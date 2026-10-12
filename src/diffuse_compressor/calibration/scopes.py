from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Iterator
from dataclasses import replace
from typing import Any

import torch
import torch.nn as nn

from ..config import CalibrationSpec, TargetConfig
from ..logging import QuantizationLogger
from ..targets import QuantTarget
from .cache import mix_seed
from .capture import apply_cache_aliases, EvalReplayCapture, first_cached_output, LayerCacheCapture
from .data import has_runnable_calibration, prepare_calibration_cache, resolve_samples
from .replay import replay_calibration_scope
from .scope_rules import assign_calibration_scopes
from .types import CalibrationScope, CalibrationScopeBatch, CaptureBinding, EvalReplayBatch, ScopeReplayState
from .utils import model_device, repartition_tensor


for _scope_type in (CalibrationScope, CalibrationScopeBatch, CaptureBinding, EvalReplayBatch, ScopeReplayState):
    _scope_type.__module__ = __name__
del _scope_type


@torch.inference_mode()
def iter_calibration_scopes(
    model: nn.Module,
    targets: Iterable[QuantTarget],
    target_config: TargetConfig | None,
    calibration: CalibrationSpec | None,
    *,
    input_stats_only: bool = False,
    capture_target_outputs: bool = True,
    capture_eval_replays: bool = True,
    logger: QuantizationLogger | None = None,
) -> Iterator[CalibrationScopeBatch]:
    """Yield calibration batches scope by scope.

    Args:
        model: Model used to replay calibration inputs.
        targets: Concrete quantization targets.
        target_config: Optional scope rules and cache aliases.
        calibration: Optional calibration settings and samples.
        input_stats_only: Capture streamed target input minima instead of full
            input row tensors.
        capture_target_outputs: Whether target output tensors should be cached.
        capture_eval_replays: Whether the consumer reads eval replay records
            from the yielded batches. When ``False``, eval module I/O is still
            captured for a scope if the next scope replays from previous-scope
            outputs, and skipped entirely otherwise.

    Yields:
        Scope batches containing target inputs, rich caches, and eval replay
        records. When no runnable calibration exists, yields empty-input
        batches so quantization can still proceed.
    """

    log = _resolve_logger(logger)
    target_list = list(targets)
    scopes = assign_calibration_scopes(model, target_list, target_config)
    total_scopes = len(scopes)
    if not has_runnable_calibration(calibration):
        log.info("- No runnable calibration data; yielding %d empty scopes", total_scopes)
        for scope in scopes:
            if calibration is not None and calibration.scope_capture_mode == "one_target":
                for target in scope.targets:
                    yield CalibrationScopeBatch(
                        scope=replace(scope, targets=(target,)), inputs={}, scope_target_count=len(scope.targets)
                    )
            else:
                yield CalibrationScopeBatch(scope=scope, inputs={}, scope_target_count=len(scope.targets))
        return

    assert calibration is not None
    cache_paths = prepare_calibration_cache(model, calibration, logger=log)
    samples = resolve_samples(calibration) if calibration.cache_mode == "disabled" or not cache_paths else []
    device = model_device(model)
    prev_scope_state = ScopeReplayState()

    log.info("- Calibrating %d scopes on %s", total_scopes, device)
    for scope_index, scope in enumerate(scopes, start=1):
        log.info("- Collecting scope %d/%d: %s (%d targets)", scope_index, total_scopes, scope.name, len(scope.targets))
        next_scope = scopes[scope_index] if scope_index < total_scopes else None
        chain_next = next_scope is not None and _scope_replays_prev_outputs(next_scope)
        chain_transform = next_scope.prev_replay_transform if chain_next else None
        if calibration.scope_capture_mode == "one_target":
            eval_replays: tuple[EvalReplayBatch, ...] | None = None
            next_state: ScopeReplayState | None = None
            for target_index, target in enumerate(scope.targets, start=1):
                log.info("  + Capturing target %d/%d: %s", target_index, len(scope.targets), target.export_name)
                batch, target_prev_state = _capture_calibration_scope_batch(
                    model,
                    scope,
                    calibration,
                    cache_paths,
                    samples,
                    device,
                    prev_scope_state,
                    scope_index=scope_index,
                    targets=(target,),
                    eval_replays=eval_replays,
                    input_stats_only=input_stats_only,
                    capture_target_outputs=capture_target_outputs,
                    capture_eval_replays=capture_eval_replays,
                    retain_prev_state=chain_next,
                    chain_transform=chain_transform,
                    # Every target pass replays the same previous state; keep it until the scope ends.
                    consume_prev=False,
                    logger=log,
                )
                if next_state is None:
                    eval_replays = batch.eval_replays
                    next_state = target_prev_state
                yield batch
                _clear_scope_batch(batch)
            prev_scope_state = next_state or ScopeReplayState()
        else:
            batch, prev_scope_state = _capture_calibration_scope_batch(
                model,
                scope,
                calibration,
                cache_paths,
                samples,
                device,
                prev_scope_state,
                scope_index=scope_index,
                input_stats_only=input_stats_only,
                capture_target_outputs=capture_target_outputs,
                capture_eval_replays=capture_eval_replays,
                retain_prev_state=chain_next,
                chain_transform=chain_transform,
                consume_prev=True,
                logger=log,
            )
            yield batch
            _clear_scope_batch(batch)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _capture_calibration_scope_batch(
    model: nn.Module,
    scope: CalibrationScope,
    calibration: CalibrationSpec,
    cache_paths,
    samples,
    device: torch.device,
    prev_scope_state: ScopeReplayState,
    *,
    scope_index: int,
    targets: tuple[QuantTarget, ...] | None = None,
    eval_replays: tuple[EvalReplayBatch, ...] | None = None,
    input_stats_only: bool = False,
    capture_target_outputs: bool = True,
    capture_eval_replays: bool = True,
    retain_prev_state: bool = True,
    chain_transform: Callable[[EvalReplayBatch], tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
    consume_prev: bool = False,
    logger: QuantizationLogger | None = None,
) -> tuple[CalibrationScopeBatch, ScopeReplayState]:
    """Capture one calibration batch for all scope targets or a target subset.

    Args:
        capture_eval_replays: Keep full scoring records for low-rank search.
        retain_prev_state: The next scope chains from this one, so every call's
            eval-module record is handed on.
        chain_transform: The next scope's ``prev_replay_transform``; input slots it
            rebuilds from outputs are not stored.
        consume_prev: Release previous-scope records as they are replayed.
        eval_replays: Scoring records from an earlier pass over the same scope;
            when given, nothing is captured from the eval module.
    """

    log = _resolve_logger(logger)
    batch_scope = scope if targets is None else replace(scope, targets=targets)
    capture = LayerCacheCapture(
        list(batch_scope.targets),
        batch_scope.captures,
        max_rows=calibration.max_rows_per_target,
        input_stats_only=input_stats_only,
        capture_target_outputs=capture_target_outputs,
        seed=calibration.seed,
    )
    first_pass = eval_replays is None
    capture_score = capture_eval_replays and first_pass
    capture_chain = retain_prev_state and first_pass
    eval_capture = EvalReplayCapture(
        scope.eval_module if capture_score or capture_chain else None,
        scope.eval_module_name if capture_score or capture_chain else None,
        calibration.max_rows_per_target,
        replay_arg_indices=scope.replay_arg_indices,
        replay_kwarg_keys=scope.replay_kwarg_keys,
        replay_transform=scope.replay_transform,
        seed=mix_seed(calibration.seed, f"{scope.name}:eval"),
        score=capture_score,
        max_score_records=calibration.max_score_records,
        chain=capture_chain,
        chain_transform=chain_transform,
    )
    capture.install()
    eval_capture.install()
    try:
        replay_calibration_scope(
            model,
            scope,
            calibration,
            cache_paths,
            samples,
            device,
            prev_scope_state,
            scope_index=scope_index,
            consume_prev=consume_prev,
            logger=log,
        )
    finally:
        eval_capture.remove()
        capture.remove()

    apply_cache_aliases(capture.layer_cache, scope.cache_aliases)
    if input_stats_only:
        inputs = {
            name: torch.tensor([value], dtype=torch.float32)
            for name, value in capture.input_mins(scope.cache_aliases).items()
        }
    else:
        inputs = capture.inputs(scope.cache_aliases)
    input_partitions = {
        name: repartition_tensor(
            tensor, sample_size=calibration.sample_size, sample_batch_size=calibration.sample_batch_size
        )
        for name, tensor in inputs.items()
    }
    layer_cache = capture.layer_cache
    captured_eval_replays = eval_capture.replays() if capture_score else eval_replays or ()
    eval_replay = captured_eval_replays[0] if captured_eval_replays else None
    chain_records = eval_capture.chain_records() if capture_chain else []
    retained_rows = _total_tensor_rows(inputs.values())
    partition_rows = _total_partition_rows(input_partitions.values())
    log.info(
        "  + Captured %d input caches (%d rows, %d partition rows) and %d eval replay batches",
        len(inputs),
        retained_rows,
        partition_rows,
        len(captured_eval_replays),
    )
    if capture_chain:
        # Every counted call appends one chain record, so the only possible shortfall is no calls at all.
        if eval_capture.num_calls:
            log.info("  + Chained %d/%d calls to the next scope", len(chain_records), eval_capture.num_calls)
        else:
            log.warning(
                "  ! No calls reached eval module %s, so nothing was chained to the next scope",
                scope.eval_module_name or scope.name,
            )
    batch = CalibrationScopeBatch(
        scope=batch_scope,
        inputs=inputs,
        input_partitions=input_partitions,
        layer_cache=layer_cache,
        eval_replay=eval_replay,
        eval_replays=captured_eval_replays,
        scope_target_count=len(scope.targets),
    )
    if not capture_chain:
        return batch, ScopeReplayState()
    if chain_records:
        return batch, ScopeReplayState(replays=chain_records)
    cached_output = first_cached_output(layer_cache)
    return batch, ScopeReplayState(outputs=() if cached_output is None else (cached_output,))


def _scope_replays_prev_outputs(scope: CalibrationScope) -> bool:
    """Return whether a scope can replay from the previous scope's outputs."""

    return scope.use_prev_scope_outputs and not scope.recompute and scope.replay_module is not None


def _clear_scope_batch(batch: CalibrationScopeBatch) -> None:
    """Release target-local caches after the caller consumes one batch."""

    for cache in batch.layer_cache.values():
        cache.clear()


def _total_tensor_rows(tensors) -> int:
    """Return the total flattened row count for captured target tensors."""

    total = 0
    for tensor in tensors:
        if tensor.numel() > 0:
            total += tensor.reshape(-1, tensor.shape[-1]).shape[0]
    return total


def _total_partition_rows(partitions_by_name) -> int:
    """Return the total row count retained after sample partitioning."""

    total = 0
    for partitions in partitions_by_name:
        for partition in partitions:
            if partition.numel() > 0:
                total += partition.reshape(-1, partition.shape[-1]).shape[0]
    return total


def _resolve_logger(logger: QuantizationLogger | None) -> QuantizationLogger:
    if logger is None:
        return QuantizationLogger.get_logger(__name__)
    return logger.for_name(__name__)
