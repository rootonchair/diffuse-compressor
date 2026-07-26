from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn

from ..config import CalibrationSpec
from ..logging import QuantizationLogger
from .data import ModuleForwardInput, iter_calibration_forward_inputs, run_forward_input
from .types import CalibrationScope, ScopeReplayState
from .utils import check_ram, has_accelerate_hooks


_REPLAY_LOGGER_NAME = "diffuse_compressor.calibration.scopes"


def replay_calibration_scope(
    model: nn.Module,
    scope: CalibrationScope,
    calibration: CalibrationSpec,
    cache_paths,
    samples,
    device: torch.device,
    prev_scope_state: ScopeReplayState,
    *,
    scope_index: int,
    logger: QuantizationLogger | None = None,
) -> None:
    """Run the forward path needed to populate installed scope hooks."""

    log = _resolve_logger(logger)
    prev_available = (
        bool(prev_scope_state.replays) if scope.prev_replay_transform is not None else bool(prev_scope_state.outputs)
    )
    if scope.use_prev_scope_outputs and prev_available and not scope.recompute and scope.replay_module is not None:
        log.info("  + Replaying %s from previous scope outputs", scope.replay_module_name or scope.name)
        with _scoped_replay_device(scope, model, device, logger=log):
            for forward_input in prev_scope_state.forward_inputs(
                scope.prev_output_transform, scope.prev_replay_transform
            ):
                _run_module_forward_input(scope.replay_module, forward_input.to(device))
                check_ram(calibration)
    elif cache_paths:
        _warn_scoped_replay_fallback(scope, model, prev_available, scope_index=scope_index, logger=log)
        _restore_model_for_full_replay(model, device, logger=log)
        replay_mode = "root replay"
        if not scope.recompute and scope.eval_module is not None:
            replay_mode = "root replay with scope early-stop"
        log.info("  + Running %s from %d cached inputs", replay_mode, len(cache_paths))
        for forward_input in iter_calibration_forward_inputs(calibration, cache_paths=cache_paths):
            replay = forward_input.to(device)
            _run_with_scope_early_stop(scope, lambda replay=replay: model(*replay.args, **replay.kwargs))
            check_ram(calibration)
    else:
        _warn_scoped_replay_fallback(scope, model, prev_available, scope_index=scope_index, logger=log)
        _restore_model_for_full_replay(model, device, logger=log)
        replay_mode = "sample forwards"
        if not scope.recompute and scope.eval_module is not None:
            replay_mode = "sample forwards with scope early-stop"
        log.info("  + Running %s from %d samples", replay_mode, len(samples))
        for forward_input in iter_calibration_forward_inputs(calibration, samples=samples):
            replay = forward_input.to(device)
            _run_with_scope_early_stop(scope, lambda replay=replay: run_forward_input(model, calibration, replay))
            check_ram(calibration)


@contextmanager
def _scoped_replay_device(
    scope: CalibrationScope, model: nn.Module, device: torch.device, *, logger: QuantizationLogger
) -> Iterator[None]:
    """Move only modules needed by previous-scope replay to a device.

    Engages automatically whenever Accelerate hooks aren't managing device
    placement for ``model`` — this is strictly cheaper than Accelerate's
    per-layer streaming for the repeated same-block access pattern of scope
    replay.
    """

    if has_accelerate_hooks(model):
        yield
        return

    modules = _scoped_replay_modules(scope)
    if not modules:
        yield
        return

    logger.info("  + Moving %d scoped replay module(s) to %s", len(modules), device)
    for module in modules:
        module.to(device)
    try:
        yield
    finally:
        for module in reversed(modules):
            module.to("cpu")
        _clear_cuda_cache(device)


def _scoped_replay_modules(scope: CalibrationScope) -> list[nn.Module]:
    """Return the minimal known modules needed to replay one scope."""

    modules: list[nn.Module] = []
    candidates = [scope.replay_module, scope.eval_module]
    candidates.extend(module for target in scope.targets for module in target.modules)
    candidates.extend(binding.module for binding in scope.captures)
    for module in candidates:
        if module is not None:
            _append_minimal_module(modules, module)
    return modules


def _append_minimal_module(modules: list[nn.Module], candidate: nn.Module) -> None:
    """Append a module unless it is already covered by another module."""

    if any(existing is candidate or _module_contains(existing, candidate) for existing in modules):
        return
    modules[:] = [existing for existing in modules if not _module_contains(candidate, existing)]
    modules.append(candidate)


def _module_contains(parent: nn.Module, child: nn.Module) -> bool:
    """Return whether ``child`` is ``parent`` or a descendant."""

    return any(module is child for module in parent.modules())


def _restore_model_for_full_replay(
    model: nn.Module,
    device: torch.device,
    *,
    logger: QuantizationLogger | None = None,
) -> None:
    """Restore the full model for replay paths that cannot run scope-local.

    Like :func:`_scoped_replay_device`, this engages automatically whenever
    Accelerate isn't already managing device placement for ``model``.
    """

    logger = _resolve_logger(logger)
    if has_accelerate_hooks(model):
        return
    full_replay_offloaded = _accelerate_cpu_offload_for_full_replay(model, device, logger=logger)
    if full_replay_offloaded:
        return
    logger.info("  + Restoring full model to %s for calibration replay", device)
    model.to(device)
    _clear_cuda_cache(device)


def _accelerate_cpu_offload_for_full_replay(
    model: nn.Module, device: torch.device, *, logger: QuantizationLogger
) -> bool:
    """Attach Accelerate CPU offload hooks for full-model replay when useful."""

    if device.type == "cpu":
        return False
    try:
        from accelerate import cpu_offload
    except ImportError:
        return False
    logger.info("  + Applying Accelerate CPU offload for full-model calibration replay on %s", device)
    cpu_offload(model, execution_device=device)
    return True


def _warn_scoped_replay_fallback(
    scope: CalibrationScope,
    model: nn.Module,
    prev_available: bool,
    *,
    scope_index: int,
    logger: QuantizationLogger,
) -> None:
    """Warn when scoped replay was expected but cannot be used.

    Scoped replay is attempted automatically whenever Accelerate isn't
    already managing device placement for ``model`` — see
    :func:`_scoped_replay_device`.
    """

    if has_accelerate_hooks(model) or not scope.use_prev_scope_outputs or scope_index <= 1:
        return
    if scope.recompute:
        reason = "scope is configured with recompute=True"
    elif scope.replay_module is None:
        reason = "scope has no replay module"
    elif not prev_available:
        reason = "previous scope outputs are unavailable"
    else:
        return
    logger.warning(
        "  ! Scoped replay offload unavailable for %s (%s); falling back to full-model replay", scope.name, reason
    )


def _clear_cuda_cache(device: torch.device) -> None:
    """Release cached CUDA blocks for an optional work device."""

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_logger(logger: QuantizationLogger | None) -> QuantizationLogger:
    if logger is None:
        return QuantizationLogger.get_logger(_REPLAY_LOGGER_NAME)
    return logger.for_name(_REPLAY_LOGGER_NAME)


def _run_module_forward_input(module: nn.Module, forward_input: ModuleForwardInput) -> Any:
    """Run a module with a normalized forward input."""

    return module(*forward_input.args, **forward_input.kwargs)


class _EarlyStopReplay(Exception):
    """Private sentinel used to stop root replay after the current scope."""


def _run_with_scope_early_stop(scope: CalibrationScope, run_forward: Callable[[], Any]) -> None:
    """Run a root forward and optionally stop after the scope eval module."""

    handle: torch.utils.hooks.RemovableHandle | None = None
    if not scope.recompute and scope.eval_module is not None:
        handle = scope.eval_module.register_forward_hook(_early_stop_hook, with_kwargs=True)
    try:
        try:
            run_forward()
        except _EarlyStopReplay:
            pass
    finally:
        if handle is not None:
            handle.remove()


def _early_stop_hook(_module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], _output: Any) -> None:
    """Stop root replay after already-registered capture hooks have run."""

    raise _EarlyStopReplay()
