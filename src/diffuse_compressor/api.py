from __future__ import annotations

import torch
import torch.nn as nn

from .artifact import ExportResult, QuantizedArtifact
from .artifact_cache import (
    load_quantization_cache,
    load_target_quantization_caches,
    save_quantization_cache,
    save_target_quantization_cache,
)
from .calibration import has_runnable_calibration, iter_calibration_scopes
from .calibration.data import cache_files as _cache_files
from .calibration.data import select_calibration_cache_files as _select_calibration_cache_files
from .calibration.utils import accelerate_hooks_temporarily_removed as _accelerate_hooks_temporarily_removed
from .config import (
    AdaNormAwqW4A16Layout,
    ActivationQuantSpec,
    AwqW4A16Layout,
    CalibrationCaptureRule,
    CalibrationScopeRule,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    GptqSpec,
    LoggingConfig,
    LowRankSolverSpec,
    NaiveSvdqLayout,
    NunchakuSvdqLayout,
    PatchRule,
    QuantizationCacheSpec,
    RangeCalibrationSpec,
    SkipRule,
    SmoothSpec,
    SvdqTargetQuant,
    SvdqLayout,
    TargetConfig,
    TargetRule,
    AwqTargetQuant,
    WeightRangeCalibrationSpec,
)
from .exporters import export_nunchaku
from .logging import QuantizationLogger
from .quantize import quantize_targets
from .patches import ShiftedConv2d, ShiftedLinear, prepare_model
from .targets import collect_quant_targets, select_unquantized_state_dict, target_shared_low_rank


def quantize_diffusion(
    model: nn.Module,
    spec: DiffusionQuantSpec,
    targets,
    calibration: CalibrationSpec | None = None,
    target_config: TargetConfig | None = None,
    logging: LoggingConfig | None = None,
    *,
    logger: QuantizationLogger | None = None,
    write_target_records: bool = True,
) -> QuantizedArtifact:
    """Quantize selected diffusion modules into an in-memory artifact.

    Args:
        model: Model containing the already-prepared target modules.
        spec: Quantization settings to apply.
        targets: Iterable of concrete targets returned by
            :func:`collect_quant_targets`.
        calibration: Optional calibration settings and samples.
        target_config: Optional target config used to select unquantized
            state-dict entries and calibration scopes.
        logging: Optional file logging configuration for this run.
        logger: Optional explicit logger for composed workflows.
        write_target_records: Write target JSONL records before returning.

    Returns:
        Quantized artifact containing quantized target tensors, unquantized
        tensors, and metadata.
    """

    logger = logger or QuantizationLogger(config=logging)
    if spec.method != "svdquant":
        raise ValueError(f"Unsupported quantization method: {spec.method!r}")
    targets = list(targets)
    logger.info("* Quantizing diffusion weights")
    logger.info("- Selected %d quantization targets", len(targets))
    activation_shifts: dict[str, float] = {}
    if (
        target_config is not None
        and has_runnable_calibration(calibration)
        and _has_activation_shift_targets(targets)
    ):
        logger.info("* Calibrating activation shifts")
        targets, activation_shifts = _apply_calibrated_activation_shifts(
            model, targets, calibration, target_config, spec, logger
        )
        logger.info("- Applied activation shifts to %d modules", len(activation_shifts))
    with _accelerate_hooks_temporarily_removed(model, logger=logger):
        unquantized = select_unquantized_state_dict(
            model,
            target_config.unquantized_patterns if target_config is not None else (),
            [name for target in targets for name in target.module_names],
        )
    logger.info("- Keeping %d unquantized tensors", len(unquantized))
    _validate_compute_device(spec.compute_device)
    cached = load_quantization_cache(spec, target_config, targets, unquantized, calibration, logger=logger)
    if cached is not None:
        logger.info("- Using cached quantized artifact")
        return cached
    cached_targets = load_target_quantization_caches(spec, target_config, targets, calibration, logger=logger)
    quantized_by_name = dict(cached_targets)
    calibration_free_targets = [
        target
        for target in targets
        if target.export_name not in quantized_by_name and _target_can_skip_calibration_replay(target)
    ]
    if calibration_free_targets:
        logger.info(
            "- Quantizing %d calibration-free target%s before calibration replay",
            len(calibration_free_targets),
            "" if len(calibration_free_targets) == 1 else "s",
        )
        for quantized_target in quantize_targets(
            calibration_free_targets, spec, calibration=calibration, logger=logger
        ):
            save_target_quantization_cache(quantized_target, spec, target_config, targets, calibration, logger=logger)
            quantized_by_name[quantized_target.target.export_name] = quantized_target
    captured_targets: set[str] = set()
    captured_scopes: list[str] = []
    scope_target_counts: dict[str, int] = {}
    eval_replay_scopes: list[str] = []
    if len(quantized_by_name) == len(targets):
        logger.info("- All %d targets are available before calibration replay; skipping scope capture", len(targets))
    else:
        remaining_targets = [target for target in targets if target.export_name not in quantized_by_name]
        for index, batch in enumerate(
            iter_calibration_scopes(
                model,
                remaining_targets,
                target_config,
                calibration,
                capture_eval_replays=_spec_wants_eval_replays(spec, remaining_targets),
                logger=logger,
            ),
            start=1,
        ):
            scope_targets = [target for target in batch.scope.targets if target.export_name not in quantized_by_name]
            logger.info("- Quantizing scope %d: %s (%d targets)", index, batch.scope.name, len(scope_targets))
            if not scope_targets:
                if calibration is not None and calibration.scope_capture_mode == "one_target":
                    batch.inputs.clear()
                    batch.input_partitions.clear()
                    batch.layer_cache.clear()
                continue
            with _accelerate_hooks_temporarily_removed(model, logger=logger, reapply=False):
                if spec.offload_model:
                    logger.info("- Offloading model to CPU while quantizing scope %s", batch.scope.name)
                    model.to("cpu")
                    _clear_cuda_cache(spec.compute_device)
                try:
                    for target in scope_targets:
                        quantized = quantize_targets(
                            [target],
                            spec,
                            calibration_inputs=batch.inputs,
                            calibration_input_partitions=batch.input_partitions,
                            layer_cache=batch.layer_cache,
                            eval_replay=batch.eval_replays or batch.eval_replay,
                            calibration=calibration,
                            logger=logger,
                        )
                        if not quantized:
                            continue
                        quantized_target = quantized[0]
                        save_target_quantization_cache(
                            quantized_target, spec, target_config, targets, calibration, logger=logger
                        )
                        quantized_by_name[quantized_target.target.export_name] = quantized_target
                finally:
                    if spec.offload_model:
                        _clear_cuda_cache(spec.compute_device)
            captured_targets.update(batch.inputs)
            if batch.scope.name not in scope_target_counts:
                captured_scopes.append(batch.scope.name)
                scope_target_counts[batch.scope.name] = batch.scope_target_count or len(batch.scope.targets)
            if (batch.eval_replays or batch.eval_replay is not None) and batch.scope.name not in eval_replay_scopes:
                eval_replay_scopes.append(batch.scope.name)
            if calibration is not None and calibration.scope_capture_mode == "one_target":
                batch.inputs.clear()
                batch.input_partitions.clear()
                batch.layer_cache.clear()
    missing_targets = [target.export_name for target in targets if target.export_name not in quantized_by_name]
    if missing_targets:
        raise RuntimeError(f"Quantization did not produce artifacts for targets: {missing_targets}")
    quantized_targets = [quantized_by_name[target.export_name] for target in targets]
    metadata = {}
    metadata["quantization"] = {"compute_device": spec.compute_device, "offload_model": spec.offload_model}
    if calibration is not None:
        metadata["calibration"] = {
            "num_samples": calibration.num_samples,
            "cache_num_samples": calibration.cache_num_samples,
            "batch_size": calibration.batch_size,
            "cache_dir": None if calibration.cache_dir is None else str(calibration.cache_dir),
            "cache_mode": calibration.cache_mode,
            "cache_records": _calibration_cache_record_metadata(calibration),
            "has_samples": calibration.samples is not None,
            "has_prompts": calibration.prompts is not None,
            "captured_targets": sorted(captured_targets),
            "captured_scopes": captured_scopes,
            "eval_replay_scopes": eval_replay_scopes,
            "scope_target_counts": scope_target_counts,
            "max_rows_per_target": calibration.max_rows_per_target,
            "scope_capture_mode": calibration.scope_capture_mode,
            "sample_size": calibration.sample_size,
            "sample_batch_size": calibration.sample_batch_size,
            "shuffle": calibration.shuffle,
            "drop_last": calibration.drop_last,
            "num_workers": calibration.num_workers,
            "eager_load_samples": calibration.eager_load_samples,
            "ram_usage_limit": calibration.ram_usage_limit,
            "artifact_cache": None
            if calibration.artifact_cache is None
            else {
                "cache_dir": None
                if calibration.artifact_cache.cache_dir is None
                else str(calibration.artifact_cache.cache_dir),
                "cache_mode": calibration.artifact_cache.cache_mode,
                "save_model": calibration.artifact_cache.save_model,
            },
            "activation_shifts": activation_shifts,
        }
    artifact = QuantizedArtifact(
        spec=spec,
        target_config=target_config,
        targets=targets,
        quantized_targets=quantized_targets,
        unquantized_state_dict=unquantized,
        metadata=metadata,
    )
    save_quantization_cache(artifact, calibration, logger=logger)
    logger.info("- Quantized %d targets", len(quantized_targets))
    if write_target_records:
        logger.write_target_records(artifact)
    return artifact


def _calibration_cache_record_metadata(calibration: CalibrationSpec) -> dict[str, int] | None:
    """Return selected and total root cache record counts for metadata."""

    if calibration.cache_mode == "disabled" or calibration.cache_dir is None:
        return None
    paths = _cache_files(calibration)
    selected = _select_calibration_cache_files(paths, calibration)
    return {"selected": len(selected), "total": len(paths)}


def _has_activation_shift_targets(targets: list) -> bool:
    """Return whether any target should use activation shifting."""

    return any(_target_shift_activations(target) for target in targets)


def _target_can_skip_calibration_replay(target) -> bool:
    """Return whether a target can be packed without calibration captures."""

    return isinstance(target.quant, AwqTargetQuant)


def _spec_wants_eval_replays(spec: DiffusionQuantSpec, targets: list) -> bool:
    """Return whether quantization will read eval replay records for any target.

    Eval replays are consumed only by search-mode low-rank solving with
    ``eval_replay`` enabled, and only for targets whose effective rank is
    positive with a shared low-rank branch.
    """

    if spec.low_rank_solver.mode != "search" or not spec.low_rank_solver.eval_replay:
        return False
    for target in targets:
        rank = getattr(target.quant, "rank", None)
        effective_rank = spec.rank if rank is None else rank
        if effective_rank > 0 and target_shared_low_rank(target):
            return True
    return False


def _apply_calibrated_activation_shifts(
    model: nn.Module,
    targets: list,
    calibration: CalibrationSpec | None,
    target_config: TargetConfig,
    spec: DiffusionQuantSpec,
    logger: QuantizationLogger | None = None,
) -> tuple[list, dict[str, float]]:
    """Patch linear targets with DeepCompressor-style lower-bound shifts.

    Args:
        model: Model to patch in place.
        targets: Current concrete targets.
        calibration: Calibration settings used to capture target inputs.
        target_config: Target configuration used to refresh target references.
        spec: Global quantization settings used for offload behavior.

    Returns:
        Refreshed targets and applied shifts by module name.
    """

    logger = logger or QuantizationLogger()
    shifted: dict[str, float] = {}
    for index, batch in enumerate(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            calibration,
            input_stats_only=True,
            capture_target_outputs=False,
            logger=logger,
        ),
        start=1,
    ):
        logger.info("- Checking activation shift scope %d: %s", index, batch.scope.name)
        with _accelerate_hooks_temporarily_removed(model, logger=logger, reapply=False):
            try:
                for target in batch.scope.targets:
                    if not _target_shift_activations(target):
                        continue
                    if all(_is_shifted_module(module, target.kind) for module in target.modules):
                        continue
                    inputs = batch.inputs.get(target.export_name)
                    if inputs is None or inputs.numel() == 0:
                        continue
                    lowerbound = float(inputs.float().amin().item())
                    if lowerbound >= 0:
                        continue
                    shift = -lowerbound
                    for module_name, module in zip(target.module_names, target.modules, strict=True):
                        if _is_shifted_module(module, target.kind):
                            continue
                        patch_type = "shift_conv" if target.kind == "conv" else "shift_linear"
                        prepare_model(model, [PatchRule(type=patch_type, module=module_name, args={"shift": shift})])  # type: ignore[arg-type]
                        shifted[module_name] = shift
                        logger.info("  + Shifted %s by %.6g", module_name, shift)
            finally:
                if spec.offload_model:
                    model.to("cpu")
                    _clear_cuda_cache(spec.compute_device)
    if not shifted:
        return targets, {}
    refreshed = collect_quant_targets(model, target_config, spec=spec)
    for target in refreshed:
        for module_name, module in zip(target.module_names, target.modules, strict=True):
            if module_name in shifted and isinstance(module, ShiftedLinear):
                module.linear.unsigned = True
            if module_name in shifted and isinstance(module, ShiftedConv2d):
                module.conv.unsigned = True
    return refreshed, shifted


def _target_shift_activations(target) -> bool:
    if isinstance(target.quant, AwqTargetQuant):
        return False
    return target.quant.shift_activations is True


def _is_shifted_module(module: nn.Module, kind: str) -> bool:
    """Return whether a target module is already shifted.

    Args:
        module: Module selected by a target.
        kind: Target kind.

    Returns:
        ``True`` when the module is the matching shifted wrapper.
    """

    if kind == "conv":
        return isinstance(module, ShiftedConv2d)
    return isinstance(module, ShiftedLinear)


def _clear_cuda_cache(device_name: str | None) -> None:
    """Release cached CUDA blocks for an optional work device."""

    if device_name is None:
        return
    try:
        device = torch.device(device_name)
    except (RuntimeError, TypeError):
        return
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_compute_device(device_name: str | None) -> None:
    """Validate the optional per-target compute device before offloading."""

    if device_name is None:
        return
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"compute_device {device_name!r} requires CUDA, but CUDA is not available")


def export_checkpoint(
    artifact: QuantizedArtifact, export: ExportSpec, *, logger: QuantizationLogger | None = None
) -> ExportResult:
    """Write a quantized artifact to a runtime-compatible checkpoint.

    Args:
        artifact: In-memory quantization result.
        export: Exporter target and output path settings.

    Returns:
        Export result with the checkpoint path and metadata.
    """

    if export.target == "nunchaku":
        log = logger or QuantizationLogger()
        log.info("* Exporting checkpoint to %s", export.output)
        return export_nunchaku(artifact, export, logger=log)
    raise ValueError(f"Unsupported export target: {export.target!r}")


def quantize_and_export(
    model: nn.Module,
    spec: DiffusionQuantSpec,
    target_config: TargetConfig,
    calibration: CalibrationSpec | None,
    export: ExportSpec,
    logging: LoggingConfig | None = None,
) -> ExportResult:
    """Run model preparation, target collection, quantization, and export.

    Args:
        model: Diffusion model to rewrite and quantize in place.
        spec: Quantization settings.
        target_config: Model-agnostic rewrite and target configuration.
        calibration: Optional calibration data and cache configuration.
        export: Export target and checkpoint path settings.
        logging: Optional file logging configuration for this run.

    Returns:
        Export result for the written checkpoint.
    """

    logger = QuantizationLogger(config=logging)
    logger.info("* Preparing model")
    prepare_model(model, target_config.patches)
    logger.info("* Collecting quantization targets")
    targets = collect_quant_targets(model, target_config, spec=spec)
    logger.info("- Collected %d quantization targets", len(targets))
    artifact = quantize_diffusion(
        model,
        spec,
        targets,
        calibration=calibration,
        target_config=target_config,
        logging=None,
        logger=logger,
        write_target_records=False,
    )
    result = export_checkpoint(artifact, export, logger=logger)
    logger.write_target_records(artifact, result)
    return result


__all__ = [
    "AdaNormAwqW4A16Layout",
    "AwqW4A16Layout",
    "CalibrationCaptureRule",
    "CalibrationSpec",
    "CalibrationScopeRule",
    "ActivationQuantSpec",
    "DiffusionQuantSpec",
    "ExportResult",
    "ExportSpec",
    "GptqSpec",
    "LoggingConfig",
    "LowRankSolverSpec",
    "NaiveSvdqLayout",
    "NunchakuSvdqLayout",
    "PatchRule",
    "QuantizationCacheSpec",
    "QuantizedArtifact",
    "RangeCalibrationSpec",
    "SkipRule",
    "SmoothSpec",
    "SvdqTargetQuant",
    "SvdqLayout",
    "TargetConfig",
    "TargetRule",
    "AwqTargetQuant",
    "WeightRangeCalibrationSpec",
    "collect_quant_targets",
    "export_checkpoint",
    "prepare_model",
    "quantize_and_export",
    "quantize_diffusion",
]
