from __future__ import annotations

import torch.nn as nn

from .artifact import ExportResult, QuantizedArtifact
from .calibration import iter_calibration_scopes
from .config import (
    CalibrationCaptureRule,
    CalibrationScopeRule,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    LowRankSolverSpec,
    PatchRule,
    SmoothSpec,
    TargetConfig,
    TargetRule,
)
from .exporters import export_nunchaku
from .methods.svdquant import quantize_targets
from .patches import prepare_model
from .targets import collect_quant_targets, select_unquantized_state_dict


def quantize_diffusion(
    model: nn.Module,
    spec: DiffusionQuantSpec,
    targets,
    calibration: CalibrationSpec | None = None,
    target_config: TargetConfig | None = None,
) -> QuantizedArtifact:
    if spec.method != "svdquant":
        raise ValueError(f"Unsupported quantization method: {spec.method!r}")
    targets = list(targets)
    quantized_targets = []
    captured_targets: set[str] = set()
    captured_scopes: list[str] = []
    scope_target_counts: dict[str, int] = {}
    eval_replay_scopes: list[str] = []
    for batch in iter_calibration_scopes(model, targets, target_config, calibration):
        quantized_targets.extend(
            quantize_targets(batch.scope.targets, spec, calibration_inputs=batch.inputs, eval_replay=batch.eval_replay)
        )
        captured_targets.update(batch.inputs)
        captured_scopes.append(batch.scope.name)
        scope_target_counts[batch.scope.name] = len(batch.scope.targets)
        if batch.eval_replay is not None:
            eval_replay_scopes.append(batch.scope.name)
    unquantized = select_unquantized_state_dict(
        model,
        target_config.unquantized_patterns if target_config is not None else (),
        [name for target in targets for name in target.module_names],
    )
    metadata = {}
    if calibration is not None:
        metadata["calibration"] = {
            "num_samples": calibration.num_samples,
            "batch_size": calibration.batch_size,
            "cache_dir": None if calibration.cache_dir is None else str(calibration.cache_dir),
            "cache_mode": calibration.cache_mode,
            "has_samples": calibration.samples is not None,
            "has_prompts": calibration.prompts is not None,
            "captured_targets": sorted(captured_targets),
            "captured_scopes": captured_scopes,
            "eval_replay_scopes": eval_replay_scopes,
            "scope_target_counts": scope_target_counts,
            "max_rows_per_target": calibration.max_rows_per_target,
            "sample_size": calibration.sample_size,
            "sample_batch_size": calibration.sample_batch_size,
            "element_size": calibration.element_size,
            "element_batch_size": calibration.element_batch_size,
            "ram_usage_limit": calibration.ram_usage_limit,
        }
    return QuantizedArtifact(
        spec=spec,
        target_config=target_config,
        targets=targets,
        quantized_targets=quantized_targets,
        unquantized_state_dict=unquantized,
        metadata=metadata,
    )


def export_checkpoint(artifact: QuantizedArtifact, export: ExportSpec) -> ExportResult:
    if export.target == "nunchaku":
        return export_nunchaku(artifact, export)
    raise ValueError(f"Unsupported export target: {export.target!r}")


def quantize_and_export(
    model: nn.Module,
    spec: DiffusionQuantSpec,
    target_config: TargetConfig,
    calibration: CalibrationSpec | None,
    export: ExportSpec,
) -> ExportResult:
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    artifact = quantize_diffusion(model, spec, targets, calibration=calibration, target_config=target_config)
    return export_checkpoint(artifact, export)


__all__ = [
    "CalibrationCaptureRule",
    "CalibrationSpec",
    "CalibrationScopeRule",
    "DiffusionQuantSpec",
    "ExportResult",
    "ExportSpec",
    "LowRankSolverSpec",
    "PatchRule",
    "QuantizedArtifact",
    "SmoothSpec",
    "TargetConfig",
    "TargetRule",
    "collect_quant_targets",
    "export_checkpoint",
    "prepare_model",
    "quantize_and_export",
    "quantize_diffusion",
]
