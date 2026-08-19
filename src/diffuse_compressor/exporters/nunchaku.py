from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import safetensors.torch
import torch.nn as nn

from ..artifact import ExportResult, QuantizedArtifact
from ..config import (
    AdaNormAwqW4A16Layout,
    AwqW4A16Layout,
    ExportSpec,
    NaiveSvdqLayout,
    NunchakuSvdqLayout,
    PatchRule,
    SvdqLayout,
    AwqTargetQuant,
    target_bias_policy,
    target_quant_metadata,
    target_weight_layout,
    weight_layout_metadata,
)
from ..logging import QuantizationLogger
from ..patches import ShiftedLinear
from ..targets import QuantTarget


@dataclass(frozen=True)
class _RuntimeManifestDiagnostic:
    kind: str
    name: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "name": self.name, "reason": self.reason}


@dataclass(frozen=True)
class _RuntimeManifestResult:
    manifest: dict[str, Any] | None
    reasons: tuple[_RuntimeManifestDiagnostic, ...] = ()

    def diagnostics(self) -> dict[str, Any]:
        return {"emitted": self.manifest is not None, "reasons": [reason.to_dict() for reason in self.reasons]}


def export_nunchaku(
    artifact: QuantizedArtifact, export: ExportSpec, logger: QuantizationLogger | None = None
) -> ExportResult:
    """Write a quantized artifact as a Nunchaku-compatible safetensors file.

    Args:
        artifact: Quantized artifact containing target and unquantized tensors.
        export: Export settings with output path.

    Returns:
        Export result containing the checkpoint path and metadata.
    """

    log = _resolve_logger(logger)
    output = Path(export.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {}
    for key, value in artifact.unquantized_state_dict.items():
        state_dict[key] = value.cpu().contiguous()
    for quantized in artifact.quantized_targets:
        prefix = quantized.target.export_name
        for suffix, tensor in quantized.state_dict.items():
            state_dict[f"{prefix}.{suffix}"] = tensor.cpu().contiguous()

    config_metadata = _metadata(artifact, logger=log)
    _write_config(output, config_metadata)
    checkpoint_metadata = _checkpoint_metadata(config_metadata)
    safetensors.torch.save_file(state_dict, str(output), metadata=checkpoint_metadata)
    log.info("- Saved %d tensors to %s", len(state_dict), output)
    return ExportResult(checkpoint_path=str(output), metadata=config_metadata)


def _metadata(artifact: QuantizedArtifact, logger: QuantizationLogger | None = None) -> dict[str, Any]:
    """Build JSON-serializable Nunchaku quantization metadata.

    Args:
        artifact: Quantized artifact being exported.

    Returns:
        Metadata dictionary written to the checkpoint config.
    """

    dtype = "fp4_e2m1_all" if artifact.spec.precision == "fp4" else "int4"
    quantized_metadata = {target.target.export_name: target.metadata for target in artifact.quantized_targets}
    metadata = {
        "method": artifact.spec.method,
        "rank": artifact.spec.rank,
        "weight": {
            "dtype": dtype,
            "group_size": artifact.spec.group_size,
            "scale_dtypes": list(artifact.spec.weight_scale_dtypes),
        },
        "gptq": {
            "enabled": artifact.spec.gptq.enabled,
            "damp_percentage": artifact.spec.gptq.damp_percentage,
            "block_size": artifact.spec.gptq.block_size,
            "num_inv_tries": artifact.spec.gptq.num_inv_tries,
            "hessian_block_size": artifact.spec.gptq.hessian_block_size,
        },
        "activation": {
            "dtype": dtype,
            "scale_dtypes": list(artifact.spec.activation_quant.scale_dtypes),
            "enabled": artifact.spec.activation_quant.enabled,
            "group_size": artifact.spec.group_size,
        },
        "targets": [
            {
                "name": target.name,
                "export_name": target.export_name,
                "modules": list(target.module_names),
                "roles": list(target.roles),
                "precision": _target_precision(target, artifact),
                "group_size": _target_group_size(target, artifact),
                "bias": target_bias_policy(target.quant),
                "quant": target_quant_metadata(target.quant),
                "weight_layout": weight_layout_metadata(target_weight_layout(target.quant)),
                "weight_scale_layout": quantized_metadata.get(target.export_name, {}).get("weight_scale_layout"),
                "runtime_tensor_layout": quantized_metadata.get(target.export_name, {}).get("runtime_tensor_layout"),
                "activation_quant": quantized_metadata.get(target.export_name, {}).get("activation_quant"),
                "gptq": quantized_metadata.get(target.export_name, {}).get("gptq"),
            }
            for target in artifact.targets
        ],
        "structural_patches": _structural_patches(artifact),
        **_artifact_metadata_for_export(artifact.metadata),
    }
    runtime_manifest = _runtime_manifest(artifact, quantized_metadata)
    metadata["runtime_manifest_diagnostics"] = runtime_manifest.diagnostics()
    if runtime_manifest.manifest is not None:
        metadata["runtime_manifest"] = runtime_manifest.manifest
    else:
        _log_runtime_manifest_omission(runtime_manifest.reasons, logger=logger)
    return metadata


def _write_config(checkpoint_path: Path, metadata: dict[str, Any]) -> None:
    config_path = _config_path(checkpoint_path)
    config_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".config.yaml")


def _checkpoint_metadata(metadata: dict[str, Any]) -> dict[str, str]:
    quantization_config = {
        "method": metadata["method"],
        "rank": metadata["rank"],
        "weight": metadata["weight"],
        "gptq": metadata["gptq"],
        "activation": metadata["activation"],
    }
    runtime_manifest = metadata.get("runtime_manifest")
    if runtime_manifest is not None:
        quantization_config["runtime_manifest"] = runtime_manifest
    return {"quantization_config": json.dumps(quantization_config, sort_keys=True)}


def _artifact_metadata_for_export(metadata: dict[str, Any]) -> dict[str, Any]:
    exported = dict(metadata)
    if "calibration" not in exported:
        return exported
    calibration = exported["calibration"]
    activation_shifts = {}
    if isinstance(calibration, dict) and isinstance(calibration.get("activation_shifts"), dict):
        activation_shifts = calibration["activation_shifts"]
    exported["calibration"] = {"activation_shifts": activation_shifts}
    return exported


def _runtime_manifest(
    artifact: QuantizedArtifact, quantized_metadata: dict[str, dict[str, Any]]
) -> _RuntimeManifestResult:
    """Build the nunchaku_lite runtime manifest when all targets conform."""

    target_entries: list[dict[str, Any]] = []
    reasons: list[_RuntimeManifestDiagnostic] = []
    for target in artifact.targets:
        entry, target_reasons = _runtime_manifest_target(
            target, artifact, quantized_metadata.get(target.export_name, {})
        )
        reasons.extend(target_reasons)
        if entry is None:
            continue
        target_entries.append(entry)
    structural_patches, patch_reasons = _runtime_manifest_patches(artifact)
    reasons.extend(patch_reasons)
    if structural_patches is None:
        if _requires_nunchaku_manifest(artifact):
            raise RuntimeError("runtime_manifest v1 does not support one or more configured structural patches")
        return _RuntimeManifestResult(None, tuple(reasons))
    if not artifact.targets:
        reasons.append(
            _RuntimeManifestDiagnostic(
                "artifact",
                "runtime_manifest",
                "no quantization targets are available; manifest v1 requires at least one target",
            )
        )
    if reasons or not target_entries or len(target_entries) != len(artifact.targets):
        if not reasons:
            reasons.append(
                _RuntimeManifestDiagnostic(
                    "artifact",
                    "runtime_manifest",
                    "one or more targets could not be represented by runtime_manifest v1",
                )
            )
        return _RuntimeManifestResult(None, tuple(reasons))
    precision = _manifest_precision(artifact)
    return _RuntimeManifestResult(
        {
            "schema": "nunchaku_lite.runtime_manifest",
            "version": 1,
            "component": "transformer",
            "nunchaku_format_version": 1,
            "producer": _producer_metadata(),
            "requirements": {
                "method": artifact.spec.method,
                "precision": precision,
                "rank": artifact.spec.rank,
                "weight_dtype": _manifest_weight_dtype(precision),
                "activation_dtype": artifact.spec.activation_quant.dtype,
            },
            "structural_patches": structural_patches,
            "targets": target_entries,
        }
    )


def _log_runtime_manifest_omission(
    reasons: tuple[_RuntimeManifestDiagnostic, ...], logger: QuantizationLogger | None = None
) -> None:
    log = _resolve_logger(logger)
    if not reasons:
        log.warning("runtime_manifest was omitted because the checkpoint is not representable by manifest v1")
        return
    for reason in reasons:
        log.warning("runtime_manifest omitted for %s %r: %s", reason.kind, reason.name, reason.reason)


def _resolve_logger(logger: QuantizationLogger | None) -> QuantizationLogger:
    if logger is None:
        return QuantizationLogger.get_logger(__name__)
    return logger.for_name(__name__)


def _runtime_manifest_target(
    target: QuantTarget, artifact: QuantizedArtifact, quantized_metadata: dict[str, Any]
) -> tuple[dict[str, Any] | None, tuple[_RuntimeManifestDiagnostic, ...]]:
    reasons: list[_RuntimeManifestDiagnostic] = []
    nunchaku_op = _nunchaku_op(target)
    if nunchaku_op is None:
        reasons.append(
            _RuntimeManifestDiagnostic(
                "target",
                target.export_name,
                f"weight layout {target_weight_layout(target.quant).name!r} is not supported by runtime_manifest v1",
            )
        )
    if nunchaku_op == "svdq_w4a4" and quantized_metadata.get("runtime_tensor_layout") != "nunchaku_packed":
        if isinstance(target_weight_layout(target.quant), NunchakuSvdqLayout):
            raise RuntimeError(
                f"Target {target.export_name!r} declares NunchakuSvdqLayout but was not packed in Nunchaku ABI layout"
            )
        reasons.append(
            _RuntimeManifestDiagnostic(
                "target",
                target.export_name,
                "uses logical SVDQ tensors; manifest v1 requires Nunchaku-packed SVDQ tensor layout",
            )
        )
    reasons.extend(_manifest_loadability_diagnostics(target))
    if reasons or nunchaku_op is None:
        return None, tuple(reasons)

    op_options = _op_options(target)
    activation = quantized_metadata.get("activation_quant")
    return (
        {
            "name": target.name,
            "checkpoint_prefix": target.export_name,
            "source_modules": list(target.module_names),
            "roles": list(target.roles),
            "kind": target.kind,
            "nunchaku_op": nunchaku_op,
            "precision": _target_precision(target, artifact),
            "group_size": _target_group_size(target, artifact),
            "rank": _target_rank(target, artifact),
            "has_bias": _target_has_bias(target, quantized_metadata),
            "op_options": op_options,
            "activation": activation,
        },
        (),
    )


def _producer_metadata() -> dict[str, str]:
    try:
        version = importlib.metadata.version("diffuse-compressor")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {"name": "diffuse_compressor", "version": version}


def _manifest_precision(artifact: QuantizedArtifact) -> str:
    precisions = {_target_precision(target, artifact) for target in artifact.targets}
    if len(precisions) == 1:
        return next(iter(precisions))
    return "mixed"


def _manifest_weight_dtype(precision: str) -> str:
    if precision == "fp4":
        return "fp4_e2m1_all"
    if precision == "int4":
        return "int4"
    return "mixed"


def _target_precision(target: QuantTarget, artifact: QuantizedArtifact) -> str:
    if isinstance(target.quant, AwqTargetQuant):
        return "int4"
    return target.quant.precision or artifact.spec.precision


def _target_group_size(target: QuantTarget, artifact: QuantizedArtifact) -> int:
    if isinstance(target.quant, AwqTargetQuant):
        return 64
    return target.quant.group_size or artifact.spec.group_size


def _target_rank(target: QuantTarget, artifact: QuantizedArtifact) -> int:
    if isinstance(target.quant, AwqTargetQuant):
        return 0
    return artifact.spec.rank if target.quant.rank is None else target.quant.rank


def _nunchaku_op(target: QuantTarget) -> str | None:
    layout = target_weight_layout(target.quant)
    if isinstance(layout, (SvdqLayout, NaiveSvdqLayout, NunchakuSvdqLayout)):
        return "svdq_w4a4"
    if isinstance(layout, AwqW4A16Layout):
        return "awq_w4a16"
    if isinstance(layout, AdaNormAwqW4A16Layout):
        return "adanorm_awq_w4a16"
    return None


def _op_options(target: QuantTarget) -> dict[str, Any]:
    layout = target_weight_layout(target.quant)
    if isinstance(layout, NunchakuSvdqLayout):
        return {"outer_scale_splits": list(layout.outer_scale_splits)}
    if isinstance(layout, AdaNormAwqW4A16Layout):
        return {"adanorm_splits": layout.splits}
    return {}


def _target_has_bias(target: QuantTarget, quantized_metadata: dict[str, Any]) -> bool:
    del quantized_metadata
    bias_policy = target_bias_policy(target.quant)
    if bias_policy == "omit":
        return False
    if bias_policy == "zero":
        return True
    return any(getattr(module, "bias", None) is not None for module in target.modules)


def _manifest_loadability_diagnostics(target: QuantTarget) -> tuple[_RuntimeManifestDiagnostic, ...]:
    reasons: list[_RuntimeManifestDiagnostic] = []
    if target.kind != "linear":
        reasons.append(
            _RuntimeManifestDiagnostic(
                "target",
                target.export_name,
                f"target kind {target.kind!r} is not supported; manifest v1 supports only linear targets",
            )
        )
    if len(target.module_names) != 1 or len(target.modules) != 1:
        reasons.append(
            _RuntimeManifestDiagnostic(
                "target",
                target.export_name,
                f"grouped target has {len(target.module_names)} source modules; manifest v1 requires exactly one source module",
            )
        )
        return tuple(reasons)
    module_name = target.module_names[0]
    if target.export_name != module_name:
        reasons.append(
            _RuntimeManifestDiagnostic(
                "target",
                target.export_name,
                f"export_name {target.export_name!r} does not match source module path {module_name!r}; "
                "manifest v1 requires the checkpoint prefix to match the source module path",
            )
        )
    module = target.modules[0]
    if _linear_module_features(module) is None:
        reasons.append(
            _RuntimeManifestDiagnostic(
                "target",
                target.export_name,
                f"source module {module_name!r} does not expose integer in_features and out_features",
            )
        )
    return tuple(reasons)


def _linear_module_features(module: nn.Module) -> tuple[int, int] | None:
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if isinstance(module, ShiftedLinear):
        return module.linear.in_features, module.linear.out_features
    return None


def _requires_nunchaku_manifest(artifact: QuantizedArtifact) -> bool:
    return any(isinstance(target_weight_layout(target.quant), NunchakuSvdqLayout) for target in artifact.targets)


def _runtime_manifest_patches(
    artifact: QuantizedArtifact,
) -> tuple[list[dict[str, Any]] | None, tuple[_RuntimeManifestDiagnostic, ...]]:
    if artifact.target_config is None:
        return [], ()
    patches = []
    reasons: list[_RuntimeManifestDiagnostic] = []
    for patch in artifact.target_config.patches:
        manifest_patch = _runtime_manifest_patch(patch)
        if manifest_patch is None:
            reasons.append(
                _RuntimeManifestDiagnostic(
                    "structural_patch",
                    patch.module,
                    f"patch type {patch.type!r} is not supported by runtime_manifest v1",
                )
            )
            continue
        patches.append(manifest_patch)
    if reasons:
        return None, tuple(reasons)
    return patches, ()


def _structural_patches(artifact: QuantizedArtifact) -> list[dict[str, Any]]:
    if artifact.target_config is None:
        return []
    patches = []
    for patch in artifact.target_config.patches:
        exported_patch = _structural_patch(patch)
        if exported_patch is not None:
            patches.append(exported_patch)
    return patches


def _structural_patch(patch: PatchRule) -> dict[str, Any] | None:
    if patch.type not in {"split_linear", "split_linear_output", "split_conv"}:
        return None
    return {"type": patch.type, "module": patch.module, "args": _jsonable_patch_args(patch.args)}


def _runtime_manifest_patch(patch: PatchRule) -> dict[str, Any] | None:
    if patch.type == "split_linear":
        patch_type = "split_linear_input"
    elif patch.type == "split_linear_output":
        patch_type = "split_linear_output"
    else:
        return None
    return {"type": patch_type, "module": patch.module, "args": _jsonable_patch_args(patch.args)}


def _jsonable_patch_args(args: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in args.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            normalized[key] = value
        elif isinstance(value, (list, tuple)):
            normalized[key] = [
                item if isinstance(item, (str, int, float, bool)) or item is None else str(item) for item in value
            ]
        else:
            normalized[key] = str(value)
    return normalized
