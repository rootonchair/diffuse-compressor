from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import safetensors.torch

from ..artifact import ExportResult, QuantizedArtifact
from ..config import ExportSpec


logger = logging.getLogger(__name__)


def export_nunchaku(artifact: QuantizedArtifact, export: ExportSpec) -> ExportResult:
    """Write a quantized artifact as a Nunchaku-compatible safetensors file.

    Args:
        artifact: Quantized artifact containing target and unquantized tensors.
        export: Export settings with output path.

    Returns:
        Export result containing the checkpoint path and metadata.
    """

    output = Path(export.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {}
    for key, value in artifact.unquantized_state_dict.items():
        state_dict[key] = value.cpu()
    for quantized in artifact.quantized_targets:
        prefix = quantized.target.export_name
        for suffix, tensor in quantized.state_dict.items():
            state_dict[f"{prefix}.{suffix}"] = tensor.cpu()

    metadata = _metadata(artifact)
    safetensors.torch.save_file(
        state_dict,
        str(output),
        metadata={"quantization_config": json.dumps(metadata, sort_keys=True)},
    )
    logger.info("- Saved %d tensors to %s", len(state_dict), output)
    return ExportResult(checkpoint_path=str(output), metadata=metadata)


def _metadata(artifact: QuantizedArtifact) -> dict[str, Any]:
    """Build JSON-serializable Nunchaku quantization metadata.

    Args:
        artifact: Quantized artifact being exported.

    Returns:
        Metadata dictionary stored under ``quantization_config``.
    """

    dtype = "fp4_e2m1_all" if artifact.spec.precision == "fp4" else "int4"
    quantized_metadata = {target.target.export_name: target.metadata for target in artifact.quantized_targets}
    return {
        "method": artifact.spec.method,
        "rank": artifact.spec.rank,
        "weight": {
            "dtype": dtype,
            "group_size": artifact.spec.group_size,
            "scale_dtypes": list(artifact.spec.weight_scale_dtypes),
        },
        "activation": {
            "dtype": artifact.spec.activation_quant.dtype,
            "scale_dtypes": list(artifact.spec.activation_quant.scale_dtypes),
            "enabled": artifact.spec.activation_quant.enabled,
        },
        "targets": [
            {
                "name": target.name,
                "export_name": target.export_name,
                "modules": list(target.module_names),
                "roles": list(target.roles),
                "precision": target.precision or artifact.spec.precision,
                "group_size": target.group_size or artifact.spec.group_size,
                "activation_quant": quantized_metadata.get(target.export_name, {}).get("activation_quant"),
            }
            for target in artifact.targets
        ],
        **artifact.metadata,
    }
