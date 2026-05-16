from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import safetensors.torch

from ..artifact import ExportResult, QuantizedArtifact
from ..config import ExportSpec


def export_nunchaku(artifact: QuantizedArtifact, export: ExportSpec) -> ExportResult:
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
    return ExportResult(checkpoint_path=str(output), metadata=metadata)


def _metadata(artifact: QuantizedArtifact) -> dict[str, Any]:
    dtype = "fp4_e2m1_all" if artifact.spec.precision == "fp4" else "int4"
    return {
        "method": artifact.spec.method,
        "rank": artifact.spec.rank,
        "weight": {"dtype": dtype, "group_size": artifact.spec.group_size},
        "targets": [
            {
                "name": target.name,
                "export_name": target.export_name,
                "modules": list(target.module_names),
                "roles": list(target.roles),
            }
            for target in artifact.targets
        ],
        **artifact.metadata,
    }
