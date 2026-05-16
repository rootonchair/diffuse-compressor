from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import DiffusionQuantSpec, TargetConfig
from .targets import QuantTarget


@dataclass
class QuantizedTarget:
    """Quantized tensors and metadata for one logical target.

    Args:
        target: Source target description used to produce the tensors.
        state_dict: Export-ready tensor mapping for this target.
        metadata: Per-target quantization metadata.
    """

    target: QuantTarget
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizedArtifact:
    """In-memory result produced by quantizing a model.

    Args:
        spec: Quantization settings used for all targets.
        target_config: Target and rewrite configuration used during collection.
        targets: Concrete targets selected from the model.
        quantized_targets: Quantized tensor groups for selected targets.
        unquantized_state_dict: CPU copy of parameters that remain unquantized.
        metadata: Artifact-level metadata, including calibration details.
    """

    spec: DiffusionQuantSpec
    target_config: TargetConfig | None
    targets: list[QuantTarget]
    quantized_targets: list[QuantizedTarget]
    unquantized_state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportResult:
    """Result returned after writing a checkpoint.

    Args:
        checkpoint_path: Path to the serialized checkpoint.
        metadata: Export metadata written or derived during serialization.
    """

    checkpoint_path: str
    metadata: dict[str, Any]
