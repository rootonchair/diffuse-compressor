from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .config import DiffusionQuantSpec, TargetConfig
from .targets import QuantTarget


@dataclass
class QuantizedTarget:
    target: QuantTarget
    state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizedArtifact:
    spec: DiffusionQuantSpec
    target_config: TargetConfig | None
    targets: list[QuantTarget]
    quantized_targets: list[QuantizedTarget]
    unquantized_state_dict: dict[str, torch.Tensor]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExportResult:
    checkpoint_path: str
    metadata: dict[str, Any]
