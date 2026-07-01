from __future__ import annotations

from dataclasses import replace

import torch

from ...backends.nunchaku.layouts import pack_awq_w4a16_target
from ...config import AwqTargetQuant, DiffusionQuantSpec, WeightRangeCalibrationSpec
from ...targets import QuantTarget


def is_awq_target(target: QuantTarget) -> bool:
    """Return whether a target uses the AWQ quantization method."""

    return isinstance(target.quant, AwqTargetQuant)


def awq_target_spec(spec: DiffusionQuantSpec) -> DiffusionQuantSpec:
    """Resolve the fixed quantization behavior for an AWQ target."""

    return replace(
        spec,
        precision="int4",
        group_size=64,
        rank=0,
        smooth=False,
        activation_quant=replace(spec.activation_quant, enabled=False),
        weight_range_calibration=WeightRangeCalibrationSpec(enabled=False),
    )


def pack_awq_target(
    target: QuantTarget, spec: DiffusionQuantSpec, weight: torch.Tensor, bias: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """Pack an AWQ target into its configured runtime layout."""

    return pack_awq_w4a16_target(target, spec, weight, bias)
