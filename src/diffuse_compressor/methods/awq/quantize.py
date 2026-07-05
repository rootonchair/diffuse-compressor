from __future__ import annotations

from dataclasses import replace

import torch

from ...backends.nunchaku.layouts import pack_awq_w4a16_target
from ...artifact import QuantizedTarget
from ...config import (
    AwqTargetQuant,
    DiffusionQuantSpec,
    WeightRangeCalibrationSpec,
    weight_layout_metadata,
    target_weight_layout,
)
from ...targets import QuantTarget
from ...quantize import ProjectorQuantizer, ProjectorTargetContext


def pack_awq_target(
    target: QuantTarget, spec: DiffusionQuantSpec, weight: torch.Tensor, bias: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """Pack an AWQ target into its configured runtime layout."""

    return pack_awq_w4a16_target(target, spec, weight, bias)


class AwqQuantizer(ProjectorQuantizer):
    """Projector quantizer for AWQ W4A16 targets."""

    def quantize(self, context: ProjectorTargetContext) -> QuantizedTarget:
        """Quantize one prepared AWQ W4A16 projector target."""

        target = context.target
        target_spec = replace(
            context.spec,
            precision="int4",
            group_size=64,
            rank=0,
            smooth=False,
            activation_quant=replace(context.spec.activation_quant, enabled=False),
            weight_range_calibration=WeightRangeCalibrationSpec(enabled=False),
        )
        state_dict = pack_awq_target(target, target_spec, context.weight, context.bias)
        context.logger.info("    - Packed AWQ W4A16 weights: group_size=%d", target_spec.group_size)
        context.logger.info("    - Finished target %s", target.export_name)
        return QuantizedTarget(
            target=target,
            state_dict=state_dict,
            metadata={
                "source_modules": list(target.module_names),
                "roles": list(target.roles),
                "rank": target_spec.rank,
                "precision": target_spec.precision,
                "group_size": target_spec.group_size,
                "weight_scale_dtypes": list(target_spec.weight_scale_dtypes),
                "compute_device": None if context.compute_device is None else str(context.compute_device),
                "calibrated": context.calibration_inputs is not None,
                "low_rank_solver": {"mode": target_spec.low_rank_solver.mode, "iterations": 0},
                "smooth": {"enabled": False},
                "activation_quant": {"enabled": False},
                "weight_range_calibration": {"enabled": False},
                "weight_layout": weight_layout_metadata(target_weight_layout(target.quant)),
            },
        )
