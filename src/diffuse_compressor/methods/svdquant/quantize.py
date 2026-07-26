from __future__ import annotations

from dataclasses import replace

import torch

from ...artifact import QuantizedTarget
from ...config import (
    ActivationQuantSpec,
    DiffusionQuantSpec,
    NaiveSvdqLayout,
    SvdqTargetQuant,
    target_bias_policy,
    target_weight_layout,
    weight_layout_metadata,
)
from ...targets import QuantTarget
from ...quantize import (
    ProjectorQuantizer,
    ProjectorTargetContext,
    total_partition_rows,
)
from ...backends.nunchaku.layouts import (
    fake_quantize_weight,
    nunchaku_nvfp4_outer_scale_rows,
    nunchaku_target_shift,
    pack_projector_state,
    uses_nunchaku_packed_layout,
    weight_scales,
)
from .ranges import (
    activation_metadata,
    activation_quant_fn,
    add_range_state,
    calibrate_activation_range,
    calibrate_output_range,
    calibrate_range,
    range_metadata,
)
from .smoothing import resolve_input_partitions, select_smooth_scale, smooth_inputs
from .factorization import low_rank_branch
from .gptq import gptq_residual_weight
from .lowrank_search import search_low_rank_branch


class SvdqQuantizer(ProjectorQuantizer):
    """Projector quantizer for SVDQuant targets."""

    def quantize(self, context: ProjectorTargetContext) -> QuantizedTarget:
        """Quantize one prepared SVDQuant projector target."""

        logger = context.logger
        target = context.target
        spec = context.spec
        precision = target.quant.precision or spec.precision
        group_size = target.quant.group_size or spec.group_size
        rank = spec.rank if target.quant.rank is None else target.quant.rank
        smooth = spec.smooth if target.quant.smooth is None else target.quant.smooth
        activation_quant = spec.activation_quant
        if isinstance(target.quant.activation_quant, ActivationQuantSpec):
            activation_quant = target.quant.activation_quant
        elif isinstance(target.quant.activation_quant, bool):
            activation_quant = replace(spec.activation_quant, enabled=target.quant.activation_quant)
        target_spec = (
            spec
            if (
                precision == spec.precision
                and group_size == spec.group_size
                and rank == spec.rank
                and smooth == spec.smooth
                and activation_quant == spec.activation_quant
            )
            else replace(
                spec,
                precision=precision,
                group_size=group_size,
                rank=rank,
                smooth=smooth,
                activation_quant=activation_quant,
            )
        )
        weight = context.weight
        bias = context.bias
        calibration_inputs = context.calibration_inputs
        calibration_input_partitions = context.calibration_input_partitions
        target_cache = context.target_cache
        eval_replay = context.eval_replay
        calibration = context.calibration
        compute_device = context.compute_device

        partitions = resolve_input_partitions(calibration_inputs, calibration_input_partitions, calibration)
        if partitions:
            partition_rows = total_partition_rows(partitions)
            logger.info(
                "    - Calibrating input activation range from %d row%s across %d partition%s",
                partition_rows,
                "" if partition_rows == 1 else "s",
                len(partitions),
                "" if len(partitions) == 1 else "s",
            )
        input_range = (
            calibrate_activation_range(partitions, target_spec.activation_quant.inputs, target_spec)
            if target_spec.activation_quant.enabled
            else None
        )
        logger.info("    - Selecting smoothing scale")
        smooth, smooth_metadata = select_smooth_scale(
            target,
            target_spec,
            weight,
            bias,
            calibration_inputs,
            partitions,
            seed=0 if calibration is None or calibration.seed is None else calibration.seed,
            sample_batch_size=-1 if calibration is None else calibration.sample_batch_size,
            logger=logger,
        )
        quant_inputs = smooth_inputs(calibration_inputs, smooth) if calibration_inputs is not None else None
        quant_input_partitions = (
            tuple(smooth_inputs(partition, smooth) for partition in partitions) if partitions else None
        )
        smooth_weight = weight * smooth.view(1, -1)
        quantize_activation = (
            activation_quant_fn(input_range)
            if target_spec.activation_quant.enabled and input_range is not None
            else None
        )

        low_rank_metadata: dict[str, object] = {"mode": target_spec.low_rank_solver.mode}
        shared_low_rank = _target_shared_low_rank(target)
        if target_spec.rank > 0 and shared_low_rank and target_spec.low_rank_solver.mode == "search":
            logger.info(
                "    - Searching low-rank branch candidates: rank=%d, max_iters=%d, eval_replay=%s",
                target_spec.rank,
                target_spec.low_rank_solver.num_iters,
                target_spec.low_rank_solver.eval_replay,
            )
            search = search_low_rank_branch(
                target=target,
                weight=smooth_weight,
                bias=bias,
                inputs=quant_inputs,
                input_partitions=quant_input_partitions,
                spec=target_spec,
                eval_replay=eval_replay,
                compute_device=compute_device,
                low_rank_fn=lambda weight, rank, inputs: low_rank_branch(
                    weight, rank, inputs, solver=target_spec.low_rank_solver
                ),
                weight_scales_fn=weight_scales,
                fake_quant_weight_fn=fake_quantize_weight,
                activation_quant_fn=quantize_activation,
                logger=logger,
            )
            low_rank = search.low_rank
            quant_weight = search.residual
            low_rank_metadata = search.metadata
        else:
            if target_spec.rank > 0 and shared_low_rank:
                logger.info(
                    "    - Computing weighted SVD low-rank branch: rank=%d, backend=%s",
                    target_spec.rank,
                    target_spec.low_rank_solver.svd_backend,
                )
            else:
                logger.info("    - Skipping low-rank branch")
            low_rank = (
                low_rank_branch(
                    smooth_weight, rank=target_spec.rank, inputs=quant_inputs, solver=target_spec.low_rank_solver
                )
                if target_spec.rank > 0 and shared_low_rank
                else None
            )
            quant_weight = smooth_weight
            if low_rank is not None:
                quant_weight = smooth_weight - low_rank[1] @ low_rank[0]
            low_rank_metadata = {
                "mode": "weighted_svd",
                "iterations": 1 if low_rank is not None else 0,
                "eval_replay": False,
                "svd_backend": target_spec.low_rank_solver.svd_backend,
                "svd_lowrank_oversample": target_spec.low_rank_solver.svd_lowrank_oversample,
                "svd_lowrank_niter": target_spec.low_rank_solver.svd_lowrank_niter,
            }

        layout = target_weight_layout(target.quant)
        gptq_metadata: dict[str, object] = {"enabled": False}
        scale = None
        if target_spec.gptq.enabled:
            scale = weight_scales(
                quant_weight,
                group_size=target_spec.group_size,
                float_point=target_spec.precision == "fp4",
            )
            quant_weight, gptq_metadata = gptq_residual_weight(
                quant_weight,
                scale,
                quant_input_partitions or (),
                target_spec,
                logger=logger,
            )

        logger.info(
            "    - Packing residual weights: precision=%s, group_size=%d",
            target_spec.precision,
            target_spec.group_size,
        )
        if scale is None:
            scale = weight_scales(
                quant_weight,
                group_size=target_spec.group_size,
                float_point=target_spec.precision == "fp4",
            )
        if target_cache is not None and target_spec.activation_quant.enabled:
            logger.info("    - Calibrating output activation range")
        output_range = (
            calibrate_output_range(target_cache, target_spec) if target_spec.activation_quant.enabled else None
        )
        if target_spec.weight_range_calibration.enabled:
            logger.info("    - Calibrating weight range")
        weight_range = (
            calibrate_range((quant_weight,), target_spec.weight_range_calibration.range, target_spec, weight_like=True)
            if target_spec.weight_range_calibration.enabled
            else None
        )
        use_nunchaku_packed_export = uses_nunchaku_packed_layout(
            quant_weight, target_spec, low_rank
        ) and not isinstance(layout, NaiveSvdqLayout)
        nunchaku_shift = nunchaku_target_shift(target) if use_nunchaku_packed_export else None
        export_bias, export_shift = (
            _nunchaku_runtime_bias_and_shift(target, bias, nunchaku_shift)
            if use_nunchaku_packed_export
            else (bias, nunchaku_shift)
        )
        state_dict, weight_scale_layout, runtime_tensor_layout = pack_projector_state(
            quant_weight,
            scale,
            target_spec,
            target,
            smooth=smooth,
            bias=export_bias,
            low_rank=low_rank,
            shift=export_shift,
            outer_scale_rows=nunchaku_nvfp4_outer_scale_rows(target, quant_weight),
        )
        add_range_state(state_dict, "weight_range", weight_range)
        logger.info("    - Finished target %s", target.export_name)
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
                "weight_scale_layout": weight_scale_layout,
                "runtime_tensor_layout": runtime_tensor_layout,
                "compute_device": None if compute_device is None else str(compute_device),
                "calibrated": calibration_inputs is not None,
                "low_rank_solver": low_rank_metadata,
                "gptq": gptq_metadata,
                "smooth": smooth_metadata,
                "activation_quant": activation_metadata(target_spec, input_range, output_range),
                "weight_range_calibration": range_metadata(target_spec.weight_range_calibration.range, weight_range)
                | {"enabled": target_spec.weight_range_calibration.enabled},
                "weight_layout": weight_layout_metadata(layout),
            },
        )


def _target_shared_low_rank(target: QuantTarget) -> bool:
    if isinstance(target.quant, SvdqTargetQuant):
        return target.quant.shared_low_rank
    return False


def _nunchaku_runtime_bias_and_shift(
    target: QuantTarget, bias: torch.Tensor | None, shift: torch.Tensor | None
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Reject Nunchaku exports that would drop activation shift compensation."""

    policy = target_bias_policy(target.quant)
    if policy == "omit":
        if shift is not None:
            raise RuntimeError(
                f"Nunchaku-packed target {target.export_name!r} cannot omit bias while using activation shift; "
                "disable activation shift or use the auto/zero bias policy"
            )
        return None, None
    return bias, shift
