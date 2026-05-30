from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from ...artifact import QuantizedTarget
from ...calibration import EvalReplayBatch, IOTensorsCache
from ...config import (
    ActivationQuantSpec,
    AdaNormAwqW4A16Layout,
    AwqW4A16Layout,
    CalibrationSpec,
    DiffusionQuantSpec,
    weight_layout_metadata,
)
from ...logging import QuantizationLogger
from ...patches import ShiftedConv2d, ShiftedLinear
from ...targets import QuantTarget
from ...backends.nunchaku.layouts import (
    fake_quantize_weight,
    nunchaku_nvfp4_outer_scale_rows,
    nunchaku_target_shift,
    pack_awq_w4a16_target,
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
from .smoothing import (
    resolve_input_partitions,
    select_smooth_scale,
    smooth_inputs,
)
from .factorization import low_rank_branch
from .lowrank_search import search_low_rank_branch


@torch.inference_mode()
def quantize_targets(
    targets: Iterable[QuantTarget],
    spec: DiffusionQuantSpec,
    calibration_inputs: dict[str, torch.Tensor] | None = None,
    calibration_input_partitions: dict[str, tuple[torch.Tensor, ...]] | None = None,
    layer_cache: dict[str, IOTensorsCache] | None = None,
    eval_replay: EvalReplayBatch | Sequence[EvalReplayBatch] | None = None,
    calibration: CalibrationSpec | None = None,
    logger: QuantizationLogger | None = None,
) -> list[QuantizedTarget]:
    """Quantize concrete SVDQuant targets.

    Args:
        targets: Concrete targets to quantize.
        spec: Quantization settings.
        calibration_inputs: Optional target input rows by export name.
        calibration_input_partitions: Optional partitioned input rows.
        layer_cache: Optional rich module I/O caches by export name.
        eval_replay: Optional eval replay batches for search-based low rank.
        calibration: Optional calibration settings for repartitioning.
        logger: Optional explicit quantization logger.

    Returns:
        Quantized target artifacts.
    """

    logger = logger or QuantizationLogger()
    targets = list(targets)
    compute_device = _resolve_compute_device(spec.compute_device)
    logger.info("- Quantizing %d SVDQuant targets", len(targets))
    if compute_device is not None:
        logger.info("- Using %s for per-target quantization compute", compute_device)
    quantized: list[QuantizedTarget] = []
    for index, target in enumerate(targets, start=1):
        if target.kind not in {"linear", "conv"}:
            raise NotImplementedError(
                f"SVDQuant currently supports linear and pointwise conv targets, got {target.kind!r}"
            )
        logger.info(
            "  + Target %d/%d: %s (%d module%s)",
            index,
            len(targets),
            target.export_name,
            len(target.modules),
            "" if len(target.modules) == 1 else "s",
        )
        inputs = calibration_inputs.get(target.export_name) if calibration_inputs is not None else None
        input_partitions = (
            calibration_input_partitions.get(target.export_name)
            if calibration_input_partitions is not None and target.export_name in calibration_input_partitions
            else None
        )
        target_cache = layer_cache.get(target.export_name) if layer_cache is not None else None
        target_started_at = logger.start_timing()
        try:
            quantized_target = _quantize_projector_target(
                target,
                spec,
                inputs,
                input_partitions,
                target_cache,
                eval_replay,
                calibration,
                compute_device=compute_device,
                logger=logger,
            )
            quantized.append(quantized_target)
            logger.stop_timing(quantized_target, target_started_at)
        finally:
            _clear_cuda_cache(compute_device)
    return quantized


def _quantize_projector_target(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    calibration_inputs: torch.Tensor | None = None,
    calibration_input_partitions: tuple[torch.Tensor, ...] | None = None,
    target_cache: IOTensorsCache | None = None,
    eval_replay: EvalReplayBatch | Sequence[EvalReplayBatch] | None = None,
    calibration: CalibrationSpec | None = None,
    *,
    compute_device: torch.device | None = None,
    logger: QuantizationLogger | None = None,
) -> QuantizedTarget:
    """Quantize one linear/pointwise-conv projector target."""

    logger = logger or QuantizationLogger()
    target_spec = _target_spec(spec, target)
    modules = _projector_modules(target)
    source_weight = _projector_weight(modules[0])
    work_device = compute_device or source_weight.device
    weight = torch.cat([_projector_weight(module).to(device=work_device) for module in modules], dim=0)
    bias = _concat_bias(modules, weight.device, weight.dtype, policy=target.export_bias)
    export_dtype = torch.bfloat16 if weight.dtype not in (torch.float16, torch.bfloat16) else weight.dtype
    weight = weight.to(device=work_device, dtype=export_dtype)
    if bias is not None:
        bias = bias.to(device=work_device, dtype=export_dtype)

    partitions = resolve_input_partitions(calibration_inputs, calibration_input_partitions, calibration)
    if partitions:
        logger.info(
            "    - Calibrating input activation range from %d partition%s",
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
    )
    quant_inputs = smooth_inputs(calibration_inputs, smooth) if calibration_inputs is not None else None
    quant_input_partitions = tuple(smooth_inputs(partition, smooth) for partition in partitions) if partitions else None
    smooth_weight = weight * smooth.view(1, -1)
    quantize_activation = (
        activation_quant_fn(input_range)
        if target_spec.activation_quant.enabled and input_range is not None
        else None
    )

    low_rank_metadata: dict[str, object] = {"mode": target_spec.low_rank_solver.mode}
    if target_spec.rank > 0 and target.shared_low_rank and target_spec.low_rank_solver.mode == "search":
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
                weight,
                rank,
                inputs,
                solver=target_spec.low_rank_solver,
            ),
            weight_scales_fn=weight_scales,
            fake_quant_weight_fn=fake_quantize_weight,
            activation_quant_fn=quantize_activation,
        )
        low_rank = search.low_rank
        quant_weight = search.residual
        low_rank_metadata = search.metadata
    else:
        if target_spec.rank > 0 and target.shared_low_rank:
            logger.info(
                "    - Computing weighted SVD low-rank branch: rank=%d, backend=%s",
                target_spec.rank,
                target_spec.low_rank_solver.svd_backend,
            )
        else:
            logger.info("    - Skipping low-rank branch")
        low_rank = (
            low_rank_branch(smooth_weight, rank=target_spec.rank, inputs=quant_inputs, solver=target_spec.low_rank_solver)
            if target_spec.rank > 0 and target.shared_low_rank
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

    logger.info(
        "    - Packing residual weights: precision=%s, group_size=%d", target_spec.precision, target_spec.group_size
    )
    if isinstance(target.weight_layout, (AwqW4A16Layout, AdaNormAwqW4A16Layout)):
        state_dict = pack_awq_w4a16_target(target, target_spec, quant_weight, bias)
        output_range = None
        weight_range = None
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
                "compute_device": None if compute_device is None else str(compute_device),
                "calibrated": calibration_inputs is not None,
                "low_rank_solver": low_rank_metadata,
                "smooth": smooth_metadata,
                "activation_quant": activation_metadata(target_spec, input_range, output_range),
                "weight_range_calibration": range_metadata(target_spec.weight_range_calibration.range, weight_range)
                | {"enabled": target_spec.weight_range_calibration.enabled},
                "weight_layout": weight_layout_metadata(target.weight_layout),
            },
        )
    scale = weight_scales(quant_weight, group_size=target_spec.group_size, float_point=target_spec.precision == "fp4")
    if target_cache is not None and target_spec.activation_quant.enabled:
        logger.info("    - Calibrating output activation range")
    output_range = calibrate_output_range(target_cache, target_spec) if target_spec.activation_quant.enabled else None
    if target_spec.weight_range_calibration.enabled:
        logger.info("    - Calibrating weight range")
    weight_range = (
        calibrate_range((quant_weight,), target_spec.weight_range_calibration.range, target_spec, weight_like=True)
        if target_spec.weight_range_calibration.enabled
        else None
    )
    nunchaku_shift = (
        nunchaku_target_shift(target) if uses_nunchaku_packed_layout(quant_weight, target_spec, low_rank) else None
    )
    state_dict, weight_scale_layout, runtime_tensor_layout = pack_projector_state(
        quant_weight,
        scale,
        target_spec,
        smooth=smooth,
        bias=bias,
        low_rank=low_rank,
        shift=nunchaku_shift,
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
            "smooth": smooth_metadata,
            "activation_quant": activation_metadata(target_spec, input_range, output_range),
            "weight_range_calibration": range_metadata(target_spec.weight_range_calibration.range, weight_range)
            | {"enabled": target_spec.weight_range_calibration.enabled},
            "weight_layout": weight_layout_metadata(target.weight_layout),
        },
    )


ProjectorModule = nn.Linear | nn.Conv2d


def _target_spec(spec: DiffusionQuantSpec, target: QuantTarget) -> DiffusionQuantSpec:
    """Apply target-level quantization overrides."""

    precision = target.precision or spec.precision
    group_size = target.group_size or spec.group_size
    rank = spec.rank if target.rank is None else target.rank
    smooth = spec.smooth if target.smooth is None else target.smooth
    activation_quant = spec.activation_quant
    if isinstance(target.activation_quant, ActivationQuantSpec):
        activation_quant = target.activation_quant
    elif isinstance(target.activation_quant, bool):
        activation_quant = replace(spec.activation_quant, enabled=target.activation_quant)
    shift_activations = spec.shift_activations if target.shift_activations is None else target.shift_activations
    if (
        precision == spec.precision
        and group_size == spec.group_size
        and rank == spec.rank
        and smooth == spec.smooth
        and activation_quant == spec.activation_quant
        and shift_activations == spec.shift_activations
    ):
        return spec
    return replace(
        spec,
        precision=precision,
        group_size=group_size,
        rank=rank,
        smooth=smooth,
        activation_quant=activation_quant,
        shift_activations=shift_activations,
    )


def _resolve_compute_device(device: str | None) -> torch.device | None:
    """Validate and resolve the optional per-target compute device."""

    if device is None:
        return None
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"compute_device {device!r} requires CUDA, but CUDA is not available")
    return resolved


def _clear_cuda_cache(device: torch.device | None) -> None:
    """Release cached CUDA blocks after one target finishes."""

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _projector_modules(target: QuantTarget) -> list[ProjectorModule]:
    """Resolve projector modules from raw or shifted target wrappers."""

    modules: list[ProjectorModule] = []
    for module in target.modules:
        if isinstance(module, ShiftedLinear):
            modules.append(module.linear)
        elif isinstance(module, ShiftedConv2d):
            modules.append(module.conv)
        elif isinstance(module, nn.Linear):
            modules.append(module)
        elif isinstance(module, nn.Conv2d):
            modules.append(module)
    if target.kind == "conv":
        for module in modules:
            _validate_pointwise_conv(module)
    return modules


def _validate_pointwise_conv(module: ProjectorModule) -> None:
    """Validate that a projector module is a supported pointwise Conv2d."""

    if not isinstance(module, nn.Conv2d):
        return
    if module.kernel_size != (1, 1) or module.groups != 1:
        raise NotImplementedError(
            "SVDQuant Conv2d targets currently require kernel_size=(1, 1) and groups=1 "
            f"(got kernel_size={module.kernel_size}, groups={module.groups})"
        )


def _projector_weight(module: ProjectorModule) -> torch.Tensor:
    """Return a projector weight matrix in ``[out, in]`` layout."""

    if isinstance(module, nn.Conv2d):
        _validate_pointwise_conv(module)
        return module.weight.detach().flatten(1)
    return module.weight.detach()


def _concat_bias(
    modules: list[ProjectorModule],
    device: torch.device,
    dtype: torch.dtype,
    *,
    policy: str = "auto",
) -> torch.Tensor | None:
    """Concatenate biases from grouped projector modules."""

    if policy == "omit":
        return None
    if policy not in {"auto", "zero"}:
        raise ValueError(f"Unsupported target export_bias policy: {policy!r}")
    if policy == "auto" and all(module.bias is None for module in modules):
        return None
    return torch.cat(
        [
            module.bias.detach().to(device=device, dtype=dtype)
            if module.bias is not None
            else torch.zeros(_projector_out_features(module), device=device, dtype=dtype)
            for module in modules
        ],
        dim=0,
    )


def _projector_out_features(module: ProjectorModule) -> int:
    """Return output feature/channel count for one projector."""

    return module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
