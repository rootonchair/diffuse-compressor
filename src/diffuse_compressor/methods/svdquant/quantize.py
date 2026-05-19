from __future__ import annotations

import logging
from dataclasses import replace
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from ...artifact import QuantizedTarget
from ...calibration import EvalReplayBatch, IOTensorsCache, repartition_tensor
from ...config import (
    ActivationQuantSpec,
    AdaNormAwqW4A16Layout,
    AwqW4A16Layout,
    CalibrationSpec,
    DiffusionQuantSpec,
    LowRankSolverSpec,
    NunchakuSvdqLayout,
    RangeCalibrationSpec,
    weight_layout_metadata,
)
from ...patches import ShiftedConv2d, ShiftedLinear
from ...targets import QuantTarget
from .lowrank_search import search_low_rank_branch
from .packing import NunchakuWeightPacker, fp4_e2m1_codebook, fp_quantize
from .smoothing import SmoothCandidate, iter_smooth_candidates, resolve_smooth_spec


logger = logging.getLogger(__name__)


@torch.inference_mode()
def quantize_targets(
    targets: Iterable[QuantTarget],
    spec: DiffusionQuantSpec,
    calibration_inputs: dict[str, torch.Tensor] | None = None,
    calibration_input_partitions: dict[str, tuple[torch.Tensor, ...]] | None = None,
    layer_cache: dict[str, IOTensorsCache] | None = None,
    eval_replay: EvalReplayBatch | Sequence[EvalReplayBatch] | None = None,
    calibration: CalibrationSpec | None = None,
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

    Returns:
        Quantized target artifacts.
    """

    targets = list(targets)
    compute_device = _resolve_compute_device(spec.compute_device)
    logger.info("- Quantizing %d SVDQuant targets", len(targets))
    if compute_device is not None:
        logger.info("- Using %s for per-target quantization compute", compute_device)
    quantized: list[QuantizedTarget] = []
    for index, target in enumerate(targets, start=1):
        if target.kind not in {"linear", "conv"}:
            raise NotImplementedError(f"SVDQuant currently supports linear and pointwise conv targets, got {target.kind!r}")
        logger.info("  + Target %d/%d: %s (%d module%s)", index, len(targets), target.export_name, len(target.modules), "" if len(target.modules) == 1 else "s")
        inputs = calibration_inputs.get(target.export_name) if calibration_inputs is not None else None
        input_partitions = (
            calibration_input_partitions.get(target.export_name)
            if calibration_input_partitions is not None and target.export_name in calibration_input_partitions
            else None
        )
        target_cache = layer_cache.get(target.export_name) if layer_cache is not None else None
        try:
            quantized.append(
                _quantize_projector_target(
                    target,
                    spec,
                    inputs,
                    input_partitions,
                    target_cache,
                    eval_replay,
                    calibration,
                    compute_device=compute_device,
                )
            )
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
) -> QuantizedTarget:
    """Quantize one linear/pointwise-conv projector target.

    Args:
        target: Concrete target whose modules are linear or pointwise Conv2d
            layers.
        spec: Quantization settings.
        calibration_inputs: Optional concatenated input rows.
        calibration_input_partitions: Optional input row partitions.
        target_cache: Optional rich I/O cache for the target.
        eval_replay: Optional eval-module replay records for search scoring.
        calibration: Optional calibration settings.

    Returns:
        Quantized target with Nunchaku Lite tensor names.
    """

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

    partitions = _resolve_input_partitions(calibration_inputs, calibration_input_partitions, calibration)
    if partitions:
        logger.info("    - Calibrating input activation range from %d partition%s", len(partitions), "" if len(partitions) == 1 else "s")
    input_range = (
        _calibrate_activation_range(partitions, target_spec.activation_quant.inputs, target_spec)
        if target_spec.activation_quant.enabled
        else None
    )
    logger.info("    - Selecting smoothing scale")
    smooth, smooth_metadata = _select_smooth_scale(target, target_spec, weight, bias, calibration_inputs, partitions)
    quant_inputs = _smooth_inputs(calibration_inputs, smooth) if calibration_inputs is not None else None
    quant_input_partitions = tuple(_smooth_inputs(partition, smooth) for partition in partitions) if partitions else None
    smooth_weight = weight * smooth.view(1, -1)
    activation_quant_fn = _activation_quant_fn(input_range) if target_spec.activation_quant.enabled and input_range is not None else None

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
            low_rank_fn=lambda weight, rank, inputs: _low_rank_branch(
                weight,
                rank,
                inputs,
                solver=target_spec.low_rank_solver,
            ),
            weight_scales_fn=_weight_scales,
            fake_quant_weight_fn=_fake_quantize_weight,
            activation_quant_fn=activation_quant_fn,
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
            _low_rank_branch(smooth_weight, rank=target_spec.rank, inputs=quant_inputs, solver=target_spec.low_rank_solver)
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

    logger.info("    - Packing residual weights: precision=%s, group_size=%d", target_spec.precision, target_spec.group_size)
    if isinstance(target.weight_layout, (AwqW4A16Layout, AdaNormAwqW4A16Layout)):
        state_dict = _pack_awq_w4a16_target(target, target_spec, quant_weight, bias)
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
                "activation_quant": _activation_metadata(target_spec, input_range, output_range),
                "weight_range_calibration": _range_metadata(target_spec.weight_range_calibration.range, weight_range)
                | {"enabled": target_spec.weight_range_calibration.enabled},
                "weight_layout": weight_layout_metadata(target.weight_layout),
            },
        )
    scale = _weight_scales(quant_weight, group_size=target_spec.group_size, float_point=target_spec.precision == "fp4")
    if target_cache is not None and target_spec.activation_quant.enabled:
        logger.info("    - Calibrating output activation range")
    output_range = _calibrate_output_range(target_cache, target_spec) if target_spec.activation_quant.enabled else None
    if target_spec.weight_range_calibration.enabled:
        logger.info("    - Calibrating weight range")
    weight_range = (
        _calibrate_range((quant_weight,), target_spec.weight_range_calibration.range, target_spec, weight_like=True)
        if target_spec.weight_range_calibration.enabled
        else None
    )
    nunchaku_shift = (
        _nunchaku_target_shift(target) if _uses_nunchaku_packed_layout(quant_weight, target_spec, low_rank) else None
    )
    state_dict, weight_scale_layout, runtime_tensor_layout = _pack_projector_state(
        quant_weight,
        scale,
        target_spec,
        smooth=smooth,
        bias=bias,
        low_rank=low_rank,
        shift=nunchaku_shift,
        outer_scale_rows=_nunchaku_nvfp4_outer_scale_rows(target, quant_weight),
    )
    _add_range_state(state_dict, "weight_range", weight_range)
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
            "activation_quant": _activation_metadata(target_spec, input_range, output_range),
            "weight_range_calibration": _range_metadata(target_spec.weight_range_calibration.range, weight_range)
            | {"enabled": target_spec.weight_range_calibration.enabled},
            "weight_layout": weight_layout_metadata(target.weight_layout),
        },
    )


def _target_spec(spec: DiffusionQuantSpec, target: QuantTarget) -> DiffusionQuantSpec:
    """Apply target-level quantization overrides.

    Args:
        spec: Global quantization settings.
        target: Concrete target that may carry overrides.

    Returns:
        Effective quantization settings for the target.
    """

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


ProjectorModule = nn.Linear | nn.Conv2d


def _projector_modules(target: QuantTarget) -> list[ProjectorModule]:
    """Resolve projector modules from raw or shifted target wrappers.

    Args:
        target: Concrete target.

    Returns:
        Linear or pointwise Conv2d child modules in export order.
    """

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
    """Validate that a projector module is a supported pointwise Conv2d.

    Args:
        module: Linear or Conv2d module.
    """

    if not isinstance(module, nn.Conv2d):
        return
    if module.kernel_size != (1, 1) or module.groups != 1:
        raise NotImplementedError(
            "SVDQuant Conv2d targets currently require kernel_size=(1, 1) and groups=1 "
            f"(got kernel_size={module.kernel_size}, groups={module.groups})"
        )


def _projector_weight(module: ProjectorModule) -> torch.Tensor:
    """Return a projector weight matrix in ``[out, in]`` layout.

    Args:
        module: Linear or pointwise Conv2d module.

    Returns:
        Two-dimensional detached weight matrix.
    """

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
    """Concatenate biases from grouped projector modules.

    Args:
        modules: Linear or pointwise Conv2d modules in export order.
        device: Device for synthesized zero-bias tensors.
        dtype: Dtype for synthesized zero-bias tensors.
        policy: Bias export policy. ``"auto"`` omits bias only when every
            module is biasless, ``"zero"`` exports synthesized zeros for
            biasless modules, and ``"omit"`` always returns ``None``.

    Returns:
        Concatenated bias, or ``None`` when all modules are biasless.
    """

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
    """Return output feature/channel count for one projector.

    Args:
        module: Linear or pointwise Conv2d module.

    Returns:
        Output feature count.
    """

    return module.out_channels if isinstance(module, nn.Conv2d) else module.out_features


def _weight_scales(weight: torch.Tensor, group_size: int, float_point: bool) -> torch.Tensor:
    """Compute per-output, per-group residual weight scales.

    Args:
        weight: Residual weight matrix in ``[out, in]`` layout.
        group_size: Number of input features per quantization group.
        float_point: Whether FP4 scale bounds should be used.

    Returns:
        Scale tensor in Nunchaku Lite grouped layout.
    """

    oc, ic = weight.shape
    if ic % group_size != 0:
        raise ValueError(f"Input features ({ic}) must be divisible by group_size ({group_size}) for Nunchaku export")
    groups = ic // group_size
    max_q = 6 if float_point else 7
    scale = weight.float().view(oc, groups, group_size).abs().amax(dim=2).clamp_min(1e-6) / max_q
    return scale.to(dtype=weight.dtype).view(oc, 1, groups, 1)


def _uses_nvfp4_split_scales(spec: DiffusionQuantSpec) -> bool:
    """Return whether a spec should export DeepCompressor-style NVFP4 scales."""

    return (
        spec.precision == "fp4"
        and spec.group_size == 16
        and tuple(spec.weight_scale_dtypes) == (None, "sfp8_e4m3_nan")
    )


def _pack_projector_state(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: DiffusionQuantSpec,
    *,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
    shift: torch.Tensor | None,
    outer_scale_rows: tuple[int, ...] | None,
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Pack residual projector tensors for export."""

    if _uses_nunchaku_packed_layout(weight, spec, low_rank):
        state, weight_scale_layout = _pack_nunchaku_projector_state(
            weight,
            scale,
            spec,
            smooth,
            bias,
            low_rank,
            shift=shift,
            outer_scale_rows=outer_scale_rows,
        )
        return state, weight_scale_layout, "nunchaku_packed"
    state, weight_scale_layout = _pack_logical_projector_state(weight, scale, spec, smooth, bias, low_rank)
    return state, weight_scale_layout, "logical"


def _uses_nunchaku_packed_layout(
    weight: torch.Tensor,
    spec: DiffusionQuantSpec,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
) -> bool:
    """Return whether packing preserves shapes expected by Nunchaku Lite."""

    packer = NunchakuWeightPacker(bits=4)
    oc, ic = weight.shape
    if oc % packer.mem_n != 0 or ic % (packer.mem_k * packer.num_k_unrolls) != 0:
        return False
    if low_rank is None:
        return True
    rank = low_rank[0].shape[0]
    low_rank_n = packer.n_pack_size * packer.num_n_lanes
    low_rank_k = packer.k_pack_size * packer.num_k_lanes * 2
    return rank % low_rank_n == 0 and ic % low_rank_k == 0 and oc % low_rank_n == 0


def _pack_logical_projector_state(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: DiffusionQuantSpec,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[dict[str, torch.Tensor], str]:
    """Pack tensors in the legacy logical layout used by tests and torch-dequant."""

    if _uses_nvfp4_split_scales(spec):
        qweight, scale_state = _pack_nvfp4_linear_weight(weight, scale)
        weight_scale_layout = "nvfp4_deepcompressor"
    else:
        qweight, wscales = _pack_linear_weight(weight, scale, float_point=spec.precision == "fp4")
        scale_state = {"wscales": wscales}
        weight_scale_layout = "effective"
    state_dict = {
        "qweight": qweight,
        "smooth_factor": smooth.detach().cpu(),
        "smooth_factor_orig": smooth.detach().cpu().clone(),
        **scale_state,
    }
    if bias is not None:
        state_dict["bias"] = bias.detach().cpu()
    if low_rank is not None:
        state_dict["proj_down"] = low_rank[0].t().contiguous().cpu()
        state_dict["proj_up"] = low_rank[1].contiguous().cpu()
    return state_dict, weight_scale_layout


def _pack_nunchaku_projector_state(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: DiffusionQuantSpec,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
    shift: torch.Tensor | None,
    *,
    outer_scale_rows: tuple[int, ...] | None,
) -> tuple[dict[str, torch.Tensor], str]:
    """Pack tensors in the Nunchaku W4A4 kernel layout."""

    if _uses_nvfp4_split_scales(spec):
        scale0, scale1 = _nvfp4_scale_leaves(weight, scale, outer_scale_rows=outer_scale_rows)
        state_dict = _pack_nunchaku_w4a4_state(
            weight,
            scale0,
            smooth,
            bias,
            low_rank,
            float_point=True,
            subscale=scale1,
            shift=shift,
        )
        return state_dict, "nvfp4_deepcompressor"
    state_dict = _pack_nunchaku_w4a4_state(
        weight,
        scale,
        smooth,
        bias,
        low_rank,
        float_point=spec.precision == "fp4",
        subscale=None,
        shift=shift,
    )
    return state_dict, "effective"


def _pack_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    float_point: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack residual weights into Nunchaku Lite INT4 layout.

    Args:
        weight: Residual weight matrix in ``[out, in]`` layout.
        scale: Per-group scales from :func:`_weight_scales`.
        float_point: Whether FP4 packing is requested.

    Returns:
        Packed int8 qweight tensor and CPU scale tensor.
    """

    oc, ic = weight.shape
    groups = scale.shape[2]
    group_size = ic // groups
    scaled = weight.float().view(oc, groups, group_size) / scale.float().view(oc, groups, 1)
    if float_point:
        qweight = fp_quantize(scaled.view(oc, ic)).to(torch.int16)
    else:
        qweight = scaled.round_().clamp_(-8, 7).to(torch.int16).view(oc, ic)
    lo = qweight[:, 0::2].bitwise_and(0xF)
    hi = qweight[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    packed = lo.bitwise_or(hi).to(torch.uint8).view(torch.int8).contiguous()
    wscales = scale.view(oc, groups).t().contiguous().cpu()
    return packed.cpu(), wscales


def _pack_nunchaku_w4a4_state(
    weight: torch.Tensor,
    scale: torch.Tensor,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    float_point: bool,
    subscale: torch.Tensor | None,
    shift: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Pack a target following DeepCompressor's Nunchaku W4A4 converter."""

    oc, ic = weight.shape
    per_tensor_scale = scale.numel() == 1
    if per_tensor_scale:
        groups = 1
        scale_for_quant = scale.view(1).expand(oc).reshape(oc, 1, 1, 1)
    else:
        groups = scale.shape[2]
        scale_for_quant = scale
    group_size = ic // groups
    if subscale is not None:
        subgroups = subscale.shape[2]
        subgroup_size = ic // subgroups
    else:
        subgroup_size = group_size
    scaled = weight.float().view(oc, groups, group_size).div(scale_for_quant.float().view(oc, groups, 1))
    if subscale is not None:
        scaled = scaled.view(oc, subgroups, subgroup_size).div(subscale.float().view(oc, subgroups, 1))
    if float_point:
        qweight = fp_quantize(scaled.view(oc, ic)).to(torch.int32)
    else:
        qweight = scaled.round_().clamp_(-8, 7).to(torch.int32).view(oc, ic)

    packer = NunchakuWeightPacker(bits=4)
    packed_weight = packer.pack_weight(packer.pad_weight(qweight)).cpu()
    packed_scale = packer.pack_scale(
        packer.pad_scale(scale_for_quant.to(dtype=weight.dtype), group_size=group_size),
        group_size=group_size if group_size < ic else -1,
    ).cpu()
    packed_subscale = (
        packer.pack_scale(
            packer.pad_scale(subscale.to(dtype=weight.dtype), group_size=subgroup_size),
            group_size=subgroup_size if subgroup_size < ic else -1,
        ).cpu()
        if subscale is not None
        else None
    )
    packed_smooth = packer.pack_scale(
        packer.pad_scale(smooth.view(-1, 1).to(dtype=weight.dtype), group_size=-1),
        group_size=-1,
    ).cpu()
    state_dict = {
        "qweight": packed_weight,
        "wscales": packed_subscale if packed_subscale is not None else packed_scale,
        "smooth_factor": packed_smooth,
        "smooth_factor_orig": packed_smooth.clone(),
    }
    if packed_subscale is not None:
        if scale.numel() == 1:
            state_dict["wtscale"] = packed_scale.view(-1)[0].view(1)
            state_dict["wcscales"] = torch.ones(oc, dtype=packed_scale.dtype, device=packed_scale.device)
        else:
            state_dict["wcscales"] = packed_scale
    if bias is not None:
        state_dict["bias"] = packer.pack_scale(
            packer.pad_scale(bias.view(-1, 1).to(dtype=weight.dtype), group_size=-1),
            group_size=-1,
        ).cpu()
    if low_rank is not None:
        proj_down = low_rank[0]
        proj_down_for_bias = proj_down.to(dtype=torch.float64)
        if smooth is not None:
            proj_down_for_bias = proj_down_for_bias.div(smooth.to(dtype=torch.float64).view(1, -1))
            proj_down = proj_down_for_bias.to(dtype=weight.dtype)
        if shift is not None:
            shift_vector = _expand_shift_for_nunchaku(shift, ic).to(device=weight.device, dtype=torch.float64)
            bias_base = torch.zeros(oc, dtype=torch.float64, device=weight.device) if bias is None else bias.to(
                device=weight.device,
                dtype=torch.float64,
            )
            correction = low_rank[1].to(dtype=torch.float64) @ proj_down_for_bias @ shift_vector.view(-1, 1)
            bias = (bias_base + correction.view(-1)).to(dtype=weight.dtype)
            state_dict["bias"] = packer.pack_scale(
                packer.pad_scale(bias.view(-1, 1), group_size=-1),
                group_size=-1,
            ).cpu()
        state_dict["proj_down"] = packer.pack_lowrank_weight(proj_down, down=True).cpu()
        state_dict["proj_up"] = packer.pack_lowrank_weight(low_rank[1], down=False).cpu()
    return state_dict


def _nunchaku_target_shift(target: QuantTarget) -> torch.Tensor | None:
    """Return the common DeepCompressor-style shift for a packed target."""

    shifts: list[torch.Tensor | None] = []
    for module in target.modules:
        if isinstance(module, ShiftedLinear):
            shifts.append(module.shift.detach().cpu())
        elif isinstance(module, ShiftedConv2d):
            shifts.append(module.shift.detach().cpu())
        else:
            shifts.append(None)
    if all(shift is None for shift in shifts):
        return None
    if any(shift is None for shift in shifts):
        raise ValueError(f"Nunchaku-packed target {target.export_name!r} mixes shifted and unshifted modules")
    first = shifts[0]
    assert first is not None
    for shift in shifts[1:]:
        assert shift is not None
        if shift.shape != first.shape or not torch.equal(shift, first):
            raise ValueError(f"Nunchaku-packed target {target.export_name!r} has inconsistent activation shifts")
    return first


def _expand_shift_for_nunchaku(shift: torch.Tensor, in_features: int) -> torch.Tensor:
    """Expand a scalar or repeated shift to one value per input feature."""

    shift = shift.flatten()
    if shift.numel() == in_features:
        return shift
    if shift.numel() == 1:
        return shift.expand(in_features)
    if in_features % shift.numel() == 0:
        return shift.view(-1, 1).expand(-1, in_features // shift.numel()).flatten()
    raise ValueError(f"shift length {shift.numel()} does not divide input feature count {in_features}")


def _nunchaku_nvfp4_outer_scale_rows(target: QuantTarget, weight: torch.Tensor) -> tuple[int, ...] | None:
    """Return fused source row chunks for DeepCompressor-style NVFP4 outer scales."""

    if isinstance(target.weight_layout, NunchakuSvdqLayout) and target.weight_layout.outer_scale_splits:
        rows = tuple(target.weight_layout.outer_scale_splits)
        if sum(rows) != weight.shape[0]:
            raise ValueError(
                f"Target {target.export_name!r} outer_scale_splits sum to {sum(rows)}, "
                f"but weight has {weight.shape[0]} output rows"
            )
        return rows
    if len(target.modules) <= 1:
        return None
    rows = tuple(_target_module_out_features(module, target.kind) for module in target.modules)
    return rows if sum(rows) == weight.shape[0] else None


def _target_module_out_features(module: nn.Module, kind: str) -> int:
    if kind == "conv":
        if isinstance(module, ShiftedConv2d):
            return module.conv.out_channels
        if isinstance(module, nn.Conv2d):
            return module.out_channels
    if isinstance(module, ShiftedLinear):
        return module.linear.out_features
    if isinstance(module, nn.Linear):
        return module.out_features
    raise TypeError(f"Unsupported target module type for output rows: {type(module).__name__}")


def _nvfp4_scale_leaves(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    outer_scale_rows: tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an effective FP4 scale into outer and FP8 micro scale leaves."""

    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("NVFP4 split-scale export requires torch.float8_e4m3fn support")
    oc, ic = weight.shape
    groups = scale.shape[2]
    if ic % 16 != 0 or groups != ic // 16:
        raise ValueError("NVFP4 split-scale export requires group_size=16")
    effective = scale.view(oc, groups).float()
    if outer_scale_rows is None:
        wcscales = (effective.amax().view(1, 1) / 448.0).clamp_min(1e-12).to(dtype=weight.dtype)
        divisor = wcscales.float()
    else:
        wcscales = torch.empty((oc, 1), dtype=weight.dtype, device=weight.device)
        offset = 0
        for rows in outer_scale_rows:
            chunk = effective[offset : offset + rows]
            chunk_scale = (chunk.amax().view(1, 1) / 448.0).clamp_min(1e-12).to(dtype=weight.dtype)
            wcscales[offset : offset + rows] = chunk_scale
            offset += rows
        divisor = wcscales.float()
    wscales = (effective / divisor).clamp(min=0.0, max=448.0)
    wscales = wscales.to(dtype=torch.float8_e4m3fn).to(dtype=weight.dtype)
    return wcscales.view(-1, 1, 1, 1), wscales.view(oc, 1, groups, 1)


def _pack_nvfp4_linear_weight(weight: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pack FP4 residual weights with DeepCompressor/Nunchaku split scales."""

    oc, ic = weight.shape
    scale0, scale1 = _nvfp4_scale_leaves(weight, scale, outer_scale_rows=tuple(1 for _ in range(oc)))
    groups = scale1.shape[2]
    wcscales = scale0.view(oc)
    wscales = scale1.view(oc, groups).to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
    scaled = (
        weight.float()
        .view(oc, groups, 16)
        .div(wcscales.float().view(oc, 1, 1))
        .div(wscales.view(oc, groups, 1))
    )
    qweight = fp_quantize(scaled.view(oc, ic)).to(torch.int16)
    lo = qweight[:, 0::2].bitwise_and(0xF)
    hi = qweight[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    packed = lo.bitwise_or(hi).to(torch.uint8).view(torch.int8).contiguous()
    return packed.cpu(), {
        "wscales": wscales.to(dtype=torch.float8_e4m3fn).t().contiguous().cpu(),
        "wcscales": wcscales.view(oc).contiguous().cpu(),
    }


def _pack_awq_w4a16_target(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """Pack an extra-weight target for Nunchaku Lite AWQ W4A16 modules."""

    if not isinstance(target.weight_layout, (AwqW4A16Layout, AdaNormAwqW4A16Layout)):
        raise ValueError("awq_w4a16 export requires an AWQ W4A16 weight layout")
    if target.kind != "linear":
        raise ValueError("awq_w4a16 weight_layout only supports linear targets")
    if len(target.modules) != 1:
        raise ValueError("awq_w4a16 weight_layout requires exactly one module per target")
    if spec.precision != "int4":
        raise ValueError("awq_w4a16 weight_layout requires precision='int4'")
    if spec.group_size != 64:
        raise ValueError("awq_w4a16 weight_layout requires group_size=64")
    if spec.rank != 0 or target.shared_low_rank:
        raise ValueError("awq_w4a16 weight_layout requires rank=0 and shared_low_rank=False")
    if spec.smooth is not False:
        raise ValueError("awq_w4a16 weight_layout requires smooth=False")
    if spec.activation_quant.enabled:
        raise ValueError("awq_w4a16 weight_layout requires activation_quant=False")
    if spec.weight_range_calibration.enabled:
        raise ValueError("awq_w4a16 weight_layout does not support weight_range_calibration")

    if isinstance(target.weight_layout, AdaNormAwqW4A16Layout):
        weight, bias = _apply_adanorm_awq_w4a16_layout(weight, bias, splits=target.weight_layout.splits)
    qweight, wscales, wzeros = _pack_awq_w4a16_weight(weight, group_size=spec.group_size)
    state_dict = {
        "qweight": qweight,
        "wscales": wscales,
        "wzeros": wzeros,
    }
    if bias is not None:
        state_dict["bias"] = bias.detach().cpu()
    return state_dict


def _pack_awq_w4a16_weight(weight: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``[out, in]`` weights into Nunchaku Lite ``AWQW4A16Linear`` layout."""

    oc, ic = weight.shape
    if group_size != 64:
        raise ValueError("Nunchaku Lite AWQ W4A16 export requires group_size=64")
    if ic % group_size != 0:
        raise ValueError(f"Input features ({ic}) must be divisible by group_size ({group_size}) for AWQ export")
    if oc % 4 != 0:
        raise ValueError(f"Output features ({oc}) must be divisible by 4 for AWQ export")
    groups = ic // group_size
    scale = weight.float().view(oc, groups, group_size).abs().amax(dim=2).clamp_min(1e-6) / 7
    export_scale = scale.to(dtype=weight.dtype)
    unsigned_codes = (
        weight.float()
        .view(oc, groups, group_size)
        .div(export_scale.float().view(oc, groups, 1))
        .add(7)
        .round()
        .clamp(0, 15)
        .to(torch.int32)
        .view(oc, ic)
    )
    code_order = _awq_w4a16_code_order(weight.device)
    ordered = unsigned_codes.view(oc, groups, group_size).index_select(dim=2, index=code_order).view(oc, groups, 8, 8)
    packed_groups = torch.zeros((oc, groups, 8), dtype=torch.int32, device=weight.device)
    for nibble in range(8):
        packed_groups.bitwise_or_((ordered[:, :, :, nibble].bitwise_and(0xF)) << (4 * nibble))
    qweight = packed_groups.view(oc // 4, 4, groups, 8).permute(0, 2, 1, 3).reshape(oc // 4, groups * 32)
    wscales = export_scale.t().contiguous()
    wzeros = (-7 * export_scale.float()).to(dtype=weight.dtype).t().contiguous()
    return qweight.cpu().contiguous(), wscales.cpu(), wzeros.cpu()


def _apply_adanorm_awq_w4a16_layout(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    splits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply DeepCompressor AdaNorm output interleave and bias offset."""

    oc, ic = weight.shape
    if splits not in {3, 6}:
        raise ValueError(f"Unsupported AdaNorm AWQ W4A16 split count: {splits!r}")
    if oc % splits != 0:
        raise ValueError(f"AdaNorm AWQ output features ({oc}) must be divisible by split count ({splits})")
    weight = weight.view(splits, oc // splits, ic).transpose(0, 1).reshape(oc, ic).contiguous()
    if bias is None:
        bias = torch.zeros(oc, dtype=weight.dtype, device=weight.device)
    bias = bias.reshape(splits, oc // splits).transpose(0, 1).contiguous()
    delta = torch.zeros(splits, dtype=bias.dtype, device=bias.device)
    delta[1] = 1
    delta[-2] = 1
    bias = bias.add(delta.view(1, splits)).reshape(oc).contiguous()
    return weight, bias


def _awq_w4a16_code_order(device: torch.device) -> torch.Tensor:
    order = []
    for packed_index in range(8):
        for nibble in range(8):
            candidates = [
                channel
                for channel in range(64)
                if ((channel // 32) * 4 + (channel % 8) // 2) == packed_index
                and (((channel % 32) // 8) + 4 * (channel % 2)) == nibble
            ]
            if len(candidates) != 1:
                raise RuntimeError("Internal AWQ W4A16 channel order construction failed")
            order.append(candidates[0])
    return torch.tensor(order, dtype=torch.long, device=device)


def _calibrate_activation_range(
    partitions: tuple[torch.Tensor, ...],
    range_spec: RangeCalibrationSpec,
    spec: DiffusionQuantSpec,
) -> dict[str, torch.Tensor | int | str | bool] | None:
    """Calibrate activation ranges from input partitions.

    Args:
        partitions: Activation row partitions.
        range_spec: Range calibration settings.
        spec: Quantization settings containing group size.

    Returns:
        Range state dictionary, or ``None`` when disabled/unavailable.
    """

    if not range_spec.enabled or not partitions:
        return None
    return _calibrate_range(partitions, range_spec, spec)


def _calibrate_output_range(
    target_cache: IOTensorsCache | None,
    spec: DiffusionQuantSpec,
) -> dict[str, torch.Tensor | int | str | bool] | None:
    """Calibrate output activation ranges from a target cache.

    Args:
        target_cache: Rich target I/O cache.
        spec: Quantization settings with activation quant config.

    Returns:
        Range state dictionary, or ``None`` when disabled/unavailable.
    """

    if target_cache is None or not spec.activation_quant.outputs.enabled:
        return None
    output = target_cache.outputs.tensor()
    if output is None:
        return None
    return _calibrate_range((output,), spec.activation_quant.outputs, spec)


def _calibrate_range(
    tensors: tuple[torch.Tensor, ...],
    range_spec: RangeCalibrationSpec,
    spec: DiffusionQuantSpec,
    *,
    weight_like: bool = False,
) -> dict[str, torch.Tensor | int | str | bool] | None:
    """Compute min/max range state and quantization parameters.

    Args:
        tensors: Tensors whose last dimension is treated as feature channels.
        range_spec: Range calibration settings.
        spec: Quantization settings containing group size.
        weight_like: Mark the returned range as weight-derived metadata.

    Returns:
        Range tensors and qparam metadata, or ``None`` when disabled.
    """

    rows = [tensor.float().reshape(-1, tensor.shape[-1]) for tensor in tensors if tensor.numel() > 0]
    if not rows or not range_spec.enabled:
        return None
    data = torch.cat(rows, dim=0)
    min_value, max_value = _range_min_max(data, range_spec, spec.group_size)
    scale, zero, qmin, qmax = _range_qparams(min_value, max_value, range_spec)
    return {
        "scale": scale.detach().cpu(),
        "zero": zero.detach().cpu(),
        "min": min_value.detach().cpu(),
        "max": max_value.detach().cpu(),
        "qmin": qmin,
        "qmax": qmax,
        "granularity": range_spec.granularity,
        "symmetric": range_spec.symmetric,
        "allow_unsigned": range_spec.allow_unsigned,
        "group_size": spec.group_size if range_spec.granularity == "group" else -1,
        "weight_like": weight_like,
    }


def _range_min_max(
    data: torch.Tensor,
    range_spec: RangeCalibrationSpec,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute min/max values at the configured granularity.

    Args:
        data: Two-dimensional rows-by-features tensor.
        range_spec: Range calibration settings.
        group_size: Feature group size for group granularity.

    Returns:
        Minimum and maximum tensors.
    """

    if range_spec.granularity == "tensor":
        return data.amin().reshape(1), data.amax().reshape(1)
    if range_spec.granularity == "channel":
        return data.amin(dim=0), data.amax(dim=0)
    if data.shape[-1] % group_size != 0:
        raise ValueError(
            f"Range calibration group granularity requires feature size {data.shape[-1]} "
            f"to be divisible by group_size {group_size}"
        )
    grouped = data.reshape(data.shape[0], data.shape[-1] // group_size, group_size)
    return grouped.amin(dim=(0, 2)), grouped.amax(dim=(0, 2))


def _range_qparams(
    min_value: torch.Tensor,
    max_value: torch.Tensor,
    range_spec: RangeCalibrationSpec,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Convert observed ranges to INT4 scale/zero/qmin/qmax.

    Args:
        min_value: Observed minimum values.
        max_value: Observed maximum values.
        range_spec: Range calibration settings.

    Returns:
        Scale tensor, zero-point tensor, qmin, and qmax.
    """

    use_unsigned = bool(range_spec.allow_unsigned and float(min_value.min()) >= 0)
    if range_spec.symmetric:
        qmin, qmax = (0, 15) if use_unsigned else (-8, 7)
        if use_unsigned:
            scale = max_value.clamp_min(range_spec.eps) / qmax
        else:
            scale = torch.maximum(min_value.abs(), max_value.abs()).clamp_min(range_spec.eps) / qmax
        zero = torch.zeros_like(scale)
        return scale, zero, qmin, qmax

    qmin, qmax = (0, 15) if use_unsigned else (-8, 7)
    scale = (max_value - min_value).clamp_min(range_spec.eps) / (qmax - qmin)
    zero = (qmin - min_value / scale).round().clamp(qmin, qmax)
    return scale, zero, qmin, qmax


def _add_range_state(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    range_state: dict[str, torch.Tensor | int | str | bool] | None,
) -> None:
    """Append range tensors to an exported target state dict.

    Args:
        state_dict: State dict to mutate.
        prefix: Tensor name prefix such as ``"input"`` or ``"output"``.
        range_state: Optional range state from calibration.
    """

    if range_state is None:
        return
    for suffix in ("scale", "zero", "min", "max"):
        value = range_state[suffix]
        assert torch.is_tensor(value)
        state_dict[f"{prefix}_{suffix}"] = value.contiguous().cpu()


def _activation_quant_fn(range_state: dict[str, torch.Tensor | int | str | bool]):
    """Create a fake activation quantizer from calibrated range state.

    Args:
        range_state: Calibrated activation range state.

    Returns:
        Callable that fake-quantizes an activation tensor.
    """

    def quantize(inputs: torch.Tensor) -> torch.Tensor:
        """Fake-quantize one activation tensor.

        Args:
            inputs: Activation tensor to quantize and dequantize.

        Returns:
            Dequantized activation tensor in the original dtype.
        """

        scale = range_state["scale"]
        zero = range_state["zero"]
        assert torch.is_tensor(scale) and torch.is_tensor(zero)
        qmin = int(range_state["qmin"])
        qmax = int(range_state["qmax"])
        group_size = int(range_state["group_size"])
        granularity = str(range_state["granularity"])
        scale = _expand_range_param(scale.to(device=inputs.device, dtype=torch.float32), inputs, granularity, group_size)
        zero = _expand_range_param(zero.to(device=inputs.device, dtype=torch.float32), inputs, granularity, group_size)
        quantized = (inputs.float() / scale + zero).round().clamp(qmin, qmax)
        return ((quantized - zero) * scale).to(dtype=inputs.dtype)

    return quantize


def _expand_range_param(
    param: torch.Tensor,
    inputs: torch.Tensor,
    granularity: str,
    group_size: int,
) -> torch.Tensor:
    """Broadcast a range parameter to an activation tensor shape.

    Args:
        param: Scale or zero-point tensor.
        inputs: Activation tensor receiving the parameter.
        granularity: Tensor, channel, or group granularity.
        group_size: Group size for group granularity.

    Returns:
        Broadcastable parameter tensor.
    """

    if granularity == "tensor":
        return param.reshape(*([1] * inputs.ndim))
    if granularity == "group":
        param = param.repeat_interleave(group_size)
    if inputs.ndim >= 3 and param.numel() == inputs.shape[1]:
        return param.reshape(1, inputs.shape[1], *([1] * (inputs.ndim - 2)))
    return param.reshape(*([1] * (inputs.ndim - 1)), inputs.shape[-1])


def _activation_metadata(
    spec: DiffusionQuantSpec,
    input_range: dict[str, torch.Tensor | int | str | bool] | None,
    output_range: dict[str, torch.Tensor | int | str | bool] | None,
) -> dict[str, object]:
    """Build activation quantization metadata for one target.

    Args:
        spec: Quantization settings.
        input_range: Optional calibrated input range state.
        output_range: Optional calibrated output range state.

    Returns:
        JSON-serializable metadata dictionary.
    """

    return {
        "enabled": spec.activation_quant.enabled,
        "dtype": spec.activation_quant.dtype,
        "static": spec.activation_quant.static,
        "scale_dtypes": list(spec.activation_quant.scale_dtypes),
        "inputs": _range_metadata(spec.activation_quant.inputs, input_range),
        "outputs": _range_metadata(spec.activation_quant.outputs, output_range),
    }


def _range_metadata(
    range_spec: RangeCalibrationSpec,
    range_state: dict[str, torch.Tensor | int | str | bool] | None,
) -> dict[str, object]:
    """Build range calibration metadata.

    Args:
        range_spec: Range calibration settings.
        range_state: Optional calibrated range state.

    Returns:
        JSON-serializable metadata dictionary.
    """

    return {
        "enabled": range_spec.enabled,
        "calibrated": range_state is not None,
        "granularity": range_spec.granularity,
        "symmetric": range_spec.symmetric,
        "allow_unsigned": range_spec.allow_unsigned,
        "qmin": None if range_state is None else int(range_state["qmin"]),
        "qmax": None if range_state is None else int(range_state["qmax"]),
        "num_scales": 0 if range_state is None else int(range_state["scale"].numel()),  # type: ignore[union-attr]
    }


def _select_smooth_scale(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    calibration_inputs: torch.Tensor | None,
    calibration_input_partitions: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select the smoothing scale for one target.

    Args:
        target: Target being quantized.
        spec: Quantization settings.
        weight: Concatenated target weight.
        bias: Optional concatenated target bias.
        calibration_inputs: Optional full input rows.
        calibration_input_partitions: Optional partitioned input rows.

    Returns:
        Smooth scale tensor and metadata.
    """

    smooth_spec = resolve_smooth_spec(spec.smooth)
    identity = torch.ones(weight.shape[1], dtype=weight.dtype, device=weight.device)
    if not smooth_spec.enabled:
        logger.info("      + Smoothing disabled")
        return identity, {"enabled": False, "searched": False, "reason": "disabled"}
    if calibration_inputs is None:
        logger.info("      + Missing calibration inputs; using identity smoothing")
        return identity, {"enabled": True, "searched": False, "reason": "missing_calibration"}

    inputs = calibration_inputs.to(device=weight.device, dtype=torch.float32).reshape(-1, weight.shape[1])
    input_partitions = tuple(
        partition.to(device=weight.device, dtype=torch.float32).reshape(-1, weight.shape[1])
        for partition in (calibration_input_partitions or (inputs,))
    )
    search_weight = weight.to(dtype=torch.float32)
    search_bias = None if bias is None else bias.to(device=weight.device, dtype=torch.float32)
    best_error = torch.tensor(float("inf"), device=weight.device)
    best_scale = identity
    best_candidate: SmoothCandidate | None = None
    num_candidates = 0
    for candidate in iter_smooth_candidates(inputs, weight, smooth_spec):
        num_candidates += 1
        logger.debug(
            "      + Smoothing candidate %d: alpha=%s beta=%s span=%s",
            num_candidates,
            candidate.alpha,
            candidate.beta,
            candidate.span,
        )
        error = _candidate_output_error(
            candidate.scale.to(device=weight.device, dtype=weight.dtype),
            input_partitions,
            search_weight,
            search_bias,
            spec,
            target.shared_low_rank,
        )
        if error < best_error:
            best_error = error
            best_scale = candidate.scale.to(device=weight.device, dtype=weight.dtype)
            best_candidate = candidate
            logger.debug("        best smoothing error: %.6g", float(best_error.cpu()))
    metadata: dict[str, object] = {
        "enabled": True,
        "searched": True,
        "strategy": smooth_spec.strategy,
        "objective": smooth_spec.objective,
        "num_candidates": num_candidates,
        "error": float(best_error.cpu()),
    }
    if best_candidate is not None:
        metadata.update(
            {
                "alpha": best_candidate.alpha,
                "beta": best_candidate.beta,
                "span": list(best_candidate.span),
            }
        )
        logger.info(
            "      + Selected smoothing candidate: alpha=%s beta=%s span=%s error=%.6g (%d candidates)",
            best_candidate.alpha,
            best_candidate.beta,
            best_candidate.span,
            float(best_error.cpu()),
            num_candidates,
        )
    else:
        logger.info("      + No smoothing candidates generated; using identity smoothing")
    return best_scale, metadata


def _candidate_output_error(
    smooth: torch.Tensor,
    input_partitions: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    spec: DiffusionQuantSpec,
    shared_low_rank: bool,
) -> torch.Tensor:
    """Score a smoothing candidate by output reconstruction error.

    Args:
        smooth: Candidate smoothing scale.
        input_partitions: Calibration input partitions.
        weight: Original target weight.
        bias: Optional target bias.
        spec: Quantization settings.
        shared_low_rank: Whether to include a shared low-rank branch.

    Returns:
        Mean squared output error.
    """

    errors: list[torch.Tensor] = []
    for inputs in input_partitions:
        smoothed_inputs = _smooth_inputs(inputs, smooth)
        expected = _linear_output(inputs, weight, bias)
        smoothed_weight = weight * smooth.view(1, -1)
        low_rank = (
            _low_rank_branch(smoothed_weight, rank=spec.rank, inputs=smoothed_inputs, solver=spec.low_rank_solver)
            if spec.rank > 0 and shared_low_rank
            else None
        )
        residual = smoothed_weight
        if low_rank is not None:
            residual = smoothed_weight - low_rank[1] @ low_rank[0]
        scale = _weight_scales(residual, group_size=spec.group_size, float_point=spec.precision == "fp4")
        approx_residual = _fake_quantize_weight(residual, scale, float_point=spec.precision == "fp4")
        approx_weight = approx_residual
        if low_rank is not None:
            approx_weight = approx_weight + low_rank[1] @ low_rank[0]
        actual = _linear_output(smoothed_inputs, approx_weight, bias)
        errors.append((actual.float() - expected.float()).pow(2).mean())
    return torch.stack(errors).mean()


def _resolve_input_partitions(
    inputs: torch.Tensor | None,
    partitions: tuple[torch.Tensor, ...] | None,
    calibration: CalibrationSpec | None,
) -> tuple[torch.Tensor, ...]:
    """Resolve calibration input partitions for quantization consumers.

    Args:
        inputs: Optional concatenated input rows.
        partitions: Optional precomputed partitions.
        calibration: Optional calibration settings for repartitioning.

    Returns:
        Tuple of input partitions.
    """

    if partitions:
        return partitions
    if inputs is None:
        return ()
    if calibration is None:
        return (inputs,)
    return repartition_tensor(
        inputs,
        sample_size=calibration.sample_size,
        sample_batch_size=calibration.sample_batch_size,
    )


def _smooth_inputs(inputs: torch.Tensor | None, smooth: torch.Tensor) -> torch.Tensor | None:
    """Apply inverse smoothing to activation inputs.

    Args:
        inputs: Optional input row tensor.
        smooth: Smoothing scale.

    Returns:
        Smoothed inputs, or ``None``.
    """

    if inputs is None:
        return None
    return inputs / smooth.to(device=inputs.device, dtype=inputs.dtype).view(1, -1)


def _linear_output(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Compute a linear output from flattened inputs.

    Args:
        inputs: Input rows in ``[rows, in]`` layout.
        weight: Weight matrix in ``[out, in]`` layout.
        bias: Optional bias vector.

    Returns:
        Output rows in ``[rows, out]`` layout.
    """

    output = inputs @ weight.t()
    if bias is not None:
        output = output + bias.view(1, -1)
    return output


def _fake_quantize_weight(weight: torch.Tensor, scale: torch.Tensor, float_point: bool) -> torch.Tensor:
    """Quantize and dequantize a residual weight matrix for scoring.

    Args:
        weight: Residual weight matrix.
        scale: Per-group quantization scales.
        float_point: Whether FP4 fake quantization is requested.

    Returns:
        Dequantized residual weight approximation.
    """

    groups = scale.shape[2]
    group_size = weight.shape[1] // groups
    max_q = 6 if float_point else 7
    qweight = weight.float().view(weight.shape[0], groups, group_size) / scale.float().view(weight.shape[0], groups, 1)
    if float_point:
        codebook = fp4_e2m1_codebook(device=weight.device, dtype=torch.float32)
        qcodes = fp_quantize(qweight.view(weight.shape[0], weight.shape[1]), codebook=codebook)
        qweight = codebook[qcodes.long()].view(weight.shape[0], groups, group_size)
        return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)
    qweight = qweight.round_().clamp_(-8, max_q)
    return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)


def _low_rank_branch(
    weight: torch.Tensor,
    rank: int,
    inputs: torch.Tensor | None = None,
    *,
    solver: LowRankSolverSpec | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a weighted or unweighted low-rank branch by SVD.

    Args:
        weight: Weight matrix in ``[out, in]`` layout.
        rank: Requested low-rank dimension.
        inputs: Optional calibration inputs used for RMS weighting.
        solver: Optional solver settings controlling the SVD backend.

    Returns:
        ``(down, up)`` matrices where ``up @ down`` approximates ``weight``.
    """

    rank = min(rank, min(weight.shape))
    if rank == 0:
        return torch.empty(0, weight.shape[1], dtype=weight.dtype, device=weight.device), torch.empty(
            weight.shape[0], 0, dtype=weight.dtype, device=weight.device
        )
    solver = solver or LowRankSolverSpec()
    svd_dtype = torch.float32
    if inputs is None:
        svd_dtype = torch.float64
        u, s, vh = _factor_low_rank_weight(weight.to(svd_dtype), rank, solver)
        down = vh[:rank].to(dtype=weight.dtype, device=weight.device)
    else:
        rms = inputs.to(device=weight.device, dtype=svd_dtype).pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        weighted = weight.to(svd_dtype) * rms.view(1, -1)
        u, s, vh = _factor_low_rank_weight(weighted, rank, solver)
        down = (vh[:rank] / rms.view(1, -1)).to(dtype=weight.dtype, device=weight.device)
    up = (u[:, :rank] * s[:rank]).to(dtype=weight.dtype, device=weight.device)
    return down, up


def _factor_low_rank_weight(
    weight: torch.Tensor,
    rank: int,
    solver: LowRankSolverSpec,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factor a weight matrix with the configured SVD backend."""

    if solver.svd_backend == "full":
        return torch.linalg.svd(weight, full_matrices=False)
    q = min(rank + solver.svd_lowrank_oversample, min(weight.shape))
    u, s, v = torch.svd_lowrank(weight, q=q, niter=solver.svd_lowrank_niter)
    return u, s, v.t()
