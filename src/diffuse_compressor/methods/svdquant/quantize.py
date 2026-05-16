from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ...artifact import QuantizedTarget
from ...calibration import EvalReplayBatch, IOTensorsCache, repartition_tensor
from ...config import CalibrationSpec, DiffusionQuantSpec, RangeCalibrationSpec
from ...targets import QuantTarget
from .lowrank_search import search_low_rank_branch
from .smoothing import SmoothCandidate, iter_smooth_candidates, resolve_smooth_spec


@torch.inference_mode()
def quantize_targets(
    targets: Iterable[QuantTarget],
    spec: DiffusionQuantSpec,
    calibration_inputs: dict[str, torch.Tensor] | None = None,
    calibration_input_partitions: dict[str, tuple[torch.Tensor, ...]] | None = None,
    layer_cache: dict[str, IOTensorsCache] | None = None,
    eval_replay: EvalReplayBatch | None = None,
    calibration: CalibrationSpec | None = None,
) -> list[QuantizedTarget]:
    quantized: list[QuantizedTarget] = []
    for target in targets:
        if target.kind != "linear":
            raise NotImplementedError(f"SVDQuant currently supports linear targets only, got {target.kind!r}")
        inputs = calibration_inputs.get(target.export_name) if calibration_inputs is not None else None
        input_partitions = (
            calibration_input_partitions.get(target.export_name)
            if calibration_input_partitions is not None and target.export_name in calibration_input_partitions
            else None
        )
        target_cache = layer_cache.get(target.export_name) if layer_cache is not None else None
        quantized.append(_quantize_linear_target(target, spec, inputs, input_partitions, target_cache, eval_replay, calibration))
    return quantized


def _quantize_linear_target(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    calibration_inputs: torch.Tensor | None = None,
    calibration_input_partitions: tuple[torch.Tensor, ...] | None = None,
    target_cache: IOTensorsCache | None = None,
    eval_replay: EvalReplayBatch | None = None,
    calibration: CalibrationSpec | None = None,
) -> QuantizedTarget:
    modules = [module for module in target.modules if isinstance(module, nn.Linear)]
    weight = torch.cat([module.weight.detach() for module in modules], dim=0)
    bias = _concat_bias(modules, weight.device, weight.dtype)
    export_dtype = torch.bfloat16 if weight.dtype not in (torch.float16, torch.bfloat16) else weight.dtype
    weight = weight.to(dtype=export_dtype)
    if bias is not None:
        bias = bias.to(dtype=export_dtype)

    partitions = _resolve_input_partitions(calibration_inputs, calibration_input_partitions, calibration)
    input_range = _calibrate_activation_range(partitions, spec.activation_quant.inputs, spec) if spec.activation_quant.enabled else None
    smooth, smooth_metadata = _select_smooth_scale(target, spec, weight, bias, calibration_inputs, partitions)
    quant_inputs = _smooth_inputs(calibration_inputs, smooth) if calibration_inputs is not None else None
    quant_input_partitions = tuple(_smooth_inputs(partition, smooth) for partition in partitions) if partitions else None
    smooth_weight = weight * smooth.view(1, -1)
    activation_quant_fn = _activation_quant_fn(input_range) if spec.activation_quant.enabled and input_range is not None else None

    low_rank_metadata: dict[str, object] = {"mode": spec.low_rank_solver.mode}
    if spec.rank > 0 and target.shared_low_rank and spec.low_rank_solver.mode == "search":
        search = search_low_rank_branch(
            target=target,
            weight=smooth_weight,
            bias=bias,
            inputs=quant_inputs,
            input_partitions=quant_input_partitions,
            spec=spec,
            eval_replay=eval_replay,
            low_rank_fn=_low_rank_branch,
            weight_scales_fn=_weight_scales,
            fake_quant_weight_fn=_fake_quantize_weight,
            activation_quant_fn=activation_quant_fn,
        )
        low_rank = search.low_rank
        quant_weight = search.residual
        low_rank_metadata = search.metadata
    else:
        low_rank = (
            _low_rank_branch(smooth_weight, rank=spec.rank, inputs=quant_inputs)
            if spec.rank > 0 and target.shared_low_rank
            else None
        )
        quant_weight = smooth_weight
        if low_rank is not None:
            quant_weight = smooth_weight - low_rank[1] @ low_rank[0]
        low_rank_metadata = {
            "mode": "weighted_svd",
            "iterations": 1 if low_rank is not None else 0,
            "eval_replay": False,
        }

    scale = _weight_scales(quant_weight, group_size=spec.group_size, float_point=spec.precision == "fp4")
    qweight, wscales = _pack_lite_linear_weight(quant_weight, scale, float_point=spec.precision == "fp4")
    output_range = _calibrate_output_range(target_cache, spec) if spec.activation_quant.enabled else None
    weight_range = (
        _calibrate_range((quant_weight,), spec.weight_range_calibration.range, spec, weight_like=True)
        if spec.weight_range_calibration.enabled
        else None
    )
    state_dict = {
        "qweight": qweight,
        "wscales": wscales,
        "smooth_factor": smooth.detach().cpu(),
        "smooth_factor_orig": smooth.detach().cpu().clone(),
    }
    if bias is not None:
        state_dict["bias"] = bias.detach().cpu()
    if low_rank is not None:
        state_dict["proj_down"] = low_rank[0].t().contiguous().cpu()
        state_dict["proj_up"] = low_rank[1].contiguous().cpu()
    _add_range_state(state_dict, "input", input_range)
    _add_range_state(state_dict, "output", output_range)
    _add_range_state(state_dict, "weight_range", weight_range)
    return QuantizedTarget(
        target=target,
        state_dict=state_dict,
        metadata={
            "source_modules": list(target.module_names),
            "roles": list(target.roles),
            "rank": spec.rank,
            "precision": spec.precision,
            "calibrated": calibration_inputs is not None,
            "low_rank_solver": low_rank_metadata,
            "smooth": smooth_metadata,
            "activation_quant": _activation_metadata(spec, input_range, output_range),
            "weight_range_calibration": _range_metadata(spec.weight_range_calibration.range, weight_range)
            | {"enabled": spec.weight_range_calibration.enabled},
        },
    )


def _concat_bias(modules: list[nn.Linear], device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if all(module.bias is None for module in modules):
        return None
    return torch.cat(
        [
            module.bias.detach() if module.bias is not None else torch.zeros(module.out_features, device=device, dtype=dtype)
            for module in modules
        ],
        dim=0,
    )


def _weight_scales(weight: torch.Tensor, group_size: int, float_point: bool) -> torch.Tensor:
    oc, ic = weight.shape
    if ic % group_size != 0:
        raise ValueError(f"Input features ({ic}) must be divisible by group_size ({group_size}) for Nunchaku export")
    groups = ic // group_size
    max_q = 6 if float_point else 7
    scale = weight.float().view(oc, groups, group_size).abs().amax(dim=2).clamp_min(1e-6) / max_q
    return scale.to(dtype=weight.dtype).view(oc, 1, groups, 1)


def _pack_lite_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    float_point: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    oc, ic = weight.shape
    groups = scale.shape[2]
    group_size = ic // groups
    scaled = weight.float().view(oc, groups, group_size) / scale.float().view(oc, groups, 1)
    if float_point:
        raise NotImplementedError("Nunchaku Lite FP4 packing is not implemented yet")
    qweight = scaled.round_().clamp_(-8, 7).to(torch.int16).view(oc, ic)
    lo = qweight[:, 0::2].bitwise_and(0xF)
    hi = qweight[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    packed = lo.bitwise_or(hi).to(torch.uint8).view(torch.int8).contiguous()
    wscales = scale.view(oc, groups).t().contiguous().cpu()
    return packed.cpu(), wscales


def _calibrate_activation_range(
    partitions: tuple[torch.Tensor, ...],
    range_spec: RangeCalibrationSpec,
    spec: DiffusionQuantSpec,
) -> dict[str, torch.Tensor | int | str | bool] | None:
    if not range_spec.enabled or not partitions:
        return None
    return _calibrate_range(partitions, range_spec, spec)


def _calibrate_output_range(
    target_cache: IOTensorsCache | None,
    spec: DiffusionQuantSpec,
) -> dict[str, torch.Tensor | int | str | bool] | None:
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
    if range_state is None:
        return
    for suffix in ("scale", "zero", "min", "max"):
        value = range_state[suffix]
        assert torch.is_tensor(value)
        state_dict[f"{prefix}_{suffix}"] = value.contiguous().cpu()


def _activation_quant_fn(range_state: dict[str, torch.Tensor | int | str | bool]):
    def quantize(inputs: torch.Tensor) -> torch.Tensor:
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
    if granularity == "tensor":
        return param.reshape(*([1] * inputs.ndim))
    if granularity == "group":
        param = param.repeat_interleave(group_size)
    return param.reshape(*([1] * (inputs.ndim - 1)), inputs.shape[-1])


def _activation_metadata(
    spec: DiffusionQuantSpec,
    input_range: dict[str, torch.Tensor | int | str | bool] | None,
    output_range: dict[str, torch.Tensor | int | str | bool] | None,
) -> dict[str, object]:
    return {
        "enabled": spec.activation_quant.enabled,
        "dtype": spec.activation_quant.dtype,
        "static": spec.activation_quant.static,
        "inputs": _range_metadata(spec.activation_quant.inputs, input_range),
        "outputs": _range_metadata(spec.activation_quant.outputs, output_range),
    }


def _range_metadata(
    range_spec: RangeCalibrationSpec,
    range_state: dict[str, torch.Tensor | int | str | bool] | None,
) -> dict[str, object]:
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
    smooth_spec = resolve_smooth_spec(spec.smooth)
    identity = torch.ones(weight.shape[1], dtype=weight.dtype, device=weight.device)
    if not smooth_spec.enabled:
        return identity, {"enabled": False, "searched": False, "reason": "disabled"}
    if calibration_inputs is None:
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
    metadata: dict[str, object] = {
        "enabled": True,
        "searched": True,
        "strategy": smooth_spec.strategy,
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
    return best_scale, metadata


def _candidate_output_error(
    smooth: torch.Tensor,
    input_partitions: tuple[torch.Tensor, ...],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    spec: DiffusionQuantSpec,
    shared_low_rank: bool,
) -> torch.Tensor:
    errors: list[torch.Tensor] = []
    for inputs in input_partitions:
        smoothed_inputs = _smooth_inputs(inputs, smooth)
        expected = _linear_output(inputs, weight, bias)
        smoothed_weight = weight * smooth.view(1, -1)
        low_rank = _low_rank_branch(smoothed_weight, rank=spec.rank, inputs=smoothed_inputs) if spec.rank > 0 and shared_low_rank else None
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
        element_size=calibration.element_size,
        element_batch_size=calibration.element_batch_size,
    )


def _smooth_inputs(inputs: torch.Tensor | None, smooth: torch.Tensor) -> torch.Tensor | None:
    if inputs is None:
        return None
    return inputs / smooth.to(device=inputs.device, dtype=inputs.dtype).view(1, -1)


def _linear_output(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    output = inputs @ weight.t()
    if bias is not None:
        output = output + bias.view(1, -1)
    return output


def _fake_quantize_weight(weight: torch.Tensor, scale: torch.Tensor, float_point: bool) -> torch.Tensor:
    groups = scale.shape[2]
    group_size = weight.shape[1] // groups
    max_q = 6 if float_point else 7
    qweight = weight.float().view(weight.shape[0], groups, group_size) / scale.float().view(weight.shape[0], groups, 1)
    if float_point:
        raise NotImplementedError("Nunchaku Lite FP4 smoothing search is not implemented yet")
    qweight = qweight.round_().clamp_(-8, max_q)
    return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)


def _low_rank_branch(
    weight: torch.Tensor,
    rank: int,
    inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = min(rank, min(weight.shape))
    if rank == 0:
        return torch.empty(0, weight.shape[1], dtype=weight.dtype, device=weight.device), torch.empty(
            weight.shape[0], 0, dtype=weight.dtype, device=weight.device
        )
    svd_dtype = torch.float32
    if inputs is None:
        svd_dtype = torch.float64
        u, s, vh = torch.linalg.svd(weight.to(svd_dtype), full_matrices=False)
        down = vh[:rank].to(dtype=weight.dtype, device=weight.device)
    else:
        rms = inputs.to(device=weight.device, dtype=svd_dtype).pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        weighted = weight.to(svd_dtype) * rms.view(1, -1)
        u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
        down = (vh[:rank] / rms.view(1, -1)).to(dtype=weight.dtype, device=weight.device)
    up = (u[:, :rank] * s[:rank]).to(dtype=weight.dtype, device=weight.device)
    return down, up
