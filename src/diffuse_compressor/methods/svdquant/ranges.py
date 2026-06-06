from __future__ import annotations

import torch

from ...calibration import IOTensorsCache
from ...config import DiffusionQuantSpec, RangeCalibrationSpec


def calibrate_activation_range(
    partitions: tuple[torch.Tensor, ...],
    range_spec: RangeCalibrationSpec,
    spec: DiffusionQuantSpec,
) -> dict[str, torch.Tensor | int | str | bool] | None:
    """Calibrate activation ranges from input partitions."""

    if not range_spec.enabled or not partitions:
        return None
    return calibrate_range(partitions, range_spec, spec)


def calibrate_output_range(
    target_cache: IOTensorsCache | None,
    spec: DiffusionQuantSpec,
) -> dict[str, torch.Tensor | int | str | bool] | None:
    """Calibrate output activation ranges from a target cache."""

    if target_cache is None or not spec.activation_quant.outputs.enabled:
        return None
    output = target_cache.outputs.tensor()
    if output is None:
        return None
    return calibrate_range((output,), spec.activation_quant.outputs, spec)


def calibrate_range(
    tensors: tuple[torch.Tensor, ...],
    range_spec: RangeCalibrationSpec,
    spec: DiffusionQuantSpec,
    *,
    weight_like: bool = False,
) -> dict[str, torch.Tensor | int | str | bool] | None:
    """Compute min/max range state and quantization parameters."""

    rows = [
        tensor.float().reshape(-1, tensor.shape[-1])
        for tensor in tensors
        if tensor.numel() > 0
    ]
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


def add_range_state(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    range_state: dict[str, torch.Tensor | int | str | bool] | None,
) -> None:
    """Append range tensors to an exported target state dict."""

    if range_state is None:
        return
    for suffix in ("scale", "zero", "min", "max"):
        value = range_state[suffix]
        assert torch.is_tensor(value)
        state_dict[f"{prefix}_{suffix}"] = value.contiguous().cpu()


def activation_quant_fn(range_state: dict[str, torch.Tensor | int | str | bool]):
    """Create a fake activation quantizer from calibrated range state."""

    def quantize(inputs: torch.Tensor) -> torch.Tensor:
        scale = range_state["scale"]
        zero = range_state["zero"]
        assert torch.is_tensor(scale) and torch.is_tensor(zero)
        qmin = int(range_state["qmin"])
        qmax = int(range_state["qmax"])
        group_size = int(range_state["group_size"])
        granularity = str(range_state["granularity"])
        scale = _expand_range_param(
            scale.to(device=inputs.device, dtype=torch.float32),
            inputs,
            granularity,
            group_size,
        )
        zero = _expand_range_param(
            zero.to(device=inputs.device, dtype=torch.float32),
            inputs,
            granularity,
            group_size,
        )
        quantized = (inputs.float() / scale + zero).round().clamp(qmin, qmax)
        return ((quantized - zero) * scale).to(dtype=inputs.dtype)

    return quantize


def activation_metadata(
    spec: DiffusionQuantSpec,
    input_range: dict[str, torch.Tensor | int | str | bool] | None,
    output_range: dict[str, torch.Tensor | int | str | bool] | None,
) -> dict[str, object]:
    """Build activation quantization metadata for one target."""

    return {
        "enabled": spec.activation_quant.enabled,
        "dtype": spec.activation_quant.dtype,
        "static": spec.activation_quant.static,
        "scale_dtypes": list(spec.activation_quant.scale_dtypes),
        "inputs": range_metadata(spec.activation_quant.inputs, input_range),
        "outputs": range_metadata(spec.activation_quant.outputs, output_range),
    }


def range_metadata(
    range_spec: RangeCalibrationSpec,
    range_state: dict[str, torch.Tensor | int | str | bool] | None,
) -> dict[str, object]:
    """Build range calibration metadata."""

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
            scale = (
                torch.maximum(min_value.abs(), max_value.abs()).clamp_min(
                    range_spec.eps
                )
                / qmax
            )
        zero = torch.zeros_like(scale)
        return scale, zero, qmin, qmax

    qmin, qmax = (0, 15) if use_unsigned else (-8, 7)
    scale = (max_value - min_value).clamp_min(range_spec.eps) / (qmax - qmin)
    zero = (qmin - min_value / scale).round().clamp(qmin, qmax)
    return scale, zero, qmin, qmax


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
    if inputs.ndim >= 3 and param.numel() == inputs.shape[1]:
        return param.reshape(1, inputs.shape[1], *([1] * (inputs.ndim - 2)))
    return param.reshape(*([1] * (inputs.ndim - 1)), inputs.shape[-1])
