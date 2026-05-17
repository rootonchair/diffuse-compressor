from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import safetensors.torch
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .core import EvaluationSpec

from ..config import PatchRule
from ..methods.svdquant.packing import fp4_e2m1_codebook
from ..patches import ShiftedLinear, prepare_model


def patch_quantized_pipeline(pipe: Any, *, model_key: str, spec: "EvaluationSpec") -> Any:
    """Patch a pipeline with an exported quantized checkpoint."""

    if spec.runtime == "none":
        return pipe
    if spec.runtime == "torch-dequant":
        return patch_pipeline_with_dequantized_checkpoint(pipe, model_key=model_key, spec=spec)
    if spec.runtime != "nunchaku-lite":
        raise RuntimeError(f"Unsupported quantized runtime: {spec.runtime!r}")
    if spec.checkpoint is None:
        raise RuntimeError("runtime='nunchaku-lite' requires EvaluationSpec.checkpoint")

    patch_transformer = _load_nunchaku_lite_patch_transformer()
    target = _nunchaku_lite_target(model_key)
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("nunchaku-lite evaluation requires the pipeline to expose a transformer")
    patch_transformer(
        pipe.transformer,
        spec.checkpoint,
        target=target,
        precision=spec.precision,
        torch_dtype=spec.torch_dtype or torch.bfloat16,
    )
    return pipe


def patch_pipeline_with_dequantized_checkpoint(pipe: Any, *, model_key: str, spec: "EvaluationSpec") -> Any:
    """Patch a pipeline by materializing packed quantized weights as PyTorch weights."""

    if spec.checkpoint is None:
        raise RuntimeError("runtime='torch-dequant' requires EvaluationSpec.checkpoint")
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("torch-dequant evaluation requires the pipeline to expose a transformer")

    state, metadata = _load_checkpoint(spec.checkpoint)
    transformer = pipe.transformer
    _prepare_transformer_for_dequant(transformer, metadata, model_key=model_key)
    _load_dequantized_transformer_state(
        transformer,
        state=state,
        metadata=metadata,
        torch_dtype=spec.torch_dtype or torch.bfloat16,
    )
    return pipe


def _load_checkpoint(path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    with safetensors.safe_open(path, framework="pt", device="cpu") as handle:
        metadata_blob = handle.metadata().get("quantization_config")
    if metadata_blob is None:
        raise RuntimeError(f"Checkpoint {path} does not contain quantization_config metadata")
    metadata = json.loads(metadata_blob)
    return safetensors.torch.load_file(path, device="cpu"), metadata


def _prepare_transformer_for_dequant(transformer: nn.Module, metadata: dict[str, Any], *, model_key: str) -> None:
    module_paths = [module for target in metadata.get("targets", []) for module in target.get("modules", [])]
    needs_flux_split = any(".proj_out.linears." in module for module in module_paths)
    if not needs_flux_split:
        _apply_activation_shift_patches(transformer, metadata)
        return
    normalized = model_key.lower()
    if not (normalized.startswith("flux.1") or normalized.startswith("flux1") or normalized.startswith("flux")):
        raise RuntimeError("torch-dequant checkpoint references split proj_out modules but model_key is not Flux-like")
    prepare_model(
        transformer,
        [PatchRule(type="split_linear", module="single_transformer_blocks.*.proj_out", args={"splits": ["out_features"]})],
    )
    _apply_activation_shift_patches(transformer, metadata)


def _apply_activation_shift_patches(transformer: nn.Module, metadata: dict[str, Any]) -> None:
    shifts = metadata.get("calibration", {}).get("activation_shifts", {})
    if not shifts:
        return
    modules = dict(transformer.named_modules())
    rules = []
    for module_name, shift in shifts.items():
        module = modules.get(module_name)
        if module is None:
            raise RuntimeError(f"torch-dequant checkpoint references missing shifted module {module_name!r}")
        if isinstance(module, ShiftedLinear):
            continue
        if not isinstance(module, nn.Linear):
            raise RuntimeError(
                f"torch-dequant activation shift expected nn.Linear at {module_name!r}, got {type(module).__name__}"
            )
        rules.append(PatchRule(type="shift_linear", module=module_name, args={"shift": float(shift)}))
    if rules:
        prepare_model(transformer, rules)


def _load_dequantized_transformer_state(
    transformer: nn.Module,
    *,
    state: dict[str, torch.Tensor],
    metadata: dict[str, Any],
    torch_dtype: torch.dtype,
) -> None:
    modules = dict(transformer.named_modules())
    for target in metadata.get("targets", []):
        export_name = str(target["export_name"])
        target_modules = [modules[name] for name in target["modules"]]
        if not all(_is_linear_like(module) for module in target_modules):
            raise RuntimeError(f"torch-dequant currently supports linear targets only, got {export_name!r}")
        weight = _reconstruct_target_weight(
            export_name=export_name,
            state=state,
            precision=str(target.get("precision", metadata.get("weight", {}).get("dtype", "int4"))),
        ).to(dtype=torch_dtype)
        bias = state.get(f"{export_name}.bias")
        if bias is not None:
            bias = bias.to(dtype=torch_dtype)
        _copy_target_weights(target_modules, weight, bias, export_name=export_name)
    setattr(transformer, "_diffuse_compressor_torch_dequant_hooks", [])


def _reconstruct_target_weight(
    *,
    export_name: str,
    state: dict[str, torch.Tensor],
    precision: str,
) -> torch.Tensor:
    qweight = state[f"{export_name}.qweight"]
    wscales = state[f"{export_name}.wscales"]
    weight = _dequantize_qweight(qweight, wscales, precision=precision)
    proj_down = state.get(f"{export_name}.proj_down")
    proj_up = state.get(f"{export_name}.proj_up")
    if proj_down is not None and proj_up is not None:
        weight = weight + proj_up.float() @ proj_down.float().t()
    smooth = state.get(f"{export_name}.smooth_factor")
    if smooth is not None:
        weight = weight / smooth.float().view(1, -1)
    return weight


def _dequantize_qweight(qweight: torch.Tensor, wscales: torch.Tensor, *, precision: str) -> torch.Tensor:
    packed = qweight.cpu().view(torch.uint8)
    lo = packed.bitwise_and(0x0F).long()
    hi = packed.bitwise_right_shift(4).bitwise_and(0x0F).long()
    codes = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.long)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    normalized = precision in {"fp4", "nvfp4", "fp4_e2m1_all", "sfp4_e2m1_all"}
    if normalized:
        values = fp4_e2m1_codebook(dtype=torch.float32)[codes]
    else:
        values = codes.float()
        values = torch.where(values >= 8, values - 16, values)
    oc, ic = values.shape
    groups = wscales.shape[0]
    if ic % groups != 0:
        raise RuntimeError(f"qweight input dimension {ic} is not divisible by {groups} scale groups")
    group_size = ic // groups
    scale = wscales.float().t().contiguous().view(oc, groups, 1)
    return (values.view(oc, groups, group_size) * scale).view(oc, ic)


def _copy_target_weights(
    modules: list[nn.Module],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    export_name: str,
) -> None:
    offset = 0
    for module in modules:
        linear = _as_linear(module)
        rows = linear.weight.shape[0]
        chunk = weight[offset : offset + rows]
        if tuple(chunk.shape) != tuple(linear.weight.shape):
            raise RuntimeError(
                f"Reconstructed weight chunk for {export_name!r} has shape {tuple(chunk.shape)}, "
                f"expected {tuple(linear.weight.shape)}"
            )
        corrected_bias = None
        if bias is not None and linear.bias is not None:
            corrected_bias = bias[offset : offset + rows]
            if isinstance(module, ShiftedLinear):
                shift = module.shift.to(device=linear.weight.device, dtype=torch.float32).view(-1, 1)
                original_bias = linear.bias.detach().float().view(-1, 1) + linear.weight.detach().float() @ shift
                corrected_bias = original_bias.view(-1) - chunk.to(device=linear.weight.device, dtype=torch.float32) @ shift.view(-1)
        linear.weight.data.copy_(chunk.to(device=linear.weight.device, dtype=linear.weight.dtype))
        if corrected_bias is not None and linear.bias is not None:
            linear.bias.data.copy_(corrected_bias.to(device=linear.bias.device, dtype=linear.bias.dtype))
        offset += rows
    if offset != weight.shape[0]:
        raise RuntimeError(f"Reconstructed target {export_name!r} left {weight.shape[0] - offset} unused rows")


def _is_linear_like(module: nn.Module) -> bool:
    return isinstance(module, nn.Linear) or isinstance(module, ShiftedLinear)


def _as_linear(module: nn.Module) -> nn.Linear:
    if isinstance(module, nn.Linear):
        return module
    if isinstance(module, ShiftedLinear):
        return module.linear
    raise TypeError(f"Module {type(module).__name__} is not linear-like")


def _register_activation_hooks(
    modules: list[nn.Module],
    *,
    export_name: str,
    target: dict[str, Any],
    state: dict[str, torch.Tensor],
) -> list[torch.utils.hooks.RemovableHandle]:
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    input_quant = _activation_quantizer_from_state(export_name, "input", target, state)
    output_quant = _activation_quantizer_from_state(export_name, "output", target, state)
    for module in modules:
        if input_quant is not None:
            hooks.append(module.register_forward_pre_hook(_activation_pre_hook(input_quant)))
        if output_quant is not None:
            hooks.append(module.register_forward_hook(_activation_output_hook(output_quant)))
    return hooks


def _activation_quantizer_from_state(
    export_name: str,
    prefix: str,
    target: dict[str, Any],
    state: dict[str, torch.Tensor],
):
    scale = state.get(f"{export_name}.{prefix}_scale")
    zero = state.get(f"{export_name}.{prefix}_zero")
    if scale is None or zero is None:
        return None
    min_value = state.get(f"{export_name}.{prefix}_min")
    group_size = int(target.get("group_size", 0))

    def quantize(inputs: torch.Tensor) -> torch.Tensor:
        qmin, qmax = _activation_qrange(min_value)
        expanded_scale = _expand_activation_param(scale.to(device=inputs.device, dtype=torch.float32), inputs, group_size)
        expanded_zero = _expand_activation_param(zero.to(device=inputs.device, dtype=torch.float32), inputs, group_size)
        quantized = (inputs.float() / expanded_scale + expanded_zero).round().clamp(qmin, qmax)
        return ((quantized - expanded_zero) * expanded_scale).to(dtype=inputs.dtype)

    return quantize


def _activation_qrange(min_value: torch.Tensor | None) -> tuple[int, int]:
    if min_value is not None and min_value.numel() > 0 and float(min_value.min()) >= 0:
        return 0, 15
    return -8, 7


def _expand_activation_param(param: torch.Tensor, inputs: torch.Tensor, group_size: int) -> torch.Tensor:
    if param.numel() == 1:
        return param.reshape(*([1] * inputs.ndim))
    if group_size > 0 and param.numel() * group_size == inputs.shape[-1]:
        param = param.repeat_interleave(group_size)
    if inputs.ndim >= 3 and param.numel() == inputs.shape[1]:
        return param.reshape(1, inputs.shape[1], *([1] * (inputs.ndim - 2)))
    if param.numel() != inputs.shape[-1]:
        raise RuntimeError(
            f"Cannot broadcast activation quantization parameter with {param.numel()} values "
            f"to input shape {tuple(inputs.shape)}"
        )
    return param.reshape(*([1] * (inputs.ndim - 1)), inputs.shape[-1])


def _activation_pre_hook(quantize):
    def hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args or not torch.is_tensor(args[0]):
            return args
        return (quantize(args[0]), *args[1:])

    return hook


def _activation_output_hook(quantize):
    def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        if torch.is_tensor(output):
            return quantize(output)
        if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
            return (quantize(output[0]), *output[1:])
        return output

    return hook


def _load_nunchaku_lite_patch_transformer():
    try:
        from nunchaku_lite import patch_transformer
    except ImportError as exc:
        raise RuntimeError("runtime='nunchaku-lite' requires the optional nunchaku_lite package") from exc
    return patch_transformer


def _nunchaku_lite_target(model_key: str) -> str:
    normalized = model_key.lower()
    if normalized.startswith("flux2") or "flux2" in normalized:
        return "flux2"
    if normalized.startswith("flux.1") or normalized.startswith("flux1") or normalized.startswith("flux"):
        return "flux"
    raise RuntimeError(f"nunchaku-lite evaluation does not support model_key={model_key!r}")
