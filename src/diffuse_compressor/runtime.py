from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import safetensors.torch
import torch
import torch.nn as nn

from .config import PatchRule
from .methods.svdquant.packing import fp4_e2m1_codebook, fp_quantize
from .patches import ShiftedLinear, prepare_model


RuntimeName = Literal["none", "nunchaku-lite", "torch-dequant"]
TorchDequantActivationMode = Literal["none", "input"]
PipelineMode = Literal["original", "quantized"]


@dataclass(frozen=True)
class RuntimePipelineSpec:
    """Configuration for loading one evaluation pipeline.

    The caller owns the evaluation loop, data loading, saving, and metrics.
    This spec only controls whether the loaded pipeline is returned as the
    original model or patched with an exported quantized checkpoint.
    """

    mode: PipelineMode = "original"
    runtime: RuntimeName = "none"
    checkpoint: str | Path | None = None
    model_key: str | None = None
    precision: str = "int4"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    torch_dequant_activation_mode: TorchDequantActivationMode = "none"

    def __post_init__(self) -> None:
        if self.mode not in {"original", "quantized"}:
            raise ValueError(f"Unsupported evaluation pipeline mode: {self.mode!r}")
        if self.runtime not in {"none", "nunchaku-lite", "torch-dequant"}:
            raise ValueError(f"Unsupported evaluation runtime: {self.runtime!r}")
        if self.torch_dequant_activation_mode not in {"none", "input"}:
            raise ValueError(f"Unsupported torch-dequant activation mode: {self.torch_dequant_activation_mode!r}")
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", Path(self.checkpoint))


def load_evaluation_pipeline(
    *,
    spec: RuntimePipelineSpec,
    pipeline: Any | None = None,
    loader: Any | None = None,
    pipeline_cls: type | None = None,
    model_id: str | None = None,
) -> Any:
    """Load one original or quantized pipeline for a user-owned eval loop.

    Exactly one source must be supplied: an existing ``pipeline``, a zero-arg
    ``loader`` callable, or ``pipeline_cls`` with ``model_id``.
    """

    pipe = _load_pipeline_source(
        pipeline=pipeline,
        loader=loader,
        pipeline_cls=pipeline_cls,
        model_id=model_id,
        torch_dtype=spec.torch_dtype,
    )
    if hasattr(pipe, "to"):
        pipe = pipe.to(spec.device)
    if spec.mode == "original":
        return pipe
    if spec.runtime == "none":
        raise ValueError("mode='quantized' requires runtime to be 'nunchaku-lite' or 'torch-dequant'")
    if spec.checkpoint is None:
        raise ValueError("mode='quantized' requires RuntimePipelineSpec.checkpoint")
    if not spec.model_key:
        raise ValueError("mode='quantized' requires RuntimePipelineSpec.model_key")
    return patch_quantized_pipeline(pipe, model_key=spec.model_key, spec=spec)


def _load_pipeline_source(
    *,
    pipeline: Any | None,
    loader: Any | None,
    pipeline_cls: type | None,
    model_id: str | None,
    torch_dtype: torch.dtype,
) -> Any:
    sources = [
        pipeline is not None,
        loader is not None,
        pipeline_cls is not None or model_id is not None,
    ]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one pipeline source: pipeline, loader, or pipeline_cls with model_id")
    if pipeline is not None:
        return pipeline
    if loader is not None:
        if not callable(loader):
            raise ValueError("loader must be callable")
        return loader()
    if pipeline_cls is None or model_id is None:
        raise ValueError("pipeline_cls and model_id must be provided together")
    return pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype)


def patch_quantized_pipeline(pipe: Any, *, model_key: str, spec: RuntimePipelineSpec) -> Any:
    """Patch a pipeline with an exported quantized checkpoint."""

    if spec.runtime == "none":
        return pipe
    if spec.runtime == "torch-dequant":
        return patch_pipeline_with_dequantized_checkpoint(pipe, model_key=model_key, spec=spec)
    if spec.runtime != "nunchaku-lite":
        raise RuntimeError(f"Unsupported quantized runtime: {spec.runtime!r}")
    if spec.checkpoint is None:
        raise RuntimeError("runtime='nunchaku-lite' requires RuntimePipelineSpec.checkpoint")

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


def patch_pipeline_with_dequantized_checkpoint(
    pipe: Any, *, model_key: str, spec: RuntimePipelineSpec
) -> Any:
    """Patch a pipeline by materializing packed quantized weights as PyTorch weights."""

    if spec.checkpoint is None:
        raise RuntimeError("runtime='torch-dequant' requires RuntimePipelineSpec.checkpoint")
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
        activation_mode=spec.torch_dequant_activation_mode,
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
    activation_mode: str = "none",
) -> None:
    modules = dict(transformer.named_modules())
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    for target in metadata.get("targets", []):
        export_name = str(target["export_name"])
        target_modules = [modules[name] for name in target["modules"]]
        if not all(_is_linear_like(module) for module in target_modules):
            raise RuntimeError(f"torch-dequant currently supports linear targets only, got {export_name!r}")
        if target.get("runtime_tensor_layout") == "nunchaku_packed":
            raise RuntimeError(
                "torch-dequant does not support Nunchaku-packed SVDQ tensors for "
                f"{export_name!r}; use runtime='nunchaku-lite' or export a logical-layout checkpoint"
            )
        weight = _reconstruct_target_weight(
            export_name=export_name,
            state=state,
            precision=str(target.get("precision", metadata.get("weight", {}).get("dtype", "int4"))),
            weight_layout=target.get("weight_layout", {"name": "svdq"}),
        ).to(dtype=torch_dtype)
        bias = state.get(f"{export_name}.bias")
        if bias is not None:
            bias = _undo_adanorm_awq_w4a16_bias(bias, target.get("weight_layout", {"name": "svdq"}))
            bias = bias.to(dtype=torch_dtype)
        _copy_target_weights(target_modules, weight, bias, export_name=export_name)
        hooks.extend(
            _register_activation_hooks(
                target_modules,
                export_name=export_name,
                target=target,
                state=state,
                mode=activation_mode,
            )
        )
    setattr(transformer, "_diffuse_compressor_torch_dequant_hooks", hooks)


def _reconstruct_target_weight(
    *,
    export_name: str,
    state: dict[str, torch.Tensor],
    precision: str,
    weight_layout: object = "svdq",
) -> torch.Tensor:
    qweight = state[f"{export_name}.qweight"]
    wscales = state[f"{export_name}.wscales"]
    wcscales = state.get(f"{export_name}.wcscales")
    wtscale = state.get(f"{export_name}.wtscale")
    layout_name = _weight_layout_name(weight_layout)
    if layout_name in {"awq_w4a16", "adanorm_awq_w4a16"}:
        weight = _dequantize_awq_w4a16_qweight(qweight, wscales, state[f"{export_name}.wzeros"])
        if layout_name == "adanorm_awq_w4a16":
            weight = _undo_adanorm_awq_w4a16_weight(weight, weight_layout)
    else:
        weight = _dequantize_qweight(qweight, wscales, precision=precision, wcscales=wcscales, wtscale=wtscale)
    proj_down = state.get(f"{export_name}.proj_down")
    proj_up = state.get(f"{export_name}.proj_up")
    if proj_down is not None and proj_up is not None:
        weight = weight + proj_up.float() @ proj_down.float().t()
    smooth = state.get(f"{export_name}.smooth_factor")
    if smooth is not None:
        weight = weight / smooth.float().view(1, -1)
    return weight


def _weight_layout_name(layout: object) -> str:
    if isinstance(layout, dict):
        return str(layout.get("name", "svdq"))
    name = getattr(layout, "name", None)
    if name is not None:
        return str(name)
    return str(layout)


def _undo_adanorm_awq_w4a16_weight(weight: torch.Tensor, layout: object) -> torch.Tensor:
    splits = _adanorm_layout_splits(layout)
    if splits is None:
        return weight
    oc, ic = weight.shape
    return weight.view(oc // splits, splits, ic).transpose(0, 1).reshape(oc, ic).contiguous()


def _undo_adanorm_awq_w4a16_bias(bias: torch.Tensor, layout: object) -> torch.Tensor:
    splits = _adanorm_layout_splits(layout)
    if splits is None:
        return bias
    oc = bias.numel()
    bias = bias.view(oc // splits, splits).clone()
    delta = torch.zeros(splits, dtype=bias.dtype, device=bias.device)
    delta[1] = 1
    delta[-2] = 1
    return bias.sub(delta.view(1, splits)).transpose(0, 1).reshape(oc).contiguous()


def _adanorm_layout_splits(layout: object) -> int | None:
    if isinstance(layout, dict):
        if layout.get("name") != "adanorm_awq_w4a16":
            return None
        splits = int(layout["splits"])
    elif getattr(layout, "name", None) == "adanorm_awq_w4a16":
        splits = int(getattr(layout, "splits"))
    else:
        return None
    if splits not in {3, 6}:
        raise RuntimeError(f"Unsupported AdaNorm AWQ W4A16 split count in metadata: {splits!r}")
    return splits


def _dequantize_awq_w4a16_qweight(qweight: torch.Tensor, wscales: torch.Tensor, wzeros: torch.Tensor) -> torch.Tensor:
    packed = qweight.cpu().to(torch.int32)
    groups, oc = wscales.shape
    rows = packed.shape[0]
    if rows * 4 != oc:
        raise RuntimeError(f"AWQ qweight output dimension {rows * 4} does not match wscales output dimension {oc}")
    if packed.shape[1] != groups * 32:
        raise RuntimeError(f"AWQ qweight shape {tuple(packed.shape)} does not match {groups} scale groups")
    codes = torch.empty((rows, 4, groups, 64), dtype=torch.float32)
    packed = packed.view(rows, groups, 4, 8)
    for packed_index, nibble, channel in _awq_w4a16_code_order():
        codes[:, :, :, channel] = (
            packed[:, :, :, packed_index].bitwise_right_shift(4 * nibble).bitwise_and(0xF).permute(0, 2, 1).float()
        )
    scale = wscales.float().t().contiguous().view(oc, groups, 1)
    zeros = wzeros.float().t().contiguous().view(oc, groups, 1)
    return (codes.view(oc, groups, 64) * scale + zeros).view(oc, groups * 64)


def _awq_w4a16_code_order() -> tuple[tuple[int, int, int], ...]:
    order = []
    for channel in range(64):
        packed_index = (channel // 32) * 4 + (channel % 8) // 2
        nibble = ((channel % 32) // 8) + 4 * (channel % 2)
        order.append((packed_index, nibble, channel))
    return tuple(order)


def _dequantize_qweight(
    qweight: torch.Tensor,
    wscales: torch.Tensor,
    *,
    precision: str,
    wcscales: torch.Tensor | None = None,
    wtscale: torch.Tensor | None = None,
) -> torch.Tensor:
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
    scale = wscales.float().t().contiguous()
    if wcscales is not None:
        scale = scale * wcscales.float().view(oc, 1)
    if wtscale is not None:
        scale = scale * wtscale.float().view(-1)[0]
    scale = scale.view(oc, groups, 1)
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
    mode: str,
) -> list[torch.utils.hooks.RemovableHandle]:
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    if mode == "none" or not _target_activation_quant_enabled(target):
        return hooks
    input_quant = _activation_input_quantizer_from_state(export_name, target, state)
    output_quant = None
    for module in modules:
        if input_quant is not None:
            hooks.append(module.register_forward_pre_hook(_activation_pre_hook(input_quant)))
        if output_quant is not None:
            hooks.append(module.register_forward_hook(_activation_output_hook(output_quant)))
    return hooks


def _target_activation_quant_enabled(target: dict[str, Any]) -> bool:
    metadata = target.get("activation_quant")
    if isinstance(metadata, dict):
        return bool(metadata.get("enabled", False))
    return metadata is None


def _activation_input_quantizer_from_state(
    export_name: str,
    target: dict[str, Any],
    state: dict[str, torch.Tensor],
):
    group_size = int(target.get("group_size", 0))
    precision = str(target.get("precision", "int4"))
    smooth = state.get(f"{export_name}.smooth_factor")
    if group_size <= 0:
        group_size = 16 if precision in {"fp4", "nvfp4", "fp4_e2m1_all", "sfp4_e2m1_all"} else 64

    def quantize(inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.float()
        expanded_smooth = None
        if smooth is not None:
            expanded_smooth = _expand_activation_smooth(
                smooth.to(device=inputs.device, dtype=torch.float32),
                inputs,
            )
            values = values / expanded_smooth
        values = _dynamic_fake_quantize_activation(
            values,
            group_size=group_size,
            float_point=precision in {"fp4", "nvfp4", "fp4_e2m1_all", "sfp4_e2m1_all"},
        )
        if expanded_smooth is not None:
            values = values * expanded_smooth
        return values.to(dtype=inputs.dtype)

    return quantize


def _expand_activation_smooth(smooth: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    if smooth.numel() == 1:
        return smooth.reshape(*([1] * inputs.ndim))
    if smooth.numel() != inputs.shape[-1]:
        raise RuntimeError(
            f"Cannot broadcast activation smooth factor with {smooth.numel()} values "
            f"to input shape {tuple(inputs.shape)}"
        )
    return smooth.reshape(*([1] * (inputs.ndim - 1)), inputs.shape[-1])


def _dynamic_fake_quantize_activation(inputs: torch.Tensor, *, group_size: int, float_point: bool) -> torch.Tensor:
    if inputs.shape[-1] % group_size != 0:
        raise RuntimeError(
            f"Cannot fake-quantize activation with last dimension {inputs.shape[-1]} "
            f"using group_size={group_size}"
        )
    original_shape = inputs.shape
    rows = inputs.reshape(-1, original_shape[-1])
    groups = original_shape[-1] // group_size
    grouped = rows.view(rows.shape[0], groups, group_size)
    max_q = 6 if float_point else 7
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / max_q
    normalized = grouped / scale
    if float_point:
        codebook = fp4_e2m1_codebook(device=inputs.device, dtype=torch.float32)
        qcodes = fp_quantize(normalized.reshape(-1), codebook=codebook)
        qvalues = codebook[qcodes.long()].view_as(grouped)
        scale = _fake_quantize_fp8_e4m3fn(scale.clamp_max(448.0))
        return (qvalues * scale).view(original_shape)
    qvalues = normalized.round().clamp(-8, 7)
    return (qvalues * scale).view(original_shape)


def _fake_quantize_fp8_e4m3fn(values: torch.Tensor) -> torch.Tensor:
    if not hasattr(torch, "float8_e4m3fn"):
        return values
    return values.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)


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
