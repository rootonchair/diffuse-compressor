from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import safetensors.torch
import torch
import torch.nn as nn

from .config import PatchRule
from .backends.nunchaku.packing import NunchakuWeightPacker, fp4_e2m1_codebook, fp_quantize
from .patches import ShiftedLinear, prepare_model


RuntimeName = Literal["none", "nunchaku-lite", "torch-dequant"]
TorchDequantActivationMode = Literal["none", "input"]
TorchDtypeName = Literal["auto", "none", "bfloat16", "float16", "float32"]
PipelineMode = Literal["original", "quantized"]
PipelineOffload = Literal["none", "model", "sequential"]


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
    nunchaku_lite_target: str | None = None
    precision: str = "int4"
    device: str = "cuda"
    torch_dtype: torch.dtype | TorchDtypeName | None = "auto"
    torch_dequant_activation_mode: TorchDequantActivationMode = "none"
    pipeline_offload: PipelineOffload = "none"

    def __post_init__(self) -> None:
        if self.mode not in {"original", "quantized"}:
            raise ValueError(f"Unsupported evaluation pipeline mode: {self.mode!r}")
        if self.runtime not in {"none", "nunchaku-lite", "torch-dequant"}:
            raise ValueError(f"Unsupported evaluation runtime: {self.runtime!r}")
        if self.torch_dequant_activation_mode not in {"none", "input"}:
            raise ValueError(f"Unsupported torch-dequant activation mode: {self.torch_dequant_activation_mode!r}")
        if self.pipeline_offload not in {"none", "model", "sequential"}:
            raise ValueError(f"Unsupported pipeline offload mode: {self.pipeline_offload!r}")
        if isinstance(self.torch_dtype, str) and self.torch_dtype not in {
            "auto",
            "none",
            "bfloat16",
            "float16",
            "float32",
        }:
            raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype!r}")
        if self.torch_dtype is not None and not isinstance(self.torch_dtype, (str, torch.dtype)):
            raise ValueError(f"Unsupported torch_dtype: {self.torch_dtype!r}")
        if self.nunchaku_lite_target is not None and not self.nunchaku_lite_target:
            raise ValueError("nunchaku_lite_target cannot be empty")
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
    ``loader`` callable, or a ``model_id`` optionally paired with a
    ``pipeline_cls``.
    """

    pipe = _load_pipeline_source(
        pipeline=pipeline,
        loader=loader,
        pipeline_cls=pipeline_cls,
        model_id=model_id,
        torch_dtype=_resolve_pipeline_torch_dtype(spec),
    )
    if spec.mode != "original":
        if spec.runtime == "none":
            raise ValueError("mode='quantized' requires runtime to be 'nunchaku-lite' or 'torch-dequant'")
        if spec.checkpoint is None:
            raise ValueError("mode='quantized' requires RuntimePipelineSpec.checkpoint")
    if spec.pipeline_offload == "none" and hasattr(pipe, "to"):
        pipe = pipe.to(spec.device)
    if spec.mode != "original":
        # Patch before enabling offload: offload hooks (sequential especially)
        # move weights to meta/offload maps, and dequantized weights copied into
        # hollowed-out parameters are silently lost, producing NaN outputs.
        pipe = patch_quantized_pipeline(pipe, spec=spec)
    if spec.pipeline_offload != "none":
        _enable_pipeline_offload(pipe, spec)
    return pipe


def _load_pipeline_source(
    *,
    pipeline: Any | None,
    loader: Any | None,
    pipeline_cls: type | None,
    model_id: str | None,
    torch_dtype: torch.dtype | None,
) -> Any:
    sources = [pipeline is not None, loader is not None, model_id is not None]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one pipeline source: pipeline, loader, or model_id")
    if pipeline is not None:
        return pipeline
    if loader is not None:
        if not callable(loader):
            raise ValueError("loader must be callable")
        return loader()
    if model_id is None:
        raise ValueError("model_id must be provided")
    if pipeline_cls is None:
        from diffusers import DiffusionPipeline

        pipeline_cls = DiffusionPipeline
    kwargs = {}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    return pipeline_cls.from_pretrained(model_id, **kwargs)


def _resolve_pipeline_torch_dtype(spec: RuntimePipelineSpec) -> torch.dtype | None:
    torch_dtype = spec.torch_dtype
    if torch_dtype is None or torch_dtype == "none":
        return None
    if isinstance(torch_dtype, torch.dtype):
        return torch_dtype
    if torch_dtype == "auto":
        if spec.checkpoint is None:
            return None
        return _infer_checkpoint_torch_dtype(Path(spec.checkpoint))
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    return dtype_map[torch_dtype]


def _infer_checkpoint_torch_dtype(checkpoint_path: Path) -> torch.dtype | None:
    if not checkpoint_path.exists():
        return None
    found: set[torch.dtype] = set()
    try:
        with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                dtype = _safetensors_dtype(handle.get_slice(key).get_dtype())
                if dtype is not None:
                    found.add(dtype)
    except Exception:
        return None
    if torch.bfloat16 in found:
        return torch.bfloat16
    if torch.float16 in found:
        return torch.float16
    if torch.float32 in found:
        return torch.float32
    return None


def _safetensors_dtype(dtype: str) -> torch.dtype | None:
    return {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}.get(dtype)


def _enable_pipeline_offload(pipe: Any, spec: RuntimePipelineSpec) -> None:
    method_name = "enable_model_cpu_offload" if spec.pipeline_offload == "model" else "enable_sequential_cpu_offload"
    method = getattr(pipe, method_name, None)
    if method is None:
        raise RuntimeError(f"Pipeline does not support {method_name}()")
    try:
        method(device=spec.device)
    except TypeError:
        method()


def patch_quantized_pipeline(pipe: Any, *, spec: RuntimePipelineSpec) -> Any:
    """Patch a pipeline with an exported quantized checkpoint."""

    if spec.runtime == "none":
        return pipe
    if spec.runtime == "torch-dequant":
        return patch_pipeline_with_dequantized_checkpoint(pipe, spec=spec)
    if spec.runtime != "nunchaku-lite":
        raise RuntimeError(f"Unsupported quantized runtime: {spec.runtime!r}")
    if spec.checkpoint is None:
        raise RuntimeError("runtime='nunchaku-lite' requires RuntimePipelineSpec.checkpoint")
    if spec.nunchaku_lite_target is None:
        raise RuntimeError("runtime='nunchaku-lite' requires RuntimePipelineSpec.nunchaku_lite_target")

    patch_transformer = _load_nunchaku_lite_patch_transformer()
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("nunchaku-lite evaluation requires the pipeline to expose a transformer")
    kwargs: dict[str, Any] = {"target": spec.nunchaku_lite_target, "precision": spec.precision}
    adapter_options = _nunchaku_lite_adapter_options(spec.checkpoint, target=spec.nunchaku_lite_target)
    if adapter_options:
        kwargs["adapter_options"] = adapter_options
    patch_transformer(pipe.transformer, spec.checkpoint, **kwargs)
    return pipe


def _nunchaku_lite_adapter_options(checkpoint_path: Path, *, target: str) -> dict[str, Any]:
    if target == "manifest":
        return {}
    metadata = _load_config(checkpoint_path)
    if metadata is None:
        return {}
    rank = metadata.get("rank")
    if isinstance(rank, int) and rank > 0:
        return {"rank": rank}
    return {}


def patch_pipeline_with_dequantized_checkpoint(pipe: Any, *, spec: RuntimePipelineSpec) -> Any:
    """Patch a pipeline by materializing packed quantized weights as PyTorch weights."""

    if spec.checkpoint is None:
        raise RuntimeError("runtime='torch-dequant' requires RuntimePipelineSpec.checkpoint")
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("torch-dequant evaluation requires the pipeline to expose a transformer")

    state, metadata = _load_checkpoint(spec.checkpoint)
    transformer = pipe.transformer
    _prepare_transformer_for_dequant(transformer, metadata)
    _load_dequantized_transformer_state(
        transformer, state=state, metadata=metadata, activation_mode=spec.torch_dequant_activation_mode
    )
    return pipe


def _load_checkpoint(path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    checkpoint_path = Path(path)
    metadata = _load_checkpoint_metadata(checkpoint_path)
    if "targets" not in metadata:
        raise RuntimeError(
            f"torch-dequant requires target metadata in {_config_path(checkpoint_path)} "
            "or a legacy full quantization_config"
        )
    return safetensors.torch.load_file(checkpoint_path, device="cpu"), metadata


def _load_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any]:
    checkpoint_metadata = _load_legacy_checkpoint_metadata(checkpoint_path)
    config_metadata = _load_config(checkpoint_path)
    if checkpoint_metadata is None and config_metadata is None:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} does not have config {_config_path(checkpoint_path)} "
            "or legacy quantization_config metadata"
        )
    return _merge_metadata(checkpoint_metadata or {}, config_metadata or {})


def _load_config(checkpoint_path: Path) -> dict[str, Any] | None:
    path = _config_path(checkpoint_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".config.yaml")


def _load_legacy_checkpoint_metadata(checkpoint_path: Path) -> dict[str, Any] | None:
    with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        metadata_blob = (handle.metadata() or {}).get("quantization_config")
    if metadata_blob is None:
        return None
    return json.loads(metadata_blob)


def _merge_metadata(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_metadata(merged[key], value)
        else:
            merged[key] = value
    return merged


def _prepare_transformer_for_dequant(transformer: nn.Module, metadata: dict[str, Any]) -> None:
    structural_patches = _structural_patch_rules_from_metadata(metadata)
    if structural_patches:
        prepare_model(transformer, structural_patches)
    _apply_activation_shift_patches(transformer, metadata)


def _structural_patch_rules_from_metadata(metadata: dict[str, Any]) -> list[PatchRule]:
    patch_records = metadata.get("structural_patches", [])
    if patch_records is None:
        return []
    if not isinstance(patch_records, list):
        raise RuntimeError("torch-dequant checkpoint structural_patches must be a list")
    patches: list[PatchRule] = []
    for index, patch_record in enumerate(patch_records):
        if not isinstance(patch_record, dict):
            raise RuntimeError(f"torch-dequant structural patch {index} must be an object")
        patch_type = patch_record.get("type")
        if patch_type not in {"split_linear", "split_linear_output", "split_conv"}:
            raise RuntimeError(f"torch-dequant structural patch {index} has unsupported type {patch_type!r}")
        module = patch_record.get("module")
        if not isinstance(module, str) or not module:
            raise RuntimeError(f"torch-dequant structural patch {index} must specify a module pattern")
        args = patch_record.get("args", {})
        if not isinstance(args, dict):
            raise RuntimeError(f"torch-dequant structural patch {index} args must be an object")
        patches.append(PatchRule(type=patch_type, module=module, args=args))
    return patches


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
    transformer: nn.Module, *, state: dict[str, torch.Tensor], metadata: dict[str, Any], activation_mode: str = "none"
) -> None:
    modules = dict(transformer.named_modules())
    hooks: list[torch.utils.hooks.RemovableHandle] = []
    for target in metadata.get("targets", []):
        export_name = str(target["export_name"])
        target_modules = _target_modules_from_metadata(modules, target, export_name=export_name)
        if not all(_is_linear_like(module) for module in target_modules):
            raise RuntimeError(f"torch-dequant currently supports linear targets only, got {export_name!r}")
        target_state = state
        if target.get("runtime_tensor_layout") == "nunchaku_packed":
            rows, columns = _target_weight_shape(target_modules, export_name=export_name)
            target_state = _unpack_nunchaku_packed_target_state(
                export_name=export_name,
                state=state,
                target=target,
                rows=rows,
                columns=columns,
                rank=int(target.get("rank", metadata.get("rank", 0)) or 0),
            )
        target_dtype = target_modules[0].weight.dtype
        weight = _reconstruct_target_weight(
            export_name=export_name,
            state=target_state,
            precision=str(target.get("precision", metadata.get("weight", {}).get("dtype", "int4"))),
            weight_layout=target.get("weight_layout", {"name": "svdq"}),
            lowrank_unsmoothed=target.get("runtime_tensor_layout") == "nunchaku_packed",
        ).to(dtype=target_dtype)
        bias = target_state.get(f"{export_name}.bias")
        if bias is not None:
            bias = _undo_adanorm_awq_w4a16_bias(bias, target.get("weight_layout", {"name": "svdq"}))
            bias = bias.to(dtype=target_dtype)
        _copy_target_weights(target_modules, weight, bias, export_name=export_name)
        hooks.extend(
            _register_activation_hooks(
                target_modules, export_name=export_name, target=target, state=target_state, mode=activation_mode
            )
        )
    setattr(transformer, "_diffuse_compressor_torch_dequant_hooks", hooks)


def _target_modules_from_metadata(
    modules: dict[str, nn.Module], target: dict[str, Any], *, export_name: str
) -> list[nn.Module]:
    target_modules = []
    for module_name in target.get("modules", []):
        if module_name not in modules:
            raise RuntimeError(
                f"torch-dequant checkpoint target {export_name!r} references missing module {module_name!r}; "
                "exported structural_patches may be required"
            )
        target_modules.append(modules[module_name])
    return target_modules


def _target_weight_shape(modules: list[nn.Module], *, export_name: str) -> tuple[int, int]:
    rows = 0
    columns: int | None = None
    for module in modules:
        linear = _as_linear(module)
        if columns is None:
            columns = linear.weight.shape[1]
        elif columns != linear.weight.shape[1]:
            raise RuntimeError(f"torch-dequant target {export_name!r} mixes input dimensions")
        rows += linear.weight.shape[0]
    if columns is None:
        raise RuntimeError(f"torch-dequant target {export_name!r} does not reference any modules")
    return rows, columns


def _unpack_nunchaku_packed_target_state(
    *, export_name: str, state: dict[str, torch.Tensor], target: dict[str, Any], rows: int, columns: int, rank: int
) -> dict[str, torch.Tensor]:
    layout_name = _weight_layout_name(target.get("weight_layout", {"name": "svdq"}))
    if layout_name in {"awq_w4a16", "adanorm_awq_w4a16"}:
        raise RuntimeError(f"torch-dequant does not support Nunchaku-packed AWQ target {export_name!r}")

    group_size = int(target.get("group_size", 0) or 0)
    if group_size <= 0:
        raise RuntimeError(f"torch-dequant packed target {export_name!r} must specify a positive group_size")
    if columns % group_size != 0:
        raise RuntimeError(
            f"torch-dequant packed target {export_name!r} input dimension {columns} "
            f"is not divisible by group_size={group_size}"
        )
    groups = columns // group_size
    packer = NunchakuWeightPacker(bits=4)
    unpacked = dict(state)
    prefix = f"{export_name}."

    qcodes = packer.unpack_weight(state[f"{prefix}qweight"], rows=rows, columns=columns)
    unpacked[f"{prefix}qweight"] = _pack_qcodes(qcodes)

    wscales = packer.unpack_scale(state[f"{prefix}wscales"], rows=rows, groups=groups, group_size=group_size)
    unpacked[f"{prefix}wscales"] = wscales.view(rows, groups).t().contiguous()

    wcscales = state.get(f"{prefix}wcscales")
    if wcscales is not None:
        if wcscales.numel() == rows:
            unpacked[f"{prefix}wcscales"] = packer.unpack_scale(wcscales, rows=rows, groups=1, group_size=-1)
        else:
            unpacked_wcscales = packer.unpack_scale(wcscales, rows=rows, groups=groups, group_size=group_size)
            unpacked[f"{prefix}wcscales"] = unpacked_wcscales.view(rows, groups).t().contiguous()

    smooth = state.get(f"{prefix}smooth_factor")
    if smooth is not None:
        unpacked[f"{prefix}smooth_factor"] = packer.unpack_scale(smooth, rows=columns, groups=1, group_size=-1)

    bias = state.get(f"{prefix}bias")
    if bias is not None:
        unpacked[f"{prefix}bias"] = packer.unpack_scale(bias, rows=rows, groups=1, group_size=-1)

    proj_down = state.get(f"{prefix}proj_down")
    proj_up = state.get(f"{prefix}proj_up")
    if proj_down is not None or proj_up is not None:
        if proj_down is None or proj_up is None:
            raise RuntimeError(f"torch-dequant packed target {export_name!r} has incomplete low-rank tensors")
        if rank <= 0:
            rank = min(proj_down.shape[1], proj_up.shape[1])
        down = packer.unpack_lowrank_weight(proj_down, down=True, rows=rank, columns=columns)
        up = packer.unpack_lowrank_weight(proj_up, down=False, rows=rows, columns=rank)
        unpacked[f"{prefix}proj_down"] = down.t().contiguous()
        unpacked[f"{prefix}proj_up"] = up.contiguous()

    return unpacked


def _pack_qcodes(qcodes: torch.Tensor) -> torch.Tensor:
    if qcodes.ndim != 2:
        raise RuntimeError(f"Expected 2D quantized weight codes, got shape {tuple(qcodes.shape)}")
    if qcodes.shape[1] % 2 != 0:
        raise RuntimeError(f"Quantized weight input dimension {qcodes.shape[1]} must be even")
    qcodes = qcodes.to(torch.int16)
    lo = qcodes[:, 0::2].bitwise_and(0xF)
    hi = qcodes[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    return lo.bitwise_or(hi).to(torch.uint8).view(torch.int8).contiguous()


def _reconstruct_target_weight(
    *,
    export_name: str,
    state: dict[str, torch.Tensor],
    precision: str,
    weight_layout: object = "svdq",
    lowrank_unsmoothed: bool = False,
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
    smooth = state.get(f"{export_name}.smooth_factor")
    # Nunchaku-packed exports fold 1/smooth into proj_down (the kernel's low-rank
    # branch consumes the unsmoothed input), so only the quantized residual is in
    # smoothed coordinates; logical-layout exports store both in smoothed space.
    if proj_down is not None and proj_up is not None and not lowrank_unsmoothed:
        weight = weight + proj_up.float() @ proj_down.float().t()
    if smooth is not None:
        weight = weight / smooth.float().view(1, -1)
    if proj_down is not None and proj_up is not None and lowrank_unsmoothed:
        weight = weight + proj_up.float() @ proj_down.float().t()
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
    modules: list[nn.Module], weight: torch.Tensor, bias: torch.Tensor | None, *, export_name: str
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
                corrected_bias = original_bias.view(-1) - chunk.to(
                    device=linear.weight.device, dtype=torch.float32
                ) @ shift.view(-1)
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
    modules: list[nn.Module], *, export_name: str, target: dict[str, Any], state: dict[str, torch.Tensor], mode: str
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


def _activation_input_quantizer_from_state(export_name: str, target: dict[str, Any], state: dict[str, torch.Tensor]):
    group_size = int(target.get("group_size", 0))
    precision = str(target.get("precision", "int4"))
    smooth = state.get(f"{export_name}.smooth_factor")
    if group_size <= 0:
        group_size = 16 if precision in {"fp4", "nvfp4", "fp4_e2m1_all", "sfp4_e2m1_all"} else 64

    def quantize(inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.float()
        expanded_smooth = None
        if smooth is not None:
            expanded_smooth = _expand_activation_smooth(smooth.to(device=inputs.device, dtype=torch.float32), inputs)
            values = values / expanded_smooth
        values = _dynamic_fake_quantize_activation(
            values, group_size=group_size, float_point=precision in {"fp4", "nvfp4", "fp4_e2m1_all", "sfp4_e2m1_all"}
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
            f"Cannot fake-quantize activation with last dimension {inputs.shape[-1]} using group_size={group_size}"
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
