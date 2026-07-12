"""Convert Qwen-Image-Edit-2509 Nunchaku checkpoints to Diffusers layout.

The source checkpoints from ``nunchaku-ai/nunchaku-qwen-image-edit-2509`` use
fused QKV projection keys. Diffusers' Nunchaku Lite loader expects the native
``QwenImageTransformer2DModel`` module names, where Q, K, and V are separate
linears. This script rewrites the transformer state dict and packages it with a
bitsandbytes 4-bit text encoder, following the layout used by the reference
``lite-infer/Qwen-Image-nunchaku-lite-nvfp4_r32-bnb4-text-encoder`` repo.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Precision = Literal["int4", "nvfp4"]

BASE_MODEL = "Qwen/Qwen-Image-Edit-2509"
NUNCHAKU_REPO = "nunchaku-ai/nunchaku-qwen-image-edit-2509"
QWEN_EDIT_PIPELINE_CLASS = "QwenImageEditPlusPipeline"
TRANSFORMER_FILE = "diffusion_pytorch_model.safetensors"

DROP_SUFFIXES = {"smooth_factor_orig"}
QKV_SPLIT_PREFIXES = {
    "attn.to_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
    "attn.add_qkv_proj": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
}
ADANORM_AWQ_SUFFIXES = ("img_mod.1", "txt_mod.1")
ADANORM_AWQ_SPLITS = 6


@dataclass(frozen=True)
class TransformerConversionSpec:
    num_layers: int
    precision: Precision
    rank: int

    @property
    def group_size(self) -> int:
        return 16 if self.precision == "nvfp4" else 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a Qwen-Image-Edit-2509 Nunchaku checkpoint into a full Diffusers pipeline directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=BASE_MODEL, help="Base Diffusers pipeline repo or local directory.")
    parser.add_argument("--nunchaku-repo", default=NUNCHAKU_REPO, help="HF repo used when checkpoint is not a path.")
    parser.add_argument(
        "--nunchaku-checkpoint",
        required=True,
        help="Source safetensors path, or filename inside --nunchaku-repo.",
    )
    parser.add_argument("--output-dir", required=True, help="Output Diffusers pipeline directory.")
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default=None, help="Override precision inference.")
    parser.add_argument("--rank", type=int, default=None, help="Override rank inference.")
    parser.add_argument("--torch-dtype", default="bfloat16", help="Pipeline/text encoder dtype.")
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Device map passed while loading the text encoder for bitsandbytes conversion.",
    )
    parser.add_argument(
        "--skip-text-encoder",
        action="store_true",
        help="Only package inherited files and converted transformer; do not quantize/save text_encoder.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve base model and source checkpoint from local HF cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_checkpoint(
        args.nunchaku_checkpoint,
        repo_id=args.nunchaku_repo,
        local_files_only=args.local_files_only,
    )
    precision = args.precision or infer_precision(checkpoint.name)
    rank = args.rank or infer_rank(checkpoint.name)

    base_snapshot = snapshot_base_components(
        args.base_model,
        output_dir=output_dir,
        include_text_encoder=args.skip_text_encoder,
        local_files_only=args.local_files_only,
    )
    transformer_config = load_json(base_snapshot / "transformer" / "config.json")
    spec = TransformerConversionSpec(
        num_layers=int(transformer_config["num_layers"]),
        precision=precision,
        rank=rank,
    )

    convert_transformer_checkpoint(
        checkpoint,
        output_dir / "transformer" / TRANSFORMER_FILE,
        spec=spec,
    )
    write_transformer_config(output_dir / "transformer" / "config.json", transformer_config, spec=spec)
    write_model_index(output_dir / "model_index.json")

    if not args.skip_text_encoder:
        quantize_text_encoder(
            args.base_model,
            output_dir=output_dir / "text_encoder",
            torch_dtype_name=args.torch_dtype,
            device_map=args.device_map,
            local_files_only=args.local_files_only,
        )


def resolve_checkpoint(checkpoint: str, *, repo_id: str, local_files_only: bool) -> Path:
    from huggingface_hub import hf_hub_download

    path = Path(checkpoint)
    if path.exists():
        return path
    return Path(hf_hub_download(repo_id, checkpoint, local_files_only=local_files_only))


def snapshot_base_components(
    model_id: str, *, output_dir: Path, include_text_encoder: bool, local_files_only: bool
) -> Path:
    from huggingface_hub import snapshot_download

    allow_patterns = [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "processor/*",
        "vae/*",
        "transformer/config.json",
    ]
    if include_text_encoder:
        allow_patterns.append("text_encoder/*")
    local_model = Path(model_id)
    snapshot = (
        local_model
        if local_model.exists()
        else Path(snapshot_download(model_id, allow_patterns=allow_patterns, local_files_only=local_files_only))
    )
    for name in ("scheduler", "tokenizer", "processor", "vae"):
        source = snapshot / name
        if source.exists():
            copy_tree(source, output_dir / name)
    transformer_source = snapshot / "transformer"
    if transformer_source.exists():
        transformer_destination = output_dir / "transformer"
        transformer_destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(transformer_source / "config.json", transformer_destination / "config.json")
    if include_text_encoder and (snapshot / "text_encoder").exists():
        copy_tree(snapshot / "text_encoder", output_dir / "text_encoder")
    model_index = snapshot / "model_index.json"
    if model_index.exists():
        shutil.copy2(model_index, output_dir / "model_index.json")
    return snapshot


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def convert_transformer_checkpoint(source: Path, destination: Path, *, spec: TransformerConversionSpec) -> None:
    import safetensors.torch
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(source, framework="pt", device="cpu") as handle:
        source_keys = set(handle.keys())
        for block in range(spec.num_layers):
            for fused_suffix, split_suffixes in QKV_SPLIT_PREFIXES.items():
                fused_prefix = f"transformer_blocks.{block}.{fused_suffix}"
                if f"{fused_prefix}.qweight" in source_keys:
                    split_fused_qkv(
                        handle,
                        converted,
                        fused_prefix=fused_prefix,
                        split_suffixes=split_suffixes,
                        precision=spec.precision,
                    )

        for key in handle.keys():
            if any(key.startswith(f"transformer_blocks.{block}.attn.to_qkv.") for block in range(spec.num_layers)):
                continue
            if any(
                key.startswith(f"transformer_blocks.{block}.attn.add_qkv_proj.") for block in range(spec.num_layers)
            ):
                continue
            if key.rsplit(".", 1)[-1] in DROP_SUFFIXES:
                continue
            converted[key] = handle.get_tensor(key)

    for target in svdq_targets(spec.num_layers):
        if has_target_state(converted, target):
            ensure_nvfp4_scales(converted, target, precision=spec.precision)
    for block in range(spec.num_layers):
        for suffix in ADANORM_AWQ_SUFFIXES:
            undo_adanorm_awq_layout(converted, f"transformer_blocks.{block}.{suffix}", splits=ADANORM_AWQ_SPLITS)

    expected_targets = set(svdq_targets(spec.num_layers)) | set(awq_targets(spec.num_layers))
    missing = [target for target in sorted(expected_targets) if not has_target_state(converted, target)]
    if missing:
        sample = ", ".join(missing[:10])
        raise RuntimeError(f"Converted checkpoint is missing {len(missing)} expected targets, first missing: {sample}")

    safetensors.torch.save_file(converted, destination)


def split_fused_qkv(
    handle, output: dict, *, fused_prefix: str, split_suffixes: tuple[str, str, str], precision: Precision
) -> None:
    from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker
    import torch

    qweight = handle.get_tensor(f"{fused_prefix}.qweight")
    if qweight.shape[0] % 3 != 0:
        raise RuntimeError(f"{fused_prefix}.qweight output rows are not divisible by 3: {tuple(qweight.shape)}")
    rows = qweight.shape[0] // 3
    columns = qweight.shape[1] * 2
    group_size = 16 if precision == "nvfp4" else 64
    packer = NunchakuWeightPacker(bits=4)
    unpacked = unpack_fused_qkv_tensors(
        handle, fused_prefix=fused_prefix, rows=qweight.shape[0], columns=columns, group_size=group_size
    )
    for index, suffix in enumerate(split_suffixes):
        target_prefix = f"{fused_prefix.rsplit('.', 1)[0]}.{suffix.rsplit('.', 1)[-1]}"
        row_slice = slice(index * rows, (index + 1) * rows)
        for source_key in handle.keys():
            if not source_key.startswith(f"{fused_prefix}."):
                continue
            tensor_name = source_key[len(fused_prefix) + 1 :]
            if tensor_name in DROP_SUFFIXES:
                continue
            tensor = handle.get_tensor(source_key)
            output[f"{target_prefix}.{tensor_name}"] = split_or_duplicate_qkv_tensor(
                tensor,
                tensor_name=tensor_name,
                row_slice=row_slice,
                packer=packer,
                unpacked=unpacked,
                group_size=group_size,
            )
        if precision == "nvfp4" and f"{target_prefix}.wtscale" not in output:
            scale_source = output.get(f"{target_prefix}.wcscales")
            dtype = scale_source.dtype if scale_source is not None else torch.bfloat16
            output[f"{target_prefix}.wtscale"] = torch.ones(1, dtype=dtype)


def unpack_fused_qkv_tensors(handle, *, fused_prefix: str, rows: int, columns: int, group_size: int) -> dict:
    from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    unpacked = {
        "qweight": packer.unpack_weight(handle.get_tensor(f"{fused_prefix}.qweight"), rows=rows, columns=columns),
    }
    if f"{fused_prefix}.wscales" in handle.keys():
        wscales = handle.get_tensor(f"{fused_prefix}.wscales")
        unpacked["wscales"] = packer.unpack_scale(
            wscales, rows=rows, groups=wscales.shape[0], group_size=group_size
        )
    for tensor_name in ("bias", "wcscales"):
        key = f"{fused_prefix}.{tensor_name}"
        if key in handle.keys():
            unpacked[tensor_name] = packer.unpack_scale(handle.get_tensor(key), rows=rows, groups=1, group_size=-1)
    key = f"{fused_prefix}.proj_up"
    if key in handle.keys():
        proj_up = handle.get_tensor(key)
        unpacked["proj_up"] = packer.unpack_lowrank_weight(proj_up, down=False, rows=rows, columns=proj_up.shape[1])
    return unpacked


def split_or_duplicate_qkv_tensor(tensor, *, tensor_name: str, row_slice: slice, packer, unpacked: dict, group_size: int):
    if tensor_name == "qweight":
        return packer.pack_weight(packer.pad_weight(unpacked[tensor_name][row_slice])).contiguous()
    if tensor_name == "wscales":
        return packer.pack_scale(
            packer.pad_scale(unpacked[tensor_name][row_slice], group_size=group_size), group_size=group_size
        )
    if tensor_name in {"bias", "wcscales"}:
        logical = unpacked[tensor_name][row_slice]
        return packer.pack_scale(packer.pad_scale(logical.view(-1, 1), group_size=-1), group_size=-1)
    if tensor_name == "proj_up":
        return packer.pack_lowrank_weight(unpacked[tensor_name][row_slice], down=False)
    if tensor_name in {"smooth_factor", "proj_down", "wtscale"}:
        return tensor.clone().contiguous()
    raise RuntimeError(f"Unsupported fused QKV tensor suffix: {tensor_name!r}")


def undo_adanorm_awq_layout(state: dict, target: str, *, splits: int) -> None:
    qweight = state.get(f"{target}.qweight")
    if qweight is None:
        return
    wscales = state.get(f"{target}.wscales")
    wzeros = state.get(f"{target}.wzeros")
    bias = state.get(f"{target}.bias")
    if wscales is None or wzeros is None:
        raise RuntimeError(f"{target} is missing AWQ scales needed to undo AdaNorm layout.")

    out_features = wscales.shape[1]
    state[f"{target}.qweight"] = undo_adanorm_awq_qweight_order(qweight, out_features=out_features, splits=splits)
    state[f"{target}.wscales"] = undo_adanorm_scale_order(wscales, splits=splits).contiguous()
    state[f"{target}.wzeros"] = undo_adanorm_scale_order(wzeros, splits=splits).contiguous()
    if bias is not None:
        state[f"{target}.bias"] = undo_adanorm_bias_order(bias, splits=splits).contiguous()


def undo_adanorm_awq_qweight_order(qweight, *, out_features: int, splits: int):
    groups = qweight.shape[1] // 32
    packed_groups = qweight.reshape(out_features // 4, groups, 4, 8).permute(0, 2, 1, 3).reshape(
        out_features, groups, 8
    )
    restored = undo_adanorm_output_order(packed_groups, splits=splits)
    return restored.reshape(out_features // 4, 4, groups, 8).permute(0, 2, 1, 3).reshape_as(qweight).contiguous()


def undo_adanorm_output_order(tensor, *, splits: int):
    out_features = tensor.shape[0]
    if out_features % splits != 0:
        raise RuntimeError(f"AdaNorm output features ({out_features}) must be divisible by {splits}.")
    chunk = out_features // splits
    return tensor.reshape(chunk, splits, *tensor.shape[1:]).transpose(0, 1).reshape_as(tensor).contiguous()


def undo_adanorm_scale_order(tensor, *, splits: int):
    return undo_adanorm_output_order(tensor.transpose(0, 1).contiguous(), splits=splits).transpose(0, 1)


def undo_adanorm_bias_order(bias, *, splits: int):
    chunk = bias.shape[0] // splits
    restored = bias.reshape(chunk, splits).clone()
    restored[:, 1] -= 1
    restored[:, -2] -= 1
    return restored.transpose(0, 1).reshape_as(bias)


def unpack_awq_qweight(qweight, *, out_features: int, in_features: int):
    import torch

    groups = in_features // 64
    packed = qweight.reshape(out_features // 4, groups, 4, 8).permute(0, 2, 1, 3).reshape(out_features, groups, 8)
    ordered = torch.empty(out_features, groups, 8, 8, dtype=torch.int32, device=qweight.device)
    packed_i32 = packed.to(torch.int32)
    for nibble in range(8):
        ordered[:, :, :, nibble] = packed_i32.bitwise_right_shift(4 * nibble).bitwise_and(0xF)
    inverse_order = torch.empty(64, dtype=torch.long, device=qweight.device)
    inverse_order[awq_w4a16_code_order(qweight.device)] = torch.arange(64, dtype=torch.long, device=qweight.device)
    return ordered.reshape(out_features, groups, 64).index_select(dim=2, index=inverse_order).reshape(
        out_features, in_features
    )


def pack_awq_qcodes(codes):
    import torch

    out_features, in_features = codes.shape
    groups = in_features // 64
    ordered = codes.reshape(out_features, groups, 64).index_select(dim=2, index=awq_w4a16_code_order(codes.device))
    ordered = ordered.reshape(out_features, groups, 8, 8).to(torch.int32)
    packed_groups = torch.zeros((out_features, groups, 8), dtype=torch.int32, device=codes.device)
    for nibble in range(8):
        packed_groups.bitwise_or_(ordered[:, :, :, nibble].bitwise_and(0xF) << (4 * nibble))
    return (
        packed_groups.reshape(out_features // 4, 4, groups, 8)
        .permute(0, 2, 1, 3)
        .reshape(out_features // 4, groups * 32)
        .contiguous()
    )


def awq_w4a16_code_order(device):
    import torch

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


def ensure_nvfp4_scales(state: dict, target: str, *, precision: Precision) -> None:
    if precision != "nvfp4":
        return
    import torch

    qweight = state.get(f"{target}.qweight")
    if qweight is None:
        return
    dtype = _scale_dtype_for_target(state, target)
    if f"{target}.wtscale" not in state:
        state[f"{target}.wtscale"] = torch.ones(1, dtype=dtype)
    if f"{target}.wcscales" not in state:
        state[f"{target}.wcscales"] = torch.ones(qweight.shape[0], dtype=dtype)


def _scale_dtype_for_target(state: dict, target: str):
    for suffix in ("wtscale", "wcscales", "bias", "smooth_factor"):
        tensor = state.get(f"{target}.{suffix}")
        if tensor is not None:
            return tensor.dtype
    import torch

    return torch.bfloat16


def has_target_state(state: dict | set[str], target: str) -> bool:
    keys = state if isinstance(state, set) else set(state)
    return f"{target}.qweight" in keys


def svdq_targets(num_layers: int) -> list[str]:
    targets: list[str] = []
    for block in range(num_layers):
        prefix = f"transformer_blocks.{block}"
        targets.extend(
            [
                f"{prefix}.attn.add_q_proj",
                f"{prefix}.attn.add_k_proj",
                f"{prefix}.attn.add_v_proj",
                f"{prefix}.attn.to_add_out",
                f"{prefix}.attn.to_out.0",
                f"{prefix}.attn.to_q",
                f"{prefix}.attn.to_k",
                f"{prefix}.attn.to_v",
                f"{prefix}.img_mlp.net.0.proj",
                f"{prefix}.img_mlp.net.2",
                f"{prefix}.txt_mlp.net.0.proj",
                f"{prefix}.txt_mlp.net.2",
            ]
        )
    return targets


def awq_targets(num_layers: int) -> list[str]:
    return [
        target
        for block in range(num_layers)
        for target in (f"transformer_blocks.{block}.img_mod.1", f"transformer_blocks.{block}.txt_mod.1")
    ]


def write_transformer_config(path: Path, config: dict, *, spec: TransformerConversionSpec) -> None:
    config = dict(config)
    config["quantization_config"] = {
        "quant_method": "nunchaku_lite",
        "compute_dtype": "bfloat16",
        "svdq_w4a4": {
            "precision": spec.precision,
            "group_size": spec.group_size,
            "rank": spec.rank,
            "targets": sorted(svdq_targets(spec.num_layers)),
        },
        "awq_w4a16": {
            "precision": "int4",
            "group_size": 64,
            "targets": sorted(awq_targets(spec.num_layers)),
        },
    }
    write_json(path, config)


def write_model_index(path: Path) -> None:
    model_index = load_json(path) if path.exists() else {}
    model_index.update(
        {
            "_class_name": QWEN_EDIT_PIPELINE_CLASS,
            "processor": ["transformers", "Qwen2VLProcessor"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
            "tokenizer": ["transformers", "Qwen2Tokenizer"],
            "transformer": ["diffusers", "QwenImageTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
        }
    )
    write_json(path, model_index)


def quantize_text_encoder(
    model_id: str, *, output_dir: Path, torch_dtype_name: str, device_map: str | None, local_files_only: bool
) -> None:
    import torch
    from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    dtype = getattr(torch, torch_dtype_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    kwargs = {
        "subfolder": "text_encoder",
        "torch_dtype": dtype,
        "quantization_config": quantization_config,
        "local_files_only": local_files_only,
    }
    if device_map and device_map != "none":
        kwargs["device_map"] = device_map
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    text_encoder.save_pretrained(output_dir, safe_serialization=True)


def infer_precision(filename: str) -> Precision:
    if "fp4" in filename or "nvfp4" in filename:
        return "nvfp4"
    if "int4" in filename:
        return "int4"
    raise ValueError(f"Cannot infer precision from checkpoint filename {filename!r}; pass --precision.")


def infer_rank(filename: str) -> int:
    match = re.search(r"_r([0-9]+)", filename)
    if match is None:
        raise ValueError(f"Cannot infer rank from checkpoint filename {filename!r}; pass --rank.")
    return int(match.group(1))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
