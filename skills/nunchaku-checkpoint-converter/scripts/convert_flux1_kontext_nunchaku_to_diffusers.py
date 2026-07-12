"""Convert FLUX.1 Kontext Dev Nunchaku checkpoints to Diffusers layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Precision = Literal["int4", "nvfp4"]

BASE_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
NUNCHAKU_REPO = "nunchaku-ai/nunchaku-flux.1-kontext-dev"
TRANSFORMER_FILE = "diffusion_pytorch_model.safetensors"

DROP_SUFFIXES = {"smooth_orig", "smooth_factor_orig"}
SUFFIX_RENAMES = {
    "lora_down": "proj_down",
    "lora_up": "proj_up",
    "smooth": "smooth_factor",
}
INT4_ACTIVATION_SHIFT = 0.171875


@dataclass(frozen=True)
class FluxKontextConversionSpec:
    num_layers: int
    num_single_layers: int
    precision: Precision
    rank: int

    @property
    def group_size(self) -> int:
        return 16 if self.precision == "nvfp4" else 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a FLUX.1 Kontext Dev Nunchaku checkpoint into a Diffusers Nunchaku Lite pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--nunchaku-repo", default=NUNCHAKU_REPO)
    parser.add_argument("--nunchaku-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--skip-text-encoder-2", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
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

    snapshot = snapshot_base_components(
        args.base_model,
        output_dir=output_dir,
        include_text_encoder_2=args.skip_text_encoder_2,
        local_files_only=args.local_files_only,
    )
    transformer_config = load_json(snapshot / "transformer" / "config.json")
    spec = FluxKontextConversionSpec(
        num_layers=int(transformer_config["num_layers"]),
        num_single_layers=int(transformer_config["num_single_layers"]),
        precision=precision,
        rank=rank,
    )

    convert_transformer_checkpoint(checkpoint, output_dir / "transformer" / TRANSFORMER_FILE, spec=spec)
    write_transformer_config(output_dir / "transformer" / "config.json", transformer_config, spec=spec)
    write_model_index(output_dir / "model_index.json")

    if not args.skip_text_encoder_2:
        quantize_text_encoder_2(
            args.base_model,
            output_dir=output_dir / "text_encoder_2",
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
    model_id: str, *, output_dir: Path, include_text_encoder_2: bool, local_files_only: bool
) -> Path:
    from huggingface_hub import snapshot_download

    allow_patterns = [
        "model_index.json",
        "scheduler/*",
        "tokenizer/*",
        "tokenizer_2/*",
        "vae/*",
        "text_encoder/*",
        "transformer/config.json",
    ]
    if include_text_encoder_2:
        allow_patterns.append("text_encoder_2/*")
    local_model = Path(model_id)
    snapshot = (
        local_model
        if local_model.exists()
        else Path(snapshot_download(model_id, allow_patterns=allow_patterns, local_files_only=local_files_only))
    )
    for name in ("scheduler", "tokenizer", "tokenizer_2", "vae", "text_encoder"):
        source = snapshot / name
        if source.exists():
            copy_tree(source, output_dir / name)
    transformer_destination = output_dir / "transformer"
    transformer_destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot / "transformer" / "config.json", transformer_destination / "config.json")
    if include_text_encoder_2 and (snapshot / "text_encoder_2").exists():
        copy_tree(snapshot / "text_encoder_2", output_dir / "text_encoder_2")
    if (snapshot / "model_index.json").exists():
        shutil.copy2(snapshot / "model_index.json", output_dir / "model_index.json")
    return snapshot


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def convert_transformer_checkpoint(source: Path, destination: Path, *, spec: FluxKontextConversionSpec) -> None:
    import safetensors
    import safetensors.torch
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[str, torch.Tensor] = {}
    with safetensors.safe_open(source, framework="pt", device="cpu") as handle:
        source_keys = set(handle.keys())
        for source_prefix, targets in fused_qkv_splits(spec).items():
            if f"{source_prefix}.qweight" in source_keys:
                split_fused_qkv(handle, converted, source_prefix=source_prefix, target_prefixes=targets, spec=spec)

        rename_map = simple_svdq_rename_map(spec)
        norm_map = attention_norm_rename_map(spec)
        fused_prefixes = tuple(f"{prefix}." for prefix in fused_qkv_splits(spec))
        for key in handle.keys():
            if key.startswith(fused_prefixes):
                continue
            prefix, suffix = split_key_suffix(key)
            if suffix in DROP_SUFFIXES:
                continue
            target_prefix = rename_map.get(prefix)
            if target_prefix is not None:
                target_suffix = SUFFIX_RENAMES.get(suffix, suffix)
                if target_prefix.endswith(".proj_out.linears.0") and target_suffix == "bias":
                    continue
                converted[f"{target_prefix}.{target_suffix}"] = handle.get_tensor(key).contiguous()
            elif prefix in norm_map:
                converted[f"{norm_map[prefix]}.{suffix}"] = handle.get_tensor(key).contiguous()
            else:
                converted[key] = handle.get_tensor(key).contiguous()

    for target in svdq_targets(spec):
        if has_target_state(converted, target):
            ensure_nvfp4_scales(converted, target, precision=spec.precision)
    apply_int4_signed_unfused_bias_compensation(converted, spec=spec)
    merge_int4_single_block_proj_out_targets(converted, spec=spec)
    pad_int4_low_rank_tensors(converted, spec=spec)
    for target, splits in awq_targets_with_splits(spec):
        undo_adanorm_awq_layout(converted, target, splits=splits)

    expected_targets = set(svdq_targets(spec)) | {target for target, _ in awq_targets_with_splits(spec)}
    missing = [target for target in sorted(expected_targets) if not has_target_state(converted, target)]
    if missing:
        raise RuntimeError(f"Converted checkpoint is missing {len(missing)} expected targets, first: {missing[:10]}")

    safetensors.torch.save_file(converted, destination)


def split_fused_qkv(handle, output: dict, *, source_prefix: str, target_prefixes: tuple[str, str, str], spec) -> None:
    from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker
    import torch

    qweight = handle.get_tensor(f"{source_prefix}.qweight")
    if qweight.shape[0] % 3 != 0:
        raise RuntimeError(f"{source_prefix}.qweight rows are not divisible by 3: {tuple(qweight.shape)}")
    rows = qweight.shape[0] // 3
    columns = qweight.shape[1] * 2
    packer = NunchakuWeightPacker(bits=4)
    unpacked = unpack_fused_qkv_tensors(
        handle,
        source_prefix=source_prefix,
        rows=qweight.shape[0],
        columns=columns,
        group_size=spec.group_size,
    )
    for index, target_prefix in enumerate(target_prefixes):
        row_slice = slice(index * rows, (index + 1) * rows)
        for source_key in handle.keys():
            if not source_key.startswith(f"{source_prefix}."):
                continue
            suffix = source_key[len(source_prefix) + 1 :]
            if suffix in DROP_SUFFIXES:
                continue
            target_suffix = SUFFIX_RENAMES.get(suffix, suffix)
            tensor = handle.get_tensor(source_key)
            output[f"{target_prefix}.{target_suffix}"] = split_or_duplicate_qkv_tensor(
                tensor,
                tensor_name=target_suffix,
                row_slice=row_slice,
                packer=packer,
                unpacked=unpacked,
                group_size=spec.group_size,
            )
        if spec.precision == "nvfp4" and f"{target_prefix}.wtscale" not in output:
            dtype = output.get(f"{target_prefix}.wcscales", torch.ones(1, dtype=torch.bfloat16)).dtype
            output[f"{target_prefix}.wtscale"] = torch.ones(1, dtype=dtype)


def unpack_fused_qkv_tensors(handle, *, source_prefix: str, rows: int, columns: int, group_size: int) -> dict:
    from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    unpacked = {
        "qweight": packer.unpack_weight(handle.get_tensor(f"{source_prefix}.qweight"), rows=rows, columns=columns)
    }
    if f"{source_prefix}.wscales" in handle.keys():
        wscales = handle.get_tensor(f"{source_prefix}.wscales")
        unpacked["wscales"] = packer.unpack_scale(
            wscales, rows=rows, groups=wscales.shape[0], group_size=group_size
        )
    for source_suffix, target_suffix in (("bias", "bias"), ("wcscales", "wcscales")):
        key = f"{source_prefix}.{source_suffix}"
        if key in handle.keys():
            unpacked[target_suffix] = packer.unpack_scale(handle.get_tensor(key), rows=rows, groups=1, group_size=-1)
    for source_suffix, target_suffix in (("lora_up", "proj_up"), ("proj_up", "proj_up")):
        key = f"{source_prefix}.{source_suffix}"
        if key in handle.keys():
            proj_up = handle.get_tensor(key)
            unpacked[target_suffix] = packer.unpack_lowrank_weight(
                proj_up, down=False, rows=rows, columns=proj_up.shape[1]
            )
            break
    return unpacked


def split_or_duplicate_qkv_tensor(tensor, *, tensor_name: str, row_slice: slice, packer, unpacked: dict, group_size: int):
    if tensor_name == "qweight":
        return packer.pack_weight(packer.pad_weight(unpacked[tensor_name][row_slice])).contiguous()
    if tensor_name == "wscales":
        return packer.pack_scale(
            packer.pad_scale(unpacked[tensor_name][row_slice], group_size=group_size), group_size=group_size
        ).contiguous()
    if tensor_name in {"bias", "wcscales"}:
        logical = unpacked[tensor_name][row_slice]
        return packer.pack_scale(packer.pad_scale(logical.view(-1, 1), group_size=-1), group_size=-1).contiguous()
    if tensor_name == "proj_up":
        return packer.pack_lowrank_weight(unpacked[tensor_name][row_slice], down=False).contiguous()
    if tensor_name in {"smooth_factor", "proj_down", "wtscale"}:
        return tensor.clone().contiguous()
    raise RuntimeError(f"Unsupported fused QKV tensor suffix: {tensor_name!r}")


def apply_int4_signed_unfused_bias_compensation(state: dict, *, spec: FluxKontextConversionSpec) -> None:
    if spec.precision != "int4":
        return
    for target in int4_shifted_down_projection_targets(spec):
        if has_target_state(state, target):
            state[f"{target}.bias"] = compensated_signed_unfused_bias(state, target, group_size=spec.group_size)


def compensated_signed_unfused_bias(state: dict, target: str, *, group_size: int):
    import torch
    from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker

    prefix = f"{target}."
    required = ("qweight", "wscales", "smooth_factor", "bias")
    missing = [suffix for suffix in required if f"{prefix}{suffix}" not in state]
    if missing:
        raise RuntimeError(f"{target} is missing tensors needed for signed int4 bias compensation: {missing}")

    qweight = state[f"{prefix}qweight"]
    rows = qweight.shape[0]
    columns = qweight.shape[1] * 2
    if columns % group_size != 0:
        raise RuntimeError(f"{target} input columns ({columns}) must be divisible by group_size={group_size}")

    packer = NunchakuWeightPacker(bits=4)
    groups = columns // group_size
    qcodes = packer.unpack_weight(qweight, rows=rows, columns=columns).to(dtype=torch.float64)
    scales = packer.unpack_scale(state[f"{prefix}wscales"], rows=rows, groups=groups, group_size=group_size)
    scales = scales.view(rows, groups).to(dtype=torch.float64)
    smooth = packer.unpack_scale(state[f"{prefix}smooth_factor"], rows=columns, groups=1, group_size=-1)
    smooth = smooth.to(dtype=torch.float64)
    bias = packer.unpack_scale(state[f"{prefix}bias"], rows=rows, groups=1, group_size=-1)
    bias_dtype = bias.dtype

    dequantized = (qcodes.view(rows, groups, group_size) * scales.view(rows, groups, 1)).view(rows, columns)
    row_sum = dequantized.div(smooth.view(1, -1)).sum(dim=1)
    compensated = bias.to(dtype=torch.float64).add(row_sum, alpha=INT4_ACTIVATION_SHIFT)
    return packer.pack_scale(
        packer.pad_scale(compensated.to(dtype=bias_dtype).view(-1, 1), group_size=-1), group_size=-1
    ).contiguous()


def merge_int4_single_block_proj_out_targets(state: dict, *, spec: FluxKontextConversionSpec) -> None:
    if spec.precision != "int4":
        return
    for block in range(spec.num_single_layers):
        target = f"single_transformer_blocks.{block}.proj_out"
        left = f"{target}.linears.0"
        right = f"{target}.linears.1"
        if not has_target_state(state, left) or not has_target_state(state, right):
            continue
        merge_split_svdq_target(state, target=target, left=left, right=right, group_size=spec.group_size)


def merge_split_svdq_target(state: dict, *, target: str, left: str, right: str, group_size: int) -> None:
    import torch
    from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    left_rows, left_columns = _packed_qweight_shape(state, left)
    right_rows, right_columns = _packed_qweight_shape(state, right)
    if left_rows != right_rows:
        raise RuntimeError(f"Cannot merge {left!r} and {right!r} with different output rows.")
    if left_columns % group_size != 0 or right_columns % group_size != 0:
        raise RuntimeError(f"Merged SVDQ target {target!r} columns must be divisible by group_size={group_size}.")

    left_groups = left_columns // group_size
    right_groups = right_columns // group_size
    left_q = packer.unpack_weight(state[f"{left}.qweight"], rows=left_rows, columns=left_columns)
    right_q = packer.unpack_weight(state[f"{right}.qweight"], rows=right_rows, columns=right_columns)
    merged_q = torch.cat([left_q, right_q], dim=1)
    state[f"{target}.qweight"] = packer.pack_weight(packer.pad_weight(merged_q)).contiguous()

    left_scales = packer.unpack_scale(
        state[f"{left}.wscales"], rows=left_rows, groups=left_groups, group_size=group_size
    ).view(left_rows, left_groups)
    right_scales = packer.unpack_scale(
        state[f"{right}.wscales"], rows=right_rows, groups=right_groups, group_size=group_size
    ).view(right_rows, right_groups)
    merged_scales = torch.cat([left_scales, right_scales], dim=1).view(left_rows, 1, left_groups + right_groups, 1)
    state[f"{target}.wscales"] = packer.pack_scale(
        packer.pad_scale(merged_scales, group_size=group_size), group_size=group_size
    ).contiguous()

    left_smooth = _unpack_vector(state, f"{left}.smooth_factor", rows=left_columns, packer=packer)
    right_smooth = _unpack_vector(state, f"{right}.smooth_factor", rows=right_columns, packer=packer)
    merged_smooth = torch.cat([left_smooth, right_smooth], dim=0)
    state[f"{target}.smooth_factor"] = packer.pack_scale(
        packer.pad_scale(merged_smooth.view(-1, 1), group_size=-1), group_size=-1
    ).contiguous()

    if f"{right}.bias" in state:
        state[f"{target}.bias"] = state[f"{right}.bias"].contiguous()
    elif f"{left}.bias" in state:
        state[f"{target}.bias"] = state[f"{left}.bias"].contiguous()

    # proj_down/proj_up are stored in the packed tile-permuted layout, so the
    # block-diagonal/concat merge must happen in logical layout.
    left_rank = state[f"{left}.proj_down"].shape[1]
    right_rank = state[f"{right}.proj_down"].shape[1]
    left_down = packer.unpack_lowrank_weight(
        state[f"{left}.proj_down"], down=True, rows=left_rank, columns=left_columns
    )
    right_down = packer.unpack_lowrank_weight(
        state[f"{right}.proj_down"], down=True, rows=right_rank, columns=right_columns
    )
    merged_down = torch.zeros(
        left_rank + right_rank, left_columns + right_columns, dtype=left_down.dtype, device=left_down.device
    )
    merged_down[:left_rank, :left_columns] = left_down
    merged_down[left_rank:, left_columns:] = right_down
    left_up = packer.unpack_lowrank_weight(
        state[f"{left}.proj_up"], down=False, rows=left_rows, columns=left_rank
    )
    right_up = packer.unpack_lowrank_weight(
        state[f"{right}.proj_up"], down=False, rows=right_rows, columns=right_rank
    )
    merged_up = torch.cat([left_up, right_up], dim=1)
    state[f"{target}.proj_down"] = packer.pack_lowrank_weight(merged_down, down=True).contiguous()
    state[f"{target}.proj_up"] = packer.pack_lowrank_weight(merged_up, down=False).contiguous()

    for prefix in (left, right):
        for suffix in ("qweight", "wscales", "smooth_factor", "bias", "proj_down", "proj_up"):
            state.pop(f"{prefix}.{suffix}", None)


def pad_int4_low_rank_tensors(state: dict, *, spec: FluxKontextConversionSpec) -> None:
    if spec.precision != "int4":
        return
    import torch
    from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker

    packer = NunchakuWeightPacker(bits=4)
    rank = int4_runtime_rank(spec)
    for key in list(state):
        if not key.endswith((".proj_down", ".proj_up")):
            continue
        tensor = state[key]
        if tensor.ndim != 2 or tensor.shape[1] >= rank:
            continue
        # Packed containers are (in_features, rank) for proj_down and
        # (out_features, rank) for proj_up; rank-padding must happen in logical
        # layout because the packed layout interleaves tiles.
        current_rank = tensor.shape[1]
        if key.endswith(".proj_down"):
            columns = tensor.shape[0]
            logical = packer.unpack_lowrank_weight(tensor, down=True, rows=current_rank, columns=columns)
            padded = torch.zeros(rank, columns, dtype=tensor.dtype, device=tensor.device)
            padded[:current_rank] = logical
            state[key] = packer.pack_lowrank_weight(padded, down=True).contiguous()
        else:
            rows = tensor.shape[0]
            logical = packer.unpack_lowrank_weight(tensor, down=False, rows=rows, columns=current_rank)
            padded = torch.zeros(rows, rank, dtype=tensor.dtype, device=tensor.device)
            padded[:, :current_rank] = logical
            state[key] = packer.pack_lowrank_weight(padded, down=False).contiguous()


def _packed_qweight_shape(state: dict, target: str) -> tuple[int, int]:
    qweight = state[f"{target}.qweight"]
    return qweight.shape[0], qweight.shape[1] * 2


def _unpack_vector(state: dict, key: str, *, rows: int, packer):
    return packer.unpack_scale(state[key], rows=rows, groups=1, group_size=-1)


def undo_adanorm_awq_layout(state: dict, target: str, *, splits: int) -> None:
    qweight = state.get(f"{target}.qweight")
    if qweight is None:
        return
    wscales = state.get(f"{target}.wscales")
    wzeros = state.get(f"{target}.wzeros")
    bias = state.get(f"{target}.bias")
    if wscales is None or wzeros is None:
        raise RuntimeError(f"{target} is missing AWQ scales needed to undo AdaNorm layout.")
    state[f"{target}.qweight"] = undo_adanorm_awq_qweight_order(qweight, out_features=wscales.shape[1], splits=splits)
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
    if splits == 6:
        restored[:, 1] -= 1
        restored[:, -2] -= 1
    elif splits == 3:
        restored[:, 1] -= 1
    return restored.transpose(0, 1).reshape_as(bias)


def fused_qkv_splits(spec: FluxKontextConversionSpec) -> dict[str, tuple[str, str, str]]:
    mapping = {}
    for block in range(spec.num_layers):
        mapping[f"transformer_blocks.{block}.qkv_proj"] = (
            f"transformer_blocks.{block}.attn.to_q",
            f"transformer_blocks.{block}.attn.to_k",
            f"transformer_blocks.{block}.attn.to_v",
        )
        mapping[f"transformer_blocks.{block}.qkv_proj_context"] = (
            f"transformer_blocks.{block}.attn.add_q_proj",
            f"transformer_blocks.{block}.attn.add_k_proj",
            f"transformer_blocks.{block}.attn.add_v_proj",
        )
    for block in range(spec.num_single_layers):
        mapping[f"single_transformer_blocks.{block}.qkv_proj"] = (
            f"single_transformer_blocks.{block}.attn.to_q",
            f"single_transformer_blocks.{block}.attn.to_k",
            f"single_transformer_blocks.{block}.attn.to_v",
        )
    return mapping


def simple_svdq_rename_map(spec: FluxKontextConversionSpec) -> dict[str, str]:
    mapping = {}
    for block in range(spec.num_layers):
        prefix = f"transformer_blocks.{block}"
        mapping.update(
            {
                f"{prefix}.out_proj": f"{prefix}.attn.to_out.0",
                f"{prefix}.out_proj_context": f"{prefix}.attn.to_add_out",
                f"{prefix}.mlp_fc1": f"{prefix}.ff.net.0.proj",
                f"{prefix}.mlp_fc2": f"{prefix}.ff.net.2",
                f"{prefix}.mlp_context_fc1": f"{prefix}.ff_context.net.0.proj",
                f"{prefix}.mlp_context_fc2": f"{prefix}.ff_context.net.2",
            }
        )
    for block in range(spec.num_single_layers):
        prefix = f"single_transformer_blocks.{block}"
        mapping.update(
            {
                f"{prefix}.out_proj": f"{prefix}.proj_out.linears.0",
                f"{prefix}.mlp_fc1": f"{prefix}.proj_mlp",
                f"{prefix}.mlp_fc2": f"{prefix}.proj_out.linears.1",
            }
        )
    return mapping


def attention_norm_rename_map(spec: FluxKontextConversionSpec) -> dict[str, str]:
    mapping = {}
    for block in range(spec.num_layers):
        prefix = f"transformer_blocks.{block}"
        mapping.update(
            {
                f"{prefix}.norm_q": f"{prefix}.attn.norm_q",
                f"{prefix}.norm_k": f"{prefix}.attn.norm_k",
                f"{prefix}.norm_added_q": f"{prefix}.attn.norm_added_q",
                f"{prefix}.norm_added_k": f"{prefix}.attn.norm_added_k",
            }
        )
    for block in range(spec.num_single_layers):
        prefix = f"single_transformer_blocks.{block}"
        mapping.update(
            {
                f"{prefix}.norm_q": f"{prefix}.attn.norm_q",
                f"{prefix}.norm_k": f"{prefix}.attn.norm_k",
            }
        )
    return mapping


def svdq_targets(spec: FluxKontextConversionSpec) -> list[str]:
    targets = []
    for block in range(spec.num_layers):
        prefix = f"transformer_blocks.{block}"
        targets.extend(
            [
                f"{prefix}.attn.to_q",
                f"{prefix}.attn.to_k",
                f"{prefix}.attn.to_v",
                f"{prefix}.attn.add_q_proj",
                f"{prefix}.attn.add_k_proj",
                f"{prefix}.attn.add_v_proj",
                f"{prefix}.attn.to_out.0",
                f"{prefix}.attn.to_add_out",
                f"{prefix}.ff.net.0.proj",
                f"{prefix}.ff.net.2",
                f"{prefix}.ff_context.net.0.proj",
                f"{prefix}.ff_context.net.2",
            ]
        )
    for block in range(spec.num_single_layers):
        prefix = f"single_transformer_blocks.{block}"
        if spec.precision == "int4":
            targets.extend(
                [
                    f"{prefix}.attn.to_q",
                    f"{prefix}.attn.to_k",
                    f"{prefix}.attn.to_v",
                    f"{prefix}.proj_out",
                    f"{prefix}.proj_mlp",
                ]
            )
        else:
            targets.extend(
                [
                    f"{prefix}.attn.to_q",
                    f"{prefix}.attn.to_k",
                    f"{prefix}.attn.to_v",
                    f"{prefix}.proj_out.linears.0",
                    f"{prefix}.proj_mlp",
                    f"{prefix}.proj_out.linears.1",
                ]
            )
    return targets


def int4_shifted_down_projection_targets(spec: FluxKontextConversionSpec) -> list[str]:
    targets = []
    for block in range(spec.num_layers):
        prefix = f"transformer_blocks.{block}"
        targets.append(f"{prefix}.ff.net.2")
        targets.append(f"{prefix}.ff_context.net.2")
    for block in range(spec.num_single_layers):
        targets.append(f"single_transformer_blocks.{block}.proj_out.linears.1")
    return targets


def int4_fused_gelu_mlp_patches(spec: FluxKontextConversionSpec) -> list[dict]:
    return []


def int4_runtime_rank(spec: FluxKontextConversionSpec) -> int:
    return max(spec.rank, spec.rank * 2)


def awq_targets_with_splits(spec: FluxKontextConversionSpec) -> list[tuple[str, int]]:
    targets = []
    for block in range(spec.num_layers):
        targets.append((f"transformer_blocks.{block}.norm1.linear", 6))
        targets.append((f"transformer_blocks.{block}.norm1_context.linear", 6))
    for block in range(spec.num_single_layers):
        targets.append((f"single_transformer_blocks.{block}.norm.linear", 3))
    return targets


def ensure_nvfp4_scales(state: dict, target: str, *, precision: Precision) -> None:
    if precision != "nvfp4" or f"{target}.qweight" not in state:
        return
    import torch

    dtype = _scale_dtype_for_target(state, target)
    if f"{target}.wtscale" not in state:
        state[f"{target}.wtscale"] = torch.ones(1, dtype=dtype)
    if f"{target}.wcscales" not in state:
        state[f"{target}.wcscales"] = torch.ones(state[f"{target}.qweight"].shape[0], dtype=dtype)


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


def split_key_suffix(key: str) -> tuple[str, str]:
    prefix, _, suffix = key.rpartition(".")
    return prefix, suffix


def write_transformer_config(path: Path, config: dict, *, spec: FluxKontextConversionSpec) -> None:
    config = dict(config)
    config["quantization_config"] = {
        "quant_method": "nunchaku_lite",
        "compute_dtype": "bfloat16",
        "patches": int4_fused_gelu_mlp_patches(spec)
        if spec.precision == "int4"
        else [
            {
                "type": "split_linear",
                "module": "single_transformer_blocks.*.proj_out",
                "args": {"splits": ["out_features"]},
            },
            *int4_fused_gelu_mlp_patches(spec),
        ],
        "svdq_w4a4": {
            "precision": spec.precision,
            "group_size": spec.group_size,
            "rank": int4_runtime_rank(spec) if spec.precision == "int4" else spec.rank,
            "targets": sorted(svdq_targets(spec)),
        },
        "awq_w4a16": {
            "precision": "int4",
            "group_size": 64,
            "targets": sorted(target for target, _ in awq_targets_with_splits(spec)),
        },
    }
    write_json(path, config)


def write_model_index(path: Path) -> None:
    model_index = load_json(path) if path.exists() else {}
    model_index.update(
        {
            "_class_name": "FluxKontextPipeline",
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "T5EncoderModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "T5TokenizerFast"],
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
    )
    write_json(path, model_index)


def quantize_text_encoder_2(
    model_id: str, *, output_dir: Path, torch_dtype_name: str, device_map: str | None, local_files_only: bool
) -> None:
    import torch
    from transformers import BitsAndBytesConfig, T5EncoderModel

    dtype = getattr(torch, torch_dtype_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
    )
    kwargs = {
        "subfolder": "text_encoder_2",
        "torch_dtype": dtype,
        "quantization_config": quantization_config,
        "local_files_only": local_files_only,
    }
    if device_map and device_map != "none":
        kwargs["device_map"] = device_map
    text_encoder = T5EncoderModel.from_pretrained(model_id, **kwargs)
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
