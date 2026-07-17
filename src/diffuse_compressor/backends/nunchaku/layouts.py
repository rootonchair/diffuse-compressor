from __future__ import annotations

import torch

import torch.nn as nn

from ...config import (
    AdaNormAwqW4A16Layout,
    AwqW4A16Layout,
    DiffusionQuantSpec,
    NaiveSvdqLayout,
    NunchakuSvdqLayout,
    target_weight_layout,
)
from ...patches import ShiftedConv2d, ShiftedLinear
from ...targets import QuantTarget
from .packing import NunchakuWeightPacker, fp4_e2m1_codebook, fp_quantize


def weight_scales(weight: torch.Tensor, group_size: int, float_point: bool) -> torch.Tensor:
    """Compute per-output, per-group residual weight scales."""

    oc, ic = weight.shape
    if ic % group_size != 0:
        raise ValueError(f"Input features ({ic}) must be divisible by group_size ({group_size}) for Nunchaku export")
    groups = ic // group_size
    max_q = 6 if float_point else 7
    scale = weight.float().view(oc, groups, group_size).abs().amax(dim=2).clamp_min(1e-6) / max_q
    return scale.to(dtype=weight.dtype).view(oc, 1, groups, 1)


def linear_output(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Compute a linear output from flattened inputs."""

    output = inputs @ weight.t()
    if bias is not None:
        output = output + bias.view(1, -1)
    return output


def fake_quantize_weight(weight: torch.Tensor, scale: torch.Tensor, float_point: bool) -> torch.Tensor:
    """Quantize and dequantize a residual weight matrix for scoring."""

    groups = scale.shape[2]
    group_size = weight.shape[1] // groups
    max_q = 6 if float_point else 7
    qweight = weight.float().view(weight.shape[0], groups, group_size) / scale.float().view(weight.shape[0], groups, 1)
    if float_point:
        codebook = fp4_e2m1_codebook(device=weight.device, dtype=torch.float32)
        qcodes = fp_quantize(qweight.view(weight.shape[0], weight.shape[1]))
        qweight = codebook[qcodes.long()].view(weight.shape[0], groups, group_size)
        return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)
    qweight = qweight.round_().clamp_(-8, max_q)
    return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)


def _uses_nvfp4_split_scales(spec: DiffusionQuantSpec) -> bool:
    """Return whether a spec should export DeepCompressor-style NVFP4 scales."""

    return (
        spec.precision == "fp4" and spec.group_size == 16 and tuple(spec.weight_scale_dtypes) == (None, "sfp8_e4m3_nan")
    )


def pack_projector_state(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: DiffusionQuantSpec,
    target: QuantTarget,
    *,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
    shift: torch.Tensor | None,
    outer_scale_rows: tuple[int, ...] | None,
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Pack residual projector tensors for export."""

    can_pack_nunchaku = uses_nunchaku_packed_layout(weight, spec, low_rank)
    layout = target_weight_layout(target.quant)
    if isinstance(layout, NunchakuSvdqLayout) and not can_pack_nunchaku:
        raise RuntimeError(
            f"Target {target.export_name!r} declares NunchakuSvdqLayout but cannot be packed in Nunchaku ABI layout"
        )
    if can_pack_nunchaku and not isinstance(layout, NaiveSvdqLayout):
        state, weight_scale_layout = _pack_nunchaku_projector_state(
            weight, scale, spec, smooth, bias, low_rank, shift=shift, outer_scale_rows=outer_scale_rows
        )
        return state, weight_scale_layout, "nunchaku_packed"
    state, weight_scale_layout = _pack_logical_projector_state(weight, scale, spec, smooth, bias, low_rank)
    return state, weight_scale_layout, "logical"


def uses_nunchaku_packed_layout(
    weight: torch.Tensor, spec: DiffusionQuantSpec, low_rank: tuple[torch.Tensor, torch.Tensor] | None
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
        scale0, scale1 = nvfp4_scale_leaves(weight, scale, outer_scale_rows=outer_scale_rows)
        state_dict = pack_nunchaku_w4a4_state(
            weight, scale0, smooth, bias, low_rank, float_point=True, subscale=scale1, shift=shift
        )
        return state_dict, "nvfp4_deepcompressor"
    state_dict = pack_nunchaku_w4a4_state(
        weight, scale, smooth, bias, low_rank, float_point=spec.precision == "fp4", subscale=None, shift=shift
    )
    return state_dict, "effective"


def _pack_linear_weight(
    weight: torch.Tensor, scale: torch.Tensor, float_point: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack residual weights into Nunchaku Lite INT4 layout."""

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


def pack_nunchaku_w4a4_state(
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
        packer.pad_scale(smooth.view(-1, 1).to(dtype=weight.dtype), group_size=-1), group_size=-1
    ).cpu()
    state_dict = {
        "qweight": packed_weight,
        "wscales": packed_subscale if packed_subscale is not None else packed_scale,
        "smooth_factor": packed_smooth,
    }
    if packed_subscale is not None:
        if scale.numel() == 1:
            state_dict["wtscale"] = packed_scale.view(-1)[0].view(1)
            state_dict["wcscales"] = torch.ones(oc, dtype=packed_scale.dtype, device=packed_scale.device)
        else:
            state_dict["wcscales"] = packed_scale
    if bias is not None:
        state_dict["bias"] = packer.pack_scale(
            packer.pad_scale(bias.view(-1, 1).to(dtype=weight.dtype), group_size=-1), group_size=-1
        ).cpu()
    if low_rank is not None:
        proj_down = low_rank[0]
        proj_down_for_bias = proj_down.to(dtype=torch.float64)
        if smooth is not None:
            proj_down_for_bias = proj_down_for_bias.div(smooth.to(dtype=torch.float64).view(1, -1))
            proj_down = proj_down_for_bias.to(dtype=weight.dtype)
        if shift is not None:
            shift_vector = _expand_shift_for_nunchaku(shift, ic).to(device=weight.device, dtype=torch.float64)
            bias_base = (
                torch.zeros(oc, dtype=torch.float64, device=weight.device)
                if bias is None
                else bias.to(device=weight.device, dtype=torch.float64)
            )
            correction = low_rank[1].to(dtype=torch.float64) @ proj_down_for_bias @ shift_vector.view(-1, 1)
            bias = (bias_base + correction.view(-1)).to(dtype=weight.dtype)
            state_dict["bias"] = packer.pack_scale(
                packer.pad_scale(bias.view(-1, 1), group_size=-1), group_size=-1
            ).cpu()
        state_dict["proj_down"] = packer.pack_lowrank_weight(proj_down, down=True).cpu()
        state_dict["proj_up"] = packer.pack_lowrank_weight(low_rank[1], down=False).cpu()
    return state_dict


def nunchaku_target_shift(target: QuantTarget) -> torch.Tensor | None:
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


def nunchaku_nvfp4_outer_scale_rows(target: QuantTarget, weight: torch.Tensor) -> tuple[int, ...] | None:
    """Return fused source row chunks for DeepCompressor-style NVFP4 outer scales."""

    layout = target_weight_layout(target.quant)
    if isinstance(layout, NunchakuSvdqLayout) and layout.outer_scale_splits:
        rows = tuple(layout.outer_scale_splits)
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
    """Return source-module output rows for grouped Nunchaku layouts."""

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


def nvfp4_scale_leaves(
    weight: torch.Tensor, scale: torch.Tensor, *, outer_scale_rows: tuple[int, ...] | None = None
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


def _pack_nvfp4_linear_weight(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pack FP4 residual weights with DeepCompressor/Nunchaku split scales."""

    oc, ic = weight.shape
    scale0, scale1 = nvfp4_scale_leaves(weight, scale, outer_scale_rows=tuple(1 for _ in range(oc)))
    groups = scale1.shape[2]
    wcscales = scale0.view(oc)
    wscales = scale1.view(oc, groups).to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
    scaled = weight.float().view(oc, groups, 16).div(wcscales.float().view(oc, 1, 1)).div(wscales.view(oc, groups, 1))
    qweight = fp_quantize(scaled.view(oc, ic)).to(torch.int16)
    lo = qweight[:, 0::2].bitwise_and(0xF)
    hi = qweight[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    packed = lo.bitwise_or(hi).to(torch.uint8).view(torch.int8).contiguous()
    return packed.cpu(), {
        "wscales": wscales.to(dtype=torch.float8_e4m3fn).t().contiguous().cpu(),
        "wcscales": wcscales.view(oc).contiguous().cpu(),
    }


def pack_awq_w4a16_target(
    target: QuantTarget, spec: DiffusionQuantSpec, weight: torch.Tensor, bias: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """Pack an AWQ target for Nunchaku Lite W4A16 modules."""

    layout = target_weight_layout(target.quant)
    if not isinstance(layout, (AwqW4A16Layout, AdaNormAwqW4A16Layout)):
        raise ValueError("awq_w4a16 export requires an AWQ W4A16 weight layout")
    if target.kind != "linear":
        raise ValueError("AWQ target export only supports linear targets")
    if len(target.modules) != 1:
        raise ValueError("AWQ target export requires exactly one module per target")
    if spec.precision != "int4":
        raise ValueError("AWQ target export requires precision='int4'")
    if spec.group_size != 64:
        raise ValueError("AWQ target export requires group_size=64")
    if spec.rank != 0:
        raise ValueError("AWQ target export requires rank=0")
    if spec.smooth is not False:
        raise ValueError("AWQ target export requires smooth=False")
    if spec.activation_quant.enabled:
        raise ValueError("AWQ target export requires activation_quant=False")
    if spec.weight_range_calibration.enabled:
        raise ValueError("AWQ target export does not support weight_range_calibration")

    if isinstance(layout, AdaNormAwqW4A16Layout):
        weight, bias = apply_adanorm_awq_w4a16_layout(weight, bias, splits=layout.splits)
    qweight, wscales, wzeros = pack_awq_w4a16_weight(weight, group_size=spec.group_size)
    state_dict = {"qweight": qweight, "wscales": wscales, "wzeros": wzeros}
    if bias is not None:
        state_dict["bias"] = bias.detach().cpu()
    return state_dict


def pack_awq_w4a16_weight(
    weight: torch.Tensor, group_size: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def apply_adanorm_awq_w4a16_layout(
    weight: torch.Tensor, bias: torch.Tensor | None, *, splits: int
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
