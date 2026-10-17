from __future__ import annotations

from dataclasses import dataclass

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
        qcodes = fp_quantize(qweight.view(weight.shape[0], weight.shape[1]), codebook=codebook)
        qweight = codebook[qcodes.long()].view(weight.shape[0], groups, group_size)
        return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)
    qweight = qweight.round_().clamp_(-8, max_q)
    return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)


def _uses_nvfp4_split_scales(spec: DiffusionQuantSpec) -> bool:
    """Return whether a spec should export DeepCompressor-style NVFP4 scales."""

    return (
        spec.precision == "fp4" and spec.group_size == 16 and tuple(spec.weight_scale_dtypes) == (None, "sfp8_e4m3_nan")
    )


@dataclass(frozen=True)
class ResidualQuantGrid:
    """Exported scales that define a residual weight's 4-bit grid, and the layout they are packed in.

    Codes are ``weight / scale / subscale`` rounded to the INT4 range or the FP4
    E2M1 codebook, and decode as ``value(code) * subscale * scale``.

    Args:
        scale: Effective per-group scale ``(oc, 1, groups, 1)``, or for NVFP4
            split scales the outer scale ``(oc|1, 1, 1, 1)``.
        subscale: FP8-rounded NVFP4 micro scale ``(oc, 1, groups, 1)``, or
            ``None`` when ``scale`` is the effective scale.
        float_point: Whether codes index the FP4 E2M1 codebook instead of INT4.
        weight_scale_layout: Checkpoint scale layout name.
        nunchaku_packed: Whether the target is packed in the Nunchaku kernel
            layout rather than the logical layout.
    """

    scale: torch.Tensor
    subscale: torch.Tensor | None
    float_point: bool
    weight_scale_layout: str
    nunchaku_packed: bool

    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """Round a full-precision residual weight to integer codes on this grid."""

        return quantize_to_codes(weight, self.scale, self.subscale, float_point=self.float_point)

    def column_divisors(self, out_features: int, in_features: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return float32 ``(oc, ic)`` outer and micro divisors for column-wise rounding."""

        outer, _ = _expand_scale(self.scale, out_features)
        outer = outer.float().view(out_features, -1)
        outer = outer.repeat_interleave(in_features // outer.shape[1], dim=1)
        if self.subscale is None:
            return outer, None
        inner = self.subscale.float().view(out_features, -1)
        return outer, inner.repeat_interleave(in_features // inner.shape[1], dim=1)

    def dequantize(self, codes: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Decode integer codes back to weights on this grid."""

        outer, inner = self.column_divisors(*codes.shape)
        values = decode_codes(codes, float_point=self.float_point)
        if inner is not None:
            values = values * inner
        return (values * outer).to(dtype=dtype)


def residual_quant_grid(
    weight: torch.Tensor,
    scale: torch.Tensor,
    spec: DiffusionQuantSpec,
    target: QuantTarget,
    *,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
) -> ResidualQuantGrid:
    """Choose the export layout for a target and derive the scales its codes are rounded against.

    Args:
        weight: Full-precision residual weight; only its shape is read.
        scale: Effective per-group scale from :func:`weight_scales`.
        spec: Quantization settings.
        target: Target being exported.
        low_rank: Low-rank branch, which constrains Nunchaku packing.
    """

    can_pack_nunchaku = uses_nunchaku_packed_layout(weight, spec, low_rank)
    layout = target_weight_layout(target.quant)
    if isinstance(layout, NunchakuSvdqLayout) and not can_pack_nunchaku:
        raise RuntimeError(
            f"Target {target.export_name!r} declares NunchakuSvdqLayout but cannot be packed in Nunchaku ABI layout"
        )
    nunchaku_packed = can_pack_nunchaku and not isinstance(layout, NaiveSvdqLayout)
    outer_scale_rows = nunchaku_nvfp4_outer_scale_rows(target, weight)
    if _uses_nvfp4_split_scales(spec):
        # The logical layout stores one outer scale per output row.
        rows = outer_scale_rows if nunchaku_packed else tuple(1 for _ in range(weight.shape[0]))
        scale0, scale1 = nvfp4_scale_leaves(weight, scale, outer_scale_rows=rows)
        return ResidualQuantGrid(scale0, scale1, True, "nvfp4_deepcompressor", nunchaku_packed)
    return ResidualQuantGrid(scale, None, spec.precision == "fp4", "effective", nunchaku_packed)


def quantize_to_codes(
    weight: torch.Tensor, scale: torch.Tensor, subscale: torch.Tensor | None, *, float_point: bool
) -> torch.Tensor:
    """Round a weight to int32 4-bit codes: INT4 values, or FP4 E2M1 code indices."""

    oc, ic = weight.shape
    scale, groups = _expand_scale(scale, oc)
    # A scale that underflowed to zero (an FP8 micro scale, or an all-zero group) decodes to zero
    # whatever the code, so clamping only keeps the division finite and the codes well defined.
    scaled = weight.float().view(oc, groups, ic // groups).div(scale.float().view(oc, groups, 1).clamp_min(1e-12))
    if subscale is not None:
        subgroups = subscale.shape[2]
        scaled = scaled.view(oc, subgroups, ic // subgroups).div(
            subscale.float().view(oc, subgroups, 1).clamp_min(1e-12)
        )
    if float_point:
        return fp_quantize(scaled.view(oc, ic)).to(torch.int32)
    return scaled.round_().clamp_(-8, 7).to(torch.int32).view(oc, ic)


def decode_codes(codes: torch.Tensor, *, float_point: bool) -> torch.Tensor:
    """Map 4-bit codes to their float32 grid values before scaling."""

    if float_point:
        return fp4_e2m1_codebook(device=codes.device, dtype=torch.float32)[codes.long()]
    return codes.float()


def _expand_scale(scale: torch.Tensor, out_features: int) -> tuple[torch.Tensor, int]:
    """Broadcast a per-tensor scale to one scale per output row; return it with its group count."""

    if scale.numel() == 1:
        return scale.view(1).expand(out_features).reshape(out_features, 1, 1, 1), 1
    return scale, scale.shape[2]


def pack_projector_state(
    codes: torch.Tensor,
    grid: ResidualQuantGrid,
    *,
    dtype: torch.dtype,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
    shift: torch.Tensor | None,
) -> tuple[dict[str, torch.Tensor], str, str]:
    """Pack residual codes and their scales for export.

    Args:
        codes: Integer codes from :meth:`ResidualQuantGrid.quantize` or GPTQ.
        grid: Grid the codes were rounded against.
        dtype: Weight dtype used for exported scales and low-rank tensors.
        smooth: Smoothing factors.
        bias: Optional bias.
        low_rank: Optional low-rank branch.
        shift: Optional activation shift folded into the bias.

    Returns:
        State dict, weight scale layout name, and runtime tensor layout name.
    """

    if grid.nunchaku_packed:
        state = pack_nunchaku_w4a4_codes(
            codes, grid.scale, smooth, bias, low_rank, dtype=dtype, subscale=grid.subscale, shift=shift
        )
        return state, grid.weight_scale_layout, "nunchaku_packed"
    return _pack_logical_projector_state(codes, grid, smooth, bias, low_rank), grid.weight_scale_layout, "logical"


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
    codes: torch.Tensor,
    grid: ResidualQuantGrid,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    """Pack tensors in the legacy logical layout used by tests and torch-dequant."""

    oc = codes.shape[0]
    if grid.subscale is not None:
        groups = grid.subscale.shape[2]
        scale_state = {
            "wscales": grid.subscale.view(oc, groups).to(dtype=torch.float8_e4m3fn).t().contiguous().cpu(),
            "wcscales": grid.scale.view(oc).contiguous().cpu(),
        }
    else:
        scale_state = {"wscales": grid.scale.view(oc, grid.scale.shape[2]).t().contiguous().cpu()}
    state_dict = {
        "qweight": _pack_nibbles(codes),
        "smooth_factor": smooth.detach().cpu(),
        **scale_state,
    }
    if bias is not None:
        state_dict["bias"] = bias.detach().cpu()
    if low_rank is not None:
        state_dict["proj_down"] = low_rank[0].t().contiguous().cpu()
        state_dict["proj_up"] = low_rank[1].contiguous().cpu()
    return state_dict


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes two per byte, low nibble first, for the logical layout."""

    lo = codes[:, 0::2].bitwise_and(0xF)
    hi = codes[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    return lo.bitwise_or(hi).to(torch.uint8).view(torch.int8).contiguous().cpu()


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
    """Quantize and pack a target following DeepCompressor's Nunchaku W4A4 converter."""

    codes = quantize_to_codes(weight, scale, subscale, float_point=float_point)
    return pack_nunchaku_w4a4_codes(
        codes, scale, smooth, bias, low_rank, dtype=weight.dtype, subscale=subscale, shift=shift
    )


def pack_nunchaku_w4a4_codes(
    codes: torch.Tensor,
    scale: torch.Tensor,
    smooth: torch.Tensor,
    bias: torch.Tensor | None,
    low_rank: tuple[torch.Tensor, torch.Tensor] | None,
    *,
    dtype: torch.dtype,
    subscale: torch.Tensor | None,
    shift: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Pack precomputed 4-bit codes and their scales in the Nunchaku W4A4 kernel layout."""

    oc, ic = codes.shape
    device = codes.device
    scale_for_quant, groups = _expand_scale(scale, oc)
    group_size = ic // groups
    subgroup_size = ic // subscale.shape[2] if subscale is not None else group_size

    packer = NunchakuWeightPacker(bits=4)
    packed_weight = packer.pack_weight(packer.pad_weight(codes)).cpu()
    packed_scale = packer.pack_scale(
        packer.pad_scale(scale_for_quant.to(dtype=dtype), group_size=group_size),
        group_size=group_size if group_size < ic else -1,
    ).cpu()
    packed_subscale = (
        packer.pack_scale(
            packer.pad_scale(subscale.to(dtype=dtype), group_size=subgroup_size),
            group_size=subgroup_size if subgroup_size < ic else -1,
        ).cpu()
        if subscale is not None
        else None
    )
    packed_smooth = packer.pack_scale(
        packer.pad_scale(smooth.view(-1, 1).to(dtype=dtype), group_size=-1), group_size=-1
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
            packer.pad_scale(bias.view(-1, 1).to(dtype=dtype), group_size=-1), group_size=-1
        ).cpu()
    if low_rank is not None:
        # MPS has no float64; keep the conservative fp64 bias fold elsewhere.
        fold_dtype = torch.float32 if device.type == "mps" else torch.float64
        proj_down = low_rank[0]
        proj_down_for_bias = proj_down.to(dtype=fold_dtype)
        if smooth is not None:
            proj_down_for_bias = proj_down_for_bias.div(smooth.to(dtype=fold_dtype).view(1, -1))
            proj_down = proj_down_for_bias.to(dtype=dtype)
        if shift is not None:
            shift_vector = _expand_shift_for_nunchaku(shift, ic).to(device=device, dtype=fold_dtype)
            bias_base = (
                torch.zeros(oc, dtype=fold_dtype, device=device)
                if bias is None
                else bias.to(device=device, dtype=fold_dtype)
            )
            correction = low_rank[1].to(dtype=fold_dtype) @ proj_down_for_bias @ shift_vector.view(-1, 1)
            bias = (bias_base + correction.view(-1)).to(dtype=dtype)
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
