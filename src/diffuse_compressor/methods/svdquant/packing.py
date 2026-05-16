from __future__ import annotations

from typing import Sequence

import torch


def ceil_divide(x: int, divisor: int) -> int:
    return (x + divisor - 1) // divisor


def pad(
    tensor: torch.Tensor | None,
    divisor: int | Sequence[int],
    dim: int | Sequence[int],
    fill_value: float | int = 0,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if isinstance(divisor, int) and divisor <= 1:
        return tensor
    if not isinstance(divisor, int) and all(value <= 1 for value in divisor):
        return tensor
    shape = list(tensor.shape)
    if isinstance(dim, int):
        if not isinstance(divisor, int):
            raise TypeError("divisor must be an int when dim is an int")
        shape[dim] = ceil_divide(shape[dim], divisor) * divisor
    else:
        divisors = [divisor] * len(dim) if isinstance(divisor, int) else list(divisor)
        for axis, axis_divisor in zip(dim, divisors, strict=True):
            shape[axis] = ceil_divide(shape[axis], axis_divisor) * axis_divisor
    result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
    result[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
    return result


def fp_quantize(x: torch.Tensor, codebook: torch.Tensor | None = None) -> torch.Tensor:
    if codebook is None:
        codebook = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=x.dtype,
            device=x.device,
        )
    return (x.unsqueeze(-1) - codebook.unsqueeze(0)).abs().argmin(dim=-1)


class MmaWeightPackerBase:
    def __init__(self, bits: int, warp_n: int, comp_n: int | None = None, comp_k: int | None = None):
        self.bits = bits
        self.comp_n = comp_n if comp_n is not None else 16
        self.comp_k = comp_k if comp_k is not None else 256 // self.bits
        self.insn_n = 8
        self.insn_k = self.comp_k
        self.num_lanes = 32
        self.num_k_lanes = 4
        self.num_n_lanes = 8
        self.warp_n = warp_n
        self.reg_k = 32 // self.bits
        self.reg_n = 1
        self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
        self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
        self.mem_k = self.comp_k
        self.mem_n = warp_n
        self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
        self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)


class NunchakuWeightPacker(MmaWeightPackerBase):
    def __init__(self, bits: int, warp_n: int = 128):
        super().__init__(bits=bits, warp_n=warp_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.dtype != torch.int32:
            raise TypeError(f"quantized weight must be torch.int32, got {weight.dtype}")
        n, k = weight.shape
        weight = weight.reshape(
            n // self.mem_n,
            self.num_n_packs,
            self.n_pack_size,
            self.num_n_lanes,
            self.reg_n,
            k // self.mem_k,
            self.num_k_packs,
            self.k_pack_size,
            self.num_k_lanes,
            self.reg_k,
        )
        weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
        if self.bits == 4:
            weight = weight.bitwise_and_(0xF)
            shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
            weight = weight.bitwise_left_shift_(shift).sum(dim=-1, dtype=torch.int32)
        else:
            raise NotImplementedError("Only 4-bit Nunchaku packing is implemented")
        return weight.view(dtype=torch.int8).view(n, -1)

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if self.check_if_micro_scale(group_size=group_size):
            return self.pack_micro_scale(scale, group_size=group_size)
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
        num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
        num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
        scale = scale.reshape(n // self.warp_n, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1)
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1) if group_size == -1 else scale.view(-1, n)

    def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if group_size != 16:
            raise ValueError("micro scale packing only supports group_size=16")
        scale = scale.to(dtype=torch.float8_e4m3fn)
        n = scale.shape[0]
        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 32
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        scale = scale.view(n // self.warp_n, num_s_packs, s_pack_size, 4, 8, -1, self.insn_k // group_size)
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, n)

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        reg_n, reg_k = 1, 2
        pack_n = self.n_pack_size * self.num_n_lanes * reg_n
        pack_k = self.k_pack_size * self.num_k_lanes * reg_k
        weight = pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
        assert weight is not None
        if down:
            r, c = weight.shape
            r_packs, c_packs = r // pack_n, c // pack_k
            weight = weight.view(r_packs, pack_n, c_packs, pack_k).permute(2, 0, 1, 3)
        else:
            c, r = weight.shape
            c_packs, r_packs = c // pack_n, r // pack_k
            weight = weight.view(c_packs, pack_n, r_packs, pack_k).permute(0, 2, 1, 3)
        weight = weight.reshape(
            c_packs, r_packs, self.n_pack_size, self.num_n_lanes, reg_n, self.k_pack_size, self.num_k_lanes, reg_k
        )
        weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
        return weight.view(c, r)

    def check_if_micro_scale(self, group_size: int) -> bool:
        return self.insn_k == group_size * 4

    def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        result = pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1))
        assert result is not None
        return result

    def pad_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if group_size > 0 and scale.numel() > scale.shape[0]:
            scale = scale.view(scale.shape[0], 1, -1, 1)
            if self.check_if_micro_scale(group_size=group_size):
                result = pad(scale, divisor=(self.warp_n, self.insn_k // group_size), dim=(0, 2), fill_value=1)
            else:
                result = pad(scale, divisor=(self.warp_n, self.num_k_unrolls), dim=(0, 2), fill_value=1)
        else:
            result = pad(scale, divisor=self.warp_n, dim=0, fill_value=1)
        assert result is not None
        return result


def convert_to_nunchaku_w4x4y16_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
    float_point: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    if weight.ndim != 2:
        raise ValueError("weight tensor must be 2D")
    device, dtype = weight.device, weight.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        weight = weight.to(torch.bfloat16)
        dtype = weight.dtype
    oc, ic = weight.shape
    if scale.numel() == 1:
        scale = scale.view(-1).expand(oc).reshape(oc, 1, 1, 1)
    if scale.ndim != 4:
        raise ValueError("scale tensor must be 4D")
    ng, group_size = scale.shape[2], ic // scale.shape[2]
    qweight = weight.to(torch.float32).view(oc, 1, ng, group_size)
    qweight = qweight.div_(scale.to(dtype=torch.float32, device=device)).view(oc, ic)
    if float_point:
        qweight = fp_quantize(qweight).to(torch.int32)
    else:
        qweight = qweight.round_().clamp_(-8, 7).to(torch.int32)

    bias = torch.zeros([oc, 1], dtype=dtype, device=device) if bias is None else bias.view(-1, 1).to(dtype=dtype)
    smooth = torch.ones([ic, 1], dtype=dtype, device=device) if smooth is None else smooth.view(-1, 1).to(dtype=dtype)

    packer = NunchakuWeightPacker(bits=4)
    qweight = packer.pack_weight(packer.pad_weight(qweight))
    packed_scale = packer.pack_scale(packer.pad_scale(scale.to(dtype=dtype), group_size=group_size), group_size=group_size)
    packed_bias = packer.pack_scale(packer.pad_scale(bias, group_size=-1), group_size=-1)
    packed_smooth = packer.pack_scale(packer.pad_scale(smooth, group_size=-1), group_size=-1)
    packed_lora = None
    if lora is not None:
        packed_lora = (
            packer.pack_lowrank_weight(lora[0].to(dtype=dtype), down=True),
            packer.pack_lowrank_weight(lora[1].to(dtype=dtype), down=False),
        )
    return qweight.cpu(), packed_scale.cpu(), packed_bias.cpu(), packed_smooth.cpu(), packed_lora
