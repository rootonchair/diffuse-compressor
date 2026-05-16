from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn

from .config import PatchRule
from .targets import _match_pattern


class SplitLinear(nn.Module):
    def __init__(self, linears: Iterable[nn.Linear], in_features_list: Iterable[int]) -> None:
        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.in_features_list = list(in_features_list)
        self.in_features = sum(self.in_features_list)
        self.out_features = self.linears[0].out_features

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> "SplitLinear":
        remaining = linear.in_features - sum(splits)
        if remaining > 0:
            splits = [*splits, remaining]
        splits = [split for split in splits if split > 0]
        if len(splits) < 2:
            raise ValueError("split_linear requires at least two positive input splits")
        linears: list[nn.Linear] = []
        start = 0
        for idx, split in enumerate(splits):
            child = nn.Linear(
                split,
                linear.out_features,
                bias=linear.bias is not None and idx == len(splits) - 1,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
            child.weight.data.copy_(linear.weight[:, start : start + split])
            if child.bias is not None and linear.bias is not None:
                child.bias.data.copy_(linear.bias)
            linears.append(child)
            start += split
        return cls(linears, splits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.split(self.in_features_list, dim=-1)
        return sum(linear(chunk.contiguous()) for linear, chunk in zip(self.linears, chunks, strict=True))


class SplitLinearOutput(nn.Module):
    def __init__(self, linears: Iterable[nn.Linear], out_features_list: Iterable[int]) -> None:
        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.out_features_list = list(out_features_list)
        self.in_features = self.linears[0].in_features
        self.out_features = sum(self.out_features_list)

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> "SplitLinearOutput":
        remaining = linear.out_features - sum(splits)
        if remaining > 0:
            splits = [*splits, remaining]
        splits = [split for split in splits if split > 0]
        if len(splits) < 2:
            raise ValueError("split_linear_output requires at least two positive output splits")
        if sum(splits) != linear.out_features:
            raise ValueError("split_linear_output splits must sum to linear.out_features")
        linears: list[nn.Linear] = []
        start = 0
        for split in splits:
            child = nn.Linear(
                linear.in_features,
                split,
                bias=linear.bias is not None,
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
            child.weight.data.copy_(linear.weight[start : start + split])
            if child.bias is not None and linear.bias is not None:
                child.bias.data.copy_(linear.bias[start : start + split])
            linears.append(child)
            start += split
        return cls(linears, splits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([linear(x) for linear in self.linears], dim=-1)


class ShiftedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, shift: torch.Tensor) -> None:
        super().__init__()
        self.linear = linear
        self.linear.shifted = True
        self.register_buffer("shift", shift.to(device=linear.weight.device, dtype=linear.weight.dtype).flatten())

    @classmethod
    def from_linear(cls, linear: nn.Linear, shift: float | torch.Tensor) -> "ShiftedLinear":
        shift_tensor = torch.as_tensor(shift, device=linear.weight.device, dtype=linear.weight.dtype).flatten()
        if shift_tensor.numel() > 1:
            if linear.in_features % shift_tensor.numel() != 0:
                raise ValueError("shift length must divide linear.in_features")
            shift_tensor = shift_tensor.view(-1, 1).expand(-1, linear.in_features // shift_tensor.numel()).flatten()
        shifted_bias = linear.weight.double() @ shift_tensor.view(-1, 1).double()
        replacement = nn.Linear(
            linear.in_features,
            linear.out_features,
            bias=True,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        replacement.weight.data.copy_(linear.weight)
        if linear.bias is None:
            replacement.bias.data.copy_((-shifted_bias).view(-1).to(linear.weight.dtype))
        else:
            replacement.bias.data.copy_((linear.bias.double() - shifted_bias.view(-1)).to(linear.weight.dtype))
        return cls(replacement, shift_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x + self.shift.view([1] * (x.ndim - 1) + [-1]))


class SplitConv2d(nn.Module):
    def __init__(self, convs: Iterable[nn.Conv2d], in_channels_list: Iterable[int]) -> None:
        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.in_channels_list = list(in_channels_list)
        self.in_channels = sum(self.in_channels_list)
        self.out_channels = self.convs[0].out_channels

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, splits: list[int]) -> "SplitConv2d":
        remaining = conv.in_channels - sum(splits)
        if remaining > 0:
            splits = [*splits, remaining]
        splits = [split for split in splits if split > 0]
        if len(splits) < 2:
            raise ValueError("split_conv requires at least two positive input-channel splits")
        convs: list[nn.Conv2d] = []
        start = 0
        for idx, split in enumerate(splits):
            child = nn.Conv2d(
                split,
                conv.out_channels,
                conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=1,
                bias=conv.bias is not None and idx == len(splits) - 1,
                padding_mode=conv.padding_mode,
                device=conv.weight.device,
                dtype=conv.weight.dtype,
            )
            child.weight.data.copy_(conv.weight[:, start : start + split])
            if child.bias is not None and conv.bias is not None:
                child.bias.data.copy_(conv.bias)
            convs.append(child)
            start += split
        return cls(convs, splits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.split(self.in_channels_list, dim=1)
        return sum(conv(chunk.contiguous()) for conv, chunk in zip(self.convs, chunks, strict=True))


class ShiftedConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, shift: torch.Tensor) -> None:
        super().__init__()
        self.conv = conv
        self.conv.shifted = True
        self.register_buffer("shift", shift.to(device=conv.weight.device, dtype=conv.weight.dtype).flatten())

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, shift: float | torch.Tensor) -> "ShiftedConv2d":
        if conv.padding != (0, 0):
            raise ValueError("shift_conv currently supports zero-padding convolutions only")
        shift_tensor = torch.as_tensor(shift, device=conv.weight.device, dtype=conv.weight.dtype).flatten()
        if shift_tensor.numel() > 1:
            if conv.in_channels % shift_tensor.numel() != 0:
                raise ValueError("shift length must divide conv.in_channels")
            if conv.padding != (0, 0):
                raise ValueError("multi-channel shift_conv only supports zero padding")
            shift_tensor = shift_tensor.view(-1, 1).expand(-1, conv.in_channels // shift_tensor.numel()).flatten()
        replacement = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
            padding_mode=conv.padding_mode,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )
        replacement.weight.data.copy_(conv.weight)
        if shift_tensor.numel() == 1:
            shifted_bias = conv.weight.double().sum(dim=(1, 2, 3)) * shift_tensor.double()
        else:
            shifted_bias = conv.weight.double().sum(dim=(2, 3)) @ shift_tensor.view(-1, 1).double()
            shifted_bias = shifted_bias.view(-1)
        if conv.bias is None:
            replacement.bias.data.copy_((-shifted_bias).to(conv.weight.dtype))
        else:
            replacement.bias.data.copy_((conv.bias.double() - shifted_bias).to(conv.weight.dtype))
        return cls(replacement, shift_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x + self.shift.view(1, -1, 1, 1))


def prepare_model(model: nn.Module, patches: Iterable[PatchRule]) -> nn.Module:
    for rule in patches:
        if rule.type == "split_linear":
            _apply_split_linear(model, rule)
        elif rule.type == "split_linear_output":
            _apply_split_linear_output(model, rule)
        elif rule.type == "shift_linear":
            _apply_shift_linear(model, rule)
        elif rule.type == "split_conv":
            _apply_split_conv(model, rule)
        elif rule.type == "shift_conv":
            _apply_shift_conv(model, rule)
        else:
            raise NotImplementedError(f"Patch type {rule.type!r} is not implemented yet")
    return model


def _apply_split_linear(model: nn.Module, rule: PatchRule) -> None:
    modules = dict(model.named_modules())
    for module_name in _matched_names(rule.module, modules):
        module = modules[module_name]
        if isinstance(module, SplitLinear):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"split_linear expected nn.Linear at {module_name!r}, got {type(module).__name__}")
        splits = _resolve_splits(module, rule.args.get("splits", []))
        _set_module(model, module_name, SplitLinear.from_linear(module, splits))


def _apply_split_linear_output(model: nn.Module, rule: PatchRule) -> None:
    modules = dict(model.named_modules())
    for module_name in _matched_names(rule.module, modules):
        module = modules[module_name]
        if isinstance(module, SplitLinearOutput):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"split_linear_output expected nn.Linear at {module_name!r}, got {type(module).__name__}")
        splits = _resolve_output_splits(module, rule.args.get("splits", []))
        _set_module(model, module_name, SplitLinearOutput.from_linear(module, splits))


def _apply_shift_linear(model: nn.Module, rule: PatchRule) -> None:
    modules = dict(model.named_modules())
    shift = rule.args.get("shift")
    if shift is None:
        raise ValueError("shift_linear requires args['shift']")
    for module_name in _matched_names(rule.module, modules):
        module = modules[module_name]
        if isinstance(module, ShiftedLinear):
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(f"shift_linear expected nn.Linear at {module_name!r}, got {type(module).__name__}")
        _set_module(model, module_name, ShiftedLinear.from_linear(module, shift))


def _apply_split_conv(model: nn.Module, rule: PatchRule) -> None:
    modules = dict(model.named_modules())
    for module_name in _matched_names(rule.module, modules):
        module = modules[module_name]
        if isinstance(module, SplitConv2d):
            continue
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"split_conv expected nn.Conv2d at {module_name!r}, got {type(module).__name__}")
        splits = _resolve_conv_splits(module, rule.args.get("splits", []))
        _set_module(model, module_name, SplitConv2d.from_conv2d(module, splits))


def _apply_shift_conv(model: nn.Module, rule: PatchRule) -> None:
    modules = dict(model.named_modules())
    shift = rule.args.get("shift")
    if shift is None:
        raise ValueError("shift_conv requires args['shift']")
    for module_name in _matched_names(rule.module, modules):
        module = modules[module_name]
        if isinstance(module, ShiftedConv2d):
            continue
        if not isinstance(module, nn.Conv2d):
            raise TypeError(f"shift_conv expected nn.Conv2d at {module_name!r}, got {type(module).__name__}")
        _set_module(model, module_name, ShiftedConv2d.from_conv2d(module, shift))


def _matched_names(pattern: str, modules: dict[str, nn.Module]) -> list[str]:
    matches = _match_pattern(pattern, modules)
    return [matches[capture] for capture in sorted(matches, key=_capture_sort_key)]


def _resolve_splits(module: nn.Linear, split_specs: Iterable[int | str]) -> list[int]:
    splits: list[int] = []
    for spec in split_specs:
        if isinstance(spec, int):
            splits.append(spec)
        elif spec == "in_features":
            splits.append(module.in_features)
        elif spec == "out_features":
            splits.append(module.out_features)
        else:
            raise ValueError(f"Unsupported split spec {spec!r}")
    return splits


def _resolve_output_splits(module: nn.Linear, split_specs: Iterable[int | str]) -> list[int]:
    splits: list[int] = []
    for spec in split_specs:
        if isinstance(spec, int):
            splits.append(spec)
        elif spec == "in_features":
            splits.append(module.in_features)
        elif spec == "out_features":
            splits.append(module.out_features)
        else:
            raise ValueError(f"Unsupported output split spec {spec!r}")
    return splits


def _resolve_conv_splits(module: nn.Conv2d, split_specs: Iterable[int | str]) -> list[int]:
    if module.groups != 1:
        raise ValueError("split_conv currently supports groups=1 only")
    splits: list[int] = []
    for spec in split_specs:
        if isinstance(spec, int):
            splits.append(spec)
        elif spec == "in_channels":
            splits.append(module.in_channels)
        elif spec == "out_channels":
            splits.append(module.out_channels)
        else:
            raise ValueError(f"Unsupported conv split spec {spec!r}")
    return splits


def _capture_sort_key(capture: tuple[str, ...]) -> tuple[tuple[int, int | str], ...]:
    return tuple((0, int(item)) if item.isdigit() else (1, item) for item in capture)


def _set_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)
