from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from .config import PatchRule
from .matching import capture_sort_key, match_pattern


class SplitLinear(nn.Module):
    """Linear replacement that splits input features across child linears.

    Args:
        linears: Child linears that each consume one input slice.
        in_features_list: Input feature counts for each child linear.
    """

    def __init__(self, linears: Iterable[nn.Linear], in_features_list: Iterable[int]) -> None:
        """Initialize a split-input linear module."""

        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.in_features_list = list(in_features_list)
        self.in_features = sum(self.in_features_list)
        self.out_features = self.linears[0].out_features

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> "SplitLinear":
        """Create a split-input replacement from an existing linear layer.

        Args:
            linear: Source linear layer.
            splits: Input feature splits. Any remaining features are appended.

        Returns:
            Replacement preserving the source layer output.
        """

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
        """Apply each child linear to its input slice and sum outputs.

        Args:
            x: Input tensor with the original feature dimension.

        Returns:
            Output tensor equivalent to the original linear layer.
        """

        chunks = x.split(self.in_features_list, dim=-1)
        return sum(linear(chunk.contiguous()) for linear, chunk in zip(self.linears, chunks, strict=True))


class SplitLinearOutput(nn.Module):
    """Linear replacement that splits output features across child linears.

    Args:
        linears: Child linears that each produce one output slice.
        out_features_list: Output feature counts for each child linear.
    """

    def __init__(self, linears: Iterable[nn.Linear], out_features_list: Iterable[int]) -> None:
        """Initialize a split-output linear module."""

        super().__init__()
        self.linears = nn.ModuleList(linears)
        self.out_features_list = list(out_features_list)
        self.in_features = self.linears[0].in_features
        self.out_features = sum(self.out_features_list)

    @classmethod
    def from_linear(cls, linear: nn.Linear, splits: list[int]) -> "SplitLinearOutput":
        """Create a split-output replacement from an existing linear layer.

        Args:
            linear: Source linear layer.
            splits: Output feature splits. Any remaining features are appended.

        Returns:
            Replacement whose concatenated output matches the source layer.
        """

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
        """Concatenate outputs from every child linear.

        Args:
            x: Input tensor for the original linear layer.

        Returns:
            Concatenated output tensor.
        """

        return torch.cat([linear(x) for linear in self.linears], dim=-1)


class ShiftedLinear(nn.Module):
    """Linear wrapper that applies a fixed input shift before the child layer.

    Args:
        linear: Bias-adjusted linear layer.
        shift: Shift vector added to inputs before applying ``linear``.
    """

    def __init__(self, linear: nn.Linear, shift: torch.Tensor) -> None:
        """Initialize a shifted linear wrapper."""

        super().__init__()
        self.linear = linear
        self.linear.shifted = True
        self.register_buffer("shift", shift.to(device=linear.weight.device, dtype=linear.weight.dtype).flatten())

    @classmethod
    def from_linear(cls, linear: nn.Linear, shift: float | torch.Tensor) -> "ShiftedLinear":
        """Create a shifted wrapper while preserving original outputs.

        Args:
            linear: Source linear layer.
            shift: Scalar or feature shift applied at runtime.

        Returns:
            Wrapper with bias adjusted so ``linear(x) == wrapper(x - shift)``.
        """

        shift_tensor = torch.as_tensor(shift, device=linear.weight.device, dtype=linear.weight.dtype).flatten()
        if shift_tensor.numel() == 1:
            shift_tensor = shift_tensor.expand(linear.in_features)
        else:
            if linear.in_features % shift_tensor.numel() != 0:
                raise ValueError("shift length must divide linear.in_features")
            shift_tensor = shift_tensor.view(-1, 1).expand(-1, linear.in_features // shift_tensor.numel()).flatten()
        # DeepCompressor used float64 here for a numerically conservative
        # weight @ shift accumulation. Float32 keeps the one-time bias fold more
        # stable than fp16/bf16 while avoiding large CUDA float64 temporaries.
        weight_f32 = linear.weight.float()
        shift_f32 = shift_tensor.view(-1, 1).float()
        shifted_bias = weight_f32 @ shift_f32
        replacement = nn.Linear(
            linear.in_features, linear.out_features, bias=True, device=linear.weight.device, dtype=linear.weight.dtype
        )
        replacement.weight.data.copy_(linear.weight)
        if linear.bias is None:
            replacement.bias.data.copy_((-shifted_bias).view(-1).to(linear.weight.dtype))
        else:
            replacement.bias.data.copy_((linear.bias.float() - shifted_bias.view(-1)).to(linear.weight.dtype))
        return cls(replacement, shift_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shifted linear transform.

        Args:
            x: Input tensor.

        Returns:
            Shifted linear output.
        """

        return self.linear(x + self.shift.view([1] * (x.ndim - 1) + [-1]))

    @property
    def weight(self) -> torch.nn.Parameter:
        """Return the wrapped linear weight for compatibility."""

        return self.linear.weight

    @property
    def bias(self) -> torch.nn.Parameter | None:
        """Return the wrapped linear bias for compatibility."""

        return self.linear.bias


class SplitConv2d(nn.Module):
    """Conv2d replacement that splits input channels across child convolutions.

    Args:
        convs: Child convolutions that each consume one channel slice.
        in_channels_list: Input channel counts for each child convolution.
    """

    def __init__(self, convs: Iterable[nn.Conv2d], in_channels_list: Iterable[int]) -> None:
        """Initialize a split-input convolution module."""

        super().__init__()
        self.convs = nn.ModuleList(convs)
        self.in_channels_list = list(in_channels_list)
        self.in_channels = sum(self.in_channels_list)
        self.out_channels = self.convs[0].out_channels

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, splits: list[int]) -> "SplitConv2d":
        """Create a split-input replacement from an existing convolution.

        Args:
            conv: Source convolution.
            splits: Input-channel splits. Any remaining channels are appended.

        Returns:
            Replacement preserving the source convolution output.
        """

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
        """Apply child convolutions to input-channel slices and sum outputs.

        Args:
            x: Input image/features tensor.

        Returns:
            Output tensor equivalent to the original convolution.
        """

        chunks = x.split(self.in_channels_list, dim=1)
        return sum(conv(chunk.contiguous()) for conv, chunk in zip(self.convs, chunks, strict=True))


class ShiftedConv2d(nn.Module):
    """Conv2d wrapper that applies a fixed channel shift before convolution.

    Args:
        conv: Bias-adjusted convolution.
        shift: Channel shift added to inputs before applying ``conv``.
    """

    def __init__(self, conv: nn.Conv2d, shift: torch.Tensor) -> None:
        """Initialize a shifted convolution wrapper."""

        super().__init__()
        self.conv = conv
        self.conv.shifted = True
        self.register_buffer("shift", shift.to(device=conv.weight.device, dtype=conv.weight.dtype).flatten())

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, shift: float | torch.Tensor) -> "ShiftedConv2d":
        """Create a shifted wrapper while preserving original conv outputs.

        Args:
            conv: Source convolution. Only zero padding is supported.
            shift: Scalar or channel shift applied at runtime.

        Returns:
            Wrapper with bias adjusted for the configured shift.
        """

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
        # DeepCompressor used float64 here for a numerically conservative
        # weight @ shift accumulation. Float32 keeps the one-time bias fold more
        # stable than fp16/bf16 while avoiding large CUDA float64 temporaries.
        weight_f32 = conv.weight.float()
        shift_f32 = shift_tensor.float()
        if shift_tensor.numel() == 1:
            shifted_bias = weight_f32.sum(dim=(1, 2, 3)) * shift_f32
        else:
            shifted_bias = weight_f32.sum(dim=(2, 3)) @ shift_f32.view(-1, 1)
            shifted_bias = shifted_bias.view(-1)
        if conv.bias is None:
            replacement.bias.data.copy_((-shifted_bias).to(conv.weight.dtype))
        else:
            replacement.bias.data.copy_((conv.bias.float() - shifted_bias).to(conv.weight.dtype))
        return cls(replacement, shift_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shifted convolution.

        Args:
            x: Input tensor in NCHW layout.

        Returns:
            Shifted convolution output.
        """

        return self.conv(x + self.shift.view(1, -1, 1, 1))


def prepare_model(model: nn.Module, patches: Iterable[PatchRule]) -> nn.Module:
    """Apply configured architecture rewrites to a model in place.

    Args:
        model: Model to mutate.
        patches: Rewrite rules to apply in order.

    Returns:
        The same model instance after patching.
    """

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
    """Apply one split-input linear patch.

    Args:
        model: Model to mutate.
        rule: Patch rule containing module pattern and split args.
    """

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
    """Apply one split-output linear patch.

    Args:
        model: Model to mutate.
        rule: Patch rule containing module pattern and split args.
    """

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
    """Apply one shifted linear patch.

    Args:
        model: Model to mutate.
        rule: Patch rule containing module pattern and ``shift`` arg.
    """

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
    """Apply one split-input convolution patch.

    Args:
        model: Model to mutate.
        rule: Patch rule containing module pattern and split args.
    """

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
    """Apply one shifted convolution patch.

    Args:
        model: Model to mutate.
        rule: Patch rule containing module pattern and ``shift`` arg.
    """

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
    """Return module names matched by a capture pattern in deterministic order.

    Args:
        pattern: Module path pattern.
        modules: Mapping of module names to modules.

    Returns:
        Matched module names sorted by wildcard captures.
    """

    matches = match_pattern(pattern, modules)
    return [matches[capture] for capture in sorted(matches, key=capture_sort_key)]


def _resolve_splits(module: nn.Linear, split_specs: Iterable[int | str]) -> list[int]:
    """Resolve input split specs for a linear layer.

    Args:
        module: Linear layer used for symbolic sizes.
        split_specs: Integer splits or symbolic feature names.

    Returns:
        Concrete input feature splits.
    """

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
    """Resolve output split specs for a linear layer.

    Args:
        module: Linear layer used for symbolic sizes.
        split_specs: Integer splits or symbolic feature names.

    Returns:
        Concrete output feature splits.
    """

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
    """Resolve input-channel split specs for a convolution.

    Args:
        module: Convolution used for symbolic channel sizes.
        split_specs: Integer splits or symbolic channel names.

    Returns:
        Concrete input channel splits.
    """

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


def _set_module(root: nn.Module, module_name: str, replacement: nn.Module) -> None:
    """Replace a named child module under a root module.

    Args:
        root: Root module containing the child.
        module_name: Dot path to the child module.
        replacement: Module to install at ``module_name``.
    """

    parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child_name, replacement)
