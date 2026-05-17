from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn

from .config import ActivationQuantSpec, SmoothSpec, TargetConfig, TargetRule


@dataclass(frozen=True)
class QuantTarget:
    """Concrete module or grouped modules selected for quantization.

    Args:
        name: Logical target name.
        modules: Module objects included in this target.
        module_names: Fully qualified model paths for ``modules``.
        export_name: Runtime/checkpoint name used for exported tensors.
        kind: Target kind, currently ``"linear"`` or ``"conv"``.
        roles: Optional semantic roles for grouped modules.
        shared_low_rank: Whether grouped modules share one low-rank branch.
        smooth_key: Optional key for sharing smoothing decisions.
        precision: Optional target-level precision override.
        group_size: Optional target-level group-size override.
        rank: Optional target-level low-rank rank override.
        smooth: Optional target-level smoothing override.
        activation_quant: Optional target-level activation quantization override.
        shift_activations: Optional target-level activation shift override.
    """

    name: str
    modules: tuple[nn.Module, ...]
    module_names: tuple[str, ...]
    export_name: str
    kind: str = "linear"
    roles: tuple[str, ...] = ()
    shared_low_rank: bool = True
    smooth_key: str | None = None
    precision: str | None = None
    group_size: int | None = None
    rank: int | None = None
    smooth: bool | SmoothSpec | None = None
    activation_quant: bool | ActivationQuantSpec | None = None
    shift_activations: bool | None = None


def collect_quant_targets(model: nn.Module, target_config: TargetConfig) -> list[QuantTarget]:
    """Expand target rules into concrete quantization targets.

    Args:
        model: Model whose named modules are matched against target patterns.
        target_config: Target configuration containing one or more rules.

    Returns:
        Ordered concrete targets with duplicate export names rejected.
    """

    modules = dict(model.named_modules())
    targets: list[QuantTarget] = []
    used_exports: set[str] = set()
    for rule in target_config.targets:
        expanded = _expand_rule(rule, modules)
        for target in expanded:
            if target.export_name in used_exports:
                raise ValueError(f"Duplicate export_name {target.export_name!r}")
            used_exports.add(target.export_name)
            _validate_target(target)
            targets.append(target)
    return targets


def select_unquantized_state_dict(
    model: nn.Module,
    patterns: Sequence[str],
    quantized_prefixes: Sequence[str],
) -> dict[str, object]:
    """Select state-dict tensors that should remain unquantized.

    Args:
        model: Source model.
        patterns: Optional fnmatch patterns. Negated patterns start with ``!``.
        quantized_prefixes: Module prefixes excluded when no explicit patterns
            are supplied.

    Returns:
        CPU state-dict mapping for tensors that should be exported unchanged.
    """

    state = model.state_dict()
    if patterns:
        selected: dict[str, object] = {}
        for key, value in state.items():
            include = False
            for pattern in patterns:
                negated = pattern.startswith("!")
                body = pattern[1:] if negated else pattern
                if fnmatch.fnmatchcase(key, body):
                    include = not negated
            if include:
                selected[key] = value.detach().cpu()
        return selected

    skipped = tuple(f"{name}." for name in quantized_prefixes)
    return {key: value.detach().cpu() for key, value in state.items() if not key.startswith(skipped)}


def _expand_rule(rule: TargetRule, modules: dict[str, nn.Module]) -> list[QuantTarget]:
    """Expand a target rule by shared wildcard captures.

    Args:
        rule: Rule with module patterns and export templates.
        modules: Mapping of model module names to module objects.

    Returns:
        Concrete targets formed from shared wildcard captures.
    """

    if not rule.modules:
        raise ValueError(f"TargetRule {rule.name!r} must contain at least one module pattern")
    matches = [_match_pattern(pattern, modules) for pattern in rule.modules]
    capture_keys = [set(items) for items in matches]
    shared_keys = set.intersection(*capture_keys)
    if not shared_keys:
        details = ", ".join(f"{pattern!r}: {sorted(items)}" for pattern, items in zip(rule.modules, capture_keys))
        raise ValueError(f"TargetRule {rule.name!r} module patterns do not share wildcard captures: {details}")

    targets: list[QuantTarget] = []
    for capture in sorted(shared_keys, key=_capture_sort_key):
        module_names = tuple(match[capture] for match in matches)
        export_name = _format_export_name(rule.export_name or rule.name, capture)
        target_name = _format_export_name(rule.name, capture)
        targets.append(
            QuantTarget(
                name=target_name,
                modules=tuple(modules[name] for name in module_names),
                module_names=module_names,
                export_name=export_name,
                kind=rule.kind,
                roles=tuple(rule.roles),
                shared_low_rank=rule.shared_low_rank,
                smooth_key=rule.smooth_key,
                precision=rule.precision,
                group_size=rule.group_size,
                rank=rule.rank,
                smooth=rule.smooth,
                activation_quant=rule.activation_quant,
                shift_activations=rule.shift_activations,
            )
        )
    return targets


def _match_pattern(pattern: str, modules: dict[str, nn.Module]) -> dict[tuple[str, ...], str]:
    """Match one wildcard module pattern against named modules.

    Args:
        pattern: Dot-path glob pattern where ``*`` captures one path segment.
        modules: Mapping of available module names to modules.

    Returns:
        Mapping from wildcard capture tuples to matched module names.
    """

    regex = _glob_to_capture_regex(pattern)
    matched: dict[tuple[str, ...], str] = {}
    for name in modules:
        match = regex.fullmatch(name)
        if not match:
            continue
        capture = tuple(match.groups())
        if capture in matched:
            raise ValueError(f"Pattern {pattern!r} ambiguously matched {matched[capture]!r} and {name!r}")
        matched[capture] = name
    if not matched:
        raise ValueError(f"Pattern {pattern!r} did not match any modules")
    return matched


def _glob_to_capture_regex(pattern: str) -> re.Pattern[str]:
    """Convert a module glob pattern into a capture regex.

    Args:
        pattern: Module path pattern using ``*`` for one segment.

    Returns:
        Compiled regex with one capture group per wildcard.
    """

    parts: list[str] = []
    for char in pattern:
        if char == "*":
            parts.append(r"([^.]+)")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts))


def _format_export_name(template: str, capture: tuple[str, ...]) -> str:
    """Format a target or export name from wildcard captures.

    Args:
        template: Python format string using capture indices.
        capture: Wildcard capture values.

    Returns:
        Formatted name.
    """

    try:
        return template.format(*capture)
    except IndexError as exc:
        raise ValueError(f"Template {template!r} references missing wildcard capture {capture}") from exc


def _capture_sort_key(capture: tuple[str, ...]) -> tuple[object, ...]:
    """Build a deterministic sort key for wildcard capture tuples.

    Args:
        capture: Wildcard capture values.

    Returns:
        Tuple that sorts numeric captures numerically and strings
        lexicographically.
    """

    key: list[object] = []
    for item in capture:
        key.append((0, int(item)) if item.isdigit() else (1, item))
    return tuple(key)


def _validate_target(target: QuantTarget) -> None:
    """Validate target module kinds and grouped shape compatibility.

    Args:
        target: Concrete target to validate.
    """

    if target.kind == "linear":
        if not all(_is_linear_like(module) for module in target.modules):
            names = ", ".join(f"{name}: {type(module).__name__}" for name, module in zip(target.module_names, target.modules))
            raise TypeError(f"Linear target {target.name!r} contains non-Linear modules: {names}")
        in_features = {_linear_in_features(module) for module in target.modules}
        if len(in_features) != 1:
            raise ValueError(f"Grouped linear target {target.name!r} has mismatched input sizes: {sorted(in_features)}")
        return
    if target.kind == "conv":
        if not all(_is_conv_like(module) for module in target.modules):
            names = ", ".join(f"{name}: {type(module).__name__}" for name, module in zip(target.module_names, target.modules))
            raise TypeError(f"Conv target {target.name!r} contains non-Conv2d modules: {names}")
        in_channels = {_conv_in_channels(module) for module in target.modules}
        if len(in_channels) != 1:
            raise ValueError(f"Grouped conv target {target.name!r} has mismatched input sizes: {sorted(in_channels)}")
        return
    raise ValueError(f"Unsupported target kind {target.kind!r} for {target.name!r}")


def _is_linear_like(module: nn.Module) -> bool:
    """Return whether a module can be quantized as a linear layer.

    Args:
        module: Module selected by a target rule.

    Returns:
        ``True`` for raw ``nn.Linear`` modules and shifted-linear wrappers.
    """

    return isinstance(module, nn.Linear) or isinstance(getattr(module, "linear", None), nn.Linear)


def _linear_in_features(module: nn.Module) -> int:
    """Return input feature count for a raw or wrapped linear module.

    Args:
        module: Linear-like module.

    Returns:
        Input feature count.
    """

    if isinstance(module, nn.Linear):
        return module.in_features
    child = getattr(module, "linear", None)
    if isinstance(child, nn.Linear):
        return child.in_features
    raise TypeError(f"Module {type(module).__name__} is not linear-like")


def _is_conv_like(module: nn.Module) -> bool:
    """Return whether a module can be quantized as a Conv2d target.

    Args:
        module: Module selected by a target rule.

    Returns:
        ``True`` for raw ``nn.Conv2d`` modules and shifted-conv wrappers.
    """

    return isinstance(module, nn.Conv2d) or isinstance(getattr(module, "conv", None), nn.Conv2d)


def _conv_in_channels(module: nn.Module) -> int:
    """Return input channel count for a raw or wrapped Conv2d module.

    Args:
        module: Conv-like module.

    Returns:
        Input channel count.
    """

    if isinstance(module, nn.Conv2d):
        return module.in_channels
    child = getattr(module, "conv", None)
    if isinstance(child, nn.Conv2d):
        return child.in_channels
    raise TypeError(f"Module {type(module).__name__} is not conv-like")
