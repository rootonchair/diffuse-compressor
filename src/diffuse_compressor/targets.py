from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass
from typing import Sequence

import torch.nn as nn

from .config import TargetConfig, TargetRule


@dataclass(frozen=True)
class QuantTarget:
    name: str
    modules: tuple[nn.Module, ...]
    module_names: tuple[str, ...]
    export_name: str
    kind: str = "linear"
    roles: tuple[str, ...] = ()
    shared_low_rank: bool = True
    smooth_key: str | None = None


def collect_quant_targets(model: nn.Module, target_config: TargetConfig) -> list[QuantTarget]:
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
            )
        )
    return targets


def _match_pattern(pattern: str, modules: dict[str, nn.Module]) -> dict[tuple[str, ...], str]:
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
    parts: list[str] = []
    for char in pattern:
        if char == "*":
            parts.append(r"([^.]+)")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts))


def _format_export_name(template: str, capture: tuple[str, ...]) -> str:
    try:
        return template.format(*capture)
    except IndexError as exc:
        raise ValueError(f"Template {template!r} references missing wildcard capture {capture}") from exc


def _capture_sort_key(capture: tuple[str, ...]) -> tuple[object, ...]:
    key: list[object] = []
    for item in capture:
        key.append((0, int(item)) if item.isdigit() else (1, item))
    return tuple(key)


def _validate_target(target: QuantTarget) -> None:
    if target.kind == "linear":
        if not all(isinstance(module, nn.Linear) for module in target.modules):
            names = ", ".join(f"{name}: {type(module).__name__}" for name, module in zip(target.module_names, target.modules))
            raise TypeError(f"Linear target {target.name!r} contains non-Linear modules: {names}")
        in_features = {module.in_features for module in target.modules if isinstance(module, nn.Linear)}
        if len(in_features) != 1:
            raise ValueError(f"Grouped linear target {target.name!r} has mismatched input sizes: {sorted(in_features)}")
        return
    if target.kind == "conv":
        if not all(isinstance(module, nn.Conv2d) for module in target.modules):
            names = ", ".join(f"{name}: {type(module).__name__}" for name, module in zip(target.module_names, target.modules))
            raise TypeError(f"Conv target {target.name!r} contains non-Conv2d modules: {names}")
        return
    raise ValueError(f"Unsupported target kind {target.kind!r} for {target.name!r}")
