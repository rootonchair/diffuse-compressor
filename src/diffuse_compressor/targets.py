from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import torch.nn as nn

from .config import SvdqTargetQuant, TargetConfig, TargetQuantSpec, TargetRule, SkipRule
from .matching import (
    capture_sort_key,
    class_name,
    format_export_name,
    match_module_classes,
    match_pattern,
    module_classes_tuple,
    module_sort_key,
)


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
        quant: Target-level quantization behavior and runtime layout policy.
    """

    name: str
    modules: tuple[nn.Module, ...]
    module_names: tuple[str, ...]
    export_name: str
    kind: str = "linear"
    roles: tuple[str, ...] = ()
    quant: TargetQuantSpec = field(default_factory=SvdqTargetQuant)


def collect_quant_targets(model: nn.Module, target_config: TargetConfig) -> list[QuantTarget]:
    """Expand target rules into concrete quantization targets.

    Args:
        model: Model whose named modules are matched against target patterns.
        target_config: Target configuration containing one or more rules.

    Returns:
        Ordered concrete targets with duplicate export names rejected.
    """

    modules = dict(model.named_modules())
    module_paths = _module_paths_by_identity(modules)
    skipped = _skipped_module_names(tuple(target_config.skips), modules)
    group_expansions: dict[int, list[QuantTarget]] = {}
    grouped_modules: set[str] = set()
    for index, rule in enumerate(target_config.targets):
        if not _is_callable_group_rule(rule):
            continue
        expanded = _expand_callable_group_rule(rule, modules, module_paths, skipped)
        group_expansions[index] = expanded
        grouped_modules.update(name for target in expanded for name in target.module_names)

    targets: list[QuantTarget] = []
    used_exports: set[str] = set()
    for index, rule in enumerate(target_config.targets):
        if index in group_expansions:
            expanded = group_expansions[index]
        else:
            expanded = _expand_rule(rule, modules, skipped=skipped, grouped_modules=grouped_modules)
        for target in expanded:
            if target.export_name in used_exports:
                raise ValueError(f"Duplicate export_name {target.export_name!r}")
            used_exports.add(target.export_name)
            _validate_target(target)
            targets.append(target)
    return targets


def target_shared_low_rank(target: QuantTarget) -> bool:
    """Return whether this concrete target should use a shared low-rank branch."""

    if isinstance(target.quant, SvdqTargetQuant):
        return target.quant.shared_low_rank
    return False


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


def _expand_rule(
    rule: TargetRule,
    modules: dict[str, nn.Module],
    *,
    skipped: set[str] | None = None,
    grouped_modules: set[str] | None = None,
) -> list[QuantTarget]:
    """Expand a target rule by shared wildcard captures.

    Args:
        rule: Rule with module patterns and export templates.
        modules: Mapping of model module names to module objects.

    Returns:
        Concrete targets formed from shared wildcard captures.
    """

    skipped = skipped or set()
    grouped_modules = grouped_modules or set()
    module_classes = module_classes_tuple(rule.module_classes)
    scope_classes = module_classes_tuple(rule.scope_module_classes)
    if not rule.modules:
        matches = [
            match_module_classes(
                modules,
                module_classes,
                scope_module_classes=scope_classes,
                omit=skipped | grouped_modules,
                allow_empty=bool(skipped or grouped_modules),
            )
        ]
        if not matches[0]:
            return []
    else:
        matches = [match_pattern(pattern, modules, module_classes=module_classes) for pattern in rule.modules]
    capture_keys = [set(items) for items in matches]
    shared_keys = set.intersection(*capture_keys)
    if not shared_keys:
        details = ", ".join(f"{pattern!r}: {sorted(items)}" for pattern, items in zip(rule.modules, capture_keys))
        raise ValueError(f"TargetRule {rule.name!r} module patterns do not share wildcard captures: {details}")

    targets: list[QuantTarget] = []
    for capture in sorted(shared_keys, key=capture_sort_key):
        module_names = tuple(match[capture] for match in matches)
        selected_skipped = [name for name in module_names if name in skipped]
        if selected_skipped:
            raise ValueError(f"TargetRule {rule.name!r} explicitly selects skipped modules: {selected_skipped}")
        selected_grouped = [name for name in module_names if name in grouped_modules]
        if selected_grouped and rule.modules:
            raise ValueError(f"TargetRule {rule.name!r} explicitly selects grouped modules: {selected_grouped}")
        target_name = _target_name(rule, capture, module_names)
        export_name = _target_export_name(rule, capture, module_names)
        targets.append(
            QuantTarget(
                name=target_name,
                modules=tuple(modules[name] for name in module_names),
                module_names=module_names,
                export_name=export_name,
                kind=rule.kind,
                roles=tuple(rule.roles),
                quant=rule.quant,
            )
        )
    return targets


def _expand_callable_group_rule(
    rule: TargetRule,
    modules: dict[str, nn.Module],
    module_paths: dict[int, str],
    skipped: set[str],
) -> list[QuantTarget]:
    parent_classes = module_classes_tuple(rule.parent_module_classes)
    if parent_classes is None or rule.member_selector is None:
        raise ValueError("Callable group rules require parent_module_classes and member_selector")

    targets: list[QuantTarget] = []
    for parent_name, parent in sorted(modules.items(), key=lambda item: module_sort_key(item[0])):
        if not parent_name or not isinstance(parent, parent_classes):
            continue
        members = rule.member_selector(parent)
        if not isinstance(members, Mapping) or not members:
            raise ValueError(f"TargetRule {rule.name!r} member_selector for {parent_name!r} must return a non-empty mapping")
        roles: list[str] = []
        module_names: list[str] = []
        for role, module in members.items():
            if not isinstance(role, str) or not role:
                raise ValueError(f"TargetRule {rule.name!r} member_selector for {parent_name!r} returned invalid role {role!r}")
            if not isinstance(module, nn.Module):
                raise TypeError(
                    f"TargetRule {rule.name!r} member_selector for {parent_name!r} role {role!r} "
                    f"returned {type(module).__name__}, expected nn.Module"
                )
            module_name = module_paths.get(id(module))
            if module_name is None:
                raise ValueError(
                    f"TargetRule {rule.name!r} member_selector for {parent_name!r} role {role!r} "
                    "returned a module that is not present in model.named_modules()"
                )
            if module_name in skipped:
                raise ValueError(f"TargetRule {rule.name!r} member_selector selected skipped module {module_name!r}")
            roles.append(role)
            module_names.append(module_name)
        module_name_tuple = tuple(module_names)
        targets.append(
            QuantTarget(
                name=_callable_target_name(rule, parent_name),
                modules=tuple(modules[name] for name in module_name_tuple),
                module_names=module_name_tuple,
                export_name=_callable_export_name(rule, parent_name),
                kind=rule.kind,
                roles=tuple(roles),
                quant=rule.quant,
            )
        )
    if not targets:
        class_names = ", ".join(class_name(cls) for cls in parent_classes)
        raise ValueError(f"No parent modules matched parent_module_classes ({class_names})")
    return targets


def _format_named_template(template: str, **kwargs: str) -> str:
    try:
        return template.format(**kwargs)
    except KeyError as exc:
        raise ValueError(f"Template {template!r} references missing format key {exc.args[0]!r}") from exc


def _target_name(rule: TargetRule, capture: tuple[str, ...], module_names: tuple[str, ...]) -> str:
    if rule.name is None:
        return module_names[0]
    return format_export_name(rule.name, capture)


def _target_export_name(rule: TargetRule, capture: tuple[str, ...], module_names: tuple[str, ...]) -> str:
    if rule.export_name is not None:
        return format_export_name(rule.export_name, capture)
    if rule.name is not None:
        return format_export_name(rule.name, capture)
    return module_names[0]


def _callable_target_name(rule: TargetRule, parent_name: str) -> str:
    if rule.name is None:
        return parent_name
    return _format_named_template(rule.name, parent_path=parent_name)


def _callable_export_name(rule: TargetRule, parent_name: str) -> str:
    if rule.export_name is None:
        return parent_name
    return _format_named_template(rule.export_name, parent_path=parent_name)


def _is_callable_group_rule(rule: TargetRule) -> bool:
    return rule.member_selector is not None


def _module_paths_by_identity(modules: dict[str, nn.Module]) -> dict[int, str]:
    return {id(module): name for name, module in modules.items() if name}


def _scope_module_names(modules: dict[str, nn.Module], scope_module_classes: tuple[type, ...] | None) -> tuple[str, ...]:
    if scope_module_classes is None:
        return ()
    return tuple(
        name
        for name, module in sorted(modules.items(), key=lambda item: module_sort_key(item[0]))
        if name and isinstance(module, scope_module_classes)
    )


def _is_in_scopes(module_name: str, scope_names: tuple[str, ...]) -> bool:
    if not scope_names:
        return True
    return any(module_name == scope or module_name.startswith(f"{scope}.") for scope in scope_names)


def _skipped_module_names(rules: Sequence[SkipRule], modules: dict[str, nn.Module]) -> set[str]:
    skipped: set[str] = set()
    for rule in rules:
        module_classes = module_classes_tuple(rule.module_classes)
        scope_classes = module_classes_tuple(rule.scope_module_classes)
        if rule.modules:
            for pattern in rule.modules:
                skipped.update(match_pattern(pattern, modules, module_classes=module_classes).values())
            continue
        if module_classes is not None:
            skipped.update(
                match_module_classes(
                    modules,
                    module_classes,
                    scope_module_classes=scope_classes,
                ).values()
            )
            continue
        skipped.update(_scope_module_names(modules, scope_classes))
    return skipped


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
