from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch.nn as nn

from ..config import CalibrationCaptureRule, TargetConfig
from ..targets import (
    QuantTarget,
    _capture_sort_key,
    _format_export_name,
    _match_module_classes,
    _match_pattern,
    _module_classes_tuple,
)
from .types import CalibrationScope, CaptureBinding, EvalReplayBatch
from .utils import is_under_scope


def assign_calibration_scopes(
    model: nn.Module,
    targets: Iterable[QuantTarget],
    target_config: TargetConfig | None,
) -> list[CalibrationScope]:
    """Assign concrete targets to configured calibration scopes."""

    target_list = list(targets)
    if not target_list:
        return []
    scope_rules = tuple(target_config.calibration_scopes) if target_config is not None else ()
    if not scope_rules:
        return [CalibrationScope(name=target.export_name, targets=(target,)) for target in target_list]

    modules = dict(model.named_modules())
    concrete_scopes: list[
        tuple[
            str,
            str,
            str,
            str,
            tuple[CaptureBinding, ...],
            dict[str, str],
            tuple[int, ...],
            tuple[str, ...],
            Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]] | None,
            Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None,
            Callable[[EvalReplayBatch], tuple[tuple[Any, ...], dict[str, Any]]] | None,
            bool,
            bool,
        ]
    ] = []
    used_names: set[str] = set()
    for rule in scope_rules:
        module_classes = _module_classes_tuple(rule.module_classes)
        match_sets = (
            [_match_pattern(pattern, modules, module_classes=module_classes) for pattern in rule.modules]
            if rule.modules
            else [_match_module_classes(modules, module_classes)]
        )
        for matches in match_sets:
            for capture in sorted(matches, key=_capture_sort_key):
                module_name = matches[capture]
                base_name = _format_export_name(rule.name, capture) if rule.name is not None else module_name
                scope_name = base_name
                if scope_name in used_names:
                    scope_name = f"{base_name}:{module_name}"
                used_names.add(scope_name)
                eval_module_name = _resolve_module_template(rule.eval_module, capture, modules) if rule.eval_module else module_name
                replay_module_name = (
                    _resolve_module_template(rule.replay_module, capture, modules) if rule.replay_module else module_name
                )
                if eval_module_name not in modules:
                    raise ValueError(f"CalibrationScopeRule {rule.name!r} eval_module {eval_module_name!r} did not match any module")
                if replay_module_name not in modules:
                    raise ValueError(
                        f"CalibrationScopeRule {rule.name!r} replay_module {replay_module_name!r} did not match any module"
                    )
                captures = _expand_capture_rules(rule.capture_modules, capture, modules)
                cache_aliases = {
                    _format_export_name(alias, capture): _format_export_name(source, capture)
                    for alias, source in rule.cache_aliases.items()
                }
                concrete_scopes.append(
                    (
                        scope_name,
                        module_name,
                        replay_module_name,
                        eval_module_name,
                        captures,
                        cache_aliases,
                        tuple(rule.replay_arg_indices),
                        tuple(rule.replay_kwarg_keys),
                        rule.replay_transform,
                        rule.prev_output_transform,
                        rule.prev_replay_transform,
                        rule.use_prev_scope_outputs,
                        rule.recompute,
                    )
                )

    assigned: dict[str, list[QuantTarget]] = {name: [] for name, *_ in concrete_scopes}
    fallback: list[CalibrationScope] = []
    for target in target_list:
        matches = [
            (
                scope_name,
                scope_module,
                replay_module_name,
                eval_module_name,
                captures,
                cache_aliases,
                replay_arg_indices,
                replay_kwarg_keys,
                replay_transform,
                prev_output_transform,
                prev_replay_transform,
                use_prev,
                recompute,
            )
            for (
                scope_name,
                scope_module,
                replay_module_name,
                eval_module_name,
                captures,
                cache_aliases,
                replay_arg_indices,
                replay_kwarg_keys,
                replay_transform,
                prev_output_transform,
                prev_replay_transform,
                use_prev,
                recompute,
            ) in concrete_scopes
            if any(is_under_scope(module_name, scope_module) for module_name in target.module_names)
        ]
        if not matches:
            fallback.append(CalibrationScope(name=target.export_name, targets=(target,)))
            continue
        scope_name, *_ = max(matches, key=lambda item: len(item[1]))
        assigned[scope_name].append(target)

    scopes = []
    for (
        name,
        scope_module,
        replay_module_name,
        eval_module_name,
        captures,
        cache_aliases,
        replay_arg_indices,
        replay_kwarg_keys,
        replay_transform,
        prev_output_transform,
        prev_replay_transform,
        use_prev,
        recompute,
    ) in concrete_scopes:
        scope_targets = assigned[name]
        if not scope_targets:
            continue
        scopes.append(
            CalibrationScope(
                name=name,
                targets=tuple(scope_targets),
                module_name=scope_module,
                replay_module_name=replay_module_name,
                replay_module=modules[replay_module_name],
                eval_module_name=eval_module_name,
                eval_module=modules[eval_module_name],
                captures=captures,
                cache_aliases=cache_aliases,
                replay_arg_indices=replay_arg_indices,
                replay_kwarg_keys=replay_kwarg_keys,
                replay_transform=replay_transform,
                prev_output_transform=prev_output_transform,
                prev_replay_transform=prev_replay_transform,
                use_prev_scope_outputs=use_prev,
                recompute=recompute,
            )
        )
    scopes.extend(fallback)
    return scopes


def _expand_capture_rules(
    rules: Sequence[CalibrationCaptureRule],
    capture: tuple[str, ...],
    modules: dict[str, nn.Module],
) -> tuple[CaptureBinding, ...]:
    """Resolve capture rules for one wildcard capture tuple."""

    bindings: list[CaptureBinding] = []
    for rule in rules:
        for pattern in rule.modules:
            module_name = _resolve_module_template(pattern, capture, modules)
            name = _format_export_name(rule.name, capture)
            if len(rule.modules) > 1:
                name = f"{name}:{module_name}"
            bindings.append(
                CaptureBinding(
                    name=name,
                    module=modules[module_name],
                    inputs=rule.inputs,
                    outputs=rule.outputs,
                    input_keys=tuple(rule.input_keys),
                    output_keys=tuple(rule.output_keys),
                    channel_dim=rule.channel_dim,
                )
            )
    return tuple(bindings)


def _resolve_module_template(template: str, capture: tuple[str, ...], modules: dict[str, nn.Module]) -> str:
    """Resolve a module template or pattern to one module name."""

    if "*" in template:
        matches = _match_pattern(template, modules)
        if capture in matches:
            return matches[capture]
        if () in matches:
            return matches[()]
        raise ValueError(f"Module pattern {template!r} did not match capture {capture}")
    module_name = _format_export_name(template, capture)
    if module_name not in modules:
        raise ValueError(f"Module template {template!r} resolved to missing module {module_name!r}")
    return module_name
