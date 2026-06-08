from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch.nn as nn

from .calibration import assign_calibration_scopes
from .config import CalibrationScopeRule, SkipRule, TargetConfig, TargetRule, target_quant_metadata
from .matching import class_name, match_module_classes, match_pattern, module_classes_tuple, scope_module_names
from .patches import prepare_model
from .targets import QuantTarget, collect_quant_targets


@dataclass(frozen=True)
class DiagnosticMessage:
    """One validation or inspection message."""

    level: str
    code: str
    message: str
    rule_index: int | None = None


@dataclass(frozen=True)
class InspectedTarget:
    """Concrete target selected by a target config."""

    name: str
    export_name: str
    modules: tuple[str, ...]
    kind: str
    roles: tuple[str, ...] = ()
    overrides: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class InspectedCapture:
    """Concrete calibration capture binding."""

    name: str
    inputs: bool
    outputs: bool
    input_keys: tuple[str | int, ...] = ()
    output_keys: tuple[str | int, ...] = ()
    channel_dim: int = -1


@dataclass(frozen=True)
class InspectedCalibrationScope:
    """Concrete calibration scope selected by a target config."""

    name: str
    targets: tuple[str, ...]
    module: str | None = None
    replay_module: str | None = None
    eval_module: str | None = None
    captures: tuple[InspectedCapture, ...] = ()
    cache_aliases: dict[str, str] = field(default_factory=dict)
    replay_arg_indices: tuple[int, ...] = ()
    replay_kwarg_keys: tuple[str, ...] = ()
    use_prev_scope_outputs: bool = True
    recompute: bool = False


@dataclass(frozen=True)
class TargetConfigReport:
    """Structured inspection result for a model and target config."""

    targets: tuple[InspectedTarget, ...]
    calibration_scopes: tuple[InspectedCalibrationScope, ...]
    skipped_modules: tuple[str, ...]
    unquantized_keys: tuple[str, ...]
    warnings: tuple[DiagnosticMessage, ...] = ()
    errors: tuple[DiagnosticMessage, ...] = ()

    @property
    def ok(self) -> bool:
        """Return whether inspection found no errors."""

        return not self.errors

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable report dictionary."""

        return asdict(self)

    def format_text(self) -> str:
        """Return a stable human-readable inspection report."""

        lines = [
            "Target config inspection",
            f"- targets: {len(self.targets)}",
            f"- calibration scopes: {len(self.calibration_scopes)}",
            f"- skipped modules: {len(self.skipped_modules)}",
            f"- unquantized keys: {len(self.unquantized_keys)}",
        ]
        if self.errors:
            lines.append("Errors:")
            lines.extend(f"  - [{message.code}] {message.message}" for message in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - [{message.code}] {message.message}" for message in self.warnings)
        if self.targets:
            lines.append("Targets:")
            for target in self.targets:
                role_text = "" if not target.roles else f" roles={list(target.roles)}"
                lines.append(f"  - {target.export_name}: {list(target.modules)} kind={target.kind}{role_text}")
        if self.calibration_scopes:
            lines.append("Calibration scopes:")
            for scope in self.calibration_scopes:
                replay = scope.replay_module or scope.module
                eval_module = scope.eval_module or scope.module
                lines.append(f"  - {scope.name}: targets={list(scope.targets)} replay={replay!r} eval={eval_module!r}")
                for capture in scope.captures:
                    sides = []
                    if capture.inputs:
                        sides.append("inputs")
                    if capture.outputs:
                        sides.append("outputs")
                    lines.append(f"      capture {capture.name}: {','.join(sides)}")
        return "\n".join(lines)


def inspect_target_config(model: nn.Module, target_config: TargetConfig) -> TargetConfigReport:
    """Apply structural patches and inspect target expansion without quantizing.

    Args:
        model: Model whose modules and state dict should be inspected. The
            model is mutated in place by ``target_config.patches`` before
            target collection, matching ``quantize_and_export`` behavior.
        target_config: Target configuration to expand.

    Returns:
        Structured report with concrete matches, warnings, and errors.
    """

    warnings: list[DiagnosticMessage] = []
    errors: list[DiagnosticMessage] = []
    try:
        prepare_model(model, target_config.patches)
    except Exception as exc:  # noqa: BLE001 - diagnostics should report config failures without raising.
        errors.append(DiagnosticMessage("error", "model_prepare_failed", str(exc)))
    modules = dict(model.named_modules())
    warnings.extend(_target_rule_warnings(modules, tuple(target_config.targets)))
    warnings.extend(_skip_rule_warnings(modules, tuple(target_config.skips)))
    warnings.extend(_scope_rule_warnings(modules, tuple(target_config.calibration_scopes)))

    skipped_modules = _skipped_modules_for_report(modules, tuple(target_config.skips), errors)
    unquantized_keys = _explicit_unquantized_keys(model, tuple(target_config.unquantized_patterns), warnings)

    targets: list[QuantTarget] = []
    try:
        targets = collect_quant_targets(model, target_config)
    except Exception as exc:  # noqa: BLE001 - diagnostics should report config failures without raising.
        errors.append(DiagnosticMessage("error", "target_collection_failed", str(exc)))

    inspected_targets = tuple(_inspect_target(target) for target in targets)
    inspected_scopes: tuple[InspectedCalibrationScope, ...] = ()
    if targets:
        try:
            inspected_scopes = tuple(
                _inspect_scope(scope) for scope in assign_calibration_scopes(model, targets, target_config)
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should report config failures without raising.
            errors.append(DiagnosticMessage("error", "calibration_scope_failed", str(exc)))
    if target_config.calibration_scopes and targets and not inspected_scopes:
        warnings.append(DiagnosticMessage("warning", "no_calibration_scopes", "No calibration scopes were assigned"))

    return TargetConfigReport(
        targets=inspected_targets,
        calibration_scopes=inspected_scopes,
        skipped_modules=tuple(sorted(skipped_modules)),
        unquantized_keys=tuple(unquantized_keys),
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def _inspect_target(target: QuantTarget) -> InspectedTarget:
    overrides: dict[str, object] = {"quant": target_quant_metadata(target.quant)}
    return InspectedTarget(
        name=target.name,
        export_name=target.export_name,
        modules=tuple(target.module_names),
        kind=target.kind,
        roles=tuple(target.roles),
        overrides=overrides,
    )


def _inspect_scope(scope) -> InspectedCalibrationScope:
    return InspectedCalibrationScope(
        name=scope.name,
        module=scope.module_name,
        replay_module=scope.replay_module_name,
        eval_module=scope.eval_module_name,
        targets=tuple(target.export_name for target in scope.targets),
        captures=tuple(
            InspectedCapture(
                name=capture.name,
                inputs=capture.inputs,
                outputs=capture.outputs,
                input_keys=tuple(capture.input_keys),
                output_keys=tuple(capture.output_keys),
                channel_dim=capture.channel_dim,
            )
            for capture in scope.captures
        ),
        cache_aliases=dict(scope.cache_aliases),
        replay_arg_indices=tuple(scope.replay_arg_indices),
        replay_kwarg_keys=tuple(scope.replay_kwarg_keys),
        use_prev_scope_outputs=scope.use_prev_scope_outputs,
        recompute=scope.recompute,
    )


def _target_rule_warnings(modules: dict[str, nn.Module], rules: tuple[TargetRule, ...]) -> list[DiagnosticMessage]:
    warnings: list[DiagnosticMessage] = []
    for index, rule in enumerate(rules):
        if rule.roles and rule.modules and len(rule.roles) != len(rule.modules):
            warnings.append(
                DiagnosticMessage(
                    "warning",
                    "target_roles_mismatch",
                    f"TargetRule {index} has {len(rule.roles)} roles for {len(rule.modules)} module patterns",
                    index,
                )
            )
        try:
            _target_rule_has_match(modules, rule)
        except Exception as exc:  # noqa: BLE001 - preflight should keep collecting diagnostics.
            warnings.append(DiagnosticMessage("warning", "target_rule_unmatched", f"TargetRule {index}: {exc}", index))
    return warnings


def _target_rule_has_match(modules: dict[str, nn.Module], rule: TargetRule) -> None:
    if rule.member_selector is not None:
        parent_classes = module_classes_tuple(rule.parent_module_classes)
        if parent_classes is None:
            raise ValueError("parent_module_classes must be provided for member_selector")
        matches = [(name, module) for name, module in modules.items() if name and isinstance(module, parent_classes)]
        if not matches:
            class_names = ", ".join(class_name(cls) for cls in parent_classes or ())
            raise ValueError(f"No parent modules matched parent_module_classes ({class_names})")
        return
    module_classes = module_classes_tuple(rule.module_classes)
    scope_classes = module_classes_tuple(rule.scope_module_classes)
    if rule.modules:
        for pattern in rule.modules:
            match_pattern(pattern, modules, module_classes=module_classes)
        return
    match_module_classes(modules, module_classes, scope_module_classes=scope_classes)


def _skip_rule_warnings(modules: dict[str, nn.Module], rules: tuple[SkipRule, ...]) -> list[DiagnosticMessage]:
    warnings: list[DiagnosticMessage] = []
    for index, rule in enumerate(rules):
        try:
            if not _skip_rule_matches(modules, rule):
                warnings.append(
                    DiagnosticMessage("warning", "skip_rule_unmatched", f"SkipRule {index} matched no modules", index)
                )
        except Exception as exc:  # noqa: BLE001 - preflight should keep collecting diagnostics.
            warnings.append(DiagnosticMessage("warning", "skip_rule_unmatched", f"SkipRule {index}: {exc}", index))
    return warnings


def _skip_rule_matches(modules: dict[str, nn.Module], rule: SkipRule) -> bool:
    module_classes = module_classes_tuple(rule.module_classes)
    scope_classes = module_classes_tuple(rule.scope_module_classes)
    if rule.modules:
        return any(match_pattern(pattern, modules, module_classes=module_classes) for pattern in rule.modules)
    if module_classes is not None:
        return bool(match_module_classes(modules, module_classes, scope_module_classes=scope_classes))
    return bool(scope_module_names(modules, scope_classes))


def _scope_rule_warnings(
    modules: dict[str, nn.Module], rules: tuple[CalibrationScopeRule, ...]
) -> list[DiagnosticMessage]:
    warnings: list[DiagnosticMessage] = []
    for index, rule in enumerate(rules):
        module_classes = module_classes_tuple(rule.module_classes)
        try:
            if rule.modules:
                for pattern in rule.modules:
                    match_pattern(pattern, modules, module_classes=module_classes)
            else:
                match_module_classes(modules, module_classes)
        except Exception as exc:  # noqa: BLE001 - preflight should keep collecting diagnostics.
            warnings.append(
                DiagnosticMessage(
                    "warning", "calibration_scope_unmatched", f"CalibrationScopeRule {index}: {exc}", index
                )
            )
    return warnings


def _skipped_modules_for_report(
    modules: dict[str, nn.Module], rules: tuple[SkipRule, ...], errors: list[DiagnosticMessage]
) -> set[str]:
    skipped: set[str] = set()
    for index, rule in enumerate(rules):
        try:
            module_classes = module_classes_tuple(rule.module_classes)
            scope_classes = module_classes_tuple(rule.scope_module_classes)
            if rule.modules:
                for pattern in rule.modules:
                    skipped.update(match_pattern(pattern, modules, module_classes=module_classes).values())
            elif module_classes is not None:
                skipped.update(
                    match_module_classes(modules, module_classes, scope_module_classes=scope_classes).values()
                )
            else:
                skipped.update(scope_module_names(modules, scope_classes))
        except Exception as exc:  # noqa: BLE001 - diagnostics should report config failures without raising.
            errors.append(DiagnosticMessage("error", "skip_rule_failed", f"SkipRule {index}: {exc}", index))
    return skipped


def _explicit_unquantized_keys(
    model: nn.Module, patterns: tuple[str, ...], warnings: list[DiagnosticMessage]
) -> list[str]:
    if not patterns:
        return []
    import fnmatch

    matched: list[str] = []
    for key in model.state_dict():
        include = False
        for pattern in patterns:
            negated = pattern.startswith("!")
            body = pattern[1:] if negated else pattern
            if fnmatch.fnmatchcase(key, body):
                include = not negated
        if include:
            matched.append(key)
    if not matched:
        warnings.append(
            DiagnosticMessage(
                "warning", "unquantized_patterns_unmatched", "unquantized_patterns matched no state-dict keys"
            )
        )
    return matched
