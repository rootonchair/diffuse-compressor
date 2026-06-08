from __future__ import annotations

import re
from typing import Sequence

import torch.nn as nn


def match_pattern(
    pattern: str, modules: dict[str, nn.Module], *, module_classes: tuple[type, ...] | None = None
) -> dict[tuple[str, ...], str]:
    """Match one wildcard module pattern against named modules."""

    regex = _glob_to_capture_regex(pattern)
    matched: dict[tuple[str, ...], str] = {}
    for name, module in modules.items():
        match = regex.fullmatch(name)
        if not match:
            continue
        if module_classes is not None and not isinstance(module, module_classes):
            continue
        capture = tuple(match.groups())
        if capture in matched:
            raise ValueError(f"Pattern {pattern!r} ambiguously matched {matched[capture]!r} and {name!r}")
        matched[capture] = name
    if not matched:
        suffix = " after module_classes filtering" if module_classes is not None else ""
        raise ValueError(f"Pattern {pattern!r} did not match any modules{suffix}")
    return matched


def match_module_classes(
    modules: dict[str, nn.Module],
    module_classes: tuple[type, ...] | None,
    *,
    scope_module_classes: tuple[type, ...] | None = None,
    omit: set[str] | None = None,
    allow_empty: bool = False,
) -> dict[tuple[str, ...], str]:
    """Match named child modules by class without using module path patterns."""

    if module_classes is None:
        raise ValueError("module_classes must be provided when modules is omitted")
    omit = omit or set()
    scope_names = scope_module_names(modules, scope_module_classes)
    candidates = {
        (name,): name
        for name, module in sorted(modules.items(), key=lambda item: module_sort_key(item[0]))
        if name and isinstance(module, module_classes) and is_in_scopes(name, scope_names)
    }
    if not candidates:
        class_names = ", ".join(class_name(cls) for cls in module_classes)
        raise ValueError(f"No child modules matched module_classes ({class_names})")
    matched = {capture: name for capture, name in candidates.items() if name not in omit}
    if not matched and not allow_empty:
        class_names = ", ".join(class_name(cls) for cls in module_classes)
        raise ValueError(f"No child modules matched module_classes ({class_names})")
    return matched


def module_classes_tuple(module_classes: type | Sequence[type] | None) -> tuple[type, ...] | None:
    """Normalize an optional class or class sequence to a tuple."""

    if module_classes is None:
        return None
    if isinstance(module_classes, type):
        return (module_classes,)
    return tuple(module_classes)


def format_export_name(template: str, capture: tuple[str, ...]) -> str:
    """Format a target, scope, or export name from wildcard captures."""

    try:
        return template.format(*capture)
    except IndexError as exc:
        raise ValueError(f"Template {template!r} references missing wildcard capture {capture}") from exc


def capture_sort_key(capture: tuple[str, ...]) -> tuple[object, ...]:
    """Build a deterministic sort key for wildcard capture tuples."""

    key: list[object] = []
    for item in capture:
        if item.isdigit():
            key.append((0, int(item)))
        elif "." in item:
            key.append((1, module_sort_key(item)))
        else:
            key.append((2, item))
    return tuple(key)


def module_sort_key(name: str) -> tuple[object, ...]:
    """Build a deterministic sort key for dotted module names."""

    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in name.split("."))


def class_name(cls: type) -> str:
    """Return a readable class name for diagnostics."""

    return f"{cls.__module__}.{cls.__qualname__}"


def scope_module_names(modules: dict[str, nn.Module], scope_module_classes: tuple[type, ...] | None) -> tuple[str, ...]:
    """Return module names that bound class-based matching scopes."""

    if scope_module_classes is None:
        return ("",)
    return tuple(
        name
        for name, module in sorted(modules.items(), key=lambda item: module_sort_key(item[0]))
        if name and isinstance(module, scope_module_classes)
    )


def is_in_scopes(module_name: str, scope_names: tuple[str, ...]) -> bool:
    """Return whether a module path is inside one of the named scopes."""

    return any(
        scope_name == "" or module_name == scope_name or module_name.startswith(f"{scope_name}.")
        for scope_name in scope_names
    )


def _glob_to_capture_regex(pattern: str) -> re.Pattern[str]:
    parts: list[str] = []
    for char in pattern:
        if char == "*":
            parts.append(r"([^.]+)")
        else:
            parts.append(re.escape(char))
    return re.compile("".join(parts))
