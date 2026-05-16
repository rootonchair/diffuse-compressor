from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from .config import CalibrationSpec, TargetConfig
from .targets import QuantTarget, _capture_sort_key, _format_export_name, _match_pattern


@dataclass(frozen=True)
class CalibrationScope:
    name: str
    targets: tuple[QuantTarget, ...]


@dataclass(frozen=True)
class CalibrationScopeBatch:
    scope: CalibrationScope
    inputs: dict[str, torch.Tensor]


def has_runnable_calibration(calibration: CalibrationSpec | None) -> bool:
    if calibration is None:
        return False
    if calibration.cache_mode != "disabled" and _cache_files(calibration):
        return True
    return bool(calibration.samples is not None or calibration.forward_fn is not None)


@torch.inference_mode()
def prepare_calibration_cache(model: nn.Module, calibration: CalibrationSpec | None) -> list[Path]:
    if calibration is None or calibration.cache_mode == "disabled" or calibration.cache_dir is None:
        return []

    cache_root = Path(calibration.cache_dir) / "caches"
    existing = sorted(cache_root.glob("*.pt"))
    if calibration.cache_mode == "reuse" and existing:
        return existing

    if calibration.cache_mode == "refresh" and cache_root.exists():
        for path in cache_root.glob("*.pt"):
            path.unlink()
    cache_root.mkdir(parents=True, exist_ok=True)

    samples = _resolve_samples(calibration)
    if not samples:
        return sorted(cache_root.glob("*.pt"))

    paths: list[Path] = []
    counter = 0

    def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        nonlocal counter
        path = cache_root / f"{counter:08d}.pt"
        torch.save({"args": _to_cpu(args), "kwargs": _to_cpu(kwargs)}, path)
        paths.append(path)
        counter += 1

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        for sample in samples:
            _run_sample(model, calibration, sample)
            _check_ram(calibration)
    finally:
        handle.remove()
    return paths


@torch.inference_mode()
def iter_calibration_scopes(
    model: nn.Module,
    targets: Iterable[QuantTarget],
    target_config: TargetConfig | None,
    calibration: CalibrationSpec | None,
) -> Iterator[CalibrationScopeBatch]:
    target_list = list(targets)
    scopes = assign_calibration_scopes(model, target_list, target_config)
    if not has_runnable_calibration(calibration):
        for scope in scopes:
            yield CalibrationScopeBatch(scope=scope, inputs={})
        return

    assert calibration is not None
    cache_paths = prepare_calibration_cache(model, calibration)
    samples = _resolve_samples(calibration) if calibration.cache_mode == "disabled" or not cache_paths else []
    device = _model_device(model)

    for scope in scopes:
        capture = _InputCapture(list(scope.targets), max_rows_per_target=calibration.max_rows_per_target)
        capture.install()
        try:
            if cache_paths:
                for path in cache_paths:
                    item = torch.load(path, map_location="cpu", weights_only=False)
                    args = _to_device(item.get("args", ()), device)
                    kwargs = _to_device(item.get("kwargs", {}), device)
                    model(*args, **kwargs)
                    _check_ram(calibration)
            else:
                for sample in samples:
                    _run_sample(model, calibration, sample)
                    _check_ram(calibration)
        finally:
            capture.remove()
        inputs = capture.inputs()
        yield CalibrationScopeBatch(scope=scope, inputs=inputs)
        del inputs, capture
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def assign_calibration_scopes(
    model: nn.Module,
    targets: Iterable[QuantTarget],
    target_config: TargetConfig | None,
) -> list[CalibrationScope]:
    target_list = list(targets)
    if not target_list:
        return []
    scope_rules = tuple(target_config.calibration_scopes) if target_config is not None else ()
    if not scope_rules:
        return [CalibrationScope(name=target.export_name, targets=(target,)) for target in target_list]

    modules = dict(model.named_modules())
    concrete_scopes: list[tuple[str, str]] = []
    used_names: set[str] = set()
    for rule in scope_rules:
        for pattern in rule.modules:
            matches = _match_pattern(pattern, modules)
            for capture in sorted(matches, key=_capture_sort_key):
                module_name = matches[capture]
                base_name = _format_export_name(rule.name, capture)
                scope_name = base_name
                if scope_name in used_names:
                    scope_name = f"{base_name}:{module_name}"
                used_names.add(scope_name)
                concrete_scopes.append((scope_name, module_name))

    assigned: dict[str, list[QuantTarget]] = {name: [] for name, _ in concrete_scopes}
    fallback: list[CalibrationScope] = []
    for target in target_list:
        matches = [
            (scope_name, scope_module)
            for scope_name, scope_module in concrete_scopes
            if any(_is_under_scope(module_name, scope_module) for module_name in target.module_names)
        ]
        if not matches:
            fallback.append(CalibrationScope(name=target.export_name, targets=(target,)))
            continue
        scope_name, _ = max(matches, key=lambda item: len(item[1]))
        assigned[scope_name].append(target)

    scopes = [
        CalibrationScope(name=name, targets=tuple(scope_targets))
        for name, scope_targets in assigned.items()
        if scope_targets
    ]
    scopes.extend(fallback)
    return scopes


class _InputCapture:
    def __init__(self, targets: list[QuantTarget], max_rows_per_target: int) -> None:
        self._targets = targets
        self._max_rows = max_rows_per_target
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._inputs: dict[str, list[torch.Tensor]] = {target.export_name: [] for target in targets}
        self._rows: dict[str, int] = {target.export_name: 0 for target in targets}

    def install(self) -> None:
        for target in self._targets:
            if not target.modules:
                continue
            self._handles.append(target.modules[0].register_forward_pre_hook(self._hook(target.export_name)))

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def inputs(self) -> dict[str, torch.Tensor]:
        return {name: torch.cat(chunks, dim=0) for name, chunks in self._inputs.items() if chunks}

    def _hook(self, export_name: str):
        def hook(_module: nn.Module, args: tuple[Any, ...]) -> None:
            if self._rows[export_name] >= self._max_rows or not args or not torch.is_tensor(args[0]):
                return
            x = args[0].detach()
            rows = x.reshape(-1, x.shape[-1]).float().cpu()
            remaining = self._max_rows - self._rows[export_name]
            if rows.shape[0] > remaining:
                rows = rows[:remaining]
            self._inputs[export_name].append(rows)
            self._rows[export_name] += rows.shape[0]

        return hook


def _run_sample(model: nn.Module, calibration: CalibrationSpec, sample: dict[str, Any]) -> None:
    if calibration.forward_fn is not None:
        calibration.forward_fn(sample)
    else:
        model(**sample)


def _cache_files(calibration: CalibrationSpec) -> list[Path]:
    if calibration.cache_dir is None:
        return []
    return sorted((Path(calibration.cache_dir) / "caches").glob("*.pt"))


def _resolve_samples(calibration: CalibrationSpec) -> list[dict[str, Any]]:
    if calibration.samples is not None:
        samples = list(calibration.samples)
    elif calibration.forward_fn is not None and calibration.prompts is not None:
        prompts = _resolve_prompts(calibration.prompts)
        samples = [{"prompt": prompt} for prompt in prompts]
    else:
        samples = []

    if calibration.num_samples is not None:
        samples = samples[: calibration.num_samples]
    return samples


def _resolve_prompts(prompts: Sequence[str] | str | Path) -> list[str]:
    if isinstance(prompts, Path):
        return _read_prompt_file(prompts)
    if isinstance(prompts, str):
        path = Path(prompts)
        if path.exists():
            return _read_prompt_file(path)
        return [prompts]
    return list(prompts)


def _read_prompt_file(path: Path) -> list[str]:
    lines = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if line:
            lines.append(line.strip("\"'"))
    return lines


def _is_under_scope(module_name: str, scope_name: str) -> bool:
    return module_name == scope_name or module_name.startswith(f"{scope_name}.")


def _model_device(model: nn.Module) -> torch.device:
    for tensor in model.parameters(recurse=True):
        return tensor.device
    for tensor in model.buffers(recurse=True):
        return tensor.device
    return torch.device("cpu")


def _to_cpu(value: Any) -> Any:
    return _tree_map(value, lambda tensor: tensor.detach().cpu())


def _to_device(value: Any, device: torch.device) -> Any:
    return _tree_map(value, lambda tensor: tensor.to(device=device))


def _tree_map(value: Any, tensor_fn) -> Any:
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: _tree_map(item, tensor_fn) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_map(item, tensor_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_map(item, tensor_fn) for item in value)
    return value


def _check_ram(calibration: CalibrationSpec) -> None:
    try:
        import psutil
    except ImportError:
        return
    usage = psutil.virtual_memory().percent / 100
    if usage > calibration.ram_usage_limit:
        raise RuntimeError(
            f"memory usage {usage:.1%} exceeds calibration ram_usage_limit {calibration.ram_usage_limit:.1%}"
        )
