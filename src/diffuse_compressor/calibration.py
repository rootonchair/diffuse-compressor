from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn

from .config import CalibrationCaptureRule, CalibrationSpec, TargetConfig
from .targets import QuantTarget, _capture_sort_key, _format_export_name, _match_pattern


@dataclass
class TensorCache:
    data: list[torch.Tensor] = field(default_factory=list)
    num_rows: int = 0
    num_total: int = 0

    def add(self, value: Any, *, max_rows: int, element_size: int = -1) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        rows = _tensor_rows(tensor)
        self.num_total += rows.shape[0]
        if self.num_rows >= max_rows:
            return
        remaining = max_rows - self.num_rows
        if rows.shape[0] > remaining:
            rows = rows[:remaining]
        if element_size > 0 and rows.shape[0] > element_size:
            rows = rows[:element_size]
        self.data.append(rows.float().cpu())
        self.num_rows += rows.shape[0]

    def tensor(self) -> torch.Tensor | None:
        if not self.data:
            return None
        return torch.cat(self.data, dim=0)

    def clear(self) -> None:
        self.data.clear()
        self.num_rows = 0


@dataclass
class IOTensorsCache:
    inputs: TensorCache = field(default_factory=TensorCache)
    outputs: TensorCache = field(default_factory=TensorCache)
    replay_args: tuple[Any, ...] | None = None
    replay_kwargs: dict[str, Any] | None = None

    def clear(self) -> None:
        self.inputs.clear()
        self.outputs.clear()
        self.replay_args = None
        self.replay_kwargs = None


@dataclass(frozen=True)
class EvalReplayBatch:
    module: nn.Module
    module_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any


@dataclass(frozen=True)
class CalibrationScope:
    name: str
    targets: tuple[QuantTarget, ...]
    module_name: str | None = None
    replay_module_name: str | None = None
    replay_module: nn.Module | None = None
    eval_module_name: str | None = None
    eval_module: nn.Module | None = None
    captures: tuple["CaptureBinding", ...] = ()
    use_prev_scope_outputs: bool = False
    recompute: bool = False


@dataclass(frozen=True)
class CalibrationScopeBatch:
    scope: CalibrationScope
    inputs: dict[str, torch.Tensor]
    layer_cache: dict[str, IOTensorsCache] = field(default_factory=dict)
    eval_replay: EvalReplayBatch | None = None


@dataclass(frozen=True)
class CaptureBinding:
    name: str
    module: nn.Module
    inputs: bool = True
    outputs: bool = False


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
    prev_scope_output: Any | None = None

    for scope in scopes:
        capture = _LayerCacheCapture(
            list(scope.targets),
            scope.captures,
            max_rows=calibration.max_rows_per_target,
            element_size=calibration.element_size,
        )
        eval_capture = _EvalReplayCapture(scope.eval_module, scope.eval_module_name, calibration.max_rows_per_target)
        capture.install()
        eval_capture.install()
        try:
            if scope.use_prev_scope_outputs and prev_scope_output is not None and not scope.recompute and scope.replay_module is not None:
                _run_replay_module(scope.replay_module, prev_scope_output, device)
                _check_ram(calibration)
            elif cache_paths:
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
            eval_capture.remove()
            capture.remove()
        inputs = capture.inputs()
        layer_cache = capture.layer_cache
        eval_replay = eval_capture.replay()
        yield CalibrationScopeBatch(scope=scope, inputs=inputs, layer_cache=layer_cache, eval_replay=eval_replay)
        prev_scope_output = eval_replay.output if eval_replay is not None else _first_cached_output(layer_cache)
        for cache in layer_cache.values():
            cache.clear()
        del inputs, layer_cache, capture, eval_capture
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
    concrete_scopes: list[tuple[str, str, str, str, tuple[CaptureBinding, ...], bool, bool]] = []
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
                concrete_scopes.append(
                    (
                        scope_name,
                        module_name,
                        replay_module_name,
                        eval_module_name,
                        captures,
                        rule.use_prev_scope_outputs,
                        rule.recompute,
                    )
                )

    assigned: dict[str, list[QuantTarget]] = {name: [] for name, *_ in concrete_scopes}
    fallback: list[CalibrationScope] = []
    for target in target_list:
        matches = [
            (scope_name, scope_module, replay_module_name, eval_module_name, captures, use_prev, recompute)
            for scope_name, scope_module, replay_module_name, eval_module_name, captures, use_prev, recompute in concrete_scopes
            if any(_is_under_scope(module_name, scope_module) for module_name in target.module_names)
        ]
        if not matches:
            fallback.append(CalibrationScope(name=target.export_name, targets=(target,)))
            continue
        scope_name, *_ = max(matches, key=lambda item: len(item[1]))
        assigned[scope_name].append(target)

    scopes = [
        CalibrationScope(
            name=name,
            targets=tuple(scope_targets),
            module_name=scope_module,
            replay_module_name=replay_module_name,
            replay_module=modules[replay_module_name],
            eval_module_name=eval_module_name,
            eval_module=modules[eval_module_name],
            captures=captures,
            use_prev_scope_outputs=use_prev,
            recompute=recompute,
        )
        for name, scope_module, replay_module_name, eval_module_name, captures, use_prev, recompute in concrete_scopes
        if (scope_targets := assigned[name])
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


class _LayerCacheCapture:
    def __init__(
        self,
        targets: list[QuantTarget],
        captures: Sequence[CaptureBinding],
        *,
        max_rows: int,
        element_size: int,
    ) -> None:
        self._bindings = self._target_bindings(targets) + list(captures)
        self._max_rows = max_rows
        self._element_size = element_size
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layer_cache: dict[str, IOTensorsCache] = {}

    def install(self) -> None:
        installed: set[tuple[int, str, str]] = set()
        for binding in self._bindings:
            self.layer_cache.setdefault(binding.name, IOTensorsCache())
            if binding.inputs:
                key = (id(binding.module), binding.name, "inputs")
                if key not in installed:
                    self._handles.append(
                        binding.module.register_forward_pre_hook(self._input_hook(binding.name), with_kwargs=True)
                    )
                    installed.add(key)
            if binding.outputs:
                key = (id(binding.module), binding.name, "outputs")
                if key not in installed:
                    self._handles.append(
                        binding.module.register_forward_hook(self._output_hook(binding.name), with_kwargs=True)
                    )
                    installed.add(key)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def inputs(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, cache in self.layer_cache.items():
            tensor = cache.inputs.tensor()
            if tensor is not None:
                result[name] = tensor
        return result

    def _input_hook(self, name: str):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            cache = self.layer_cache.setdefault(name, IOTensorsCache())
            if cache.replay_args is None:
                cache.replay_args = _to_cpu(args)
                cache.replay_kwargs = _to_cpu(kwargs)
            cache.inputs.add((args, kwargs), max_rows=self._max_rows, element_size=self._element_size)

        return hook

    def _output_hook(self, name: str):
        def hook(_module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
            self.layer_cache.setdefault(name, IOTensorsCache()).outputs.add(
                output,
                max_rows=self._max_rows,
                element_size=self._element_size,
            )

        return hook

    @staticmethod
    def _target_bindings(targets: list[QuantTarget]) -> list[CaptureBinding]:
        bindings = []
        for target in targets:
            if target.modules:
                bindings.append(CaptureBinding(name=target.export_name, module=target.modules[0], inputs=True))
        return bindings


class _EvalReplayCapture:
    def __init__(self, module: nn.Module | None, module_name: str | None, max_rows: int) -> None:
        self._module = module
        self._module_name = module_name
        self._max_rows = max_rows
        self._rows = 0
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._replay: EvalReplayBatch | None = None

    def install(self) -> None:
        if self._module is None:
            return
        self._handle = self._module.register_forward_hook(self._hook, with_kwargs=True)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def replay(self) -> EvalReplayBatch | None:
        return self._replay

    def _hook(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        if self._replay is not None or self._rows >= self._max_rows:
            return
        rows = _first_tensor_rows(args, kwargs)
        if rows <= 0:
            return
        self._rows += min(rows, self._max_rows - self._rows)
        self._replay = EvalReplayBatch(
            module=module,
            module_name=self._module_name or "",
            args=_to_cpu(args),
            kwargs=_to_cpu(kwargs),
            output=_to_cpu(output),
        )


def _expand_capture_rules(
    rules: Sequence[CalibrationCaptureRule],
    capture: tuple[str, ...],
    modules: dict[str, nn.Module],
) -> tuple[CaptureBinding, ...]:
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
                )
            )
    return tuple(bindings)


def _resolve_module_template(template: str, capture: tuple[str, ...], modules: dict[str, nn.Module]) -> str:
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


def _run_replay_module(module: nn.Module, value: Any, device: torch.device) -> None:
    value = _to_device(value, device)
    if torch.is_tensor(value):
        module(value)
    elif isinstance(value, tuple):
        module(*value)
    elif isinstance(value, list):
        module(*value)
    elif isinstance(value, dict):
        module(**value)
    else:
        raise TypeError(f"Cannot replay previous scope output of type {type(value).__name__}")


def _first_cached_output(layer_cache: dict[str, IOTensorsCache]) -> Any | None:
    for cache in layer_cache.values():
        tensor = cache.outputs.tensor()
        if tensor is not None:
            return tensor
    return None


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


def _first_tensor(value: Any) -> torch.Tensor | None:
    stack = [value]
    while stack:
        item = stack.pop(0)
        if torch.is_tensor(item):
            return item.detach()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return None


def _tensor_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    return tensor.detach().reshape(-1, tensor.shape[-1])


def _first_tensor_rows(*values: Any) -> int:
    stack = list(values)
    while stack:
        value = stack.pop(0)
        if torch.is_tensor(value):
            return _tensor_rows(value).shape[0]
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return 0


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
