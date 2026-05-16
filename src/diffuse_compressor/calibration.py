from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .config import CalibrationCaptureRule, CalibrationSpec, TargetConfig
from .targets import QuantTarget, _capture_sort_key, _format_export_name, _match_pattern


@dataclass
class TensorCache:
    data: list[torch.Tensor] = field(default_factory=list)
    num_rows: int = 0
    num_total: int = 0
    num_samples: int = 0
    orig_device: torch.device | None = None
    channel_dim: int = -1

    def add(self, value: Any, *, max_rows: int, element_size: int = -1, channel_dim: int | None = None) -> None:
        tensor = _first_tensor(value)
        if tensor is None:
            return
        if self.orig_device is None:
            self.orig_device = tensor.device
        self.num_samples += _sample_count(tensor)
        rows = _tensor_rows(tensor, self.channel_dim if channel_dim is None else channel_dim)
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
        self.num_samples = 0

    def repartition(
        self,
        *,
        sample_size: int = -1,
        sample_batch_size: int = -1,
        element_size: int = -1,
        element_batch_size: int = -1,
    ) -> tuple[torch.Tensor, ...]:
        tensor = self.tensor()
        if tensor is None:
            return ()
        return repartition_tensor(
            tensor,
            sample_size=sample_size,
            sample_batch_size=sample_batch_size,
            element_size=element_size,
            element_batch_size=element_batch_size,
        )


@dataclass
class TensorsCache:
    tensors: dict[str, TensorCache] = field(default_factory=dict)
    primary_key: str | None = None
    num_samples: int = 0

    def add(
        self,
        value: Any,
        *,
        max_rows: int,
        element_size: int = -1,
        keys: Sequence[str | int] = (),
        channel_dim: int = -1,
    ) -> None:
        named = _named_tensors(value)
        selected = _select_named_tensors(named, keys)
        if not selected:
            return
        self.num_samples += 1
        for key, tensor in selected:
            str_key = str(key)
            if self.primary_key is None:
                self.primary_key = str_key
            cache = self.tensors.setdefault(str_key, TensorCache(channel_dim=channel_dim))
            cache.add(tensor, max_rows=max_rows, element_size=element_size, channel_dim=channel_dim)

    def tensor(self, key: str | int | None = None) -> torch.Tensor | None:
        cache = self._cache(key)
        return None if cache is None else cache.tensor()

    def keyed_tensors(self) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for key, cache in self.tensors.items():
            tensor = cache.tensor()
            if tensor is not None:
                result[key] = tensor
        return result

    def clear(self) -> None:
        for cache in self.tensors.values():
            cache.clear()
        self.tensors.clear()
        self.primary_key = None
        self.num_samples = 0

    def alias(self, alias: str, source: str) -> None:
        if source in self.tensors:
            self.tensors[alias] = self.tensors[source]
            if self.primary_key is None:
                self.primary_key = alias

    def repartition(
        self,
        key: str | int | None = None,
        *,
        sample_size: int = -1,
        sample_batch_size: int = -1,
        element_size: int = -1,
        element_batch_size: int = -1,
    ) -> tuple[torch.Tensor, ...]:
        cache = self._cache(key)
        if cache is None:
            return ()
        return cache.repartition(
            sample_size=sample_size,
            sample_batch_size=sample_batch_size,
            element_size=element_size,
            element_batch_size=element_batch_size,
        )

    def _cache(self, key: str | int | None) -> TensorCache | None:
        if key is None:
            key = self.primary_key
        if key is None:
            return None
        return self.tensors.get(str(key))


@dataclass
class IOTensorsCache:
    inputs: TensorsCache = field(default_factory=TensorsCache)
    outputs: TensorsCache = field(default_factory=TensorsCache)
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
class ModuleForwardInput:
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device) -> "ModuleForwardInput":
        return ModuleForwardInput(args=_to_device(self.args, device), kwargs=_to_device(self.kwargs, device))


class CalibrationCacheDataset(Dataset[ModuleForwardInput]):
    def __init__(self, paths: Sequence[Path], *, eager_load: bool = False) -> None:
        self.paths = list(paths)
        self.items = [_load_cached_forward_input(path) for path in self.paths] if eager_load else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> ModuleForwardInput:
        if self.items is not None:
            return self.items[index]
        return _load_cached_forward_input(self.paths[index])


class CalibrationSampleDataset(Dataset[ModuleForwardInput]):
    def __init__(self, samples: Sequence[dict[str, Any]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ModuleForwardInput:
        return _sample_to_forward_input(self.samples[index])


@dataclass(frozen=True)
class ScopeReplayState:
    outputs: tuple[Any, ...] = ()

    def forward_inputs(
        self,
        transform: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
    ) -> tuple[ModuleForwardInput, ...]:
        return tuple(_prev_output_to_forward_input(output, transform) for output in self.outputs)


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
    cache_aliases: Mapping[str, str] = field(default_factory=dict)
    replay_arg_indices: tuple[int, ...] = ()
    replay_kwarg_keys: tuple[str, ...] = ()
    replay_transform: Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    prev_output_transform: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    use_prev_scope_outputs: bool = False
    recompute: bool = False


@dataclass(frozen=True)
class CalibrationScopeBatch:
    scope: CalibrationScope
    inputs: dict[str, torch.Tensor]
    input_partitions: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    layer_cache: dict[str, IOTensorsCache] = field(default_factory=dict)
    eval_replay: EvalReplayBatch | None = None
    eval_replays: tuple[EvalReplayBatch, ...] = ()


@dataclass(frozen=True)
class CaptureBinding:
    name: str
    module: nn.Module
    inputs: bool = True
    outputs: bool = False
    input_keys: tuple[str | int, ...] = ()
    output_keys: tuple[str | int, ...] = ()
    channel_dim: int = -1


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
        for forward_input in _iter_calibration_forward_inputs(calibration, samples=samples, batch_size=1, drop_last=False):
            _run_forward_input(model, calibration, forward_input)
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
    prev_scope_state = ScopeReplayState()

    for scope in scopes:
        capture = _LayerCacheCapture(
            list(scope.targets),
            scope.captures,
            max_rows=calibration.max_rows_per_target,
            element_size=calibration.element_size,
        )
        eval_capture = _EvalReplayCapture(
            scope.eval_module,
            scope.eval_module_name,
            calibration.max_rows_per_target,
            replay_arg_indices=scope.replay_arg_indices,
            replay_kwarg_keys=scope.replay_kwarg_keys,
            replay_transform=scope.replay_transform,
        )
        capture.install()
        eval_capture.install()
        try:
            if (
                scope.use_prev_scope_outputs
                and prev_scope_state.outputs
                and not scope.recompute
                and scope.replay_module is not None
            ):
                for forward_input in prev_scope_state.forward_inputs(scope.prev_output_transform):
                    _run_module_forward_input(scope.replay_module, forward_input.to(device))
                    _check_ram(calibration)
            elif cache_paths:
                for forward_input in _iter_calibration_forward_inputs(calibration, cache_paths=cache_paths):
                    replay = forward_input.to(device)
                    model(*replay.args, **replay.kwargs)
                    _check_ram(calibration)
            else:
                for forward_input in _iter_calibration_forward_inputs(calibration, samples=samples):
                    _run_forward_input(model, calibration, forward_input.to(device))
                    _check_ram(calibration)
        finally:
            eval_capture.remove()
            capture.remove()
        _apply_cache_aliases(capture.layer_cache, scope.cache_aliases)
        inputs = capture.inputs(scope.cache_aliases)
        input_partitions = {
            name: repartition_tensor(
                tensor,
                sample_size=calibration.sample_size,
                sample_batch_size=calibration.sample_batch_size,
                element_size=calibration.element_size,
                element_batch_size=calibration.element_batch_size,
            )
            for name, tensor in inputs.items()
        }
        layer_cache = capture.layer_cache
        eval_replay = eval_capture.replay()
        eval_replays = eval_capture.replays()
        yield CalibrationScopeBatch(
            scope=scope,
            inputs=inputs,
            input_partitions=input_partitions,
            layer_cache=layer_cache,
            eval_replay=eval_replay,
            eval_replays=eval_replays,
        )
        prev_outputs = tuple(replay.output for replay in eval_replays)
        if not prev_outputs:
            cached_output = _first_cached_output(layer_cache)
            prev_outputs = () if cached_output is None else (cached_output,)
        prev_scope_state = ScopeReplayState(outputs=prev_outputs)
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
            bool,
            bool,
        ]
    ] = []
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
                use_prev,
                recompute,
            ) in concrete_scopes
            if any(_is_under_scope(module_name, scope_module) for module_name in target.module_names)
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
                use_prev_scope_outputs=use_prev,
                recompute=recompute,
            )
        )
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

    def inputs(self, aliases: Mapping[str, str] | None = None) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, cache in self.layer_cache.items():
            tensor = cache.inputs.tensor()
            if tensor is not None:
                result[name] = tensor
        if aliases:
            for alias, source in aliases.items():
                if alias not in result and source in result:
                    result[alias] = result[source]
        return result

    def _input_hook(self, name: str):
        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            binding = self._binding(name)
            cache = self.layer_cache.setdefault(name, IOTensorsCache())
            if cache.replay_args is None:
                cache.replay_args = _to_cpu(args)
                cache.replay_kwargs = _to_cpu(kwargs)
            cache.inputs.add(
                (args, kwargs),
                max_rows=self._max_rows,
                element_size=self._element_size,
                keys=() if binding is None else binding.input_keys,
                channel_dim=-1 if binding is None else binding.channel_dim,
            )

        return hook

    def _output_hook(self, name: str):
        def hook(_module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
            binding = self._binding(name)
            self.layer_cache.setdefault(name, IOTensorsCache()).outputs.add(
                output,
                max_rows=self._max_rows,
                element_size=self._element_size,
                keys=() if binding is None else binding.output_keys,
                channel_dim=-1 if binding is None else binding.channel_dim,
            )

        return hook

    def _binding(self, name: str) -> CaptureBinding | None:
        for binding in self._bindings:
            if binding.name == name:
                return binding
        return None

    @staticmethod
    def _target_bindings(targets: list[QuantTarget]) -> list[CaptureBinding]:
        bindings = []
        for target in targets:
            if target.modules:
                bindings.append(CaptureBinding(name=target.export_name, module=target.modules[0], inputs=True, outputs=True))
                for module in target.modules[1:]:
                    bindings.append(CaptureBinding(name=target.export_name, module=module, inputs=False, outputs=True))
        return bindings


class _EvalReplayCapture:
    def __init__(
        self,
        module: nn.Module | None,
        module_name: str | None,
        max_rows: int,
        *,
        replay_arg_indices: Sequence[int] = (),
        replay_kwarg_keys: Sequence[str] = (),
        replay_transform: Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
    ) -> None:
        self._module = module
        self._module_name = module_name
        self._max_rows = max_rows
        self._replay_arg_indices = tuple(replay_arg_indices)
        self._replay_kwarg_keys = tuple(replay_kwarg_keys)
        self._replay_transform = replay_transform
        self._rows = 0
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._replays: list[EvalReplayBatch] = []

    def install(self) -> None:
        if self._module is None:
            return
        self._handle = self._module.register_forward_hook(self._hook, with_kwargs=True)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def replay(self) -> EvalReplayBatch | None:
        return self._replays[0] if self._replays else None

    def replays(self) -> tuple[EvalReplayBatch, ...]:
        return tuple(self._replays)

    def _hook(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        if self._rows >= self._max_rows:
            return
        rows = _first_tensor_rows(args, kwargs)
        if rows <= 0:
            return
        self._rows += min(rows, self._max_rows - self._rows)
        replay_args, replay_kwargs = _filter_replay_inputs(
            args,
            kwargs,
            arg_indices=self._replay_arg_indices,
            kwarg_keys=self._replay_kwarg_keys,
        )
        if self._replay_transform is not None:
            replay_args, replay_kwargs = self._replay_transform(replay_args, replay_kwargs)
        self._replays.append(
            EvalReplayBatch(
                module=module,
                module_name=self._module_name or "",
                args=_to_cpu(replay_args),
                kwargs=_to_cpu(replay_kwargs),
                output=_to_cpu(output),
            )
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
                    input_keys=tuple(rule.input_keys),
                    output_keys=tuple(rule.output_keys),
                    channel_dim=rule.channel_dim,
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


def _run_module_forward_input(module: nn.Module, forward_input: ModuleForwardInput) -> Any:
    return module(*forward_input.args, **forward_input.kwargs)


def _prev_output_to_forward_input(
    output: Any,
    transform: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
) -> ModuleForwardInput:
    if transform is not None:
        args, kwargs = transform(output)
        return ModuleForwardInput(args=tuple(args), kwargs=dict(kwargs))
    if torch.is_tensor(output):
        return ModuleForwardInput(args=(output,))
    if isinstance(output, tuple):
        return ModuleForwardInput(args=output)
    if isinstance(output, list):
        return ModuleForwardInput(args=tuple(output))
    if isinstance(output, dict):
        return ModuleForwardInput(kwargs=output)
    raise TypeError(f"Cannot replay previous scope output of type {type(output).__name__}")


def _apply_cache_aliases(layer_cache: dict[str, IOTensorsCache], aliases: Mapping[str, str]) -> None:
    for alias, source in aliases.items():
        if alias in layer_cache or source not in layer_cache:
            continue
        layer_cache[alias] = layer_cache[source]


def _first_cached_output(layer_cache: dict[str, IOTensorsCache]) -> Any | None:
    for cache in layer_cache.values():
        tensor = cache.outputs.tensor()
        if tensor is not None:
            return tensor
    return None


def _iter_calibration_forward_inputs(
    calibration: CalibrationSpec,
    *,
    cache_paths: Sequence[Path] | None = None,
    samples: Sequence[dict[str, Any]] | None = None,
    batch_size: int | None = None,
    drop_last: bool | None = None,
) -> Iterator[ModuleForwardInput]:
    if cache_paths is not None:
        dataset: Dataset[ModuleForwardInput] = CalibrationCacheDataset(
            cache_paths,
            eager_load=calibration.eager_load_samples,
        )
    else:
        dataset = CalibrationSampleDataset(samples or ())

    batch_size = calibration.batch_size if batch_size is None else batch_size
    generator = None
    if calibration.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(calibration.seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=calibration.shuffle,
        drop_last=calibration.drop_last if drop_last is None else drop_last,
        num_workers=calibration.num_workers,
        collate_fn=_batch_forward_inputs,
        generator=generator,
    )
    yield from loader


def _load_cached_forward_input(path: Path) -> ModuleForwardInput:
    item = torch.load(path, map_location="cpu", weights_only=False)
    return ModuleForwardInput(args=tuple(item.get("args", ())), kwargs=dict(item.get("kwargs", {})))


def _sample_to_forward_input(sample: dict[str, Any]) -> ModuleForwardInput:
    return ModuleForwardInput(kwargs=dict(sample))


def _batch_forward_inputs(inputs: Sequence[ModuleForwardInput]) -> ModuleForwardInput:
    if len(inputs) == 1:
        return inputs[0]
    args = _batch_sequence([item.args for item in inputs])
    kwargs = _batch_mapping([item.kwargs for item in inputs])
    return ModuleForwardInput(args=args, kwargs=kwargs)


def _batch_sequence(values: Sequence[tuple[Any, ...]]) -> tuple[Any, ...]:
    if not values or any(len(value) != len(values[0]) for value in values):
        return tuple(values[0]) if values else ()
    return tuple(_batch_values([value[index] for value in values]) for index in range(len(values[0])))


def _batch_mapping(values: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        return {}
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        return dict(values[0])
    return {key: _batch_values([value[key] for value in values]) for key in values[0]}


def _batch_values(values: Sequence[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if all(torch.is_tensor(value) and value.shape == first.shape for value in values):
        return torch.cat([value for value in values], dim=0)
    if all(isinstance(value, dict) for value in values):
        return _batch_mapping(values)  # type: ignore[arg-type]
    if all(isinstance(value, tuple) and len(value) == len(first) for value in values):
        return _batch_sequence(values)  # type: ignore[arg-type]
    return list(values)


def _run_forward_input(model: nn.Module, calibration: CalibrationSpec, forward_input: ModuleForwardInput) -> None:
    if calibration.forward_fn is not None:
        sample = dict(forward_input.kwargs)
        if forward_input.args:
            sample["__args__"] = forward_input.args
        calibration.forward_fn(sample)
    else:
        model(*forward_input.args, **forward_input.kwargs)


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


def _named_tensors(value: Any) -> list[tuple[str, torch.Tensor]]:
    if _is_forward_pair(value):
        args, kwargs = value
        result: list[tuple[str, torch.Tensor]] = []
        for index, item in enumerate(args):
            result.extend((f"arg{index}" if key == "" else f"arg{index}.{key}", tensor) for key, tensor in _flatten_named_tensors(item))
        for key, item in kwargs.items():
            result.extend((str(key) if nested == "" else f"{key}.{nested}", tensor) for nested, tensor in _flatten_named_tensors(item))
        return result
    return _flatten_named_tensors(value)


def _flatten_named_tensors(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(prefix, value.detach())]
    if isinstance(value, dict):
        result: list[tuple[str, torch.Tensor]] = []
        for key, item in value.items():
            nested = str(key) if prefix == "" else f"{prefix}.{key}"
            result.extend(_flatten_named_tensors(item, nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            nested = str(index) if prefix == "" else f"{prefix}.{index}"
            result.extend(_flatten_named_tensors(item, nested))
        return result
    return []


def _select_named_tensors(
    named: Sequence[tuple[str, torch.Tensor]],
    keys: Sequence[str | int],
) -> list[tuple[str, torch.Tensor]]:
    if not keys:
        return list(named[:1])
    wanted = {f"arg{key}" if isinstance(key, int) else str(key) for key in keys}
    return [(key, tensor) for key, tensor in named if key in wanted]


def _is_forward_pair(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], tuple)
        and isinstance(value[1], dict)
    )


def _filter_replay_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    arg_indices: Sequence[int],
    kwarg_keys: Sequence[str],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    replay_args = tuple(args[index] for index in arg_indices if -len(args) <= index < len(args)) if arg_indices else args
    replay_kwargs = {key: kwargs[key] for key in kwarg_keys if key in kwargs} if kwarg_keys else kwargs
    return replay_args, replay_kwargs


def _tensor_rows(tensor: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(-1, 1)
    tensor = tensor.detach()
    if channel_dim < 0:
        channel_dim += tensor.ndim
    if channel_dim != tensor.ndim - 1:
        tensor = tensor.movedim(channel_dim, -1)
    return tensor.reshape(-1, tensor.shape[-1])


def _sample_count(tensor: torch.Tensor) -> int:
    return 1 if tensor.ndim == 0 else int(tensor.shape[0])


def repartition_tensor(
    tensor: torch.Tensor,
    *,
    sample_size: int = -1,
    sample_batch_size: int = -1,
    element_size: int = -1,
    element_batch_size: int = -1,
) -> tuple[torch.Tensor, ...]:
    rows = tensor.reshape(-1, tensor.shape[-1])
    limit = rows.shape[0]
    if sample_size > 0:
        limit = min(limit, sample_size)
    if element_size > 0:
        limit = min(limit, element_size)
    rows = rows[:limit]
    batch_size = element_batch_size if element_batch_size > 0 else sample_batch_size
    if batch_size <= 0:
        return (rows,)
    return tuple(rows[index : index + batch_size] for index in range(0, rows.shape[0], batch_size))


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
