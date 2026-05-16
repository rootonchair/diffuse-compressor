from __future__ import annotations

import gc
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
import torch.nn as nn

from ..config import CalibrationCaptureRule, CalibrationSpec, TargetConfig
from ..targets import QuantTarget, _capture_sort_key, _format_export_name, _match_pattern
from .cache import IOTensorsCache
from .data import (
    ModuleForwardInput,
    has_runnable_calibration,
    iter_calibration_forward_inputs,
    prepare_calibration_cache,
    resolve_samples,
    run_forward_input,
)
from .utils import (
    check_ram,
    filter_replay_inputs,
    first_tensor_rows,
    is_under_scope,
    model_device,
    repartition_tensor,
    to_cpu,
    to_device,
)


@dataclass(frozen=True)
class EvalReplayBatch:
    module: nn.Module
    module_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any


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
    samples = resolve_samples(calibration) if calibration.cache_mode == "disabled" or not cache_paths else []
    device = model_device(model)
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
                    check_ram(calibration)
            elif cache_paths:
                for forward_input in iter_calibration_forward_inputs(calibration, cache_paths=cache_paths):
                    replay = forward_input.to(device)
                    model(*replay.args, **replay.kwargs)
                    check_ram(calibration)
            else:
                for forward_input in iter_calibration_forward_inputs(calibration, samples=samples):
                    run_forward_input(model, calibration, forward_input.to(device))
                    check_ram(calibration)
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
                cache.replay_args = to_cpu(args)
                cache.replay_kwargs = to_cpu(kwargs)
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
        rows = first_tensor_rows(args, kwargs)
        if rows <= 0:
            return
        self._rows += min(rows, self._max_rows - self._rows)
        replay_args, replay_kwargs = filter_replay_inputs(
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
                args=to_cpu(replay_args),
                kwargs=to_cpu(replay_kwargs),
                output=to_cpu(output),
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
