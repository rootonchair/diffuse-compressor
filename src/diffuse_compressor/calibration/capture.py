from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ..targets import QuantTarget
from .cache import IOTensorsCache
from .types import CaptureBinding, EvalReplayBatch
from .utils import (
    filter_replay_inputs,
    first_tensor_rows,
    named_tensors,
    select_named_tensors,
    to_cpu,
)


class LayerCacheCapture:
    """Install hooks that capture target and auxiliary module I/O."""

    def __init__(
        self,
        targets: list[QuantTarget],
        captures: Sequence[CaptureBinding],
        *,
        max_rows: int | None,
        input_stats_only: bool = False,
        capture_target_outputs: bool = True,
    ) -> None:
        """Initialize hook bindings and cache storage."""

        self._bindings = self._target_bindings(
            targets,
            capture_outputs=capture_target_outputs,
        ) + list(captures)
        self._max_rows = max_rows
        self._input_stats_only = input_stats_only
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layer_cache: dict[str, IOTensorsCache] = {}

    def install(self) -> None:
        """Register forward hooks for every configured binding."""

        installed: set[tuple[int, str, str]] = set()
        for binding in self._bindings:
            self.layer_cache.setdefault(binding.name, IOTensorsCache())
            if binding.inputs:
                key = (id(binding.module), binding.name, "inputs")
                if key not in installed:
                    self._handles.append(
                        binding.module.register_forward_pre_hook(
                            self._input_hook(binding.name), with_kwargs=True
                        )
                    )
                    installed.add(key)
            if binding.outputs:
                key = (id(binding.module), binding.name, "outputs")
                if key not in installed:
                    self._handles.append(
                        binding.module.register_forward_hook(
                            self._output_hook(binding.name), with_kwargs=True
                        )
                    )
                    installed.add(key)

    def remove(self) -> None:
        """Remove installed hooks and clear hook handles."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def inputs(
        self, aliases: Mapping[str, str] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return captured input tensors, optionally applying aliases."""

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

    def input_mins(self, aliases: Mapping[str, str] | None = None) -> dict[str, float]:
        """Return streamed input minima, optionally applying aliases."""

        result = {
            name: cache.input_min
            for name, cache in self.layer_cache.items()
            if cache.input_min is not None
        }
        if aliases:
            for alias, source in aliases.items():
                if alias not in result and source in result:
                    result[alias] = result[source]
        return result

    def _input_hook(self, name: str):
        """Build a forward pre-hook that records module inputs."""

        def hook(
            _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> None:
            binding = self._binding(name)
            cache = self.layer_cache.setdefault(name, IOTensorsCache())
            if self._input_stats_only:
                selected = select_named_tensors(
                    named_tensors((args, kwargs)),
                    () if binding is None else binding.input_keys,
                )
                for _key, tensor in selected:
                    value = float(tensor.detach().amin().item())
                    cache.input_min = (
                        value
                        if cache.input_min is None
                        else min(cache.input_min, value)
                    )
                return
            if cache.replay_args is None:
                cache.replay_args = to_cpu(args)
                cache.replay_kwargs = to_cpu(kwargs)
            cache.inputs.add(
                (args, kwargs),
                max_rows=self._max_rows,
                keys=() if binding is None else binding.input_keys,
                channel_dim=-1 if binding is None else binding.channel_dim,
            )

        return hook

    def _output_hook(self, name: str):
        """Build a forward hook that records module outputs."""

        def hook(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            binding = self._binding(name)
            self.layer_cache.setdefault(name, IOTensorsCache()).outputs.add(
                output,
                max_rows=self._max_rows,
                keys=() if binding is None else binding.output_keys,
                channel_dim=-1 if binding is None else binding.channel_dim,
            )

        return hook

    def _binding(self, name: str) -> CaptureBinding | None:
        """Find the binding registered for a cache name."""

        for binding in self._bindings:
            if binding.name == name:
                return binding
        return None

    @staticmethod
    def _target_bindings(
        targets: list[QuantTarget], *, capture_outputs: bool
    ) -> list[CaptureBinding]:
        """Create default capture bindings for target modules."""

        bindings = []
        for target in targets:
            if target.modules:
                channel_dim = 1 if target.kind == "conv" else -1
                bindings.append(
                    CaptureBinding(
                        name=target.export_name,
                        module=target.modules[0],
                        inputs=True,
                        outputs=capture_outputs,
                        channel_dim=channel_dim,
                    )
                )
                for module in target.modules[1:]:
                    bindings.append(
                        CaptureBinding(
                            name=target.export_name,
                            module=module,
                            inputs=False,
                            outputs=capture_outputs,
                            channel_dim=channel_dim,
                        )
                    )
        return bindings


class EvalReplayCapture:
    """Capture eval-module replay records for objective scoring."""

    def __init__(
        self,
        module: nn.Module | None,
        module_name: str | None,
        max_rows: int | None,
        *,
        replay_arg_indices: Sequence[int] = (),
        replay_kwarg_keys: Sequence[str] = (),
        replay_transform: Callable[
            [tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]
        ]
        | None = None,
    ) -> None:
        """Initialize eval replay capture state."""

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
        """Register the eval-module forward hook when a module is configured."""

        if self._module is None:
            return
        self._handle = self._module.register_forward_hook(self._hook, with_kwargs=True)

    def remove(self) -> None:
        """Remove the eval replay hook if it is installed."""

        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def replay(self) -> EvalReplayBatch | None:
        """Return the first captured replay record."""

        return self._replays[0] if self._replays else None

    def replays(self) -> tuple[EvalReplayBatch, ...]:
        """Return all captured replay records."""

        return tuple(self._replays)

    def _hook(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        """Capture one eval-module input/output pair."""

        if self._max_rows is not None and self._rows >= self._max_rows:
            return
        rows = first_tensor_rows(args, kwargs)
        if rows <= 0:
            return
        self._rows += (
            rows if self._max_rows is None else min(rows, self._max_rows - self._rows)
        )
        replay_args, replay_kwargs = filter_replay_inputs(
            args,
            kwargs,
            arg_indices=self._replay_arg_indices,
            kwarg_keys=self._replay_kwarg_keys,
        )
        if self._replay_transform is not None:
            replay_args, replay_kwargs = self._replay_transform(
                replay_args, replay_kwargs
            )
        self._replays.append(
            EvalReplayBatch(
                module=module,
                module_name=self._module_name or "",
                args=to_cpu(replay_args),
                kwargs=to_cpu(replay_kwargs),
                output=to_cpu(output),
            )
        )


def apply_cache_aliases(
    layer_cache: dict[str, IOTensorsCache], aliases: Mapping[str, str]
) -> None:
    """Add alias keys for existing layer caches."""

    for alias, source in aliases.items():
        if alias in layer_cache or source not in layer_cache:
            continue
        layer_cache[alias] = layer_cache[source]


def first_cached_output(layer_cache: dict[str, IOTensorsCache]) -> Any | None:
    """Return the first available cached output tensor."""

    for cache in layer_cache.values():
        tensor = cache.outputs.tensor()
        if tensor is not None:
            return tensor
    return None
