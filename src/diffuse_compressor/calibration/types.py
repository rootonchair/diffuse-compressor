from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from ..targets import QuantTarget
from .cache import IOTensorsCache
from .data import ModuleForwardInput


@dataclass(frozen=True)
class EvalReplayBatch:
    """Captured eval-module replay sample for low-rank search.

    Args:
        module: Module to replay for objective scoring.
        module_name: Fully qualified module name.
        args: CPU positional arguments for replay.
        kwargs: CPU keyword arguments for replay.
        output: CPU reference output captured from the original forward pass.
    """

    module: nn.Module
    module_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: Any


@dataclass(frozen=True)
class ScopeReplayState:
    """Outputs retained from the previous calibration scope.

    Args:
        outputs: CPU outputs captured from the previous scope.
        replays: CPU eval replay records captured from the previous scope.
    """

    outputs: tuple[Any, ...] = ()
    replays: tuple[EvalReplayBatch, ...] = ()

    def forward_inputs(
        self,
        transform: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
        replay_transform: Callable[[EvalReplayBatch], tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
    ) -> tuple[ModuleForwardInput, ...]:
        """Convert retained outputs into replayable forward inputs."""

        if replay_transform is not None:
            return tuple(_prev_replay_to_forward_input(replay, replay_transform) for replay in self.replays)
        return tuple(_prev_output_to_forward_input(output, transform) for output in self.outputs)


@dataclass(frozen=True)
class CalibrationScope:
    """Concrete calibration scope containing targets and replay modules."""

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
    prev_replay_transform: Callable[[EvalReplayBatch], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    use_prev_scope_outputs: bool = True
    recompute: bool = False


@dataclass(frozen=True)
class CalibrationScopeBatch:
    """Calibration data captured for one scope."""

    scope: CalibrationScope
    inputs: dict[str, torch.Tensor]
    input_partitions: dict[str, tuple[torch.Tensor, ...]] = field(default_factory=dict)
    layer_cache: dict[str, IOTensorsCache] = field(default_factory=dict)
    eval_replay: EvalReplayBatch | None = None
    eval_replays: tuple[EvalReplayBatch, ...] = ()
    scope_target_count: int | None = None


@dataclass(frozen=True)
class CaptureBinding:
    """Bind a module to an input/output cache name."""

    name: str
    module: nn.Module
    inputs: bool = True
    outputs: bool = False
    input_keys: tuple[str | int, ...] = ()
    output_keys: tuple[str | int, ...] = ()
    channel_dim: int = -1


def _prev_replay_to_forward_input(
    replay: EvalReplayBatch, transform: Callable[[EvalReplayBatch], tuple[tuple[Any, ...], dict[str, Any]]]
) -> ModuleForwardInput:
    """Convert a previous eval replay record into replay inputs."""

    args, kwargs = transform(replay)
    return ModuleForwardInput(args=tuple(args), kwargs=dict(kwargs))


def _prev_output_to_forward_input(
    output: Any, transform: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
) -> ModuleForwardInput:
    """Convert a previous scope output into replay inputs."""

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
