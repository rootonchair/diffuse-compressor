from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ..targets import QuantTarget
from .cache import IOTensorsCache, mix_seed, TensorsCache
from .types import CaptureBinding, EvalReplayBatch
from .utils import filter_replay_inputs, first_tensor_rows, named_tensors, select_named_tensors, to_cpu


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
        seed: int | None = 0,
    ) -> None:
        """Initialize hook bindings and cache storage.

        ``seed`` drives the sampling of rows once ``max_rows`` is
        reached; each cache name mixes in its own hash.
        """

        self._bindings = self._target_bindings(targets, capture_outputs=capture_target_outputs) + list(captures)
        self._max_rows = max_rows
        self._input_stats_only = input_stats_only
        self._seed = seed
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layer_cache: dict[str, IOTensorsCache] = {}

    def install(self) -> None:
        """Register forward hooks for every configured binding."""

        installed: set[tuple[int, str, str]] = set()
        for binding in self._bindings:
            self._cache(binding.name)
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
        """Remove installed hooks and clear hook handles."""

        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def inputs(self, aliases: Mapping[str, str] | None = None) -> dict[str, torch.Tensor]:
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

        result = {name: cache.input_min for name, cache in self.layer_cache.items() if cache.input_min is not None}
        if aliases:
            for alias, source in aliases.items():
                if alias not in result and source in result:
                    result[alias] = result[source]
        return result

    def _input_hook(self, name: str):
        """Build a forward pre-hook that records module inputs."""

        def hook(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            binding = self._binding(name)
            cache = self._cache(name)
            if self._input_stats_only:
                selected = select_named_tensors(
                    named_tensors((args, kwargs)), () if binding is None else binding.input_keys
                )
                for _key, tensor in selected:
                    value = float(tensor.detach().amin().item())
                    cache.input_min = value if cache.input_min is None else min(cache.input_min, value)
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

        def hook(_module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any], output: Any) -> None:
            binding = self._binding(name)
            self._cache(name).outputs.add(
                output,
                max_rows=self._max_rows,
                keys=() if binding is None else binding.output_keys,
                channel_dim=-1 if binding is None else binding.channel_dim,
            )

        return hook

    def _cache(self, name: str) -> IOTensorsCache:
        """Return the cache for a name, creating it with seeded row sampling."""

        cache = self.layer_cache.get(name)
        if cache is None:
            cache = self.layer_cache[name] = IOTensorsCache(
                inputs=TensorsCache(seed=mix_seed(self._seed, f"{name}:inputs")),
                outputs=TensorsCache(seed=mix_seed(self._seed, f"{name}:outputs")),
            )
        return cache

    def _binding(self, name: str) -> CaptureBinding | None:
        """Find the binding registered for a cache name."""

        for binding in self._bindings:
            if binding.name == name:
                return binding
        return None

    @staticmethod
    def _target_bindings(targets: list[QuantTarget], *, capture_outputs: bool) -> list[CaptureBinding]:
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


class _CallSampler:
    """Seeded uniform sample of whole calls with a fixed or row-derived capacity."""

    def __init__(self, capacity: int | None, *, max_rows: int | None = None, seed: int | None = 0) -> None:
        self._capacity = capacity
        self._max_rows = max_rows
        self._seed = seed
        self._generator: torch.Generator | None = None

    def slot(self, call_index: int, rows: int) -> int | None:
        """Return the slot for call ``call_index`` (0-based), or ``None`` to skip it."""

        if self._capacity is None:
            if self._max_rows is None:
                return call_index
            self._capacity = max(1, self._max_rows // rows)
        if call_index < self._capacity:
            return call_index
        if self._generator is None:
            self._generator = torch.Generator()
            if self._seed is None:
                self._generator.seed()
            else:
                self._generator.manual_seed(self._seed)
        slot = int(torch.randint(call_index + 1, (1,), generator=self._generator).item())
        return slot if slot < self._capacity else None


class EvalReplayCapture:
    """Capture whole-call records of an eval module.

    Two independent stores are filled from the same forward hook:

    * Scoring records: full input/output records used to score low-rank
      candidates. Their count is ``max_score_records``, or ``max_rows`` divided by
      the row count of one call when unset, sampled across all calls.
    * Chain records: what the next chained scope replays. Every call is kept,
      matching DeepCompressor's layer-to-layer propagation. Input slots that the next scope's
      ``chain_transform`` overwrites with this module's output are stored as
      ``None``, so each call costs about one hidden state instead of two.
    """

    def __init__(
        self,
        module: nn.Module | None,
        module_name: str | None,
        max_rows: int | None,
        *,
        replay_arg_indices: Sequence[int] = (),
        replay_kwarg_keys: Sequence[str] = (),
        replay_transform: Callable[[tuple[Any, ...], dict[str, Any]], tuple[tuple[Any, ...], dict[str, Any]]]
        | None = None,
        seed: int | None = 0,
        score: bool = True,
        max_score_records: int | None = None,
        chain: bool = False,
        chain_transform: Callable[[EvalReplayBatch], tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
    ) -> None:
        """Initialize eval replay capture state."""

        self._module = module
        self._module_name = module_name
        self._replay_arg_indices = tuple(replay_arg_indices)
        self._replay_kwarg_keys = tuple(replay_kwarg_keys)
        self._replay_transform = replay_transform
        self._score = _CallSampler(max_score_records, max_rows=max_rows, seed=seed) if score else None
        self._chain = chain
        self._chain_transform = chain_transform
        self._carried: tuple[frozenset[int], frozenset[str]] | None = None
        self._num_calls = 0
        self._handle: torch.utils.hooks.RemovableHandle | None = None
        self._replays: list[EvalReplayBatch] = []
        self._chain_records: list[EvalReplayBatch] = []

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

    @property
    def num_calls(self) -> int:
        """Number of eval-module calls observed."""

        return self._num_calls

    def replay(self) -> EvalReplayBatch | None:
        """Return the first scoring record."""

        return self._replays[0] if self._replays else None

    def replays(self) -> tuple[EvalReplayBatch, ...]:
        """Return the scoring records (full inputs and outputs)."""

        return tuple(self._replays)

    def chain_records(self) -> list[EvalReplayBatch]:
        """Return the records handed to the next chained scope."""

        return self._chain_records

    def _hook(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any) -> None:
        """Capture one eval-module call into the scoring and chain stores."""

        rows = first_tensor_rows(args, kwargs)
        if rows <= 0:
            return
        call_index = self._num_calls
        self._num_calls += 1
        score_slot = None if self._score is None else self._score.slot(call_index, rows)
        chain_slot = len(self._chain_records) if self._chain else None
        if score_slot is None and chain_slot is None:
            return
        replay_args, replay_kwargs = filter_replay_inputs(
            args, kwargs, arg_indices=self._replay_arg_indices, kwarg_keys=self._replay_kwarg_keys
        )
        if self._replay_transform is not None:
            replay_args, replay_kwargs = self._replay_transform(replay_args, replay_kwargs)
        cpu_output = to_cpu(output)
        name = self._module_name or ""
        if score_slot is not None:
            cpu_args, cpu_kwargs = to_cpu(replay_args), to_cpu(replay_kwargs)
            _store(self._replays, score_slot, EvalReplayBatch(module, name, cpu_args, cpu_kwargs, cpu_output))
            if chain_slot is not None:
                # Reuse the CPU copies; only the carried slots differ.
                chain_args, chain_kwargs = self._strip_carried(module, cpu_args, cpu_kwargs, cpu_output)
                _store(self._chain_records, chain_slot, EvalReplayBatch(module, name, chain_args, chain_kwargs, cpu_output))
            return
        # Strip before copying so the carried input is never moved to CPU.
        chain_args, chain_kwargs = self._strip_carried(module, replay_args, replay_kwargs, output)
        _store(
            self._chain_records,
            chain_slot,
            EvalReplayBatch(module, name, to_cpu(chain_args), to_cpu(chain_kwargs), cpu_output),
        )

    def _strip_carried(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Replace input slots the next scope fills from ``output`` with ``None``."""

        if self._chain_transform is None:
            # The next scope rebuilds its inputs from outputs alone.
            return (), {}
        if self._carried is None:
            self._carried = _carried_slots(self._chain_transform, module, self._module_name or "", args, kwargs, output)
        arg_slots, kwarg_slots = self._carried
        stripped_args = tuple(None if index in arg_slots else value for index, value in enumerate(args))
        stripped_kwargs = {key: None if key in kwarg_slots else value for key, value in kwargs.items()}
        return stripped_args, stripped_kwargs


def _store(records: list[EvalReplayBatch], slot: int, record: EvalReplayBatch) -> None:
    if slot == len(records):
        records.append(record)
    else:
        records[slot] = record


def _carried_slots(
    transform: Callable[[EvalReplayBatch], tuple[tuple[Any, ...], dict[str, Any]]],
    module: nn.Module,
    module_name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    output: Any,
) -> tuple[frozenset[int], frozenset[str]]:
    """Find the input slots a chain transform overwrites with this module's output."""

    carried_values = [output, *(output if isinstance(output, (tuple, list)) else ())]

    def is_carried(value: Any) -> bool:
        return any(value is candidate for candidate in carried_values)

    next_args, next_kwargs = transform(EvalReplayBatch(module, module_name, args, kwargs, output))
    arg_slots = frozenset(index for index, value in enumerate(next_args) if index < len(args) and is_carried(value))
    kwarg_slots = frozenset(key for key, value in next_kwargs.items() if key in kwargs and is_carried(value))
    return arg_slots, kwarg_slots


def apply_cache_aliases(layer_cache: dict[str, IOTensorsCache], aliases: Mapping[str, str]) -> None:
    """Add alias keys for existing layer caches."""

    for alias, source in aliases.items():
        if alias in layer_cache or source not in layer_cache:
            continue
        layer_cache[alias] = layer_cache[source]
