from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .utils import (
    first_tensor,
    named_tensors,
    repartition_tensor,
    sample_count,
    select_named_tensors,
    tensor_rows,
)


@dataclass
class TensorCache:
    data: list[torch.Tensor] = field(default_factory=list)
    num_rows: int = 0
    num_total: int = 0
    num_samples: int = 0
    orig_device: torch.device | None = None
    channel_dim: int = -1

    def add(self, value: Any, *, max_rows: int, element_size: int = -1, channel_dim: int | None = None) -> None:
        tensor = first_tensor(value)
        if tensor is None:
            return
        if self.orig_device is None:
            self.orig_device = tensor.device
        self.num_samples += sample_count(tensor)
        rows = tensor_rows(tensor, self.channel_dim if channel_dim is None else channel_dim)
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
        selected = select_named_tensors(named_tensors(value), keys)
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
