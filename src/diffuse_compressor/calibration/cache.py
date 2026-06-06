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
    """CPU cache for flattened rows from one tensor stream.

    Args:
        data: Cached CPU row tensors.
        num_rows: Number of retained rows.
        num_total: Number of rows observed before truncation.
        num_samples: Number of samples observed.
        orig_device: Device of the first observed tensor.
        channel_dim: Dimension treated as the channel/feature axis.
    """

    data: list[torch.Tensor] = field(default_factory=list)
    num_rows: int = 0
    num_total: int = 0
    num_samples: int = 0
    orig_device: torch.device | None = None
    channel_dim: int = -1

    def add(
        self, value: Any, *, max_rows: int | None, channel_dim: int | None = None
    ) -> None:
        """Append rows from the first tensor contained in a value.

        Args:
            value: Tensor or nested structure containing a tensor.
            max_rows: Optional maximum retained rows across the cache.
            channel_dim: Optional channel dimension override.
        """

        tensor = first_tensor(value)
        if tensor is None:
            return
        if self.orig_device is None:
            self.orig_device = tensor.device
        self.num_samples += sample_count(tensor)
        rows = tensor_rows(
            tensor, self.channel_dim if channel_dim is None else channel_dim
        )
        self.num_total += rows.shape[0]
        if max_rows is not None:
            if self.num_rows >= max_rows:
                return
            remaining = max_rows - self.num_rows
            if rows.shape[0] > remaining:
                rows = rows[:remaining]
        self.data.append(rows.float().cpu())
        self.num_rows += rows.shape[0]

    def tensor(self) -> torch.Tensor | None:
        """Return cached rows as one tensor.

        Returns:
            Concatenated CPU tensor, or ``None`` when the cache is empty.
        """

        if not self.data:
            return None
        return torch.cat(self.data, dim=0)

    def clear(self) -> None:
        """Release cached tensors and reset retained sample counters."""

        self.data.clear()
        self.num_rows = 0
        self.num_samples = 0

    def repartition(
        self,
        *,
        sample_size: int = -1,
        sample_batch_size: int = -1,
    ) -> tuple[torch.Tensor, ...]:
        """Split cached rows into bounded partitions.

        Args:
            sample_size: Maximum sample rows to keep, or ``-1`` for all.
            sample_batch_size: Partition size from sample batching.

        Returns:
            Tuple of CPU row tensors.
        """

        tensor = self.tensor()
        if tensor is None:
            return ()
        return repartition_tensor(
            tensor,
            sample_size=sample_size,
            sample_batch_size=sample_batch_size,
        )


@dataclass
class TensorsCache:
    """Named collection of tensor row caches for structured module I/O.

    Args:
        tensors: Mapping from tensor key to its row cache.
        primary_key: Default key used when a caller does not request one.
        num_samples: Number of structured values added.
    """

    tensors: dict[str, TensorCache] = field(default_factory=dict)
    primary_key: str | None = None
    num_samples: int = 0

    def add(
        self,
        value: Any,
        *,
        max_rows: int | None,
        keys: Sequence[str | int] = (),
        channel_dim: int = -1,
    ) -> None:
        """Add selected tensors from a structured value.

        Args:
            value: Tensor, forward ``(args, kwargs)`` pair, or nested structure.
            max_rows: Optional maximum retained rows per selected tensor.
            keys: Optional tensor keys or argument indices to retain.
            channel_dim: Channel dimension used when flattening tensors.
        """

        selected = select_named_tensors(named_tensors(value), keys)
        if not selected:
            return
        self.num_samples += 1
        for key, tensor in selected:
            str_key = str(key)
            if self.primary_key is None:
                self.primary_key = str_key
            cache = self.tensors.setdefault(
                str_key, TensorCache(channel_dim=channel_dim)
            )
            cache.add(tensor, max_rows=max_rows, channel_dim=channel_dim)

    def tensor(self, key: str | int | None = None) -> torch.Tensor | None:
        """Return one cached tensor by key.

        Args:
            key: Tensor key, or ``None`` to use the primary key.

        Returns:
            Concatenated CPU tensor, or ``None`` when no cache exists.
        """

        cache = self._cache(key)
        return None if cache is None else cache.tensor()

    def keyed_tensors(self) -> dict[str, torch.Tensor]:
        """Return all non-empty cached tensors.

        Returns:
            Mapping from cache key to concatenated CPU tensor.
        """

        result: dict[str, torch.Tensor] = {}
        for key, cache in self.tensors.items():
            tensor = cache.tensor()
            if tensor is not None:
                result[key] = tensor
        return result

    def clear(self) -> None:
        """Release every named tensor cache and reset metadata."""

        for cache in self.tensors.values():
            cache.clear()
        self.tensors.clear()
        self.primary_key = None
        self.num_samples = 0

    def alias(self, alias: str, source: str) -> None:
        """Expose an existing cache under another key.

        Args:
            alias: New key to register.
            source: Existing key whose cache should be reused.
        """

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
    ) -> tuple[torch.Tensor, ...]:
        """Split one named tensor cache into bounded partitions.

        Args:
            key: Tensor key, or ``None`` to use the primary key.
            sample_size: Maximum sample rows to keep, or ``-1`` for all.
            sample_batch_size: Partition size from sample batching.

        Returns:
            Tuple of CPU row tensors.
        """

        cache = self._cache(key)
        if cache is None:
            return ()
        return cache.repartition(
            sample_size=sample_size,
            sample_batch_size=sample_batch_size,
        )

    def _cache(self, key: str | int | None) -> TensorCache | None:
        """Resolve a cache by explicit or primary key.

        Args:
            key: Requested tensor key, or ``None`` for the primary key.

        Returns:
            Matching tensor cache, or ``None``.
        """

        if key is None:
            key = self.primary_key
        if key is None:
            return None
        return self.tensors.get(str(key))


@dataclass
class IOTensorsCache:
    """Input/output cache for one captured module.

    Args:
        inputs: Structured input tensor cache.
        outputs: Structured output tensor cache.
        replay_args: Optional CPU replay positional arguments for the module.
        replay_kwargs: Optional CPU replay keyword arguments for the module.
    """

    inputs: TensorsCache = field(default_factory=TensorsCache)
    outputs: TensorsCache = field(default_factory=TensorsCache)
    replay_args: tuple[Any, ...] | None = None
    replay_kwargs: dict[str, Any] | None = None
    input_min: float | None = None

    def clear(self) -> None:
        """Release cached inputs, outputs, and replay arguments."""

        self.inputs.clear()
        self.outputs.clear()
        self.replay_args = None
        self.replay_kwargs = None
        self.input_min = None
