from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from typing import Any, Sequence

import torch


def to_cpu(value: Any) -> Any:
    """Move every tensor in a nested value to CPU."""

    return tree_map(value, lambda tensor: tensor.detach().cpu())


def to_device(value: Any, device: torch.device) -> Any:
    """Move every tensor in a nested value to a device."""

    return tree_map(value, lambda tensor: tensor.to(device=device))


def first_tensor(value: Any) -> torch.Tensor | None:
    """Find the first tensor in a nested structure."""

    stack = [value]
    while stack:
        item = stack.pop(0)
        if torch.is_tensor(item):
            return item.detach()
        if isinstance(item, dict):
            stack.extend(item.values())
        elif _is_dataclass_instance(item):
            stack.extend(_dataclass_values(item))
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return None


def named_tensors(value: Any) -> list[tuple[str, torch.Tensor]]:
    """Flatten tensors from a value into stable names."""

    if is_forward_pair(value):
        args, kwargs = value
        result: list[tuple[str, torch.Tensor]] = []
        for index, item in enumerate(args):
            result.extend(
                (f"arg{index}" if key == "" else f"arg{index}.{key}", tensor)
                for key, tensor in flatten_named_tensors(item)
            )
        for key, item in kwargs.items():
            result.extend(
                (str(key) if nested == "" else f"{key}.{nested}", tensor)
                for nested, tensor in flatten_named_tensors(item)
            )
        return result
    return flatten_named_tensors(value)


def flatten_named_tensors(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    """Recursively flatten tensors and preserve structural paths."""

    if torch.is_tensor(value):
        return [(prefix, value.detach())]
    if isinstance(value, dict):
        result: list[tuple[str, torch.Tensor]] = []
        for key, item in value.items():
            nested = str(key) if prefix == "" else f"{prefix}.{key}"
            result.extend(flatten_named_tensors(item, nested))
        return result
    if _is_dataclass_instance(value):
        result = []
        for key, item in _dataclass_items(value):
            nested = key if prefix == "" else f"{prefix}.{key}"
            result.extend(flatten_named_tensors(item, nested))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for index, item in enumerate(value):
            nested = str(index) if prefix == "" else f"{prefix}.{index}"
            result.extend(flatten_named_tensors(item, nested))
        return result
    return []


def select_named_tensors(
    named: Sequence[tuple[str, torch.Tensor]], keys: Sequence[str | int]
) -> list[tuple[str, torch.Tensor]]:
    """Select tensors by flattened key."""

    if not keys:
        return list(named[:1])
    wanted = {f"arg{key}" if isinstance(key, int) else str(key) for key in keys}
    return [(key, tensor) for key, tensor in named if key in wanted]


def is_forward_pair(value: Any) -> bool:
    """Return whether a value is a normalized ``(args, kwargs)`` pair."""

    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], tuple) and isinstance(value[1], dict)


def tensor_rows(tensor: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
    """Flatten a tensor into ``[rows, channels]`` form."""

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


def sample_count(tensor: torch.Tensor) -> int:
    """Estimate sample count from a tensor."""

    return 1 if tensor.ndim == 0 else int(tensor.shape[0])


def repartition_tensor(
    tensor: torch.Tensor, *, sample_size: int = -1, sample_batch_size: int = -1
) -> tuple[torch.Tensor, ...]:
    """Limit and split tensor rows for calibration consumers."""

    rows = tensor.reshape(-1, tensor.shape[-1])
    limit = rows.shape[0]
    if sample_size > 0:
        limit = min(limit, sample_size)
    rows = rows[:limit]
    if sample_batch_size <= 0:
        return (rows,)
    return tuple(rows[index : index + sample_batch_size] for index in range(0, rows.shape[0], sample_batch_size))


def first_tensor_rows(*values: Any) -> int:
    """Return the row count of the first tensor found in values."""

    stack = list(values)
    while stack:
        value = stack.pop(0)
        if torch.is_tensor(value):
            return tensor_rows(value).shape[0]
        if isinstance(value, dict):
            stack.extend(value.values())
        elif _is_dataclass_instance(value):
            stack.extend(_dataclass_values(value))
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return 0


def tree_map(value: Any, tensor_fn) -> Any:
    """Apply a function to every tensor in a nested value."""

    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: tree_map(item, tensor_fn) for key, item in value.items()}
    if _is_dataclass_instance(value):
        return replace(
            value,
            **{
                field.name: tree_map(getattr(value, field.name), tensor_fn)
                for field in fields(value)
                if field.init
            },
        )
    if isinstance(value, list):
        return [tree_map(item, tensor_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_map(item, tensor_fn) for item in value)
    return value


def _is_dataclass_instance(value: Any) -> bool:
    """Return whether value is a dataclass instance, not a dataclass type."""

    return is_dataclass(value) and not isinstance(value, type)


def _dataclass_items(value: Any) -> list[tuple[str, Any]]:
    """Return field-name/value pairs for dataclass traversal."""

    return [(field.name, getattr(value, field.name)) for field in fields(value)]


def _dataclass_values(value: Any) -> list[Any]:
    """Return dataclass field values for stack-based traversal."""

    return [item for _name, item in _dataclass_items(value)]
