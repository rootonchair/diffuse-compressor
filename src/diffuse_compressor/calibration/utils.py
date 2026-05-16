from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from ..config import CalibrationSpec


def is_under_scope(module_name: str, scope_name: str) -> bool:
    return module_name == scope_name or module_name.startswith(f"{scope_name}.")


def model_device(model: nn.Module) -> torch.device:
    for tensor in model.parameters(recurse=True):
        return tensor.device
    for tensor in model.buffers(recurse=True):
        return tensor.device
    return torch.device("cpu")


def to_cpu(value: Any) -> Any:
    return tree_map(value, lambda tensor: tensor.detach().cpu())


def to_device(value: Any, device: torch.device) -> Any:
    return tree_map(value, lambda tensor: tensor.to(device=device))


def first_tensor(value: Any) -> torch.Tensor | None:
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


def named_tensors(value: Any) -> list[tuple[str, torch.Tensor]]:
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
    if torch.is_tensor(value):
        return [(prefix, value.detach())]
    if isinstance(value, dict):
        result: list[tuple[str, torch.Tensor]] = []
        for key, item in value.items():
            nested = str(key) if prefix == "" else f"{prefix}.{key}"
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
    named: Sequence[tuple[str, torch.Tensor]],
    keys: Sequence[str | int],
) -> list[tuple[str, torch.Tensor]]:
    if not keys:
        return list(named[:1])
    wanted = {f"arg{key}" if isinstance(key, int) else str(key) for key in keys}
    return [(key, tensor) for key, tensor in named if key in wanted]


def is_forward_pair(value: Any) -> bool:
    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], tuple) and isinstance(value[1], dict)


def filter_replay_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    arg_indices: Sequence[int],
    kwarg_keys: Sequence[str],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    replay_args = tuple(args[index] for index in arg_indices if -len(args) <= index < len(args)) if arg_indices else args
    replay_kwargs = {key: kwargs[key] for key in kwarg_keys if key in kwargs} if kwarg_keys else kwargs
    return replay_args, replay_kwargs


def tensor_rows(tensor: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
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


def first_tensor_rows(*values: Any) -> int:
    stack = list(values)
    while stack:
        value = stack.pop(0)
        if torch.is_tensor(value):
            return tensor_rows(value).shape[0]
        if isinstance(value, dict):
            stack.extend(value.values())
        elif isinstance(value, (list, tuple)):
            stack.extend(value)
    return 0


def tree_map(value: Any, tensor_fn) -> Any:
    if torch.is_tensor(value):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: tree_map(item, tensor_fn) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_map(item, tensor_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_map(item, tensor_fn) for item in value)
    return value


def check_ram(calibration: CalibrationSpec) -> None:
    try:
        import psutil
    except ImportError:
        return
    usage = psutil.virtual_memory().percent / 100
    if usage > calibration.ram_usage_limit:
        raise RuntimeError(
            f"memory usage {usage:.1%} exceeds calibration ram_usage_limit {calibration.ram_usage_limit:.1%}"
        )
