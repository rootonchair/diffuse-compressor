from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from ..config import CalibrationSpec


def is_under_scope(module_name: str, scope_name: str) -> bool:
    """Return whether a module path belongs to a scope path.

    Args:
        module_name: Fully qualified module name to test.
        scope_name: Scope root module name.

    Returns:
        ``True`` when ``module_name`` is the scope or a descendant.
    """

    return module_name == scope_name or module_name.startswith(f"{scope_name}.")


def model_device(model: nn.Module) -> torch.device:
    """Find the first parameter or buffer device for a model.

    Args:
        model: Module to inspect.

    Returns:
        First parameter/buffer device, or CPU for parameterless modules.
    """

    for tensor in model.parameters(recurse=True):
        return tensor.device
    for tensor in model.buffers(recurse=True):
        return tensor.device
    return torch.device("cpu")


def to_cpu(value: Any) -> Any:
    """Move every tensor in a nested value to CPU.

    Args:
        value: Tensor or nested Python structure.

    Returns:
        Structure with detached CPU tensors.
    """

    return tree_map(value, lambda tensor: tensor.detach().cpu())


def to_device(value: Any, device: torch.device) -> Any:
    """Move every tensor in a nested value to a device.

    Args:
        value: Tensor or nested Python structure.
        device: Destination torch device.

    Returns:
        Structure with tensors moved to ``device``.
    """

    return tree_map(value, lambda tensor: tensor.to(device=device))


def first_tensor(value: Any) -> torch.Tensor | None:
    """Find the first tensor in a nested structure.

    Args:
        value: Tensor or nested Python structure.

    Returns:
        Detached tensor, or ``None`` if no tensor is present.
    """

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
    """Flatten tensors from a value into stable names.

    Args:
        value: Tensor, nested structure, or forward ``(args, kwargs)`` pair.

    Returns:
        List of ``(name, detached_tensor)`` pairs.
    """

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
    """Recursively flatten tensors and preserve structural paths.

    Args:
        value: Tensor or nested dict/list/tuple structure.
        prefix: Name prefix accumulated during recursion.

    Returns:
        List of ``(path, detached_tensor)`` pairs.
    """

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
    """Select tensors by flattened key.

    Args:
        named: Flattened named tensor pairs.
        keys: Requested keys. Integer keys select positional ``argN`` names.

    Returns:
        Selected tensor pairs, or the first tensor when no keys are specified.
    """

    if not keys:
        return list(named[:1])
    wanted = {f"arg{key}" if isinstance(key, int) else str(key) for key in keys}
    return [(key, tensor) for key, tensor in named if key in wanted]


def is_forward_pair(value: Any) -> bool:
    """Return whether a value is a normalized ``(args, kwargs)`` pair.

    Args:
        value: Value to inspect.

    Returns:
        ``True`` when the value is a tuple of positional args and keyword args.
    """

    return isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], tuple) and isinstance(value[1], dict)


def filter_replay_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    arg_indices: Sequence[int],
    kwarg_keys: Sequence[str],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Filter replay inputs to requested positional and keyword subsets.

    Args:
        args: Original positional arguments.
        kwargs: Original keyword arguments.
        arg_indices: Positional argument indices to keep, or empty for all.
        kwarg_keys: Keyword argument names to keep, or empty for all.

    Returns:
        Filtered positional and keyword arguments.
    """

    replay_args = tuple(args[index] for index in arg_indices if -len(args) <= index < len(args)) if arg_indices else args
    replay_kwargs = {key: kwargs[key] for key in kwarg_keys if key in kwargs} if kwarg_keys else kwargs
    return replay_args, replay_kwargs


def tensor_rows(tensor: torch.Tensor, channel_dim: int = -1) -> torch.Tensor:
    """Flatten a tensor into ``[rows, channels]`` form.

    Args:
        tensor: Tensor to flatten.
        channel_dim: Dimension to preserve as the channel axis.

    Returns:
        Detached two-dimensional tensor rows.
    """

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
    """Estimate sample count from a tensor.

    Args:
        tensor: Tensor to inspect.

    Returns:
        Leading dimension size, or ``1`` for scalars.
    """

    return 1 if tensor.ndim == 0 else int(tensor.shape[0])


def repartition_tensor(
    tensor: torch.Tensor,
    *,
    sample_size: int = -1,
    sample_batch_size: int = -1,
    element_size: int = -1,
    element_batch_size: int = -1,
) -> tuple[torch.Tensor, ...]:
    """Limit and split tensor rows for calibration consumers.

    Args:
        tensor: Tensor whose last dimension is treated as channels.
        sample_size: Maximum sample rows to keep, or ``-1`` for all.
        sample_batch_size: Partition size from sample batching.
        element_size: Maximum element rows to keep, or ``-1`` for all.
        element_batch_size: Partition size from element batching.

    Returns:
        Tuple of row partitions.
    """

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
    """Return the row count of the first tensor found in values.

    Args:
        *values: Values or nested structures to search.

    Returns:
        Number of flattened rows, or ``0`` when no tensor is present.
    """

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
    """Apply a function to every tensor in a nested value.

    Args:
        value: Tensor or nested Python structure.
        tensor_fn: Callable applied to each tensor.

    Returns:
        Structure with tensors replaced by ``tensor_fn`` results.
    """

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
    """Abort calibration when system RAM usage exceeds the configured limit.

    Args:
        calibration: Calibration settings containing ``ram_usage_limit``.

    Raises:
        RuntimeError: If ``psutil`` is available and current memory usage is
            above the configured limit.
    """

    try:
        import psutil
    except ImportError:
        return
    usage = psutil.virtual_memory().percent / 100
    if usage > calibration.ram_usage_limit:
        raise RuntimeError(
            f"memory usage {usage:.1%} exceeds calibration ram_usage_limit {calibration.ram_usage_limit:.1%}"
        )
