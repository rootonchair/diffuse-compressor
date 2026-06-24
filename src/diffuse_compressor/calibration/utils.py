from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from ..config import CalibrationSpec
from ..logging import QuantizationLogger
from ..tensor_utils import (
    first_tensor,
    first_tensor_rows,
    flatten_named_tensors,
    is_forward_pair,
    named_tensors,
    repartition_tensor,
    sample_count,
    select_named_tensors,
    tensor_rows,
    to_cpu,
    to_device,
    tree_map,
)


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
        Accelerate hook execution device, first parameter/buffer device, or
        CPU for parameterless modules.
    """

    hook_device = _accelerate_execution_device(model)
    if hook_device is not None:
        return hook_device
    for tensor in model.parameters(recurse=True):
        return tensor.device
    for tensor in model.buffers(recurse=True):
        return tensor.device
    return torch.device("cpu")


def has_accelerate_hooks(model: nn.Module) -> bool:
    """Return whether any module has an Accelerate hook attached."""

    return any(getattr(module, "_hf_hook", None) is not None for module in model.modules())


def remove_accelerate_hooks(model: nn.Module, logger: QuantizationLogger | None = None) -> bool:
    """Remove Accelerate hooks from a model if they are present."""

    if not has_accelerate_hooks(model):
        return False
    try:
        from accelerate.hooks import remove_hook_from_submodules
    except ImportError:
        return False
    with torch.inference_mode():
        remove_hook_from_submodules(model)
    log = QuantizationLogger.get_logger(__name__) if logger is None else logger.for_name(__name__)
    log.info("- Removed Accelerate hooks from model")
    return True


def _accelerate_execution_device(model: nn.Module) -> torch.device | None:
    """Return an Accelerate offload execution device when one is attached."""

    for module in model.modules():
        if (device := _hook_execution_device(getattr(module, "_hf_hook", None))) is not None:
            return device
    return None


def _hook_execution_device(hook: Any) -> torch.device | None:
    """Resolve a possibly chained Accelerate hook execution device."""

    device = getattr(hook, "execution_device", None)
    if device is not None:
        try:
            return torch.device(device)
        except (RuntimeError, TypeError):
            pass
    for child in getattr(hook, "hooks", ()):
        if (child_device := _hook_execution_device(child)) is not None:
            return child_device
    return None


def filter_replay_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any], *, arg_indices: Sequence[int], kwarg_keys: Sequence[str]
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

    replay_args = (
        tuple(args[index] for index in arg_indices if -len(args) <= index < len(args)) if arg_indices else args
    )
    replay_kwargs = {key: kwargs[key] for key in kwarg_keys if key in kwargs} if kwarg_keys else kwargs
    return replay_args, replay_kwargs


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
