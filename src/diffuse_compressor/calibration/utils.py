from __future__ import annotations

from contextlib import contextmanager, nullcontext
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


def materialized_state_dict(model: nn.Module) -> dict[str, torch.Tensor] | None:
    """Return a CPU state dict of an Accelerate-offloaded model, leaving its hooks intact.

    ``model.state_dict()`` on an offloaded model yields ``meta`` tensors. Removing
    the hooks to work around that is destructive for a pipeline using *model* CPU
    offload: diffusers links those hooks into a chain, and re-applying sequential
    offload in their place makes the next forward pass call ``.to("cpu")`` on meta
    parameters. Materializing one module at a time keeps the chain untouched.

    Returns ``None`` when the model has no Accelerate hooks, so callers can use the
    ordinary path.
    """

    if not has_accelerate_hooks(model):
        return None
    try:
        from accelerate.utils import align_module_device
    except ImportError:
        return None

    expected = set(model.state_dict().keys())
    materialized: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        own = list(module.named_parameters(recurse=False)) + list(module.named_buffers(recurse=False))
        if not own:
            continue
        prefix = f"{name}." if name else ""
        context = align_module_device(module) if getattr(module, "_hf_hook", None) is not None else nullcontext()
        try:
            with context:
                for leaf_name, tensor in own:
                    key = f"{prefix}{leaf_name}"
                    if key in expected and tensor is not None:
                        materialized[key] = tensor.detach().to("cpu")
        except NotImplementedError:
            # Sequential offload can leave a submodule on meta while the hook that
            # owns its weights sits on an ancestor. Fall back to the hook-removal
            # path, which restores sequential offload faithfully.
            return None
    if expected - materialized.keys():
        return None
    return materialized


def reapply_accelerate_offload(
    model: nn.Module, device: torch.device, logger: QuantizationLogger | None = None
) -> bool:
    """Re-attach Accelerate sequential CPU offload hooks after a temporary removal.

    Mirrors the ``accelerate.cpu_offload`` call diffusers' own
    ``enable_sequential_cpu_offload`` makes per pipeline component, so a model
    temporarily materialized via :func:`remove_accelerate_hooks` (e.g. to read
    a real ``state_dict()``) can resume forward-pass-driven offloading
    afterward.
    """

    try:
        from accelerate import cpu_offload
    except ImportError:
        return False
    offload_buffers = len(model._parameters) > 0
    cpu_offload(model, device, offload_buffers=offload_buffers)
    log = QuantizationLogger.get_logger(__name__) if logger is None else logger.for_name(__name__)
    log.info("- Re-applied Accelerate sequential CPU offload to model")
    return True


@contextmanager
def accelerate_hooks_temporarily_removed(
    model: nn.Module, logger: QuantizationLogger | None = None, *, reapply: bool = True
):
    """Temporarily strip Accelerate offload hooks for direct tensor access.

    Direct weight mutation (in-place quantization) and plain ``state_dict()``
    reads are incompatible with an Accelerate-offloaded model, whose
    parameters live on the ``meta`` device between forward passes. This
    removes any hooks for the duration of the ``with`` block and, by default,
    restores sequential CPU offload afterward (even on exception), so callers
    can safely mix direct tensor access with subsequent forward-pass-driven
    calibration replay on the same model instance.

    Pass ``reapply=False`` when the caller's own hook-presence-aware device
    management (e.g. ``iter_calibration_scopes``'s scoped-replay or dynamic
    full-replay reattachment) will take over placement for whatever forward
    passes follow, instead of falling back to Accelerate's slower per-layer
    streaming.
    """

    device = model_device(model)
    removed = remove_accelerate_hooks(model, logger=logger)
    try:
        yield removed
    finally:
        if removed and reapply:
            reapply_accelerate_offload(model, device, logger=logger)


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
