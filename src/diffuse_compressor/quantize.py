from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .artifact import QuantizedTarget
from .calibration import EvalReplayBatch, IOTensorsCache
from .config import AwqTargetQuant, CalibrationSpec, DiffusionQuantSpec, SvdqTargetQuant, target_bias_policy
from .logging import QuantizationLogger
from .patches import ShiftedConv2d, ShiftedLinear
from .targets import QuantTarget

ProjectorModule = nn.Linear | nn.Conv2d


@dataclass(frozen=True)
class ProjectorTargetContext:
    """Prepared projector target inputs shared by projector quantizers."""

    target: QuantTarget
    spec: DiffusionQuantSpec
    modules: list[ProjectorModule]
    weight: torch.Tensor
    bias: torch.Tensor | None
    calibration_inputs: torch.Tensor | None
    calibration_input_partitions: tuple[torch.Tensor, ...] | None
    target_cache: IOTensorsCache | None
    eval_replay: EvalReplayBatch | Sequence[EvalReplayBatch] | None
    calibration: CalibrationSpec | None
    compute_device: torch.device | None
    logger: QuantizationLogger


class ProjectorQuantizer(ABC):
    """Quantizer interface for projector targets."""

    @abstractmethod
    def quantize(self, context: ProjectorTargetContext) -> QuantizedTarget:
        """Quantize a prepared projector target."""


@torch.inference_mode()
def quantize_targets(
    targets: Iterable[QuantTarget],
    spec: DiffusionQuantSpec,
    calibration_inputs: dict[str, torch.Tensor] | None = None,
    calibration_input_partitions: dict[str, tuple[torch.Tensor, ...]] | None = None,
    layer_cache: dict[str, IOTensorsCache] | None = None,
    eval_replay: EvalReplayBatch | Sequence[EvalReplayBatch] | None = None,
    calibration: CalibrationSpec | None = None,
    logger: QuantizationLogger | None = None,
) -> list[QuantizedTarget]:
    """Quantize concrete linear/pointwise-conv projector targets."""

    logger = logger or QuantizationLogger()
    targets = list(targets)
    compute_device = _resolve_compute_device(spec.compute_device)
    logger.info("- Quantizing %d projector targets", len(targets))
    if compute_device is not None:
        logger.info("- Using %s for per-target quantization compute", compute_device)
    quantized: list[QuantizedTarget] = []
    for index, target in enumerate(targets, start=1):
        if target.kind not in {"linear", "conv"}:
            raise NotImplementedError(
                f"Projector quantization currently supports linear and pointwise conv targets, got {target.kind!r}"
            )
        quantizer = _quantizer_for_target(target)
        logger.info(
            "  + Target %d/%d: %s (%d module%s, method=%s)",
            index,
            len(targets),
            target.export_name,
            len(target.modules),
            "" if len(target.modules) == 1 else "s",
            quantizer.__class__.__name__,
        )
        inputs = calibration_inputs.get(target.export_name) if calibration_inputs is not None else None
        input_partitions = (
            calibration_input_partitions.get(target.export_name)
            if calibration_input_partitions is not None and target.export_name in calibration_input_partitions
            else None
        )
        target_cache = layer_cache.get(target.export_name) if layer_cache is not None else None
        target_started_at = logger.start_timing()
        try:
            context = _projector_context(
                target,
                spec,
                inputs,
                input_partitions,
                target_cache,
                eval_replay,
                calibration,
                compute_device=compute_device,
                logger=logger,
            )
            quantized_target = quantizer.quantize(context)
            quantized.append(quantized_target)
            logger.stop_timing(quantized_target, target_started_at)
        finally:
            _clear_cuda_cache(compute_device)
    return quantized


def _projector_context(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    calibration_inputs: torch.Tensor | None,
    calibration_input_partitions: tuple[torch.Tensor, ...] | None,
    target_cache: IOTensorsCache | None,
    eval_replay: EvalReplayBatch | Sequence[EvalReplayBatch] | None,
    calibration: CalibrationSpec | None,
    *,
    compute_device: torch.device | None,
    logger: QuantizationLogger,
) -> ProjectorTargetContext:
    modules = _projector_modules(target)
    source_weight = _projector_weight(modules[0])
    work_device = compute_device or source_weight.device
    weight = torch.cat([_projector_weight(module).to(device=work_device) for module in modules], dim=0)
    bias = _concat_bias(modules, weight.device, weight.dtype, policy=target_bias_policy(target.quant))
    export_dtype = torch.bfloat16 if weight.dtype not in (torch.float16, torch.bfloat16) else weight.dtype
    weight = weight.to(device=work_device, dtype=export_dtype)
    if bias is not None:
        bias = bias.to(device=work_device, dtype=export_dtype)
    return ProjectorTargetContext(
        target=target,
        spec=spec,
        modules=modules,
        weight=weight,
        bias=bias,
        calibration_inputs=calibration_inputs,
        calibration_input_partitions=calibration_input_partitions,
        target_cache=target_cache,
        eval_replay=eval_replay,
        calibration=calibration,
        compute_device=compute_device,
        logger=logger,
    )


def _quantizer_for_target(target: QuantTarget) -> ProjectorQuantizer:
    quantizer_cls = _registered_quantizers().get(type(target.quant))
    if quantizer_cls is not None:
        return quantizer_cls()
    raise NotImplementedError(f"No projector quantizer registered for target {target.export_name!r}")


def _registered_quantizers() -> dict[type, type[ProjectorQuantizer]]:
    from .methods.awq.quantize import AwqQuantizer
    from .methods.svdquant.quantize import SvdqQuantizer

    return {
        AwqTargetQuant: AwqQuantizer,
        SvdqTargetQuant: SvdqQuantizer,
    }


def _resolve_compute_device(device: str | None) -> torch.device | None:
    """Validate and resolve the optional per-target compute device."""

    if device is None:
        return None
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"compute_device {device!r} requires CUDA, but CUDA is not available")
    return resolved


def _clear_cuda_cache(device: torch.device | None) -> None:
    """Release cached CUDA blocks after one target finishes."""

    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def total_partition_rows(partitions: tuple[torch.Tensor, ...]) -> int:
    """Return the total flattened row count across calibration partitions."""

    total = 0
    for partition in partitions:
        if partition.numel() > 0:
            total += partition.reshape(-1, partition.shape[-1]).shape[0]
    return total


def _projector_modules(target: QuantTarget) -> list[ProjectorModule]:
    """Resolve projector modules from raw or shifted target wrappers."""

    modules: list[ProjectorModule] = []
    for module in target.modules:
        if isinstance(module, ShiftedLinear):
            modules.append(module.linear)
        elif isinstance(module, ShiftedConv2d):
            modules.append(module.conv)
        elif isinstance(module, nn.Linear):
            modules.append(module)
        elif isinstance(module, nn.Conv2d):
            modules.append(module)
    if target.kind == "conv":
        for module in modules:
            _validate_pointwise_conv(module)
    return modules


def _validate_pointwise_conv(module: ProjectorModule) -> None:
    """Validate that a projector module is a supported pointwise Conv2d."""

    if not isinstance(module, nn.Conv2d):
        return
    if module.kernel_size != (1, 1) or module.groups != 1:
        raise NotImplementedError(
            "Projector Conv2d targets currently require kernel_size=(1, 1) and groups=1 "
            f"(got kernel_size={module.kernel_size}, groups={module.groups})"
        )


def _projector_weight(module: ProjectorModule) -> torch.Tensor:
    """Return a projector weight matrix in ``[out, in]`` layout."""

    if isinstance(module, nn.Conv2d):
        _validate_pointwise_conv(module)
        return module.weight.detach().flatten(1)
    return module.weight.detach()


def _concat_bias(
    modules: list[ProjectorModule], device: torch.device, dtype: torch.dtype, *, policy: str = "auto"
) -> torch.Tensor | None:
    """Concatenate biases from grouped projector modules."""

    if policy == "omit":
        return None
    if policy not in {"auto", "zero"}:
        raise ValueError(f"Unsupported target export_bias policy: {policy!r}")
    if policy == "auto" and all(module.bias is None for module in modules):
        return None
    return torch.cat(
        [
            module.bias.detach().to(device=device, dtype=dtype)
            if module.bias is not None
            else torch.zeros(_projector_out_features(module), device=device, dtype=dtype)
            for module in modules
        ],
        dim=0,
    )


def _projector_out_features(module: ProjectorModule) -> int:
    """Return output feature/channel count for one projector."""

    return module.out_channels if isinstance(module, nn.Conv2d) else module.out_features
