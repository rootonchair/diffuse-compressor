from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...calibration import EvalReplayBatch
from ...config import DiffusionQuantSpec, LowRankSolverSpec
from ...targets import QuantTarget


@dataclass(frozen=True)
class LowRankSearchResult:
    """Result from the search-based low-rank solver.

    Args:
        low_rank: ``(down, up)`` low-rank matrices.
        residual: Residual weight selected after candidate scoring.
        metadata: Solver diagnostics and objective values.
    """

    low_rank: tuple[torch.Tensor, torch.Tensor]
    residual: torch.Tensor
    metadata: dict[str, object]


@torch.inference_mode()
def search_low_rank_branch(
    target: QuantTarget,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor | None,
    input_partitions: tuple[torch.Tensor, ...] | None,
    spec: DiffusionQuantSpec,
    eval_replay: EvalReplayBatch | None,
    low_rank_fn: Callable[[torch.Tensor, int, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
    weight_scales_fn: Callable[[torch.Tensor, int, bool], torch.Tensor],
    fake_quant_weight_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
    activation_quant_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> LowRankSearchResult:
    """Search for a low-rank branch using residual quantization candidates.

    Args:
        target: Target being quantized.
        weight: Smoothed target weight.
        bias: Optional target bias.
        inputs: Optional calibration input rows.
        input_partitions: Optional calibration input partitions.
        spec: Quantization settings containing solver options.
        eval_replay: Optional eval-module replay for objective scoring.
        low_rank_fn: Callable that builds a low-rank branch.
        weight_scales_fn: Callable that computes residual weight scales.
        fake_quant_weight_fn: Callable that quantizes and dequantizes weights.
        activation_quant_fn: Optional fake activation quantizer.

    Returns:
        Selected low-rank branch, residual weight, and metadata.
    """

    solver = spec.low_rank_solver
    rank = min(spec.rank, min(weight.shape))
    empty = (
        torch.empty(0, weight.shape[1], dtype=weight.dtype, device=weight.device),
        torch.empty(weight.shape[0], 0, dtype=weight.dtype, device=weight.device),
    )
    if rank == 0 or not target.shared_low_rank:
        return LowRankSearchResult(
            low_rank=empty,
            residual=weight,
            metadata={"mode": solver.mode, "iterations": 0, "reason": "disabled"},
        )

    search_inputs = None if inputs is None else _sample_inputs(inputs, solver.sample_size).to(device=weight.device, dtype=weight.dtype)
    search_partitions = tuple(
        _sample_inputs(partition, solver.sample_size).to(device=weight.device, dtype=weight.dtype)
        for partition in (input_partitions or (() if search_inputs is None else (search_inputs,)))
    )
    baseline = _quantized_weight(weight, spec, weight_scales_fn, fake_quant_weight_fn) if solver.compensate else torch.zeros_like(weight)
    best_low_rank = empty
    best_residual = weight
    best_error: torch.Tensor | None = None
    stopped_early = False
    errors: list[float] = []

    for iteration in range(solver.num_iters):
        # DeepCompressor initializes each low-rank search candidate from the
        # residual weight only; activations participate in scoring, not in the
        # branch SVD itself.
        candidate_low_rank = low_rank_fn(weight - baseline, rank, None)
        candidate_branch = candidate_low_rank[1] @ candidate_low_rank[0]
        candidate_residual = weight - candidate_branch
        candidate_quantized_residual = _quantized_weight(candidate_residual, spec, weight_scales_fn, fake_quant_weight_fn)
        error = _score_candidate(
            target=target,
            residual=candidate_quantized_residual,
            low_rank=candidate_low_rank,
            bias=bias,
            input_partitions=search_partitions,
            expected_weight=weight,
            spec=spec,
            solver=solver,
            eval_replay=eval_replay,
            activation_quant_fn=activation_quant_fn,
        )
        errors.append(float(error.cpu()))
        if best_error is None or error < best_error:
            best_error = error
            best_low_rank = candidate_low_rank
            best_residual = candidate_quantized_residual
            baseline = candidate_quantized_residual
        elif solver.early_stop:
            stopped_early = True
            break
        else:
            baseline = candidate_quantized_residual

    metadata = {
        "mode": solver.mode,
        "iterations": len(errors),
        "best_error": None if best_error is None else float(best_error.cpu()),
        "errors": errors,
        "early_stop": solver.early_stop,
        "stopped_early": stopped_early,
        "compensate": solver.compensate,
        "activation_quant": bool(solver.activation_quant or activation_quant_fn is not None),
        "objective": solver.objective,
        "eval_replay": bool(eval_replay is not None and solver.eval_replay),
    }
    return LowRankSearchResult(low_rank=best_low_rank, residual=best_residual, metadata=metadata)


def _quantized_weight(
    weight: torch.Tensor,
    spec: DiffusionQuantSpec,
    weight_scales_fn: Callable[[torch.Tensor, int, bool], torch.Tensor],
    fake_quant_weight_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
) -> torch.Tensor:
    """Quantize and dequantize a weight matrix using configured precision.

    Args:
        weight: Weight matrix to approximate.
        spec: Quantization settings.
        weight_scales_fn: Scale computation callable.
        fake_quant_weight_fn: Fake quantization callable.

    Returns:
        Dequantized weight approximation.
    """

    scale = weight_scales_fn(weight, group_size=spec.group_size, float_point=spec.precision == "fp4")
    return fake_quant_weight_fn(weight, scale, spec.precision == "fp4")


def _score_candidate(
    *,
    target: QuantTarget,
    residual: torch.Tensor,
    low_rank: tuple[torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None,
    input_partitions: tuple[torch.Tensor, ...],
    expected_weight: torch.Tensor,
    spec: DiffusionQuantSpec,
    solver: LowRankSolverSpec,
    eval_replay: EvalReplayBatch | None,
    activation_quant_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Score one low-rank/residual candidate.

    Args:
        target: Target being quantized.
        residual: Quantized residual candidate.
        low_rank: Candidate low-rank branch.
        bias: Optional target bias.
        input_partitions: Calibration input partitions.
        expected_weight: Floating-point reference weight.
        spec: Quantization settings.
        solver: Low-rank solver settings.
        eval_replay: Optional eval replay record.
        activation_quant_fn: Optional fake activation quantizer.

    Returns:
        Mean squared reconstruction error.
    """

    if eval_replay is not None and solver.eval_replay:
        replay_error = _score_eval_replay(target, residual, low_rank, solver, eval_replay, activation_quant_fn)
        if replay_error is not None:
            return replay_error
    if not input_partitions:
        branch = low_rank[1] @ low_rank[0]
        return (residual.float() + branch.float() - expected_weight.float()).pow(2).mean()
    branch = low_rank[1] @ low_rank[0]
    errors: list[torch.Tensor] = []
    for inputs in input_partitions:
        score_inputs = _quantize_activations(inputs, solver, activation_quant_fn)
        expected = _linear(score_inputs, expected_weight, bias)
        actual = _linear(score_inputs, residual + branch, bias)
        errors.append((actual.float() - expected.float()).pow(2).mean())
    return torch.stack(errors).mean()


def _score_eval_replay(
    target: QuantTarget,
    residual: torch.Tensor,
    low_rank: tuple[torch.Tensor, torch.Tensor],
    solver: LowRankSolverSpec,
    replay: EvalReplayBatch,
    activation_quant_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor | None:
    """Score a candidate by replaying an eval module with patched weights.

    Args:
        target: Target whose module weights are temporarily patched.
        residual: Quantized residual candidate.
        low_rank: Candidate low-rank branch.
        solver: Low-rank solver settings.
        replay: Captured eval-module replay record.
        activation_quant_fn: Optional fake activation quantizer.

    Returns:
        Replay output MSE, or ``None`` when replay cannot be applied.
    """

    if not target.modules:
        return None
    original_weights = [module.weight.data for module in target.modules if isinstance(module, nn.Linear)]
    if len(original_weights) != len(target.modules):
        return None
    branch = low_rank[1] @ low_rank[0]
    residual_chunks = list(residual.split([module.weight.shape[0] for module in target.modules], dim=0))
    branch_chunks = list(branch.split([module.weight.shape[0] for module in target.modules], dim=0))
    handles = []
    device = original_weights[0].device
    try:
        for module, quantized_weight, branch_weight in zip(target.modules, residual_chunks, branch_chunks, strict=True):
            module.weight.data = quantized_weight.to(device=module.weight.device, dtype=module.weight.dtype)
            handles.append(module.register_forward_hook(_branch_hook(branch_weight.to(device=module.weight.device, dtype=module.weight.dtype))))
            if solver.activation_quant or activation_quant_fn is not None:
                handles.append(module.register_forward_pre_hook(_activation_quant_hook(solver, activation_quant_fn)))
        args = _to_device(replay.args, device)
        kwargs = _to_device(replay.kwargs, device)
        expected = _to_device(replay.output, device)
        actual = replay.module(*args, **kwargs)
        return _tree_mse(actual, expected)
    finally:
        for module, original in zip(target.modules, original_weights, strict=True):
            module.weight.data = original
        for handle in handles:
            handle.remove()


def _branch_hook(branch_weight: torch.Tensor):
    """Create a hook that adds low-rank branch output to a linear result.

    Args:
        branch_weight: Low-rank branch weight for one grouped module chunk.

    Returns:
        Forward hook that adds ``F.linear(input, branch_weight)``.
    """

    def hook(_module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        """Apply the branch contribution to one module output.

        Args:
            _module: Hooked module, unused.
            args: Forward positional arguments.
            output: Original module output.

        Returns:
            Output plus low-rank branch contribution when possible.
        """

        if not args or not torch.is_tensor(args[0]) or not torch.is_tensor(output):
            return output
        return output + F.linear(args[0], branch_weight)

    return hook


def _activation_quant_hook(
    solver: LowRankSolverSpec,
    activation_quant_fn: Callable[[torch.Tensor], torch.Tensor] | None,
):
    """Create a pre-hook that fake-quantizes first tensor input.

    Args:
        solver: Low-rank solver settings.
        activation_quant_fn: Optional calibrated fake activation quantizer.

    Returns:
        Forward pre-hook for module inputs.
    """

    def hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        """Fake-quantize the first positional input tensor.

        Args:
            _module: Hooked module, unused.
            args: Forward positional arguments.

        Returns:
            Positional arguments with the first tensor fake-quantized.
        """

        if not args or not torch.is_tensor(args[0]):
            return args
        return (_quantize_activations(args[0], solver, activation_quant_fn), *args[1:])

    return hook


def _quantize_activations(
    inputs: torch.Tensor,
    solver: LowRankSolverSpec,
    activation_quant_fn: Callable[[torch.Tensor], torch.Tensor] | None,
) -> torch.Tensor:
    """Apply calibrated or simple fake activation quantization.

    Args:
        inputs: Activation tensor.
        solver: Low-rank solver settings.
        activation_quant_fn: Optional calibrated fake activation quantizer.

    Returns:
        Quantized/dequantized activations when enabled, otherwise inputs.
    """

    if activation_quant_fn is not None:
        return activation_quant_fn(inputs)
    if solver.activation_quant:
        return _fake_quantize_activations(inputs)
    return inputs


def _fake_quantize_activations(inputs: torch.Tensor) -> torch.Tensor:
    """Fake-quantize activations with per-channel INT4 absmax scales.

    Args:
        inputs: Activation tensor.

    Returns:
        Dequantized activation approximation.
    """

    scale = inputs.float().abs().amax(dim=0, keepdim=True).clamp_min(1e-6) / 7
    return (inputs.float() / scale).round_().clamp_(-8, 7).mul_(scale).to(dtype=inputs.dtype)


def _linear(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """Compute linear outputs for flattened calibration rows.

    Args:
        inputs: Input rows.
        weight: Weight matrix in ``[out, in]`` layout.
        bias: Optional bias vector.

    Returns:
        Output rows.
    """

    output = inputs @ weight.t()
    if bias is not None:
        output = output + bias.view(1, -1)
    return output


def _sample_inputs(inputs: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Flatten and optionally truncate input rows.

    Args:
        inputs: Input tensor.
        sample_size: Maximum rows to retain, or ``-1`` for all.

    Returns:
        Two-dimensional sampled rows.
    """

    rows = inputs.reshape(-1, inputs.shape[-1])
    if sample_size > 0 and rows.shape[0] > sample_size:
        rows = rows[:sample_size]
    return rows


def _tree_mse(actual: Any, expected: Any) -> torch.Tensor:
    """Compute MSE across matching tensors in two nested outputs.

    Args:
        actual: Actual replay output.
        expected: Reference replay output.

    Returns:
        Mean squared error, or infinity when structures do not match.
    """

    actual_tensors = _flatten_tensors(actual)
    expected_tensors = _flatten_tensors(expected)
    if not actual_tensors or len(actual_tensors) != len(expected_tensors):
        return torch.tensor(float("inf"))
    errors = [
        (actual_tensor.float() - expected_tensor.to(device=actual_tensor.device).float()).pow(2).mean()
        for actual_tensor, expected_tensor in zip(actual_tensors, expected_tensors, strict=True)
        if actual_tensor.shape == expected_tensor.shape
    ]
    if not errors:
        return torch.tensor(float("inf"))
    return torch.stack(errors).mean()


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    """Flatten tensors from a nested output structure.

    Args:
        value: Tensor or nested dict/list/tuple structure.

    Returns:
        Tensors in deterministic traversal order.
    """

    if torch.is_tensor(value):
        return [value]
    if isinstance(value, dict):
        tensors = []
        for key in sorted(value):
            tensors.extend(_flatten_tensors(value[key]))
        return tensors
    if isinstance(value, (list, tuple)):
        tensors = []
        for item in value:
            tensors.extend(_flatten_tensors(item))
        return tensors
    return []


def _to_device(value: Any, device: torch.device) -> Any:
    """Move tensors in a nested structure to a device.

    Args:
        value: Tensor or nested Python structure.
        device: Destination device.

    Returns:
        Structure with tensors moved to ``device``.
    """

    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value
