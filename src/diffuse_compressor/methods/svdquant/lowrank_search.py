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
    low_rank: tuple[torch.Tensor, torch.Tensor]
    residual: torch.Tensor
    metadata: dict[str, object]


@torch.inference_mode()
def search_low_rank_branch(
    target: QuantTarget,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    inputs: torch.Tensor | None,
    spec: DiffusionQuantSpec,
    eval_replay: EvalReplayBatch | None,
    low_rank_fn: Callable[[torch.Tensor, int, torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]],
    weight_scales_fn: Callable[[torch.Tensor, int, bool], torch.Tensor],
    fake_quant_weight_fn: Callable[[torch.Tensor, torch.Tensor, bool], torch.Tensor],
) -> LowRankSearchResult:
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
            inputs=search_inputs,
            expected_weight=weight,
            spec=spec,
            solver=solver,
            eval_replay=eval_replay,
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
        "activation_quant": solver.activation_quant,
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
    scale = weight_scales_fn(weight, group_size=spec.group_size, float_point=spec.precision == "fp4")
    return fake_quant_weight_fn(weight, scale, spec.precision == "fp4")


def _score_candidate(
    *,
    target: QuantTarget,
    residual: torch.Tensor,
    low_rank: tuple[torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None,
    inputs: torch.Tensor | None,
    expected_weight: torch.Tensor,
    spec: DiffusionQuantSpec,
    solver: LowRankSolverSpec,
    eval_replay: EvalReplayBatch | None,
) -> torch.Tensor:
    if eval_replay is not None and solver.eval_replay:
        replay_error = _score_eval_replay(target, residual, low_rank, solver, eval_replay)
        if replay_error is not None:
            return replay_error
    if inputs is None:
        branch = low_rank[1] @ low_rank[0]
        return (residual.float() + branch.float() - expected_weight.float()).pow(2).mean()
    score_inputs = _fake_quantize_activations(inputs) if solver.activation_quant else inputs
    expected = _linear(score_inputs, expected_weight, bias)
    branch = low_rank[1] @ low_rank[0]
    actual = _linear(score_inputs, residual + branch, bias)
    return (actual.float() - expected.float()).pow(2).mean()


def _score_eval_replay(
    target: QuantTarget,
    residual: torch.Tensor,
    low_rank: tuple[torch.Tensor, torch.Tensor],
    solver: LowRankSolverSpec,
    replay: EvalReplayBatch,
) -> torch.Tensor | None:
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
            if solver.activation_quant:
                handles.append(module.register_forward_pre_hook(_activation_quant_hook))
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
    def hook(_module: nn.Module, args: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        if not args or not torch.is_tensor(args[0]) or not torch.is_tensor(output):
            return output
        return output + F.linear(args[0], branch_weight)

    return hook


def _activation_quant_hook(_module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
    if not args or not torch.is_tensor(args[0]):
        return args
    return (_fake_quantize_activations(args[0]), *args[1:])


def _fake_quantize_activations(inputs: torch.Tensor) -> torch.Tensor:
    scale = inputs.float().abs().amax(dim=0, keepdim=True).clamp_min(1e-6) / 7
    return (inputs.float() / scale).round_().clamp_(-8, 7).mul_(scale).to(dtype=inputs.dtype)


def _linear(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    output = inputs @ weight.t()
    if bias is not None:
        output = output + bias.view(1, -1)
    return output


def _sample_inputs(inputs: torch.Tensor, sample_size: int) -> torch.Tensor:
    rows = inputs.reshape(-1, inputs.shape[-1])
    if sample_size > 0 and rows.shape[0] > sample_size:
        rows = rows[:sample_size]
    return rows


def _tree_mse(actual: Any, expected: Any) -> torch.Tensor:
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
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value
