from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, Sequence

import torch

from ...backends.nunchaku.layouts import linear_output
from ...backends.nunchaku.packing import fp4_e2m1_codebook, fp_quantize
from ...calibration import repartition_tensor
from ...config import CalibrationSpec, DiffusionQuantSpec, LowRankSolverSpec, SmoothSpec
from ...logging import QuantizationLogger
from ...targets import QuantTarget, target_shared_low_rank


@dataclass(frozen=True)
class SmoothCandidate:
    """One smoothing scale candidate.

    Args:
        scale: Per-input-channel smoothing scale.
        alpha: Activation-span exponent used for the candidate.
        beta: Weight-span exponent used for the candidate.
        span: Names of the activation and weight span estimators.
    """

    scale: torch.Tensor
    alpha: float
    beta: float
    span: tuple[str, str]


@dataclass(frozen=True)
class SmoothEvaluation:
    """Scored smoothing candidate."""

    candidate: SmoothCandidate
    error: torch.Tensor


@dataclass(frozen=True)
class SmoothSearchResult:
    """Result returned by a smoothing search strategy."""

    best_candidate: SmoothCandidate | None
    best_error: torch.Tensor | None
    num_candidates: int
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SmoothSpanContext:
    """Precomputed activation/weight spans used to build candidates."""

    alpha_span: torch.Tensor
    beta_span: torch.Tensor
    span: tuple[str, str]


EvaluateSmoothCandidates = Callable[[Sequence[SmoothCandidate]], Sequence[SmoothEvaluation]]


class SmoothSearchStrategy(Protocol):
    """Internal candidate scheduling interface for smoothing search."""

    def search(
        self,
        spec: SmoothSpec,
        span_contexts: Sequence[SmoothSpanContext],
        evaluate_candidates: EvaluateSmoothCandidates,
    ) -> SmoothSearchResult:
        """Schedule smoothing candidates and return the best evaluation."""
        ...


class ManualSmoothSearchStrategy:
    """Evaluate the fixed manual smoothing candidate for each span context."""

    def search(
        self,
        spec: SmoothSpec,
        span_contexts: Sequence[SmoothSpanContext],
        evaluate_candidates: EvaluateSmoothCandidates,
    ) -> SmoothSearchResult:
        candidates = tuple(
            _smooth_candidate(context, alpha, beta, spec.eps)
            for context in span_contexts
            for alpha, beta in _manual_alpha_beta_pairs(spec)
        )
        return _best_search_result(evaluate_candidates(candidates))


class RandomSmoothSearchStrategy:
    """Evaluate a deterministic random subset of the grid candidates."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def search(
        self,
        spec: SmoothSpec,
        span_contexts: Sequence[SmoothSpanContext],
        evaluate_candidates: EvaluateSmoothCandidates,
    ) -> SmoothSearchResult:
        pool = tuple(
            _smooth_candidate(context, alpha, beta, spec.eps)
            for context in span_contexts
            for alpha, beta in _grid_alpha_beta_pairs(spec, allow_random=True)
        )
        samples = _random_samples(spec)
        candidates = _sample_random_candidates(pool, samples, seed=self.seed)
        result = _best_search_result(evaluate_candidates(candidates))
        return SmoothSearchResult(
            best_candidate=result.best_candidate,
            best_error=result.best_error,
            num_candidates=result.num_candidates,
            metadata={"samples": samples, "actual_samples": len(candidates), "seed": self.seed},
        )


class GridSmoothSearchStrategy:
    """Evaluate the DeepCompressor-style smoothing grid."""

    def search(
        self,
        spec: SmoothSpec,
        span_contexts: Sequence[SmoothSpanContext],
        evaluate_candidates: EvaluateSmoothCandidates,
    ) -> SmoothSearchResult:
        candidates = tuple(
            _smooth_candidate(context, alpha, beta, spec.eps)
            for context in span_contexts
            for alpha, beta in _grid_alpha_beta_pairs(spec)
        )
        return _best_search_result(evaluate_candidates(candidates))


def resolve_smooth_spec(value: bool | SmoothSpec) -> SmoothSpec:
    """Normalize a boolean or explicit smoothing spec.

    Args:
        value: Existing spec or boolean enable flag.

    Returns:
        Concrete smoothing spec.
    """

    if isinstance(value, SmoothSpec):
        return value
    return SmoothSpec(enabled=value)


def resolve_smooth_search_strategy(spec: SmoothSpec, *, seed: int = 0) -> SmoothSearchStrategy:
    """Resolve the internal search strategy for a smoothing spec."""

    if spec.strategy == "manual":
        return ManualSmoothSearchStrategy()
    if spec.strategy == "grid_search":
        return GridSmoothSearchStrategy()
    if spec.strategy == "random_search":
        return RandomSmoothSearchStrategy(seed=seed)
    raise ValueError(f"Unsupported smoothing strategy: {spec.strategy!r}")


def build_smooth_span_contexts(
    inputs: torch.Tensor, weight: torch.Tensor, spec: SmoothSpec
) -> tuple[SmoothSpanContext, ...]:
    """Build precomputed span contexts for smoothing search.

    Args:
        inputs: Calibration input rows.
        weight: Target weight matrix.
        spec: Smoothing configuration.

    Returns:
        Span contexts used by smoothing search strategies.
    """

    inputs = _sample_inputs(inputs, spec.sample_size).to(dtype=torch.float32)
    weight = weight.to(dtype=torch.float32)
    return build_smooth_span_contexts_from_partitions((inputs,), weight, spec)


def build_smooth_span_contexts_from_partitions(
    input_partitions: Sequence[torch.Tensor], weight: torch.Tensor, spec: SmoothSpec
) -> tuple[SmoothSpanContext, ...]:
    """Build smoothing span contexts from bounded activation partitions.

    Args:
        input_partitions: Calibration input row partitions.
        weight: Target weight matrix.
        spec: Smoothing configuration.

    Returns:
        Span contexts used by smoothing search strategies.
    """

    sampled_partitions = _sample_input_partitions(input_partitions, spec.sample_size)
    weight = weight.to(dtype=torch.float32)
    return tuple(
        SmoothSpanContext(
            alpha_span=_activation_span_from_partitions(
                sampled_partitions, alpha_span_name, eps=spec.eps, device=weight.device
            ),
            beta_span=_weight_span(weight, beta_span_name, eps=spec.eps),
            span=(alpha_span_name, beta_span_name),
        )
        for alpha_span_name, beta_span_name in spec.spans
    )


def _manual_alpha_beta_pairs(spec: SmoothSpec) -> tuple[tuple[float, float], ...]:
    """Generate manual alpha/beta smoothing exponent pairs."""

    if spec.strategy == "manual":
        if spec.beta < 0:
            if not 0 <= spec.alpha <= 1:
                raise ValueError("manual smooth alpha must be in [0, 1] when beta is negative")
            return ((spec.alpha, 1 - spec.alpha),)
        if spec.alpha < 0:
            if not 0 <= spec.beta <= 1:
                raise ValueError("manual smooth beta must be in [0, 1] when alpha is negative")
            return ((1 - spec.beta, spec.beta),)
        if not 0 <= spec.alpha <= 1 or not 0 <= spec.beta <= 1:
            raise ValueError("manual smooth alpha and beta must be in [0, 1]")
        if spec.alpha == 0 and spec.beta == 0:
            raise ValueError("manual smooth alpha and beta cannot both be zero")
        return ((spec.alpha, spec.beta),)
    raise ValueError(f"ManualSmoothSearchStrategy does not support strategy {spec.strategy!r}")


def _grid_alpha_beta_pairs(spec: SmoothSpec, *, allow_random: bool = False) -> tuple[tuple[float, float], ...]:
    """Generate DeepCompressor-style grid-search alpha/beta pairs."""

    if spec.strategy != "grid_search" and not (allow_random and spec.strategy == "random_search"):
        raise ValueError(f"GridSmoothSearchStrategy does not support strategy {spec.strategy!r}")
    choices = [i / spec.num_grids for i in range(1, spec.num_grids)]
    if spec.alpha > 0:
        if spec.beta > 0:
            return tuple([(0, 0)] + [(alpha, alpha) for alpha in choices])
        if spec.beta == 0:
            return tuple([(0, 0)] + [(alpha, 0) for alpha in choices])
        if spec.beta == -1:
            return tuple([(0, 0)] + [(alpha, 1 - alpha) for alpha in choices])
        if spec.beta == -2:
            return tuple([(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices])
        return tuple(
            [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, beta) for alpha in choices for beta in choices]
        )
    if spec.alpha == 0:
        if spec.beta > 0:
            return tuple([(0, 0)] + [(0, beta) for beta in choices])
        if spec.beta == 0:
            return tuple([(0, 0)] + [(alpha, 0) for alpha in choices] + [(0, beta) for beta in choices])
        if spec.beta == -1:
            return tuple([(0, 0)] + [(0, beta) for beta in choices] + [(alpha, 1 - alpha) for alpha in choices])
        if spec.beta == -2:
            return tuple(
                [(0, 0)]
                + [(alpha, 0) for alpha in choices]
                + [(0, beta) for beta in choices]
                + [(alpha, 1 - alpha) for alpha in choices]
            )
        return tuple(
            [(0, 0)]
            + [(alpha, 0) for alpha in choices]
            + [(0, beta) for beta in choices]
            + [(alpha, beta) for alpha in choices for beta in choices]
        )
    if spec.beta > 0:
        return tuple([(0, 0)] + [(1 - beta, beta) for beta in choices])
    if spec.beta == 0:
        return tuple([(0, 0)] + [(alpha, 0) for alpha in choices])
    if spec.beta == -1:
        return tuple([(0, 0)] + [(alpha, 1 - alpha) for alpha in choices])
    if spec.beta == -2:
        return tuple([(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices])
    return tuple(
        [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, beta) for alpha in choices for beta in choices]
    )


def _random_samples(spec: SmoothSpec) -> int:
    """Return the configured random search sample count."""

    value = spec.strategy_options.get("random_samples", 8)
    if not isinstance(value, int) or value <= 0:
        raise ValueError("smooth strategy_options['random_samples'] must be a positive integer")
    return value


def _sample_random_candidates(
    candidates: Sequence[SmoothCandidate], samples: int, *, seed: int
) -> tuple[SmoothCandidate, ...]:
    """Sample smoothing candidates deterministically, preserving identity first."""

    if samples >= len(candidates):
        return tuple(candidates)
    identity_index = next(
        (index for index, candidate in enumerate(candidates) if candidate.alpha == 0 and candidate.beta == 0), None
    )
    rng = random.Random(seed)
    if identity_index is None:
        indices = rng.sample(range(len(candidates)), samples)
    else:
        remaining = [index for index in range(len(candidates)) if index != identity_index]
        indices = [identity_index] + rng.sample(remaining, samples - 1)
    return tuple(candidates[index] for index in indices)


def _smooth_candidate(context: SmoothSpanContext, alpha: float, beta: float, eps: float) -> SmoothCandidate:
    """Build one smoothing candidate from a span context and alpha/beta pair."""

    if alpha == 0 and beta == 0:
        scale = torch.ones_like(context.alpha_span)
    else:
        scale = context.alpha_span.pow(alpha)
        if beta > 0:
            scale = scale / context.beta_span.pow(beta)
        scale = _sanitize_scale(scale, eps=eps)
    return SmoothCandidate(scale=scale.to(device=context.beta_span.device), alpha=alpha, beta=beta, span=context.span)


def _best_search_result(evaluations: Sequence[SmoothEvaluation]) -> SmoothSearchResult:
    """Select the best smoothing evaluation by scalar error."""

    if not evaluations:
        return SmoothSearchResult(best_candidate=None, best_error=None, num_candidates=0)
    best = min(evaluations, key=lambda item: float(item.error.detach().cpu()))
    return SmoothSearchResult(best_candidate=best.candidate, best_error=best.error, num_candidates=len(evaluations))


def _sample_inputs(inputs: torch.Tensor, sample_size: int) -> torch.Tensor:
    """Flatten and optionally truncate calibration inputs.

    Args:
        inputs: Input tensor whose last dimension is features.
        sample_size: Maximum rows to retain, or ``-1`` for all.

    Returns:
        Two-dimensional sampled input rows.
    """

    rows = inputs.reshape(-1, inputs.shape[-1])
    if sample_size > 0 and rows.shape[0] > sample_size:
        rows = rows[:sample_size]
    return rows


def _sample_input_partitions(input_partitions: Sequence[torch.Tensor], sample_size: int) -> tuple[torch.Tensor, ...]:
    """Flatten and optionally truncate calibration input partitions."""

    sampled: list[torch.Tensor] = []
    remaining = sample_size
    for partition in input_partitions:
        rows = partition.reshape(-1, partition.shape[-1])
        if rows.numel() == 0:
            continue
        if sample_size > 0:
            if remaining <= 0:
                break
            if rows.shape[0] > remaining:
                rows = rows[:remaining]
            remaining -= rows.shape[0]
        sampled.append(rows)
    return tuple(sampled)


def _activation_span(inputs: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    """Compute per-channel activation span.

    Args:
        inputs: Input rows.
        mode: Span estimator name, ``"absmax"`` or ``"rms"``.
        eps: Positive lower bound.

    Returns:
        Per-channel activation span.
    """

    if mode == "absmax":
        span = inputs.abs().amax(dim=0)
    elif mode == "rms":
        span = inputs.pow(2).mean(dim=0).sqrt()
    else:
        raise ValueError(f"Unsupported activation smooth span: {mode!r}")
    return span.clamp_min(eps)


def _activation_span_from_partitions(
    input_partitions: Sequence[torch.Tensor], mode: str, eps: float, *, device: torch.device
) -> torch.Tensor:
    """Compute per-channel activation span from bounded partitions."""

    if not input_partitions:
        raise ValueError("smoothing span calibration requires at least one input partition")
    span: torch.Tensor | None = None
    sum_squares: torch.Tensor | None = None
    count = 0
    for partition in input_partitions:
        rows = partition.to(device=device, dtype=torch.float32).reshape(-1, partition.shape[-1])
        if rows.numel() == 0:
            continue
        if mode == "absmax":
            partition_span = rows.abs().amax(dim=0)
            span = partition_span if span is None else torch.maximum(span, partition_span)
        elif mode == "rms":
            partition_sum = rows.pow(2).sum(dim=0)
            sum_squares = partition_sum if sum_squares is None else sum_squares + partition_sum
            count += rows.shape[0]
        else:
            raise ValueError(f"Unsupported activation smooth span: {mode!r}")
    if mode == "rms":
        if sum_squares is None or count == 0:
            raise ValueError("smoothing span calibration requires non-empty input partitions")
        span = (sum_squares / count).sqrt()
    if span is None:
        raise ValueError("smoothing span calibration requires non-empty input partitions")
    return span.clamp_min(eps)


def _weight_span(weight: torch.Tensor, mode: str, eps: float) -> torch.Tensor:
    """Compute per-input-channel weight span.

    Args:
        weight: Weight matrix in ``[out, in]`` layout.
        mode: Span estimator name, ``"absmax"`` or ``"rms"``.
        eps: Positive lower bound.

    Returns:
        Per-input-channel weight span.
    """

    if mode == "absmax":
        span = weight.abs().amax(dim=0)
    elif mode == "rms":
        span = weight.pow(2).mean(dim=0).sqrt()
    else:
        raise ValueError(f"Unsupported weight smooth span: {mode!r}")
    return span.clamp_min(eps)


def _sanitize_scale(scale: torch.Tensor, eps: float) -> torch.Tensor:
    """Replace invalid smoothing scales and clamp to a floor.

    Args:
        scale: Candidate scale tensor.
        eps: Positive lower bound.

    Returns:
        Finite, positive scale tensor.
    """

    scale = torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
    return scale.clamp_min(eps)


def select_smooth_scale(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    calibration_inputs: torch.Tensor | None,
    calibration_input_partitions: tuple[torch.Tensor, ...] | None = None,
    seed: int = 0,
    sample_batch_size: int = -1,
    logger: QuantizationLogger | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Select the smoothing scale for one target."""

    log = _resolve_logger(logger)
    smooth_spec = resolve_smooth_spec(spec.smooth)
    identity = torch.ones(weight.shape[1], dtype=weight.dtype, device=weight.device)
    if not smooth_spec.enabled:
        log.info("      + Smoothing disabled")
        return identity, {"enabled": False, "searched": False, "reason": "disabled"}
    if calibration_inputs is None:
        log.info("      + Missing calibration inputs; using identity smoothing")
        return identity, {"enabled": True, "searched": False, "reason": "missing_calibration"}

    input_partitions = tuple(
        partition.reshape(-1, weight.shape[1]) for partition in (calibration_input_partitions or (calibration_inputs,))
    )
    search_weight = weight.to(dtype=torch.float32)
    search_bias = None if bias is None else bias.to(device=weight.device, dtype=torch.float32)
    best_error = torch.tensor(float("inf"), device=weight.device)
    best_scale = identity
    span_contexts = build_smooth_span_contexts_from_partitions(input_partitions, weight, smooth_spec)
    partition_baselines = _prepare_partition_baselines(input_partitions, search_weight, search_bias)

    def evaluate_candidates(candidates: Sequence[SmoothCandidate]) -> tuple[SmoothEvaluation, ...]:
        candidates = tuple(candidates)
        chunks = _chunk_candidates(candidates, sample_batch_size)
        errors: list[torch.Tensor] = []
        for chunk in _iter_smoothing_progress(chunks, target.export_name, log):
            log.debug("      + Scoring %d smoothing candidate(s)", len(chunk))
            errors.extend(
                _score_smoothing_candidates(
                    chunk,
                    partition_baselines,
                    search_weight,
                    search_bias,
                    spec,
                    target_shared_low_rank(target),
                )
            )
        return tuple(
            SmoothEvaluation(candidate=candidate, error=error) for candidate, error in zip(candidates, errors)
        )

    search = resolve_smooth_search_strategy(smooth_spec, seed=seed).search(
        smooth_spec, span_contexts, evaluate_candidates
    )
    best_candidate = search.best_candidate
    if best_candidate is not None:
        best_scale = best_candidate.scale.to(device=weight.device, dtype=weight.dtype)
        best_error = search.best_error if search.best_error is not None else best_error
        log.debug("        best smoothing error: %.6g", float(best_error.cpu()))
    metadata: dict[str, object] = {
        "enabled": True,
        "searched": True,
        "strategy": smooth_spec.strategy,
        "objective": smooth_spec.objective,
        "num_candidates": search.num_candidates,
        "error": float(best_error.cpu()),
    }
    if search.metadata:
        metadata["search"] = search.metadata
    if best_candidate is not None:
        metadata.update({"alpha": best_candidate.alpha, "beta": best_candidate.beta, "span": list(best_candidate.span)})
        log.info(
            "      + Selected smoothing candidate: alpha=%s beta=%s span=%s error=%.6g (%d candidates)",
            best_candidate.alpha,
            best_candidate.beta,
            best_candidate.span,
            float(best_error.cpu()),
            search.num_candidates,
        )
    else:
        log.info("      + No smoothing candidates generated; using identity smoothing")
    return best_scale, metadata


def _iter_smoothing_progress(
    chunks: tuple[tuple[SmoothCandidate, ...], ...], target_name: str, logger: QuantizationLogger
) -> Iterable[tuple[SmoothCandidate, ...]]:
    """Show smoothing candidate-batch progress when tqdm is installed and useful."""

    if not chunks or not logger.isEnabledFor(20) or not sys.stderr.isatty():
        return chunks
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return chunks
    return tqdm(
        chunks,
        total=len(chunks),
        desc=f"Smoothing {target_name}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )


def _resolve_logger(logger: QuantizationLogger | None) -> QuantizationLogger:
    if logger is None:
        return QuantizationLogger.get_logger(__name__)
    return logger.for_name(__name__)


def _prepare_partition_baselines(
    input_partitions: tuple[torch.Tensor, ...], weight: torch.Tensor, bias: torch.Tensor | None
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...]:
    """Cast each calibration partition once and precompute its unsmoothed reference output
    and per-channel input RMS.

    All three are candidate-invariant across the whole smoothing search, so this only needs
    to run once per target instead of once per candidate.
    """

    baselines: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for partition in input_partitions:
        inputs = partition.to(device=weight.device, dtype=weight.dtype).reshape(-1, weight.shape[1])
        if inputs.numel() == 0:
            continue
        base_rms = inputs.float().pow(2).mean(dim=0).sqrt()
        baselines.append((inputs, linear_output(inputs, weight, bias), base_rms))
    return tuple(baselines)


def _chunk_candidates(
    candidates: tuple[SmoothCandidate, ...], sample_batch_size: int
) -> tuple[tuple[SmoothCandidate, ...], ...]:
    """Split candidates into batches bounded by ``sample_batch_size`` (<= 0 means unbounded)."""

    if not candidates:
        return ()
    if sample_batch_size <= 0 or sample_batch_size >= len(candidates):
        return (candidates,)
    return tuple(
        candidates[index : index + sample_batch_size] for index in range(0, len(candidates), sample_batch_size)
    )


def _score_smoothing_candidates(
    candidates: Sequence[SmoothCandidate],
    partition_baselines: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    spec: DiffusionQuantSpec,
    shared_low_rank: bool,
) -> tuple[torch.Tensor, ...]:
    """Score a batch of smoothing candidates at once by output reconstruction error.

    Smoothed inputs are never materialized: ``(X / s) @ W̃ᵀ == X @ (W̃ / s)ᵀ`` folds the
    per-channel scale into the candidate weight batch, and ``rms(X / s) == rms(X) / s``
    derives the SVD row-weighting from the precomputed baseline RMS. This keeps the
    candidate-batch memory at ``[N, out, in]`` instead of ``[N, rows, in]``.
    """

    num_candidates = len(candidates)
    if num_candidates == 0:
        return ()
    if not partition_baselines:
        return tuple(torch.tensor(float("inf"), device=weight.device) for _ in range(num_candidates))

    smooth_batch = torch.stack(
        [candidate.scale.to(device=weight.device, dtype=weight.dtype) for candidate in candidates]
    )
    smoothed_weight = weight.unsqueeze(0) * smooth_batch.unsqueeze(1)
    partition_errors: list[torch.Tensor] = []
    for inputs, expected, base_rms in partition_baselines:
        low_rank = (
            _batched_low_rank_branch(
                smoothed_weight,
                rank=spec.rank,
                rms=(base_rms.unsqueeze(0) / smooth_batch.float()).clamp_min(1e-6),
                solver=spec.low_rank_solver,
            )
            if spec.rank > 0 and shared_low_rank
            else None
        )
        residual = smoothed_weight
        if low_rank is not None:
            down, up = low_rank
            residual = smoothed_weight - torch.bmm(up, down)
        scale = _batched_weight_scales(residual, group_size=spec.group_size, float_point=spec.precision == "fp4")
        approx_residual = _batched_fake_quantize_weight(residual, scale, float_point=spec.precision == "fp4")
        approx_weight = approx_residual
        if low_rank is not None:
            down, up = low_rank
            approx_weight = approx_weight + torch.bmm(up, down)
        folded = approx_weight / smooth_batch.unsqueeze(1)
        folded2d = folded.permute(2, 0, 1).reshape(weight.shape[1], -1)
        actual = (inputs @ folded2d).view(inputs.shape[0], num_candidates, weight.shape[0])
        if bias is not None:
            actual = actual + bias.view(1, 1, -1)
        partition_errors.append((actual.float() - expected.unsqueeze(1).float()).pow(2).mean(dim=(0, 2)))
    mean_errors = torch.stack(partition_errors).mean(dim=0)
    return tuple(mean_errors[index] for index in range(num_candidates))


def _batched_low_rank_branch(
    weight: torch.Tensor, rank: int, rms: torch.Tensor, solver: LowRankSolverSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched sibling of :func:`low_rank_branch` for scoring many smoothing candidates at once.

    ``weight`` and ``rms`` carry a leading candidate-batch dimension (``[N, out, in]`` and
    ``[N, in]``); ``torch.linalg.svd``/``torch.svd_lowrank`` both natively support it. The
    caller derives each candidate's input RMS as ``rms(X) / s`` instead of materializing
    smoothed inputs.
    """

    num_candidates, out_features, in_features = weight.shape
    rank = min(rank, out_features, in_features)
    if rank == 0:
        down = torch.empty(num_candidates, 0, in_features, dtype=weight.dtype, device=weight.device)
        up = torch.empty(num_candidates, out_features, 0, dtype=weight.dtype, device=weight.device)
        return down, up
    svd_dtype = torch.float32
    rms = rms.to(device=weight.device, dtype=svd_dtype)
    weighted = weight.to(svd_dtype) * rms.unsqueeze(1)
    if solver.svd_backend == "full":
        u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
    else:
        q = min(rank + solver.svd_lowrank_oversample, out_features, in_features)
        u, s, v = torch.svd_lowrank(weighted, q=q, niter=solver.svd_lowrank_niter)
        vh = v.transpose(-1, -2)
    down = (vh[:, :rank] / rms.unsqueeze(1)).to(dtype=weight.dtype, device=weight.device)
    up = (u[:, :, :rank] * s[:, :rank].unsqueeze(1)).to(dtype=weight.dtype, device=weight.device)
    return down, up


def _batched_weight_scales(weight: torch.Tensor, group_size: int, float_point: bool) -> torch.Tensor:
    """Batched sibling of :func:`weight_scales` for scoring many smoothing candidates at once."""

    num_candidates, out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({group_size})")
    groups = in_features // group_size
    max_q = 6 if float_point else 7
    scale = (
        weight.float().view(num_candidates, out_features, groups, group_size).abs().amax(dim=3).clamp_min(1e-6)
        / max_q
    )
    return scale.to(dtype=weight.dtype).view(num_candidates, out_features, 1, groups, 1)


def _batched_fake_quantize_weight(weight: torch.Tensor, scale: torch.Tensor, float_point: bool) -> torch.Tensor:
    """Batched sibling of :func:`fake_quantize_weight` for scoring many smoothing candidates at once."""

    num_candidates, out_features, in_features = weight.shape
    groups = scale.shape[3]
    group_size = in_features // groups
    max_q = 6 if float_point else 7
    qweight = weight.float().view(num_candidates, out_features, groups, group_size) / scale.float().view(
        num_candidates, out_features, groups, 1
    )
    if float_point:
        codebook = fp4_e2m1_codebook(device=weight.device, dtype=torch.float32)
        qcodes = fp_quantize(qweight.view(num_candidates, out_features, in_features), codebook=codebook)
        qweight = codebook[qcodes.long()].view(num_candidates, out_features, groups, group_size)
        return (qweight * scale.float().view(num_candidates, out_features, groups, 1)).view_as(weight).to(
            dtype=weight.dtype
        )
    qweight = qweight.round_().clamp_(-8, max_q)
    return (qweight * scale.float().view(num_candidates, out_features, groups, 1)).view_as(weight).to(
        dtype=weight.dtype
    )


def resolve_input_partitions(
    inputs: torch.Tensor | None, partitions: tuple[torch.Tensor, ...] | None, calibration: CalibrationSpec | None
) -> tuple[torch.Tensor, ...]:
    """Resolve calibration input partitions for quantization consumers."""

    if partitions:
        return partitions
    if inputs is None:
        return ()
    if calibration is None:
        return (inputs,)
    return repartition_tensor(
        inputs, sample_size=calibration.sample_size, sample_batch_size=calibration.sample_batch_size
    )


def smooth_inputs(inputs: torch.Tensor | None, smooth: torch.Tensor) -> torch.Tensor | None:
    """Apply inverse smoothing to activation inputs."""

    if inputs is None:
        return None
    return inputs / smooth.to(device=inputs.device, dtype=inputs.dtype).view(1, -1)
