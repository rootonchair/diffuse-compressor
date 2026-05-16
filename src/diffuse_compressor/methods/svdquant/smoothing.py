from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

from ...config import SmoothSpec


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


def smooth_alpha_beta_pairs(spec: SmoothSpec) -> list[tuple[float, float]]:
    """Generate alpha/beta smoothing exponent pairs.

    Args:
        spec: Smoothing configuration.

    Returns:
        Ordered exponent pairs for manual or grid-search smoothing.
    """

    if spec.strategy == "manual":
        if spec.beta < 0:
            if not 0 <= spec.alpha <= 1:
                raise ValueError("manual smooth alpha must be in [0, 1] when beta is negative")
            return [(spec.alpha, 1 - spec.alpha)]
        if spec.alpha < 0:
            if not 0 <= spec.beta <= 1:
                raise ValueError("manual smooth beta must be in [0, 1] when alpha is negative")
            return [(1 - spec.beta, spec.beta)]
        if not 0 <= spec.alpha <= 1 or not 0 <= spec.beta <= 1:
            raise ValueError("manual smooth alpha and beta must be in [0, 1]")
        if spec.alpha == 0 and spec.beta == 0:
            raise ValueError("manual smooth alpha and beta cannot both be zero")
        return [(spec.alpha, spec.beta)]

    choices = [i / spec.num_grids for i in range(1, spec.num_grids)]
    if spec.alpha > 0:
        if spec.beta > 0:
            return [(0, 0)] + [(alpha, alpha) for alpha in choices]
        if spec.beta == 0:
            return [(0, 0)] + [(alpha, 0) for alpha in choices]
        if spec.beta == -1:
            return [(0, 0)] + [(alpha, 1 - alpha) for alpha in choices]
        if spec.beta == -2:
            return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices]
        return [(0, 0)] + [(alpha, 0) for alpha in choices] + [
            (alpha, beta) for alpha in choices for beta in choices
        ]
    if spec.alpha == 0:
        if spec.beta > 0:
            return [(0, 0)] + [(0, beta) for beta in choices]
        if spec.beta == 0:
            return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(0, beta) for beta in choices]
        if spec.beta == -1:
            return [(0, 0)] + [(0, beta) for beta in choices] + [(alpha, 1 - alpha) for alpha in choices]
        if spec.beta == -2:
            return (
                [(0, 0)]
                + [(alpha, 0) for alpha in choices]
                + [(0, beta) for beta in choices]
                + [(alpha, 1 - alpha) for alpha in choices]
            )
        return (
            [(0, 0)]
            + [(alpha, 0) for alpha in choices]
            + [(0, beta) for beta in choices]
            + [(alpha, beta) for alpha in choices for beta in choices]
        )
    if spec.beta > 0:
        return [(0, 0)] + [(1 - beta, beta) for beta in choices]
    if spec.beta == 0:
        return [(0, 0)] + [(alpha, 0) for alpha in choices]
    if spec.beta == -1:
        return [(0, 0)] + [(alpha, 1 - alpha) for alpha in choices]
    if spec.beta == -2:
        return [(0, 0)] + [(alpha, 0) for alpha in choices] + [(alpha, 1 - alpha) for alpha in choices]
    return [(0, 0)] + [(alpha, 0) for alpha in choices] + [
        (alpha, beta) for alpha in choices for beta in choices
    ]


def iter_smooth_candidates(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    spec: SmoothSpec,
) -> Iterator[SmoothCandidate]:
    """Yield smoothing candidates from calibration inputs and weights.

    Args:
        inputs: Calibration input rows.
        weight: Target weight matrix.
        spec: Smoothing configuration.

    Yields:
        Candidate scale tensors and metadata.
    """

    inputs = _sample_inputs(inputs, spec.sample_size).to(dtype=torch.float32)
    weight = weight.to(dtype=torch.float32)
    for alpha_span_name, beta_span_name in spec.spans:
        alpha_span = _activation_span(inputs, alpha_span_name, eps=spec.eps)
        beta_span = _weight_span(weight, beta_span_name, eps=spec.eps)
        for alpha, beta in smooth_alpha_beta_pairs(spec):
            if alpha == 0 and beta == 0:
                scale = torch.ones_like(alpha_span)
            else:
                scale = alpha_span.pow(alpha)
                if beta > 0:
                    scale = scale / beta_span.pow(beta)
                scale = _sanitize_scale(scale, eps=spec.eps)
            yield SmoothCandidate(scale=scale.to(device=weight.device), alpha=alpha, beta=beta, span=(alpha_span_name, beta_span_name))


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
