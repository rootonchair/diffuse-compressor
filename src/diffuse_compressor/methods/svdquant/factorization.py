from __future__ import annotations

import torch

from ...config import LowRankSolverSpec


def low_rank_branch(
    weight: torch.Tensor, rank: int, inputs: torch.Tensor | None = None, *, solver: LowRankSolverSpec | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a weighted or unweighted low-rank branch by SVD."""

    rank = min(rank, min(weight.shape))
    if rank == 0:
        return torch.empty(0, weight.shape[1], dtype=weight.dtype, device=weight.device), torch.empty(
            weight.shape[0], 0, dtype=weight.dtype, device=weight.device
        )
    solver = solver or LowRankSolverSpec()
    svd_dtype = torch.float32
    if inputs is None:
        svd_dtype = torch.float64
        u, s, vh = _factor_low_rank_weight(weight.to(svd_dtype), rank, solver)
        down = vh[:rank].to(dtype=weight.dtype, device=weight.device)
    else:
        rms = inputs.to(device=weight.device, dtype=svd_dtype).pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        weighted = weight.to(svd_dtype) * rms.view(1, -1)
        u, s, vh = _factor_low_rank_weight(weighted, rank, solver)
        down = (vh[:rank] / rms.view(1, -1)).to(dtype=weight.dtype, device=weight.device)
    up = (u[:, :rank] * s[:rank]).to(dtype=weight.dtype, device=weight.device)
    return down, up


def _factor_low_rank_weight(
    weight: torch.Tensor, rank: int, solver: LowRankSolverSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factor a weight matrix with the configured SVD backend."""

    if solver.svd_backend == "full":
        return torch.linalg.svd(weight, full_matrices=False)
    q = min(rank + solver.svd_lowrank_oversample, min(weight.shape))
    u, s, v = torch.svd_lowrank(weight, q=q, niter=solver.svd_lowrank_niter)
    return u, s, v.t()
