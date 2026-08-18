"""Smoothing candidate batches must stay inside a memory budget on wide projections."""

import pytest
import torch

from diffuse_compressor.methods.svdquant.smoothing import _chunk_candidates


def _candidates(count: int) -> tuple[object, ...]:
    return tuple(object() for _ in range(count))


def _cuda_weight(rows: int, columns: int) -> torch.Tensor:
    """Return a CUDA tensor with the given shape without allocating rows*columns floats."""

    return torch.zeros(1, dtype=torch.float32, device="cuda").expand(rows * columns).reshape(rows, columns)


def test_chunking_is_unbounded_without_a_weight():
    candidates = _candidates(39)
    assert _chunk_candidates(candidates, -1) == (candidates,)


def test_sample_batch_size_still_bounds_chunks():
    chunks = _chunk_candidates(_candidates(39), 8)
    assert [len(chunk) for chunk in chunks] == [8, 8, 8, 8, 7]


def test_cpu_weights_are_not_capped():
    candidates = _candidates(39)
    weight = torch.zeros(4, 4, dtype=torch.float32, device="cpu")
    assert _chunk_candidates(candidates, -1, weight) == (candidates,)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="budget cap only applies to CUDA weights")
def test_wide_cuda_weight_caps_chunk_below_sample_batch_size(monkeypatch):
    """A 18432x3072 float32 weight costs 216 MiB per candidate, so 32 would need ~6.8 GiB."""

    monkeypatch.setenv("DIFFUSE_COMPRESSOR_SMOOTH_CANDIDATE_BUDGET_GIB", "2")
    chunks = _chunk_candidates(_candidates(39), 32, _cuda_weight(18432, 3072))

    assert max(len(chunk) for chunk in chunks) <= 9
    assert sum(len(chunk) for chunk in chunks) == 39


@pytest.mark.skipif(not torch.cuda.is_available(), reason="budget cap only applies to CUDA weights")
def test_narrow_cuda_weight_keeps_a_single_chunk(monkeypatch):
    monkeypatch.setenv("DIFFUSE_COMPRESSOR_SMOOTH_CANDIDATE_BUDGET_GIB", "2")
    chunks = _chunk_candidates(_candidates(39), -1, _cuda_weight(3072, 3072))

    assert len(chunks) == 1
