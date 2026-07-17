import pytest
import torch

from diffuse_compressor.backends.nunchaku.packing import fp4_e2m1_codebook, fp_quantize


def _reference_fp_quantize(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    return (x.unsqueeze(-1) - codebook.unsqueeze(0)).abs().argmin(dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fp_quantize_bucketization_matches_e2m1_distance_search(dtype):
    codebook = fp4_e2m1_codebook(dtype=dtype)
    midpoints = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=dtype)
    lower = torch.nextafter(midpoints, torch.full_like(midpoints, float("-inf")))
    upper = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
    special = torch.cat(
        [
            codebook,
            midpoints,
            -midpoints,
            lower,
            upper,
            torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan")], dtype=dtype),
        ]
    )
    generator = torch.Generator().manual_seed(0)
    random = (torch.randn(100_000, generator=generator) * 10).to(dtype)
    values = torch.cat([special, random])

    expected = _reference_fp_quantize(values, codebook)

    assert torch.equal(fp_quantize(values), expected)


def test_fp_quantize_custom_codebook_uses_equivalent_chunked_search():
    values = torch.linspace(-3, 3, 100_003)
    codebook = torch.tensor([-3.0, -1.0, 0.0, 2.0])

    expected = _reference_fp_quantize(values, codebook)

    assert torch.equal(fp_quantize(values, codebook=codebook), expected)
