import pytest
import torch

from diffuse_compressor.backends.nunchaku.packing import fp4_e2m1_codebook, fp_quantize


def _reference_e2m1_bucket_quantize(x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    positive = codebook[:8]
    thresholds = (positive[:-1] + positive[1:]) / 2
    codes = torch.bucketize(x.abs(), thresholds, right=False)
    negative = x.lt(0) & codes.ne(0)
    codes.add_(negative, alpha=8)
    codes.masked_fill_(~x.isfinite(), 0)
    return codes


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_fp_quantize_bucketization_uses_e2m1_codebook_thresholds(dtype):
    codebook = fp4_e2m1_codebook(dtype=dtype)
    midpoints = (codebook[:7] + codebook[1:8]) / 2
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

    expected = _reference_e2m1_bucket_quantize(values, codebook)

    assert torch.equal(fp_quantize(values), expected)


def test_fp_quantize_custom_e2m1_codebook_derives_thresholds():
    values = torch.tensor([-3.1, -3.0, -2.0, -1.9, -0.24, 0.24, 1.9, 2.0, 3.0, 3.1])
    codebook = fp4_e2m1_codebook() * 0.5

    expected = _reference_e2m1_bucket_quantize(values, codebook)

    assert torch.equal(fp_quantize(values, codebook=codebook), expected)


def test_fp_quantize_rejects_unsupported_custom_codebook():
    values = torch.linspace(-3, 3, 17)
    codebook = torch.tensor([-3.0, -1.0, 0.0, 2.0])

    with pytest.raises(ValueError, match="E2M1-compatible"):
        fp_quantize(values, codebook=codebook)
