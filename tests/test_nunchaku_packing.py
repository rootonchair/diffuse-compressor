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


@pytest.mark.parametrize(
    "precision, group_size, scale_dtypes, layout",
    [
        ("int4", 64, (None,), "naive"),
        ("int4", 64, (None,), "nunchaku"),
        ("fp4", 32, (None,), "naive"),
        ("fp4", 16, (None, "sfp8_e4m3_nan"), "naive"),
        ("fp4", 16, (None, "sfp8_e4m3_nan"), "nunchaku"),
    ],
)
def test_residual_quant_grid_codes_round_trip(precision, group_size, scale_dtypes, layout):
    from torch import nn

    from diffuse_compressor import DiffusionQuantSpec, SvdqTargetQuant, TargetConfig, TargetRule, collect_quant_targets
    from diffuse_compressor.backends.nunchaku.layouts import fake_quantize_weight, residual_quant_grid, weight_scales
    from diffuse_compressor.config import NaiveSvdqLayout, NunchakuSvdqLayout

    model = nn.Sequential(nn.Linear(256, 256))
    weight_layout = NaiveSvdqLayout() if layout == "naive" else NunchakuSvdqLayout()
    config = TargetConfig(
        targets=[
            TargetRule(name="p", modules=["0"], export_name="p", quant=SvdqTargetQuant(weight_layout=weight_layout))
        ]
    )
    target = collect_quant_targets(model, config)[0]
    spec = DiffusionQuantSpec(precision=precision, group_size=group_size, rank=0, weight_scale_dtypes=scale_dtypes)
    torch.manual_seed(0)
    weight = torch.randn(256, 256, dtype=torch.bfloat16)
    scale = weight_scales(weight, group_size, precision == "fp4")

    grid = residual_quant_grid(weight, scale, spec, target, low_rank=None)
    codes = grid.quantize(weight)
    decoded = grid.dequantize(codes, torch.float32)

    assert grid.nunchaku_packed is (layout == "nunchaku")
    assert (grid.subscale is not None) is (len(scale_dtypes) == 2)
    # Decoded values sit on the grid: quantizing them again returns the same codes.
    assert torch.equal(grid.quantize(decoded), codes)
    if grid.subscale is None:
        torch.testing.assert_close(decoded, fake_quantize_weight(weight.float(), scale.float(), precision == "fp4"))
