# GPTQ Residual Rounding

`GptqSpec` enables optional GPTQ rounding for SVDQuant residual weights. It is
applied after smoothing and low-rank residual construction, before final INT4
or FP4/NVFP4 packing. GPTQ uses captured calibration inputs to build a Hessian,
so it requires runnable calibration data or reusable calibration caches.

## Programmatic Usage

```python
from diffuse_compressor import DiffusionQuantSpec, GptqSpec

spec = DiffusionQuantSpec(
    precision="fp4",
    group_size=16,
    weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
    gptq=GptqSpec(
        enabled=True,
        damp_percentage=0.01,
        block_size=128,
        num_inv_tries=250,
        hessian_block_size=512,
    ),
)
```

GPTQ works with both `precision="int4"` and `precision="fp4"`; the NVFP4
example is the FP4 precision path with 16-wide groups and split FP8 scale
metadata.

## FLUX.2 Klein 4B Example

Run the baseline example as usual:

```bash
python examples/text_to_image/quantize_flux2_klein_4b.py --precision nvfp4
```

Run the GPTQ variant:

```bash
python examples/text_to_image/quantize_gptq_flux2_klein_4b.py --precision nvfp4
```

The GPTQ example accepts `--gptq-damp-percentage`, `--gptq-block-size`,
`--gptq-num-inv-tries`, and `--gptq-hessian-block-size`. Its default checkpoint
name uses `svdq-gptq-*`, and its cache key uses `<precision>-gptq` so baseline
and GPTQ artifacts do not collide.

The following table compares NVFP4 quantization with and without GPTQ against
the original FLUX.2 Klein 4B output using one QDiff prompt, 1024x1024
resolution, four inference steps, guidance scale 1.0, and torch-dequant
evaluation:

| Version | Wall time | MSE ↓ | MAE ↓ | RMSE ↓ | PSNR ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| SVDQ | 3h 24m | 0.112661 | 0.275128 | 0.335650 | 9.4823 |
| SVDQ + GPTQ | 3h 25m | 0.094146 | 0.254980 | 0.306832 | 10.2620 |

(Wall time is the quantization time on an NVIDIA RTX PRO 6000 Blackwell GPU.)

These results show that applying GPTQ with SVDQ significantly improves the
quantized model quality in this local evaluation.
