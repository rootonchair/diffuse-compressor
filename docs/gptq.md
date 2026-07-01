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

## FLUX.2 Klein 4B Smoke Metrics

The following smoke run compares NVFP4 with and without GPTQ against the
original FLUX.2 Klein 4B output using one qdiff prompt, 512x512 resolution, one
inference step, and torch-dequant evaluation. These numbers are useful PR
evidence for wiring and relative behavior; they are not a benchmark-scale
quality claim.

| Checkpoint | MSE ↓ | MAE ↓ | RMSE ↓ | PSNR ↑ |
| --- | ---: | ---: | ---: | ---: |
| `svdq-nvfp4_r32-flux2-klein-4b-smoke.safetensors` | 0.092596 | 0.241747 | 0.304295 | 10.3341 |
| `svdq-gptq-nvfp4_r32-flux2-klein-4b-smoke.safetensors` | 0.083922 | 0.228643 | 0.289694 | 10.7612 |

Generated artifacts were written under
`outputs/checkpoints/` and `outputs/eval/flux2-klein-4b/` for local inspection.

## FLUX.2 Klein 4B Smoke Timing

The same smoke quantization settings used one calibration sample, 512x512
calibration generation, one inference step, `scope-capture-mode=one-target`,
and `cache-mode=reuse`.

| Run | Wall Time | Notes |
| --- | ---: | --- |
| NVFP4 baseline | 3h 24m 1s | Combined active runtime from the initial run plus a resume that reused 109/120 cached target artifacts |
| NVFP4 + GPTQ | 3h 25m 1s | Single run with GPTQ residual rounding enabled for all 120 targets |

In this smoke run, GPTQ added about 61 seconds of wall time compared with the
completed baseline run. Treat this as local PR evidence; timing varies with GPU,
cache state, sample count, resolution, and replay/offload settings.
