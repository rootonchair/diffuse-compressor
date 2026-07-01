# Example Configs

The `examples/text_to_image/`, `examples/image_to_image/`, and
`examples/text_to_video/` folders contain user-side configs for the diffusion
model families represented by upstream DeepCompressor SVDQuant diffusion
configs. The library core still does not know these architectures; the examples
specify target module patterns, grouped QKV/KV behavior, fused projection
splitting, pointwise Conv2d targets, CLI defaults, prompt handling, and
calibration wiring where needed. Each model's defaults and helper code stay next
to that model's quantization config.

For start-to-finish task guides, see
[text_to_image_end_to_end_guide.md](text_to_image_end_to_end_guide.md) and
[image_to_image_end_to_end_guide.md](image_to_image_end_to_end_guide.md).
For adapting the examples to a different architecture, see
[adding_new_model.md](adding_new_model.md).

Install the package normally for Diffusers-backed example support:

```bash
python -m pip install -e .
```

Install example data-loading extras when using examples that download
calibration images from Hugging Face datasets:

```bash
python -m pip install -e ".[examples]"
```

Nunchaku Lite runtime patching still requires installing `nunchaku_lite` from
its release or private package channel. The `nunchaku-lite` extra is kept as an
explicit marker for that optional runtime, but it does not install a public PyPI
package.

## Supported Examples

| Example | Upstream model id | Defaults | Notes |
| --- | --- | --- | --- |
| `text_to_image/quantize_flux1_schnell.py` | `black-forest-labs/FLUX.1-schnell` | 4 steps, guidance 0.0, calib batch 16 | Flux double/single blocks, grouped QKV/add-QKV, split single block output projection |
| `text_to_image/quantize_flux1_dev.py` | `black-forest-labs/FLUX.1-dev` | 50 steps, guidance 3.5, calib batch 16 | Same target layout as Schnell |
| `text_to_image/quantize_flux2_klein_4b.py` | `black-forest-labs/FLUX.2-klein-4B` | 4 steps, guidance 1.0, calib batch 1 | FLUX.2 double/single blocks, grouped QKV/add-QKV, split fused single-block QKV+MLP projections |
| `text_to_image/quantize_gptq_flux2_klein_4b.py` | `black-forest-labs/FLUX.2-klein-4B` | Same as 4B with GPTQ enabled | Convenience entry point for GPTQ residual rounding with separate `*-gptq` checkpoint and cache paths |
| `text_to_image/quantize_flux2_klein_9b.py` | `black-forest-labs/FLUX.2-klein-9B` | 4 steps, guidance 1.0, calib batch 1 | Same FLUX.2 layout as 4B with wider 9B split sizes |
| `text_to_image/quantize_pixart_sigma.py` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | 20 steps, guidance 4.5, calib batch 256 | Self-attention QKV, cross-attention KV, MLP projections |
| `text_to_image/quantize_sana_1_6b.py` | `Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632` | 20 steps, guidance 4.5, calib batch 256 | Adds pointwise Conv2d FFN targets; depthwise conv is intentionally not quantized |
| `image_to_image/quantize_longcat_image_edit.py` | `meituan-longcat/LongCat-Image-Edit-Turbo` | 8 steps, guidance 1.0, calib batch 1 | Image-edit calibration from the `validation` split of `VyoJ/NHR-Edit-Change_Only`; exact module-path targets for generic manifest loading |
| `text_to_image/quantize_ernie_image.py` | `baidu/ERNIE-Image` | 50 steps, guidance 4.0, calib batch 1 | Exact module-path manifest targets for repeated block SVDQ; prompt enhancer disabled for calibration |
| `text_to_image/quantize_ernie_image_turbo.py` | `baidu/ERNIE-Image-Turbo` | 8 steps, guidance 1.0, calib batch 1 | Same ERNIE manifest layout as the base model with Turbo defaults |
| `text_to_image/quantize_lens_turbo.py` | `microsoft/Lens-Turbo` | 4 steps, guidance 1.0, calib batch 1 | Requires Microsoft's external `lens` package; Lens MMDiT block targets with fused image/text QKV splits |

## Command Matrix

Run one model and precision:

```bash
python examples/text_to_image/quantize_flux1_schnell.py --precision int4
python examples/text_to_image/quantize_flux1_schnell.py --precision nvfp4
python examples/text_to_image/quantize_flux1_dev.py --precision int4
python examples/text_to_image/quantize_flux1_dev.py --precision nvfp4
python examples/text_to_image/quantize_flux2_klein_4b.py --precision int4
python examples/text_to_image/quantize_flux2_klein_4b.py --precision nvfp4
python examples/text_to_image/quantize_gptq_flux2_klein_4b.py --precision nvfp4
python examples/text_to_image/quantize_flux2_klein_9b.py --precision int4
python examples/text_to_image/quantize_flux2_klein_9b.py --precision nvfp4
python examples/text_to_image/quantize_pixart_sigma.py --precision int4
python examples/text_to_image/quantize_pixart_sigma.py --precision nvfp4
python examples/text_to_image/quantize_sana_1_6b.py --precision int4
python examples/text_to_image/quantize_sana_1_6b.py --precision nvfp4
python examples/image_to_image/quantize_longcat_image_edit.py --precision int4
python examples/image_to_image/quantize_longcat_image_edit.py --precision nvfp4
python examples/text_to_image/quantize_ernie_image.py --precision int4
python examples/text_to_image/quantize_ernie_image.py --precision nvfp4
python examples/text_to_image/quantize_ernie_image_turbo.py --precision int4
python examples/text_to_image/quantize_ernie_image_turbo.py --precision nvfp4
python examples/text_to_image/quantize_lens_turbo.py --precision int4
python examples/text_to_image/quantize_lens_turbo.py --precision nvfp4
```

For GPTQ configuration details and custom spec usage, see
[`gptq.md`](gptq.md).

Example CLIs write run logs by default under `outputs/logs`: a text
quantization run log and a `.targets.jsonl` file with per-target elapsed time
and low-rank solver records. Error fields such as `best_error` and `errors` are
available only for `LowRankSolverSpec(mode="search")`; the default
`weighted_svd` mode builds one low-rank branch directly and does not score
candidates. Use `--log-dir <path>` to choose another directory or
`--no-run-log` to disable these files.

To run several upstream examples for one precision, call the model entry points
directly from a shell loop:

```bash
for example in \
  text_to_image/quantize_flux1_schnell.py \
  text_to_image/quantize_flux1_dev.py \
  text_to_image/quantize_flux2_klein_4b.py \
  text_to_image/quantize_flux2_klein_9b.py \
  text_to_image/quantize_pixart_sigma.py \
  text_to_image/quantize_sana_1_6b.py \
  image_to_image/quantize_longcat_image_edit.py \
  text_to_image/quantize_ernie_image.py \
  text_to_image/quantize_ernie_image_turbo.py \
  text_to_image/quantize_lens_turbo.py; do
  python "examples/${example}" --precision int4
done
```

INT4 examples use `rank=32`, `group_size=64`, INT4 residual packing,
activation shift, DeepCompressor-style low-rank search, and projection
smoothing search. NVFP4 examples use `rank=32`, `group_size=16`,
`weight_scale_dtypes=(None, "sfp8_e4m3_nan")`, no activation shift, and the
same search/smoothing flow. For Flux and PixArt NVFP4, extra norm/AdaLN linear
weights are exported as target-level INT4 weight-only overrides to mirror the
upstream precision overlay.

The default output path is
`outputs/checkpoints/svdq-<precision>_r32-<model>.safetensors`; calibration
root input caches and artifact caches are stored under
`outputs/calibration/<model>/<precision>/...` unless `--cache-dir` is supplied.
For lower peak VRAM in the upstream Diffusers examples, combine pipeline CPU
offload with per-target quantization offload, for example
`--pipeline-offload model --offload-model --compute-device cuda`. Calibration
captures remain CPU-backed, and only the active scope or target is moved to the
compute device while it is being replayed or quantized.

## Adapting Target Configs

The `*_target_config()` functions in the corresponding task-folder model files
are meant to be copied and edited for new model architectures. The core
question is not "is this model Flux or PixArt?", but "which modules should
become each exported runtime projection?"

Use [adding_new_model.md](adding_new_model.md) for the full porting workflow:
inspect the module tree, start with a minimal class scan, refine explicit
targets and grouped projections, add patches only for fused modules, add
calibration scopes, inspect the config, then run a tiny quantization/eval/infer
loop. Use [configuration.md](configuration.md) for the detailed rule reference
and small runnable recipes.
