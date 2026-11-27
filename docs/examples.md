# Example Configs

The `examples/text_to_image/`, `examples/image_to_image/`, and
`examples/text_to_video/` folders (each split into `calibrated/` and
`data_free/` scripts) contain user-side configs for the diffusion
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
| `text_to_image/quantize_qwen_image_2_1.py` | `Qwen/Qwen-Image-2.1` | 40 steps, 1024², CFG off (`--true-cfg-scale` to enable), calib batch 1 (required) | Generic-scanner targets (block linears as SVDQ, shared `modulation.1` as AWQ) with chained single-stream block scopes; calibrates with the pipeline KV cache disabled; keeps 32768 activation rows per target (above the 12288-wide `img_mlp.out` input); `--gptq` enables GPTQ residual rounding |

## Command Matrix

Run one model and precision:

```bash
python examples/text_to_image/calibrated/quantize_flux1_schnell.py --precision int4
python examples/text_to_image/calibrated/quantize_flux1_schnell.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_flux1_dev.py --precision int4
python examples/text_to_image/calibrated/quantize_flux1_dev.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_flux2_klein_4b.py --precision int4
python examples/text_to_image/calibrated/quantize_flux2_klein_4b.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_gptq_flux2_klein_4b.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_flux2_klein_9b.py --precision int4
python examples/text_to_image/calibrated/quantize_flux2_klein_9b.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_pixart_sigma.py --precision int4
python examples/text_to_image/calibrated/quantize_pixart_sigma.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_sana_1_6b.py --precision int4
python examples/text_to_image/calibrated/quantize_sana_1_6b.py --precision nvfp4
python examples/image_to_image/calibrated/quantize_longcat_image_edit.py --precision int4
python examples/image_to_image/calibrated/quantize_longcat_image_edit.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_ernie_image.py --precision int4
python examples/text_to_image/calibrated/quantize_ernie_image.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_ernie_image_turbo.py --precision int4
python examples/text_to_image/calibrated/quantize_ernie_image_turbo.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_lens_turbo.py --precision int4
python examples/text_to_image/calibrated/quantize_lens_turbo.py --precision nvfp4
python examples/text_to_image/calibrated/quantize_qwen_image_2_1.py --precision int4
python examples/text_to_image/calibrated/quantize_qwen_image_2_1.py --precision nvfp4
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

### Data-free quantization

Data-free scripts quantize a denoiser without any calibration data. They load
only the transformer weights (no text encoders, VAE or pipeline) and never run
the model. Smoothing uses weight-only scales `1 / absmax(W[:, j]) ** beta`
(`--smooth-exponent`, default 0.5, or 0 with `--hadamard`) and the low-rank
branch is an SVD of the smoothed weight.

Each task folder splits its scripts into `calibrated/` and `data_free/`. A
model with a calibrated script can have a data-free script of the same name
that exports the same model-specific layout:

| Model | Script | Default output |
| --- | --- | --- |
| FLUX.2 [klein] 4B | `examples/text_to_image/data_free/quantize_flux2_klein_4b.py` | `svdq-<precision>_r32-flux2-klein-4b-datafree.safetensors` |
| FLUX.2 [klein] 9B | `examples/text_to_image/data_free/quantize_flux2_klein_9b.py` | `svdq-<precision>_r32-flux2-klein-9b-datafree.safetensors` |
| LTX-2.3 | `examples/text_to_video/data_free/quantize_ltx2_3.py` | `svdq-<precision>_r32-ltx2.3-datafree.safetensors` |
| LTX-2.3 Distilled | `examples/text_to_video/data_free/quantize_ltx2_3_distilled.py` | `svdq-<precision>_r32-ltx2.3-distilled-datafree.safetensors` |

Any other Diffusers model can use the generic `data_free/quantize_hf.py`
(`examples/text_to_image/data_free/quantize_hf.py` or
`examples/text_to_video/data_free/quantize_hf.py`), which finds layers with the
same generic scanner as `calibrated/quantize_hf.py`.

```bash
python examples/text_to_image/data_free/quantize_flux2_klein_4b.py --precision nvfp4
python examples/text_to_image/data_free/quantize_flux2_klein_4b.py --hadamard
# -> outputs/checkpoints/svdq-int4_r32-flux2-klein-4b-datafree-hadamard.safetensors
python examples/text_to_video/data_free/quantize_ltx2_3.py --offload-model --compute-device cuda
python examples/text_to_video/data_free/quantize_hf.py MODEL_ID --inspect-config
```

With `--offload-model` the transformer stays on CPU and each target is moved to
`--compute-device` while it is quantized. The default
`--svd-backend svd_lowrank` is a randomized approximation of that SVD, so
repeated runs differ slightly; pass `--svd-backend full` for the exact,
reproducible decomposition. Artifact caches go to
`<cache-dir>/<precision>/artifacts-datafree` (the cache root defaults to the
calibrated script's), so they never overwrite the resume state of a calibrated
run. Files in one cache folder are named per target, so use a separate
`--cache-dir` when comparing different `--smooth-exponent` or `--hadamard`
settings. FLUX.2 klein-9B's `--synthetic-calibration` is different: it
calibrates on synthetic denoising trajectories and stays in
`quantize_flux2_klein_9b.py`.

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
