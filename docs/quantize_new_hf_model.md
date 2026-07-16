# Quantize A New Hugging Face Model And Load It With `from_pretrained`

This tutorial covers the generic text-to-image path from a dense Hugging Face
Diffusers pipeline to a local, quantized pipeline that loads with the ordinary
`DiffusionPipeline.from_pretrained()` API. FLUX.2 Klein 4B is used as a concrete,
tested example, but the same commands accept another Diffusers model id or a
local pipeline directory.

The workflow has three artifacts:

1. a dense base pipeline used for calibration;
2. a manifest-bearing quantized transformer checkpoint;
3. a packaged Diffusers pipeline containing the base pipeline files and the
   quantized transformer.

`from_pretrained()` loads the third artifact, not the standalone checkpoint.

## 1. Install The Quantizer And Runtime

From the repository root, install the package and example dependencies:

```bash
python -m pip install -e ".[examples]"
```

Install `nunchaku_lite` from its release or private package channel. It is not
installed by this repository's `nunchaku-lite` extra. The installed Diffusers
version must also provide the Nunchaku Lite quantizer and
`NunchakuLiteQuantizationConfig`. When working from a Diffusers checkout, for
example:

```bash
python -m pip install -e /path/to/diffusers
```

To package one or more text encoders with bitsandbytes 4-bit weights, install
the optional runtime dependencies in the same environment:

```bash
python -m pip install -U bitsandbytes accelerate transformers
```

Quantization requires a CUDA-capable machine in practice. Ensure any required
Hugging Face model license has been accepted and authenticate with
`huggingface-cli login` when the base repository is gated.

The commands below use FLUX.2 Klein 4B and INT4 explicitly so that each command
can be copied without first defining shell variables. Replace `--precision
int4` and the `int4` portion of output paths with `nvfp4` to build NVFP4
residual weights instead.

## 2. Inspect What Will Be Quantized

Always inspect a new model before calibration:

```bash
python examples/text_to_image/quantize_hf.py black-forest-labs/FLUX.2-klein-4B \
  --precision int4 \
  --rank 32 \
  --inspect-config
```

The generic scanner treats model structure as follows:

- compatible linears inside homogeneous, repeated `ModuleList` block stacks
  become SVDQ W4A4 targets;
- recognized normalization/modulation linears become AWQ W4A16 targets;
- embeddings, final projections, and other linears outside repeated stacks stay
  in their dense dtype;
- if the model has no repeated homogeneous stack, compatible linears use the
  broad fallback scan.

For FLUX.2 Klein 4B, the expected result is 100 SVDQ targets, 3 AWQ modulation
targets, and 6 outer linears left dense. The report should have no missing
patterns, duplicate export names, or invalid calibration scopes.

Exclude an unwanted target by repeating `--skip` with shell-style path globs:

```bash
python examples/text_to_image/quantize_hf.py black-forest-labs/FLUX.2-klein-4B \
  --inspect-config \
  --skip 'transformer_blocks.*.debug_projection'
```

Do not continue with the generic path merely because inspection completes if
the runtime architecture needs grouped QKV/KV tensors, split fused projections,
synthetic export names, pointwise convolutions, or another structural rewrite.
Those models need a model-specific `TargetConfig`; see
[Adding A New Model](adding_new_model.md).

## 3. Produce The Checkpoint

Use a persistent input/artifact cache for the production run:

```bash
python examples/text_to_image/quantize_hf.py black-forest-labs/FLUX.2-klein-4B \
  --precision int4 \
  --rank 32 \
  --num-samples 128 \
  --batch-size 1 \
  --sample-batch-size 32 \
  --compute-device cuda \
  --pipeline-offload model \
  --svd-backend svd_lowrank
```

`--cache-mode reuse`, the default, reuses compatible input and quantization
artifacts on a later invocation. Use `--cache-mode refresh` after changing
calibration inputs or when a clean recalibration is required.

When the calibration input cache is built or refreshed, its dense generated
images are saved under
`outputs/calibration/flux-2-klein-4b/int4/inputs/samples/`. Inspect a few of
these reference images to confirm that the base pipeline, prompts, and inference
settings are working before evaluating the quantized pipeline.

If memory is tight, retain `--batch-size 1` and add `--offload-model`; use
`--pipeline-offload sequential` for lower VRAM at the cost of speed. See
[Low-Memory Quantization](low_memory_quantization.md) for the full tradeoffs.

The standalone safetensors file contains quantized tensors, untouched dense
transformer tensors, and `quantization_config.runtime_manifest` metadata. It is
an intermediate artifact; it is not yet a complete Diffusers repository.

## 4. Package A Diffusers Pipeline

Combine the quantized transformer with the non-transformer files from the base
pipeline:

```bash
python examples/convert_nunchaku_lite_diffusers.py \
  --checkpoint outputs/checkpoints/svdq-int4_r32-flux-2-klein-4b.safetensors \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --bnb4-text-encoder text_encoder \
  --compute-dtype bfloat16
```

The converter removes the dense transformer weights, copies in the quantized
checkpoint, and writes the compact Nunchaku Lite configuration into
`transformer/config.json`. It deliberately refuses to overwrite an existing
output directory.

Without `--output-dir`, this command writes to
`outputs/diffusers/FLUX.2-klein-4B-nunchaku-lite-int4-bnb4-text-encoder`. The
destination uses the source model name and the transformer precision stored in
the checkpoint; selecting any BNB4 text encoder adds the
`-bnb4-text-encoder` suffix. Pass `--output-dir` when a custom location is
needed; an explicit path is used exactly as provided.

`--bnb4-text-encoder` is optional and repeatable. It converts only the named
Transformers components to bitsandbytes NF4 with BF16 compute; it does not
change the Nunchaku INT4 or NVFP4 transformer precision. Omit the option to keep
all text encoders dense.

Pipelines may expose several text encoders, and quantizing all of them is not
always useful. For example, FLUX.1 has a smaller CLIP `text_encoder` and a much
larger T5 `text_encoder_2`. To match packages that keep CLIP dense and quantize
only T5, use:

```bash
--bnb4-text-encoder text_encoder_2
```

To quantize several encoders explicitly, repeat the option:

```bash
--bnb4-text-encoder text_encoder \
--bnb4-text-encoder text_encoder_2
```

Every selected name must be a Transformers text-encoder component declared in
the pipeline's `model_index.json`. See the official
[Diffusers bitsandbytes guide](https://huggingface.co/docs/diffusers/quantization/bitsandbytes)
for backend requirements and supported hardware.

Conversion also validates that the checkpoint has a Nunchaku Lite runtime
manifest, contains supported SVDQ/AWQ tensors, has no structural patches, and
maps every runtime target from one source module to the identical checkpoint
prefix. A failure here normally means the model requires a model-specific
runtime layout rather than different calibration settings.

## 5. Load With `from_pretrained` And Generate An Image

The packaged directory now behaves like a normal local Diffusers pipeline:

```python
import torch
from diffusers import DiffusionPipeline

pipeline_dir = "outputs/diffusers/FLUX.2-klein-4B-nunchaku-lite-int4-bnb4-text-encoder"

pipe = DiffusionPipeline.from_pretrained(
    pipeline_dir,
    device_map="cuda",
)

generator = torch.Generator(device="cuda").manual_seed(12345)
image = pipe(
    prompt="A glass robot in a greenhouse, cinematic lighting",
    num_inference_steps=4,
    guidance_scale=1.0,
    height=512,
    width=512,
    generator=generator,
).images[0]

image.save("quantized-output.png")
```

## Troubleshooting

### Diffusers reports an unknown `nunchaku_lite` quantization method

Install a Diffusers version that includes Nunchaku Lite support, and verify that
`nunchaku_lite` itself is importable in the same Python environment.

### The converter says the runtime manifest is missing

Read the checkpoint sidecar's `runtime_manifest_diagnostics`. Generic loading
requires one linear source module per identical checkpoint prefix, supported
Nunchaku packing, and no runtime structural patches. See the
[runtime manifest specification](nunchaku_lite_manifest_v1.md).

### Inspection selects the wrong modules

Use `--skip` for isolated false-positive targets. Write a model-specific config
when the architecture needs targets added, grouped, renamed, or structurally
split; the generic CLI intentionally does not provide an include override.

### CUDA runs out of memory

Use model or sequential pipeline offload, `--offload-model`, batch size 1, and a
smaller `--sample-batch-size`. Reducing calibration samples also lowers cache
and replay work, but may reduce final quality.

### The pipeline loads but the image is incoherent

Confirm the loaded SVDQ/AWQ counts, verify that the dense base model generates a
good image with the same prompt and settings, and rerun with a full calibration
set. If the problem persists, inspect target grouping and runtime layout before
tuning numeric quantization settings.
