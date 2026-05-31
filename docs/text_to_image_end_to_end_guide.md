# Text-To-Image End-To-End Guide

This guide shows a complete text-to-image path using
`black-forest-labs/FLUX.2-klein-4B`: inspect the target config, quantize,
evaluate, and run single-prompt inference.

The example uses the model-specific config in
`examples/text_to_image/quantize_flux2_klein_4b.py`. The package core remains
model-agnostic; FLUX.2 target paths, fused projection splits, grouped QKV
targets, and calibration defaults live in the example.

## 1. Install

Install the package and example dependencies:

```bash
python -m pip install -e ".[examples]"
```

Install evaluation metric dependencies if you want benchmark metrics:

```bash
python -m pip install -e ".[eval]"
```

For Nunchaku Lite inference or evaluation, install `nunchaku_lite` from its
release or private package channel. Without it, use `--runtime torch-dequant`
for correctness/debug evaluation.

## 2. Inspect The FLUX.2 Klein 4B Config

Before spending time on calibration, inspect the resolved target config:

```bash
python examples/text_to_image/quantize_flux2_klein_4b.py \
  --precision int4 \
  --inspect-config
```

The report should list concrete FLUX.2 block targets, grouped QKV projections,
split fused single-block projections, calibration scopes, and no collection
errors.

## 3. Quantize FLUX.2 Klein 4B INT4

Run the default INT4 quantization:

```bash
python examples/text_to_image/quantize_flux2_klein_4b.py \
  --precision int4 \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --num-samples 128 \
  --batch-size 1 \
  --output outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors
```

By default this writes:

- checkpoint: `outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors`
- sidecar config: `outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.config.yaml`
- calibration/artifact caches: `outputs/calibration/flux2-klein-4b/int4/...`
- run logs: `outputs/logs/svdq-int4_r32-flux2-klein-4b*.log` and
  `.targets.jsonl`

To build NVFP4-style weights instead:

```bash
python examples/text_to_image/quantize_flux2_klein_4b.py \
  --precision nvfp4 \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --num-samples 128 \
  --batch-size 1 \
  --output outputs/checkpoints/svdq-nvfp4_r32-flux2-klein-4b.safetensors
```

## 4. Lower-Memory Quantization

For tighter VRAM and calibration RAM, start with:

```bash
python examples/text_to_image/quantize_flux2_klein_4b.py \
  --precision int4 \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --num-samples 64 \
  --cache-num-samples 64 \
  --batch-size 1 \
  --sample-batch-size 32 \
  --scope-capture-mode one-target \
  --pipeline-offload sequential \
  --offload-model \
  --compute-device cuda \
  --output outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors
```

Important knobs:

- `--pipeline-offload model` or `sequential`: uses Diffusers/Accelerate CPU
  offload during calibration forward passes.
- `--offload-model --compute-device cuda`: keeps per-target quantization math
  on GPU while allowing the rest of the transformer to move off GPU.
- `--batch-size 1`: reduces pipeline and calibration replay batch memory.
- `--cache-num-samples`: limits reused cached transformer input records.
- `--scope-capture-mode one-target`: lowers peak activation cache memory by
  replaying a scope once per target.
- `--sample-batch-size`: partitions smoothing, range calibration, and low-rank
  scoring work.

See [low_memory_quantization.md](low_memory_quantization.md) for more detail.

## 5. Evaluate Original Outputs

Generate an original-model reference run first:

```bash
python -m evaluation.evaluate_image_generation \
  --mode original \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --steps 4 \
  --guidance-scale 1.0 \
  --height 1024 \
  --width 1024 \
  --num-samples 128 \
  --batch-size 1 \
  --pipeline-offload model \
  --output-dir outputs/eval/flux2-klein-4b/original
```

The generated samples are written under:

```text
outputs/eval/flux2-klein-4b/original/samples/qdiff-128/
```

## 6. Evaluate The Quantized Checkpoint

Use `torch-dequant` when you want a pure PyTorch debug path that does not need
Nunchaku Lite:

```bash
python -m evaluation.evaluate_image_generation \
  --mode quantized \
  --runtime torch-dequant \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --checkpoint outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors \
  --steps 4 \
  --guidance-scale 1.0 \
  --height 1024 \
  --width 1024 \
  --num-samples 128 \
  --batch-size 1 \
  --pipeline-offload model \
  --ref-root outputs/eval/flux2-klein-4b/original \
  --output-dir outputs/eval/flux2-klein-4b/int4-torch-dequant
```

Use Nunchaku Lite when `nunchaku_lite` is installed and you want the runtime
path:

```bash
python -m evaluation.evaluate_image_generation \
  --mode quantized \
  --runtime nunchaku-lite \
  --nunchaku-lite-target flux2 \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --checkpoint outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors \
  --precision int4 \
  --steps 4 \
  --guidance-scale 1.0 \
  --height 1024 \
  --width 1024 \
  --num-samples 128 \
  --batch-size 1 \
  --pipeline-offload model \
  --ref-root outputs/eval/flux2-klein-4b/original \
  --output-dir outputs/eval/flux2-klein-4b/int4-nunchaku-lite
```

Each evaluation writes generated images plus `results.json` under the selected
`--output-dir`. The `--ref-root` directory is used for generated-vs-original
metrics.

## 7. Run Single-Prompt Inference

For Nunchaku Lite inference, use `load_nunchaku_pipeline` so the quantized
transformer is installed while the Diffusers pipeline is being loaded:

```python
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from nunchaku_lite import load_nunchaku_pipeline

checkpoint = "outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors"
pipe = load_nunchaku_pipeline(
    "black-forest-labs/FLUX.2-klein-4B",
    pipeline_cls=Flux2KleinPipeline,
    checkpoint=checkpoint,
    target="flux2",
    precision="int4",
    torch_dtype=torch.bfloat16,
    device="cuda",
    adapter_options={"rank": 32},
)
pipe.enable_model_cpu_offload(device="cuda")

generator = torch.Generator(device="cuda").manual_seed(0)
image = pipe(
    prompt="A glass robot in a greenhouse, cinematic lighting",
    height=1024,
    width=1024,
    num_inference_steps=4,
    guidance_scale=1.0,
    generator=generator,
).images[0]
output = Path("outputs/infer/flux2-klein-4b-int4.png")
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output)
```

## 8. Reuse Or Refresh Caches

Use `--cache-mode reuse` for normal iteration. Use `--cache-mode refresh` after
changing settings that affect calibration inputs, such as prompt file, image
size, steps, guidance scale, model id, or precision-specific cache paths.

Generated checkpoints, calibration caches, logs, and inference outputs should
stay under `outputs/` or another ignored path.
