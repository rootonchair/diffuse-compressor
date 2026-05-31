# Image-To-Image End-To-End Guide

This guide shows a complete image-to-image path using
`meituan-longcat/LongCat-Image-Edit-Turbo`: inspect the target config,
quantize, evaluate, and run single-image inference.

The example uses the model-specific config in
`examples/image_to_image/quantize_longcat_image_edit.py`. The package core
remains model-agnostic; LongCat module paths, exact manifest targets,
image-edit calibration data, and pipeline call details live in the example.

## 1. Install

Install the package, example dependencies, and dataset support:

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

## 2. Inspect The LongCat Config

Before calibration, inspect the resolved LongCat target config:

```bash
python examples/image_to_image/quantize_longcat_image_edit.py \
  --precision int4 \
  --inspect-config
```

The report should list exact LongCat transformer targets, manifest-compatible
export names, calibration scopes, and no collection errors.

## 3. Quantize LongCat INT4

Run the default INT4 quantization. LongCat quantization uses the validation
split of `VyoJ/NHR-Edit-Change_Only` by default:

```bash
python examples/image_to_image/quantize_longcat_image_edit.py \
  --precision int4 \
  --model-id meituan-longcat/LongCat-Image-Edit-Turbo \
  --image-edit-dataset VyoJ/NHR-Edit-Change_Only \
  --image-edit-split validation \
  --num-samples 128 \
  --height 512 \
  --width 512 \
  --batch-size 1 \
  --output outputs/checkpoints/svdq-int4_r32-longcat-image-edit.safetensors
```

By default this writes:

- checkpoint: `outputs/checkpoints/svdq-int4_r32-longcat-image-edit.safetensors`
- sidecar config: `outputs/checkpoints/svdq-int4_r32-longcat-image-edit.config.yaml`
- calibration/artifact caches: `outputs/calibration/longcat-image-edit/int4/...`
- run logs: `outputs/logs/svdq-int4_r32-longcat-image-edit*.log` and
  `.targets.jsonl`

To build NVFP4-style weights instead:

```bash
python examples/image_to_image/quantize_longcat_image_edit.py \
  --precision nvfp4 \
  --model-id meituan-longcat/LongCat-Image-Edit-Turbo \
  --image-edit-dataset VyoJ/NHR-Edit-Change_Only \
  --image-edit-split validation \
  --num-samples 128 \
  --height 512 \
  --width 512 \
  --batch-size 1 \
  --output outputs/checkpoints/svdq-nvfp4_r32-longcat-image-edit.safetensors
```

## 4. Lower-Memory Quantization

For tighter VRAM and calibration RAM, start with:

```bash
python examples/image_to_image/quantize_longcat_image_edit.py \
  --precision nvfp4 \
  --model-id meituan-longcat/LongCat-Image-Edit-Turbo \
  --image-edit-dataset VyoJ/NHR-Edit-Change_Only \
  --image-edit-split validation \
  --num-samples 64 \
  --cache-num-samples 64 \
  --height 512 \
  --width 512 \
  --batch-size 1 \
  --sample-batch-size 32 \
  --scope-capture-mode one-target \
  --pipeline-offload model \
  --offload-model \
  --compute-device cuda \
  --output outputs/checkpoints/svdq-nvfp4_r32-longcat-image-edit.safetensors
```

Important knobs:

- `--height` and `--width`: control calibration image dimensions. Refresh the
  cache after changing them.
- `--pipeline-offload model` or `sequential`: uses Diffusers/Accelerate CPU
  offload during image-edit calibration forwards.
- `--offload-model --compute-device cuda`: keeps per-target quantization math
  on GPU while allowing the rest of the transformer to move off GPU.
- `--cache-num-samples`: limits reused cached transformer input records.
- `--scope-capture-mode one-target`: lowers peak activation cache memory by
  replaying a scope once per target.
- `--sample-batch-size`: partitions smoothing, range calibration, and low-rank
  scoring work.

See [low_memory_quantization.md](low_memory_quantization.md) for more detail.

## 5. Evaluate Original Outputs

Generate an original-model reference run on the held-out test split:

```bash
python -m evaluation.evaluate_image_generation \
  --mode original \
  --model-id meituan-longcat/LongCat-Image-Edit-Turbo \
  --task image-edit \
  --benchmark NHR-Edit-Change_Only \
  --image-edit-dataset VyoJ/NHR-Edit-Change_Only \
  --image-edit-split test \
  --image-edit-input-size 512 \
  --steps 8 \
  --guidance-scale 1.0 \
  --num-samples 100 \
  --batch-size 1 \
  --pipeline-offload model \
  --output-dir outputs/eval/longcat-image-edit/original
```

The generated samples and benchmark targets are written under the selected
`--output-dir`.

## 6. Evaluate The Quantized Checkpoint

Use `torch-dequant` when you want a pure PyTorch debug path that does not need
Nunchaku Lite:

```bash
python -m evaluation.evaluate_image_generation \
  --mode quantized \
  --runtime torch-dequant \
  --model-id meituan-longcat/LongCat-Image-Edit-Turbo \
  --task image-edit \
  --checkpoint outputs/checkpoints/svdq-nvfp4_r32-longcat-image-edit.safetensors \
  --precision nvfp4 \
  --benchmark NHR-Edit-Change_Only \
  --image-edit-dataset VyoJ/NHR-Edit-Change_Only \
  --image-edit-split test \
  --image-edit-input-size 512 \
  --steps 8 \
  --guidance-scale 1.0 \
  --num-samples 100 \
  --batch-size 1 \
  --pipeline-offload model \
  --ref-root outputs/eval/longcat-image-edit/original \
  --output-dir outputs/eval/longcat-image-edit/nvfp4-torch-dequant
```

Use Nunchaku Lite when `nunchaku_lite` is installed and you want the runtime
path:

```bash
python -m evaluation.evaluate_image_generation \
  --mode quantized \
  --runtime nunchaku-lite \
  --nunchaku-lite-target manifest \
  --model-id meituan-longcat/LongCat-Image-Edit-Turbo \
  --task image-edit \
  --checkpoint outputs/checkpoints/svdq-nvfp4_r32-longcat-image-edit.safetensors \
  --precision nvfp4 \
  --benchmark NHR-Edit-Change_Only \
  --image-edit-dataset VyoJ/NHR-Edit-Change_Only \
  --image-edit-split test \
  --image-edit-input-size 512 \
  --steps 8 \
  --guidance-scale 1.0 \
  --num-samples 100 \
  --batch-size 1 \
  --pipeline-offload model \
  --ref-root outputs/eval/longcat-image-edit/original \
  --output-dir outputs/eval/longcat-image-edit/nvfp4-nunchaku-lite
```

Each evaluation writes generated images plus `results.json` under the selected
`--output-dir`. The `--ref-root` directory is used for generated-vs-original
metrics.

## 7. Run Single-Image Inference

For Nunchaku Lite inference, use `load_nunchaku_pipeline` so the quantized
transformer is installed while the Diffusers pipeline is being loaded:

```python
from pathlib import Path

import torch
from diffusers import LongCatImageEditPipeline
from nunchaku_lite import load_nunchaku_pipeline
from PIL import Image

checkpoint = "outputs/checkpoints/svdq-nvfp4_r32-longcat-image-edit.safetensors"
pipe = load_nunchaku_pipeline(
    "meituan-longcat/LongCat-Image-Edit-Turbo",
    pipeline_cls=LongCatImageEditPipeline,
    checkpoint=checkpoint,
    target="manifest",
    precision="fp4",
    torch_dtype=torch.bfloat16,
    device="cuda",
)
pipe.enable_model_cpu_offload(device="cuda")

input_image = Image.open("inputs/longcat-edit-source.png").convert("RGB")
generator = torch.Generator(device="cuda").manual_seed(0)
image = pipe(
    image=[input_image],
    prompt=["Replace the background with a misty mountain lake"],
    negative_prompt="",
    num_inference_steps=8,
    guidance_scale=1.0,
    generator=[generator],
    height=512,
    width=512,
).images[0]
output = Path("outputs/infer/longcat-image-edit-nvfp4.png")
output.parent.mkdir(parents=True, exist_ok=True)
image.save(output)
```

The `precision="fp4"` runtime option is the value passed to Nunchaku Lite for
checkpoints produced by the `--precision nvfp4` export overlay.

## 8. Reuse Or Refresh Caches

Use `--cache-mode reuse` for normal iteration. Use `--cache-mode refresh` after
changing settings that affect calibration inputs, such as dataset split, image
size, steps, guidance scale, model id, or precision-specific cache paths.

Generated checkpoints, calibration caches, logs, and inference outputs should
stay under `outputs/` or another ignored path.
