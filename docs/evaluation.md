# Evaluation Helpers

`diffuse_compressor.runtime` provides a focused evaluation pipeline loader.
It loads a normal pipeline or patches a loaded pipeline with an exported
quantized checkpoint, while your project owns the Dataset, DataLoader,
generation loop, image saving, and metrics.

```python
from diffuse_compressor.runtime import RuntimePipelineSpec, load_evaluation_pipeline

pipe = load_evaluation_pipeline(
    model_id="black-forest-labs/FLUX.1-schnell",
    spec=RuntimePipelineSpec(
        mode="quantized",
        runtime="torch-dequant",
        checkpoint="outputs/checkpoints/svdq-int4_r32-flux.1-schnell.safetensors",
        device="cuda",
    ),
)

for batch in dataloader:
    with torch.inference_mode():
        images = pipe(**batch["pipeline_kwargs"]).images
```

Set `--runtime torch-dequant` to evaluate an exported packed checkpoint through
ordinary PyTorch modules without installing Nunchaku Lite. This path
dequantizes packed weights, folds low-rank and smoothing tensors into module
weights, replays structural patches from the config, and replays calibrated
activation-shift wrappers. By default it does not fake-quantize activations. For
debug parity checks, pass `--torch-dequant-activation-mode input` to fake
quantize/dequantize SVDQuant target inputs with a dynamic per-row/per-group
quantizer that applies each target's smoothing factor before quantization.
Static exported output ranges are not replayed in this path because the
Nunchaku W4A4 runtime quantizes the next target input dynamically instead.
W4A16 extra-weight targets remain weight-only. This mode is an approximation
of the fused Nunchaku W4A4 kernels and is intended for correctness/debug
evaluation rather than performance.

Set `--runtime nunchaku-lite` to evaluate through Nunchaku Lite. Manifest
checkpoints declare their runtime ABI in safetensors metadata. For older or
target-specific checkpoints that rely on adapter options, evaluation reads
the adjacent checkpoint config for values such as low-rank branch rank.

## Image Generation Benchmark Example

For a fuller DeepCompressor-style image-generation run where original and
quantized models are evaluated separately, see:

```bash
python -m evaluation.evaluate_image_generation \
  --mode original \
  --model-id black-forest-labs/FLUX.1-schnell \
  --steps 4 \
  --guidance-scale 0.0 \
  --height 1024 \
  --width 1024 \
  --benchmark MJHQ \
  --output-dir outputs/eval/flux.1-schnell/original \
  --num-samples 1024

python -m evaluation.evaluate_image_generation \
  --mode quantized \
  --model-id black-forest-labs/FLUX.1-schnell \
  --steps 4 \
  --guidance-scale 0.0 \
  --height 1024 \
  --width 1024 \
  --runtime torch-dequant \
  --checkpoint outputs/checkpoints/svdq-int4_r32-flux.1-schnell.safetensors \
  --benchmark MJHQ \
  --ref-root outputs/eval/flux.1-schnell/original \
  --output-dir outputs/eval/flux.1-schnell/int4 \
  --num-samples 1024
```

Use `--benchmark DCI` for the sDCI prompt/image benchmark instead. For MJHQ and
DCI, the example downloads the benchmark images and prompts through
`datasets`, writes the ground-truth images under `targets/MJHQ-N` or
`targets/sDCI-N`, and uses them for `with_gt` metrics. `--ref-root` remains the
separately generated original-model output root for `with_orig` metrics. The
example imports metric packages only when requested. Install the benchmark and
metric extras with `python -m pip install -e ".[eval]"`.

## LongCat Image Edit

LongCat Image Edit evaluation uses the held-out `test` split by default, while
quantization uses `validation` by default:

```bash
python -m evaluation.evaluate_image_generation \
  --mode quantized \
  --model-id meituan-longcat/LongCat-Image-Edit-Turbo \
  --task image-edit \
  --steps 8 \
  --guidance-scale 1.0 \
  --runtime nunchaku-lite \
  --nunchaku-lite-target manifest \
  --checkpoint outputs/checkpoints/svdq-nvfp4_r32-longcat-image-edit.safetensors \
  --precision nvfp4 \
  --benchmark NHR-Edit-Change_Only \
  --pipeline-offload model \
  --output-dir outputs/eval/longcat-image-edit/nvfp4 \
  --num-samples 100
```
