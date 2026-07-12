"""Write model cards for converted Z-Image-Turbo rank-32 repositories."""

from __future__ import annotations

import json
from pathlib import Path


PROMPT = "A cinematic portrait of a red fox in a misty forest at sunrise, detailed fur, volumetric light"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    for precision in ("int4", "nvfp4"):
        display = "INT4" if precision == "int4" else "NVFP4"
        source_precision = "int4" if precision == "int4" else "fp4"
        output = Path(f"outputs/converted/z-image-turbo-nunchaku-lite-{precision}_r32-bnb4-text-encoder")
        smoke = Path(f"outputs/inference_smoke/z-image-turbo-{precision}")
        converted = load(smoke / f"converted_{precision}_9step.json")
        rows = (
            f"| Converted Diffusers Nunchaku Lite {display} r32 + BNB4 Qwen3 | "
            f"{converted['latency_sec_mean']:.2f} s (stdev {converted['latency_sec_stdev']:.2f} s) | "
            f"{converted['max_device_memory_used_gib']:.2f} GiB |"
        )
        if precision == "nvfp4":
            native = load(smoke / "native_fp4_9step.json")
            rows += (
                f"\n| Original Nunchaku NVFP4 r32 + BF16 Qwen3 | "
                f"{native['latency_sec_mean']:.2f} s (stdev {native['latency_sec_stdev']:.2f} s) | "
                f"{native['max_device_memory_used_gib']:.2f} GiB |"
            )
            delta = native["image_delta_converted_vs_native"]
            quality = f"Same-precision pixel MAE is {delta['mae_pixel']:.2f} and RMSE is {delta['rmse_pixel']:.2f}."
        else:
            cross = load(smoke / "native_fp4_cross_precision_9step.json")
            delta = cross["image_delta_converted_vs_native"]
            quality = (
                "Native Nunchaku 1.x selects NVFP4 kernels on Blackwell, so the comparison uses native NVFP4 as a "
                f"visual reference only (cross-precision MAE {delta['mae_pixel']:.2f}, RMSE {delta['rmse_pixel']:.2f})."
            )
        hardware = "Blackwell" if precision == "nvfp4" else "Turing, Ampere, Ada, or Blackwell (not Hopper)"
        text = f"""---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
pipeline_tag: text-to-image
library_name: diffusers
tags:
  - z-image
  - nunchaku
  - svdquant
  - {precision}
  - quantization
---

# Z-Image-Turbo Nunchaku Lite {display} r32

Diffusers-loadable conversion of:

- Base model: `Tongyi-MAI/Z-Image-Turbo`
- Source repo: `nunchaku-ai/nunchaku-z-image-turbo`
- Source checkpoint: `svdq-{source_precision}_r32-z-image-turbo.safetensors`

The transformer uses `quant_method: nunchaku_lite`, {display} SVDQ, group size {16 if precision == 'nvfp4' else 64}, rank 32, and 238 quantized targets. Packed fused QKV and SwiGLU projections are split in logical layout into the stock Z-Image graph. The Qwen3 text encoder is packaged with BitsAndBytes 4-bit NF4.

## Benchmark

| Checkpoint | Latency | Max VRAM |
| --- | ---: | ---: |
{rows}

RTX 5090, 1024×1024, 9 scheduler steps (8 DiT forwards), guidance scale 0, seed 42, one warmup and three measured runs. VRAM is peak total device usage sampled through `nvidia-smi`. The native row uses the base model's BF16 Qwen3 encoder.

## Output Comparison

![Native reference (left) and converted output (right)](output_comparison.png)

Both images use the same prompt, seed, scheduler, resolution, and step count. {quality}

## Run

Requires the Hugging Face `kernels` package and an NVIDIA {hardware} GPU.

```python
import torch
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "lite-infer/{output.name}",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt={PROMPT!r},
    height=1024,
    width=1024,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("output.png")
```
"""
        (output / "README.md").write_text(text)


if __name__ == "__main__":
    main()
