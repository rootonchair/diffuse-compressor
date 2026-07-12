"""Write publish-ready model cards for the six converted FLUX.1 repositories."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("outputs")
PROMPT = "A cinematic photograph of a red fox standing in a misty forest at sunrise, detailed fur, volumetric light"
MODELS = {
    "schnell": {
        "title": "FLUX.1 Schnell",
        "base": "black-forest-labs/FLUX.1-schnell",
        "source": "nunchaku-ai/nunchaku-flux.1-schnell",
        "license": "license: apache-2.0",
        "steps": 4,
        "guidance": 0.0,
    },
    "krea-dev": {
        "title": "FLUX.1 Krea Dev",
        "base": "black-forest-labs/FLUX.1-Krea-dev",
        "source": "nunchaku-ai/nunchaku-flux.1-krea-dev",
        "license": "license: other\nlicense_name: flux-1-dev-non-commercial-license\nlicense_link: https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev/blob/main/LICENSE.md",
        "steps": 28,
        "guidance": 3.5,
    },
    "dev": {
        "title": "FLUX.1 Dev",
        "base": "black-forest-labs/FLUX.1-dev",
        "source": "nunchaku-ai/nunchaku-flux.1-dev",
        "license": "license: other\nlicense_name: flux-1-dev-non-commercial-license\nlicense_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md",
        "steps": 28,
        "guidance": 3.5,
    },
}


def report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    for model, info in MODELS.items():
        for precision in ("int4", "nvfp4"):
            output = ROOT / "converted" / f"flux.1-{model}-nunchaku-lite-{precision}_r32-bnb4-text-encoder"
            smoke = ROOT / "inference_smoke" / f"flux.1-{model}-{precision}"
            steps = info["steps"]
            converted = report(smoke / f"converted_{precision}_{steps}step.json")
            display_precision = "INT4" if precision == "int4" else "NVFP4"
            source_precision = "int4" if precision == "int4" else "fp4"
            checkpoint = f"svdq-{source_precision}_r32-flux.1-{model}.safetensors"
            converted_row = (
                f"| Converted Diffusers Nunchaku Lite {display_precision} r32 + BNB4 T5 | "
                f"{converted['latency_sec_mean']:.2f} s (stdev {converted['latency_sec_stdev']:.2f} s) | "
                f"{converted['max_device_memory_used_gib']:.2f} GiB |"
            )
            compiled_row = ""
            if model == "dev" and precision == "nvfp4":
                compiled = report(
                    ROOT
                    / "inference_smoke"
                    / "flux.1-dev-nvfp4-compile-1024"
                    / "converted_nvfp4_compile_28step.json"
                )
                compiled_row = (
                    f"\n| Converted Diffusers Nunchaku Lite NVFP4 r32 + BNB4 T5 + `torch.compile` | "
                    f"{compiled['latency_sec_mean']:.2f} s (stdev {compiled['latency_sec_stdev']:.2f} s) | "
                    f"{compiled['max_device_memory_used_gib']:.2f} GiB |"
                )
            if precision == "nvfp4":
                native = report(smoke / f"native_fp4_{steps}step.json")
                native_row = (
                    f"\n| Original Nunchaku NVFP4 r32 + BF16 T5 | "
                    f"{native['latency_sec_mean']:.2f} s (stdev {native['latency_sec_stdev']:.2f} s) | "
                    f"{native['max_device_memory_used_gib']:.2f} GiB |"
                )
                quality = native["image_delta_converted_vs_native"]
                comparison_note = (
                    f"The native and converted {display_precision} images use identical inputs. Pixel MAE is "
                    f"{quality['mae_pixel']:.2f} and RMSE is {quality['rmse_pixel']:.2f}."
                )
            else:
                native_row = ""
                cross = report(smoke / f"native_fp4_cross_precision_{steps}step.json")
                quality = cross["image_delta_converted_vs_native"]
                comparison_note = (
                    "Native Nunchaku 1.x refuses INT4 checkpoints on Blackwell GPUs, so a same-precision native "
                    "benchmark was unavailable on the RTX 5090. The comparison uses the native NVFP4 checkpoint "
                    f"as a visual reference only (cross-precision MAE {quality['mae_pixel']:.2f}, RMSE "
                    f"{quality['rmse_pixel']:.2f}); these values are not a conversion-parity metric."
                )
            hardware = (
                "a Blackwell NVIDIA GPU for NVFP4 kernels"
                if precision == "nvfp4"
                else "a Turing, Ampere, Ada, or Blackwell NVIDIA GPU; Hopper is unsupported for INT4 kernels"
            )
            native_benchmark_note = (
                " The native row uses the base model's BF16 T5 encoder." if precision == "nvfp4" else ""
            )
            readme = f"""---
{info['license']}
base_model: {info['base']}
pipeline_tag: text-to-image
library_name: diffusers
tags:
  - flux
  - nunchaku
  - svdquant
  - {precision}
  - quantization
---

# {info['title']} Nunchaku Lite {display_precision} r32

Diffusers-loadable conversion of:

- Base model: `{info['base']}`
- Source repo: `{info['source']}`
- Source checkpoint: `{checkpoint}`

The transformer uses `quant_method: nunchaku_lite`, {display_precision} SVDQ with group size {16 if precision == 'nvfp4' else 64}, runtime rank 64, 418 SVDQ targets, and 76 AWQ W4A16 targets. The CLIP encoder is copied from the base model and T5 `text_encoder_2` is BitsAndBytes 4-bit NF4. Fused QKV modules are split in logical tensor layout; single-block `proj_out` is merged from attention and MLP projections; low-rank tensors are logically padded to rank 64.{' INT4 shifted down-projection biases are compensated for signed-unfused Diffusers execution.' if precision == 'int4' else ' NVFP4 outer scales are reconciled without overflowing FP8 group scales.'}

## Benchmark

| Checkpoint | Latency | Max VRAM |
| --- | ---: | ---: |
{converted_row}{compiled_row}{native_row}

RTX 5090, 1024×1024, {steps} steps, guidance scale {info['guidance']}, one warmup and three measured runs, full GPU placement. VRAM is peak total device usage sampled with `nvidia-smi`, including allocations outside PyTorch's caching allocator.{native_benchmark_note}

## Output Comparison

![Native reference (left) and converted output (right)](output_comparison.png)

Both images use the same prompt, seed 0, scheduler, resolution, and step count. {comparison_note}

## Run

Requires the Hugging Face `kernels` package and {hardware}.

```python
import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "lite-infer/{output.name}",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt={PROMPT!r},
    generator=torch.Generator("cuda").manual_seed(0),
    width=1024,
    height=1024,
    num_inference_steps={steps},
    guidance_scale={info['guidance']},
).images[0]
image.save("output.png")
```
"""
            (output / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
