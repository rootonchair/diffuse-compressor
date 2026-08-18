"""Benchmark the denoise+decode stage with the transformer fully resident on the GPU.

Motivation: a large dense pipeline may not fit in VRAM *or* in the container's CPU
RAM, so ``enable_model_cpu_offload`` gets OOM-killed and ``enable_sequential_cpu_offload``
measures weight streaming rather than the model. Neither is comparable against a
quantized build that runs resident.

This runs in two phases so nothing is held at once:

1. Load only the text encoder, encode the prompt, write the embeddings out, exit.
2. Load only transformer + VAE + scheduler, put them on the GPU, and time
   ``pipe(prompt_embeds=...)``.

The timed region is identical for dense and quantized builds: denoise + decode,
with text encoding excluded. That makes the numbers directly comparable.

    python benchmark_denoiser_resident.py --phase encode --model-id <repo> --out embeds.pt
    python benchmark_denoiser_resident.py --phase bench --model-id <repo> --embeds embeds.pt \
        --output-dir <dir> --tag <tag>
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, Flux2Transformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from mem_poll import GpuMemPoller

PROMPT = "A glass robot tending orchids in a sunlit greenhouse, cinematic lighting, highly detailed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("encode", "bench"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--embeds", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reference-image", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--tag", default=None)
    return parser.parse_args()


def encode(args: argparse.Namespace) -> None:
    """Phase 1: text encoder only, on GPU, then persist the embeddings."""

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_id, transformer=None, vae=None, dtype=torch.bfloat16
    )
    pipe.text_encoder.to("cuda")
    with torch.no_grad():
        prompt_embeds = pipe.encode_prompt(prompt=[args.prompt], device=torch.device("cuda"))
    if isinstance(prompt_embeds, tuple):
        prompt_embeds = prompt_embeds[0]
    torch.save(prompt_embeds.cpu(), args.out)
    print(f"wrote {args.out} shape={tuple(prompt_embeds.shape)} dtype={prompt_embeds.dtype}")


def bench(args: argparse.Namespace) -> None:
    """Phase 2: transformer + VAE resident on GPU; time denoise + decode."""

    transformer = Flux2Transformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", dtype=torch.bfloat16
    )
    vae = AutoencoderKLFlux2.from_pretrained(args.model_id, subfolder="vae", dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = Flux2KleinPipeline(
        scheduler=scheduler, vae=vae, text_encoder=None, tokenizer=None, transformer=transformer
    )
    pipe.to("cuda")

    quantized = sum(
        1 for m in transformer.modules() if type(m).__name__ in {"SVDQW4A4Linear", "AWQW4A16Linear"}
    )
    prompt_embeds = torch.load(args.embeds).to("cuda", torch.bfloat16)

    def run():
        generator = torch.Generator("cuda").manual_seed(args.seed)
        return pipe(
            prompt_embeds=prompt_embeds,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        ).images[0]

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies = []
    image = None
    with GpuMemPoller() as memory_poller:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            image = run()
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{args.tag}_{args.steps}step.png"
    image.save(image_path)

    report = {
        "model_id": args.model_id,
        "stage": "denoise+decode, transformer resident on GPU, text encoding excluded",
        "quantized_modules": quantized,
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "latency_sec_mean": statistics.mean(latencies),
        "latency_sec_all": latencies,
        "max_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_device_memory_used_gib": memory_poller.peak_gib,
        "output_image": str(image_path),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    if args.runs > 1:
        report["latency_sec_stdev"] = statistics.stdev(latencies)

    if args.reference_image:
        import numpy as np
        from PIL import Image

        reference = np.asarray(Image.open(args.reference_image).convert("RGB")).astype(np.float64)
        ours = np.asarray(image.convert("RGB")).astype(np.float64)
        if ours.shape == reference.shape:
            delta = ours - reference
            report["image_delta_vs_reference"] = {
                "mae_pixel": float(np.abs(delta).mean()),
                "rmse_pixel": float(np.sqrt((delta**2).mean())),
            }
            comparison = Image.new("RGB", (reference.shape[1] * 2, reference.shape[0]))
            comparison.paste(Image.open(args.reference_image).convert("RGB"), (0, 0))
            comparison.paste(image, (reference.shape[1], 0))
            comparison_path = output_dir / f"comparison_{args.tag}_vs_reference.png"
            comparison.save(comparison_path)
            report["comparison_image"] = str(comparison_path)

    (output_dir / f"{args.tag}_{args.steps}step.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.phase == "encode":
        encode(args)
    else:
        bench(args)
    gc.collect()


if __name__ == "__main__":
    main()
