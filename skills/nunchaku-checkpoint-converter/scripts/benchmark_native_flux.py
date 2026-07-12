"""Benchmark an upstream Nunchaku FLUX checkpoint against a converted image."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mem_poll import GpuMemPoller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--bnb-text-encoder",
        default=None,
        help="Optional packaged BNB4 T5 encoder. Omit to benchmark the base model's BF16 T5.",
    )
    parser.add_argument("--precision", choices=("int4", "fp4"), required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--converted-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import diffusers
    import numpy as np
    import torch
    from PIL import Image
    from diffusers import FluxPipeline
    from nunchaku import NunchakuFluxTransformer2dModel
    from transformers import T5EncoderModel

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    text_encoder = (
        T5EncoderModel.from_pretrained(args.bnb_text_encoder, torch_dtype=torch.bfloat16)
        if args.bnb_text_encoder is not None
        else None
    )
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(
        args.checkpoint, precision=args.precision, torch_dtype=torch.bfloat16
    )
    pipeline_kwargs = {"transformer": transformer, "torch_dtype": torch.bfloat16}
    if text_encoder is not None:
        pipeline_kwargs["text_encoder_2"] = text_encoder
    pipe = FluxPipeline.from_pretrained(args.base_model, **pipeline_kwargs).to("cuda")

    def run():
        generator = torch.Generator("cuda").manual_seed(args.seed)
        return pipe(
            prompt=args.prompt,
            generator=generator,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
        ).images[0]

    for _ in range(args.warmup):
        run()
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    result = None
    with GpuMemPoller() as memory_poller:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = run()
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)

    native_path = output_dir / f"{args.tag}_{args.steps}step.png"
    result.save(native_path)
    converted = Image.open(args.converted_image).convert("RGB")
    native = np.asarray(result.convert("RGB")).astype(np.float64)
    converted_array = np.asarray(converted).astype(np.float64)
    if native.shape != converted_array.shape:
        raise ValueError(f"Native shape {native.shape} != converted shape {converted_array.shape}")
    delta = converted_array - native
    comparison = Image.new("RGB", (result.width * 2, result.height))
    comparison.paste(result, (0, 0))
    comparison.paste(converted, (result.width, 0))
    comparison_path = output_dir / "output_comparison.png"
    comparison.save(comparison_path)
    report = {
        "base_model": args.base_model,
        "checkpoint": args.checkpoint,
        "precision": args.precision,
        "text_encoder_2": "bnb4" if args.bnb_text_encoder is not None else "bf16",
        "prompt": args.prompt,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "latency_sec_all": latencies,
        "latency_sec_mean": statistics.mean(latencies),
        "latency_sec_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "max_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_device_memory_used_gib": memory_poller.peak_gib,
        "native_image": str(native_path),
        "converted_image": args.converted_image,
        "comparison_image": str(comparison_path),
        "image_delta_converted_vs_native": {
            "mae_pixel": float(np.abs(delta).mean()),
            "rmse_pixel": float(np.sqrt((delta**2).mean())),
            "max_abs_pixel": int(np.abs(delta).max()),
        },
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    report_path = output_dir / f"{args.tag}_{args.steps}step.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
