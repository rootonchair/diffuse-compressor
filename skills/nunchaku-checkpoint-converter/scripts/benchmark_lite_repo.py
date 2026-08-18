"""Benchmark a packaged Nunchaku Lite repo, standard or Modular, into the card JSON schema.

Handles the two things ``benchmark_pipeline.py`` cannot:

* **Modular repos.** ``ModularPipeline.load_components()`` ignores
  ``quantization_config`` and silently yields a dense transformer, so the
  quantized transformer is loaded explicitly and swapped in with
  ``update_components``.
* **Denoisers too large to sit beside their text encoder.** ``--embeds`` runs the
  timed region on precomputed prompt embeddings, so a dense baseline can be
  measured fully resident instead of streaming under offload.

Emits the same fields ``write_nunchaku_lite_readme.py`` reads.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from mem_poll import GpuMemPoller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--pipeline-class", default=None, help="Standard pipeline class; omit for modular repos.")
    parser.add_argument("--transformer-class", default=None, help="Required for modular repos.")
    parser.add_argument("--modular", action="store_true")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--reference-image", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    return parser.parse_args()


def build_pipeline(args):
    """Return (pipe, replaced_module_count)."""

    import diffusers
    from diffusers.quantizers.nunchaku.utils import AWQW4A16Linear, SVDQW4A4Linear

    if args.modular:
        from diffusers.modular_pipelines import ModularPipeline

        pipe = ModularPipeline.from_pretrained(args.pipeline_dir)
        pipe.load_components(torch_dtype=torch.bfloat16)
        if args.transformer_class:
            model_cls = getattr(diffusers, args.transformer_class)
            # load_components() ignores quantization_config; swap in the quantized one.
            pipe.update_components(
                transformer=model_cls.from_pretrained(
                    args.pipeline_dir, subfolder="transformer", torch_dtype=torch.bfloat16
                )
            )
    else:
        pipe = getattr(diffusers, args.pipeline_class).from_pretrained(
            args.pipeline_dir, torch_dtype=torch.bfloat16
        )
    pipe.to("cuda")
    replaced = sum(
        1 for m in pipe.transformer.modules() if isinstance(m, (SVDQW4A4Linear, AWQW4A16Linear))
    )
    return pipe, replaced


def main() -> None:
    args = parse_args()
    pipe, replaced = build_pipeline(args)

    call = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "height": args.height,
        "width": args.width,
    }
    if args.guidance_scale is not None:
        call["guidance_scale"] = args.guidance_scale
    if args.modular:
        call["output"] = "images"

    def run():
        generator = torch.Generator("cuda").manual_seed(args.seed)
        out = pipe(generator=generator, **call)
        return out[0] if args.modular else out.images[0]

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    latencies, image = [], None
    with GpuMemPoller() as poller:
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
        "pipeline_dir": args.pipeline_dir,
        "modular": args.modular,
        "replaced_modules": replaced,
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
        "max_device_memory_used_gib": poller.peak_gib,
        "output_image": str(image_path),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    if args.runs > 1:
        report["latency_sec_stdev"] = statistics.stdev(latencies)

    if args.reference_image:
        import numpy as np
        from PIL import Image

        ref = Image.open(args.reference_image).convert("RGB")
        a, b = np.asarray(image.convert("RGB"), dtype=np.float64), np.asarray(ref, dtype=np.float64)
        if a.shape == b.shape:
            report["image_delta_vs_reference"] = {
                "mae_pixel": float(np.abs(a - b).mean()),
                "rmse_pixel": float(np.sqrt(((a - b) ** 2).mean())),
            }
            side = Image.new("RGB", (b.shape[1] * 2, b.shape[0]))
            side.paste(ref, (0, 0))
            side.paste(image, (b.shape[1], 0))
            side.save(output_dir / f"comparison_{args.tag}_vs_reference.png")

    (output_dir / f"{args.tag}_{args.steps}step.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"{args.tag}: replaced={replaced} {report['latency_sec_mean']:.2f}s "
          f"{report['max_device_memory_used_gib']:.2f} GiB")


if __name__ == "__main__":
    main()
