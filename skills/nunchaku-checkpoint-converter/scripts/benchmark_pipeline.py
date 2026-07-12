"""Model-agnostic benchmark runner for a converted Nunchaku Lite pipeline directory.

Measures end-to-end latency (warmup + N measured runs), peak CUDA memory, and
optionally pixel MAE/RMSE against a reference image, then writes the output
image, a side-by-side comparison, and a JSON sidecar with all parameters.

Example:

    python benchmark_pipeline.py \
        --pipeline-dir outputs/converted/flux.1-kontext-dev-nunchaku-lite-int4_r32-bnb4-text-encoder \
        --pipeline-class FluxKontextPipeline \
        --prompt "Make Pikachu hold a sign that says 'Nunchaku is awesome'" \
        --image input.png \
        --steps 28 --seed 0 \
        --pipe-kwarg guidance_scale=2.5 --pipe-kwarg max_area=262144 \
        --reference-image native_baseline.png \
        --output-dir outputs/benchmark/flux-kontext-int4 --tag converted_int4
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from mem_poll import GpuMemPoller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--pipeline-class", required=True, help="Diffusers pipeline class name, e.g. FluxKontextPipeline")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", default=None, help="Optional input image for edit pipelines")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pipe-kwarg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra pipeline call kwarg; VALUE is parsed as JSON, falling back to string",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--offload", choices=("none", "model", "sequential"), default="none")
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Compile the transformer with torch.compile before warmup; compile/warmup time is reported separately.",
    )
    parser.add_argument("--reference-image", default=None, help="Baseline image for MAE/RMSE and comparison PNG")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True, help="File name stem for outputs")
    return parser.parse_args()


def parse_pipe_kwargs(pairs: list[str]) -> dict:
    kwargs = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if not key or not raw:
            raise ValueError(f"--pipe-kwarg must be KEY=VALUE, got {pair!r}")
        try:
            kwargs[key] = json.loads(raw)
        except json.JSONDecodeError:
            kwargs[key] = raw
    return kwargs


def main() -> None:
    args = parse_args()
    import diffusers
    import numpy as np
    import torch
    from PIL import Image

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_cls = getattr(diffusers, args.pipeline_class)
    pipe = pipeline_cls.from_pretrained(args.pipeline_dir, torch_dtype=torch.bfloat16)
    if args.offload == "model":
        pipe.enable_model_cpu_offload()
    elif args.offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.to("cuda")
    if args.torch_compile:
        pipe.transformer = torch.compile(pipe.transformer)

    call_kwargs = {"prompt": args.prompt, "num_inference_steps": args.steps}
    call_kwargs.update(parse_pipe_kwargs(args.pipe_kwarg))
    if args.image is not None:
        call_kwargs["image"] = Image.open(args.image).convert("RGB")

    def run():
        generator = torch.Generator("cuda").manual_seed(args.seed)
        return pipe(generator=generator, **call_kwargs).images[0]

    torch.cuda.synchronize()
    warmup_started = time.perf_counter()
    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    warmup_sec = time.perf_counter() - warmup_started

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    result = None
    with GpuMemPoller() as memory_poller:
        for _ in range(args.runs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = run()
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    output_png = output_dir / f"{args.tag}_{args.steps}step.png"
    result.save(output_png)

    report = {
        "pipeline_dir": args.pipeline_dir,
        "pipeline_class": args.pipeline_class,
        "prompt": args.prompt,
        "input_image": args.image,
        "seed": args.seed,
        "num_inference_steps": args.steps,
        "pipe_kwargs": parse_pipe_kwargs(args.pipe_kwarg),
        "offload": args.offload,
        "warmup_runs": args.warmup,
        "warmup_and_compile_sec": warmup_sec,
        "torch_compile": args.torch_compile,
        "measured_runs": args.runs,
        "latency_sec_mean": statistics.mean(latencies),
        "latency_sec_all": latencies,
        "max_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_device_memory_used_gib": memory_poller.peak_gib,
        "output_image": str(output_png),
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    if args.runs > 1:
        report["latency_sec_stdev"] = statistics.stdev(latencies)

    if args.reference_image:
        reference = Image.open(args.reference_image).convert("RGB")
        ours = np.asarray(result.convert("RGB")).astype(np.float64)
        base = np.asarray(reference).astype(np.float64)
        if ours.shape != base.shape:
            raise ValueError(f"Output shape {ours.shape} != reference shape {base.shape}")
        delta = ours - base
        report["reference_image"] = args.reference_image
        report["image_delta_vs_reference"] = {
            "mae_pixel": float(np.abs(delta).mean()),
            "rmse_pixel": float(np.sqrt((delta**2).mean())),
            "max_abs_pixel": int(np.abs(delta).max()),
        }
        comparison = Image.new("RGB", (base.shape[1] * 2, base.shape[0]))
        comparison.paste(reference, (0, 0))
        comparison.paste(result, (base.shape[1], 0))
        comparison_png = output_dir / f"comparison_{args.tag}_vs_reference.png"
        comparison.save(comparison_png)
        report["comparison_image"] = str(comparison_png)

    report_path = output_dir / f"{args.tag}_{args.steps}step.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
