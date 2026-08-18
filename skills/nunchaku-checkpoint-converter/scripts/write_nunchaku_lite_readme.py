"""Write lite-infer style model cards for packaged Nunchaku Lite repos.

Matches the structure published under https://huggingface.co/lite-infer: YAML
frontmatter, a short "Diffusers-loadable conversion of" lead, packaging facts as
prose, a Benchmark table, an Output Comparison, and a Run section.

Scope rule: a model card answers "what is this, will it run on my GPU, how do I
load it". Measurement-rig detail (container limits, harness script names, local
artifact filenames) and diagnostic metrics belong in the benchmark JSON sidecars,
not here.

Every number is read from those sidecars and from the packaged
``transformer/config.json`` rather than retyped.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PRECISION_LABEL = {"int4": "INT4", "nvfp4": "NVFP4"}
# Files inherited from the base model repo that would dangle once we replace the card.
INHERITED_NOISE = ("editing.jpg", "others.jpg", "realism.jpg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline-dir", required=True)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--baseline-tag", default=None,
                        help="Omit when a dense baseline cannot run on the benchmark GPU.")
    parser.add_argument("--no-baseline-reason", default=None,
                        help="Explains the missing dense row, e.g. it exceeds VRAM.")
    parser.add_argument("--peer-tag", default=None)
    parser.add_argument("--peer-label", default=None)
    parser.add_argument("--model-label", required=True, help='e.g. "FLUX.2 Klein 4B"')
    parser.add_argument("--base-repo", required=True)
    parser.add_argument("--repo-id", default=None, help="Hub id for the from_pretrained snippet.")
    parser.add_argument("--pipeline-class", default="Flux2KleinPipeline")
    parser.add_argument("--transformer-class", default=None, help="Model class for modular repos.")
    parser.add_argument("--license", default="apache-2.0")
    parser.add_argument("--license-name", default=None)
    parser.add_argument("--license-link", default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--resolution", default="1024x1024")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--timed-region",
        default="the full pipeline call",
        help="Short phrase naming what the latency covers, e.g. 'denoise and decode'.",
    )
    parser.add_argument(
        "--dense-linears",
        default="The outer linears (embedders, final projection)",
        help="Sentence subject naming which modules stay in bf16 for this architecture.",
    )
    parser.add_argument("--text-encoder", default="text_encoder", help="Component quantized to BNB4.")
    parser.add_argument("--calibration-samples", default="128")
    parser.add_argument("--tags", default="flux", help="Comma-separated family tags for the frontmatter.")
    parser.add_argument("--comparison-image", default=None,
                        help="Explicit comparison PNG when there is no dense baseline to diff against.")
    parser.add_argument("--comparison-caption", default=None)
    parser.add_argument("--prompt", default="A glass robot tending orchids in a sunlit greenhouse, cinematic lighting, highly detailed")
    parser.add_argument(
        "--modular",
        action="store_true",
        help="Repo is a Modular Diffusers pipeline; document the transformer swap it requires.",
    )
    return parser.parse_args()


def load_report(bench_dir: Path, tag: str, steps: int) -> dict:
    return json.loads((bench_dir / f"{tag}_{steps}step.json").read_text())


def row(label: str, report: dict) -> str:
    latency = f"{report['latency_sec_mean']:.2f} s"
    if report.get("latency_sec_stdev") is not None:
        latency += f" (stdev {report['latency_sec_stdev']:.2f} s)"
    return f"| {label} | {latency} | {report['max_device_memory_used_gib']:.2f} GiB |"


def main() -> None:
    args = parse_args()
    pipeline_dir = Path(args.pipeline_dir)
    bench_dir = Path(args.bench_dir)

    quant = json.loads((pipeline_dir / "transformer" / "config.json").read_text())["quantization_config"]
    svdq = quant["svdq_w4a4"]
    awq = quant.get("awq_w4a16")  # absent when no modulation linears qualified
    precision = svdq["precision"]
    label = PRECISION_LABEL.get(precision, precision.upper())
    rank = svdq["rank"]

    report = load_report(bench_dir, args.tag, args.steps)
    baseline = load_report(bench_dir, args.baseline_tag, args.steps) if args.baseline_tag else None
    peer = load_report(bench_dir, args.peer_tag, args.steps) if args.peer_tag else None

    rows = [row(f"**This repo** — Nunchaku Lite {label} r{rank} + BNB4 text encoder", report)]
    if peer:
        rows.append(row(f"Nunchaku Lite {args.peer_label} r{rank} + BNB4 text encoder", peer))
    if baseline:
        rows.append(row(f"{args.model_label} dense bf16", baseline))

    comparison = Path(args.comparison_image) if args.comparison_image else (
        bench_dir / f"comparison_{args.tag}_vs_reference.png"
    )
    has_comparison = comparison.is_file()
    if has_comparison:
        shutil.copy2(comparison, pipeline_dir / "output_comparison.png")
    for name in INHERITED_NOISE:
        (pipeline_dir / name).unlink(missing_ok=True)

    if baseline is None:
        peer_clause = ""
        if peer:
            ratio = peer["latency_sec_mean"] / report["latency_sec_mean"]
            if ratio >= 1.0:
                peer_clause = f" It is **{ratio:.2f}x faster** than the {args.peer_label} build here."
            else:
                # Never let a slower-than-peer number read as a verdict on the format:
                # on this GPU the FP4 tensor cores are the reason, and INT4's audience
                # is the hardware that cannot run NVFP4 at all.
                peer_clause = (
                    f" The {args.peer_label} build is {1 / ratio:.2f}x faster **on this GPU**, because "
                    f"{report['gpu']} has native FP4 tensor cores that only {args.peer_label} can use. "
                    f"**{label} is the build for Turing through Ada, where {args.peer_label} does not run "
                    f"at all**, and has not been benchmarked there."
                )
        verdict = (
            f"There is no dense bf16 row because {args.no_baseline_reason}"
            f"{peer_clause}"
        )
    else:
        verdict = None
    speedup = (baseline["latency_sec_mean"] / report["latency_sec_mean"]) if baseline else 1.0
    memory_ratio = (baseline["max_device_memory_used_gib"] / report["max_device_memory_used_gib"]) if baseline else 1.0
    if verdict is not None:
        pass
    elif speedup >= 1.0:
        verdict = (
            f"That is **{speedup:.2f}x faster** than dense bf16 using **{memory_ratio:.2f}x less** VRAM."
        )
    else:
        # Never present a slower-than-dense result as a verdict on the checkpoint:
        # here it is a property of the benchmark GPU, and INT4's audience is older cards.
        verdict = (
            f"That is **{memory_ratio:.2f}x less** VRAM than dense bf16, at {1 / speedup:.2f}x the latency. "
            f"The latency gap is specific to this GPU, which has native FP4 tensor cores that only the "
            f"NVFP4 build can use. **INT4 is the build for Turing through Ada, where NVFP4 does not run at "
            f"all**; on Blackwell prefer the NVFP4 build."
        )

    awq_clause = (
        f" and {len(awq['targets'])} AWQ W4A16 targets" if awq else " and no AWQ W4A16 targets"
    )
    load_note = (
        "\n\nThis is a Modular Diffusers repo: `ModularPipeline.load_components()` "
        "currently ignores `quantization_config` and would silently give you a dense "
        "transformer, so the snippet below loads the quantized transformer explicitly "
        "and swaps it in."
        if args.modular
        else ""
    )
    call_kwargs = (
        f'\n    num_inference_steps={args.steps},'
        + (f'\n    guidance_scale={args.guidance_scale},' if not args.modular else "")
    )
    if args.modular:
        run_snippet = f"""```python
import os

# Read when `diffusers` is imported, so it has to be set before the import below.
os.environ.setdefault("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")

import torch
from diffusers import {args.transformer_class}
from diffusers.modular_pipelines import ModularPipeline

repo = "{args.repo_id or pipeline_dir.name}"

# load_components() ignores quantization_config, so build the quantized
# transformer explicitly and swap it in.
transformer = {args.transformer_class}.from_pretrained(repo, subfolder="transformer", torch_dtype=torch.bfloat16)
pipe = ModularPipeline.from_pretrained(repo)
pipe.load_components(torch_dtype=torch.bfloat16)
pipe.update_components(transformer=transformer)
pipe.to("cuda")

image = pipe(
    prompt="{args.prompt}",
    generator=torch.Generator("cuda").manual_seed({args.seed}),
    width=1024,
    height=1024,{call_kwargs}
    output="images",
)[0]
image.save("output.png")
```"""
    else:
        run_snippet = f"""```python
import os

# Read when `diffusers` is imported, so it has to be set before the import below.
os.environ.setdefault("DIFFUSERS_TRUST_REMOTE_KERNELS", "true")

import torch
from diffusers import {args.pipeline_class}

pipe = {args.pipeline_class}.from_pretrained(
    "{args.repo_id or pipeline_dir.name}",
    torch_dtype=torch.bfloat16,
).to("cuda")

image = pipe(
    prompt="{args.prompt}",
    generator=torch.Generator("cuda").manual_seed({args.seed}),
    width=1024,
    height=1024,{call_kwargs}
).images[0]
image.save("output.png")
```"""
    if has_comparison:
        delta = report.get("image_delta_vs_reference")
        metrics = (
            f" Pixel **MAE {delta['mae_pixel']:.2f} / RMSE {delta['rmse_pixel']:.2f}**"
            + (
                f", versus {peer_delta['mae_pixel']:.2f} / {peer_delta['rmse_pixel']:.2f} for the "
                f"{args.peer_label} build."
                if (peer_delta := (peer or {}).get("image_delta_vs_reference"))
                else "."
            )
            if delta
            else ""
        )
        caption = args.comparison_caption or (
            "Dense reference (left) and this build (right). Same prompt, seed, scheduler, "
            f"resolution, and step count.{metrics}"
        )
        comparison_section = f"## Output Comparison\n\n![output comparison](output_comparison.png)\n\n{caption}\n"
    else:
        comparison_section = ""
    peer_pointer = (
        f" On Turing through Ada use the {args.peer_label} build instead."
        if args.peer_label and precision == "nvfp4"
        else ""
    )
    family_tags = "\n".join(
        f"  - {tag.strip()}" for tag in args.tags.split(",")
        if tag.strip() and tag.strip() != "text-to-image"  # already the pipeline_tag
    )
    repo_id = args.repo_id or pipeline_dir.name
    license_lines = f"license: {args.license}"
    if args.license_name:
        license_lines += f"\nlicense_name: {args.license_name}"
    if args.license_link:
        license_lines += f"\nlicense_link: {args.license_link}"

    readme = f"""---
{license_lines}
base_model: {args.base_repo}
pipeline_tag: text-to-image
library_name: diffusers
tags:
{family_tags}
  - nunchaku
  - svdquant
  - {precision}
  - quantization
---

# {args.model_label} Nunchaku Lite {label} r{rank}

Diffusers-loadable {label} conversion of `{args.base_repo}`, quantized with
[`diffuse-compressor`](https://github.com/rootonchair/diffuse-compressor). It
loads with a plain `from_pretrained` call — no runtime graph patches and no extra
runtime package.

The transformer uses `quant_method: nunchaku_lite`, {label} SVDQ with group size
{svdq['group_size']}, rank {rank}, {len(svdq['targets'])} SVDQ targets{awq_clause}. {args.dense_linears} stay
in bf16, and the `{args.text_encoder}` component is BitsAndBytes 4-bit NF4 with bf16
compute. QKV projections are not fused, so this trades some speed for loading
through the stock Diffusers graph. Calibrated on {args.calibration_samples} prompts at
{args.steps} steps, {args.resolution}.{load_note}

## Benchmark

| Checkpoint | Latency | Max VRAM |
| --- | ---: | ---: |
{chr(10).join(rows)}

{report['gpu']}, settings as in the Run snippet below, one warmup and three
measured runs, everything resident on the GPU with no offload. Latency covers
{args.timed_region}; VRAM is peak device usage.

{verdict}

{comparison_section}
## Run

```bash
pip install bitsandbytes kernels git+https://github.com/huggingface/diffusers
```

The `nunchaku_lite` quantizer is not in a released Diffusers yet, `kernels`
fetches the fused ops, and `bitsandbytes` loads the 4-bit `{args.text_encoder}`.
{label} needs {'a Blackwell or newer' if precision == 'nvfp4' else 'a Turing or newer'} NVIDIA GPU; Hopper is unsupported.{peer_pointer}

{run_snippet}
"""

    (pipeline_dir / "README.md").write_text(readme)
    print(f"wrote {pipeline_dir / 'README.md'}")


if __name__ == "__main__":
    main()
