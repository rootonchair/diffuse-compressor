"""Evaluate BF16 and optional quantized upstream diffusion outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from diffuse_compressor.evaluation import EvaluationSample, EvaluationSpec, evaluate_pipeline_pair
from examples.upstream_diffusion_svdquant import (
    MODEL_DEFAULTS,
    ModelKey,
    Precision,
    UPSTREAM_QDIFF_PROMPT_SOURCE,
    configure_logging,
    standard_prompt_records,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the upstream evaluation CLI parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", choices=tuple(MODEL_DEFAULTS), default="flux.1-schnell")
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default="int4")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--runtime", choices=("none", "nunchaku-lite", "torch-dequant"), default="none")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--prompt-file", default=UPSTREAM_QDIFF_PROMPT_SOURCE)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--skip-bf16", action="store_true")
    parser.add_argument("--skip-quantized", action="store_true")
    return parser


def main() -> None:
    """Run BF16 and optional quantized generation for one upstream model."""

    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    model_key: ModelKey = args.model_key
    precision: Precision = args.precision
    defaults = MODEL_DEFAULTS[model_key]
    torch_dtype = _resolve_torch_dtype(args.torch_dtype or defaults.torch_dtype)
    checkpoint = args.checkpoint or _default_checkpoint(defaults.output_prefix, precision)
    output_dir = args.output_dir or f"outputs/eval/{defaults.output_prefix}/{precision}"
    records = standard_prompt_records(args.num_samples, prompt_file=args.prompt_file)
    samples = [
        EvaluationSample(
            filename=str(record["filename"]),
            prompt=str(record["prompt"]),
            seed=int(record["seed"]),
        )
        for record in records
    ]

    import diffusers

    pipeline_cls = getattr(diffusers, defaults.pipeline_name)
    evaluate_pipeline_pair(
        model_id=defaults.model_id,
        pipeline_cls=pipeline_cls,
        model_key=model_key,
        samples=samples,
        spec=EvaluationSpec(
            output_dir=output_dir,
            checkpoint=checkpoint,
            runtime=args.runtime,
            precision="fp4" if precision == "nvfp4" else "int4",
            height=args.height,
            width=args.width,
            steps=args.steps if args.steps is not None else defaults.steps,
            guidance_scale=args.guidance_scale if args.guidance_scale is not None else defaults.guidance_scale,
            device=args.device,
            torch_dtype=torch_dtype,
            skip_bf16=args.skip_bf16,
            skip_quantized=args.skip_quantized,
        ),
    )


def _default_checkpoint(output_prefix: str, precision: Precision) -> str:
    return f"outputs/checkpoints/svdq-{precision}_r32-{output_prefix}.safetensors"


def _resolve_torch_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name!r}")


if __name__ == "__main__":
    main()
