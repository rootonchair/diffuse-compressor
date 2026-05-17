"""User-side FLUX.2 Klein 4B config for Nunchaku Lite-compatible SVDQuant.

The diffuse_compressor library remains model-agnostic. This example names the
Flux2 modules and the fused projection split sizes required by Nunchaku Lite.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from diffuse_compressor import (
    CalibrationScopeRule,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    PatchRule,
    TargetConfig,
    TargetRule,
    quantize_and_export,
)


MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
OUTPUT = "outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors"
CALIBRATION_CACHE = "outputs/calibration/flux2-klein-4b"


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


def flux2_klein_target_config(
    *,
    single_qkv_features: int = 9216,
    single_attn_features: int = 3072,
) -> TargetConfig:
    return TargetConfig(
        patches=[
            PatchRule(
                type="split_linear_output",
                module="single_transformer_blocks.*.attn.to_qkv_mlp_proj",
                args={"splits": [single_qkv_features]},
            ),
            PatchRule(
                type="split_linear",
                module="single_transformer_blocks.*.attn.to_out",
                args={"splits": [single_attn_features]},
            ),
        ],
        calibration_scopes=[
            CalibrationScopeRule("transformer_blocks.{0}", ["transformer_blocks.*"]),
            CalibrationScopeRule("single_transformer_blocks.{0}", ["single_transformer_blocks.*"]),
        ],
        targets=[
            TargetRule(
                name="double_qkv",
                modules=[
                    "transformer_blocks.*.attn.to_q",
                    "transformer_blocks.*.attn.to_k",
                    "transformer_blocks.*.attn.to_v",
                ],
                export_name="transformer_blocks.{0}.attn.to_qkv",
                roles=["q", "k", "v"],
            ),
            TargetRule(
                name="double_added_qkv",
                modules=[
                    "transformer_blocks.*.attn.add_q_proj",
                    "transformer_blocks.*.attn.add_k_proj",
                    "transformer_blocks.*.attn.add_v_proj",
                ],
                export_name="transformer_blocks.{0}.attn.to_added_qkv",
                roles=["add_q", "add_k", "add_v"],
            ),
            TargetRule("double_out", ["transformer_blocks.*.attn.to_out.0"], "transformer_blocks.{0}.attn.to_out.0"),
            TargetRule("double_added_out", ["transformer_blocks.*.attn.to_add_out"], "transformer_blocks.{0}.attn.to_add_out"),
            TargetRule("double_ff_in", ["transformer_blocks.*.ff.linear_in"], "transformer_blocks.{0}.ff.linear_in"),
            TargetRule("double_ff_out", ["transformer_blocks.*.ff.linear_out"], "transformer_blocks.{0}.ff.linear_out"),
            TargetRule(
                "double_context_ff_in",
                ["transformer_blocks.*.ff_context.linear_in"],
                "transformer_blocks.{0}.ff_context.linear_in",
            ),
            TargetRule(
                "double_context_ff_out",
                ["transformer_blocks.*.ff_context.linear_out"],
                "transformer_blocks.{0}.ff_context.linear_out",
            ),
            TargetRule(
                "single_qkv",
                ["single_transformer_blocks.*.attn.to_qkv_mlp_proj.linears.0"],
                "single_transformer_blocks.{0}.attn.qkv_proj",
            ),
            TargetRule(
                "single_mlp_fc1",
                ["single_transformer_blocks.*.attn.to_qkv_mlp_proj.linears.1"],
                "single_transformer_blocks.{0}.attn.mlp_fc1",
            ),
            TargetRule(
                "single_out",
                ["single_transformer_blocks.*.attn.to_out.linears.0"],
                "single_transformer_blocks.{0}.attn.out_proj",
            ),
            TargetRule(
                "single_mlp_fc2",
                ["single_transformer_blocks.*.attn.to_out.linears.1"],
                "single_transformer_blocks.{0}.attn.mlp_fc2",
            ),
        ],
    )


def standard_prompts(num_samples: int = 128) -> list[str]:
    seeds = [
        "A cinematic portrait of a glass robot in a greenhouse",
        "A small cabin beside a frozen lake at sunrise",
        "A red train crossing a stone bridge in the mountains",
        "A detailed product photo of a translucent sneaker",
        "A cozy library with floating paper lanterns",
        "An astronaut repairing a satellite above Earth",
        "A watercolor city street after rain",
        "A dragon-shaped kite flying over a beach festival",
        "A macro photo of dew on a blue flower",
        "A medieval market square lit by candles",
        "A sleek electric motorcycle in a studio",
        "A fantasy castle carved into a cliff",
        "A chef plating noodles in a neon diner",
        "A quiet desert observatory under the Milky Way",
        "A toy car racing through a cardboard city",
        "A fashion editorial shot with reflective fabric",
    ]
    return [seeds[idx % len(seeds)] for idx in range(num_samples)]


def batched_samples(prompts: list[str], batch_size: int) -> list[dict]:
    samples = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch_prompts = prompts[start:end]
        seeds = list(range(start, end))
        filenames = [f"{index:04d}-0" for index in range(start, end)]
        samples.append(
            {
                "filename": filenames[0] if len(filenames) == 1 else filenames,
                "prompt": batch_prompts[0] if len(batch_prompts) == 1 else batch_prompts,
                "seed": seeds[0] if len(seeds) == 1 else seeds,
            }
        )
    return samples


def save_diffusers_images(result: object, sample: dict, output_dir: Path) -> None:
    images = getattr(result, "images", None)
    if images is None:
        raise ValueError("Diffusers calibration output must expose an images attribute")
    filenames = _as_list(sample.get("filename"))
    if not filenames:
        filenames = [f"{int(seed):04d}-0" for seed in _as_list(sample.get("seed"))]
    if len(filenames) != len(images):
        raise ValueError(f"Expected {len(filenames)} image filenames, got {len(images)} images")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, image in zip(filenames, images, strict=True):
        image.save(output_dir / f"{filename}.png")


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()

    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to("cuda")
    prompts = standard_prompts(args.num_samples)

    def run_sample(sample: dict) -> None:
        if isinstance(sample["seed"], list):
            generator = [torch.Generator(device="cuda").manual_seed(int(seed)) for seed in sample["seed"]]
        else:
            generator = torch.Generator(device="cuda").manual_seed(int(sample["seed"]))
        return pipe(
            prompt=sample["prompt"],
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            generator=generator,
        )

    samples = batched_samples(prompts, args.batch_size)
    quantize_and_export(
        model=pipe.transformer,
        spec=DiffusionQuantSpec(precision="int4", rank=32, group_size=64),
        target_config=flux2_klein_target_config(),
        calibration=CalibrationSpec(
            samples=samples,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            cache_dir=CALIBRATION_CACHE,
            cache_mode="reuse",
            seed=0,
            forward_fn=run_sample,
            output_dir=Path(CALIBRATION_CACHE) / "samples",
            output_save_fn=save_diffusers_images,
            shared_input_keys=("txt_ids", "img_ids"),
            max_rows_per_target=4096,
        ),
        export=ExportSpec(output=Path(args.output)),
    )


if __name__ == "__main__":
    main()
