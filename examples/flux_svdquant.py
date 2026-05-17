"""Example DeepCompressor-style Flux target config.

This file is intentionally just user-side configuration: diffuse_compressor
does not hard-code Flux module names.
"""

import logging

import torch
from diffusers import FluxPipeline

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


def configure_logging() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


target_config = TargetConfig(
    patches=[
        PatchRule(
            type="split_linear",
            module="single_transformer_blocks.*.proj_out",
            args={"splits": ["out_features"]},
        )
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
            export_name="transformer_blocks.{0}.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule(
            name="double_context_qkv",
            modules=[
                "transformer_blocks.*.attn.add_q_proj",
                "transformer_blocks.*.attn.add_k_proj",
                "transformer_blocks.*.attn.add_v_proj",
            ],
            export_name="transformer_blocks.{0}.qkv_proj_context",
            roles=["add_q", "add_k", "add_v"],
        ),
        TargetRule("double_out", ["transformer_blocks.*.attn.to_out.0"], "transformer_blocks.{0}.out_proj"),
        TargetRule(
            "double_context_out",
            ["transformer_blocks.*.attn.to_add_out"],
            "transformer_blocks.{0}.out_proj_context",
        ),
        TargetRule("double_mlp_fc1", ["transformer_blocks.*.ff.net.0.proj"], "transformer_blocks.{0}.mlp_fc1"),
        TargetRule("double_mlp_fc2", ["transformer_blocks.*.ff.net.2"], "transformer_blocks.{0}.mlp_fc2"),
        TargetRule(
            "double_context_mlp_fc1",
            ["transformer_blocks.*.ff_context.net.0.proj"],
            "transformer_blocks.{0}.mlp_context_fc1",
        ),
        TargetRule(
            "double_context_mlp_fc2",
            ["transformer_blocks.*.ff_context.net.2"],
            "transformer_blocks.{0}.mlp_context_fc2",
        ),
        TargetRule(
            name="single_qkv",
            modules=[
                "single_transformer_blocks.*.attn.to_q",
                "single_transformer_blocks.*.attn.to_k",
                "single_transformer_blocks.*.attn.to_v",
            ],
            export_name="single_transformer_blocks.{0}.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule(
            "single_out",
            ["single_transformer_blocks.*.proj_out.linears.0"],
            "single_transformer_blocks.{0}.out_proj",
        ),
        TargetRule("single_mlp_fc1", ["single_transformer_blocks.*.proj_mlp"], "single_transformer_blocks.{0}.mlp_fc1"),
        TargetRule(
            "single_mlp_fc2",
            ["single_transformer_blocks.*.proj_out.linears.1"],
            "single_transformer_blocks.{0}.mlp_fc2",
        ),
    ],
)


def main() -> None:
    configure_logging()
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")
    quantize_and_export(
        model=pipe.transformer,
        spec=DiffusionQuantSpec(precision="int4", rank=32, group_size=64, shift_activations=True),
        target_config=target_config,
        calibration=CalibrationSpec(
            prompts="examples/prompts/qdiff.yaml",
            num_samples=128,
            batch_size=16,
            shared_input_keys=("txt_ids", "img_ids"),
        ),
        export=ExportSpec(output="outputs/checkpoints/svdq-int4_r32-flux.1-schnell.safetensors"),
    )


if __name__ == "__main__":
    main()
