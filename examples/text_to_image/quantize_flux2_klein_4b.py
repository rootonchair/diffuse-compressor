"""Quantize FLUX.2 Klein 4B with upstream DeepCompressor-style SVDQuant config."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from diffuse_compressor import (
    CalibrationScopeRule,
    CalibrationSpec,
    ExportSpec,
    LoggingConfig,
    NunchakuSvdqLayout,
    PatchRule,
    QuantizationCacheSpec,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    inspect_target_config,
    quantize_and_export,
)

try:
    from .utils import (
        DEFAULT_QDIFF_PROMPT_FILE as DEFAULT_QDIFF_PROMPT_FILE,
        Precision,
        batched_samples,
        default_arg_parser,
        load_pipeline,
        pipeline_forward_fn,
        save_diffusers_images,
        standard_prompt_records,
        svdquant_spec,
    )
except ImportError:
    from utils import (
        DEFAULT_QDIFF_PROMPT_FILE as DEFAULT_QDIFF_PROMPT_FILE,
        Precision,
        batched_samples,
        default_arg_parser,
        load_pipeline,
        pipeline_forward_fn,
        save_diffusers_images,
        standard_prompt_records,
        svdquant_spec,
    )


def _flux_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next Flux block input from the previous block replay."""

    if not isinstance(replay.output, tuple) or len(replay.output) != 2:
        raise TypeError(
            "Flux block replay output must be (encoder_hidden_states, hidden_states)"
        )
    encoder_hidden_states, hidden_states = replay.output
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = hidden_states
        kwargs["encoder_hidden_states"] = encoder_hidden_states
        return (), kwargs
    args = list(replay.args)
    if len(args) < 2:
        raise TypeError(
            "Flux block replay args must include hidden_states and encoder_hidden_states"
        )
    args[0] = hidden_states
    args[1] = encoder_hidden_states
    return tuple(args), {}


def _flux2_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next Flux.2 block input from the previous block replay."""

    if isinstance(replay.output, tuple):
        return _flux_block_prev_replay_transform(replay)
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = replay.output
        kwargs["encoder_hidden_states"] = None
        return (), kwargs
    args = list(replay.args)
    if not args:
        raise TypeError("Flux.2 block replay args must include hidden_states")
    args[0] = replay.output
    if len(args) > 1:
        args[1] = None
    return tuple(args), {}


def run_model_cli() -> None:
    """Load one example pipeline and run quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = "outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors"
    parser = default_arg_parser(
        "black-forest-labs/FLUX.2-klein-4B",
        output,
        steps=4,
        guidance_scale=1.0,
        batch_size=1,
        height=1024,
        width=1024,
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = (
            f"outputs/checkpoints/svdq-{args.precision}_r32-flux2-klein-4b.safetensors"
        )
    cache_dir = args.cache_dir or "outputs/calibration/flux2-klein-4b"
    pipe = load_pipeline(
        "Flux2KleinPipeline",
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
    )
    target_config = flux2_klein_target_config(
        args.precision,
        single_qkv_features=9216,
        single_attn_features=3072,
    )
    if args.inspect_config:
        print(inspect_target_config(pipe.transformer, target_config).format_text())
        return
    records = standard_prompt_records(args.num_samples, prompt_file=args.prompt_file)
    forward_fn = pipeline_forward_fn(
        pipe,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        use_pe=None,
    )
    artifact_cache = None
    if cache_dir is not None and args.cache_mode != "disabled":
        artifact_cache = QuantizationCacheSpec(
            cache_dir=Path(cache_dir) / args.precision / "artifacts",
            cache_mode=args.cache_mode,
        )
    output_dir = (
        None
        if cache_dir is None
        else Path(cache_dir) / args.precision / "inputs" / "samples"
    )
    quantize_and_export(
        model=pipe.transformer,
        spec=svdquant_spec(
            args.precision,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            compute_device=args.compute_device
            or (args.device if args.offload_model else None),
            offload_model=args.offload_model,
        ),
        target_config=target_config,
        calibration=CalibrationSpec(
            samples=batched_samples(records, args.batch_size),
            num_samples=args.num_samples,
            cache_num_samples=args.num_samples
            if args.cache_num_samples is None
            else args.cache_num_samples,
            batch_size=args.batch_size,
            cache_dir=None
            if cache_dir is None
            else Path(cache_dir) / args.precision / "inputs",
            cache_mode=args.cache_mode,
            forward_fn=forward_fn,
            output_dir=output_dir,
            output_save_fn=save_diffusers_images,
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size or args.batch_size,
            artifact_cache=artifact_cache,
        ),
        export=ExportSpec(output=Path(args.output)),
        logging=LoggingConfig(
            enabled=not args.no_run_log,
            log_dir=args.log_dir,
            name=Path(args.output).stem,
        ),
    )


def flux2_klein_target_config(
    precision: Precision = "int4",
    *,
    single_qkv_features: int = 9216,
    single_attn_features: int = 3072,
    use_nunchaku_layout: bool = True,
) -> TargetConfig:
    """Return a FLUX.2 Klein target config for upstream SVDQuant examples."""

    del precision
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Attention,
        Flux2SingleTransformerBlock,
        Flux2TransformerBlock,
    )

    def qkv_members(attn: Flux2Attention) -> dict[str, torch.nn.Module]:
        return {"q": attn.to_q, "k": attn.to_k, "v": attn.to_v}

    def added_qkv_members(attn: Flux2Attention) -> dict[str, torch.nn.Module]:
        return {
            "add_q": attn.add_q_proj,
            "add_k": attn.add_k_proj,
            "add_v": attn.add_v_proj,
        }

    single_qkv_target = {
        "modules": ["single_transformer_blocks.*.attn.to_qkv_mlp_proj.linears.0"],
        "export_name": "single_transformer_blocks.{0}.attn.qkv_proj",
    }
    if use_nunchaku_layout:
        single_qkv_target["quant"] = SvdqTargetQuant(
            weight_layout=NunchakuSvdqLayout(
                outer_scale_splits=(
                    single_attn_features,
                    single_attn_features,
                    single_attn_features,
                )
            )
        )

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
            CalibrationScopeRule(
                module_classes=Flux2TransformerBlock,
                prev_replay_transform=_flux2_block_prev_replay_transform,
            ),
            CalibrationScopeRule(
                module_classes=Flux2SingleTransformerBlock,
                prev_replay_transform=_flux2_block_prev_replay_transform,
            ),
        ],
        targets=[
            TargetRule(
                parent_module_classes=Flux2Attention,
                member_selector=qkv_members,
                export_name="{parent_path}.to_qkv",
            ),
            TargetRule(
                parent_module_classes=Flux2Attention,
                member_selector=added_qkv_members,
                export_name="{parent_path}.to_added_qkv",
            ),
            TargetRule(
                scope_module_classes=Flux2TransformerBlock,
                module_classes=torch.nn.Linear,
            ),
            TargetRule(**single_qkv_target),
            TargetRule(
                modules=["single_transformer_blocks.*.attn.to_qkv_mlp_proj.linears.1"],
                export_name="single_transformer_blocks.{0}.attn.mlp_fc1",
            ),
            TargetRule(
                modules=["single_transformer_blocks.*.attn.to_out.linears.0"],
                export_name="single_transformer_blocks.{0}.attn.out_proj",
            ),
            TargetRule(
                modules=["single_transformer_blocks.*.attn.to_out.linears.1"],
                export_name="single_transformer_blocks.{0}.attn.mlp_fc2",
            ),
        ],
    )


if __name__ == "__main__":
    run_model_cli()
