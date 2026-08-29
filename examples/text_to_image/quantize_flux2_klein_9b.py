"""Quantize FLUX.2 Klein 9B with upstream DeepCompressor-style SVDQuant config."""

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

from utils import (
    DEFAULT_QDIFF_PROMPT_FILE,
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
    output = "outputs/checkpoints/svdq-int4_r32-flux2-klein-9b.safetensors"
    parser = default_arg_parser(
        "black-forest-labs/FLUX.2-klein-9B",
        output,
        steps=4,
        guidance_scale=1.0,
        batch_size=1,
        height=1024,
        width=1024,
    )
    parser.add_argument(
        "--data-free",
        action="store_true",
        help="Quantize without calibration data: weight-span smoothing plus plain "
        "SVD low-rank, loading only the transformer. Prompt and sample options "
        "are ignored.",
    )
    parser.add_argument(
        "--synthetic-calibration",
        action="store_true",
        help="With --data-free: calibrate on synthetic denoising trajectories "
        "(Gaussian latents plus fabricated text embeddings) driven through the "
        "transformer alone, enabling the full smoothing search and "
        "activation-weighted SVD without any dataset.",
    )
    parser.add_argument(
        "--synthetic-text",
        choices=("gaussian", "encoder"),
        default="gaussian",
        help="Text embeddings for --synthetic-calibration: random Gaussian "
        "(default, loads nothing extra) or the real text encoder run once on "
        "bundled prompts.",
    )
    args = parser.parse_args()
    if args.synthetic_calibration and not args.data_free:
        parser.error("--synthetic-calibration requires --data-free")
    if args.output == output:
        variant = (
            f"synthcal-{args.synthetic_text}-" if args.synthetic_calibration else ""
        )
        args.output = (
            f"outputs/checkpoints/svdq-{variant}{args.precision}_r32-flux2-klein-9b.safetensors"
        )
    cache_dir = args.cache_dir or "outputs/calibration/flux2-klein-9b"
    if args.data_free:
        from quantize_hf import load_denoiser_only

        pipe = None
        model = load_denoiser_only(args.model_id, device=args.device)
    else:
        pipe = load_pipeline(
            "Flux2KleinPipeline",
            args.model_id,
            device=args.device,
            pipeline_offload=args.pipeline_offload,
            dtype=torch.bfloat16,
        )
        model = pipe.transformer
    target_config = flux2_klein_target_config(
        args.precision,
        single_qkv_features=12288,
        single_attn_features=4096,
    )
    if args.inspect_config:
        print(inspect_target_config(model, target_config).format_text())
        return
    artifact_dir = (
        f"artifacts-synthetic-{args.synthetic_text}"
        if args.synthetic_calibration
        else "artifacts"
    )
    artifact_cache = None
    if cache_dir is not None and args.cache_mode != "disabled":
        artifact_cache = QuantizationCacheSpec(
            cache_dir=Path(cache_dir) / args.precision / artifact_dir,
            cache_mode=args.cache_mode,
        )
    if args.data_free and args.synthetic_calibration:
        from flux2_synthetic import synthetic_denoise_forward_fn, synthetic_prompt_embeds

        prompt_embeds = synthetic_prompt_embeds(
            model.config,
            args.num_samples,
            mode=args.synthetic_text,
            model_id=args.model_id,
            device=args.device,
            prompt_file=args.prompt_file,
        )
        calibration = CalibrationSpec(
            samples=[{"seed": index} for index in range(args.num_samples)],
            num_samples=args.num_samples,
            cache_num_samples=args.num_samples * args.steps
            if args.cache_num_samples is None
            else args.cache_num_samples,
            batch_size=args.batch_size,
            cache_dir=None
            if cache_dir is None
            else Path(cache_dir) / args.precision / f"synthetic-{args.synthetic_text}",
            cache_mode=args.cache_mode,
            forward_fn=synthetic_denoise_forward_fn(
                model,
                model_id=args.model_id,
                height=args.height,
                width=args.width,
                steps=args.steps,
                prompt_embeds=prompt_embeds,
                device=args.device,
            ),
            shared_input_keys=("txt_ids", "img_ids"),
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size,
            artifact_cache=artifact_cache,
            max_rows_per_target=4096,  # Cap sampled activation rows per target to speed up quantization.
        )
    elif args.data_free:
        calibration = (
            None
            if artifact_cache is None
            else CalibrationSpec(artifact_cache=artifact_cache)
        )
    else:
        records = standard_prompt_records(
            args.num_samples, prompt_file=args.prompt_file
        )
        forward_fn = pipeline_forward_fn(
            pipe,
            height=args.height,
            width=args.width,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            device=args.device,
            use_pe=None,
        )
        output_dir = (
            None
            if cache_dir is None
            else Path(cache_dir) / args.precision / "inputs" / "samples"
        )
        calibration = CalibrationSpec(
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
            sample_batch_size=args.sample_batch_size,
            artifact_cache=artifact_cache,
            max_rows_per_target=4096,  # Cap sampled activation rows per target to speed up quantization.
        )
    quantize_and_export(
        model=model,
        spec=svdquant_spec(
            args.precision,
            data_free=args.data_free and not args.synthetic_calibration,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            compute_device=args.compute_device
            or (args.device if args.offload_model else None),
            offload_model=args.offload_model,
        ),
        target_config=target_config,
        calibration=calibration,
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

    single_qkv_quant = SvdqTargetQuant(
        weight_layout=NunchakuSvdqLayout(
            outer_scale_splits=(
                single_attn_features,
                single_attn_features,
                single_attn_features,
            )
        )
    )

    single_qkv_target = {
        "modules": ["single_transformer_blocks.*.attn.to_qkv_mlp_proj.linears.0"],
        "export_name": "single_transformer_blocks.{0}.attn.qkv_proj",
        "quant": single_qkv_quant,
    }

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
                modules=("single_transformer_blocks.0",),
                module_classes=Flux2SingleTransformerBlock,
                use_prev_scope_outputs=False,
            ),
            CalibrationScopeRule(
                modules=("single_transformer_blocks.*",),
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
