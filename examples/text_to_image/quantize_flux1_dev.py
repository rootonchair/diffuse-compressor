"""Quantize FLUX.1 Dev with upstream DeepCompressor-style SVDQuant config."""

from __future__ import annotations

import logging
from pathlib import Path


from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    CalibrationScopeRule,
    CalibrationSpec,
    ExportSpec,
    LoggingConfig,
    PatchRule,
    QuantizationCacheSpec,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    AwqTargetQuant,
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


def run_model_cli() -> None:
    """Load one example pipeline and run quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = "outputs/checkpoints/svdq-int4_r32-flux.1-dev.safetensors"
    parser = default_arg_parser(
        "black-forest-labs/FLUX.1-dev",
        output,
        steps=50,
        guidance_scale=3.5,
        batch_size=16,
        height=1024,
        width=1024,
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = (
            f"outputs/checkpoints/svdq-{args.precision}_r32-flux.1-dev.safetensors"
        )
    cache_dir = args.cache_dir or "outputs/calibration/flux.1-dev"
    pipe = load_pipeline(
        "FluxPipeline",
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
    )
    target_config = flux1_target_config(args.precision)
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
            shared_input_keys=("txt_ids", "img_ids"),
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size or args.batch_size,
            artifact_cache=artifact_cache,
            max_rows_per_target=4096,  # Cap sampled activation rows per target to speed up quantization.
        ),
        export=ExportSpec(output=Path(args.output)),
        logging=LoggingConfig(
            enabled=not args.no_run_log,
            log_dir=args.log_dir,
            name=Path(args.output).stem,
        ),
    )


def _flux_extra_weight_targets() -> list[TargetRule]:
    """Return NVFP4 Flux extra INT4 target rules."""

    return [
        TargetRule(
            modules=["transformer_blocks.*.norm1.linear"],
            quant=AwqTargetQuant(layout=AdaNormAwqW4A16Layout(splits=6)),
        ),
        TargetRule(
            modules=["transformer_blocks.*.norm1_context.linear"],
            quant=AwqTargetQuant(layout=AdaNormAwqW4A16Layout(splits=6)),
        ),
        TargetRule(
            modules=["single_transformer_blocks.*.norm.linear"],
            quant=AwqTargetQuant(layout=AdaNormAwqW4A16Layout(splits=3)),
        ),
    ]


def flux1_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return a Flux.1 target config for upstream SVDQuant examples."""

    from diffusers.models.transformers.transformer_flux import (
        FluxSingleTransformerBlock,
        FluxTransformerBlock,
    )

    down_proj_shift_activations = True if precision == "int4" else None
    targets = [
        TargetRule(
            modules=[
                "transformer_blocks.*.attn.to_q",
                "transformer_blocks.*.attn.to_k",
                "transformer_blocks.*.attn.to_v",
            ],
            export_name="transformer_blocks.{0}.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule(
            modules=[
                "transformer_blocks.*.attn.add_q_proj",
                "transformer_blocks.*.attn.add_k_proj",
                "transformer_blocks.*.attn.add_v_proj",
            ],
            export_name="transformer_blocks.{0}.qkv_proj_context",
            roles=["add_q", "add_k", "add_v"],
        ),
        TargetRule(
            modules=["transformer_blocks.*.attn.to_out.0"],
            export_name="transformer_blocks.{0}.out_proj",
        ),
        TargetRule(
            modules=["transformer_blocks.*.attn.to_add_out"],
            export_name="transformer_blocks.{0}.out_proj_context",
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff.net.0.proj"],
            export_name="transformer_blocks.{0}.mlp_fc1",
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff.net.2"],
            export_name="transformer_blocks.{0}.mlp_fc2",
            quant=SvdqTargetQuant(shift_activations=down_proj_shift_activations),
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff_context.net.0.proj"],
            export_name="transformer_blocks.{0}.mlp_context_fc1",
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff_context.net.2"],
            export_name="transformer_blocks.{0}.mlp_context_fc2",
            quant=SvdqTargetQuant(shift_activations=down_proj_shift_activations),
        ),
        TargetRule(
            modules=[
                "single_transformer_blocks.*.attn.to_q",
                "single_transformer_blocks.*.attn.to_k",
                "single_transformer_blocks.*.attn.to_v",
            ],
            export_name="single_transformer_blocks.{0}.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule(
            modules=["single_transformer_blocks.*.proj_out.linears.0"],
            export_name="single_transformer_blocks.{0}.out_proj",
            quant=SvdqTargetQuant(bias="zero"),
        ),
        TargetRule(
            modules=["single_transformer_blocks.*.proj_mlp"],
            export_name="single_transformer_blocks.{0}.mlp_fc1",
        ),
        TargetRule(
            modules=["single_transformer_blocks.*.proj_out.linears.1"],
            export_name="single_transformer_blocks.{0}.mlp_fc2",
            quant=SvdqTargetQuant(shift_activations=down_proj_shift_activations),
        ),
    ]
    if precision == "nvfp4":
        targets.extend(_flux_extra_weight_targets())
    return TargetConfig(
        patches=[
            PatchRule(
                type="split_linear",
                module="single_transformer_blocks.*.proj_out",
                args={"splits": ["out_features"]},
            )
        ],
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=FluxTransformerBlock,
                prev_replay_transform=_flux_block_prev_replay_transform,
            ),
            CalibrationScopeRule(
                module_classes=FluxSingleTransformerBlock,
                prev_replay_transform=_flux_block_prev_replay_transform,
            ),
        ],
        targets=targets,
    )


if __name__ == "__main__":
    run_model_cli()
