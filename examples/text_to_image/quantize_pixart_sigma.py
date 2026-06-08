"""Quantize PixArt Sigma with upstream DeepCompressor-style SVDQuant config."""

from __future__ import annotations

import logging
from pathlib import Path


from diffuse_compressor import (
    CalibrationScopeRule,
    CalibrationSpec,
    ExportSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    TargetConfig,
    TargetRule,
    AwqTargetQuant,
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


def _hidden_states_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next single-stream block input from the previous hidden states."""

    if isinstance(replay.output, tuple):
        raise TypeError("Block replay output must be a hidden-state tensor")
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = replay.output
        return (), kwargs
    args = list(replay.args)
    if not args:
        raise TypeError("Block replay args must include hidden_states")
    args[0] = replay.output
    return tuple(args), {}


def run_model_cli() -> None:
    """Load one example pipeline and run quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = "outputs/checkpoints/svdq-int4_r32-pixart-sigma.safetensors"
    parser = default_arg_parser(
        "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        output,
        steps=20,
        guidance_scale=4.5,
        batch_size=256,
        height=1024,
        width=1024,
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = (
            f"outputs/checkpoints/svdq-{args.precision}_r32-pixart-sigma.safetensors"
        )
    cache_dir = args.cache_dir or "outputs/calibration/pixart-sigma"
    pipe = load_pipeline(
        "PixArtSigmaPipeline",
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
    )
    target_config = pixart_sigma_target_config(args.precision)
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


def pixart_sigma_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return a PixArt Sigma target config for upstream SVDQuant examples."""

    from diffusers.models.attention import BasicTransformerBlock

    targets = [
        TargetRule(
            modules=[
                "transformer_blocks.*.attn1.to_q",
                "transformer_blocks.*.attn1.to_k",
                "transformer_blocks.*.attn1.to_v",
            ],
            export_name="transformer_blocks.{0}.attn1.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule(
            modules=["transformer_blocks.*.attn1.to_out.0"],
            export_name="transformer_blocks.{0}.attn1.out_proj",
        ),
        TargetRule(
            modules=["transformer_blocks.*.attn2.to_q"],
            export_name="transformer_blocks.{0}.attn2.q_proj",
        ),
        TargetRule(
            modules=[
                "transformer_blocks.*.attn2.to_k",
                "transformer_blocks.*.attn2.to_v",
            ],
            export_name="transformer_blocks.{0}.attn2.kv_proj",
            roles=["k", "v"],
        ),
        TargetRule(
            modules=["transformer_blocks.*.attn2.to_out.0"],
            export_name="transformer_blocks.{0}.attn2.out_proj",
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff.net.0.proj"],
            export_name="transformer_blocks.{0}.mlp_fc1",
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff.net.2"],
            export_name="transformer_blocks.{0}.mlp_fc2",
        ),
    ]
    if precision == "nvfp4":
        targets.append(
            TargetRule(
                modules=["adaln_single.linear"],
                quant=AwqTargetQuant(),
            )
        )
    return TargetConfig(
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=BasicTransformerBlock,
                prev_replay_transform=_hidden_states_prev_replay_transform,
            )
        ],
        targets=targets,
    )


if __name__ == "__main__":
    run_model_cli()
