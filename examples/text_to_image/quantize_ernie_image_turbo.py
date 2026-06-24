"""Quantize ERNIE-Image Turbo with upstream DeepCompressor-style SVDQuant config."""

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


def _ernie_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next ERNIE block input from the previous block replay."""

    if isinstance(replay.output, tuple):
        raise TypeError("ERNIE block replay output must be a tensor")
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["x"] = replay.output
        return (), kwargs
    args = list(replay.args)
    if not args:
        raise TypeError("ERNIE block replay args must include x")
    args[0] = replay.output
    return tuple(args), {}


def run_model_cli() -> None:
    """Load one example pipeline and run quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = "outputs/checkpoints/svdq-int4_r32-ernie-image-turbo.safetensors"
    parser = default_arg_parser(
        "baidu/ERNIE-Image-Turbo",
        output,
        steps=8,
        guidance_scale=1.0,
        batch_size=1,
        height=1024,
        width=1024,
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = f"outputs/checkpoints/svdq-{args.precision}_r32-ernie-image-turbo.safetensors"
    cache_dir = args.cache_dir or "outputs/calibration/ernie-image-turbo"
    pipe = load_pipeline(
        "ErnieImagePipeline",
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
    )
    target_config = ernie_image_target_config(args.precision)
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
        use_pe=False,
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
            max_rows_per_target=4096,  # Cap sampled activation rows per target to speed up quantization.
        ),
        export=ExportSpec(output=Path(args.output)),
        logging=LoggingConfig(
            enabled=not args.no_run_log,
            log_dir=args.log_dir,
            name=Path(args.output).stem,
        ),
    )


def ernie_image_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return an ERNIE-Image target config for manifest-driven export."""

    from diffusers.models.transformers.transformer_ernie_image import (
        ErnieImageSharedAdaLNBlock,
    )

    block_targets = [
        "layers.*.self_attention.to_q",
        "layers.*.self_attention.to_k",
        "layers.*.self_attention.to_v",
        "layers.*.self_attention.to_out.0",
        "layers.*.mlp.gate_proj",
        "layers.*.mlp.up_proj",
        "layers.*.mlp.linear_fc2",
    ]
    targets = [TargetRule(modules=[pattern]) for pattern in block_targets]
    return TargetConfig(
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=ErnieImageSharedAdaLNBlock,
                prev_replay_transform=_ernie_block_prev_replay_transform,
            )
        ],
        targets=targets,
    )


if __name__ == "__main__":
    run_model_cli()
