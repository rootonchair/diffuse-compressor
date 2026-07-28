"""Quantize FLUX.2 Klein 4B with GPTQ-enabled SVDQuant config."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import torch

from diffuse_compressor import (
    CalibrationSpec,
    ExportSpec,
    GptqSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    inspect_target_config,
    quantize_and_export,
)

from quantize_flux2_klein_4b import flux2_klein_target_config
from utils import (
    batched_samples,
    default_arg_parser,
    load_pipeline,
    pipeline_forward_fn,
    save_diffusers_images,
    standard_prompt_records,
    svdquant_spec,
)


def run_model_cli() -> None:
    """Load FLUX.2 Klein 4B and run GPTQ-enabled quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = "outputs/checkpoints/svdq-gptq-int4_r32-flux2-klein-4b.safetensors"
    parser = default_arg_parser(
        "black-forest-labs/FLUX.2-klein-4B",
        output,
        steps=4,
        guidance_scale=1.0,
        batch_size=1,
        height=1024,
        width=1024,
    )
    parser.add_argument("--gptq-damp-percentage", type=float, default=0.01)
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--gptq-num-inv-tries", type=int, default=250)
    parser.add_argument("--gptq-hessian-block-size", type=int, default=512)
    args = parser.parse_args()
    if args.output == output:
        args.output = f"outputs/checkpoints/svdq-gptq-{args.precision}_r32-flux2-klein-4b.safetensors"
    cache_dir = args.cache_dir or "outputs/calibration/flux2-klein-4b"
    cache_key = f"{args.precision}-gptq"
    pipe = load_pipeline(
        "Flux2KleinPipeline",
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
        dtype=torch.bfloat16,
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
            cache_dir=Path(cache_dir) / cache_key / "artifacts",
            cache_mode=args.cache_mode,
        )
    output_dir = None if cache_dir is None else Path(cache_dir) / cache_key / "inputs" / "samples"
    base_spec = svdquant_spec(
        args.precision,
        svd_backend=args.svd_backend,
        svd_lowrank_oversample=args.svd_lowrank_oversample,
        svd_lowrank_niter=args.svd_lowrank_niter,
        compute_device=args.compute_device or (args.device if args.offload_model else None),
        offload_model=args.offload_model,
    )
    quantize_and_export(
        model=pipe.transformer,
        spec=replace(
            base_spec,
            gptq=GptqSpec(
                enabled=True,
                damp_percentage=args.gptq_damp_percentage,
                block_size=args.gptq_block_size,
                num_inv_tries=args.gptq_num_inv_tries,
                hessian_block_size=args.gptq_hessian_block_size,
            ),
        ),
        target_config=target_config,
        calibration=CalibrationSpec(
            samples=batched_samples(records, args.batch_size),
            num_samples=args.num_samples,
            cache_num_samples=args.num_samples if args.cache_num_samples is None else args.cache_num_samples,
            batch_size=args.batch_size,
            cache_dir=None if cache_dir is None else Path(cache_dir) / cache_key / "inputs",
            cache_mode=args.cache_mode,
            forward_fn=forward_fn,
            output_dir=output_dir,
            output_save_fn=save_diffusers_images,
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size,
            artifact_cache=artifact_cache,
            max_rows_per_target=4096,
        ),
        export=ExportSpec(output=Path(args.output)),
        logging=LoggingConfig(
            enabled=not args.no_run_log,
            log_dir=args.log_dir,
            name=Path(args.output).stem,
        ),
    )


if __name__ == "__main__":
    run_model_cli()
