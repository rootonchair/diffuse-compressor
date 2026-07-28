"""Quantize Microsoft Lens-Turbo with upstream DeepCompressor-style SVDQuant config."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, Literal

import torch
from torch import nn

from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    AwqTargetQuant,
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
    PipelineOffload,
    Precision,
    batched_samples,
    make_generator,
    save_diffusers_images,
    standard_prompt_records,
    svdquant_spec,
    _torch_dtype,
)


TextEncoderDevice = Literal["auto", "cpu"]


def _missing_lens_error() -> RuntimeError:
    return RuntimeError(
        "The Lens-Turbo example requires Microsoft's Lens inference package. "
        "Install it from https://github.com/microsoft/Lens before running this script."
    )


def _force_pipeline_execution_device(pipe, device: str) -> None:
    """Force Diffusers intermediates to use the transformer execution device."""

    # Lens does not expose a public way to keep the text encoder on CPU while
    # making Diffusers-style pipeline internals allocate intermediates on the
    # transformer device. Keep this workaround local to that split-device path.
    base_cls = pipe.__class__
    forced_cls = type(
        f"{base_cls.__name__}WithForcedExecutionDevice",
        (base_cls,),
        {
            "_execution_device": property(
                lambda self: torch.device(
                    getattr(self, "_diffuse_compressor_execution_device")
                )
            )
        },
    )
    pipe.__class__ = forced_cls
    pipe._diffuse_compressor_execution_device = str(torch.device(device))


def _lens_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next Lens block input from the previous block replay."""

    if not isinstance(replay.output, tuple) or len(replay.output) != 2:
        raise TypeError(
            "Lens block replay output must be (encoder_hidden_states, hidden_states)"
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
            "Lens block replay args must include hidden_states and encoder_hidden_states"
        )
    args[0] = hidden_states
    args[1] = encoder_hidden_states
    return tuple(args), {}


def default_arg_parser(
    model_id: str,
    output: str,
    *,
    steps: int,
    guidance_scale: float,
    batch_size: int,
    base_resolution: int = 1024,
    aspect_ratio: str = "1:1",
) -> argparse.ArgumentParser:
    """Create the Lens-Turbo CLI parser."""

    parser = argparse.ArgumentParser(
        description="Quantize Microsoft Lens-Turbo with the shared SVDQuant example config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--precision",
        choices=("int4", "nvfp4"),
        default="int4",
        help="Weight precision overlay.",
    )
    parser.add_argument(
        "--model-id",
        default=model_id,
        help="Hugging Face model id or local Lens pipeline directory.",
    )
    parser.add_argument(
        "--output", default=output, help="Output safetensors checkpoint path."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration prompts to use.",
    )
    parser.add_argument(
        "--cache-num-samples",
        type=int,
        default=None,
        help=(
            "Number of cached model-forward records to replay. Defaults to --num-samples; "
            "use -1 to replay every cached calibration record."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        help="Calibration DataLoader batch size.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=-1,
        help="Activation row batch size for smoothing, range calibration, and low-rank scoring.",
    )
    parser.add_argument(
        "--scope-capture-mode",
        choices=("all-targets", "one-target"),
        default="all-targets",
        help="Capture every target in a scope at once, or replay each scope once per target to lower peak RAM.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Calibration/artifact cache root; defaults to outputs/calibration/lens-turbo.",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("reuse", "refresh", "disabled"),
        default="reuse",
        help="Reuse existing calibration caches, refresh them, or disable disk caching.",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_QDIFF_PROMPT_FILE,
        help="QDiff-style prompt YAML path.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Pipeline execution device.",
    )
    parser.add_argument(
        "--compute-device",
        default=None,
        help="Optional per-target quantization compute device; useful with --offload-model.",
    )
    parser.add_argument(
        "--offload-model",
        action="store_true",
        help="Move the transformer back to CPU between quantization work.",
    )
    parser.add_argument(
        "--pipeline-offload",
        choices=("none", "model", "sequential"),
        default="none",
        help="Enable Lens/Diffusers pipeline CPU offload while collecting calibration inputs.",
    )
    parser.add_argument(
        "--base-resolution",
        type=int,
        default=base_resolution,
        help="Lens base resolution, for example 1024 or 1440.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default=aspect_ratio,
        help="Lens aspect ratio bucket, for example 1:1, 16:9, or 9:16.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=steps,
        help="Denoising steps for calibration forwards.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=guidance_scale,
        help="Guidance scale for calibration forwards.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Pipeline compute dtype.",
    )
    parser.add_argument(
        "--text-encoder-device",
        choices=("auto", "cpu"),
        default="auto",
        help="Keep Lens GPT-OSS text encoding on CPU to reduce CUDA VRAM.",
    )
    parser.add_argument(
        "--disable-mxfp4",
        action="store_true",
        help="Ask the Lens GPT-OSS text encoder loader to dequantize MXFP4 weights to --dtype.",
    )
    parser.add_argument(
        "--svd-backend",
        choices=("full", "svd_lowrank"),
        default="svd_lowrank",
        help="Low-rank decomposition backend.",
    )
    parser.add_argument(
        "--svd-lowrank-oversample",
        type=int,
        default=10,
        help="Oversampling rank for torch.svd_lowrank.",
    )
    parser.add_argument(
        "--svd-lowrank-niter",
        type=int,
        default=4,
        help="Power iterations for torch.svd_lowrank.",
    )
    parser.add_argument(
        "--log-dir",
        default="outputs/logs",
        help="Directory for quantization process and target logs.",
    )
    parser.add_argument(
        "--no-run-log", action="store_true", help="Disable quantization run log files."
    )
    parser.add_argument(
        "--inspect-config",
        action="store_true",
        help="Print target-config diagnostics and exit.",
    )
    return parser


def run_model_cli() -> None:
    """Load Lens-Turbo and run quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = "outputs/checkpoints/svdq-int4_r32-lens-turbo.safetensors"
    parser = default_arg_parser(
        "microsoft/Lens-Turbo",
        output,
        steps=4,
        guidance_scale=1.0,
        batch_size=1,
        base_resolution=1024,
        aspect_ratio="1:1",
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = (
            f"outputs/checkpoints/svdq-{args.precision}_r32-lens-turbo.safetensors"
        )
    cache_dir = args.cache_dir or "outputs/calibration/lens-turbo"
    pipe = load_pipeline(
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
        dtype=_torch_dtype(args.dtype),
        disable_mxfp4=args.disable_mxfp4,
        text_encoder_device=args.text_encoder_device,
    )
    target_config = lens_turbo_target_config(args.precision)
    if args.inspect_config:
        print(inspect_target_config(pipe.transformer, target_config).format_text())
        return
    records = standard_prompt_records(args.num_samples, prompt_file=args.prompt_file)
    forward_fn = pipeline_forward_fn(
        pipe,
        base_resolution=args.base_resolution,
        aspect_ratio=args.aspect_ratio,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        text_encoder_device=args.text_encoder_device,
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
            sample_batch_size=args.sample_batch_size,
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


def load_pipeline(
    model_id: str,
    *,
    device: str,
    pipeline_offload: PipelineOffload = "none",
    dtype: torch.dtype = torch.bfloat16,
    disable_mxfp4: bool = False,
    text_encoder_device: TextEncoderDevice = "auto",
):
    """Load a Lens pipeline from the external Microsoft Lens package."""

    try:
        from lens import LensGptOssEncoder, LensPipeline
    except ImportError as exc:
        raise _missing_lens_error() from exc

    text_encoder_kwargs = {"subfolder": "text_encoder", "dtype": dtype}
    try:
        from transformers import Mxfp4Config

        text_encoder_kwargs["quantization_config"] = Mxfp4Config(
            dequantize=disable_mxfp4
        )
    except ImportError:
        pass
    text_encoder = LensGptOssEncoder.from_pretrained(model_id, **text_encoder_kwargs)
    pipe = LensPipeline.from_pretrained(
        model_id, text_encoder=text_encoder, torch_dtype=dtype
    )
    if text_encoder_device == "cpu":
        pipe.text_encoder.to("cpu")
        pipe.transformer.to(device)
        pipe.vae.to(device)
        _force_pipeline_execution_device(pipe, device)
        return pipe
    if pipeline_offload == "none":
        return pipe.to(device)
    method_name = (
        "enable_model_cpu_offload"
        if pipeline_offload == "model"
        else "enable_sequential_cpu_offload"
    )
    method = getattr(pipe, method_name, None)
    if method is None:
        raise RuntimeError(f"LensPipeline does not support {method_name}()")
    try:
        method(device=device)
    except TypeError:
        method()
    return pipe


def pipeline_forward_fn(
    pipe,
    *,
    base_resolution: int,
    aspect_ratio: str,
    steps: int,
    guidance_scale: float,
    device: str,
    text_encoder_device: TextEncoderDevice = "auto",
) -> Callable[[dict], object]:
    """Create a calibration forward function for a Lens pipeline."""

    def forward(sample: dict) -> object:
        prompt_kwargs = {"prompt": sample["prompt"]}
        if text_encoder_device == "cpu":
            text_device = torch.device("cpu")
            execution_device = torch.device(device)
            prompt_embeds, prompt_mask, negative_prompt_embeds, negative_prompt_mask = (
                pipe.encode_prompt(
                    prompt=sample["prompt"],
                    negative_prompt="",
                    device=text_device,
                )
            )
            prompt_kwargs = {
                "prompt": "",
                "prompt_embeds": [item.to(execution_device) for item in prompt_embeds],
                "prompt_mask": prompt_mask.to(execution_device),
                "negative_prompt_embeds": [
                    item.to(execution_device) for item in negative_prompt_embeds
                ],
                "negative_prompt_mask": negative_prompt_mask.to(execution_device),
            }
        return pipe(
            **prompt_kwargs,
            base_resolution=base_resolution,
            aspect_ratio=aspect_ratio,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=make_generator(sample.get("seed", 0), device=device),
        )

    return forward


def lens_turbo_target_config(
    precision: Precision = "int4",
    *,
    inner_dim: int = 1536,
) -> TargetConfig:
    """Return a Lens-Turbo target config."""

    try:
        from diffusers.models.attention import FeedForward
        from lens.transformer import GateMLP, LensJointAttention, LensTransformerBlock
    except ImportError as exc:
        raise _missing_lens_error() from exc

    targets = [
        TargetRule(module_classes=nn.Linear, scope_module_classes=LensJointAttention),
        TargetRule(
            module_classes=nn.Linear, scope_module_classes=(GateMLP, FeedForward)
        ),
    ]
    if precision == "nvfp4":
        targets.extend(
            [
                TargetRule(
                    modules=["transformer_blocks.*.img_mod.*"],
                    module_classes=nn.Linear,
                    quant=AwqTargetQuant(layout=AdaNormAwqW4A16Layout(splits=6)),
                ),
                TargetRule(
                    modules=["transformer_blocks.*.txt_mod.*"],
                    module_classes=nn.Linear,
                    quant=AwqTargetQuant(layout=AdaNormAwqW4A16Layout(splits=6)),
                ),
            ]
        )

    return TargetConfig(
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=LensTransformerBlock,
                prev_replay_transform=_lens_block_prev_replay_transform,
            )
        ],
        targets=targets,
    )


if __name__ == "__main__":
    run_model_cli()
