"""Quantize LTX-2.3 Diffusers with upstream DeepCompressor-style SVDQuant config."""

from __future__ import annotations

import argparse
import logging
from functools import partial
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from diffuse_compressor import (
    AwqTargetQuant,
    AwqW4A16Layout,
    CalibrationScopeRule,
    CalibrationSpec,
    ExportSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    SkipRule,
    TargetConfig,
    TargetRule,
    collect_quant_targets,
    inspect_target_config,
    quantize_and_export,
)

from utils import (
    DEFAULT_QDIFF_PROMPT_FILE,
    PipelineOffload,
    Precision,
    batched_samples,
    make_generator,
    standard_prompt_records,
    svdquant_spec,
    _torch_dtype,
)


def _ltx2_default_negative_prompt() -> str:
    """Return the upstream LTX2 default negative prompt when Diffusers provides it."""

    from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

    return DEFAULT_NEGATIVE_PROMPT


def _ltx2_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next LTX2 block input from the previous block replay."""

    if not isinstance(replay.output, tuple) or len(replay.output) != 2:
        raise TypeError(
            "LTX2 block replay output must be (hidden_states, audio_hidden_states)"
        )
    hidden_states, audio_hidden_states = replay.output
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = hidden_states
        kwargs["audio_hidden_states"] = audio_hidden_states
        return (), kwargs
    args = list(replay.args)
    if len(args) < 2:
        raise TypeError(
            "LTX2 block replay args must include hidden_states and audio_hidden_states"
        )
    args[0] = hidden_states
    args[1] = audio_hidden_states
    return tuple(args), {}


def default_arg_parser(
    model_id: str,
    output: str,
    *,
    steps: int,
    guidance_scale: float,
    batch_size: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    default_cache_dir: str = "outputs/calibration/ltx2.3",
) -> argparse.ArgumentParser:
    """Create the LTX-2.3 CLI parser."""

    parser = argparse.ArgumentParser(
        description="Quantize LTX-2.3 Diffusers transformer with the shared SVDQuant example config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--precision",
        choices=("int4", "nvfp4"),
        default="int4",
        help="Weight precision overlay.",
    )
    parser.add_argument(
        "--fuse-qkv",
        action="store_true",
        help="Group compatible LTX attention QKV/KV projections into fused quantization targets.",
    )
    parser.add_argument(
        "--model-id",
        default=model_id,
        help="Hugging Face model id or local Diffusers pipeline directory.",
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
        help=f"Calibration/artifact cache root; defaults to {default_cache_dir}.",
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
        help="Enable Diffusers pipeline CPU offload while collecting calibration inputs.",
    )
    parser.add_argument(
        "--height", type=int, default=height, help="Calibration video height."
    )
    parser.add_argument(
        "--width", type=int, default=width, help="Calibration video width."
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=num_frames,
        help="Calibration video frame count.",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=frame_rate,
        help="Calibration video frame rate.",
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
        help="Video guidance scale for calibration forwards.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=_ltx2_default_negative_prompt(),
        help="Negative prompt used for calibration forwards.",
    )
    parser.add_argument(
        "--stg-scale",
        type=float,
        default=0.0,
        help="Spatio-temporal guidance scale for calibration forwards.",
    )
    parser.add_argument(
        "--modality-scale",
        type=float,
        default=1.0,
        help="Video modality guidance scale for calibration forwards.",
    )
    parser.add_argument(
        "--guidance-rescale",
        type=float,
        default=0.0,
        help="Guidance rescale for calibration forwards.",
    )
    parser.add_argument(
        "--decode-timestep",
        type=float,
        default=0.0,
        help="VAE decode timestep for calibration forwards.",
    )
    parser.add_argument(
        "--decode-noise-scale",
        type=float,
        default=None,
        help="VAE decode noise scale for calibration forwards.",
    )
    parser.add_argument(
        "--use-cross-timestep",
        action="store_true",
        help="Enable cross-timestep conditioning in the LTX2 pipeline.",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=1024,
        help="Maximum tokenizer sequence length.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Pipeline compute dtype.",
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
        "--gptq",
        action="store_true",
        help="Enable GPTQ residual-weight rounding after low-rank compensation.",
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
    parser.add_argument(
        "--text-encoder-quant",
        choices=("none", "8bit", "4bit"),
        default="none",
        help="Quantize the LTX2 text encoder via bitsandbytes to cut host RAM/VRAM footprint.",
    )
    return parser


def run_model_cli() -> None:
    """Load LTX-2.3 Diffusers and run quantization."""

    run_ltx2_3_cli(
        model_id="dg845/LTX-2.3-Diffusers",
        output_template="outputs/checkpoints/svdq-{precision}_r32-ltx2.3.safetensors",
        default_cache_dir="outputs/calibration/ltx2.3",
        steps=30,
        guidance_scale=3.0,
        batch_size=1,
        height=512,
        width=768,
        num_frames=121,
        frame_rate=24.0,
    )


def run_ltx2_3_cli(
    *,
    model_id: str,
    output_template: str,
    default_cache_dir: str,
    steps: int,
    guidance_scale: float,
    batch_size: int,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    sigmas: list[float] | None = None,
) -> None:
    """Load one LTX-2.3-family Diffusers pipeline and run quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = output_template.format(precision="int4")
    parser = default_arg_parser(
        model_id,
        output,
        steps=steps,
        guidance_scale=guidance_scale,
        batch_size=batch_size,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=frame_rate,
        default_cache_dir=default_cache_dir,
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = output_template.format(precision=args.precision)
    cache_dir = args.cache_dir or default_cache_dir
    pipe = load_ltx2_pipeline(
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
        dtype=_torch_dtype(args.dtype),
        text_encoder_quant=args.text_encoder_quant,
    )
    target_config = ltx2_3_target_config(args.precision, fuse_qkv=args.fuse_qkv)
    if args.inspect_config:
        print(inspect_target_config(pipe.transformer, target_config).format_text())
        return
    validate_ltx2_3_nunchaku_lite_manifest_targets(pipe.transformer, target_config)
    records = standard_prompt_records(args.num_samples, prompt_file=args.prompt_file)
    forward_fn = ltx2_forward_fn(
        pipe,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        sigmas=sigmas,
        negative_prompt=args.negative_prompt,
        stg_scale=args.stg_scale,
        modality_scale=args.modality_scale,
        guidance_rescale=args.guidance_rescale,
        decode_timestep=args.decode_timestep,
        decode_noise_scale=args.decode_noise_scale,
        use_cross_timestep=args.use_cross_timestep,
        max_sequence_length=args.max_sequence_length,
        device=args.device,
    )
    artifact_cache = None
    if cache_dir is not None and args.cache_mode != "disabled":
        artifact_cache = QuantizationCacheSpec(
            cache_dir=Path(cache_dir) / args.precision / "artifacts",
            cache_mode=args.cache_mode,
        )
    output_dir = Path(cache_dir) / args.precision / "inputs" / "samples"
    output_save_fn = partial(
        save_ltx2_videos,
        frame_rate=args.frame_rate,
        audio_sample_rate=_ltx2_audio_sample_rate(pipe),
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
            gptq=args.gptq,
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
            output_save_fn=output_save_fn,
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size,
            artifact_cache=artifact_cache,
            max_rows_per_target=1024,  # Cap sampled activation rows per target to speed up quantization.
        ),
        export=ExportSpec(output=Path(args.output)),
        logging=LoggingConfig(
            enabled=not args.no_run_log,
            log_dir=args.log_dir,
            name=Path(args.output).stem,
        ),
    )


def load_ltx2_pipeline(
    model_id: str,
    *,
    device: str,
    pipeline_offload: PipelineOffload = "none",
    dtype: torch.dtype = torch.bfloat16,
    text_encoder_quant: str = "none",
):
    """Load an LTX2 Diffusers pipeline."""

    from diffusers import LTX2Pipeline

    from_pretrained_kwargs = {}
    if text_encoder_quant != "none":
        from diffusers.quantizers import PipelineQuantizationConfig

        from_pretrained_kwargs["quantization_config"] = PipelineQuantizationConfig(
            quant_backend=f"bitsandbytes_{text_encoder_quant}",
            quant_kwargs={"load_in_8bit": True}
            if text_encoder_quant == "8bit"
            else {"load_in_4bit": True},
            components_to_quantize=["text_encoder"],
        )
    pipe = LTX2Pipeline.from_pretrained(model_id, torch_dtype=dtype, **from_pretrained_kwargs)
    if pipeline_offload == "none":
        return pipe.to(device)
    method_name = (
        "enable_model_cpu_offload"
        if pipeline_offload == "model"
        else "enable_sequential_cpu_offload"
    )
    method = getattr(pipe, method_name, None)
    if method is None:
        raise RuntimeError(f"LTX2Pipeline does not support {method_name}()")
    try:
        method(device=device)
    except TypeError:
        method()
    return pipe


def ltx2_forward_fn(
    pipe,
    *,
    height: int,
    width: int,
    num_frames: int,
    frame_rate: float,
    steps: int,
    guidance_scale: float,
    negative_prompt: str,
    stg_scale: float,
    modality_scale: float,
    guidance_rescale: float,
    decode_timestep: float,
    decode_noise_scale: float | None,
    use_cross_timestep: bool,
    max_sequence_length: int,
    device: str,
    sigmas: list[float] | None = None,
) -> Callable[[dict], object]:
    """Create a calibration forward function for an LTX2 pipeline."""

    def forward(sample: dict) -> object:
        return pipe(
            prompt=sample["prompt"],
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=steps,
            sigmas=sigmas,
            guidance_scale=guidance_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            decode_timestep=decode_timestep,
            decode_noise_scale=decode_noise_scale,
            use_cross_timestep=use_cross_timestep,
            max_sequence_length=max_sequence_length,
            output_type="np",
            return_dict=False,
            generator=make_generator(sample.get("seed", 0), device=device),
        )

    return forward


def save_ltx2_videos(
    result: object,
    sample: dict,
    output_dir: Path,
    *,
    frame_rate: float,
    audio_sample_rate: int | None = None,
) -> None:
    """Save generated LTX2 calibration videos using calibration sample filenames."""

    frames, audio = _ltx2_result_media(result)
    videos = _batched_media(frames)
    audios = _batched_audio(audio, len(videos))
    filenames = _as_list(sample.get("filename"))
    if not filenames:
        filenames = [f"{int(seed):04d}-0" for seed in _as_list(sample.get("seed"))]
    if len(filenames) != len(videos):
        raise ValueError(
            f"Expected {len(filenames)} video filenames, got {len(videos)} videos"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, video, audio_item in zip(filenames, videos, audios, strict=True):
        _encode_ltx2_video(
            video,
            frame_rate=frame_rate,
            output_path=output_dir / f"{filename}.mp4",
            audio=audio_item,
            audio_sample_rate=audio_sample_rate,
        )


def _ltx2_result_media(result: object) -> tuple[object, object | None]:
    frames = getattr(result, "frames", None)
    audio = getattr(result, "audio", None)
    if frames is not None:
        return frames, audio
    if isinstance(result, tuple) and result:
        return result[0], result[1] if len(result) > 1 else None
    raise ValueError("LTX2 calibration output must expose frames or return (frames, audio)")


def _batched_media(media: object) -> list[object]:
    if isinstance(media, list):
        if media and isinstance(media[0], list):
            return list(media)
        return [media]
    ndim = getattr(media, "ndim", None)
    if ndim == 5:
        return list(media)
    return [media]


def _batched_audio(audio: object | None, count: int) -> list[object | None]:
    if audio is None:
        return [None] * count
    if isinstance(audio, list):
        if len(audio) == count:
            return list(audio)
        if count == 1:
            return [audio]
    shape = getattr(audio, "shape", None)
    if shape is not None and len(shape) >= 3 and shape[0] == count:
        return list(audio)
    if count == 1:
        return [audio]
    raise ValueError(f"Expected {count} audio items, got incompatible audio shape")


def _encode_ltx2_video(
    video: object,
    *,
    frame_rate: float,
    output_path: Path,
    audio: object | None,
    audio_sample_rate: int | None,
) -> None:
    from diffusers.utils import encode_video

    kwargs = {}
    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when saving LTX2 audio")
        if torch.is_tensor(audio):
            audio = audio.float().cpu()
        else:
            audio = torch.as_tensor(audio).float().cpu()
        kwargs["audio"] = audio
        kwargs["audio_sample_rate"] = audio_sample_rate
    encode_video(video, fps=int(frame_rate), output_path=str(output_path), **kwargs)


def _ltx2_audio_sample_rate(pipe) -> int | None:
    vocoder = getattr(pipe, "vocoder", None)
    config = getattr(vocoder, "config", None)
    sample_rate = getattr(config, "output_sampling_rate", None)
    return int(sample_rate) if sample_rate is not None else None


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def ltx2_3_target_config(precision: Precision = "int4", *, fuse_qkv: bool = False) -> TargetConfig:
    """Return an LTX-2.3 target config for Diffusers pipelines."""

    from diffusers.models.transformers.transformer_ltx2 import LTX2VideoTransformerBlock

    targets = []
    if fuse_qkv:
        targets.extend(_ltx2_fused_qkv_targets())
    targets.append(
        TargetRule(
            scope_module_classes=LTX2VideoTransformerBlock,
            module_classes=nn.Linear,
        )
    )
    return TargetConfig(
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=LTX2VideoTransformerBlock,
                prev_replay_transform=_ltx2_block_prev_replay_transform,
            )
        ],
        skips=[
            SkipRule(modules=["transformer_blocks.*.*.to_gate_logits"]),
        ],
        targets=targets,
    )


def _ltx2_fused_qkv_targets() -> list[TargetRule]:
    """Return grouped LTX attention targets whose members share activation inputs."""

    return [
        TargetRule(
            name="video_self_qkv",
            modules=[
                "transformer_blocks.*.attn1.to_q",
                "transformer_blocks.*.attn1.to_k",
                "transformer_blocks.*.attn1.to_v",
            ],
            export_name="transformer_blocks.{0}.attn1.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule(
            name="audio_self_qkv",
            modules=[
                "transformer_blocks.*.audio_attn1.to_q",
                "transformer_blocks.*.audio_attn1.to_k",
                "transformer_blocks.*.audio_attn1.to_v",
            ],
            export_name="transformer_blocks.{0}.audio_attn1.qkv_proj",
            roles=["q", "k", "v"],
        ),
        TargetRule(
            name="video_text_kv",
            modules=[
                "transformer_blocks.*.attn2.to_k",
                "transformer_blocks.*.attn2.to_v",
            ],
            export_name="transformer_blocks.{0}.attn2.kv_proj",
            roles=["k", "v"],
        ),
        TargetRule(
            name="audio_text_kv",
            modules=[
                "transformer_blocks.*.audio_attn2.to_k",
                "transformer_blocks.*.audio_attn2.to_v",
            ],
            export_name="transformer_blocks.{0}.audio_attn2.kv_proj",
            roles=["k", "v"],
        ),
        TargetRule(
            name="audio_to_video_kv",
            modules=[
                "transformer_blocks.*.audio_to_video_attn.to_k",
                "transformer_blocks.*.audio_to_video_attn.to_v",
            ],
            export_name="transformer_blocks.{0}.audio_to_video_attn.kv_proj",
            roles=["k", "v"],
        ),
        TargetRule(
            name="video_to_audio_kv",
            modules=[
                "transformer_blocks.*.video_to_audio_attn.to_k",
                "transformer_blocks.*.video_to_audio_attn.to_v",
            ],
            export_name="transformer_blocks.{0}.video_to_audio_attn.kv_proj",
            roles=["k", "v"],
        ),
    ]


def validate_ltx2_3_nunchaku_lite_manifest_targets(
    transformer: nn.Module, target_config: TargetConfig
) -> None:
    """Reject LTX2 targets that the nunchaku_lite manifest runtime cannot load."""

    targets = collect_quant_targets(transformer, target_config)
    unsupported_gate_targets = sorted(
        target.export_name
        for target in targets
        if not (
            isinstance(target.quant, AwqTargetQuant)
            and isinstance(target.quant.layout, AwqW4A16Layout)
        )
        and (
            target.export_name.endswith(".to_gate_logits")
            or any(module_name.endswith(".to_gate_logits") for module_name in target.module_names)
        )
    )
    if unsupported_gate_targets:
        formatted = ", ".join(unsupported_gate_targets)
        raise RuntimeError(
            "LTX-2.3 nunchaku_lite manifest export requires attention gate logits "
            f"to be dense or AWQ W4A16; skip or change quantization targets for: {formatted}"
        )


if __name__ == "__main__":
    run_model_cli()
