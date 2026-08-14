"""Quantize the MiniMax H3 FL2VA transformer for Nunchaku SVDQuant."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from diffuse_compressor import (
    AwqTargetQuant,
    AwqW4A16Layout,
    CalibrationScopeRule,
    CalibrationSpec,
    ExportSpec,
    GptqSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    inspect_target_config,
    quantize_and_export,
)
from examples.text_to_image.utils import (
    batched_samples,
    make_generator,
    standard_prompt_records,
)
from examples.text_to_video.utils import DEFAULT_QDIFF_PROMPT_FILE, svdquant_spec


DEFAULT_MODEL_ID = "MiniMaxAI/MiniMax-H3"
_H3_STACKS = (
    ("token_refiner.refiner_blocks", "token_refiner.blocks"),
    ("transformer_blocks", "blocks"),
)
_H3_PROJECTION_RENAMES = (
    ("attn.to_out.0", "attn.out_proj"),
    ("ff.net.0.proj", "mlp.fc1"),
    ("ff.net.2", "mlp.fc2"),
)


def format_minimax_h3_prompt(prompt: str) -> str:
    """Wrap a plain caption in MiniMax H3's native audiovisual prompt form."""

    prompt = prompt.strip()
    if "integrated_multimodal_description:" in prompt:
        return prompt
    return (
        f"integrated_multimodal_description: [Shot 1] {prompt}\n\n"
        "overall_soundscape: Natural ambient sound synchronized with the visible action.\n\n"
        "non_diegetic_music: N/A"
    )


def minimax_h3_prompt_records(num_samples: int, prompt_file: str | Path) -> list[dict]:
    """Load deterministic calibration records and apply H3 prompt formatting."""

    records = standard_prompt_records(num_samples, prompt_file)
    return [
        {**record, "prompt": format_minimax_h3_prompt(str(record["prompt"]))}
        for record in records
    ]


def _h3_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next H3 block input from the previous block replay."""

    hidden_states = replay.output
    if not torch.is_tensor(hidden_states):
        raise TypeError("MiniMax H3 block replay output must be a hidden-state tensor")
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        if "hidden_states" not in kwargs:
            raise TypeError("MiniMax H3 block replay kwargs must include hidden_states")
        kwargs["hidden_states"] = hidden_states
        return (), kwargs
    args = list(replay.args)
    if not args:
        raise TypeError("MiniMax H3 block replay args must include hidden_states")
    args[0] = hidden_states
    return tuple(args), {}


def minimax_h3_target_config() -> TargetConfig:
    """Return the Nunchaku target map for the FL2VA transformer.

    Attention and feed-forward projections use SVDQuant W4A4. Per-block
    AdaLN projections use plain AWQ W4A16. Input, timestep, context, and final
    projections stay in their original BF16/FP32 dtype.
    """

    targets = []
    for source_stack, export_stack in _H3_STACKS:
        targets.append(
            TargetRule(
                name=f"{export_stack}.qkv",
                modules=tuple(
                    f"{source_stack}.*.attn.to_{role}" for role in ("q", "k", "v")
                ),
                export_name=f"{export_stack}.{{0}}.attn.qkv_proj",
                roles=("q", "k", "v"),
                quant=SvdqTargetQuant(),
            )
        )
        targets.extend(
            TargetRule(
                modules=(f"{source_stack}.*.{source_projection}",),
                export_name=f"{export_stack}.{{0}}.{export_projection}",
                quant=SvdqTargetQuant(),
            )
            for source_projection, export_projection in _H3_PROJECTION_RENAMES
        )
    targets.append(
        TargetRule(
            modules=("transformer_blocks.*.adaln_proj.linear",),
            export_name="blocks.{0}.adaln_proj.linear",
            quant=AwqTargetQuant(layout=AwqW4A16Layout()),
        )
    )
    return TargetConfig(
        targets=targets,
        calibration_scopes=(
            CalibrationScopeRule(
                modules=("token_refiner.refiner_blocks.*",),
                prev_replay_transform=_h3_block_prev_replay_transform,
            ),
            CalibrationScopeRule(
                modules=("transformer_blocks.0",),
                use_prev_scope_outputs=False,
            ),
            CalibrationScopeRule(
                modules=("transformer_blocks.*",),
                prev_replay_transform=_h3_block_prev_replay_transform,
            ),
        ),
    )


def load_minimax_h3_pipeline(
    model_id: str,
    *,
    device: str,
    pipeline_offload: str,
    memory_reserve_margin: str,
    text_encoder_path: str | None = None,
    decode_outputs: bool = False,
):
    """Load the FL2VA pipeline with optional latent-only VAE proxies."""

    try:
        from diffusers import ComponentsManager, ModularPipeline
    except ImportError as exc:
        raise ImportError(
            "MiniMax H3 requires a Diffusers build containing MiniMaxH3Blocks "
            "and MiniMaxH3Transformer3DModel."
        ) from exc

    manager = ComponentsManager() if pipeline_offload == "auto" else None
    pipe = ModularPipeline.from_pretrained(model_id, components_manager=manager)
    names = pipe.pretrained_component_names
    if not decode_outputs:
        names = [name for name in names if name not in {"vae", "audio_vae"}]
    load_kwargs = {"dtype": torch.bfloat16}
    component_paths = {}
    component_subfolders = {}
    local_model_dir = Path(model_id)
    if local_model_dir.is_dir():
        for name in names:
            spec = pipe.get_component_spec(name)
            component_paths[name] = str(local_model_dir)
            component_subfolders[name] = spec.subfolder
    if text_encoder_path is not None:
        component_paths["text_encoder"] = text_encoder_path
        component_subfolders["text_encoder"] = ""
    if component_paths:
        load_kwargs.update(
            pretrained_model_name_or_path=component_paths,
            subfolder=component_subfolders,
        )
    pipe.load_components(names=names, **load_kwargs)

    if not decode_outputs:
        video_spec = pipe.get_component_spec("vae")
        audio_spec = pipe.get_component_spec("audio_vae")

        def load_config(spec):
            config_model_id = (
                str(local_model_dir)
                if local_model_dir.is_dir()
                else spec.pretrained_model_name_or_path
            )
            return spec.type_hint.load_config(
                config_model_id,
                subfolder=spec.subfolder,
                revision=spec.revision,
            )

        video_config = load_config(video_spec)
        audio_config = load_config(audio_spec)
        pipe.vae = SimpleNamespace(
            config=SimpleNamespace(**video_config),
            spatial_compression_ratio=16,
        )
        pipe.audio_vae = SimpleNamespace(config=SimpleNamespace(**audio_config))
    if manager is None:
        pipe.to(device)
    else:
        manager.enable_auto_cpu_offload(
            device=device, memory_reserve_margin=memory_reserve_margin
        )
    return pipe


def minimax_h3_forward_fn(pipe, args):
    """Build the FL2VA calibration forward callable."""

    def forward(sample: dict):
        return pipe(
            prompt=sample["prompt"],
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.steps,
            generator=make_generator(sample.get("seed", 0), device=args.device),
            output_type="pil" if args.save_calibration_videos else "latent",
            output=["videos", "audio", "sampling_rate"],
        )

    return forward


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def save_minimax_h3_videos(state, sample: dict, output_dir: Path, *, fps: int) -> None:
    """Mux calibration video and audio returned in a Modular PipelineState."""

    from diffusers.utils.export_utils import encode_video

    videos = _as_list(state.get("videos"))
    audios = _as_list(state.get("audio"))
    sampling_rates = _as_list(state.get("sampling_rate"))
    filenames = _as_list(sample.get("filename"))
    if not filenames:
        filenames = [f"{int(seed):04d}-0" for seed in _as_list(sample.get("seed"))]
    if len(filenames) != len(videos):
        raise ValueError(f"Expected {len(filenames)} videos, got {len(videos)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (filename, video) in enumerate(zip(filenames, videos, strict=True)):
        audio = audios[index] if index < len(audios) else None
        sampling_rate = sampling_rates[index] if index < len(sampling_rates) else None
        encode_video(
            video,
            fps=fps,
            output_path=str(output_dir / f"{filename}.mp4"),
            audio=audio,
            audio_sample_rate=sampling_rate,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Quantize MiniMax H3 FL2VA to a Nunchaku-compatible SVDQuant checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--text-encoder-path")
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default="nvfp4")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--output")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--cache-num-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=-1)
    parser.add_argument(
        "--row-sampling",
        choices=("head", "reservoir"),
        default="reservoir",
    )
    parser.add_argument("--max-eval-replays", type=int, default=4)
    parser.add_argument(
        "--scope-capture-mode",
        choices=("all-targets", "one-target"),
        default="all-targets",
    )
    parser.add_argument("--prompt-file", default=DEFAULT_QDIFF_PROMPT_FILE)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--svd-backend", choices=("full", "svd_lowrank"), default="svd_lowrank"
    )
    parser.add_argument("--svd-lowrank-oversample", type=int, default=10)
    parser.add_argument("--svd-lowrank-niter", type=int, default=4)
    parser.add_argument("--gptq", action="store_true")
    parser.add_argument("--gptq-damp-percentage", type=float, default=0.01)
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--gptq-num-inv-tries", type=int, default=250)
    parser.add_argument("--gptq-hessian-block-size", type=int, default=512)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--compute-device")
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--pipeline-offload", choices=("none", "auto"), default="none")
    parser.add_argument("--memory-reserve-margin", default="12GB")
    parser.add_argument("--cache-dir")
    parser.add_argument(
        "--cache-mode", choices=("reuse", "refresh", "disabled"), default="reuse"
    )
    parser.add_argument("--save-model-cache", action="store_true")
    parser.add_argument("--save-calibration-videos", action="store_true")
    parser.add_argument("--log-dir", default="outputs/logs")
    parser.add_argument("--no-run-log", action="store_true")
    parser.add_argument("--inspect-config", action="store_true")
    return parser


def _slug(model_id: str) -> str:
    return re.sub(
        r"[^a-z0-9]+", "-", model_id.rstrip("/").split("/")[-1].lower()
    ).strip("-")


def validate_minimax_h3_args(args) -> None:
    """Validate generation geometry before loading the 120+ GiB pipeline."""

    if args.batch_size != 1:
        raise ValueError(
            "MiniMax H3 supports exactly one prompt per packed sequence; use --batch-size 1"
        )
    if args.height <= 0 or args.width <= 0 or args.height % 32 or args.width % 32:
        raise ValueError("MiniMax H3 height and width must be positive multiples of 32")
    if not 124 <= args.num_frames <= 362 or (args.num_frames - 5) % 17:
        raise ValueError(
            "MiniMax H3 num_frames must be 17*n+5 in the supported 5-15 second range (124-362)"
        )
    if args.steps < 2:
        raise ValueError(
            "MiniMax H3 steps includes the terminal sigma and must be at least 2"
        )


def run() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    validate_minimax_h3_args(args)
    pipe = load_minimax_h3_pipeline(
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
        memory_reserve_margin=args.memory_reserve_margin,
        text_encoder_path=args.text_encoder_path,
        decode_outputs=args.save_calibration_videos,
    )
    model = pipe.transformer
    target_config = minimax_h3_target_config()
    report = inspect_target_config(model, target_config)
    print(report.format_text())
    if not report.ok:
        raise ValueError("MiniMax H3 target configuration inspection failed")
    if args.inspect_config:
        return

    precision_label = "fp4" if args.precision == "nvfp4" else "int4"
    method_label = "svdq-gptq" if args.gptq else "svdq"
    output = Path(
        args.output
        or f"outputs/checkpoints/{method_label}-{precision_label}_r{args.rank}-{_slug(args.model_id)}-fl2va.safetensors"
    )
    cache_dir = Path(
        args.cache_dir or f"outputs/calibration/{_slug(args.model_id)}-fl2va"
    )
    artifact_label = (
        f"{'gptq-' if args.gptq else ''}{args.row_sampling}"
        f"-n{args.num_samples}-replay{args.max_eval_replays or 'rows'}-artifacts"
    )
    artifact_cache = (
        None
        if args.cache_mode == "disabled"
        else QuantizationCacheSpec(
            cache_dir / args.precision / artifact_label,
            args.cache_mode,
            save_model=args.save_model_cache,
        )
    )
    spec = replace(
        svdquant_spec(
            args.precision,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            compute_device=args.compute_device or args.device,
            offload_model=args.offload_model,
        ),
        rank=args.rank,
        gptq=GptqSpec(
            enabled=args.gptq,
            damp_percentage=args.gptq_damp_percentage,
            block_size=args.gptq_block_size,
            num_inv_tries=args.gptq_num_inv_tries,
            hessian_block_size=args.gptq_hessian_block_size,
        ),
    )
    save_outputs = args.save_calibration_videos
    quantize_and_export(
        model=model,
        spec=spec,
        target_config=target_config,
        calibration=CalibrationSpec(
            samples=batched_samples(
                minimax_h3_prompt_records(args.num_samples, args.prompt_file),
                args.batch_size,
            ),
            num_samples=args.num_samples,
            cache_num_samples=(
                args.num_samples
                if args.cache_num_samples is None
                else args.cache_num_samples
            ),
            batch_size=args.batch_size,
            cache_dir=cache_dir / args.precision / "inputs",
            cache_mode=args.cache_mode,
            forward_fn=minimax_h3_forward_fn(pipe, args),
            output_dir=cache_dir / args.precision / "inputs" / "samples"
            if save_outputs
            else None,
            output_save_fn=partial(save_minimax_h3_videos, fps=args.fps)
            if save_outputs
            else None,
            max_rows_per_target=4096,
            row_sampling=args.row_sampling,
            max_eval_replays=args.max_eval_replays,
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size,
            artifact_cache=artifact_cache,
        ),
        export=ExportSpec(output=output),
        logging=LoggingConfig(
            enabled=not args.no_run_log,
            log_dir=args.log_dir,
            name=output.stem,
        ),
    )


if __name__ == "__main__":
    run()
