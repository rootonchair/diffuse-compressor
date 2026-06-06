"""Quantize FLUX.1 Dev with upstream DeepCompressor-style SVDQuant config."""

from __future__ import annotations

import argparse
import logging
import random
import re
from pathlib import Path
from typing import Callable, Literal

import torch

from diffuse_compressor import (
    ActivationQuantSpec,
    AdaNormAwqW4A16Layout,
    CalibrationScopeRule,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    LoggingConfig,
    LowRankSolverSpec,
    PatchRule,
    QuantizationCacheSpec,
    RangeCalibrationSpec,
    SmoothSpec,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    AwqTargetQuant,
    inspect_target_config,
    quantize_and_export,
)


Precision = Literal["int4", "nvfp4"]
SvdBackend = Literal["full", "svd_lowrank"]
PromptRecord = dict[str, object]
DEFAULT_QDIFF_PROMPT_FILE = (
    Path(__file__).resolve().parents[1] / "prompts" / "qdiff.yaml"
)


def svdquant_spec(
    precision: Precision,
    *,
    svd_backend: SvdBackend = "full",
    svd_lowrank_oversample: int = 10,
    svd_lowrank_niter: int = 4,
    compute_device: str | None = None,
    offload_model: bool = False,
) -> DiffusionQuantSpec:
    """Build an upstream-style SVDQuant spec for one precision overlay."""

    if precision == "int4":
        return DiffusionQuantSpec(
            shift_activations=True,
            compute_device=compute_device,
            offload_model=offload_model,
            low_rank_solver=_low_rank_solver(
                svd_backend=svd_backend,
                svd_lowrank_oversample=svd_lowrank_oversample,
                svd_lowrank_niter=svd_lowrank_niter,
            ),
            smooth=_smooth_spec(),
            activation_quant=ActivationQuantSpec(
                enabled=True,
                static=False,
                inputs=RangeCalibrationSpec(granularity="group", allow_unsigned=True),
            ),
        )
    if precision == "nvfp4":
        return DiffusionQuantSpec(
            precision="fp4",
            group_size=16,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            compute_device=compute_device,
            offload_model=offload_model,
            low_rank_solver=_low_rank_solver(
                svd_backend=svd_backend,
                svd_lowrank_oversample=svd_lowrank_oversample,
                svd_lowrank_niter=svd_lowrank_niter,
            ),
            smooth=_smooth_spec(),
            activation_quant=ActivationQuantSpec(
                enabled=True,
                static=False,
                scale_dtypes=("sfp8_e4m3_nan",),
                inputs=RangeCalibrationSpec(granularity="group", allow_unsigned=True),
            ),
        )
    raise ValueError(f"Unsupported precision: {precision!r}")


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


def default_arg_parser(
    model_id: str,
    output: str,
    *,
    steps: int,
    guidance_scale: float,
    batch_size: int,
    height: int = 1024,
    width: int = 1024,
) -> argparse.ArgumentParser:
    """Create a shared CLI parser for upstream diffusion examples."""

    parser = argparse.ArgumentParser(
        description="Quantize one supported Diffusers transformer with the shared SVDQuant example config.",
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
        help="Hugging Face model id or local pipeline directory.",
    )
    parser.add_argument(
        "--output", default=output, help="Output safetensors checkpoint path."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=128,
        help="Number of calibration prompts or records to use.",
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
        default=None,
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
        help="Calibration/artifact cache root; defaults to outputs/calibration/<model>.",
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
        "--height", type=int, default=height, help="Calibration image height."
    )
    parser.add_argument(
        "--width", type=int, default=width, help="Calibration image width."
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
        "--svd-backend",
        choices=("full", "svd_lowrank"),
        default="full",
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
        ),
        export=ExportSpec(output=Path(args.output)),
        logging=LoggingConfig(
            enabled=not args.no_run_log,
            log_dir=args.log_dir,
            name=Path(args.output).stem,
        ),
    )


def load_pipeline(
    pipeline_name: str,
    model_id: str,
    *,
    device: str,
    pipeline_offload: Literal["none", "model", "sequential"] = "none",
):
    """Load a diffusers pipeline by class name."""

    import diffusers

    pipeline_cls = getattr(diffusers, pipeline_name)
    pipe = pipeline_cls.from_pretrained(model_id)
    if pipeline_offload == "none":
        return pipe.to(device)
    method_name = (
        "enable_model_cpu_offload"
        if pipeline_offload == "model"
        else "enable_sequential_cpu_offload"
    )
    method = getattr(pipe, method_name, None)
    if method is None:
        raise RuntimeError(f"{pipeline_name} does not support {method_name}()")
    try:
        method(device=device)
    except TypeError:
        method()
    return pipe


def pipeline_forward_fn(
    pipe,
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    device: str,
    use_pe: bool | None = None,
) -> Callable[[dict], object]:
    """Create a calibration forward function for a diffusers pipeline."""

    def forward(sample: dict) -> object:
        kwargs = {
            "prompt": sample["prompt"],
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "generator": make_generator(sample.get("seed", 0), device=device),
        }
        if use_pe is not None:
            kwargs["use_pe"] = use_pe
        return pipe(**kwargs)

    return forward


def save_diffusers_images(result: object, sample: dict, output_dir: Path) -> None:
    """Save generated Diffusers images using calibration sample filenames."""

    images = getattr(result, "images", None)
    if images is None:
        raise ValueError("Diffusers calibration output must expose an images attribute")
    filenames = _as_list(sample.get("filename"))
    if not filenames:
        filenames = [f"{int(seed):04d}-0" for seed in _as_list(sample.get("seed"))]
    if len(filenames) != len(images):
        raise ValueError(
            f"Expected {len(filenames)} image filenames, got {len(images)} images"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, image in zip(filenames, images, strict=True):
        image.save(output_dir / f"{filename}.png")


def standard_prompt_records(
    num_samples: int,
    prompt_file: str | Path = DEFAULT_QDIFF_PROMPT_FILE,
) -> list[PromptRecord]:
    """Return qdiff calibration prompt records."""

    meta = _parse_qdiff_prompt_yaml(Path(prompt_file).read_text(encoding="utf-8"))
    names = list(meta)
    if num_samples > 0:
        random.Random(0).shuffle(names)
        names = sorted(names[:num_samples])
    return [
        {
            "filename": f"{name}-0",
            "prompt": meta[name],
            "seed": _hash_str_to_int(f"{name}-0"),
        }
        for name in names
    ]


def batched_samples(
    prompts: list[str] | list[PromptRecord], batch_size: int
) -> list[dict]:
    """Pack prompts and seeds into calibration sample dictionaries."""

    samples = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch = prompts[start:end]
        if batch and isinstance(batch[0], dict):
            filenames = [str(item["filename"]) for item in batch]  # type: ignore[index]
            prompt_batch = [str(item["prompt"]) for item in batch]  # type: ignore[index]
            seeds = [int(item["seed"]) for item in batch]  # type: ignore[index]
        else:
            filenames = [f"{index:04d}-0" for index in range(start, end)]
            prompt_batch = [str(item) for item in batch]
            seeds = list(range(start, end))
        sample = {
            "filename": filenames[0] if len(filenames) == 1 else filenames,
            "prompt": prompt_batch[0] if len(prompt_batch) == 1 else prompt_batch,
            "seed": seeds[0] if len(seeds) == 1 else seeds,
        }
        samples.append(sample)
    return samples


def _as_list(value: object) -> list:
    """Return a scalar or sequence value as a list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _parse_qdiff_prompt_yaml(text: str) -> dict[str, str]:
    """Parse the simple key/value qdiff prompt YAML without adding PyYAML."""

    prompts: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []
    entry_pattern = re.compile(r"^'?(?P<key>\d{4})'?:\s*(?P<value>.*)$")

    def flush() -> None:
        if current_key is not None:
            prompts[current_key] = _normalize_qdiff_value(" ".join(current_value))

    for line in text.splitlines():
        if not line.strip():
            continue
        match = entry_pattern.match(line)
        if match:
            flush()
            current_key = match.group("key")
            current_value = [match.group("value").strip()]
        elif current_key is not None and line[0].isspace():
            current_value.append(line.strip())
        else:
            raise ValueError(f"Unsupported qdiff prompt line: {line!r}")
    flush()
    return prompts


def _normalize_qdiff_value(value: str) -> str:
    """Normalize a qdiff YAML scalar."""

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace("''", "'")


def _hash_str_to_int(value: str) -> int:
    """Hash a string the same way upstream DeepCompressor seeds samples."""

    modulus = 10**9 + 7
    hash_int = 0
    for char in value:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def make_generator(seed: int | list[int], device: str = "cuda"):
    """Create one or more deterministic torch generators."""

    if isinstance(seed, list):
        return [torch.Generator(device=device).manual_seed(int(item)) for item in seed]
    return torch.Generator(device=device).manual_seed(int(seed))


def _low_rank_solver(
    *,
    svd_backend: SvdBackend = "full",
    svd_lowrank_oversample: int = 10,
    svd_lowrank_niter: int = 4,
) -> LowRankSolverSpec:
    """Return the upstream-style low-rank search spec."""

    return LowRankSolverSpec(
        mode="search",
        num_iters=100,
        early_stop=True,
        svd_backend=svd_backend,
        svd_lowrank_oversample=svd_lowrank_oversample,
        svd_lowrank_niter=svd_lowrank_niter,
    )


def _smooth_spec() -> SmoothSpec:
    """Return the upstream-style projection smoothing spec."""

    return SmoothSpec(
        enabled=True,
        strategy="grid_search",
        objective="outputs_error",
        alpha=0.5,
        beta=-2,
        num_grids=20,
        spans=(("absmax", "absmax"),),
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
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff_context.net.0.proj"],
            export_name="transformer_blocks.{0}.mlp_context_fc1",
        ),
        TargetRule(
            modules=["transformer_blocks.*.ff_context.net.2"],
            export_name="transformer_blocks.{0}.mlp_context_fc2",
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
