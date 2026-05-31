"""Quantize LongCat Image Edit with upstream DeepCompressor-style SVDQuant config."""

import argparse
import logging
import sys
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
    TargetConfig,
    TargetRule,
    inspect_target_config,
    quantize_and_export,
)


Precision = Literal["int4", "nvfp4"]
SvdBackend = Literal["full", "svd_lowrank"]
PromptRecord = dict[str, object]


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
        raise TypeError("Flux block replay output must be (encoder_hidden_states, hidden_states)")
    encoder_hidden_states, hidden_states = replay.output
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = hidden_states
        kwargs["encoder_hidden_states"] = encoder_hidden_states
        return (), kwargs
    args = list(replay.args)
    if len(args) < 2:
        raise TypeError("Flux block replay args must include hidden_states and encoder_hidden_states")
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
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default="int4", help="Weight precision overlay.")
    parser.add_argument("--model-id", default=model_id, help="Hugging Face model id or local pipeline directory.")
    parser.add_argument("--output", default=output, help="Output safetensors checkpoint path.")
    parser.add_argument("--num-samples", type=int, default=128, help="Number of calibration prompts or records to use.")
    parser.add_argument(
        "--cache-num-samples",
        type=int,
        default=None,
        help=(
            "Number of cached model-forward records to replay. Defaults to --num-samples; "
            "use -1 to replay every cached calibration record."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=batch_size, help="Calibration DataLoader batch size.")
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
    parser.add_argument("--image-edit-dataset", default="VyoJ/NHR-Edit-Change_Only", help="Dataset id used by image-edit examples.")
    parser.add_argument("--image-edit-split", default="validation", help="Dataset split used by image-edit calibration.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Pipeline execution device.")
    parser.add_argument("--compute-device", default=None, help="Optional per-target quantization compute device; useful with --offload-model.")
    parser.add_argument("--offload-model", action="store_true", help="Move the transformer back to CPU between quantization work.")
    parser.add_argument(
        "--pipeline-offload",
        choices=("none", "model", "sequential"),
        default="none",
        help="Enable Diffusers pipeline CPU offload while collecting calibration inputs.",
    )
    parser.add_argument("--height", type=int, default=height, help="Calibration image height.")
    parser.add_argument("--width", type=int, default=width, help="Calibration image width.")
    parser.add_argument("--steps", type=int, default=steps, help="Denoising steps for calibration forwards.")
    parser.add_argument("--guidance-scale", type=float, default=guidance_scale, help="Guidance scale for calibration forwards.")
    parser.add_argument("--svd-backend", choices=("full", "svd_lowrank"), default="full", help="Low-rank decomposition backend.")
    parser.add_argument("--svd-lowrank-oversample", type=int, default=10, help="Oversampling rank for torch.svd_lowrank.")
    parser.add_argument("--svd-lowrank-niter", type=int, default=4, help="Power iterations for torch.svd_lowrank.")
    parser.add_argument("--log-dir", default="outputs/logs", help="Directory for quantization process and target logs.")
    parser.add_argument("--no-run-log", action="store_true", help="Disable quantization run log files.")
    parser.add_argument("--inspect-config", action="store_true", help="Print target-config diagnostics and exit.")
    return parser


def run_model_cli() -> None:
    """Load one example pipeline and run quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    output = "outputs/checkpoints/svdq-int4_r32-longcat-image-edit.safetensors"
    parser = default_arg_parser(
        "meituan-longcat/LongCat-Image-Edit-Turbo",
        output,
        steps=8,
        guidance_scale=1.0,
        batch_size=1,
        height=512,
        width=512,
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = f"outputs/checkpoints/svdq-{args.precision}_r32-longcat-image-edit.safetensors"
    cache_dir = args.cache_dir or "outputs/calibration/longcat-image-edit"
    pipe = load_pipeline(
        "LongCatImageEditPipeline",
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
    )
    target_config = longcat_image_edit_target_config(args.precision)
    if args.inspect_config:
        print(inspect_target_config(pipe.transformer, target_config).format_text())
        return
    records = image_edit_records(args.num_samples, dataset=args.image_edit_dataset, split=args.image_edit_split)
    forward_fn = image_edit_forward_fn(
        pipe,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        height=args.height,
        width=args.width,
    )
    artifact_cache = None
    if cache_dir is not None and args.cache_mode != "disabled":
        artifact_cache = QuantizationCacheSpec(cache_dir=Path(cache_dir) / args.precision / "artifacts", cache_mode=args.cache_mode)
    output_dir = None if cache_dir is None else Path(cache_dir) / args.precision / "inputs" / "samples"
    quantize_and_export(
        model=pipe.transformer,
        spec=svdquant_spec(
            args.precision,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            compute_device=args.compute_device or (args.device if args.offload_model else None),
            offload_model=args.offload_model,
        ),
        target_config=target_config,
        calibration=CalibrationSpec(
            samples=batched_samples(records, args.batch_size),
            num_samples=args.num_samples,
            cache_num_samples=args.num_samples if args.cache_num_samples is None else args.cache_num_samples,
            batch_size=args.batch_size,
            cache_dir=None if cache_dir is None else Path(cache_dir) / args.precision / "inputs",
            cache_mode=args.cache_mode,
            forward_fn=forward_fn,
            output_dir=output_dir,
            output_save_fn=save_diffusers_images,
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size or args.batch_size,
            artifact_cache=artifact_cache,
        ),
        export=ExportSpec(output=Path(args.output)),
        logging=LoggingConfig(enabled=not args.no_run_log, log_dir=args.log_dir, name=Path(args.output).stem),
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
    method_name = "enable_model_cpu_offload" if pipeline_offload == "model" else "enable_sequential_cpu_offload"
    method = getattr(pipe, method_name, None)
    if method is None:
        raise RuntimeError(f"{pipeline_name} does not support {method_name}()")
    try:
        method(device=device)
    except TypeError:
        method()
    return pipe


def image_edit_forward_fn(
    pipe,
    *,
    steps: int,
    guidance_scale: float,
    device: str,
    height: int | None = None,
    width: int | None = None,
) -> Callable[[dict], object]:
    """Create a calibration forward function for LongCat image-edit pipelines."""

    def forward(sample: dict) -> object:
        return _call_image_edit_pipeline(
            pipe,
            height=height,
            width=width,
            image=sample["image"],
            prompt=sample["prompt"],
            negative_prompt="",
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=make_generator(sample.get("seed", 0), device=device),
        )

    return forward


def _call_image_edit_pipeline(pipe, *, height: int | None, width: int | None, **kwargs):
    """Call an image-edit pipeline, overriding LongCat target dimensions when requested."""

    if height is None or width is None:
        return pipe(**kwargs)
    if height <= 0 or width <= 0:
        raise ValueError("image-edit calibration height and width must be positive")
    module = sys.modules.get(pipe.__class__.__module__)
    calculate_dimensions = getattr(module, "calculate_dimensions", None) if module is not None else None
    if not callable(calculate_dimensions):
        return pipe(**kwargs)
    target_height = height if height % 16 == 0 else (height // 16 + 1) * 16
    target_width = width if width % 16 == 0 else (width // 16 + 1) * 16

    def fixed_dimensions(_target_area, _ratio):
        return target_width, target_height

    setattr(module, "calculate_dimensions", fixed_dimensions)
    try:
        return pipe(**kwargs)
    finally:
        setattr(module, "calculate_dimensions", calculate_dimensions)


def save_diffusers_images(result: object, sample: dict, output_dir: Path) -> None:
    """Save generated Diffusers images using calibration sample filenames."""

    images = getattr(result, "images", None)
    if images is None:
        raise ValueError("Diffusers calibration output must expose an images attribute")
    filenames = _as_list(sample.get("filename"))
    if not filenames:
        filenames = [f"{int(seed):04d}-0" for seed in _as_list(sample.get("seed"))]
    if len(filenames) != len(images):
        raise ValueError(f"Expected {len(filenames)} image filenames, got {len(images)} images")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, image in zip(filenames, images, strict=True):
        image.save(output_dir / f"{filename}.png")


def image_edit_records(
    num_samples: int,
    *,
    dataset: str = "VyoJ/NHR-Edit-Change_Only",
    split: str = "validation",
    image_size: int = 512,
) -> list[PromptRecord]:
    """Return LongCat image-edit calibration records from a Hugging Face dataset."""

    import datasets

    loaded = datasets.load_dataset(dataset, split=split)
    records: list[PromptRecord] = []
    limit = len(loaded) if num_samples < 0 else min(num_samples, len(loaded))
    for index in range(limit):
        row = loaded[index]
        filename = str(row.get("filename") or row.get("sample_id") or index)
        image = row.get("source_image") or row.get("source")
        if image is None:
            raise ValueError(f"Image-edit sample {filename!r} does not contain source_image or source")
        records.append(
            {
                "filename": filename,
                "prompt": str(row.get("prompt") or row.get("edit_instruction")),
                "image": _resize_image_edit_image(image, image_size),
                "seed": _hash_str_to_int(filename),
            }
        )
    return records


def batched_samples(prompts: list[str] | list[PromptRecord], batch_size: int) -> list[dict]:
    """Pack prompts and seeds into calibration sample dictionaries."""

    samples = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch = prompts[start:end]
        if batch and isinstance(batch[0], dict):
            filenames = [str(item["filename"]) for item in batch]  # type: ignore[index]
            prompt_batch = [str(item["prompt"]) for item in batch]  # type: ignore[index]
            seeds = [int(item["seed"]) for item in batch]  # type: ignore[index]
            images = [item["image"] for item in batch if "image" in item]  # type: ignore[operator]
        else:
            filenames = [f"{index:04d}-0" for index in range(start, end)]
            prompt_batch = [str(item) for item in batch]
            seeds = list(range(start, end))
            images = []
        sample = {
            "filename": filenames[0] if len(filenames) == 1 else filenames,
            "prompt": prompt_batch[0] if len(prompt_batch) == 1 else prompt_batch,
            "seed": seeds[0] if len(seeds) == 1 else seeds,
        }
        if images:
            if len(images) != len(filenames):
                raise ValueError("Image-edit records must include one image per prompt")
            sample["image"] = images[0] if len(images) == 1 else images
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


def _resize_image_edit_image(image: object, image_size: int) -> object:
    """Center-crop and resize an image-edit source image when possible."""

    if image_size <= 0 or not hasattr(image, "size") or not hasattr(image, "resize"):
        return image
    width, height = image.size
    crop = min(width, height)
    left = (width - crop) // 2
    top = (height - crop) // 2
    image = image.crop((left, top, left + crop, top + crop)) if hasattr(image, "crop") else image
    image = image.resize((image_size, image_size))
    return image.convert("RGB") if hasattr(image, "convert") else image


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
            shared_low_rank=False,
            precision="int4",
            group_size=64,
            rank=0,
            smooth=False,
            activation_quant=False,
            shift_activations=False,
            weight_layout=AdaNormAwqW4A16Layout(splits=6),
        ),
        TargetRule(
            modules=["transformer_blocks.*.norm1_context.linear"],
            shared_low_rank=False,
            precision="int4",
            group_size=64,
            rank=0,
            smooth=False,
            activation_quant=False,
            shift_activations=False,
            weight_layout=AdaNormAwqW4A16Layout(splits=6),
        ),
        TargetRule(
            modules=["single_transformer_blocks.*.norm.linear"],
            shared_low_rank=False,
            precision="int4",
            group_size=64,
            rank=0,
            smooth=False,
            activation_quant=False,
            shift_activations=False,
            weight_layout=AdaNormAwqW4A16Layout(splits=3),
        ),
    ]


def longcat_image_edit_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return a LongCat Image Edit target config for manifest-driven export."""

    from diffusers.models.transformers.transformer_longcat_image import (
        LongCatImageSingleTransformerBlock,
        LongCatImageTransformerBlock,
    )

    target_patterns = [
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
        "transformer_blocks.*.attn.to_out.0",
        "transformer_blocks.*.attn.add_q_proj",
        "transformer_blocks.*.attn.add_k_proj",
        "transformer_blocks.*.attn.add_v_proj",
        "transformer_blocks.*.attn.to_add_out",
        "transformer_blocks.*.ff.net.0.proj",
        "transformer_blocks.*.ff.net.2",
        "transformer_blocks.*.ff_context.net.0.proj",
        "transformer_blocks.*.ff_context.net.2",
        "single_transformer_blocks.*.attn.to_q",
        "single_transformer_blocks.*.attn.to_k",
        "single_transformer_blocks.*.attn.to_v",
        "single_transformer_blocks.*.proj_mlp",
        "single_transformer_blocks.*.proj_out.linears.0",
        "single_transformer_blocks.*.proj_out.linears.1",
    ]
    targets = [TargetRule(modules=[pattern]) for pattern in target_patterns]
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
                module_classes=LongCatImageTransformerBlock,
                prev_replay_transform=_flux_block_prev_replay_transform,
            ),
            CalibrationScopeRule(
                module_classes=LongCatImageSingleTransformerBlock,
                prev_replay_transform=_flux_block_prev_replay_transform,
            ),
        ],
        targets=targets,
    )


if __name__ == "__main__":
    run_model_cli()
