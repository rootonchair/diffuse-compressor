"""Quantize a general Hugging Face text-to-video Diffusers pipeline."""

from __future__ import annotations

import argparse
import fnmatch
import inspect
import logging
import re
import sys
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    inspect_target_config,
    quantize_and_export,
)
from diffuse_compressor.backends.nunchaku.packing import NunchakuWeightPacker
from examples.text_to_image.utils import (
    batched_samples,
    make_generator,
    standard_prompt_records,
    svdquant_spec,
)


_MODULATION_TOKENS = re.compile(r"(?:^|[._])(adaln|ada_norm|norm\d*|modulation|mod)(?:[._]|$)", re.I)
_STRONG_MODULATION_PATH = re.compile(r"(?:^|[._])(adaln|ada_norm|modulation|(?:img|txt|image|text)_mod)(?:[._]|$)", re.I)


def _class_has_modulation_signal(class_name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", class_name.lower())
    return any(
        token in compact
        for token in ("adaln", "adanorm", "adalayernormzero", "adaptivelayernormzero", "modulation")
    )


def _repeated_block_roots(model: nn.Module) -> tuple[str, ...]:
    """Return child paths from homogeneous, repeated ``ModuleList`` stacks."""

    roots: list[str] = []
    for container_name, container in model.named_modules():
        if not isinstance(container, nn.ModuleList) or len(container) < 2:
            continue
        child_type = type(container[0])
        if not all(type(child) is child_type for child in container):
            continue
        prefix = f"{container_name}." if container_name else ""
        roots.extend(f"{prefix}{index}" for index in range(len(container)))
    return tuple(roots)


def _is_under_any(name: str, roots: Sequence[str]) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in roots)


@dataclass(frozen=True)
class ScanResult:
    """Target configuration plus scanner diagnostics."""

    target_config: TargetConfig
    svdq_targets: tuple[str, ...]
    awq_targets: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    ambiguous: tuple[str, ...]

    def format_text(self) -> str:
        lines = [f"SVDQ targets ({len(self.svdq_targets)}):", *[f"  {x}" for x in self.svdq_targets]]
        lines += [f"AWQ W4A16 targets ({len(self.awq_targets)}):", *[f"  {x}" for x in self.awq_targets]]
        if self.skipped:
            lines += ["Skipped:", *[f"  {name}: {reason}" for name, reason in self.skipped]]
        if self.ambiguous:
            lines += ["Ambiguous modulation candidates kept as SVDQ:", *[f"  {x}" for x in self.ambiguous]]
        return "\n".join(lines)


def scan_linear_targets(
    model: nn.Module,
    *,
    precision: str = "int4",
    rank: int = 32,
    skip: Sequence[str] = (),
) -> ScanResult:
    """Scan stock linear paths into independent SVDQ and standard AWQ targets.

    When the model exposes homogeneous repeated block stacks, SVDQ is limited
    to those blocks. This preserves outer embeddings and output projections in
    their floating-point dtype without relying on model-family path names.
    """

    packer = NunchakuWeightPacker(bits=4)
    residual_k = packer.mem_k * packer.num_k_unrolls
    low_rank_n = packer.n_pack_size * packer.num_n_lanes
    low_rank_k = packer.k_pack_size * packer.num_k_lanes * 2
    modules = dict(model.named_modules())
    block_roots = _repeated_block_roots(model)
    svdq: list[str] = []
    awq: list[str] = []
    skipped: list[tuple[str, str]] = []
    ambiguous: list[str] = []

    for name, module in modules.items():
        if not name or not isinstance(module, nn.Linear):
            continue
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in skip):
            skipped.append((name, "selected by --skip"))
            continue

        parent_name, _, _ = name.rpartition(".")
        parent = modules.get(parent_name)
        parent_type = type(parent).__name__ if parent is not None else ""
        path_signal = bool(_MODULATION_TOKENS.search(name))
        type_signal = _class_has_modulation_signal(parent_type)
        strong_path_signal = bool(_STRONG_MODULATION_PATH.search(name))
        is_awq = strong_path_signal or (path_signal and type_signal)
        if path_signal and not is_awq:
            ambiguous.append(name)

        if is_awq:
            if module.in_features % 64 or module.out_features % 4:
                skipped.append((name, "AWQ requires input divisible by 64 and output divisible by 4"))
            else:
                awq.append(name)
            continue

        if block_roots and not _is_under_any(name, block_roots):
            skipped.append((name, "outside repeated block stacks"))
            continue

        invalid = []
        if module.out_features % packer.mem_n:
            invalid.append(f"output not divisible by {packer.mem_n}")
        if module.in_features % residual_k:
            invalid.append(f"input not divisible by {residual_k}")
        if rank and (rank % low_rank_n or module.in_features % low_rank_k or module.out_features % low_rank_n):
            invalid.append("low-rank geometry is not packable")
        if invalid:
            skipped.append((name, "; ".join(invalid)))
        else:
            svdq.append(name)

    rules = [
        TargetRule(modules=(name,), export_name=name, quant=AwqTargetQuant(layout=AwqW4A16Layout()))
        for name in awq
    ]
    rules += [TargetRule(modules=(name,), export_name=name, quant=SvdqTargetQuant()) for name in svdq]
    scopes = tuple(
        CalibrationScopeRule(modules=(root,), use_prev_scope_outputs=False)
        for root in block_roots
        if any(_is_under_any(name, (root,)) for name in svdq)
    )
    return ScanResult(
        TargetConfig(targets=rules, calibration_scopes=scopes),
        tuple(svdq),
        tuple(awq),
        tuple(skipped),
        tuple(ambiguous),
    )


def discover_denoiser(pipe):
    """Return the pipeline's stock transformer or UNet denoiser."""

    for name in ("transformer", "unet"):
        component = getattr(pipe, name, None)
        if isinstance(component, nn.Module):
            return name, component
    raise ValueError("Pipeline must expose a torch module as `transformer` or `unet`")


def load_auto_pipeline(model_id: str, *, device: str, pipeline_offload: str):
    """Load the pipeline class declared by its Diffusers model index."""

    from diffusers import DiffusionPipeline

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    if pipeline_offload == "none":
        return pipe.to(device)
    method = getattr(pipe, "enable_model_cpu_offload" if pipeline_offload == "model" else "enable_sequential_cpu_offload")
    try:
        method(device=device)
    except TypeError:
        method()
    return pipe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generically quantize a Diffusers text-to-video model.")
    parser.add_argument("model_id", help="Hugging Face model id or local Diffusers pipeline")
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default="int4")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--output")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--cache-num-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int, default=-1)
    parser.add_argument(
        "--scope-capture-mode",
        choices=("all-targets", "one-target"),
        default="all-targets",
        help=(
            "all-targets replays each scope once; one-target replays it once per target, "
            "which lowers peak RAM but multiplies replay cost by the target count."
        ),
    )
    parser.add_argument("--prompt-file", default=Path(__file__).parent.parent / "prompts" / "qdiff.yaml")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--svd-backend", choices=("full", "svd_lowrank"), default="svd_lowrank")
    parser.add_argument("--svd-lowrank-oversample", type=int, default=10)
    parser.add_argument("--svd-lowrank-niter", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-device")
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--pipeline-offload", choices=("none", "model", "sequential"), default="none")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-mode", choices=("reuse", "refresh", "disabled"), default="reuse")
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--inspect-config", action="store_true")
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    return parser


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.rstrip("/").split("/")[-1].lower()).strip("-")


def _forward_fn(pipe, args):
    signature = inspect.signature(pipe.__call__)
    accepted = set(signature.parameters)

    def forward(sample: dict):
        kwargs = {"prompt": sample["prompt"], "num_inference_steps": args.steps, "guidance_scale": args.guidance_scale}
        optional = {"height": args.height, "width": args.width, "num_frames": args.num_frames, "generator": make_generator(sample.get("seed", 0), device=args.device)}
        kwargs.update({key: value for key, value in optional.items() if key in accepted})
        return pipe(**kwargs)

    return forward


def _as_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def save_diffusers_videos(result: object, sample: dict, output_dir: Path, *, fps: int) -> None:
    """Save generated Diffusers frame sequences as MP4 calibration references."""

    from diffusers.utils import export_to_video

    videos = getattr(result, "frames", None)
    if videos is None:
        raise ValueError("Diffusers video calibration output must expose a frames attribute")
    filenames = _as_list(sample.get("filename"))
    if not filenames:
        filenames = [f"{int(seed):04d}-0" for seed in _as_list(sample.get("seed"))]
    videos = list(videos)
    if len(filenames) == 1 and len(videos) != 1:
        videos = [videos]
    if len(filenames) != len(videos):
        raise ValueError(f"Expected {len(filenames)} video filenames, got {len(videos)} videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, frames in zip(filenames, videos, strict=True):
        export_to_video(list(frames), str(output_dir / f"{filename}.mp4"), fps=fps)


def run() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    pipe = load_auto_pipeline(args.model_id, device=args.device, pipeline_offload=args.pipeline_offload)
    _, model = discover_denoiser(pipe)
    scan = scan_linear_targets(model, precision=args.precision, rank=args.rank, skip=args.skip)
    print(scan.format_text())
    if args.inspect_config:
        print(inspect_target_config(model, scan.target_config).format_text())
        return
    if not scan.svdq_targets and not scan.awq_targets:
        raise ValueError("No compatible linear targets were discovered")
    output = Path(args.output or f"outputs/checkpoints/svdq-{args.precision}_r{args.rank}-{_slug(args.model_id)}.safetensors")
    cache_dir = Path(args.cache_dir or f"outputs/calibration/{_slug(args.model_id)}")
    artifact_cache = None if args.cache_mode == "disabled" else QuantizationCacheSpec(cache_dir / args.precision / "artifacts", args.cache_mode)
    sample_output_dir = cache_dir / args.precision / "inputs" / "samples"
    spec = replace(
        svdquant_spec(
            args.precision,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            compute_device=args.compute_device or (args.device if args.offload_model else None),
            offload_model=args.offload_model,
        ),
        rank=args.rank,
    )
    quantize_and_export(
        model, spec, scan.target_config,
        CalibrationSpec(samples=batched_samples(standard_prompt_records(args.num_samples, args.prompt_file), args.batch_size), num_samples=args.num_samples,
                        cache_num_samples=args.num_samples if args.cache_num_samples is None else args.cache_num_samples,
                        batch_size=args.batch_size, cache_dir=cache_dir / args.precision / "inputs", cache_mode=args.cache_mode,
                        forward_fn=_forward_fn(pipe, args), max_rows_per_target=4096, artifact_cache=artifact_cache,
                        output_dir=sample_output_dir, output_save_fn=partial(save_diffusers_videos, fps=args.fps),
                        scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
                        sample_batch_size=args.sample_batch_size),
        ExportSpec(output=output), LoggingConfig(name=output.stem),
    )


if __name__ == "__main__":
    run()
