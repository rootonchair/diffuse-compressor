"""Shared model-scanning helpers for generic Hugging Face quantization examples."""

from __future__ import annotations

import argparse
import fnmatch
import inspect
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import nn

from diffuse_compressor import (
    AwqTargetQuant,
    AwqW4A16Layout,
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
from examples.text_to_image.utils import batched_samples, make_generator, standard_prompt_records, svdquant_spec


Task = Literal["text-to-image", "image-to-image", "text-to-video"]
_MODULATION_TOKENS = re.compile(r"(?:^|[._])(adaln|ada_norm|norm\d*|modulation|mod)(?:[._]|$)", re.I)
_STRONG_MODULATION_PATH = re.compile(r"(?:^|[._])(adaln|ada_norm|modulation|(?:img|txt|image|text)_mod)(?:[._]|$)", re.I)


def _class_has_modulation_signal(class_name: str) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", class_name.lower())
    return any(
        token in compact
        for token in ("adaln", "adanorm", "adalayernormzero", "adaptivelayernormzero", "modulation")
    )


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
    include: Sequence[str] = (),
    skip: Sequence[str] = (),
) -> ScanResult:
    """Scan stock linear paths into independent SVDQ and standard AWQ targets."""

    packer = NunchakuWeightPacker(bits=4)
    residual_k = packer.mem_k * packer.num_k_unrolls
    low_rank_n = packer.n_pack_size * packer.num_n_lanes
    low_rank_k = packer.k_pack_size * packer.num_k_lanes * 2
    modules = dict(model.named_modules())
    svdq: list[str] = []
    awq: list[str] = []
    skipped: list[tuple[str, str]] = []
    ambiguous: list[str] = []

    for name, module in modules.items():
        if not name or not isinstance(module, nn.Linear):
            continue
        if include and not any(fnmatch.fnmatchcase(name, pattern) for pattern in include):
            skipped.append((name, "not selected by --include"))
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
    return ScanResult(TargetConfig(targets=rules), tuple(svdq), tuple(awq), tuple(skipped), tuple(ambiguous))


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


def build_parser(task: Task) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Generically quantize a Diffusers {task} model.")
    parser.add_argument("model_id", help="Hugging Face model id or local Diffusers pipeline")
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default="int4")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--output")
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--cache-num-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sample-batch-size", type=int)
    parser.add_argument("--scope-capture-mode", choices=("all-targets", "one-target"), default="one-target")
    parser.add_argument("--prompt-file", default=Path(__file__).parent / "prompts" / "qdiff.yaml")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--svd-backend", choices=("full", "svd_lowrank"), default="full")
    parser.add_argument("--svd-lowrank-oversample", type=int, default=10)
    parser.add_argument("--svd-lowrank-niter", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compute-device")
    parser.add_argument("--offload-model", action="store_true")
    parser.add_argument("--pipeline-offload", choices=("none", "model", "sequential"), default="none")
    parser.add_argument("--cache-dir")
    parser.add_argument("--cache-mode", choices=("reuse", "refresh", "disabled"), default="reuse")
    parser.add_argument("--include", action="append", default=[])
    parser.add_argument("--skip", action="append", default=[])
    parser.add_argument("--inspect-config", action="store_true")
    if task == "text-to-video":
        parser.add_argument("--num-frames", type=int, default=49)
    if task == "image-to-image":
        parser.add_argument("--dataset", required=True)
        parser.add_argument("--dataset-config")
        parser.add_argument("--dataset-split", default="validation")
        parser.add_argument("--image-column")
        parser.add_argument("--prompt-column")
    return parser


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.rstrip("/").split("/")[-1].lower()).strip("-")


def _records(args, task: Task) -> list[dict]:
    if task != "image-to-image":
        return standard_prompt_records(args.num_samples, args.prompt_file)
    import datasets

    data = datasets.load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    columns = set(data.column_names)
    image_col = args.image_column or next((x for x in ("image", "source_image", "source", "input_image") if x in columns), None)
    prompt_col = args.prompt_column or next((x for x in ("prompt", "edit_instruction", "instruction", "text") if x in columns), None)
    if image_col is None or prompt_col is None:
        raise ValueError(f"Could not discover image/prompt columns from {sorted(columns)}")
    limit = len(data) if args.num_samples < 0 else min(args.num_samples, len(data))
    return [{"filename": str(i), "image": data[i][image_col], "prompt": str(data[i][prompt_col]), "seed": i} for i in range(limit)]


def _batched_records(records: list[dict], batch_size: int) -> list[dict]:
    if not records or "image" not in records[0]:
        return batched_samples(records, batch_size)
    batches = []
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        batches.append({key: values[0] if len(values) == 1 else values for key in rows[0] for values in [[row[key] for row in rows]]})
    return batches


def _forward_fn(pipe, args, task: Task):
    signature = inspect.signature(pipe.__call__)
    accepted = set(signature.parameters)

    def forward(sample: dict):
        kwargs = {"prompt": sample["prompt"], "num_inference_steps": args.steps, "guidance_scale": args.guidance_scale}
        optional = {"height": args.height, "width": args.width, "generator": make_generator(sample.get("seed", 0), device=args.device)}
        if task == "text-to-video":
            optional["num_frames"] = args.num_frames
        if task == "image-to-image":
            optional["image"] = sample["image"]
        kwargs.update({key: value for key, value in optional.items() if key in accepted})
        return pipe(**kwargs)

    return forward


def run(task: Task) -> None:
    args = build_parser(task).parse_args()
    pipe = load_auto_pipeline(args.model_id, device=args.device, pipeline_offload=args.pipeline_offload)
    _, model = discover_denoiser(pipe)
    scan = scan_linear_targets(model, precision=args.precision, rank=args.rank, include=args.include, skip=args.skip)
    print(scan.format_text())
    if args.inspect_config:
        print(inspect_target_config(model, scan.target_config).format_text())
        return
    if not scan.svdq_targets and not scan.awq_targets:
        raise ValueError("No compatible linear targets were discovered")
    output = Path(args.output or f"outputs/checkpoints/svdq-{args.precision}_r{args.rank}-{_slug(args.model_id)}.safetensors")
    cache_dir = Path(args.cache_dir or f"outputs/calibration/{_slug(args.model_id)}")
    artifact_cache = None if args.cache_mode == "disabled" else QuantizationCacheSpec(cache_dir / args.precision / "artifacts", args.cache_mode)
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
        CalibrationSpec(samples=_batched_records(_records(args, task), args.batch_size), num_samples=args.num_samples,
                        cache_num_samples=args.num_samples if args.cache_num_samples is None else args.cache_num_samples,
                        batch_size=args.batch_size, cache_dir=cache_dir / args.precision / "inputs", cache_mode=args.cache_mode,
                        forward_fn=_forward_fn(pipe, args, task), max_rows_per_target=4096, artifact_cache=artifact_cache,
                        scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
                        sample_batch_size=args.sample_batch_size or args.batch_size),
        ExportSpec(output=output), LoggingConfig(name=output.stem),
    )
