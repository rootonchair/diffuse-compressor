"""Quantize a Diffusers denoiser with data-free SVDQuant (no calibration data).

Only the transformer/unet weights are loaded: no text encoders, VAE or
pipeline, and the model is never run. Smoothing uses weight-only scales
``s = 1 / absmax(W[:, j]) ** beta`` and the low-rank branch is an SVD of the
smoothed weight. Works the same for image and video models.

    python examples/quantize_data_free.py black-forest-labs/FLUX.2-klein-4B \
        --target-config flux2-klein --precision nvfp4
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch import nn

from diffuse_compressor import (
    CalibrationSpec,
    ExportSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    TargetConfig,
    inspect_target_config,
    quantize_and_export,
)
from examples.text_to_image.utils import svdquant_spec

TARGET_CONFIGS = ("auto", "flux2-klein", "ltx2.3")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_id", help="Hugging Face model id or local Diffusers model/pipeline directory")
    parser.add_argument(
        "--target-config",
        choices=TARGET_CONFIGS,
        default="auto",
        help="Which layers to quantize: 'auto' scans the denoiser's linear layers generically; "
        "the others use the hand-written target lists from the calibrated model scripts.",
    )
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default="int4")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument(
        "--smooth-exponent",
        type=float,
        default=None,
        help="Weight-span smoothing exponent in [0, 1] (0 disables smoothing, 1 fully flattens "
        "per-channel weight spans). Defaults to 0 with --hadamard, else 0.5.",
    )
    parser.add_argument(
        "--hadamard",
        action="store_true",
        help="Enable online 64-block Hadamard rotation of the residual input basis (int4 only).",
    )
    parser.add_argument(
        "--svd-backend",
        choices=("full", "svd_lowrank"),
        default="svd_lowrank",
        help="svd_lowrank is a randomized approximation (runs differ slightly); full is exact.",
    )
    parser.add_argument("--svd-lowrank-oversample", type=int, default=10)
    parser.add_argument("--svd-lowrank-niter", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--offload-model",
        action="store_true",
        help="Keep the denoiser on CPU and move each target to the compute device only while it "
        "is quantized, for models that don't fit on the GPU.",
    )
    parser.add_argument("--compute-device", help="Device for per-target quantization (default: --device).")
    parser.add_argument("--skip", action="append", default=[], help="With --target-config auto: glob of module names to leave dense.")
    parser.add_argument("--fuse-qkv", action="store_true", help="With --target-config ltx2.3: fuse q/k/v projections.")
    parser.add_argument("--output", help="Checkpoint path (default: outputs/checkpoints/svdq-<precision>_r<rank>-<model>-datafree[-hadamard].safetensors)")
    parser.add_argument("--cache-dir", help="Artifact cache root (default: outputs/calibration/<model>).")
    parser.add_argument("--cache-mode", choices=("reuse", "refresh", "disabled"), default="reuse")
    parser.add_argument("--inspect-config", action="store_true", help="Print the matched targets and exit.")
    parser.add_argument("--log-dir", default="outputs/logs", help="Directory for quantization process and target logs.")
    parser.add_argument("--no-run-log", action="store_true")
    return parser


def slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.rstrip("/").split("/")[-1].lower()).strip("-")


def default_output(model_id: str, *, precision: str, rank: int, hadamard: bool) -> Path:
    suffix = "-datafree" + ("-hadamard" if hadamard else "")
    return Path(f"outputs/checkpoints/svdq-{precision}_r{rank}-{slug(model_id)}{suffix}.safetensors")


def artifact_cache_dir(cache_dir: str | None, model_id: str, precision: str) -> Path:
    # Kept apart from calibrated runs' ``artifacts``: cache files are named per
    # target, so sharing a folder would overwrite the other run's resume state.
    return Path(cache_dir or f"outputs/calibration/{slug(model_id)}") / precision / "artifacts-datafree"


def load_denoiser(model_id: str, *, device: str, dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """Load only the transformer/unet component."""

    from diffusers import AutoModel

    errors: list[str] = []
    for subfolder in ("transformer", "unet"):
        try:
            model = AutoModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=dtype)
        except (OSError, ValueError) as exc:
            errors.append(f"{subfolder}: {exc}")
            continue
        return model.to(device)
    raise ValueError(f"Could not load a transformer or unet component from {model_id!r}:\n" + "\n".join(errors))


def _load_example_module(relative: str, utils_module: str) -> ModuleType:
    """Import a calibrated model script whose ``from utils import ...`` must see ``utils_module``."""

    import importlib

    utils = importlib.import_module(utils_module)
    path = _REPO_ROOT / "examples" / relative
    previous = sys.modules.get("utils")
    sys.modules["utils"] = utils
    try:
        spec = importlib.util.spec_from_file_location(f"_data_free_{path.stem}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous
    return module


def flux2_klein_targets(model: nn.Module, precision: str) -> TargetConfig:
    klein = _load_example_module("text_to_image/quantize_flux2_klein_4b.py", "examples.text_to_image.utils")
    inner = model.config.num_attention_heads * model.config.attention_head_dim
    return klein.flux2_klein_target_config(precision, single_qkv_features=3 * inner, single_attn_features=inner)


def ltx2_3_targets(model: nn.Module, precision: str, *, fuse_qkv: bool) -> TargetConfig:
    ltx = _load_example_module("text_to_video/quantize_ltx2_3.py", "examples.text_to_video.utils")
    target_config = ltx.ltx2_3_target_config(precision, fuse_qkv=fuse_qkv)
    ltx.validate_ltx2_3_nunchaku_lite_manifest_targets(model, target_config)
    return target_config


def resolve_target_config(model: nn.Module, args: argparse.Namespace) -> TargetConfig:
    if args.target_config == "flux2-klein":
        return flux2_klein_targets(model, args.precision)
    if args.target_config == "ltx2.3":
        return ltx2_3_targets(model, args.precision, fuse_qkv=args.fuse_qkv)
    from examples.text_to_image.quantize_hf import scan_linear_targets

    scan = scan_linear_targets(model, precision=args.precision, rank=args.rank, skip=args.skip)
    print(scan.format_text())
    if not scan.svdq_targets and not scan.awq_targets:
        raise ValueError("No compatible linear targets were discovered")
    return scan.target_config


def run(argv: list[str] | None = None) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.hadamard and args.precision != "int4":
        parser.error("--hadamard supports --precision int4 only")
    if args.skip and args.target_config != "auto":
        parser.error("--skip applies to --target-config auto only")
    if args.fuse_qkv and args.target_config != "ltx2.3":
        parser.error("--fuse-qkv applies to --target-config ltx2.3 only")
    # Validate the spec before loading weights.
    spec = replace(
        svdquant_spec(
            args.precision,
            data_free=True,
            smooth_exponent=args.smooth_exponent,
            hadamard=args.hadamard,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            compute_device=args.compute_device or (args.device if args.offload_model else None),
            offload_model=args.offload_model,
        ),
        rank=args.rank,
    )
    model = load_denoiser(args.model_id, device="cpu" if args.offload_model else args.device)
    target_config = resolve_target_config(model, args)
    if args.inspect_config:
        print(inspect_target_config(model, target_config).format_text())
        return
    output = Path(args.output) if args.output else default_output(
        args.model_id, precision=args.precision, rank=args.rank, hadamard=args.hadamard
    )
    calibration = None
    if args.cache_mode != "disabled":
        calibration = CalibrationSpec(
            artifact_cache=QuantizationCacheSpec(
                cache_dir=artifact_cache_dir(args.cache_dir, args.model_id, args.precision),
                cache_mode=args.cache_mode,
            )
        )
    quantize_and_export(
        model=model,
        spec=spec,
        target_config=target_config,
        calibration=calibration,
        export=ExportSpec(output=output),
        logging=LoggingConfig(enabled=not args.no_run_log, log_dir=args.log_dir, name=output.stem),
    )


if __name__ == "__main__":
    run()
