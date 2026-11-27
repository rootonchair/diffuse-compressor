"""Quantize any Diffusers text-to-image denoiser with data-free SVDQuant (no calibration data).

Only the transformer/unet weights are loaded: no text encoders, VAE or
pipeline, and the model is never run. Smoothing uses weight-only scales
``s = 1 / absmax(W[:, j]) ** beta`` and the low-rank branch is an SVD of the
smoothed weight. Layers are found by the generic scanner from
``calibrated/quantize_hf.py``; models with a dedicated script in this folder
export their model-specific layout instead.

    python examples/text_to_image/data_free/quantize_hf.py MODEL_ID --precision nvfp4
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch import nn

from diffuse_compressor import CalibrationSpec, ExportSpec, LoggingConfig, inspect_target_config, quantize_and_export
from examples.text_to_image.calibrated.quantize_hf import scan_linear_targets
from examples.text_to_image.utils import (
    check_data_free_args,
    data_free_arg_parser,
    data_free_artifact_cache,
    data_free_output,
    data_free_spec,
)


def slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model_id.rstrip("/").split("/")[-1].lower()).strip("-")


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


def run(argv: list[str] | None = None) -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = data_free_arg_parser()
    parser.add_argument("--skip", action="append", default=[], help="Glob of module names to leave dense.")
    args = parser.parse_args(argv)
    check_data_free_args(parser, args)
    spec = data_free_spec(args)
    model = load_denoiser(
        args.model_id, device="cpu" if args.offload_model else args.device, dtype=getattr(torch, args.dtype)
    )
    scan = scan_linear_targets(model, precision=args.precision, rank=args.rank, skip=args.skip)
    print(scan.format_text())
    if args.inspect_config:
        print(inspect_target_config(model, scan.target_config).format_text())
        return
    if not scan.svdq_targets and not scan.awq_targets:
        raise ValueError("No compatible linear targets were discovered")
    output = data_free_output(args, slug(args.model_id))
    artifact_cache = data_free_artifact_cache(args, f"outputs/calibration/{slug(args.model_id)}")
    quantize_and_export(
        model=model,
        spec=spec,
        target_config=scan.target_config,
        calibration=None if artifact_cache is None else CalibrationSpec(artifact_cache=artifact_cache),
        export=ExportSpec(output=output),
        logging=LoggingConfig(enabled=not args.no_run_log, log_dir=args.log_dir, name=output.stem),
    )


if __name__ == "__main__":
    run()
