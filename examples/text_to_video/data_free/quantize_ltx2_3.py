"""Quantize LTX-2.3 Diffusers with data-free SVDQuant (no calibration data).

Loads only the transformer (no text encoder, VAEs or vocoder) and exports the
same targets as ``calibrated/quantize_ltx2_3.py``, including its nunchaku-lite manifest
check. See ``data_free_arg_parser`` for the options.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from diffuse_compressor import CalibrationSpec, ExportSpec, LoggingConfig, inspect_target_config, quantize_and_export

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.text_to_video.calibrated.quantize_ltx2_3 import ltx2_3_target_config, validate_ltx2_3_nunchaku_lite_manifest_targets
from examples.text_to_video.utils import (
    check_data_free_args,
    data_free_arg_parser,
    data_free_artifact_cache,
    data_free_output,
    data_free_spec,
)


def load_ltx2_transformer(model_id: str, *, device: str, dtype: torch.dtype) -> nn.Module:
    """Load only the LTX2 transformer."""

    from diffusers import LTX2VideoTransformer3DModel

    return LTX2VideoTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype).to(device)


def run_model_cli(argv: list[str] | None = None) -> None:
    """Load the LTX-2.3 transformer and run data-free quantization."""

    run_ltx2_3_data_free_cli(
        model_id="dg845/LTX-2.3-Diffusers",
        model_name="ltx2.3",
        default_cache_dir="outputs/calibration/ltx2.3",
        argv=argv,
    )


def run_ltx2_3_data_free_cli(
    *, model_id: str, model_name: str, default_cache_dir: str, argv: list[str] | None = None
) -> None:
    """Run data-free quantization for one LTX-2.3-family transformer."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = data_free_arg_parser(model_id)
    parser.add_argument("--fuse-qkv", action="store_true", help="Fuse q/k/v projections into one target.")
    args = parser.parse_args(argv)
    check_data_free_args(parser, args)
    spec = data_free_spec(args)
    target_config = ltx2_3_target_config(args.precision, fuse_qkv=args.fuse_qkv)
    transformer = load_ltx2_transformer(
        args.model_id, device="cpu" if args.offload_model else args.device, dtype=getattr(torch, args.dtype)
    )
    if args.inspect_config:
        print(inspect_target_config(transformer, target_config).format_text())
        return
    validate_ltx2_3_nunchaku_lite_manifest_targets(transformer, target_config)
    output = data_free_output(args, model_name)
    artifact_cache = data_free_artifact_cache(args, default_cache_dir)
    quantize_and_export(
        model=transformer,
        spec=spec,
        target_config=target_config,
        calibration=None if artifact_cache is None else CalibrationSpec(artifact_cache=artifact_cache),
        export=ExportSpec(output=output),
        logging=LoggingConfig(enabled=not args.no_run_log, log_dir=args.log_dir, name=output.stem),
    )


if __name__ == "__main__":
    run_model_cli()
