"""Quantize FLUX.2 Klein 9B with data-free SVDQuant (no calibration data).

Loads only the transformer and exports the same targets as
``quantize_flux2_klein_9b.py``. See ``data_free_arg_parser`` for the options.
For calibration on synthetic denoising trajectories instead, use
``quantize_flux2_klein_9b.py --synthetic-calibration``.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from diffuse_compressor import CalibrationSpec, ExportSpec, LoggingConfig, inspect_target_config, quantize_and_export

from quantize_flux2_klein_9b import flux2_klein_target_config
from utils import (
    check_data_free_args,
    data_free_arg_parser,
    data_free_artifact_cache,
    data_free_output,
    data_free_spec,
)

MODEL_ID = "black-forest-labs/FLUX.2-klein-9B"


def load_flux2_transformer(model_id: str, *, device: str, dtype: torch.dtype) -> nn.Module:
    """Load only the FLUX.2 transformer."""

    from diffusers import Flux2Transformer2DModel

    return Flux2Transformer2DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=dtype).to(device)


def run_model_cli(argv: list[str] | None = None) -> None:
    """Load the FLUX.2 Klein 9B transformer and run data-free quantization."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = data_free_arg_parser(MODEL_ID)
    args = parser.parse_args(argv)
    check_data_free_args(parser, args)
    spec = data_free_spec(args)
    target_config = flux2_klein_target_config(args.precision, single_qkv_features=12288, single_attn_features=4096)
    model = load_flux2_transformer(
        args.model_id, device="cpu" if args.offload_model else args.device, dtype=getattr(torch, args.dtype)
    )
    if args.inspect_config:
        print(inspect_target_config(model, target_config).format_text())
        return
    output = data_free_output(args, "flux2-klein-9b")
    artifact_cache = data_free_artifact_cache(args, "outputs/calibration/flux2-klein-9b")
    quantize_and_export(
        model=model,
        spec=spec,
        target_config=target_config,
        calibration=None if artifact_cache is None else CalibrationSpec(artifact_cache=artifact_cache),
        export=ExportSpec(output=output),
        logging=LoggingConfig(enabled=not args.no_run_log, log_dir=args.log_dir, name=output.stem),
    )


if __name__ == "__main__":
    run_model_cli()
