"""Quantize LTX-2.3 Distilled Diffusers with data-free SVDQuant (no calibration data)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.text_to_video.data_free.quantize_ltx2_3 import run_ltx2_3_data_free_cli


def run_model_cli(argv: list[str] | None = None) -> None:
    """Load the LTX-2.3 Distilled transformer and run data-free quantization."""

    run_ltx2_3_data_free_cli(
        model_id="dg845/LTX-2.3-Distilled-Diffusers",
        model_name="ltx2.3-distilled",
        default_cache_dir="outputs/calibration/ltx2.3-distilled",
        argv=argv,
    )


if __name__ == "__main__":
    run_model_cli()
