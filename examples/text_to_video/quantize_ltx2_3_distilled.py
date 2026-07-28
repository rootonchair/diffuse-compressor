"""Quantize LTX-2.3 Distilled Diffusers with the shared LTX SVDQuant config."""

from __future__ import annotations

from quantize_ltx2_3 import run_ltx2_3_cli


def run_model_cli() -> None:
    """Load LTX-2.3 Distilled Diffusers and run quantization."""

    from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

    run_ltx2_3_cli(
        model_id="dg845/LTX-2.3-Distilled-Diffusers",
        output_template=(
            "outputs/checkpoints/svdq-{precision}_r32-ltx2.3-distilled.safetensors"
        ),
        default_cache_dir="outputs/calibration/ltx2.3-distilled",
        steps=8,
        guidance_scale=1.0,
        batch_size=1,
        height=512,
        width=768,
        num_frames=121,
        frame_rate=24.0,
        sigmas=DISTILLED_SIGMA_VALUES,
    )


if __name__ == "__main__":
    run_model_cli()
