"""Quantize PixArt Sigma with upstream DeepCompressor-style SVDQuant config."""

from upstream_diffusion_svdquant import run_model_cli


if __name__ == "__main__":
    run_model_cli("pixart-sigma")
