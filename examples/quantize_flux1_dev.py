"""Quantize FLUX.1 Dev with upstream DeepCompressor-style SVDQuant config."""

from upstream_diffusion_svdquant import run_model_cli


if __name__ == "__main__":
    run_model_cli("flux.1-dev")
