"""Quantize Sana 1.6B with upstream DeepCompressor-style SVDQuant config."""

from upstream_diffusion_svdquant import run_model_cli


if __name__ == "__main__":
    run_model_cli("sana-1.6b")
