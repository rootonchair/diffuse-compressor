"""Quantize a general Hugging Face image-to-image Diffusers pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.quantize_hf import run


if __name__ == "__main__":
    run("image-to-image")
