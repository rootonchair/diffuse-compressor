#!/bin/bash
set -e

source /venv/main/bin/activate
cd "$(dirname "${BASH_SOURCE[0]}")"

python -u examples/text_to_video/quantize_hf.py \
  Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --precision int4 \
  --height 480 \
  --width 832 \
  --num-frames 81 \
  --steps 50 \
  --guidance-scale 5.0 \
  --compute-device cuda \
  --num-samples 4 \
  --cache-num-samples 128 \
  --sample-batch-size 8 \
  --prompt-file examples/prompts/qdiff.yaml \
  --gptq
