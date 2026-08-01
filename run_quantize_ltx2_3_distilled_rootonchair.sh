#!/bin/bash
set -e

source /venv/main/bin/activate
cd "$(dirname "${BASH_SOURCE[0]}")"

# rm -rf outputs/calibration/ltx2.3-distilled

python -u examples/text_to_video/quantize_ltx2_3_distilled.py \
  --model-id rootonchair/LTX-2.3-Distilled-v1.1-Diffusers \
  --precision int4 \
  --text-encoder-quant 8bit \
  --pipeline-offload sequential \
  --offload-model \
  --compute-device cuda \
  --num-samples 32 \
  --cache-num-samples 128 \
  --sample-batch-size 4 \
  --prompt-file examples/prompts/ltx2_audio_diverse.yaml \
  --gptq
