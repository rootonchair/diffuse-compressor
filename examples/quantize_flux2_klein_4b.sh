#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="${PYTHONPATH:-}:src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

MODEL_ID="${MODEL_ID:-black-forest-labs/FLUX.2-klein-4B}"
OUTPUT="${OUTPUT:-outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
STEPS="${STEPS:-4}"

mkdir -p "$(dirname "$OUTPUT")" outputs/calibration/flux2-klein-4b

python examples/flux2_klein_4b_svdquant.py \
  --model-id "$MODEL_ID" \
  --output "$OUTPUT" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --steps "$STEPS"
