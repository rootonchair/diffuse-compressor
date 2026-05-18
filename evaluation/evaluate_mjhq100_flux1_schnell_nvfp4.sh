#!/usr/bin/env bash
set -euo pipefail

cd /mnt/disks/workspace/research/diffuse_compressor

CHECKPOINT="/mnt/disks/workspace/research/diffuse_compressor/outputs/checkpoints/svdq-nvfp4_r32-flux.1-schnell.safetensors"
BASE_DIR="outputs/eval/flux.1-schnell/mjhq-100"
ORIG_DIR="$BASE_DIR/original"
QUANT_DIR="$BASE_DIR/nvfp4"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BATCH_SIZE="${BATCH_SIZE:-1}"
METRICS="${METRICS:-clip_iqa clip_score image_reward fid}"

test -f "$CHECKPOINT"

env PYTHONPATH=src:. python -m evaluation.evaluate_image_generation \
  --mode original \
  --model-key flux.1-schnell \
  --benchmark MJHQ \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --metrics $METRICS \
  --output-dir "$ORIG_DIR"

env PYTHONPATH=src:. python -m evaluation.evaluate_image_generation \
  --mode quantized \
  --model-key flux.1-schnell \
  --runtime torch-dequant \
  --checkpoint "$CHECKPOINT" \
  --precision nvfp4 \
  --benchmark MJHQ \
  --ref-root "$ORIG_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --metrics $METRICS \
  --output-dir "$QUANT_DIR"
