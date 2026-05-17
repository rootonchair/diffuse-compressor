#!/usr/bin/env bash
set -euo pipefail

checkpoint="${CHECKPOINT:-outputs/checkpoints/svdq-nvfp4_r32-flux.1-schnell.safetensors}"
output_dir="${OUTPUT_DIR:-outputs/eval/flux.1-schnell/nvfp4-torch-dequant}"
runtime="${RUNTIME:-torch-dequant}"
samples="${SAMPLES:-128}"
skip_bf16="${SKIP_BF16:-true}"

args=(
  examples/evaluate_upstream_diffusion.py
  --model-key flux.1-schnell
  --precision nvfp4
  --checkpoint "${checkpoint}"
  --runtime "${runtime}"
  --output-dir "${output_dir}"
  --num-samples "${samples}"
)

if [[ "${skip_bf16}" == "true" ]]; then
  args+=(--skip-bf16)
fi

PYTHONPATH="${PYTHONPATH:-src:.}" \
python "${args[@]}"
