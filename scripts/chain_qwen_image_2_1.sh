#!/usr/bin/env bash
# Quantize Qwen-Image 2.1 in INT4 then NVFP4, then compare every available checkpoint against BF16.
#
# Usage (settings pass through to run_qwen_image_2_1.sh):
#   NUM_SAMPLES=32 RANK=64 CACHE_NUM_SAMPLES=1280 CACHE_MODE=refresh nohup scripts/chain_qwen_image_2_1.sh &
#
# The comparison includes BF16 plus every svdq[-gptq]-{int4,nvfp4}_r*-qwen-image-2-1.safetensors in
# outputs/checkpoints, so earlier ranks appear next to the new ones.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
CK=outputs/checkpoints
RANK=${RANK:-32}
export RANK
METHOD=svdq; TAG=""; [ "${GPTQ:-0}" = 1 ] && { METHOD=svdq-gptq; TAG=gptq-; }

finished_ok() {  # <log> <checkpoint>
    [ -s "$2" ] && grep -q "Exporting checkpoint" "$1" && ! grep -q "Traceback" "$1"
}

for precision in ${PRECISIONS:-int4 nvfp4}; do
    echo "[$(date +%T)] starting $precision (rank $RANK, ${NUM_SAMPLES:-8} prompts)"
    out=$(PRECISION=$precision scripts/run_qwen_image_2_1.sh) || { echo "$out"; exit 1; }
    echo "$out"
    pid=$(sed -n 's/^PID: *//p' <<<"$out"); log=$(sed -n 's/^Log: *//p' <<<"$out")
    while kill -0 "$pid" 2>/dev/null; do sleep 30; done
    if ! finished_ok "$log" "$CK/${METHOD}-${precision}_r${RANK}-qwen-image-2-1.safetensors"; then
        echo "[$(date +%T)] $precision failed; stopping"; grep -A5 Traceback "$log" | tail -20; exit 1
    fi
    echo "[$(date +%T)] $precision done"
done

args=()
for checkpoint in $(ls $CK/svdq-*int4_r*-qwen-image-2-1.safetensors $CK/svdq-*nvfp4_r*-qwen-image-2-1.safetensors 2>/dev/null); do
    name=$(basename "$checkpoint" .safetensors | sed 's/^svdq-//; s/-qwen-image-2-1$//')
    args+=(--checkpoint "$name=$checkpoint")
done
echo "[$(date +%T)] running comparison: ${args[*]}"
# The comparison needs nunchaku_lite and the Hugging Face `kernels` package importable; point
# RUNTIME_PYTHONPATH at them if they are not installed.
PYTHONPATH=${RUNTIME_PYTHONPATH:+$RUNTIME_PYTHONPATH:}$REPO/src \
python3 scripts/compare_qwen_image_2_1.py "${args[@]}" --output-dir outputs/compare/qwen-image-2-1-${TAG}r${RANK}
echo "[$(date +%T)] chain finished (exit $?)"
