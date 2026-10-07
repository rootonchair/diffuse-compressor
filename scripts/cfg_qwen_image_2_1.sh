#!/usr/bin/env bash
# Quantize Qwen-Image 2.1 r64 with CFG-calibrated inputs (true_cfg_scale 4.0, negative prompt " "):
# INT4 at 4096 rows, NVFP4 at 32768 rows. Inputs are recorded once (2560 calls) and shared.
# Usage: scripts/cfg_qwen_image_2_1.sh [pid-to-wait-for]
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
WAIT_PID=${1:-}
CK=outputs/checkpoints
CACHE=outputs/calibration/qwen-image-2-1-cfg4
export PYTHONPATH=$REPO/src
[ -n "${HF_HUB_CACHE:-}" ] && export HF_HUB_CACHE
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
[ -n "$WAIT_PID" ] && while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
while gpu_busy || { sleep 60; gpu_busy; }; do sleep 60; done
echo "[$(date +%T)] GPU idle; starting CFG quantization"
run() {  # <precision> <rows> <tag>
    local prec=$1 rows=$2 tag=$3 out=$CK/svdq-${1}_r64-qwen-image-2-1-cfg4$3.safetensors log=outputs/logs/cfg4-$1$3.log
    python3 examples/text_to_image/quantize_qwen_image_2_1.py --precision "$prec" --rank 64 \
        --true-cfg-scale 4.0 --negative-prompt " " \
        --num-samples 32 --cache-num-samples 1280 --steps 40 --height 1024 --width 1024 --device cuda \
        --cache-mode reuse --cache-dir "$CACHE" --max-rows-per-target "$rows" --output "$out" > "$log" 2>&1
    local status=$?
    echo "[$(date +%T)] cfg4 $prec rows=$rows exit=$status; $(grep -m1 -o 'Saved [0-9]* calibration input cache records' "$log");" \
         "$(grep -m1 -o 'Chained [0-9]*/[0-9]* calls' "$log")"
    [ $status -eq 0 ] && [ -s "$out" ] && rm -rf "$CACHE/$prec/artifacts"
}
run int4 4096 -rows4k || exit 1
# The recorded calibration inputs come from the BF16 pipeline, so NVFP4 reuses INT4's.
mkdir -p "$CACHE/nvfp4" && ln -sfn "$REPO/$CACHE/int4/inputs" "$CACHE/nvfp4/inputs"
run nvfp4 32768 ""
echo "[$(date +%T)] CFG quantization finished"
