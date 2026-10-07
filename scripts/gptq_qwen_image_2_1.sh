#!/usr/bin/env bash
# Quantize Qwen-Image 2.1 r64 with GPTQ at each run's row limit, reusing the recorded calibration
# calls (32 prompts x 40 steps = 1280). Waits for an idle GPU first.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$REPO"
CK=outputs/checkpoints
export PYTHONPATH=$REPO/src
[ -n "${HF_HUB_CACHE:-}" ] && export HF_HUB_CACHE
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while gpu_busy || { sleep 60; gpu_busy; }; do sleep 60; done
echo "[$(date +%T)] GPU idle; starting GPTQ runs"
run() {  # <precision> <rows> <tag>
    local prec=$1 rows=$2 tag=$3
    local dir=outputs/calibration/qwen-image-2-1-gptq-$prec$tag out=$CK/svdq-gptq-${prec}_r64-qwen-image-2-1$tag.safetensors
    mkdir -p "$dir/$prec" && ln -sfn "$REPO/outputs/calibration/qwen-image-2-1/$prec/inputs" "$dir/$prec/inputs"
    local log=outputs/logs/gptq-$prec$tag.log
    python3 examples/text_to_image/quantize_qwen_image_2_1.py --precision "$prec" --rank 64 --gptq \
        --num-samples 32 --cache-num-samples 1280 --steps 40 --height 1024 --width 1024 --device cuda \
        --cache-mode reuse --cache-dir "$dir" --max-rows-per-target "$rows" --output "$out" > "$log" 2>&1
    local status=$?
    echo "[$(date +%T)] gptq $prec rows=$rows exit=$status; chained: $(grep -m1 -o 'Chained [0-9]*/[0-9]* calls' "$log");" \
         "gptq targets: $(grep -c 'Applying GPTQ' "$log"); rank-deficient warnings: $(grep -c 'rank-deficient' "$log")"
    [ $status -eq 0 ] && [ -s "$out" ] && rm -rf "$dir"
}
# RUNS selects a subset, e.g. RUNS="nvfp4:32768:" (precision:rows:tag); default is all three.
for spec in ${RUNS:-int4:4096:-rows4k nvfp4:4096:-rows4k nvfp4:32768:}; do
    IFS=: read -r prec rows tag <<<"$spec"
    run "$prec" "$rows" "$tag"
done
echo "[$(date +%T)] GPTQ runs finished"
