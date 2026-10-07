#!/usr/bin/env bash
# Quantize Qwen-Image 2.1 with diffuse-compressor.
#
# Usage:
#   scripts/run_qwen_image_2_1.sh                    # INT4, no CFG
#   PRECISION=nvfp4 scripts/run_qwen_image_2_1.sh
#   CFG=4.0 scripts/run_qwen_image_2_1.sh            # also calibrate negative-prompt passes
#   GPTQ=1 scripts/run_qwen_image_2_1.sh             # GPTQ residual rounding
#   HF_HUB_CACHE=/path/to/hub scripts/run_qwen_image_2_1.sh   # use a non-default local weights cache
#
# Runs in the background with nohup; follow progress with:
#   tail -f "$LOG"   (path printed at start)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PRECISION=${PRECISION:-int4}          # int4 | nvfp4
RANK=${RANK:-32}
NUM_SAMPLES=${NUM_SAMPLES:-8}         # prompts; each prompt runs a full STEPS-step generation
STEPS=${STEPS:-40}
# Calibration replays this many recorded transformer calls (not prompts).
# NUM_SAMPLES x STEPS calls are recorded (x2 with CFG); the default of NUM_SAMPLES would replay only 128 of them.
CACHE_NUM_SAMPLES=${CACHE_NUM_SAMPLES:-512}
CFG=${CFG:-1.0}                       # >1 enables CFG calibration with NEGATIVE_PROMPT
NEGATIVE_PROMPT=${NEGATIVE_PROMPT:-" "}
CACHE_MODE=${CACHE_MODE:-reuse}       # reuse | refresh | disabled
MAX_ROWS=${MAX_ROWS:-32768}           # activation rows kept per target; keep above the widest input (12288)
GPTQ=${GPTQ:-0}                       # 1 enables GPTQ residual rounding
MIN_FREE_GB=${MIN_FREE_GB:-30}        # input cache (~8 GB) + artifact cache (~12 GB) + checkpoint (~4 GB)

SUFFIX=""
if awk "BEGIN{exit !($CFG > 1)}"; then SUFFIX="-cfg${CFG}"; fi
METHOD=svdq; [ "$GPTQ" = 1 ] && METHOD=svdq-gptq
NAME="${METHOD}-${PRECISION}_r${RANK}-qwen-image-2-1${SUFFIX}"
OUT_DIR=$REPO/outputs/checkpoints
CACHE_DIR=$REPO/outputs/calibration/qwen-image-2-1${SUFFIX}
LOG=$REPO/outputs/logs/${NAME}.$(date +%Y%m%d-%H%M%S).log
mkdir -p "$OUT_DIR" "$CACHE_DIR" "$(dirname "$LOG")"

# Caches and checkpoints are large; fail early instead of dying at the final save.
free_gb=$(df --output=avail -BG "$REPO" | tail -1 | tr -dc '0-9')
if [ "$CACHE_MODE" != "disabled" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    echo "Only ${free_gb} GB free on $(df --output=target "$REPO" | tail -1); need ${MIN_FREE_GB} GB." >&2
    echo "Free space (e.g. rm -rf $REPO/outputs/qi21_smoke) or rerun with CACHE_MODE=disabled." >&2
    exit 1
fi

cd "$REPO"
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
[ -n "${HF_HUB_CACHE:-}" ] && export HF_HUB_CACHE
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

ARGS=(
    --precision "$PRECISION"
    --rank "$RANK"
    --num-samples "$NUM_SAMPLES"
    --cache-num-samples "$CACHE_NUM_SAMPLES"
    --steps "$STEPS"
    --height 1024 --width 1024
    --device cuda
    --cache-mode "$CACHE_MODE"
    --max-rows-per-target "$MAX_ROWS"
    --cache-dir "$CACHE_DIR"
    --output "$OUT_DIR/${NAME}.safetensors"
)
if [ "$GPTQ" = 1 ]; then ARGS+=(--gptq); fi
if [ -n "$SUFFIX" ]; then
    ARGS+=(--true-cfg-scale "$CFG" --negative-prompt "$NEGATIVE_PROMPT")
fi

echo "Log:        $LOG"
echo "Checkpoint: $OUT_DIR/${NAME}.safetensors"
nohup python3 examples/text_to_image/quantize_qwen_image_2_1.py "${ARGS[@]}" > "$LOG" 2>&1 &
echo "PID:        $!"
