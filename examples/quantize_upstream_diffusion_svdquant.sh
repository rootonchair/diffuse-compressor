#!/usr/bin/env bash
set -euo pipefail

precision="${1:-int4}"
samples="${SAMPLES:-128}"

python examples/quantize_flux1_schnell.py --precision "${precision}" --num-samples "${samples}"
python examples/quantize_flux1_dev.py --precision "${precision}" --num-samples "${samples}"
python examples/quantize_flux2_klein_4b.py --precision "${precision}" --num-samples "${samples}"
python examples/quantize_flux2_klein_9b.py --precision "${precision}" --num-samples "${samples}"
python examples/quantize_pixart_sigma.py --precision "${precision}" --num-samples "${samples}"
python examples/quantize_sana_1_6b.py --precision "${precision}" --num-samples "${samples}"
