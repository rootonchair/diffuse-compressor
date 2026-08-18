---
name: nunchaku-checkpoint-converter
description: Convert original Nunchaku checkpoints into Diffusers-loadable Nunchaku Lite model repos (like lite-infer/*-nunchaku-lite-*-bnb4-text-encoder), model-agnostically. Use when packaging prequantized Nunchaku checkpoints for plain `from_pretrained` loading, splitting fused modules, merging split modules into a single Diffusers linear, undoing fused-GELU/unsigned-INT4 bias shifts, padding low-rank ranks, quantizing companion text encoders with BitsAndBytes, writing the output repo README, benchmarking against the native baseline, or debugging Diffusers loads that succeed but generate bad images because of packed-tensor layout mistakes.
---

# Nunchaku Checkpoint Converter

Turn an original Nunchaku checkpoint into a Diffusers pipeline directory that
loads with plain `DiffusionPipeline.from_pretrained(...)` via the Nunchaku Lite
quantizer — no runtime graph patches — then prove it is numerically usable and
package it with a README and benchmark.

The output must look like a `lite-infer/*` model repo, e.g.
`outputs/converted/qwen-image-edit-2509-lightning-4steps-nunchaku-lite-int4_r32-bnb4-text-encoder`:
pipeline components + quantized transformer + `README.md` with a benchmark table
and output comparison image.

## The one rule that prevents most bugs

Nunchaku tensors are stored in **packed, tile-permuted layouts**. Many have
ordinary-looking shapes (`proj_up` packed is the same shape as logical), so
shape checks and strict state-dict loads prove nothing about element order.

**Never slice, concatenate, block-diagonalize, or zero-pad a packed tensor in
container space.** Always: `unpack → edit in logical layout → repack` with
`diffuse_compressor.backends.nunchaku.packing.NunchakuWeightPacker`. A violation
loads cleanly and produces plausible-but-noisy images — the hardest failure
mode to notice. Validate with distinctive-valued tensors; constant fills
(`torch.ones`) are permutation-invariant and cannot catch packing bugs.

## Core Workflow

1. **Survey both sides.**
   - Dump source checkpoint keys/shapes/dtypes with `safetensors.safe_open`
     (header-only read; no tensor loads needed).
   - Instantiate or read the base Diffusers transformer config; list the module
     paths that exist in the *stock* graph. The Lite quantizer can only replace
     modules that exist — every converted target key must map onto one.
   - Compare against a known-good converted repo when one exists.

2. **Classify every source module into a conversion situation** (details and
   code patterns in `references/nunchaku-diffusers-conversion.md`):

   | Situation | Signal | Action |
   |---|---|---|
   | 1:1 rename | source and Diffusers module shapes match | rename keys, rename suffixes (`lora_down→proj_down`, `lora_up→proj_up`, `smooth→smooth_factor`), drop `*_orig` |
   | Fused → split | source has one module (e.g. `qkv_proj`), Diffusers has several (`to_q/to_k/to_v`) | unpack output-side tensors, slice rows logically, repack; duplicate input-side tensors |
   | Split → merged | Diffusers has one linear (e.g. FLUX single-block `proj_out`) where source has two (`out_proj` + `mlp_fc2`) | check group alignment first; merge logically: qweight/wscales concat on input dim, smooth concat, proj_down block-diagonal, proj_up concat, biases summed logically |
   | Shifted down-projection (INT4) | source was calibrated fused-GELU + shift + unsigned | add `shift · rowsum(Ŵ/smooth)` back into the bias (signed-unfused compensation) |
   | Rank uniformity | Lite config has one global `rank` but targets differ | zero-pad low-rank pairs to max rank **in logical layout** |
   | AWQ layout | AdaNorm-style interleaved AWQ modules | permute packed rows/scales/zeros/bias to Diffusers order; never dequantize |
   | NVFP4 scales | missing `wtscale`/`wcscales` | synthesize ones; note per-tensor `wtscale` blocks naive merges |

3. **Write the transformer config** with a compact `nunchaku_lite`
   `quantization_config` (svdq/awq target lists, precision, group size, rank).
   Do not emit keys the loader ignores (e.g. `patches`).

4. **Package the pipeline**: copy scheduler/tokenizer(s)/vae/processor and
   `model_index.json` from the base model; quantize large text encoders with
   BitsAndBytes 4-bit NF4 unless the user asks otherwise. Be explicit when the
   output omits or inherits a full-precision component.

5. **Validate up the ladder — do not skip rungs**
   (`references/nunchaku-diffusers-conversion.md` § Validation):
   1. Unit tests for every layout transform, using distinctive values through
      real pack/unpack round-trips.
   2. Key match: converted state dict vs expected targets, exact.
   3. Layer-level A/B with real kernels: converted module output vs source
      layout output on identical inputs (this is what catches packing bugs;
      expect ≤ ~2% relative error from accumulation noise, exact match on
      zeroed input halves).
   4. Pipeline load with the intended Diffusers checkout.
   5. Full multi-step inference; compare against the native Nunchaku baseline
      image. Same-class MAE (roughly the split-vs-native distance) passes; a
      several-times-larger MAE means a layout bug, not "quantization loss".

6. **Benchmark and write the README**
   (`references/packaging-readme-benchmark.md`): latency (warmup + ≥3 measured
   runs), peak CUDA memory, output comparison image vs the native baseline,
   and a runnable `from_pretrained` snippet. `scripts/benchmark_pipeline.py`
   is a model-agnostic runner.

## References

- `references/nunchaku-diffusers-conversion.md` — model-agnostic layout rules,
  all conversion situations with code, validation harness patterns.
- `references/packaging-readme-benchmark.md` — output repo structure, README
  template, benchmark protocol, acceptance criteria.
- `references/qwen-image-edit-2509.md` — model-specific notes for the bundled
  Qwen example only.

## Bundled Examples

- `scripts/convert_qwen_image_edit_2509_nunchaku_to_diffusers.py` — fused QKV
  split, normal AWQ, BNB4 text encoder packaging.
- `scripts/convert_flux1_kontext_nunchaku_to_diffusers.py` — split→merged
  single-block `proj_out`, signed-unfused bias compensation, logical rank
  padding, AdaNorm AWQ undo, NVFP4 scale synthesis.
- `scripts/benchmark_pipeline.py` — generic latency/VRAM/MAE benchmark runner
  for a converted pipeline directory.
- `scripts/benchmark_lite_repo.py` — benchmarks a packaged repo, standard or
  Modular, into the card JSON schema; loads the quantized transformer
  explicitly because `load_components()` ignores `quantization_config`.
- `scripts/benchmark_denoiser_resident.py` — two-phase encode-then-denoise run
  for a dense baseline too large to sit beside its text encoder, so it is
  measured resident instead of streaming under offload.
- `scripts/write_nunchaku_lite_readme.py` — writes the lite-infer style model
  card from those JSON sidecars and the packaged `transformer/config.json`.

Treat the converters as implementation examples: copy their patterns, not their
module names or layer counts.
