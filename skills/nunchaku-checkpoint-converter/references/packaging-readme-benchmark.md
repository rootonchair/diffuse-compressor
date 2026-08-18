# Packaging, README, and Benchmark for Converted Repos

The deliverable is a self-contained Diffusers model directory in the style of
`lite-infer/*-nunchaku-lite-<precision>_r<rank>-bnb4-text-encoder`, e.g.
`outputs/converted/qwen-image-edit-2509-lightning-4steps-nunchaku-lite-int4_r32-bnb4-text-encoder`.

## Directory Structure

```text
<repo>/
  model_index.json          # pipeline class + component classes
  README.md                 # see template below
  output_comparison.png     # converted vs native baseline, same seed/params
  scheduler/                # copied from base model (or variant-specific)
  tokenizer/ tokenizer_2/   # copied from base model
  processor/                # if the pipeline uses one
  vae/                      # copied from base model
  text_encoder/             # small encoders: copied as-is
  text_encoder_2/           # large encoders: BitsAndBytes 4-bit NF4
  transformer/
    config.json             # base config + nunchaku_lite quantization_config
    diffusion_pytorch_model.safetensors
```

Naming convention: `<model>-nunchaku-lite-<precision>_r<rank>-bnb4-text-encoder`.
Append a short experiment suffix for non-canonical variants.

Text encoder quantization (unless the user asks otherwise):

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_compute_dtype=torch.bfloat16)
```

## README Template

Follow the structure of the known-good Qwen example. Sections, in order:

1. **Title** — model + variant + precision + rank.
2. **Provenance** — bullet list: base Diffusers model repo, source Nunchaku
   checkpoint repo, exact source safetensors filename, any LoRA fused into the
   source checkpoint.
3. **Packaging facts** — `quant_method: nunchaku_lite`, SVDQ precision / group
   size / rank, SVDQ and AWQ target counts, text encoder quantization config,
   any scheduler changes. Note conversion transforms that affect numerics
   (e.g. "single-block proj_out merged from out_proj + mlp_fc2; INT4
   down-projections converted to signed-unfused with bias compensation").
4. **Benchmark** — table (see protocol below) plus one sentence stating steps,
   resolution, warmup/run counts, and what the baseline row is.
5. **Output Comparison** — embed `output_comparison.png`; state that both
   images used the same input, prompt, seed, scheduler, and step count.
6. **Run** — a complete, copy-pasteable snippet: plain
   `Pipeline.from_pretrained(<repo>, torch_dtype=torch.bfloat16)`, real
   pipeline call with the recommended parameters, seed fixed. State every
   install the snippet needs -- `kernels`, `bitsandbytes` when a component is
   BNB4, and Diffusers from git while `nunchaku_lite` is unreleased -- and set
   `DIFFUSERS_TRUST_REMOTE_KERNELS` inside the snippet, before `diffusers` is
   imported, since it is read at import time. Hardware: INT4 needs Turing or
   newer, NVFP4 needs Blackwell or newer, Hopper unsupported.

## Benchmark Protocol

Use `scripts/benchmark_pipeline.py` or replicate its procedure:

- Fixed prompt, seed, steps, and resolution; record all of them in a JSON
  sidecar next to the images (prompt, seed, steps, guidance, sizes, torch
  version, GPU name).
- **Latency**: 1 warmup run, then mean of ≥ 3 measured runs, end-to-end
  pipeline call.
- **Peak VRAM**: `torch.cuda.reset_peak_memory_stats()` before the measured
  run; report `max_memory_allocated` in GiB. Note the offload mode used
  (`.to("cuda")` vs `enable_model_cpu_offload()`) — numbers are not comparable
  across modes.
- **Baseline row**: the original Nunchaku checkpoint through the native
  Nunchaku runtime, same parameters, when the native runtime is available.
  Otherwise benchmark only the converted pipeline and say so.
- **Quality**: pixel MAE / RMSE between converted and baseline outputs, plus a
  side-by-side `output_comparison.png`. Calibrate the acceptance threshold on
  a known-good pair first; same-class MAE passes, several-times-larger MAE is
  a conversion bug (see the validation ladder).

Benchmark table format:

```markdown
| Checkpoint | Latency | Max VRAM |
| --- | ---: | ---: |
| Converted Diffusers Nunchaku Lite INT4 r32 + BNB4 text encoder | 12.60 s | 21.21 GiB |
| Original Nunchaku INT4 r32 safetensors | 11.27 s | 35.13 GiB |
```

## Reporting

Keep every generated artifact (output PNGs, comparison PNG, JSON sidecars)
under a dedicated smoke/benchmark directory in the workspace
(`outputs/inference_smoke/<model>_*` in this repo), and reference the exact
files from the README and from any issue/doc write-up. Never overwrite a
previous experiment's artifacts — suffix new runs instead, so regressions stay
diffable.
