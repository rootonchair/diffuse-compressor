# Low-Memory Quantization Settings

This guide describes the main settings for reducing CPU RAM and GPU VRAM during
diffusion SVDQuant calibration and export.

## Recommended LongCat Command

For LongCat image-edit quantization on a memory-constrained machine, start with:

```bash
python examples/image_to_image/quantize_longcat_image_edit.py \
  --precision nvfp4 \
  --cache-mode reuse \
  --num-samples 128 \
  --cache-num-samples 128 \
  --height 512 \
  --width 512 \
  --scope-capture-mode one-target \
  --offload-model \
  --compute-device cuda
```

If cache records already exist, `--cache-mode reuse` avoids regenerating them.
If the cache is stale or incompatible with the current pipeline arguments, use
`--cache-mode refresh` once.

For LongCat image-edit, `--height` and `--width` are wired through a
calibration-time target-size override. Refresh the cache after changing them.

## CPU RAM Controls

### `--cache-num-samples`

Limits how many cached root forward records are replayed during calibration.
This is the most important knob when a cache directory contains many denoising
step records.

```bash
--cache-num-samples 128
```

Behavior:

- `None` or omitted: use all cached records.
- `-1`: use all cached records.
- Positive integer: deterministically select that many cached `.pt` records
  using the calibration seed.

This is separate from `--num-samples`.

### `--num-samples`

Limits raw calibration data records before cache generation.

```bash
--num-samples 128
```

For LongCat, each image-edit data record can produce multiple cached transformer
input records because every denoising step calls the transformer. For example,
100 image-edit records with 8 steps can produce about 800 cached records.

When reusing an existing cache, `--num-samples` does not limit cached records.
Use `--cache-num-samples` for that.

### `--scope-capture-mode one-target`

Captures one target at a time inside each calibration scope:

```bash
--scope-capture-mode one-target
```

This lowers peak CPU RAM because the implementation keeps only one target's
input/output activation cache at a time. Eval replay records are still retained
once per scope so later targets can reuse the same scope-level objective.

Use `all-targets` only when you have enough RAM and want fewer replay passes.

### `sample_size` and `sample_batch_size`

`CalibrationSpec.sample_size` limits flattened activation rows used by
partitioned calibration consumers. `sample_batch_size` splits those rows into
smaller partitions.

The upstream CLI exposes:

```bash
--sample-batch-size 32
```

Use a moderate value such as `32`, `64`, or `128` for LongCat. Leaving this at
`1` creates many tiny partitions and can make scoring inefficient.

## GPU VRAM Controls

### `--offload-model`

Moves the transformer to CPU while quantizing a captured scope, then restores
only the modules needed for the next calibration replay when possible.

```bash
--offload-model --compute-device cuda
```

Calibration replay and the active scope evaluation still need GPU memory. The
offload mainly reduces VRAM between capture and per-target quantization work.

If scoped replay is unavailable, the implementation falls back to restoring the
full model for correctness and emits a warning.

### `--pipeline-offload`

Uses Diffusers or Accelerate pipeline offload:

```bash
--pipeline-offload model
```

This is separate from `--offload-model`. Pipeline offload controls the pipeline
while generating or replaying cached inputs; `--offload-model` controls this
repo's quantization-time model residency.

Be careful when combining both. Accelerate hooks own module movement, so this
repo avoids manual moves where those hooks are detected.

`sequential` works with quantization: weights that Accelerate leaves on the
`meta` device are materialized on demand for the direct reads quantization
needs, without detaching the hooks calibration replay still depends on. It is
the only choice for a denoiser too large to be resident as a single component,
which is what `model` offload requires.

### `--compute-device`

Selects the device used for per-target quantization compute:

```bash
--compute-device cuda
```

When unset, quantization uses the target weight's current device. With
`--offload-model`, setting `--compute-device cuda` allows per-target math to run
on GPU while the rest of the model can stay offloaded.

## Memory Estimate For LongCat

With:

```bash
--cache-num-samples 128
--scope-capture-mode one-target
```

the worst CPU RAM peak is typically the current target cache plus scope eval
replay records. For LongCat image MLP targets, this can still be around
80-100 GiB before normal Python and PyTorch overhead.

Without `one-target`, a full first scope can retain all target caches together
and may need several hundred GiB for 128 cache records.

Without `--cache-num-samples`, a reused cache with 800 records can multiply
these estimates by about 6.25.

## Practical Presets

Lower CPU RAM:

```bash
--cache-num-samples 64 --scope-capture-mode one-target --sample-batch-size 32
```

Balanced:

```bash
--cache-num-samples 128 --scope-capture-mode one-target --sample-batch-size 64
```

Higher quality, more RAM:

```bash
--cache-num-samples 256 --scope-capture-mode one-target --sample-batch-size 128
```

For maximum calibration coverage, omit `--cache-num-samples` or set it to `-1`,
but expect CPU RAM to grow with the number of cached transformer input records.
