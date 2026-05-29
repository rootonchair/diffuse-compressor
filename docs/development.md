# Development Guide

This repository is a model-agnostic diffusion quantization library. The core
package does not hard-code Flux, PixArt, Sana, image editing, or video
architectures. Model-specific structure belongs in examples through
`TargetConfig`, `TargetRule`, `PatchRule`, and `CalibrationScopeRule`.

## Core Flow

The main orchestration path is:

```text
model
  -> prepare_model()
  -> collect_quant_targets()
  -> iter_calibration_scopes()
  -> quantize_targets()
  -> export_checkpoint()
```

Start with `src/diffuse_compressor/api.py` and
`src/diffuse_compressor/config.py`. `quantize_and_export()` is the main public
entry point. `config.py` defines the dataclasses that control quantization,
target collection, calibration, and export.

## Target Configs And Patches

Target matching lives in `src/diffuse_compressor/targets.py`. A `TargetRule`
turns wildcard module paths into concrete quantization targets. Group modules
only when they consume the same activation tensor, such as Q/K/V projections.

Model rewrites live in `src/diffuse_compressor/patches.py`. Use patches for
generic transformations, such as splitting fused linear projections before
target collection. Keep architecture-specific names in `examples/`, not in the
library core.

## Calibration

Calibration is the most fragile subsystem. The main files are:

- `calibration/data.py`: sample resolution, DataLoader collation, root input
  cache creation, and custom `forward_fn` execution.
- `calibration/scopes.py`: scope assignment, replay, hooks, previous-scope
  replay, and activation capture.
- `calibration/cache.py`: tensor cache containers.
- `calibration/utils.py`: tensor tree helpers, repartitioning, and RAM checks.

Flux calibration uses block-local replay to avoid replaying the full
transformer for every scope. The first scoped replay can early-stop after the
current block, and later scopes can use previous eval replay records to rebuild
block inputs with preserved kwargs such as `temb`, `image_rotary_emb`, and
`joint_attention_kwargs`.

## Quantization And Export

SVDQuant code lives under `src/diffuse_compressor/methods/svdquant/`.
`quantize_targets()` handles grouped weights, low-rank branches, smoothing,
residual quantization, and activation or weight range tensors.

Export lives in `src/diffuse_compressor/exporters/nunchaku.py`. It writes one
safetensors checkpoint containing quantized target tensors, selected
unquantized tensors, plus an adjacent config documented in
`docs/checkpoint_metadata.md`. Safetensors metadata keeps compact compatibility
fields and, when available, the Nunchaku Lite runtime ABI manifest documented
in `docs/nunchaku_lite_manifest_v1.md`.

## Evaluation

Evaluation/runtime helpers live in `src/diffuse_compressor/runtime.py`. They
load or patch one pipeline at a time for user-owned evaluation loops. Keep
runtime-specific code behind adapters because the core package exports
checkpoints but does not implement quantized inference kernels.

## Examples

Most architecture knowledge is in:

- `examples/text_to_image/`: model-owned target configs for Flux, Flux2,
  PixArt, Sana, and ERNIE, including their CLI defaults and calibration helpers.
- `examples/image_to_image/`: image-edit target configs such as LongCat.
- `examples/text_to_video/`: text-to-video target sketches.
- `evaluation/evaluate_image_generation.py`: DeepCompressor-style image
  generation and metrics for one original or quantized run.

When adding a model, first add or adapt an example `TargetConfig`. Only move
logic into `src/diffuse_compressor` when it is genuinely model-agnostic.

## Testing

Run focused tests for the subsystem you change:

```bash
pytest tests/test_targets_and_patches.py
pytest tests/test_calibration_streaming.py
pytest tests/test_quantize_export.py
pytest tests/test_upstream_diffusion_examples.py
```

Before handing off a broad change, run:

```bash
pytest
```

Use `tests/test_calibration_streaming.py` for calibration replay regressions,
especially previous-scope replay, cache batching, and activation capture.
