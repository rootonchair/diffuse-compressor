# Development Guide

This repository is a model-agnostic diffusion quantization library. The core
package does not hard-code Flux, PixArt, Sana, image editing, or video
architectures. Model-specific structure belongs in examples through
`TargetConfig`, `TargetRule`, `PatchRule`, and `CalibrationScopeRule`.

## Repository Layout

```text
src/diffuse_compressor/
  api.py                       Public orchestration API
  config.py                    Dataclass configs for specs, targets, patches, calibration, export
  targets.py                   Model-agnostic target matching and grouped target expansion
  patches.py                   Generic architecture rewrite modules
  calibration/                 Disk-backed calibration replay and scoped activation capture
    cache.py                   Tensor and module I/O cache containers
    data.py                    Calibration sample datasets and root input cache preparation
    scopes.py                  Scope assignment, replay, hooks, and capture batching
    utils.py                   Tensor/tree helpers, RAM guard, and repartitioning
  artifact.py                  In-memory quantized artifact containers
  methods/svdquant/quantize.py SVDQuant quantization orchestration
  backends/nunchaku/           Nunchaku Lite tensor layout and packing helpers
  exporters/nunchaku.py        Safetensors export with quantization metadata

examples/
  prompts/
    qdiff.yaml                 Vendored qdiff calibration prompts
  text_to_image/               Text-to-image model quantization examples
  image_to_image/              Image-edit quantization examples
  text_to_video/               Text-to-video target config sketch

tests/
  test_targets_and_patches.py  Target grouping and patch behavior tests
  test_calibration_streaming.py Disk cache, scope streaming, and RAM guard tests
  test_quantize_export.py      Quantization, calibration, and export tests
  test_flux2_example_config.py Flux2 config and Nunchaku Lite compatibility tests
  test_upstream_diffusion_examples.py Upstream model config/export smoke tests
```

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

The convenience API `quantize_and_export()` runs the full sequence:

```python
prepare_model()
collect_quant_targets()
quantize_diffusion()
export_checkpoint()
```

Detailed component flow:

```text
User model
  |
  |  TargetConfig.patches
  v
prepare_model()
  - applies generic rewrites such as split_linear and split_linear_output
  - exposes child Linear modules that can be targeted independently
  |
  |  TargetConfig.targets
  v
collect_quant_targets()
  - matches module path patterns against model.named_modules()
  - expands wildcards into concrete QuantTarget objects
  - groups multiple modules into one export target when configured
  |
  |  optional CalibrationSpec + TargetConfig.calibration_scopes
  v
prepare_calibration_cache()
  - records model forward inputs under cache_dir/caches/*.pt
  - reuses or refreshes cached inputs according to cache_mode
  |
  v
iter_calibration_scopes()
  - assigns targets to user-configured scopes
  - replays cached model inputs one scope at a time
  - captures current-scope target inputs and optional declared module I/O on CPU
  - can keep a scope eval replay and use previous scope outputs for simple sequential replay
  - clears scope activations before moving to the next scope
  |
  |  DiffusionQuantSpec
  v
quantize_targets()
  - concatenates grouped linear or pointwise Conv2d weights when needed
  - computes a low-rank SVD branch
  - uses current-scope activations for weighted SVD when calibration is provided
  - quantizes residual weights to INT4
  - emits Nunchaku Lite parameter names and tensor shapes
  |
  |  ExportSpec
  v
export_checkpoint()
  - combines quantized targets with untouched non-target parameters
  - writes one safetensors checkpoint
  - writes config metadata
  - stores method/rank/weight/activation compatibility metadata in quantization_config
  - adds runtime_manifest to quantization_config when available
```

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
checkpoints but does not implement quantized inference kernels. See
[evaluation.md](evaluation.md) for the runtime helper API and benchmark
commands.

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
See [examples.md](examples.md) for the full example matrix and adaptation
notes.

## Extending The Library

The intended extension points are:

- Add a new quantization method under `src/diffuse_compressor/methods/`.
- Add a dispatcher branch in `api.py` once the method has a stable interface.
- Add generic patch types to `patches.py` only when they are architecture
  independent.
- Add exporters under `src/diffuse_compressor/exporters/`.
- Add model-specific configs under `examples/` or in downstream projects, not
  in the library core.

For future methods inspired by `cache-dit`, the recommended shape is to keep
method-specific data structures inside a new method package and reuse the
existing target collection, patching, calibration replay, and export artifact
interfaces where possible.

Backlog items are tracked in [backlog.md](backlog.md).

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
