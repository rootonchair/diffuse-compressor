# diffuse_compressor

`diffuse_compressor` is a model-agnostic quantization toolkit for diffusion
transformers. The initial implementation focuses on SVDQuant for linear
projections and pointwise Conv2d projections, and exports checkpoints that can
be loaded by Nunchaku Lite and Nunchaku-style runtimes.

The library deliberately does not know about Flux, Flux2, image editing
pipelines, or text-to-video architectures. Model-specific structure is provided
by user-side configuration: which modules should be quantized, which modules
should be grouped, and which fused modules should be split before quantization.
This keeps the core reusable while still allowing configs to reproduce
DeepCompressor-style architecture rewrites.

## Goals

- Provide a small Python API for diffusion model quantization.
- Keep model architecture knowledge outside the library core.
- Support grouped targets such as QKV, added QKV, or KV projections through
  config rather than hard-coded adapters.
- Support pointwise Conv2d projector targets used by architectures such as
  Sana, while leaving depthwise/spatial convolutions unquantized by config.
- Support generic module rewrites needed by quantized runtimes, such as
  splitting fused linear projections.
- Use DeepCompressor-style calibration storage: cache model forward inputs to
  disk, replay them by configured scope, and clear activation RAM after each
  scope.
- Export single-file safetensors checkpoints compatible with Nunchaku Lite
  module names and tensor layout.
- Leave clear extension points for future quantization methods and exporters.

## Non-Goals

- The core library does not auto-detect Flux, Flux2, SDXL, Wan, HunyuanVideo,
  or any other architecture.
- It does not provide a runtime inference engine.
- It does not currently implement every DeepCompressor optimization pass.
- FP4/NVFP4 export records DeepCompressor-style scale dtype metadata and uses
  the current FP4 residual packing path; runtime support still depends on the
  loader consuming the checkpoint.

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
  methods/svdquant/quantize.py SVDQuant implementation and Nunchaku Lite tensor layout
  methods/svdquant/packing.py  Legacy/utility packing helpers
  exporters/nunchaku.py        Safetensors export with quantization metadata

examples/
  flux_svdquant.py             Flux-style user config example
  flux2_klein_4b_svdquant.py   FLUX.2 Klein 4B user config and quantization script
  quantize_flux2_klein_4b.sh   Bash wrapper for full FLUX.2 Klein 4B quantization
  upstream_diffusion_svdquant.py Shared upstream DeepCompressor diffusion configs
  quantize_flux1_schnell.py    FLUX.1 Schnell INT4/NVFP4 quantization
  quantize_flux1_dev.py        FLUX.1 Dev INT4/NVFP4 quantization
  quantize_flux2_klein_4b.py   FLUX.2 Klein 4B INT4/NVFP4 quantization
  quantize_flux2_klein_9b.py   FLUX.2 Klein 9B INT4/NVFP4 quantization
  quantize_pixart_sigma.py     PixArt Sigma INT4/NVFP4 quantization
  quantize_sana_1_6b.py        Sana 1.6B INT4/NVFP4 quantization
  quantize_upstream_diffusion_svdquant.sh Matrix runner for the upstream examples
  text_to_video_svdquant.py    Text-to-video target config sketch

tests/
  test_targets_and_patches.py  Target grouping and patch behavior tests
  test_calibration_streaming.py Disk cache, scope streaming, and RAM guard tests
  test_quantize_export.py      Quantization, calibration, and export tests
  test_flux2_example_config.py Flux2 config and Nunchaku Lite compatibility tests
  test_upstream_diffusion_examples.py Upstream model config/export smoke tests
```

## Component Flow

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
  - stores quantization_config metadata
```

The convenience API `quantize_and_export()` runs the full sequence:

```python
prepare_model()
collect_quant_targets()
quantize_diffusion()
export_checkpoint()
```

## Upstream Diffusion Examples

The `examples/upstream_diffusion_svdquant.py` module contains user-side
configs for every diffusion model family represented by upstream
DeepCompressor SVDQuant diffusion configs. The library core still does not
know these architectures; the examples specify target module patterns,
grouped QKV/KV behavior, fused projection splitting, and pointwise Conv2d
targets where needed.

| Example | Upstream model id | Defaults | Notes |
| --- | --- | --- | --- |
| `quantize_flux1_schnell.py` | `black-forest-labs/FLUX.1-schnell` | 4 steps, guidance 0.0, calib batch 16 | Flux double/single blocks, grouped QKV/add-QKV, split single block output projection |
| `quantize_flux1_dev.py` | `black-forest-labs/FLUX.1-dev` | 50 steps, guidance 3.5, calib batch 16 | Same target layout as Schnell |
| `quantize_flux2_klein_4b.py` | `black-forest-labs/FLUX.2-klein-4B` | 4 steps, guidance 1.0, calib batch 1 | FLUX.2 double/single blocks, grouped QKV/add-QKV, split fused single-block QKV+MLP projections |
| `quantize_flux2_klein_9b.py` | `black-forest-labs/FLUX.2-klein-9B` | 4 steps, guidance 1.0, calib batch 1 | Same FLUX.2 layout as 4B with wider 9B split sizes |
| `quantize_pixart_sigma.py` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | 20 steps, guidance 4.5, calib batch 256 | Self-attention QKV, cross-attention KV, MLP projections |
| `quantize_sana_1_6b.py` | `Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632` | 20 steps, guidance 4.5, calib batch 256 | Adds pointwise Conv2d FFN targets; depthwise conv is intentionally not quantized |

Run one model and precision:

```bash
python examples/quantize_flux1_schnell.py --precision int4
python examples/quantize_flux1_schnell.py --precision nvfp4
python examples/quantize_flux1_dev.py --precision int4
python examples/quantize_flux1_dev.py --precision nvfp4
python examples/quantize_flux2_klein_4b.py --precision int4
python examples/quantize_flux2_klein_4b.py --precision nvfp4
python examples/quantize_flux2_klein_9b.py --precision int4
python examples/quantize_flux2_klein_9b.py --precision nvfp4
python examples/quantize_pixart_sigma.py --precision int4
python examples/quantize_pixart_sigma.py --precision nvfp4
python examples/quantize_sana_1_6b.py --precision int4
python examples/quantize_sana_1_6b.py --precision nvfp4
```

Or run the whole upstream model matrix for one precision:

```bash
examples/quantize_upstream_diffusion_svdquant.sh int4
examples/quantize_upstream_diffusion_svdquant.sh nvfp4
```

INT4 examples use `rank=32`, `group_size=64`, INT4 residual packing, activation
shift, DeepCompressor-style low-rank search, and projection smoothing search.
NVFP4 examples use `rank=32`, `group_size=16`,
`weight_scale_dtypes=(None, "sfp8_e4m3_nan")`, and the same search/smoothing
flow. For Flux and PixArt NVFP4, extra norm/AdaLN linear weights are exported
as target-level INT4 weight-only overrides to mirror the upstream precision
overlay.

The default output path is
`outputs/checkpoints/svdq-<precision>_r32-<model>.safetensors`; calibration
root input caches and artifact caches are stored under
`outputs/calibration/<model>/<precision>/...` unless `--cache-dir` is supplied.

### Adapting Target Configs

The `*_target_config()` functions in
`examples/upstream_diffusion_svdquant.py` are meant to be copied and edited for
new model architectures. The core question is not "is this model Flux or
PixArt?", but "which modules should become each exported runtime projection?"
The same module also includes `module_class_selector_config()` as a compact
example for class-only target and scope selectors.

Start by printing the model module tree:

```python
for name, module in model.named_modules():
    if module.__class__.__name__ in {"Linear", "Conv2d"}:
        print(name, module)
```

Then build `TargetRule`s from the runtime projection layout:

- Use one `TargetRule` for one exported projection tensor family.
- Use a scoped class scan when the module path is already the export name, for
  example `TargetRule(scope_module_classes=BlockCls, module_classes=nn.Linear)`.
- Use grouped rules when several modules form one exported runtime projection.
  Grouping can be path-based with multiple `modules` patterns, or class-based
  with `parent_module_classes` plus a `member_selector` callable. Put modules in
  one group only when they consume the same activation tensor and should share
  one low-rank branch. Typical examples are self-attention Q/K/V or
  cross-attention K/V.
- Do not group projections that consume different inputs. Cross-attention Q
  usually consumes hidden states, while K/V consume encoder states, so Q should
  be separate from K/V.
- Use wildcard captures for repeated blocks. A rule with
  `modules=["blocks.*.attn.q", "blocks.*.attn.k", "blocks.*.attn.v"]` produces
  one target per block, and `export_name="blocks.{0}.attn.qkv"` reuses the
  block index captured by `*`.
- Add `module_classes` when broad patterns should only select specific module
  implementations. Path patterns remain the primary selector, so wildcard
  captures and export-name formatting keep working:

  ```python
  TargetRule(
      name="block_proj",
      modules=["blocks.*.proj"],
      export_name="blocks.{0}.proj",
      module_classes=nn.Linear,
  )
  ```

- Omit `name` and `modules` when class identity alone is the intended selector.
  This creates one target per named child module, skips the root model object,
  and uses the matched module path as the target and export name:

  ```python
  TargetRule(module_classes=MyProjection)
  ```

- For callable groups, return an ordered mapping from role name to child module:

  ```python
  TargetRule(
      parent_module_classes=AttentionCls,
      member_selector=lambda attn: {"q": attn.to_q, "k": attn.to_k, "v": attn.to_v},
      export_name="{parent_path}.qkv_proj",
  )
  ```

  The mapping keys become `roles`, and `{parent_path}` is available in `name`
  and `export_name`.
- Use `SkipRule` to exclude modules from broad class scans. Explicit path rules
  that select a skipped module still raise, so skips do not hide typos.
- Set `kind="conv"` only for pointwise `nn.Conv2d` projector modules with
  `kernel_size=(1, 1)` and `groups=1`. Depthwise convs and spatial convs should
  stay out of `TargetConfig.targets` unless a dedicated quantization path is
  added.
- Use target-level overrides such as `precision="int4"`, `group_size=64`,
  `rank=0`, `smooth=False`, `activation_quant=False`, and
  `shift_activations=False` for extra-weight policies, like the NVFP4 configs
  that keep selected norm linears in INT4 W4A16-style form.

Use `PatchRule`s only for generic rewrites needed before matching targets. For
example, Flux single blocks split a fused output projection before the target
rules match its child linears. If a new architecture has a fused QKV or fused
QKV+MLP projection, split it first and then target the exposed children.

Use `CalibrationScopeRule`s to control memory and replay granularity. A normal
transformer stack usually has one scope rule per repeated block collection,
for example `CalibrationScopeRule("blocks.{0}", ["blocks.*"])`. More complex
architectures can add `capture_modules`, `cache_aliases`, and replay argument
filters, but the first pass should keep scopes aligned with the blocks that
own the target projections.

Scope rules accept the same `module_classes` selector. With `modules` present,
the class selector filters path matches. With `modules` and `name` omitted,
each named child module matching the class becomes one scope and the module path
is used as the scope name:

```python
CalibrationScopeRule(module_classes=MyTransformerBlock)
```

Before running a full quantization job, test a new config on a tiny model or a
single real block:

```python
prepare_model(model, target_config.patches)
targets = collect_quant_targets(model, target_config)
for target in targets:
    print(target.export_name, target.module_names, target.kind)
```

The expected result is a complete list of runtime projection names with no
missing modules, no duplicate `export_name`s, and grouped targets ordered the
same way the runtime expects them in the exported checkpoint.

## Configuration Model

### Quantization Spec

`DiffusionQuantSpec` describes the quantization method and numeric settings:

```python
DiffusionQuantSpec(
    method="svdquant",
    precision="int4",
    rank=32,
    group_size=64,
)
```

### Patch Rules

`PatchRule` describes generic module rewrites. These are not tied to any model
family.

Supported patch types:

- `split_linear`: split a linear by input features; child outputs are summed.
- `split_linear_output`: split a linear by output features; child outputs are
  concatenated.
- `split_conv`: split a convolution by input channels.
- `shift_linear`: fold an activation shift into a linear module.
- `shift_conv`: fold an activation shift into a convolution module.

Example:

```python
PatchRule(
    type="split_linear_output",
    module="single_transformer_blocks.*.attn.to_qkv_mlp_proj",
    args={"splits": [9216]},
)
```

### Target Rules

`TargetRule` describes which modules become a quantized export target. A rule
with one module pattern creates one target per matched module. A rule with
multiple module patterns creates one grouped target for each shared wildcard
capture tuple.

```python
TargetRule(
    modules=[
        "transformer_blocks.*.attn.to_q",
        "transformer_blocks.*.attn.to_k",
        "transformer_blocks.*.attn.to_v",
    ],
    export_name="transformer_blocks.{0}.attn.to_qkv",
    roles=["q", "k", "v"],
)
```

Grouping is positional: `roles[0]` labels `modules[0]`, `roles[1]` labels
`modules[1]`, and so on. Wildcard captures must line up across grouped module
patterns. In the example above, the shared capture is the block index from `*`,
so `transformer_blocks.0.attn.to_q`, `to_k`, and `to_v` become one grouped
target exported as `transformer_blocks.0.attn.to_qkv`.

Without wildcards, multiple exact module paths form one group keyed by the
empty capture tuple. With mismatched wildcards, only captures present in every
pattern form groups; if there is no shared capture, target collection raises.

`module_classes` can be a class or a sequence of classes. When combined with
`modules`, it filters the modules matched by the path patterns. When `modules`
is omitted, it selects every named child module whose instance matches one of
the classes; if `name` and `export_name` are also omitted, the module path is
used for both. `scope_module_classes` constrains a class scan to descendants of
matching block modules.

Callable grouping is useful when child property names are architecture details
but the parent module class is stable:

```python
TargetRule(
    parent_module_classes=AttentionCls,
    member_selector=lambda attn: {"q": attn.to_q, "k": attn.to_k, "v": attn.to_v},
    export_name="{parent_path}.qkv_proj",
)
```

Callable group members are collected before broad scans. This means a later
`TargetRule(scope_module_classes=BlockCls, module_classes=nn.Linear)` can pick
up the remaining linears without also quantizing Q/K/V as standalone targets.

Target-level export controls are available for runtime-specific tensor
contracts. `weight_layout=AwqW4A16Layout()` exports a single INT4 linear target
in the Nunchaku Lite AWQ W4A16 extra-weight layout, while
`weight_layout=AdaNormAwqW4A16Layout(splits=3 or 6)` applies the
DeepCompressor AdaNorm W4A16 export transform. `export_bias="zero"` writes a
synthesized zero bias for biasless modules, which is useful when a runtime
expects split projections to expose separate bias tensors.

### Skip Rules

`SkipRule` excludes modules from broad class scans without making grouping
implicit:

```python
TargetConfig(
    targets=[TargetRule(scope_module_classes=BlockCls, module_classes=nn.Linear)],
    skips=[SkipRule(modules=["blocks.*.linear_norm"])],
)
```

Skips can match `modules`, `module_classes`, or class scans scoped by
`scope_module_classes`. Explicit `TargetRule(modules=[...])` entries that select
skipped or already-grouped modules raise an error.

### Calibration Scope Rules

`CalibrationScopeRule` defines the replay granularity for calibration. Scopes
are user-provided so the core library stays model-agnostic.

```python
CalibrationScopeRule(
    name="transformer_blocks.{0}",
    modules=["transformer_blocks.*"],
    eval_module="transformer_blocks.*",
    cache_aliases={
        "transformer_blocks.{0}.attn.to_k": "transformer_blocks.{0}.attn.to_q",
        "transformer_blocks.{0}.attn.to_v": "transformer_blocks.{0}.attn.to_q",
    },
    replay_kwarg_keys=("hidden_states", "encoder_hidden_states", "temb"),
    capture_modules=[
        CalibrationCaptureRule(
            name="transformer_blocks.{0}.attn_io",
            modules=["transformer_blocks.*.attn"],
            inputs=True,
            outputs=True,
            input_keys=("hidden_states", "encoder_hidden_states"),
        ),
    ],
)
```

Targets whose module paths are under `transformer_blocks.0` are calibrated
together, then their activation cache is cleared before `transformer_blocks.1`
is replayed. If no scopes are configured, the library falls back to one target
per scope to avoid holding all target activations in RAM.

`CalibrationScopeRule` also supports `module_classes`. This is useful when a
model exposes stable block classes but path names vary between checkpoints.
Path patterns can still be provided to preserve wildcard captures, or omitted
to create one scope per matching child module path.

Scope capture is keyed and model-agnostic. `input_keys` and `output_keys` select
positional keys such as `"arg0"` or keyword keys such as `"hidden_states"`.
`cache_aliases` lets grouped targets reuse another captured cache, which covers
QKV-style behavior without hardcoding attention architecture names.
`replay_arg_indices`, `replay_kwarg_keys`, and `replay_transform` filter or
rewrite eval-module replay inputs for complex blocks.

### DeepCompressor SVDQuant Mapping

DeepCompressor SVDQuant for diffusion is configured by merging its default
diffusion config, `configs/svdquant/__default__.yaml`, precision config such as
`configs/svdquant/int4.yaml`, and a model config such as
`configs/model/flux.1-dev.yaml`. In `diffuse_compressor`, numeric quantization
settings map to `DiffusionQuantSpec`, calibration storage maps to
`CalibrationSpec`, and model architecture choices move into `TargetConfig`.

| DeepCompressor setting | Meaning | `diffuse_compressor` equivalent |
| --- | --- | --- |
| `quant.wgts.dtype: sint4` | INT4 weight quantization | `DiffusionQuantSpec(precision="int4")` |
| `quant.wgts.dtype: sfp4_e2m1_all` | FP4/NVFP4 residual weight quantization | `DiffusionQuantSpec(precision="fp4")` |
| `quant.wgts.group_shapes: [[1, 64, 1, 1, 1]]` | 64-wide input/channel groups | `DiffusionQuantSpec(group_size=64)` |
| `quant.wgts.group_shapes: [[-1, -1], [1, 16, 1, 1, 1]]` | NVFP4 two-level weight quantization: tensor-level scale plus 16-wide micro groups | `DiffusionQuantSpec(group_size=16, weight_scale_dtypes=(None, "sfp8_e4m3_nan"))`; current exporter records both scale dtypes and packs the 16-wide FP4 weight path |
| `quant.wgts.scale_dtypes: [null]` | Weight scales remain unquantized/model dtype | `DiffusionQuantSpec(weight_scale_dtypes=(None,))` |
| `quant.wgts.scale_dtypes: [null, sfp8_e4m3_nan]` | NVFP4 outer scale remains model dtype and micro scales use SFP8 | `DiffusionQuantSpec(weight_scale_dtypes=(None, "sfp8_e4m3_nan"))` |
| `quant.wgts.low_rank.rank: 32` | SVD low-rank branch rank | `DiffusionQuantSpec(rank=32)` |
| `quant.wgts.enable_low_rank: true` | Enable low-rank branch | `rank > 0` and `TargetRule.shared_low_rank=True` |
| `quant.wgts.low_rank.exclusive: false` | Share low-rank branch across grouped projections | Group modules in one `TargetRule` |
| `quant.wgts.low_rank.num_iters`, `early_stop`, `compensate` | DeepCompressor-style low-rank search controls | `LowRankSolverSpec(mode="search", num_iters=..., early_stop=..., compensate=...)` |
| `quant.wgts.low_rank.objective` | Low-rank search objective | `LowRankSolverSpec(objective="outputs_error")`; only output error is supported |
| `quant.wgts.low_rank.degree` | Error norm degree for search | `LowRankSolverSpec(degree=...)` |
| `quant.wgts.low_rank.sample_size` | Number of samples for low-rank scoring | `LowRankSolverSpec(sample_size=...)` |
| `quant.wgts.low_rank.sample_batch_size` | Low-rank scoring sample batch size | `CalibrationSpec(sample_batch_size=...)` for replay partitions; `LowRankSolverSpec(sample_size=...)` for solver subsampling |
| DeepCompressor `torch.svd_lowrank` acceleration | Approximate low-rank branch SVD | `LowRankSolverSpec(svd_backend="svd_lowrank", svd_lowrank_oversample=10, svd_lowrank_niter=4)` |
| `quant.wgts.low_rank.skips` / `quant.wgts.skips` | Skip model parts | Do not include those modules in `TargetConfig.targets` |
| `quant.wgts.calib_range.*` | Weight dynamic-range calibration state | `WeightRangeCalibrationSpec(...)` exports calibrated residual weight range tensors |
| `quant.ipts.dtype: sint4` | Runtime activation quantization | `ActivationQuantSpec(enabled=True, dtype="int4", ...)` exports activation scale/zero tensors |
| `quant.ipts.dtype: sfp4_e2m1_all` | FP4/NVFP4 runtime activation quantization | Not a separate runtime activation packer yet; activation range metadata still uses `ActivationQuantSpec`, while weight FP4 export is supported |
| `quant.ipts.group_shapes: [[1, 64, 1, 1, 1]]` | INT4 runtime activation groups | `RangeCalibrationSpec(granularity="group")` with `DiffusionQuantSpec(group_size=64)` |
| `quant.ipts.group_shapes: [[1, 16, 1, 1, 1]]` | NVFP4 runtime activation groups | `RangeCalibrationSpec(granularity="group")` with `DiffusionQuantSpec(group_size=16)` for exported activation range metadata |
| `quant.ipts.scale_dtypes: [null]` | INT4 activation scales remain model dtype | `ActivationQuantSpec(scale_dtypes=(None,))` |
| `quant.ipts.scale_dtypes: [sfp8_e4m3_nan]` | NVFP4 activation scale dtype metadata | `ActivationQuantSpec(scale_dtypes=("sfp8_e4m3_nan",))` |
| `quant.ipts.static: true` / `false` | Static activation metadata vs dynamic/runtime activation intent | `ActivationQuantSpec(static=True/False)` |
| `quant.ipts.allow_unsigned: true` | Allow unsigned activation paths | `RangeCalibrationSpec(allow_unsigned=True)` |
| `quant.ipts.calib_range.*` | Input activation range calibration | `ActivationQuantSpec(inputs=RangeCalibrationSpec(...))` |
| `quant.opts.calib_range.*` | Output activation range calibration | `ActivationQuantSpec(outputs=RangeCalibrationSpec(...))` |
| `quant.enable_smooth` / `quant.smooth.proj.*` | SmoothQuant-style projection smoothing | `SmoothSpec(...)` passed through `DiffusionQuantSpec.smooth` |
| `quant.smooth.proj.objective`, `strategy`, `alpha`, `beta`, `num_grids`, `spans` | Projection smoothing search objective and search space | `SmoothSpec(objective="outputs_error", strategy=..., alpha=..., beta=..., num_grids=..., spans=...)` |
| `quant.smooth.proj.granularity`, `allow_low_rank`, `fuse_when_possible`, `skips` | Architecture-aware projection smoothing policy | Partially user-owned through `TargetRule`; full parity not modeled yet |
| `quant.smooth.proj.element_batch_size`, `sample_batch_size`, `element_size`, `sample_size` | Projection smoothing calibration batching/subsampling | `CalibrationSpec(element_batch_size=..., sample_batch_size=..., element_size=..., sample_size=...)` plus `SmoothSpec(sample_size=...)` |
| `quant.enable_extra_wgts: true` | Quantize selected extra modules with a different weight config, used by NVFP4 config | Add extra `TargetRule(...)` entries for those modules with target-level overrides |
| `quant.extra_wgts.dtype: sint4` | Extra weights use INT4 instead of FP4 | `TargetRule(..., precision="int4", group_size=64)` |
| `quant.extra_wgts.group_shapes: [[1, 64, 1, 1, 1]]` | Extra weights use 64-wide groups | `TargetRule(..., group_size=64)` |
| Runtime extra-weight AWQ layout | Nunchaku Lite loads selected extra weights as W4A16 AWQ modules | `TargetRule(..., weight_layout=AwqW4A16Layout(), precision="int4", group_size=64, rank=0, smooth=False, activation_quant=False)` |
| Runtime AdaNorm AWQ layout | DeepCompressor/Nunchaku AdaNorm extra weights use interleaved W4A16 export | `TargetRule(..., weight_layout=AdaNormAwqW4A16Layout(splits=3 or 6), precision="int4", group_size=64, rank=0, smooth=False, activation_quant=False)` |
| `quant.extra_wgts.scale_dtypes: [null]` | Extra weight scales remain unquantized/model dtype | Use the default `weight_scale_dtypes=(None,)` semantics for those INT4 targets |
| `quant.extra_wgts.includes: [transformer_norm, transformer_add_norm]` | Architecture semantic include list for extra weights | Model-agnostic core does not know these labels; user config should add matching `TargetRule`s for the corresponding modules |
| `quant.develop_dtype` | Internal calibration/search dtype | Not exposed; current internals use float32/float64 where needed |
| `pipeline.shift_activations: true` | Shift activation lower-bound outliers into weights | `DiffusionQuantSpec(shift_activations=True)` calibrates scalar lower-bound shifts from target inputs; manual `PatchRule(type="shift_linear", ...)` remains available |
| `quant.calib.data` | Named DeepCompressor calibration dataset | User supplies `CalibrationSpec(samples=...)`, `prompts=...`, or `forward_fn=...` |
| `quant.calib.path` | Calibration cache path | `CalibrationSpec(cache_dir=...)` |
| `quant.calib.num_samples: 128` | Number of calibration samples | `CalibrationSpec(num_samples=128)` |
| `quant.calib.batch_size` | Calibration batch size | `CalibrationSpec(batch_size=...)` |
| DeepCompressor calibration dataset/data loader | Cached sample replay | `CalibrationSpec(batch_size=..., shuffle=..., drop_last=..., num_workers=..., eager_load_samples=...)` |
| DeepCompressor model/block structs | Block-wise calibration replay | `TargetConfig.calibration_scopes` |

Equivalent INT4 SVDQuant skeleton:

```python
spec = DiffusionQuantSpec(
    method="svdquant",
    precision="int4",
    rank=32,
    group_size=64,
    low_rank_solver=LowRankSolverSpec(
        mode="search",
        num_iters=100,
        early_stop=True,
        compensate=False,
        activation_quant=True,
        eval_replay=True,
        svd_backend="svd_lowrank",
        svd_lowrank_oversample=10,
        svd_lowrank_niter=4,
        degree=2,
    ),
    smooth=SmoothSpec(
        enabled=True,
        objective="outputs_error",
        strategy="grid_search",
        alpha=0.5,
        beta=-2,
        num_grids=20,
        spans=(("absmax", "absmax"),),
    ),
    activation_quant=ActivationQuantSpec(
        enabled=True,
        static=True,
        scale_dtypes=(None,),
        inputs=RangeCalibrationSpec(granularity="channel", allow_unsigned=True),
        outputs=RangeCalibrationSpec(granularity="tensor"),
    ),
    weight_range_calibration=WeightRangeCalibrationSpec(
        enabled=True,
        range=RangeCalibrationSpec(granularity="group"),
    ),
)

calibration = CalibrationSpec(
    samples=samples,
    num_samples=128,
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    eager_load_samples=False,
    cache_dir="outputs/calibration/flux",
    cache_mode="reuse",
    max_rows_per_target=4096,
    ram_usage_limit=0.90,
    forward_fn=run_sample,
    artifact_cache=QuantizationCacheSpec(
        cache_dir="outputs/calibration/flux/artifacts",
        cache_mode="reuse",
    ),
)

target_config = TargetConfig(
    patches=[...],
    calibration_scopes=[...],
    targets=[...],
)
```

The main translation difference is ownership: DeepCompressor derives many
module groups and block scopes from architecture-specific structs, while
`diffuse_compressor` keeps that knowledge in user-provided config.

## Usage

### Basic API

```python
from diffuse_compressor import (
    DiffusionQuantSpec,
    ExportSpec,
    LowRankSolverSpec,
    QuantizationCacheSpec,
    SmoothSpec,
    quantize_and_export,
)
from examples.flux2_klein_4b_svdquant import flux2_klein_target_config

result = quantize_and_export(
    model=pipe.transformer,
    spec=DiffusionQuantSpec(
        precision="int4",
        rank=32,
        group_size=64,
        smooth=SmoothSpec(enabled=True),
    ),
    target_config=flux2_klein_target_config(),
    calibration=None,
    export=ExportSpec(output="outputs/checkpoints/model.safetensors"),
)

print(result.checkpoint_path)
```

### Calibration-Aware SVD

If `CalibrationSpec.samples` is provided, the library first records model
forward inputs to `cache_dir/caches/*.pt` as individual sample records, then
replays those cached records through a loader that honors
`CalibrationSpec.batch_size`, `shuffle`, `drop_last`, `num_workers`, and
`seed`. By default cached records are loaded lazily from disk; set
`eager_load_samples=True` to load selected cached records into the dataset up
front. For plain modules, samples can be passed directly into `model(**sample)`.
For diffusion pipelines, pass a `forward_fn` that closes over the full pipeline:

```python
import torch
from diffuse_compressor import CalibrationSpec

def run_sample(sample: dict) -> None:
    pipe(
        prompt=sample["prompt"],
        height=1024,
        width=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator(device="cuda").manual_seed(sample["seed"]),
    )

calibration = CalibrationSpec(
    samples=[
        {"prompt": "A glass robot in a greenhouse", "seed": 0},
        {"prompt": "A red train crossing a mountain bridge", "seed": 1},
    ],
    cache_dir="outputs/calibration/my-model",
    cache_mode="reuse",
    batch_size=16,
    shuffle=False,
    drop_last=False,
    num_workers=0,
    forward_fn=run_sample,
    max_rows_per_target=4096,
    ram_usage_limit=0.90,
)
```

`cache_mode="reuse"` reuses existing cache files, `"refresh"` rewrites them,
and `"disabled"` bypasses disk caching and replays live samples for each scope.
Captured activations are used to compute a weighted SVD branch for each target.
When smoothing is enabled, the same captured activations are used to search
projection smooth factors before weighted SVD and residual weight quantization.
When richer scope capture is configured, each yielded scope also exposes a
DeepCompressor-style layer cache with keyed module inputs, outputs, replay
args, replay kwargs, cache aliases, and repartitioning helpers. These caches are
scope-local and are cleared before the next scope is processed.
If no runnable calibration or reusable cache is provided, the quantizer falls
back to identity smoothing and weight-only SVD.

When `ActivationQuantSpec(enabled=True)` is configured, target input/output
activation ranges are calibrated from the same scoped caches and recorded in
checkpoint metadata, but are not emitted as `input_*` or `output_*` checkpoint
tensors. Nunchaku-style SVDQuant computes activation scales dynamically at
runtime. When `WeightRangeCalibrationSpec(enabled=True)` is configured,
calibrated residual weight range tensors are exported as `weight_range_*`.
These tensors are generic runtime metadata; architecture-specific runtime
consumption remains the exporter/runtime's responsibility.

`CalibrationSpec.artifact_cache` persists quantization artifacts separately
from root model-input caches. The cache writes DeepCompressor-style component
files (`smooth.pt`, `branch.pt`, `wgts.pt`, `acts.pt`, `scale.pt`, and
`model.pt`) and reuses `model.pt` only when the quant spec, target config, and
target export names match the saved cache key.

### Full FLUX.2 Klein 4B Example

The FLUX.2 example lives outside the library core and is just a user config plus
a script:

```bash
./examples/quantize_flux2_klein_4b.sh
```

Override defaults with environment variables:

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_SAMPLES=128 \
BATCH_SIZE=1 \
OUTPUT=outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors \
./examples/quantize_flux2_klein_4b.sh
```

The Python entry point is:

```bash
PYTHONPATH=src python examples/flux2_klein_4b_svdquant.py \
  --model-id black-forest-labs/FLUX.2-klein-4B \
  --num-samples 128 \
  --batch-size 1 \
  --output outputs/checkpoints/svdq-int4_r32-flux2-klein-4b.safetensors
```

## Checkpoint Compatibility

The Nunchaku exporter writes one safetensors file containing:

- Quantized target parameters under configured `export_name` prefixes.
- Untouched non-target model parameters required for strict runtime loading.
- `quantization_config` metadata describing method, rank, precision, group
  size, and target mapping.

For Nunchaku Lite, quantized linear targets use keys such as:

```text
transformer_blocks.0.attn.to_qkv.qweight
transformer_blocks.0.attn.to_qkv.wscales
transformer_blocks.0.attn.to_qkv.smooth_factor
transformer_blocks.0.attn.to_qkv.smooth_factor_orig
transformer_blocks.0.attn.to_qkv.proj_down
transformer_blocks.0.attn.to_qkv.proj_up
```

The test suite includes a strict tiny Flux2 load through `nunchaku_lite` to
guard this layout.

## Evaluation Helpers

`diffuse_compressor.runtime` provides a focused evaluation pipeline loader.
It loads a normal pipeline or patches a loaded pipeline with an exported
quantized checkpoint, while your project owns the Dataset, DataLoader,
generation loop, image saving, and metrics.

```python
from diffuse_compressor.runtime import RuntimePipelineSpec, load_evaluation_pipeline

pipe = load_evaluation_pipeline(
    pipeline_cls=FluxPipeline,
    model_id="black-forest-labs/FLUX.1-schnell",
    spec=RuntimePipelineSpec(
        mode="quantized",
        runtime="torch-dequant",
        checkpoint="outputs/checkpoints/svdq-int4_r32-flux.1-schnell.safetensors",
        model_key="flux.1-schnell",
        device="cuda",
        torch_dtype=torch.bfloat16,
    ),
)

for batch in dataloader:
    with torch.inference_mode():
        images = pipe(**batch["pipeline_kwargs"]).images
```

Set `--runtime torch-dequant` to evaluate an exported packed checkpoint through
ordinary PyTorch modules without installing Nunchaku Lite. This path
dequantizes packed weights, folds low-rank and smoothing tensors into module
weights, and replays calibrated activation-shift wrappers. By default it does
not fake-quantize activations. For debug parity checks, pass
`--torch-dequant-activation-mode input` to fake quantize/dequantize SVDQuant
target inputs with a dynamic per-row/per-group quantizer that applies each
target's smoothing factor before quantization.
Static exported output ranges are not replayed in this path because the
Nunchaku W4A4 runtime quantizes the next target input dynamically instead.
W4A16 extra-weight targets remain weight-only. This mode is an approximation
of the fused Nunchaku W4A4 kernels and is intended for correctness/debug
evaluation rather than performance.

For a fuller DeepCompressor-style image-generation run where original and
quantized models are evaluated separately, see:

```bash
PYTHONPATH=src:. python evaluation/evaluate_image_generation.py \
  --mode original \
  --model-key flux.1-schnell \
  --benchmark MJHQ \
  --output-dir outputs/eval/flux.1-schnell/original \
  --num-samples 1024

PYTHONPATH=src:. python evaluation/evaluate_image_generation.py \
  --mode quantized \
  --model-key flux.1-schnell \
  --runtime torch-dequant \
  --checkpoint outputs/checkpoints/svdq-int4_r32-flux.1-schnell.safetensors \
  --benchmark MJHQ \
  --ref-root outputs/eval/flux.1-schnell/original \
  --output-dir outputs/eval/flux.1-schnell/int4 \
  --num-samples 1024
```

Use `--benchmark DCI` for the sDCI prompt/image benchmark instead. For MJHQ and
DCI, the example downloads the benchmark images and prompts through
`datasets`, writes the ground-truth images under `targets/MJHQ-N` or
`targets/sDCI-N`, and uses them for `with_gt` metrics. `--ref-root` remains the
separately generated original-model output root for `with_orig` metrics. The
example imports metric packages only when requested. Install the metric extras with
`python -m pip install -e ".[eval]"`.

## Extending the Library

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

Backlog items from the DeepCompressor SVDQuant mapping:

- High priority: implement true NVFP4/Nunchaku weight scale layout parity.
  Split effective FP4 residual scales into FP8 micro `wscales`, optional
  per-output-channel `wcscales`, and optional scalar `wtscale` instead of
  collapsing them into BF16 `wscales`.
- Resolved: W4A16/AdaLN asymmetric zero-point export parity. Extra-weight
  AdaLN/norm linears now use DeepCompressor-compatible AWQ W4A16 packing,
  `wzeros = -7 * wscales`, unsigned `q = round(weight / scale + 7)`, and
  AdaNorm split interleaving via `AdaNormAwqW4A16Layout`.
- Extend smoothing beyond the implemented target-local projection search:
  add full DeepCompressor projection policy parity for `granularity`,
  `allow_low_rank`, `fuse_when_possible`, and `skips`.
- Extend generic scope replay beyond multi-eval replay scoring toward full
  DeepCompressor `iter_layer_activations` parity with module output needs
  functions and architecture-specific traversal helpers.
- Add optional user-side semantic skip preset helpers for categories such as
  `embed`, `resblock_shortcut`, `resblock_time_proj`, `transformer_proj_in`,
  `transformer_proj_out`, `transformer_norm`, `transformer_add_norm`,
  `down_sample`, and `up_sample`, while keeping core target discovery
  model-agnostic.
- Consider a future `format="w4a16"` or `target_kind="w4a16"` target preset
  once multiple configs need the same explicit extra-weight override bundle.
- Add GPTQ kernel calibration support for `configs/svdquant/gptq.yaml`.

## Development

Install in editable mode:

```bash
pip install -e .
```

Run tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
```

Current expected result:

```text
14 passed
```
