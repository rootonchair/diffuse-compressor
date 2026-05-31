# Example Configs

The `examples/text_to_image/`, `examples/image_to_image/`, and
`examples/text_to_video/` folders contain user-side configs for the diffusion
model families represented by upstream DeepCompressor SVDQuant diffusion
configs. The library core still does not know these architectures; the examples
specify target module patterns, grouped QKV/KV behavior, fused projection
splitting, pointwise Conv2d targets, CLI defaults, prompt handling, and
calibration wiring where needed. Each model's defaults and helper code stay next
to that model's quantization config.

For start-to-finish task guides, see
[text_to_image_end_to_end_guide.md](text_to_image_end_to_end_guide.md) and
[image_to_image_end_to_end_guide.md](image_to_image_end_to_end_guide.md).

Install the package normally for Diffusers-backed example support:

```bash
python -m pip install -e .
```

Install example data-loading extras when using examples that download
calibration images from Hugging Face datasets:

```bash
python -m pip install -e ".[examples]"
```

Nunchaku Lite runtime patching still requires installing `nunchaku_lite` from
its release or private package channel. The `nunchaku-lite` extra is kept as an
explicit marker for that optional runtime, but it does not install a public PyPI
package.

## Supported Examples

| Example | Upstream model id | Defaults | Notes |
| --- | --- | --- | --- |
| `text_to_image/quantize_flux1_schnell.py` | `black-forest-labs/FLUX.1-schnell` | 4 steps, guidance 0.0, calib batch 16 | Flux double/single blocks, grouped QKV/add-QKV, split single block output projection |
| `text_to_image/quantize_flux1_dev.py` | `black-forest-labs/FLUX.1-dev` | 50 steps, guidance 3.5, calib batch 16 | Same target layout as Schnell |
| `text_to_image/quantize_flux2_klein_4b.py` | `black-forest-labs/FLUX.2-klein-4B` | 4 steps, guidance 1.0, calib batch 1 | FLUX.2 double/single blocks, grouped QKV/add-QKV, split fused single-block QKV+MLP projections |
| `text_to_image/quantize_flux2_klein_9b.py` | `black-forest-labs/FLUX.2-klein-9B` | 4 steps, guidance 1.0, calib batch 1 | Same FLUX.2 layout as 4B with wider 9B split sizes |
| `text_to_image/quantize_pixart_sigma.py` | `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | 20 steps, guidance 4.5, calib batch 256 | Self-attention QKV, cross-attention KV, MLP projections |
| `text_to_image/quantize_sana_1_6b.py` | `Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632` | 20 steps, guidance 4.5, calib batch 256 | Adds pointwise Conv2d FFN targets; depthwise conv is intentionally not quantized |
| `image_to_image/quantize_longcat_image_edit.py` | `meituan-longcat/LongCat-Image-Edit-Turbo` | 8 steps, guidance 1.0, calib batch 1 | Image-edit calibration from the `validation` split of `VyoJ/NHR-Edit-Change_Only`; exact module-path targets for generic manifest loading |
| `text_to_image/quantize_ernie_image.py` | `baidu/ERNIE-Image` | 50 steps, guidance 4.0, calib batch 1 | Exact module-path manifest targets; repeated block SVDQ plus INT4 AWQ extra linears; prompt enhancer disabled for calibration |
| `text_to_image/quantize_ernie_image_turbo.py` | `baidu/ERNIE-Image-Turbo` | 8 steps, guidance 1.0, calib batch 1 | Same ERNIE manifest layout as the base model with Turbo defaults |

## Command Matrix

Run one model and precision:

```bash
python examples/text_to_image/quantize_flux1_schnell.py --precision int4
python examples/text_to_image/quantize_flux1_schnell.py --precision nvfp4
python examples/text_to_image/quantize_flux1_dev.py --precision int4
python examples/text_to_image/quantize_flux1_dev.py --precision nvfp4
python examples/text_to_image/quantize_flux2_klein_4b.py --precision int4
python examples/text_to_image/quantize_flux2_klein_4b.py --precision nvfp4
python examples/text_to_image/quantize_flux2_klein_9b.py --precision int4
python examples/text_to_image/quantize_flux2_klein_9b.py --precision nvfp4
python examples/text_to_image/quantize_pixart_sigma.py --precision int4
python examples/text_to_image/quantize_pixart_sigma.py --precision nvfp4
python examples/text_to_image/quantize_sana_1_6b.py --precision int4
python examples/text_to_image/quantize_sana_1_6b.py --precision nvfp4
python examples/image_to_image/quantize_longcat_image_edit.py --precision int4
python examples/image_to_image/quantize_longcat_image_edit.py --precision nvfp4
python examples/text_to_image/quantize_ernie_image.py --precision int4
python examples/text_to_image/quantize_ernie_image.py --precision nvfp4
python examples/text_to_image/quantize_ernie_image_turbo.py --precision int4
python examples/text_to_image/quantize_ernie_image_turbo.py --precision nvfp4
```

Example CLIs write run logs by default under `outputs/logs`: a text
quantization run log and a `.targets.jsonl` file with per-target elapsed time
and low-rank error records. Use `--log-dir <path>` to choose another directory
or `--no-run-log` to disable these files.

To run several upstream examples for one precision, call the model entry points
directly from a shell loop:

```bash
for example in \
  text_to_image/quantize_flux1_schnell.py \
  text_to_image/quantize_flux1_dev.py \
  text_to_image/quantize_flux2_klein_4b.py \
  text_to_image/quantize_flux2_klein_9b.py \
  text_to_image/quantize_pixart_sigma.py \
  text_to_image/quantize_sana_1_6b.py \
  image_to_image/quantize_longcat_image_edit.py \
  text_to_image/quantize_ernie_image.py \
  text_to_image/quantize_ernie_image_turbo.py; do
  python "examples/${example}" --precision int4
done
```

INT4 examples use `rank=32`, `group_size=64`, INT4 residual packing,
activation shift, DeepCompressor-style low-rank search, and projection
smoothing search. NVFP4 examples use `rank=32`, `group_size=16`,
`weight_scale_dtypes=(None, "sfp8_e4m3_nan")`, no activation shift, and the
same search/smoothing flow. For Flux and PixArt NVFP4, extra norm/AdaLN linear
weights are exported as target-level INT4 weight-only overrides to mirror the
upstream precision overlay.

The default output path is
`outputs/checkpoints/svdq-<precision>_r32-<model>.safetensors`; calibration
root input caches and artifact caches are stored under
`outputs/calibration/<model>/<precision>/...` unless `--cache-dir` is supplied.
For lower peak VRAM in the upstream Diffusers examples, combine pipeline CPU
offload with per-target quantization offload, for example
`--pipeline-offload model --offload-model --compute-device cuda`. Calibration
captures remain CPU-backed, and only the active scope or target is moved to the
compute device while it is being replayed or quantized.

## Adapting Target Configs

The `*_target_config()` functions in the corresponding task-folder model files
are meant to be copied and edited for new model architectures. The core
question is not "is this model Flux or PixArt?", but "which modules should
become each exported runtime projection?" For new architectures, start with a
small class-only selector using `CalibrationScopeRule(module_classes=...)` and
`TargetRule(scope_module_classes=..., module_classes=...)`, then make the export
names and grouped projections explicit as the runtime format requires.

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

By default, each scope replays from the previous scope output after the first
root replay. This is the intended path for sequential transformer stacks. Set
`use_prev_scope_outputs=False` when scopes are independent branches or when the
next scope cannot consume the previous scope output without a custom
`prev_output_transform` or `prev_replay_transform`.

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

For a supported inspection workflow, use `inspect_target_config()` or pass
`--inspect-config` to a full example script. The structured report and smaller
configuration recipes are documented in [configuration.md](configuration.md).
