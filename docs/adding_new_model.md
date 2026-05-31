# Adding A New Model

This guide shows how to adapt `diffuse_compressor` to a new diffusion model
architecture. The library core is model-agnostic; model structure belongs in a
user-side `TargetConfig`, usually copied from the closest file under
`examples/text_to_image/` or `examples/image_to_image/`.

The goal is to answer three questions before running calibration:

- Which modules become quantized runtime projection tensors?
- Which structural rewrites are needed before those modules can be selected?
- Which repeated blocks should be replayed as calibration scopes?

## 1. Pick A Starting Point

Start from the closest existing example:

- Flux-like text-to-image transformer: `examples/text_to_image/quantize_flux1_schnell.py`
- Flux.2-style grouped attention: `examples/text_to_image/quantize_flux2_klein_4b.py`
- PixArt or Sana-style transformer: `examples/text_to_image/quantize_pixart_sigma.py`
- Image-edit pipeline with image inputs: `examples/image_to_image/quantize_longcat_image_edit.py`

Keep model-specific code in the example or downstream project. Move code into
`src/diffuse_compressor/` only when it is genuinely architecture-independent.

## 2. Inspect The Module Tree

Load the dense model, then print the modules that could become quantization
targets:

```python
for name, module in model.named_modules():
    class_name = module.__class__.__name__
    if class_name in {"Linear", "Conv2d"}:
        print(name, module)
```

Look for repeated block paths, attention projections, MLP projections, fused
linears, pointwise convolutions, norm modulation linears, and final heads that
should stay unquantized.

## 3. Start With A Minimal Target Config

Begin with a broad class scan scoped to one repeated block class. This gives a
complete target list quickly and makes inspection useful before you refine
export names:

```python
import torch

from diffuse_compressor import CalibrationScopeRule, TargetConfig, TargetRule


def my_model_target_config() -> TargetConfig:
    return TargetConfig(
        calibration_scopes=[
            CalibrationScopeRule(module_classes=MyTransformerBlock),
        ],
        targets=[
            TargetRule(
                scope_module_classes=MyTransformerBlock,
                module_classes=torch.nn.Linear,
            ),
        ],
    )
```

Use `kind="conv"` only for pointwise `Conv2d` projector modules with
`kernel_size=(1, 1)` and `groups=1`. Leave depthwise and spatial convolutions
out of the target list unless a dedicated quantization path is added.

## 4. Refine Targets For Runtime Export

Once the broad scan is correct, decide whether the checkpoint must load through
the generic Nunchaku Lite manifest path. That decision controls how strict each
`TargetRule` must be.

For manifest-compatible exports, write one rule per loadable linear module
after patches:

```python
TargetRule(
    modules=["blocks.*.attn.q"],
    export_name="blocks.{0}.attn.q",
    quant=SvdqTargetQuant(weight_layout=NunchakuSvdqLayout()),
)
```

This works because every expanded target has exactly one source module, and
the checkpoint prefix equals the patched module path:
`blocks.0.attn.q -> blocks.0.attn.q`.

The same rule applies to standalone W4A16 linear targets, such as norm
modulation or AdaLN projection modules. Keep them as one-module targets and keep
`export_name` equal to the module path:

These modules should be quantized as standalone W4A16 weights instead of normal
SVDQuant projection targets. In diffusion transformers, they usually feed
scale/shift/gate parameters, not attention or MLP projections. They do not get
activation quantization, smoothing, or a low-rank branch.

```python
TargetRule(
    modules=["blocks.*.norm_modulation"],
    export_name="blocks.{0}.norm_modulation",
    quant=W4A16TargetQuant(layout=AwqW4A16Layout()),
)
```

Use `AwqW4A16Layout()` for a plain standalone W4A16 linear. Use
`AdaNormAwqW4A16Layout(splits=3)` or `AdaNormAwqW4A16Layout(splits=6)` when the
runtime expects the AdaNorm modulation tensor to be split and interleaved in the
DeepCompressor/Nunchaku format. `W4A16TargetQuant` keeps the required behavior
together: INT4 residual weights, 64-wide groups, rank 0, no smoothing, no
activation quantization, no activation shifting, and no shared low-rank branch.

These `TargetRule` shapes disable the generic manifest even though checkpoint
export may still succeed:

- `kind="conv"` because manifest v1 supports only linear runtime targets.
- A grouped path rule with multiple source modules:

```python
TargetRule(
    modules=["blocks.*.attn.q", "blocks.*.attn.k", "blocks.*.attn.v"],
    export_name="blocks.{0}.attn.qkv",
    roles=("q", "k", "v"),
)
```

- A callable group that returns multiple child modules:

```python
TargetRule(
    parent_module_classes=Attention,
    member_selector=lambda attn: {"q": attn.to_q, "k": attn.to_k, "v": attn.to_v},
    export_name="{parent_path}.qkv",
)
```

- A synthetic export prefix that differs from the source module path:

```python
TargetRule(
    modules=["blocks.*.attn.q"],
    export_name="blocks.{0}.q_proj",
)
```

  The expanded target uses checkpoint prefix `blocks.0.q_proj`, but the source
  module is `blocks.0.attn.q`; manifest v1 requires those names to match.
- `SvdqTargetQuant(weight_layout=NaiveSvdqLayout())` or any SVDQ target that
  exports logical tensors instead of Nunchaku-packed tensors.
- A selected module that does not expose integer `in_features` and
  `out_features`.

Use grouped or synthetic `TargetRule`s only when the checkpoint will be loaded
by a model-specific runtime adapter, by a debug/dequant workflow, or by a future
manifest schema that understands the grouping.

Additional target rules:

- Use one `TargetRule` for one exported tensor family.
- Group modules only when they consume the same activation tensor and should
  share one low-rank branch.
- Do not group projections that consume different inputs. Cross-attention Q
  usually consumes hidden states, while K/V consume encoder states.
- Use `SkipRule` to remove final heads, debug heads, and other non-runtime
  modules from broad scans.
- Use `NunchakuSvdqLayout()` when you want export to fail if the target cannot
  be packed into the Nunchaku SVDQ ABI.

## 5. Add Patches Only When Needed

Use `PatchRule`s when targetable child modules do not exist yet. Common cases
are fused QKV projections, fused QKV+MLP projections, and fused output
projections:

```python
PatchRule(
    type="split_linear_output",
    module="single_blocks.*.attn.to_qkv_mlp",
    args={"splits": [hidden_size * 3]},
)
```

Patch first, then target the exposed child modules. Avoid adding
architecture-specific patch behavior to the library core.

## 6. Add Calibration Scopes

Calibration scopes control replay granularity and memory. Start with one scope
per repeated block class:

```python
CalibrationScopeRule(module_classes=MyTransformerBlock)
```

Class-based scopes use the matched module path as the scope name and replay
module. This is the least brittle starting point when porting a new
architecture.

For transformer stacks where the next block cannot directly consume the
previous scope output, keep the class selector and add a transform that adapts
the previous replay record to the block forward signature:

```python
CalibrationScopeRule(
    module_classes=MyTransformerBlock,
    prev_replay_transform=my_block_prev_replay_transform,
)
```

Add path-template `eval_module`, `capture_modules`, or `cache_aliases` only
after the class-based scope inspects correctly and a grouped target needs a
specific child-module replay cache.

Sequential transformer stacks usually use the previous scope output as the next
scope input. If scopes are independent branches, set
`use_prev_scope_outputs=False` or provide a `prev_output_transform` /
`prev_replay_transform` that matches the block forward signature.

## 7. Inspect Before Quantizing

Inspect the config against a real dense model before running calibration:

```python
from diffuse_compressor import inspect_target_config

report = inspect_target_config(model, my_model_target_config())
print(report.format_text())
assert report.ok
```

For a full example script, add a `--inspect-config` path that loads the model,
prints the report, and exits before calibration or quantization.

The expected report has:

- no missing module patterns;
- no duplicate `export_name`s;
- no skipped modules selected by explicit rules;
- grouped targets ordered the same way the runtime expects checkpoint tensors;
- calibration scopes that cover the target modules they are meant to replay.

For lower-level debugging, collect targets directly:

```python
from diffuse_compressor import collect_quant_targets, prepare_model

target_config = my_model_target_config()
prepare_model(model, target_config.patches)
targets = collect_quant_targets(model, target_config)
for target in targets:
    print(target.export_name, target.module_names, target.kind)
```

## 8. Quantize, Evaluate, And Infer

After inspection passes, run a tiny calibration first:

```bash
python examples/text_to_image/quantize_my_model.py \
  --precision int4 \
  --num-samples 2 \
  --cache-mode disabled
```

Replace `quantize_my_model.py` with the copied example script for your model.
Then increase calibration samples, evaluate original and quantized outputs, and
run a single-prompt Nunchaku Lite inference check. The task guides show the
end-to-end command shape:

- [Text-to-image end-to-end guide](text_to_image_end_to_end_guide.md)
- [Image-to-image end-to-end guide](image_to_image_end_to_end_guide.md)

If the checkpoint does not load in the runtime, inspect exported key names and
layout choices before changing quantization settings. Runtime naming and tensor
layout mismatches are more common than numeric failures during a first port.
When a manifest is omitted, check the sidecar checkpoint config for
`runtime_manifest_diagnostics`; see
[checkpoint_metadata.md](checkpoint_metadata.md) and
[nunchaku_lite_manifest_v1.md](nunchaku_lite_manifest_v1.md).
