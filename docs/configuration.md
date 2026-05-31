# Configuration Guide

`diffuse_compressor` keeps model-specific structure in Python dataclass
configs. A target config answers which model modules become runtime projection
tensors, which modules stay unquantized, and how calibration replay is scoped.
For full model example defaults and a longer target-adaptation walkthrough, see
[examples.md](examples.md). For DeepCompressor setting equivalents, see
[deepcompressor_mapping.md](deepcompressor_mapping.md).

Before running a full quantization job, inspect the config against a real model:

```python
from diffuse_compressor import inspect_target_config

report = inspect_target_config(model, target_config)
print(report.format_text())
assert report.ok
```

Example quantization scripts also support:

```bash
python examples/text_to_image/quantize_flux1_schnell.py --inspect-config
```

This loads the pipeline/model, prints concrete targets and calibration scopes,
then exits before calibration or quantization.

## Quantization Spec

`DiffusionQuantSpec` describes the quantization method and numeric settings:

```python
DiffusionQuantSpec(
    method="svdquant",
    precision="int4",
    rank=32,
    group_size=64,
    compute_device=None,
    offload_model=False,
)
```

## Patch Rules

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

## Recipe Map

All recipes are local-only and use tiny `torch.nn.Module` models. They do not
download Diffusers pipelines, run calibration, quantize, or export checkpoints.

| Recipe | Use it for | Related sections |
| --- | --- | --- |
| `examples/recipes/path_grouping.py` | Path patterns, wildcard captures, grouped QKV, export-name templates | Target Rules |
| `examples/recipes/class_scan_and_skips.py` | `module_classes`, `scope_module_classes`, broad scans, `SkipRule`, `unquantized_patterns` | Target Rules, Skips And Overrides |
| `examples/recipes/callable_grouping.py` | `parent_module_classes`, `member_selector`, `{parent_path}` export names | Target Rules |
| `examples/recipes/calibration_scopes.py` | Block calibration scopes, replay/eval modules, capture rules, cache aliases | Calibration Scopes |
| `examples/recipes/extra_weight_overrides.py` | Target-level precision/rank/smoothing overrides, AWQ and AdaNorm layouts | Skips And Overrides |
| `examples/recipes/inspect_target_config.py` | End-to-end sampler plus intentionally broken config diagnostics | Inspection Output |

## Target Rules

Use one `TargetRule` for one exported runtime tensor family.

Single projection:

```python
TargetRule(modules=["blocks.*.attn.out"], export_name="blocks.{0}.attn.out")
```

Grouped QKV projection:

```python
TargetRule(
    modules=["blocks.*.attn.q", "blocks.*.attn.k", "blocks.*.attn.v"],
    export_name="blocks.{0}.attn.qkv",
    roles=("q", "k", "v"),
)
```

Wildcard captures must line up across grouped module patterns. The `*` capture
is reused by `{0}` in `export_name`.

Without wildcards, multiple exact module paths form one group keyed by the
empty capture tuple. With mismatched wildcards, only captures present in every
pattern form groups; if there is no shared capture, target collection raises.

Runnable example: `examples/recipes/path_grouping.py`.

That recipe defines two repeated blocks and shows how one grouped QKV rule:

- matches `blocks.0.attn.to_q`, `to_k`, and `to_v` as one target;
- emits `blocks.0.attn.to_qkv` by formatting `{0}` with the block index;
- repeats the same grouping for `blocks.1`;
- adds separate output and MLP projection targets.

Class-based scan:

```python
TargetRule(scope_module_classes=Block, module_classes=nn.Linear)
```

This selects named child modules whose class matches `module_classes`, limited
to descendants of matching `scope_module_classes`.

Runnable example: `examples/recipes/class_scan_and_skips.py`.

That recipe uses a broad class scan to select all `nn.Linear` modules under
each `Block`, then uses skips so debug-only heads and the final output layer do
not become quantized targets.

Callable grouping:

```python
TargetRule(
    parent_module_classes=Attention,
    member_selector=lambda attn: {"q": attn.q, "k": attn.k, "v": attn.v},
    export_name="{parent_path}.qkv",
)
```

Use callable groups when child attributes are stable but path names are not.

Runnable example: `examples/recipes/callable_grouping.py`.

That recipe groups `query`, `key`, and `value` attributes from each matched
`Attention` parent and exports them as `{parent_path}.qkv`. This pattern is
useful when module paths vary but the parent class and child attributes are
stable.

Target-level export controls are available for runtime-specific tensor
contracts. `SvdqLayout()` keeps the backward-compatible auto SVDQ behavior,
`NaiveSvdqLayout()` forces the logical/torch-dequant-friendly SVDQ tensor
layout, and `NunchakuSvdqLayout()` requires the packed Nunchaku kernel ABI.
`weight_layout=AwqW4A16Layout()` exports a single INT4 linear target in the
Nunchaku Lite AWQ W4A16 extra-weight layout, while
`weight_layout=AdaNormAwqW4A16Layout(splits=3 or 6)` applies the
DeepCompressor AdaNorm W4A16 export transform. `export_bias="zero"` writes a
synthesized zero bias for biasless modules, which is useful when a runtime
expects split projections to expose separate bias tensors.

## Skips And Overrides

Use `SkipRule` to remove modules from broad class scans:

```python
TargetConfig(
    targets=[TargetRule(scope_module_classes=Block, module_classes=nn.Linear)],
    skips=[SkipRule(modules=["blocks.*.norm_linear"])],
)
```

Explicit path targets that select skipped modules still fail. This keeps skips
from hiding typos.

Runnable example: `examples/recipes/class_scan_and_skips.py`.

That recipe also sets `unquantized_patterns=["final.*"]`, so inspection reports
which state-dict keys would be preserved outside the quantized target set.

Use target-level overrides for runtime-specific tensor contracts:

Use `SvdqLayout()` for the default auto-selected SVDQ layout,
`NaiveSvdqLayout()` to force logical SVDQ tensors, and
`NunchakuSvdqLayout()` to require the packed Nunchaku SVDQ ABI.

```python
TargetRule(
    modules=["blocks.*.norm_modulation"],
    export_name="blocks.{0}.norm_modulation",
    precision="int4",
    group_size=64,
    rank=0,
    smooth=False,
    activation_quant=False,
    weight_layout=AwqW4A16Layout(),
)
```

Runnable example: `examples/recipes/extra_weight_overrides.py`.

That recipe shows two extra-weight patterns:

- plain W4A16-style export with `AwqW4A16Layout()`;
- AdaNorm modulation export with `AdaNormAwqW4A16Layout(splits=6)`.

Both use target-level overrides such as `rank=0`, `smooth=False`, and
`activation_quant=False` so the target behaves like an extra weight rather than
a normal SVDQuant projection.

## Calibration Scopes

Use `CalibrationScopeRule` to replay and clear calibration activations by block:

```python
CalibrationScopeRule(name="blocks.{0}", modules=["blocks.*"])
```

Targets under `blocks.0` are calibrated together, then cleared before `blocks.1`
is replayed. If no scopes are configured, each target gets its own scope.

Add replay/eval modules and capture aliases when grouped targets need shared
activation caches:

```python
CalibrationScopeRule(
    name="blocks.{0}",
    modules=["blocks.*"],
    eval_module="blocks.*.attn",
    replay_module="blocks.*",
    cache_aliases={
        "blocks.{0}.attn.k": "blocks.{0}.attn.q",
        "blocks.{0}.attn.v": "blocks.{0}.attn.q",
    },
    capture_modules=[
        CalibrationCaptureRule(
            name="blocks.{0}.attn_io",
            modules=["blocks.*.attn"],
            inputs=True,
            outputs=True,
            input_keys=("hidden_states",),
        )
    ],
)
```

Runnable example: `examples/recipes/calibration_scopes.py`.

That recipe demonstrates:

- one scope per repeated block;
- `replay_module` for the module rerun during calibration replay;
- `eval_module` for low-rank candidate scoring;
- `capture_modules` for named input/output caches;
- `cache_aliases` so grouped K/V targets can reuse the Q input cache when they
  share the same activation source.

Scope capture is keyed and model-agnostic. `input_keys` and `output_keys` select
positional keys such as `"arg0"` or keyword keys such as `"hidden_states"`.
`cache_aliases` lets grouped targets reuse another captured cache, which covers
QKV-style behavior without hardcoding attention architecture names.
`replay_arg_indices`, `replay_kwarg_keys`, and `replay_transform` filter or
rewrite eval-module replay inputs for complex blocks.
`prev_output_transform` and `prev_replay_transform` adapt previous-scope replay
when a block has multiple streams or invariant conditioning arguments.

## Inspection Output

`inspect_target_config()` returns `TargetConfigReport` with:

- `targets`: concrete target names, export names, module paths, roles, kinds,
  and target overrides.
- `calibration_scopes`: concrete scopes, assigned targets, replay/eval modules,
  captures, and cache aliases.
- `skipped_modules`: modules selected by `SkipRule`s.
- `unquantized_keys`: state-dict keys matched by explicit
  `unquantized_patterns`.
- `warnings`: unmatched rules or suspicious config shape.
- `errors`: failures from target collection or scope assignment.

Use `report.to_dict()` for structured output and `report.format_text()` for
CLI-friendly output.

Runnable example: `examples/recipes/inspect_target_config.py`.

That recipe prints several valid reports and one intentionally broken config.
The broken config demonstrates how diagnostics report unmatched target patterns,
unmatched calibration scopes, unmatched unquantized patterns, and collection
errors without running quantization.
