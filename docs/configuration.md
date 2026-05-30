# Configuration Guide

`diffuse_compressor` keeps model-specific structure in Python dataclass
configs. A target config answers which model modules become runtime projection
tensors, which modules stay unquantized, and how calibration replay is scoped.

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
