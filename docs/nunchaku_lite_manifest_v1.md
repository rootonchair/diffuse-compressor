# Nunchaku Lite Runtime Manifest v1

This document is the contract between `diffuse_compressor` checkpoint exports
and `nunchaku_lite` runtime loading. The manifest is intended for
`nunchaku_lite` maintainers and coding agents implementing generic or
adapter-light runtime patching.

The manifest is not descriptive metadata only. If a checkpoint contains this
manifest, `diffuse_compressor` asserts that the checkpoint tensors already
conform to the declared Nunchaku Lite ABI:

```text
nunchaku_format_version + nunchaku_op + precision
=> tensor names, shapes, packing, scale keys, and runtime module contract
```

If `diffuse_compressor` cannot emit tensors matching that ABI, it must omit the
manifest or fail export with a clear error.

## Location

The manifest is stored inside safetensors metadata under:

```json
{
  "quantization_config": {
    "method": "svdquant",
    "rank": 32,
    "weight": {},
    "activation": {},
    "runtime_manifest": {}
  }
}
```

Detailed package-level torch-dequant metadata is not stored beside the manifest
in safetensors metadata. It lives in the checkpoint config documented in
`checkpoint_metadata.md`.

## Top-Level Schema

```json
{
  "schema": "nunchaku_lite.runtime_manifest",
  "version": 1,
  "component": "transformer",
  "nunchaku_format_version": 1,
  "producer": {
    "name": "diffuse_compressor",
    "version": "0.1.0"
  },
  "requirements": {
    "method": "svdquant",
    "precision": "fp4",
    "rank": 32,
    "weight_dtype": "fp4_e2m1_all",
    "activation_dtype": "int4"
  },
  "structural_patches": [],
  "targets": []
}
```

Required fields:

- `schema`: must be exactly `"nunchaku_lite.runtime_manifest"`.
- `version`: manifest schema version. This document defines version `1`.
- `component`: pipeline component patched by Nunchaku Lite, usually
  `"transformer"`.
- `nunchaku_format_version`: Nunchaku checkpoint ABI version. Version `1`
  means tensor layout is defined by Nunchaku Lite conventions for each op.
- `producer`: exporting tool identity.
- `requirements`: global quantization summary. `precision` and `weight_dtype`
  are the common effective target values, or `"mixed"` when target-level
  overrides produce more than one value. Runtime loaders must use each target's
  own `precision` for module and tensor ABI selection.
- `structural_patches`: ordered model rewrites to apply before replacement.
- `targets`: ordered runtime target declarations.

## Nunchaku Format Version 1

For `nunchaku_format_version = 1`, `nunchaku_lite` owns the exact checkpoint
tensor convention for every `(nunchaku_op, precision)` pair.

Examples of ABI-owned behavior:

- Logical vs Nunchaku-packed tensor layout.
- `qweight`, `wscales`, `wcscales`, `wtscale`, and `wzeros` key meanings.
- Scalar vs per-channel outer scales.
- Packed vs plain low-rank tensors.
- Packed vs plain bias and smooth tensors.
- Whether a runtime module expects `wzeros`.

Consumers must not infer these details from `precision` alone. `precision`
selects numeric quantization format; `nunchaku_op` and
`nunchaku_format_version` define the runtime tensor ABI.

## Structural Patches

`structural_patches` is an ordered list. Nunchaku Lite must apply these rewrites
before target replacement or shape validation.

Supported patch types in v1:

```json
{
  "type": "split_linear_output",
  "module": "single_transformer_blocks.*.attn.to_qkv_mlp_proj",
  "args": {
    "splits": [12288]
  }
}
```

```json
{
  "type": "split_linear_input",
  "module": "single_transformer_blocks.*.attn.to_out",
  "args": {
    "splits": [4096]
  }
}
```

`splits` values are JSON-safe integers or symbolic strings already understood
by the corresponding implementation. Current diffuse_compressor examples may
use `"out_features"` for Flux-style split-input projections.

Unsupported structural patch types must be rejected with an explicit error.

## Target Entries

Each `targets` entry declares one exported runtime target:

```json
{
  "name": "single_transformer_blocks.0.attn.to_qkv_mlp_proj.linears.0",
  "checkpoint_prefix": "single_transformer_blocks.0.attn.qkv_proj",
  "source_modules": [
    "single_transformer_blocks.0.attn.to_qkv_mlp_proj.linears.0"
  ],
  "roles": [],
  "kind": "linear",
  "nunchaku_op": "svdq_w4a4",
  "precision": "fp4",
  "group_size": 16,
  "rank": 32,
  "has_bias": true,
  "op_options": {
    "outer_scale_splits": [4096, 4096, 4096]
  },
  "activation": {}
}
```

Required target fields:

- `checkpoint_prefix`: prefix for tensors in the safetensors file.
- `source_modules`: model module paths that produced the target.
- `roles`: semantic grouping roles such as `["q", "k", "v"]`.
- `kind`: `"linear"` or `"conv"`. Nunchaku Lite v1 may reject unsupported
  kinds.
- `nunchaku_op`: runtime op contract.
- `precision`: target precision, `"int4"` or `"fp4"`.
- `group_size`: target quantization group size.
- `rank`: target low-rank rank.
- `has_bias`: whether the runtime target has a bias tensor.
- `op_options`: op-specific options.
- `activation`: target activation quantization metadata.

Supported `nunchaku_op` values in v1:

- `svdq_w4a4`: SVDQuant W4A4 runtime module.
- `awq_w4a16`: AWQ W4A16 runtime module.
- `adanorm_awq_w4a16`: AdaNorm modulation AWQ W4A16 runtime module.

Op-specific options:

- `svdq_w4a4`: `outer_scale_splits`, optional list of output-row chunks for
  fused NVFP4 outer scales.
- `adanorm_awq_w4a16`: `adanorm_splits`, required split count.
- `awq_w4a16`: no v1 options.

For the generic Nunchaku Lite manifest adapter, each v1 target must name one
loadable module after structural patches: `checkpoint_prefix` must equal the
only item in `source_modules`. Grouped projections with synthetic checkpoint
prefixes, such as fused QKV exports, require an architecture adapter or a
future manifest schema that explicitly defines grouping semantics.

## Nunchaku Lite Loader Obligations

A compliant loader must:

1. Reject unsupported `schema`, `version`, `nunchaku_format_version`, or
   `nunchaku_op`.
2. Apply `structural_patches` before module replacement.
3. Select runtime modules and expected tensor keys from:
   `nunchaku_format_version + nunchaku_op + precision`.
4. Use `checkpoint_prefix` as the state-dict prefix.
5. Validate missing keys, unexpected keys, tensor shapes, and dtypes before or
   during final load.
6. Use architecture adapters or named forward templates where forward semantics
   cannot be inferred from this manifest.

This manifest does not encode arbitrary forward graphs, attention semantics,
rotary embedding packing, or pipeline-level behavior. It provides the target
and tensor ABI contract needed for adapter-light patching.

## Diffuse Compressor Exporter Obligations

When emitting `runtime_manifest`, `diffuse_compressor` must:

1. Export tensors in the declared Nunchaku Lite ABI.
2. Map target layouts to `nunchaku_op` values:
   - `SvdqLayout` and `NunchakuSvdqLayout` -> `svdq_w4a4`
   - `AwqW4A16Layout` -> `awq_w4a16`
   - `AdaNormAwqW4A16Layout` -> `adanorm_awq_w4a16`
3. Put layout-specific hints in `op_options`, not in separate physical layout
   fields.
4. Omit the manifest when an export contains logical or otherwise
   non-Nunchaku-ABI tensors.
5. Fail export clearly when a target explicitly requests a Nunchaku ABI layout
   but cannot be packed into that ABI.
6. Omit the manifest for grouped or synthetic targets that the generic manifest
   adapter cannot resolve as module paths after structural patches.
