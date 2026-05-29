# Checkpoint Config Schema

`diffuse_compressor` writes quantized tensors to a safetensors checkpoint and
writes package-level config metadata next to it:

```text
model.safetensors
model.config.yaml
```

The config content is JSON formatted with a `.yaml` extension, so it remains
YAML-compatible without adding a required YAML dependency.

Safetensors checkpoint metadata intentionally keeps only compatibility fields.
Every exported checkpoint stores `method`, `rank`, `weight`, and `activation`
under `quantization_config`. When a generic Nunchaku Lite runtime manifest is
available, the same object also contains `runtime_manifest`:

```json
{
  "quantization_config": {
    "method": "svdquant",
    "rank": 32,
    "weight": {
      "dtype": "fp4_e2m1_all",
      "group_size": 16,
      "scale_dtypes": [
        null,
        "sfp8_e4m3_nan"
      ]
    },
    "activation": {
      "dtype": "int4",
      "scale_dtypes": [
        "sfp8_e4m3_nan"
      ],
      "enabled": true
    },
    "runtime_manifest": {
      "schema": "nunchaku_lite.runtime_manifest",
      "version": 1
    }
  }
}
```

Safetensors stores metadata values as strings, so the actual
`quantization_config` metadata value is the JSON-encoded string form of the
object above.

Detailed target metadata, structural patches, calibration activation shifts,
and cache provenance live in the adjacent config. Torch-dequant uses that
config instead of safetensors metadata.

## Config Shape

```json
{
  "method": "svdquant",
  "rank": 32,
  "weight": {
    "dtype": "int4",
    "group_size": 64,
    "scale_dtypes": [null, null]
  },
  "activation": {
    "dtype": "int4",
    "scale_dtypes": [null],
    "enabled": true
  },
  "targets": [],
  "structural_patches": [],
  "calibration": {
    "activation_shifts": {}
  },
  "quantization": {},
  "artifact_cache": {}
}
```

Required config fields:

- `method`: quantization method name, currently `"svdquant"`.
- `rank`: default low-rank branch rank.
- `weight`: default residual weight quantization settings.
- `activation`: default activation quantization settings.
- `targets`: ordered target metadata entries for torch-dequant.
- `structural_patches`: package-native rewrites to replay before resolving
  target module paths.

Optional fields:

- `calibration.activation_shifts`: activation shifts needed by torch-dequant.
- `quantization`: execution choices such as `compute_device` and
  `offload_model`.
- `artifact_cache`: local cache provenance for in-memory/debug use.

## Target Entries

Each `targets` item describes one exported target:

```json
{
  "name": "blocks.0.attn.qkv",
  "export_name": "blocks.0.attn.qkv",
  "modules": ["blocks.0.attn.q", "blocks.0.attn.k", "blocks.0.attn.v"],
  "roles": ["q", "k", "v"],
  "precision": "int4",
  "group_size": 64,
  "export_bias": "auto",
  "weight_layout": {
    "name": "svdq"
  },
  "weight_scale_layout": "logical",
  "runtime_tensor_layout": "logical",
  "activation_quant": {
    "enabled": true
  }
}
```

The exported tensor keys use `export_name` as their prefix, for example
`<export_name>.qweight`, `<export_name>.wscales`,
`<export_name>.smooth_factor`, `<export_name>.proj_down`, and
`<export_name>.proj_up`.

## Structural Patches

`structural_patches` is an ordered list of package-native rewrites:

```json
[
  {
    "type": "split_linear",
    "module": "single_transformer_blocks.*.proj_out",
    "args": {
      "splits": ["out_features"]
    }
  }
]
```

Supported config patch types:

- `split_linear`
- `split_linear_output`
- `split_conv`

These are the names accepted by `prepare_model()`. The Nunchaku Lite runtime
manifest uses its own ABI names, for example `split_linear_input`.

## Calibration Metadata

Exported calibration metadata only keeps activation shift values:

```json
{
  "activation_shifts": {
    "blocks.0.ff.net.0.proj": 1.25
  }
}
```

Other calibration execution details such as cache paths, sample counts,
captured target names, data-loader settings, and RAM limits are not written to
the exported config.

## Runtime Manifest

This config is not the Nunchaku Lite runtime ABI. Generic Nunchaku Lite loading
uses `quantization_config.runtime_manifest` inside safetensors metadata when
that manifest is present.

See [`nunchaku_lite_manifest_v1.md`](nunchaku_lite_manifest_v1.md) for the
runtime manifest schema.
