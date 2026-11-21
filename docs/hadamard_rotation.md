# Hadamard Rotation

`RotationSpec` enables an optional online block-wise Walsh-Hadamard rotation of
the SVDQuant W4A4 residual. Rotation spreads activation and weight outliers
across each 64-channel block before 4-bit rounding. It is off by default. With
it off, the quantization flow and exported checkpoints are identical to runs
without the setting.

## Enabling It

```python
from diffuse_compressor import DiffusionQuantSpec, RotationSpec

spec = DiffusionQuantSpec(
    precision="int4",
    group_size=64,
    rotation=RotationSpec(),  # type="hadamard", block_size=64
)
```

Example scripts opt in with this one field. Rotation changes the error surface,
so re-search smoothing alphas and low-rank candidates with rotation on. Do not
reuse settings tuned without it.

## Basis Convention

For a linear layer with weight `W [N, K]`, smoothing `s`, and low-rank branch
`L1` (down) and `L2` (up), the runtime computes:

```text
x_s = x / s
y   = x_s @ L1^T @ L2^T + Q(x_s @ H) @ Q(R @ H)^T + b
R   = W @ diag(s) - L2 @ L1
H   = blockdiag(H64, ..., H64),   H64 = scipy.linalg.hadamard(64) / 8
```

`H64` uses Sylvester ordering. It is symmetric and orthonormal, so `H^-1 = H`.

- Rotated: the residual `R` along its input dimension, and the smoothed
  activation that enters the 4-bit quantizer.
- Not rotated: the raw input, the smoothing factors, the low-rank branch and
  its input (LoRA-down runs on the unrotated `x_s`), the bias, and the output.

Every step that rounds the residual or its activations does so in the rotated
basis: smoothing-candidate scoring, low-rank candidate scoring (including eval
replay), activation range calibration, the GPTQ Hessian, and the final packing.
The code applies this through one input-basis transform,
`diffuse_compressor.methods.svdquant.rotation.InputRotation`. Its default, `IDENTITY`, is a
no-op.

## Constraints

- Every SVDQuant target must be a linear layer with `in_features` divisible
  by 64. Quantization raises `ValueError` otherwise. Layers are never skipped
  silently, because a skipped layer would need its own runtime flag. Quantize
  incompatible layers in a separate run or with AWQ W4A16, which does not use
  the W4A4 activation quantizer and is never rotated.
- Rotation requires a signed activation path. Targets with
  `shift_activations=True` quantize their (post-GELU) input unsigned, and in
  nunchaku-lite that input is quantized inside the previous GEMM's epilogue,
  which never applies the transform. Rotating them would silently give wrong
  outputs, so quantization fails before calibration and names those targets.
  Opt them out with `SvdqTargetQuant(rotation=False)`. They are exported
  unrotated while every other target keeps the rotation. On FLUX.1-dev the
  text-stream `ff_context.net.2` layers lose up to 30 dB with rotation, so
  per-target opt-out is the intended setting for those layers anyway.
- Only `block_size=64` with Sylvester ordering is supported. Random sign flips
  (`D @ H`), rotating the LoRA-down input, fused-GLU activation quantization,
  and Conv2d targets are not supported.
- Rotation is validated for INT4 only, and NVFP4 is refused by default. With
  INT4's `group_size=64`, a quantization group is wider than an outlier's
  neighbourhood, so spreading the outlier helps. NVFP4's 16-wide micro-scales
  already isolate outliers, and H64 spreads each one across four scale groups,
  which is expected to raise the error. `precision="fp4"` with rotation, at the
  spec level or as a per-target override, raises `ValueError` unless you pass
  `RotationSpec(allow_fp4=True)`. The kernel supports it, so set that only to
  experiment, and measure before relying on it.

## Inference

The exported residual is `Q(R @ H)`, so the checkpoint is not interchangeable
with an unrotated one.

- **nunchaku-lite:** inference uses nunchaku-lite's separate Hadamard quantize
  op, `quantize_w4a4_act_hadamard_fuse_lora` (CUDA only; rootonchair/nunchaku-lite#17).
  MPS, MLX, and CUDA builds without that op refuse rotated checkpoints at load.
  The kernel's fp16/bf16 butterfly FWHT matches the fp32 reference up to
  half-precision rounding.
- **torch-dequant:** the runtime in this repository reads the per-target
  `rotation` metadata and undoes it. Rotated checkpoints can still be checked
  on CPU or MPS.

## Checkpoint Flag

Rotation is declared by name and no matrix is stored. `"hadamard"` means
exactly one transform: each contiguous block of 64 input channels of
`x / smooth`, in logical channel order, is multiplied by the orthonormal
Sylvester matrix `H64 / 8`. The low-rank branch reads the unrotated input, and
the residual is packed as `Q(R @ blockdiag(H64 / 8))`.
`nunchaku_lite.ops.hadamard_rotate` is the reference transform, and
`tests/test_rotation.py` checks the exported packing against it. A different
transform needs a new name.

Checkpoints with at least one rotated target declare
`runtime_manifest.nunchaku_format_version = 2`. Unrotated exports stay at 1.
A loader that only supports version 1 therefore refuses a rotated checkpoint
instead of loading it without the transform. The stored manifest is required:
older nunchaku-lite versions ignore the model-level declaration below, so the
exporter raises instead of writing a Nunchaku-packed rotated checkpoint whose
targets manifest v1 cannot describe (grouped targets, or checkpoint prefixes
that differ from the source module path). Logical-layout checkpoints, which
only the torch-dequant runtime loads, are exempt.

The declaration appears in three places:

- In the sidecar config's target entry, as `"rotation": {"type": "hadamard"}`.
- In `runtime_manifest.targets[*].op_options.rotation`, as
  `{"type": "hadamard"}` with no other fields.
- In `quantization_config.input_rotation` in the safetensors metadata, as
  `{"format_version": 2, "type": "hadamard", "targets": [...]}`. `targets`
  lists the checkpoint prefix (`<prefix>.qweight`) of every rotated layer. It
  is optional for nunchaku-lite, which takes the rotated targets from the
  manifest and refuses the checkpoint unless `targets` equals them.

Pre-release checkpoints, with `{"type": "block_matrix", ...}` metadata and
`<export_name>.input_transform` tensors, are refused by both nunchaku-lite and
the torch-dequant runtime. Their packed weights are the same, so rewriting the
metadata and dropping the tensors converts them without re-quantizing.

Unrotated exports omit the key everywhere. The manifest v1 parser in
nunchaku-lite ignores unknown `op_options`. A loader without rotation support
would therefore load a rotated checkpoint without the rotation and give wrong
outputs instead of failing. Loaders must check this key.
