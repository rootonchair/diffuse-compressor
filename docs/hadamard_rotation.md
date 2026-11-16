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

- **nunchaku-lite:** inference requires the nunchaku-lite W4A4 kernel built
  with `hadamard=True`, which is CUDA only. The kernel's fp16/bf16 butterfly
  FWHT matches the fp32 reference up to half-precision rounding.
- **torch-dequant:** the runtime in this repository reads the per-target
  `rotation` metadata and undoes it. Rotated checkpoints can still be checked
  on CPU or MPS.

## Checkpoint Flag

The transform is stored as data, not named. Each rotated target records
`"rotation": {"type": "block_matrix", "block_size": 64, "shared": true,
"generator": "hadamard"}` and stores the blocks in a float32
`<export_name>.input_transform` tensor (`[64, 64]` shared, or `[K/64, 64, 64]`
with one block per 64 channels). Loaders and kernels must apply the stored
matrix. `generator` is only a hint. A kernel with a fast path for a known matrix,
such as today's FWHT for `hadamard`, may use it, but the tensor is
authoritative. Any orthogonal block matrix is valid, so a better rotation method
needs no format change. Checkpoints written before this change with
`{"type": "hadamard"}` still load.

Checkpoints with at least one rotated target declare
`runtime_manifest.nunchaku_format_version = 2`. Unrotated exports stay at 1.
A loader that only supports version 1 therefore refuses a rotated checkpoint
instead of loading it without the transform. Pre-release checkpoints that
carry `{"type": "hadamard"}` without an `input_transform` tensor are not
supported. Re-export them.

The metadata appears in three places:

- In the sidecar config's target entry. This is per target.
- In `runtime_manifest.targets[*].op_options.rotation`. This is per target,
  and is where the kernel flag is selected.
- In `quantization_config.rotation` in the safetensors metadata. This is a
  model-level summary for loaders that do not read the manifest.

Unrotated exports omit the key everywhere. The manifest v1 parser in
nunchaku-lite ignores unknown `op_options`. A loader without rotation support
would therefore load a rotated checkpoint without the rotation and give wrong
outputs instead of failing. Loaders must check this key.
