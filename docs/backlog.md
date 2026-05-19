# Backlog

## High Priority

### Add torch-dequant support for Nunchaku-packed SVDQ weights

New NVFP4/SVDQ exports may write targets with
`runtime_tensor_layout="nunchaku_packed"`. The current `torch-dequant`
runtime rejects those targets, so newly exported Nunchaku-compatible
checkpoints cannot use the PyTorch dequantization path for validation,
evaluation fallback, or checkpoint debugging.

Acceptance criteria:

- `runtime="torch-dequant"` can load checkpoints containing
  `runtime_tensor_layout="nunchaku_packed"` SVDQ targets.
- Packed `qweight` is unpacked and dequantized according to the Nunchaku W4A4
  tensor ABI.
- Packed low-rank `proj_down` and `proj_up` tensors are reconstructed correctly.
- `wscales`, `wcscales`, `wtscale`, `smooth_factor`, bias, and activation-shift
  behavior are preserved.
- Existing logical-layout `torch-dequant` checkpoints keep working unchanged.
- Tests cover a packed export from `AlignedModel` and compare reconstructed
  weights against a logical/reference reconstruction within expected tolerance.
