# Backlog

## Open Source Readiness And Maintainability

- Add open-source project metadata and process files: `LICENSE`,
  `CONTRIBUTING.md`, `SECURITY.md`, changelog/release notes, and CI workflow.
- Harden `.pt` cache loading. Current artifact and calibration caches use
  `torch.load(..., weights_only=False)`, which should not be used on untrusted
  files. Prefer safer formats or explicit trust boundaries in docs and APIs.

## DeepCompressor SVDQuant Parity

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
