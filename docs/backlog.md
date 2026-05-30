# Backlog

## Open Source Readiness And Maintainability

- Add open-source project metadata and process files: `LICENSE`,
  `CONTRIBUTING.md`, `SECURITY.md`, changelog/release notes, and CI workflow.
- Harden `.pt` cache loading. Current artifact and calibration caches use
  `torch.load(..., weights_only=False)`, which should not be used on untrusted
  files. Prefer safer formats or explicit trust boundaries in docs and APIs.
- Add calibration cache manifests and invalidation keys that include model id,
  pipeline parameters, image size, dtype, step count, code/config version, and
  target config identity.
- Document or redesign in-place model mutation. `prepare_model()`,
  `quantize_and_export()`, and activation-shift calibration mutate the supplied
  model, which should be explicit in public docs and ideally paired with a
  copy/restore helper or non-mutating workflow.
- Make export ABI selection explicit. Nunchaku packed vs logical layout is
  currently shape-dependent; public configs should be able to require a layout
  and fail clearly when the ABI cannot be produced.
- Surface missing runtime manifests as warnings or errors. The exporter can
  silently omit `runtime_manifest` for unsupported/grouped targets, which makes
  runtime compatibility failures hard to diagnose.
- Replace the global singleton logging stream tee with a library-friendly
  logging design. Rewriting `sys.stdout` and `sys.stderr` is not thread-safe
  and is awkward inside notebooks, services, or larger applications.
- Improve public configuration ergonomics. Target and calibration configs are
  powerful but stringly typed; add schema docs, validation tools, inspection
  commands, and smaller recipe-style examples for common patterns.
- Clean local/untracked scripts before release and decide which shell scripts
  are supported examples versus local experiment launchers.

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
