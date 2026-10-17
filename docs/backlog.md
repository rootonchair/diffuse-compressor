# Backlog

## Open Source Readiness And Maintainability

- Add open-source project metadata and process files: `LICENSE`,
  `CONTRIBUTING.md`, `SECURITY.md`, changelog/release notes, and CI workflow.
- Harden `.pt` cache loading. Current artifact and calibration caches use
  `torch.load(..., weights_only=False)`, which should not be used on untrusted
  files. Prefer safer formats or explicit trust boundaries in docs and APIs.
- Guard image calibration cache reuse against partial output sets. If an
  interrupted run leaves fewer generated calibration images than `--num-samples`,
  `--cache-mode reuse` can still replay existing `inputs/caches/*.pt` records
  and proceed with incomplete calibration data. Later fix: detect incomplete
  `inputs/samples/*.png` output sets, log the mismatch, and refresh/regenerate
  the calibration input cache before quantization.

## Quantization Memory

- Bound smoothing-candidate scoring by row count, not only by weight size.
  `_score_smoothing_candidates` (`methods/svdquant/smoothing.py`) materializes
  `actual = inputs @ folded2d` as float32 `[rows, candidates, out_features]`, plus
  same-sized `actual - expected` and `.pow(2)` temporaries. `_chunk_candidates`
  caps a candidate batch only by the weight-shaped tensors (2 GiB budget from
  `777ebbe`), so peak VRAM grows linearly with `max_rows_per_target`. On
  Qwen-Image 2.1 NVFP4 r64 with 32768 rows, a 12288x4096 projection scores about
  10 candidates per batch: about 16 GB per activation-sized tensor and about
  48 GB at peak, versus about 2 GB per tensor at 4096 rows. The process reached
  92.6 of 97.9 GB without running out of memory. Fix: include
  `rows * out_features * 4` bytes per candidate in the chunk budget; candidate
  selection is unchanged because each candidate is scored independently.
  Workaround: pass `--sample-batch-size 4096` to score rows in partitions.
- Release pipeline components that calibration no longer needs. Example scripts
  load the whole pipeline onto the GPU, and the text encoder (about 16 GB for
  Qwen-Image 2.1) stays resident during replay and quantization even when the
  calibration inputs are reused from the disk cache.

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
