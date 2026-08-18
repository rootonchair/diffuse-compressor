"""Quantize Krea-2-Turbo, chaining block replay so calibration does not grow with depth.

The generic scanner gives every scope ``use_prev_scope_outputs=False``, so each
block replays the model *from the root* up to itself. On a 28-block model that is
O(N^2) replay work, and under the sequential CPU offload Krea's 26.3 GB
transformer requires, every replay streams the whole model. Measured on the INT4
pass, per-target cost rose from ~500 s in block 26 to ~2280 s in block 27.

This script keeps the generic scanner's target set byte-for-byte -- so the
checkpoint keeps its 1:1 module-to-prefix mapping, an empty ``structural_patches``
list, and a clean runtime manifest -- and only replaces the calibration scopes
with chained ones. Each scope then replays a single block, taking the previous
block's output as its input.

Usage mirrors ``quantize_hf.py``:

    python examples/text_to_image/quantize_krea2_turbo.py --precision nvfp4 \
        --num-samples 32 --steps 8 --guidance-scale 0.0 \
        --pipeline-offload sequential --compute-device cuda --offload-model
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diffuse_compressor import (
    CalibrationScopeRule,
    CalibrationSpec,
    ExportSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    TargetConfig,
    inspect_target_config,
    quantize_and_export,
)
from examples.text_to_image.quantize_hf import (
    _forward_fn,
    _slug,
    build_parser,
    discover_denoiser,
    load_auto_pipeline,
    scan_linear_targets,
)
from examples.text_to_image.utils import (
    batched_samples,
    save_diffusers_images,
    standard_prompt_records,
    svdquant_spec,
)

MODEL_ID = "krea/Krea-2-Turbo"


def _krea2_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Feed the previous block's output in as this block's ``hidden_states``.

    Both Krea2 block types return a single tensor that is exactly the next
    block's ``hidden_states``. The remaining inputs (``temb``,
    ``image_rotary_emb``, ``attention_mask``) are shared across blocks, so they
    carry over from the previous replay unchanged.
    """

    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = replay.output
        return (), kwargs
    args = list(replay.args)
    if not args:
        raise TypeError("Krea2 block replay must include hidden_states")
    args[0] = replay.output
    return tuple(args), {}


def krea2_block_stacks() -> tuple[tuple[str, type], ...]:
    """Return each repeated block stack and its block class, in forward order."""

    from diffusers.models.transformers.transformer_krea2 import (
        Krea2TextFusionBlock,
        Krea2TransformerBlock,
    )

    return (
        ("text_fusion.layerwise_blocks", Krea2TextFusionBlock),
        ("text_fusion.refiner_blocks", Krea2TextFusionBlock),
        ("transformer_blocks", Krea2TransformerBlock),
    )


def krea2_calibration_scopes() -> tuple[CalibrationScopeRule, ...]:
    """Chain replay within each stack, restarting at every stack head.

    The three stacks carry different tensors, so each one's first block must not
    consume the previous stack's output.
    """

    scopes: list[CalibrationScopeRule] = []
    for prefix, block_class in krea2_block_stacks():
        scopes.append(
            CalibrationScopeRule(
                modules=(f"{prefix}.0",),
                module_classes=block_class,
                use_prev_scope_outputs=False,
            )
        )
        scopes.append(
            CalibrationScopeRule(
                modules=(f"{prefix}.*",),
                module_classes=block_class,
                prev_replay_transform=_krea2_block_prev_replay_transform,
            )
        )
    return tuple(scopes)


def krea2_target_config(model, *, precision: str, rank: int, skip=()) -> TargetConfig:
    """Return the generic scanner's targets with chained calibration scopes.

    Reusing ``scan_linear_targets`` is deliberate: it keeps the exported target
    names identical to the generic path, which is what preserves runtime-manifest
    compatibility and lets ``convert_nunchaku_lite_diffusers.py`` package the
    result. Only the scopes change.
    """

    scan = scan_linear_targets(model, precision=precision, rank=rank, skip=skip)
    return replace(scan.target_config, calibration_scopes=krea2_calibration_scopes()), scan


def run() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser(MODEL_ID).parse_args()

    pipe = load_auto_pipeline(args.model_id, device=args.device, pipeline_offload=args.pipeline_offload)
    _, model = discover_denoiser(pipe)
    target_config, scan = krea2_target_config(model, precision=args.precision, rank=args.rank, skip=args.skip)
    print(scan.format_text())
    print(f"Chained calibration scopes: {len(target_config.calibration_scopes)} rules across "
          f"{len(krea2_block_stacks())} block stacks")
    if args.inspect_config:
        print(inspect_target_config(model, target_config).format_text())
        return
    if not scan.svdq_targets and not scan.awq_targets:
        raise ValueError("No compatible linear targets were discovered")

    output = Path(
        args.output or f"outputs/checkpoints/svdq-{args.precision}_r{args.rank}-{_slug(args.model_id)}.safetensors"
    )
    cache_dir = Path(args.cache_dir or f"outputs/calibration/{_slug(args.model_id)}")
    artifact_cache = (
        None if args.cache_mode == "disabled" else QuantizationCacheSpec(cache_dir / args.precision / "artifacts", args.cache_mode)
    )
    spec = replace(
        svdquant_spec(
            args.precision,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            compute_device=args.compute_device or (args.device if args.offload_model else None),
            offload_model=args.offload_model,
        ),
        rank=args.rank,
    )
    quantize_and_export(
        model,
        spec,
        target_config,
        CalibrationSpec(
            samples=batched_samples(standard_prompt_records(args.num_samples, args.prompt_file), args.batch_size),
            num_samples=args.num_samples,
            cache_num_samples=args.num_samples if args.cache_num_samples is None else args.cache_num_samples,
            batch_size=args.batch_size,
            cache_dir=cache_dir / args.precision / "inputs",
            cache_mode=args.cache_mode,
            forward_fn=_forward_fn(pipe, args),
            max_rows_per_target=4096,
            artifact_cache=artifact_cache,
            output_dir=cache_dir / args.precision / "inputs" / "samples",
            output_save_fn=save_diffusers_images,
            scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
            sample_batch_size=args.sample_batch_size,
        ),
        ExportSpec(output=output),
        LoggingConfig(name=output.stem),
    )


if __name__ == "__main__":
    run()
