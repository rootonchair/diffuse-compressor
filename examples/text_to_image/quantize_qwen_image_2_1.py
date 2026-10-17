"""Quantize Qwen-Image 2.1 with chained single-stream block replay.

The generic ``quantize_hf.py`` scanner already finds the right targets for
``QwenImage21Transformer2DModel``: every linear in the 32 single-stream blocks as
SVDQuant, and the shared top-level ``modulation.1`` projection as AWQ. This script
keeps that target set byte-for-byte -- so the checkpoint keeps its 1:1
module-to-prefix mapping and a clean runtime manifest -- and changes only what
the generic path gets wrong for this pipeline:

* Calibration forwards. ``QwenImage21Pipeline`` has no ``guidance_scale``
  argument; classifier-free guidance is ``true_cfg_scale`` plus a
  ``negative_prompt``, and it is off by default, as in the model card.
* The prefix KV cache. By default the pipeline prefills text K/V on step 0 and
  later steps only recompute target-image tokens from a stateful cache object.
  Calibration records and replays transformer calls out of context, so it runs
  with ``use_kv_cache=False``: every step processes the full joint sequence.
* Calibration scopes. Each block replays from the previous block's output
  instead of from the model root, so replay cost does not grow with depth.

Calibration batch size must be 1: ``img_shapes`` and ``img_mask`` describe one
call's token layout and cannot be concatenated across cached records.

Usage (set HF_HUB_CACHE if the weights live in a non-default Hugging Face cache):

    python examples/text_to_image/quantize_qwen_image_2_1.py --precision int4 \
        --num-samples 128 --steps 40

Add ``--true-cfg-scale 4.0`` (and optionally ``--negative-prompt``) to calibrate
the negative-prompt passes too when the model will be served with CFG.

``--gptq`` enables GPTQ residual rounding. Its Hessian is built from the cached
activation rows, so ``--max-rows-per-target`` (default 32768) is kept well above
the widest target input (12288, ``img_mlp.out``); with fewer rows than input
channels the Hessian is rank-deficient and only damping holds it together.
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
    GptqSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    TargetConfig,
    inspect_target_config,
    quantize_and_export,
)
from examples.text_to_image.quantize_hf import (
    _slug,
    build_parser as build_generic_parser,
    discover_denoiser,
    load_auto_pipeline,
    scan_linear_targets,
)
from examples.text_to_image.utils import (
    batched_samples,
    make_generator,
    save_diffusers_images,
    standard_prompt_records,
    svdquant_spec,
)

MODEL_ID = "Qwen/Qwen-Image-2.1"
# About 2.7x the widest target input (12288), so per-target activation statistics
# and GPTQ Hessians are full rank.
DEFAULT_MAX_ROWS_PER_TARGET = 32768


def _qwen_image_21_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Feed the previous block's output in as this block's ``hidden_states``.

    A Qwen-Image 2.1 block returns the joint hidden states as a single tensor.
    Its other inputs (``modulation``, ``rotary_emb``, ``attention_mask``,
    ``target_token_mask``, ``segments``, ``key_valid``) are shared by every
    block, so they carry over from the previous replay unchanged.
    """

    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = replay.output
        return (), kwargs
    args = list(replay.args)
    if not args:
        raise TypeError("Qwen-Image 2.1 block replay must include hidden_states")
    args[0] = replay.output
    return tuple(args), {}


def qwen_image_21_calibration_scopes() -> tuple[CalibrationScopeRule, ...]:
    """Recompute the first block from the root, then chain the rest."""

    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

    return (
        CalibrationScopeRule(
            modules=("transformer_blocks.0",),
            module_classes=QwenImage21TransformerBlock,
            use_prev_scope_outputs=False,
        ),
        CalibrationScopeRule(
            modules=("transformer_blocks.*",),
            module_classes=QwenImage21TransformerBlock,
            prev_replay_transform=_qwen_image_21_block_prev_replay_transform,
        ),
    )


def qwen_image_21_target_config(model, *, precision: str, rank: int, skip=()) -> TargetConfig:
    """Return the generic scanner's targets with chained calibration scopes.

    Keeping the scanner's targets preserves the exported names, which is what
    keeps the runtime manifest valid and lets ``convert_nunchaku_lite_diffusers.py``
    package the result. Only the scopes change.
    """

    scan = scan_linear_targets(model, precision=precision, rank=rank, skip=skip)
    return replace(scan.target_config, calibration_scopes=qwen_image_21_calibration_scopes()), scan


def qwen_image_21_forward_fn(pipe, args):
    """Build the calibration forward: KV cache off, CFG only when requested."""

    def forward(sample: dict):
        kwargs = {
            "prompt": sample["prompt"],
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "generator": make_generator(sample.get("seed", 0), device=args.device),
            "use_kv_cache": False,
        }
        if args.true_cfg_scale > 1:
            kwargs["true_cfg_scale"] = args.true_cfg_scale
            kwargs["negative_prompt"] = args.negative_prompt
        return pipe(**kwargs)

    return forward


def build_parser():
    """Extend the generic CLI with Qwen-Image 2.1 defaults and CFG flags."""

    parser = build_generic_parser(MODEL_ID)
    parser.set_defaults(height=1024, width=1024, steps=40)
    for action in parser._actions:
        if action.dest == "guidance_scale":
            action.help = "Unused: Qwen-Image 2.1 has no guidance_scale; use --true-cfg-scale."
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=1.0,
        help="Classifier-free guidance scale for calibration; values above 1 also run negative-prompt passes.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=" ",
        help="Negative prompt used when --true-cfg-scale is above 1.",
    )
    parser.add_argument(
        "--max-rows-per-target",
        type=int,
        default=DEFAULT_MAX_ROWS_PER_TARGET,
        help="Activation rows kept per target (sampled uniformly across all calibration calls).",
    )
    parser.add_argument("--gptq", action="store_true", help="Enable GPTQ residual rounding.")
    parser.add_argument("--gptq-damp-percentage", type=float, default=0.01)
    parser.add_argument("--gptq-block-size", type=int, default=128)
    parser.add_argument("--gptq-num-inv-tries", type=int, default=250)
    parser.add_argument("--gptq-hessian-block-size", type=int, default=512)
    return parser


def qwen_image_21_spec(args):
    """Build the quantization spec from CLI arguments."""

    spec = replace(
        svdquant_spec(
            args.precision,
            svd_backend=args.svd_backend,
            svd_lowrank_oversample=args.svd_lowrank_oversample,
            svd_lowrank_niter=args.svd_lowrank_niter,
            # Scoped block replay moves each block back to CPU afterwards, so without an
            # explicit compute device the per-target math would run on CPU.
            compute_device=args.compute_device or args.device,
            offload_model=args.offload_model,
        ),
        rank=args.rank,
    )
    if args.gptq:
        spec = replace(
            spec,
            gptq=GptqSpec(
                enabled=True,
                damp_percentage=args.gptq_damp_percentage,
                block_size=args.gptq_block_size,
                num_inv_tries=args.gptq_num_inv_tries,
                hessian_block_size=args.gptq_hessian_block_size,
            ),
        )
    return spec


def run() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError(
            "Qwen-Image 2.1 calibration requires --batch-size 1: img_shapes and img_mask describe one call's "
            "token layout and cannot be concatenated across cached records"
        )

    pipe = load_auto_pipeline(args.model_id, device=args.device, pipeline_offload=args.pipeline_offload)
    _, model = discover_denoiser(pipe)
    target_config, scan = qwen_image_21_target_config(model, precision=args.precision, rank=args.rank, skip=args.skip)
    print(scan.format_text())
    print(f"Chained calibration scopes: {len(target_config.calibration_scopes)} rules over transformer_blocks")
    if args.inspect_config:
        print(inspect_target_config(model, target_config).format_text())
        return
    if not scan.svdq_targets and not scan.awq_targets:
        raise ValueError("No compatible linear targets were discovered")

    method = "svdq-gptq" if args.gptq else "svdq"
    output = Path(
        args.output or f"outputs/checkpoints/{method}-{args.precision}_r{args.rank}-{_slug(args.model_id)}.safetensors"
    )
    cache_dir = Path(args.cache_dir or f"outputs/calibration/{_slug(args.model_id)}")
    # Per-target artifacts are enough to resume an interrupted run; the combined model copy
    # (about 7.5 GB at rank 64) would only duplicate the exported checkpoint.
    artifact_cache = (
        None
        if args.cache_mode == "disabled"
        else QuantizationCacheSpec(cache_dir / args.precision / "artifacts", args.cache_mode, save_model=False)
    )
    quantize_and_export(
        model,
        qwen_image_21_spec(args),
        target_config,
        CalibrationSpec(
            samples=batched_samples(standard_prompt_records(args.num_samples, args.prompt_file), args.batch_size),
            num_samples=args.num_samples,
            cache_num_samples=args.num_samples if args.cache_num_samples is None else args.cache_num_samples,
            batch_size=args.batch_size,
            cache_dir=cache_dir / args.precision / "inputs",
            cache_mode=args.cache_mode,
            forward_fn=qwen_image_21_forward_fn(pipe, args),
            max_rows_per_target=args.max_rows_per_target,
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
