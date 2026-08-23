"""Quantize Ideogram v4 fast/instant with calibrated SVDQuant and chained block replay.

The fal Ideogram v4 releases (``fal/ideogram-v4-fast``, 20-step and
``fal/ideogram-v4-instant``, 8-step) are transformer-only, gated repositories:
the tokenizer, text encoder, VAE, and scheduler come from Ideogram AI's gated
``ideogram-ai/ideogram-4-nf4-diffusers`` repository, and Diffusers 0.39.0's
mandatory CFG plumbing is satisfied with a zero-parameter unconditional stand-in
instead of loading the repository's second diffusion transformer. Loading the
NF4-serialized text encoder requires ``bitsandbytes``.

Both checkpoints are CFG-distilled, so calibration replay always runs with
``guidance_scale=1.0`` and the model card's flow-matching schedule
(``mu=0.0``, ``std=1.75``); ``Ideogram4Pipeline`` raises if a guidance scale is
combined with its default per-step guidance schedule, which is why this script
carries its own forward function instead of the generic one. Calibration
prompts are wrapped into Ideogram 4's structured JSON caption format.

The generic scanner's target set is kept byte-for-byte -- 1:1 module-to-prefix
mapping, empty ``structural_patches``, converter-compatible runtime manifest --
and only the calibration scopes are replaced with chained ones, so the 34-block
``layers`` stack replays one block per scope instead of replaying from the root
(O(N^2) over the stack).

Usage:

    python examples/text_to_image/quantize_ideogram_v4.py --precision nvfp4 \
        --num-samples 32 --steps 20 --compute-device cuda

    python examples/text_to_image/quantize_ideogram_v4.py fal/ideogram-v4-instant \
        --precision nvfp4 --num-samples 32 --steps 8 --compute-device cuda
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

import torch

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
    _slug,
    build_parser,
    discover_denoiser,
    scan_linear_targets,
)
from examples.text_to_image.utils import (
    batched_samples,
    make_generator,
    save_diffusers_images,
    standard_prompt_records,
    svdquant_spec,
)

MODEL_ID = "fal/ideogram-v4-fast"
COMPONENTS_REPO = "ideogram-ai/ideogram-4-nf4-diffusers"
COMPONENTS_REVISION = "1874bc70267ba2c823a7239e1d70dd308c8d64dc"


class ZeroUnconditionalTransformer(torch.nn.Module):
    """Zero-parameter stand-in for Diffusers 0.39.0's mandatory CFG branch.

    With ``guidance_scale=1.0`` the pipeline blends
    ``1.0 * conditional + 0.0 * unconditional``, so the zero output never
    contributes; no second diffusion transformer is loaded or run.
    """

    def __init__(self, dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        self.register_buffer("_dtype_anchor", torch.empty(0, dtype=dtype), persistent=False)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype_anchor.dtype

    def forward(self, *, hidden_states: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        return (torch.zeros_like(hidden_states),)


def load_ideogram_pipeline(
    model_id: str,
    *,
    device: str,
    pipeline_offload: str,
    components_revision: str = COMPONENTS_REVISION,
):
    """Assemble ``Ideogram4Pipeline`` from the transformer-only fal repository."""

    from diffusers import Ideogram4Pipeline, Ideogram4Transformer2DModel

    transformer = Ideogram4Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    pipe = Ideogram4Pipeline.from_pretrained(
        COMPONENTS_REPO,
        revision=components_revision,
        transformer=transformer,
        unconditional_transformer=None,
        torch_dtype=torch.bfloat16,
    )
    pipe.register_modules(unconditional_transformer=ZeroUnconditionalTransformer())
    if pipeline_offload == "none":
        return pipe.to(device)
    method = getattr(pipe, "enable_model_cpu_offload" if pipeline_offload == "model" else "enable_sequential_cpu_offload")
    try:
        method(device=device)
    except TypeError:
        method()
    return pipe


def _ideogram_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Feed the previous block's output in as this block's ``hidden_states``.

    ``Ideogram4TransformerBlock`` returns a single tensor that is exactly the
    next block's ``hidden_states``. The remaining inputs (``attention_mask``,
    ``image_rotary_emb``, ``adaln_input``) are shared across blocks, so they
    carry over from the previous replay unchanged.
    """

    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = replay.output
        return (), kwargs
    args = list(replay.args)
    if not args:
        raise TypeError("Ideogram4 block replay must include hidden_states")
    args[0] = replay.output
    return tuple(args), {}


def ideogram_calibration_scopes() -> tuple[CalibrationScopeRule, ...]:
    """Chain replay through the single ``layers`` stack."""

    from diffusers.models.transformers.transformer_ideogram4 import Ideogram4TransformerBlock

    return (
        CalibrationScopeRule(
            modules=("layers.0",),
            module_classes=Ideogram4TransformerBlock,
            use_prev_scope_outputs=False,
        ),
        CalibrationScopeRule(
            modules=("layers.*",),
            module_classes=Ideogram4TransformerBlock,
            prev_replay_transform=_ideogram_block_prev_replay_transform,
        ),
    )


def ideogram_target_config(model, *, precision: str, rank: int, skip=()) -> tuple[TargetConfig, object]:
    """Return the generic scanner's targets with chained calibration scopes."""

    scan = scan_linear_targets(model, precision=precision, rank=rank, skip=skip)
    return replace(scan.target_config, calibration_scopes=ideogram_calibration_scopes()), scan


def _json_caption_records(num_samples: int, prompt_file) -> list[dict]:
    """Wrap plain calibration prompts into Ideogram 4's structured caption format."""

    records = standard_prompt_records(num_samples, prompt_file)
    for record in records:
        record["prompt"] = json.dumps(
            {"high_level_description": record["prompt"]}, ensure_ascii=False, separators=(",", ":")
        )
    return records


def _ideogram_forward_fn(pipe, args):
    """Run calibration generation with the CFG-distilled inference settings.

    ``guidance_schedule=None`` is required alongside ``guidance_scale``; the
    schedule parameters follow the fal model cards (``mu=0.0``, ``std=1.75``).
    """

    def forward(sample: dict):
        return pipe(
            sample["prompt"],
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=1.0,
            guidance_schedule=None,
            mu=0.0,
            std=1.75,
            max_sequence_length=args.max_sequence_length,
            generator=make_generator(sample.get("seed", 0), device=args.device),
        )

    return forward


def run() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = build_parser(MODEL_ID)
    parser.set_defaults(steps=20, height=1024, width=1024)
    parser.add_argument(
        "--components-revision",
        default=COMPONENTS_REVISION,
        help="Revision of the ideogram-ai components repository to load",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help=(
            "Text conditioning length. The 53248-dim text features are padded to this "
            "length, so lowering it cuts calibration activation memory when prompts are short."
        ),
    )
    args = parser.parse_args()

    pipe = load_ideogram_pipeline(
        args.model_id,
        device=args.device,
        pipeline_offload=args.pipeline_offload,
        components_revision=args.components_revision,
    )
    _, model = discover_denoiser(pipe)
    target_config, scan = ideogram_target_config(model, precision=args.precision, rank=args.rank, skip=args.skip)
    print(scan.format_text())
    print(f"Chained calibration scopes: {len(target_config.calibration_scopes)} rules over the layers stack")
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
            samples=batched_samples(_json_caption_records(args.num_samples, args.prompt_file), args.batch_size),
            num_samples=args.num_samples,
            cache_num_samples=args.num_samples if args.cache_num_samples is None else args.cache_num_samples,
            batch_size=args.batch_size,
            cache_dir=cache_dir / args.precision / "inputs",
            cache_mode=args.cache_mode,
            forward_fn=_ideogram_forward_fn(pipe, args),
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
