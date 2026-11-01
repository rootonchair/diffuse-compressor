"""Quantize FLUX.3 [klein] (step-distilled, 16 steps, no CFG) to Nunchaku Lite SVDQuant.

``Flux3Transformer2DModel`` has four homogeneous block stacks: per-mode blocks for
the noisy image (``mode_blocks.image``), the reference images
(``mode_blocks.image_cond``) and the text (``txt_mode_blocks``), then 28 joint
``single_transformer_blocks``. Every linear in those blocks has separate Q/K/V,
output and feed-forward projections, so the generic ``quantize_hf.py`` scanner
finds the right SVDQuant targets with a 1:1 module-to-prefix mapping, which keeps
the runtime manifest valid for ``convert_nunchaku_lite_diffusers.py``. The
per-token modulation heads, input embedders and the output AdaLN/projection stay
BF16.

What the generic path gets wrong for this model, and this script handles:

* Pipeline. FLUX.3 ships as a Diffusers modular pipeline whose first step is a
  9B prompt upsampler. Calibration removes that step and encodes the prompts as
  given, so calibration does not depend on sampled upsampler text.
* Reference-image blocks. ``mode_blocks.image_cond`` only runs when the pipeline
  receives reference images. Calibration records may therefore carry
  ``images``; without any, those blocks would see no activations, so the script
  refuses unless they are skipped explicitly.
* Memory. The text encoder runs first, alone, and its embeddings are reused by a
  denoise-only pipeline holding just the transformer and VAE.

Each block is replayed from the cached root calls with early stop rather than
chained: the per-token modulation tensors (``(1, tokens, 9216)`` per call) make
in-RAM chain records too large for a full calibration set.

Calibration prompts come from a JSON file ``{"records": [...]}``; each record has
a unique ``filename``, ``prompt``, ``seed`` and optionally ``width``, ``height``
and ``images`` (reference image paths, relative to the JSON file). A qdiff YAML
file is also accepted for text-to-image only calibration, together with
``--skip 'mode_blocks.image_cond.*'``.

``examples/prompts/flux3_klein_calibration.json`` describes the calibration set
used for the published lite-infer checkpoints (32 qdiff prompts, 16 detailed
captions and 16 edits). It is a specification, not a records file: each edit's
reference image is first rendered by the BF16 model from ``reference_prompt``,
and the records then point at those images.

Usage::

    python examples/text_to_image/quantize_flux3_klein.py /path/to/flux-3-image-klein \\
        --precision nvfp4 --prompt-file calibration.json --gptq
    # The second precision reuses the cached BF16 calibration inputs.
    python examples/text_to_image/quantize_flux3_klein.py /path/to/flux-3-image-klein \\
        --precision int4 --prompt-file calibration.json --gptq

Requires Diffusers with FLUX.3 support (0.42.1 or newer, Python 3.11+).
"""

from __future__ import annotations

import gc
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diffuse_compressor import (
    CalibrationSpec,
    ExportSpec,
    LoggingConfig,
    QuantizationCacheSpec,
    TargetConfig,
    inspect_target_config,
    quantize_and_export,
)
from examples.text_to_image.quantize_hf import _slug, scan_linear_targets
from examples.text_to_image.quantize_qwen_image_2_1 import (
    build_parser as build_qwen_parser,
    qwen_image_21_spec,
)
from examples.text_to_image.utils import make_generator, standard_prompt_records

MODEL_ID = "black-forest-labs/flux-3-image-klein"
DEFAULT_STEPS = 16
# About 3.5x the widest target input (9216, ``ff.linear_out``), so GPTQ Hessians are full rank.
DEFAULT_MAX_ROWS_PER_TARGET = 32768
IMAGE_COND_BLOCKS = "mode_blocks.image_cond.*"


def load_calibration_records(path: str | Path, num_samples: int) -> list[dict]:
    """Load JSON calibration records (reference paths made absolute) or qdiff YAML prompts."""

    path = Path(path)
    if path.suffix != ".json":
        return standard_prompt_records(num_samples, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if "records" not in data:
        raise ValueError(
            f"{path} has no 'records' list. A calibration specification such as "
            "examples/prompts/flux3_klein_calibration.json must first be turned into records "
            "(with BF16-rendered reference images for its edits); see this script's docstring."
        )
    records = data["records"]
    names = [record["filename"] for record in records]
    if len(set(names)) != len(names):
        raise ValueError("Calibration record filenames must be unique; prompt embeddings are keyed by filename")
    if num_samples > 0:
        records = records[:num_samples]
    resolved = []
    for record in records:
        record = dict(record)
        record["images"] = [str((path.parent / image).resolve()) for image in record.get("images", ())]
        for image in record["images"]:
            if not Path(image).is_file():
                raise FileNotFoundError(f"Calibration reference image is missing: {image}")
        resolved.append(record)
    return resolved


def flux3_blocks(*, keep=("text_encoder", "vae_encoder", "denoise", "decode")):
    """Return the distilled FLUX.3 blocks without the prompt upsampler (and optionally other steps)."""

    from diffusers import Flux3DistilledAutoBlocks

    blocks = Flux3DistilledAutoBlocks()
    for name in list(blocks.sub_blocks):
        if name not in keep:
            blocks.sub_blocks.pop(name)
    return blocks


def load_flux3_pipeline(model_id: str, *, device: str, keep, components=None):
    """Initialise a modular pipeline from ``model_id`` with only the components ``keep`` needs."""

    import torch

    pipe = flux3_blocks(keep=keep).init_pipeline(model_id)
    pipe.load_components(names=components, torch_dtype=torch.bfloat16)
    pipe.to(device)
    return pipe


def encode_prompts(model_id: str, prompts: list[str], *, device: str) -> list:
    """Encode each prompt with the pipeline's own text-encoder step; free the encoder afterwards."""

    import torch

    pipe = load_flux3_pipeline(model_id, device=device, keep=("text_encoder",))
    embeddings = []
    with torch.inference_mode():
        for prompt in prompts:
            embeddings.append(pipe(prompt=prompt, output="prompt_embeds").cpu())
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


def flux3_forward_fn(pipe, embeddings: dict, args):
    """Run the denoise-only pipeline from precomputed prompt embeddings."""

    def forward(sample: dict):
        from PIL import Image

        kwargs = {
            "prompt_embeds": embeddings[sample["filename"]].to(args.device),
            "height": int(sample.get("height") or args.height),
            "width": int(sample.get("width") or args.width),
            "num_inference_steps": args.steps,
            "generator": make_generator(sample.get("seed", 0), device=args.device),
        }
        if sample.get("images"):
            kwargs["image"] = [Image.open(path).convert("RGB") for path in sample["images"]]
        return pipe(**kwargs, output="images")

    return forward


def save_calibration_images(result, sample: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(result):
        suffix = "" if len(result) == 1 else f"-{index}"
        image.save(output_dir / f"{sample['filename']}{suffix}.png")


def flux3_target_config(model, *, precision: str, rank: int, skip=()) -> tuple[TargetConfig, object]:
    """Scanner targets over the four block stacks, each block replayed from the root with early stop."""

    scan = scan_linear_targets(model, precision=precision, rank=rank, skip=skip)
    expected_roots = ("mode_blocks.image.", "mode_blocks.image_cond.", "txt_mode_blocks.", "single_transformer_blocks.")
    stray = [name for name in scan.svdq_targets if not name.startswith(expected_roots)]
    if stray or scan.awq_targets:
        raise ValueError(f"Unexpected FLUX.3 targets outside the block stacks: {stray + list(scan.awq_targets)}")
    return scan.target_config, scan


def build_parser():
    parser = build_qwen_parser()
    parser.description = "Quantize FLUX.3 [klein] to Nunchaku Lite SVDQuant."
    parser.set_defaults(
        model_id=MODEL_ID, height=1024, width=1024, steps=DEFAULT_STEPS, max_rows_per_target=DEFAULT_MAX_ROWS_PER_TARGET
    )
    for action in parser._actions:
        if action.dest == "model_id":
            action.help = f"Hugging Face model id or local modular pipeline (default: {MODEL_ID})"
        if action.dest in {"guidance_scale", "true_cfg_scale", "negative_prompt"}:
            action.help = "Unused: the distilled FLUX.3 model runs without classifier-free guidance."
    return parser


def run() -> None:
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args()
    if args.batch_size != 1:
        raise ValueError("FLUX.3 encodes one prompt per call; calibration requires --batch-size 1")

    import random
    import torch
    from diffusers import Flux3Transformer2DModel

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_calibration_records(args.prompt_file, args.num_samples)
    if not any(record.get("images") for record in records) and not any(
        pattern.startswith("mode_blocks.image_cond") for pattern in args.skip
    ):
        raise ValueError(
            f"No calibration record has reference images, so {IMAGE_COND_BLOCKS} would see no activations; "
            f"add image-conditioned records or pass --skip '{IMAGE_COND_BLOCKS}'"
        )

    method = "svdq-gptq" if args.gptq else "svdq"
    output = Path(
        args.output or f"outputs/checkpoints/{method}-{args.precision}_r{args.rank}-{_slug(args.model_id)}.safetensors"
    )
    cache_dir = Path(args.cache_dir or f"outputs/calibration/{_slug(args.model_id)}")
    # Root calls are BF16 model inputs, identical for both precisions, so they are captured once.
    input_dir = cache_dir / "inputs"
    has_cache = args.cache_mode == "reuse" and any((input_dir / "caches").glob("*.pt"))

    pipe = None
    forward = None
    if has_cache:
        # Cached root calls replace the pipeline entirely; only the transformer is needed.
        model = Flux3Transformer2DModel.from_pretrained(args.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
        # Root replay runs on the model's device; left on CPU, every scope replays all calls on CPU.
        # With --offload-model it stays on CPU and replay offloads blocks to the device as needed.
        if not args.offload_model:
            model.to(args.device)
    else:
        prompts = [record["prompt"] for record in records]
        embeddings = dict(zip((r["filename"] for r in records), encode_prompts(args.model_id, prompts, device=args.device)))
        pipe = load_flux3_pipeline(args.model_id, device=args.device, keep=("vae_encoder", "denoise", "decode"))
        model = pipe.transformer
        forward = flux3_forward_fn(pipe, embeddings, args)

    target_config, scan = flux3_target_config(model, precision=args.precision, rank=args.rank, skip=args.skip)
    print(scan.format_text())
    if args.inspect_config:
        print(inspect_target_config(model, target_config).format_text())
        return

    artifact_cache = (
        None
        if args.cache_mode == "disabled"
        else QuantizationCacheSpec(cache_dir / args.precision / "artifacts", args.cache_mode, save_model=False)
    )
    calibration = CalibrationSpec(
        samples=records,
        num_samples=len(records),
        seed=args.seed,
        cache_num_samples=-1 if args.cache_num_samples is None else args.cache_num_samples,
        batch_size=1,
        cache_dir=input_dir,
        cache_mode=args.cache_mode,
        forward_fn=forward,
        max_rows_per_target=args.max_rows_per_target,
        artifact_cache=artifact_cache,
        output_dir=None if forward is None else input_dir / "samples",
        output_save_fn=save_calibration_images,
        scope_capture_mode=args.scope_capture_mode.replace("-", "_"),
        sample_batch_size=args.sample_batch_size,
    )
    if pipe is not None:
        # Free the VAE before quantization; block replay manages the transformer's placement itself.
        model.to("cpu")
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
    quantize_and_export(
        model,
        qwen_image_21_spec(args),
        target_config,
        calibration,
        ExportSpec(output=output),
        LoggingConfig(name=output.stem),
    )


if __name__ == "__main__":
    run()
