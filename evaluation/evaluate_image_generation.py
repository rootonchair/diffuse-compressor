"""DeepCompressor-style image generation evaluation example.

This script intentionally owns the dataset, DataLoader, generation loop, image
saving, and metrics. The package only provides ``load_evaluation_pipeline`` so
projects can keep their evaluation code in ordinary PyTorch style.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from diffuse_compressor.runtime import RuntimePipelineSpec, load_evaluation_pipeline
from evaluation.datasets import DCIDataset, MJHQDataset, PromptDataset
from examples.upstream_diffusion_svdquant import (
    MODEL_DEFAULTS,
    UPSTREAM_QDIFF_PROMPT_SOURCE,
    configure_logging,
    standard_prompt_records,
)


EvalDataset = PromptDataset | MJHQDataset | DCIDataset


def _load_eval_dataset(args: argparse.Namespace) -> tuple[EvalDataset, str]:
    """Load a qdiff-style prompt dataset or a supported benchmark dataset."""

    if args.benchmark is None:
        prompt_file = args.prompt_file or UPSTREAM_QDIFF_PROMPT_SOURCE
        records = standard_prompt_records(args.num_samples, prompt_file=prompt_file)
        return PromptDataset(records), _sample_set_name(prompt_file)
    if args.benchmark == "MJHQ":
        dataset = MJHQDataset(args.num_samples)
        return dataset, dataset.sample_set_name
    if args.benchmark == "DCI":
        dataset = DCIDataset(args.num_samples)
        return dataset, dataset.sample_set_name
    raise ValueError(f"Unsupported benchmark: {args.benchmark!r}")


def _save_target_images(records: Sequence[dict[str, Any]], output_dir: Path) -> Path | None:
    """Save benchmark target images and return their directory."""

    if not any("target_image" in record for record in records):
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        image = record.get("target_image")
        if image is None:
            continue
        image.save(output_dir / f"{record['filename']}.png")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the image generation evaluation parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("original", "quantized"), required=True)
    parser.add_argument("--model-key", choices=tuple(MODEL_DEFAULTS), default="flux.1-schnell")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--pipeline-cls", default=None, help="Diffusers class name or fully qualified class path.")
    parser.add_argument("--runtime", choices=("none", "nunchaku-lite", "torch-dequant"), default="none")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--precision", choices=("int4", "fp4", "nvfp4"), default="int4")
    parser.add_argument("--torch-dequant-activation-mode", choices=("none", "input"), default="input")
    parser.add_argument("--output-dir", required=True)
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--prompt-file", default=None)
    data.add_argument("--benchmark", choices=("MJHQ", "DCI"), default=None)
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["clip_iqa", "clip_score", "image_reward", "fid"],
        choices=(
            "clip_iqa",
            "clip_score",
            "image_reward",
            "fid",
            "psnr",
            "lpips",
            "ssim",
            "mse",
            "mae",
            "rmse",
        ),
    )
    parser.add_argument("--ref-root", default=None, help="Original run output root for with_orig metrics.")
    parser.add_argument("--gt-root", default=None, help="Ground-truth image folder for with_gt similarity/FID metrics.")
    return parser


def main() -> None:
    """Run one original or quantized image-generation evaluation."""

    configure_logging()
    args = build_parser().parse_args()
    defaults = MODEL_DEFAULTS[args.model_key]
    model_id = args.model_id or defaults.model_id
    pipeline_cls = _resolve_pipeline_cls(args.pipeline_cls or defaults.pipeline_name)
    torch_dtype = _resolve_torch_dtype(args.torch_dtype or defaults.torch_dtype)
    steps = args.steps if args.steps is not None else defaults.steps
    guidance_scale = args.guidance_scale if args.guidance_scale is not None else defaults.guidance_scale
    dataset, sample_set_name = _load_eval_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "samples" / f"{sample_set_name}-{len(dataset)}"
    target_dir = _save_target_images(dataset.records, output_dir / "targets" / f"{sample_set_name}-{len(dataset)}")
    gt_root = Path(args.gt_root) if args.gt_root else target_dir
    sample_dir.mkdir(parents=True, exist_ok=True)

    pipe = load_evaluation_pipeline(
        pipeline_cls=pipeline_cls,
        model_id=model_id,
        spec=RuntimePipelineSpec(
            mode=args.mode,
            runtime=args.runtime,
            checkpoint=args.checkpoint,
            model_key=args.model_key,
            precision="fp4" if args.precision == "nvfp4" else args.precision,
            device=args.device,
            torch_dtype=torch_dtype,
            torch_dequant_activation_mode=args.torch_dequant_activation_mode,
        ),
    )
    _generate_images(
        pipe,
        dataloader,
        sample_dir,
        height=args.height,
        width=args.width,
        steps=steps,
        guidance_scale=guidance_scale,
        device=args.device,
    )

    metrics = compute_image_metrics(
        sample_dir,
        prompts={str(record["filename"]): str(record["prompt"]) for record in dataset.records},
        metrics=args.metrics,
        ref_root=_resolve_compare_dir(args.ref_root, sample_set_name, len(dataset)) if args.ref_root else None,
        gt_root=gt_root,
        device=args.device,
    )
    result = {
        "mode": args.mode,
        "model_key": args.model_key,
        "model_id": model_id,
        "runtime": args.runtime,
        "checkpoint": args.checkpoint,
        "num_samples": len(dataset),
        "sample_dir": str(sample_dir),
        "target_dir": None if target_dir is None else str(target_dir),
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _generate_images(
    pipe: Any,
    dataloader: DataLoader,
    output_dir: Path,
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    device: str,
) -> None:
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=False)
    with torch.inference_mode():
        for batch in dataloader:
            filenames = [str(item) for item in _as_list(batch["filename"])]
            if all((output_dir / f"{filename}.png").exists() for filename in filenames):
                continue
            prompts = [str(item) for item in _as_list(batch["prompt"])]
            seeds = [int(item) for item in _as_list(batch["seed"])]
            generators = [_make_generator(seed, device) for seed in seeds]
            output = pipe(
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generators,
            )
            images = getattr(output, "images", None)
            if images is None:
                raise ValueError("pipeline output must expose an images attribute")
            if len(images) != len(filenames):
                raise ValueError(f"Expected {len(filenames)} images, got {len(images)}")
            for filename, image in zip(filenames, images, strict=True):
                image.save(output_dir / f"{filename}.png")


def compute_image_metrics(
    gen_dir: Path,
    *,
    prompts: dict[str, str],
    metrics: Sequence[str],
    ref_root: Path | None,
    gt_root: Path | None,
    device: str,
) -> dict[str, dict[str, float | str]]:
    """Compute DeepCompressor-style metric groups for generated images."""

    requested = set(metrics)
    results: dict[str, dict[str, float | str]] = {}
    with_gt: dict[str, float | str] = {}
    with_orig: dict[str, float | str] = {}

    for metric in ("clip_iqa", "clip_score"):
        if metric in requested:
            with_gt[metric] = _compute_clip_metric(metric, gen_dir, prompts, device=device)
    if "image_reward" in requested:
        with_gt["image_reward"] = _compute_image_reward(gen_dir, prompts)
    if gt_root is not None:
        with_gt.update(_compute_pair_metrics(requested, gt_root, gen_dir, device=device))
        if "fid" in requested:
            with_gt["fid"] = _compute_fid(gt_root, gen_dir)
    elif "fid" in requested:
        with_gt["fid"] = "skipped: --gt-root is required for generated-vs-ground-truth FID"
    if with_gt:
        results["with_gt"] = with_gt

    if ref_root is not None:
        with_orig.update(_compute_pair_metrics(requested, ref_root, gen_dir, device=device))
        if "fid" in requested:
            with_orig["fid"] = _compute_fid(ref_root, gen_dir)
        results["with_orig"] = with_orig
    return results


def _compute_pair_metrics(metrics: set[str], ref_dir: Path, gen_dir: Path, *, device: str) -> dict[str, float]:
    names = _common_image_names(ref_dir, gen_dir)
    if not names:
        raise ValueError(f"No matching images found in {ref_dir} and {gen_dir}")
    results: dict[str, float] = {}
    basic = {"mse", "mae", "rmse", "psnr"} & metrics
    if basic:
        totals = {metric: 0.0 for metric in basic}
        for name in names:
            ref = _image_float_tensor(ref_dir / name)
            gen = _image_float_tensor(gen_dir / name)
            mse = torch.mean((gen - ref) ** 2).item()
            mae = torch.mean((gen - ref).abs()).item()
            values = {
                "mse": mse,
                "mae": mae,
                "rmse": math.sqrt(mse),
                "psnr": float("inf") if mse == 0 else 10 * math.log10(1.0 / mse),
            }
            for metric in basic:
                totals[metric] += values[metric]
        results.update({metric: value / len(names) for metric, value in totals.items()})
    if "ssim" in metrics:
        metric = _torchmetrics_image_metric("ssim").to(device)
        for name in names:
            metric.update(
                _image_float_tensor(gen_dir / name).unsqueeze(0).to(device),
                _image_float_tensor(ref_dir / name).unsqueeze(0).to(device),
            )
        results["ssim"] = float(metric.compute().item())
    if "lpips" in metrics:
        metric = _torchmetrics_image_metric("lpips").to(device)
        for name in names:
            metric.update(
                _image_float_tensor(gen_dir / name).unsqueeze(0).to(device),
                _image_float_tensor(ref_dir / name).unsqueeze(0).to(device),
            )
        results["lpips"] = float(metric.compute().item())
    return results


def _compute_clip_metric(metric_name: str, gen_dir: Path, prompts: dict[str, str], *, device: str) -> float:
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            f"Metric {metric_name!r} requires transformers. Install evaluation dependencies with "
            "python -m pip install -e '.[eval]'."
        ) from exc
    model_id = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_id)
    scores = []
    if metric_name == "clip_iqa":
        anchors = _clip_text_features(model, processor, ["Good photo.", "Bad photo."], device)
    for filename, prompt in prompts.items():
        image = _image_uint8_tensor(gen_dir / f"{filename}.png")
        image_features = _clip_image_features(model, processor, image, device)
        if metric_name == "clip_iqa":
            logits = 100 * image_features @ anchors.t()
            scores.append(float(logits.reshape(-1, 2).softmax(-1)[:, 0].mean().item()))
        elif metric_name == "clip_score":
            text_features = _clip_text_features(model, processor, [prompt], device)
            score = torch.clamp(100 * (image_features * text_features).sum(dim=-1), min=0)
            scores.append(float(score.mean().item()))
        else:
            raise ValueError(f"Unsupported CLIP metric: {metric_name!r}")
    return sum(scores) / len(scores)


def _clip_image_features(model: Any, processor: Any, image: torch.Tensor, device: str) -> torch.Tensor:
    processed = processor(images=[image.cpu()], return_tensors="pt", padding=True)
    with torch.inference_mode():
        features = model.get_image_features(processed["pixel_values"].to(device))
    return _normalize_clip_features(features)


def _clip_text_features(model: Any, processor: Any, texts: Sequence[str], device: str) -> torch.Tensor:
    processed = processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
    kwargs = {key: value.to(device) for key, value in processed.items() if key in {"input_ids", "attention_mask"}}
    with torch.inference_mode():
        features = model.get_text_features(**kwargs)
    return _normalize_clip_features(features)


def _normalize_clip_features(features: Any) -> torch.Tensor:
    tensor = _clip_feature_tensor(features).float()
    return tensor / tensor.norm(p=2, dim=-1, keepdim=True)


def _clip_feature_tensor(features: Any) -> torch.Tensor:
    if torch.is_tensor(features):
        return features
    for name in ("text_embeds", "image_embeds", "pooler_output"):
        value = getattr(features, name, None)
        if torch.is_tensor(value):
            return value
    if isinstance(features, (tuple, list)):
        for value in features:
            if torch.is_tensor(value) and value.ndim == 2:
                return value
    raise TypeError(f"Unsupported CLIP feature output type: {type(features)!r}")


def _compute_image_reward(gen_dir: Path, prompts: dict[str, str]) -> float:
    try:
        _patch_image_reward_transformers_compat()
        import ImageReward as RM
    except ImportError as exc:
        raise RuntimeError(
            "Metric 'image_reward' requires ImageReward. Install it separately or omit image_reward from --metrics."
        ) from exc
    model = RM.load("ImageReward-v1.0")
    scores = []
    for filename, prompt in prompts.items():
        with torch.inference_mode():
            scores.append(float(model.score(prompt, str(gen_dir / f"{filename}.png"))))
    return sum(scores) / len(scores)


def _patch_image_reward_transformers_compat() -> None:
    """Expose legacy BLIP helpers for ImageReward under Transformers 5.x."""

    try:
        from transformers import BertTokenizer
        import transformers.modeling_utils as modeling_utils
        import transformers.pytorch_utils as pytorch_utils
    except ImportError:
        return
    if not hasattr(modeling_utils, "apply_chunking_to_forward"):
        modeling_utils.apply_chunking_to_forward = pytorch_utils.apply_chunking_to_forward
    if not hasattr(modeling_utils, "prune_linear_layer"):
        modeling_utils.prune_linear_layer = pytorch_utils.prune_linear_layer
    if not hasattr(modeling_utils, "find_pruneable_heads_and_indices"):
        modeling_utils.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices
    if not hasattr(modeling_utils.PreTrainedModel, "all_tied_weights_keys"):
        modeling_utils.PreTrainedModel.all_tied_weights_keys = property(
            lambda self: {key: None for key in (getattr(self, "_tied_weights_keys", None) or [])}
        )
    if not hasattr(modeling_utils.PreTrainedModel, "get_head_mask"):
        modeling_utils.PreTrainedModel.get_head_mask = _get_head_mask
    _patch_bert_tokenizer_special_token_ids(BertTokenizer)


def _patch_bert_tokenizer_special_token_ids(tokenizer_cls: type) -> None:
    if not hasattr(tokenizer_cls, "additional_special_tokens"):
        tokenizer_cls.additional_special_tokens = property(
            lambda self: getattr(self, "_image_reward_additional_special_tokens", [])
        )
    if not hasattr(tokenizer_cls, "additional_special_tokens_ids"):
        tokenizer_cls.additional_special_tokens_ids = property(
            lambda self: getattr(self, "_image_reward_additional_special_tokens_ids", [])
        )
    if getattr(tokenizer_cls.add_special_tokens, "_image_reward_compat", False):
        return
    original_add_special_tokens = tokenizer_cls.add_special_tokens

    def add_special_tokens(self: Any, special_tokens_dict: dict[str, Any], *args: Any, **kwargs: Any) -> int:
        added = original_add_special_tokens(self, special_tokens_dict, *args, **kwargs)
        tokens = special_tokens_dict.get("additional_special_tokens")
        if tokens is not None:
            if isinstance(tokens, str):
                tokens = [tokens]
            object.__setattr__(self, "_image_reward_additional_special_tokens", list(tokens))
            object.__setattr__(self, "_image_reward_additional_special_tokens_ids", self.convert_tokens_to_ids(tokens))
        return added

    add_special_tokens._image_reward_compat = True
    tokenizer_cls.add_special_tokens = add_special_tokens


def _find_pruneable_heads_and_indices(
    heads: Sequence[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in heads:
        head -= sum(1 for pruned_head in already_pruned_heads if pruned_head < head)
        mask[head] = 0
    index = torch.arange(len(mask.view(-1)))[mask.view(-1).eq(1)].long()
    return heads, index


def _get_head_mask(self: Any, head_mask: torch.Tensor | None, num_hidden_layers: int, is_attention_chunked: bool = False):
    if head_mask is None:
        return [None] * num_hidden_layers
    if head_mask.dim() == 1:
        head_mask = head_mask[None, None, :, None, None].expand(num_hidden_layers, -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask[:, None, :, None, None]
    if is_attention_chunked:
        head_mask = head_mask.unsqueeze(-1)
    return head_mask.to(dtype=getattr(self, "dtype", torch.float32))


def _compute_fid(ref_dir: Path, gen_dir: Path) -> float:
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError(
            "Metric 'fid' requires clean-fid. Install evaluation dependencies with python -m pip install -e '.[eval]'."
        ) from exc
    return float(fid.compute_fid(str(ref_dir), str(gen_dir), mode="clean"))


def _torchmetrics_image_metric(metric_name: str):
    try:
        from torchmetrics.image import (
            LearnedPerceptualImagePatchSimilarity,
            StructuralSimilarityIndexMeasure,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"Metric {metric_name!r} requires torchmetrics. Install evaluation dependencies with "
            "python -m pip install -e '.[eval]'."
        ) from exc
    if metric_name == "ssim":
        return StructuralSimilarityIndexMeasure(data_range=(0, 1))
    if metric_name == "lpips":
        return LearnedPerceptualImagePatchSimilarity(normalize=True)
    raise ValueError(f"Unsupported torchmetrics image metric: {metric_name!r}")


def _image_float_tensor(path: Path) -> torch.Tensor:
    return _image_uint8_tensor(path).float() / 255.0


def _image_uint8_tensor(path: Path) -> torch.Tensor:
    from PIL import Image

    image = Image.open(path).convert("RGB")
    width, height = image.size
    data = torch.frombuffer(image.tobytes(), dtype=torch.uint8).clone()
    return data.view(height, width, 3).permute(2, 0, 1).contiguous()


def _common_image_names(ref_dir: Path, gen_dir: Path) -> list[str]:
    ref_names = {path.name for path in ref_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    gen_names = {path.name for path in gen_dir.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}}
    return sorted(ref_names & gen_names)


def _resolve_compare_dir(root: str, sample_set_name: str, num_samples: int) -> Path:
    path = Path(root)
    nested = path / "samples" / f"{sample_set_name}-{num_samples}"
    return nested if nested.exists() else path


def _resolve_pipeline_cls(name: str) -> type:
    module_name, class_name = name.rsplit(".", 1) if "." in name else ("diffusers", name)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _sample_set_name(prompt_file: str) -> str:
    source = str(prompt_file)
    if source.startswith(("http://", "https://")):
        return "qdiff"
    return Path(source).stem


def _as_list(value: Any) -> list[Any]:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _make_generator(seed: int, device: str) -> torch.Generator:
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except RuntimeError:
        return torch.Generator().manual_seed(int(seed))


if __name__ == "__main__":
    main()
