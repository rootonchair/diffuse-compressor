from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import torch

from .runtime import patch_quantized_pipeline


logger = logging.getLogger(__name__)


RuntimeName = Literal["none", "nunchaku-lite", "torch-dequant"]
TorchDequantActivationMode = Literal["none", "input"]


@dataclass(frozen=True)
class EvaluationSample:
    """One deterministic generation sample."""

    filename: str
    prompt: str
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"filename": self.filename, "prompt": self.prompt, "seed": self.seed}


@dataclass(frozen=True)
class EvaluationSpec:
    """Configuration for BF16 and optional quantized generation."""

    output_dir: str | Path
    checkpoint: str | Path | None = None
    runtime: RuntimeName = "none"
    precision: str = "int4"
    height: int = 1024
    width: int = 1024
    steps: int = 4
    guidance_scale: float = 0.0
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.bfloat16
    skip_bf16: bool = False
    skip_quantized: bool = False
    torch_dequant_activation_mode: TorchDequantActivationMode = "none"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.checkpoint is not None:
            object.__setattr__(self, "checkpoint", Path(self.checkpoint))
        if self.runtime not in {"none", "nunchaku-lite", "torch-dequant"}:
            raise ValueError(f"Unsupported evaluation runtime: {self.runtime!r}")
        if self.torch_dequant_activation_mode not in {"none", "input"}:
            raise ValueError(f"Unsupported torch-dequant activation mode: {self.torch_dequant_activation_mode!r}")

    def settings(self) -> dict[str, Any]:
        return {
            "height": self.height,
            "width": self.width,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype).replace("torch.", ""),
            "precision": self.precision,
            "skip_bf16": self.skip_bf16,
            "skip_quantized": self.skip_quantized,
            "torch_dequant_activation_mode": self.torch_dequant_activation_mode,
        }


@dataclass(frozen=True)
class EvaluationOutput:
    """Saved output image for one sample."""

    filename: str
    prompt: str
    seed: int
    path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "prompt": self.prompt,
            "seed": self.seed,
            "path": self.path,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation output manifest."""

    model_id: str
    model_key: str
    runtime: RuntimeName
    checkpoint: str | None
    settings: dict[str, Any]
    samples: list[dict[str, Any]]
    bf16: list[EvaluationOutput] = field(default_factory=list)
    quantized: list[EvaluationOutput] = field(default_factory=list)
    bf16_status: str = "generated"
    quantized_status: str = "generated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_key": self.model_key,
            "runtime": self.runtime,
            "checkpoint": self.checkpoint,
            "settings": self.settings,
            "samples": self.samples,
            "bf16": {
                "status": self.bf16_status,
                "outputs": [output.to_dict() for output in self.bf16],
            },
            "quantized": {
                "status": self.quantized_status,
                "outputs": [output.to_dict() for output in self.quantized],
            },
        }

    def save_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def evaluate_pipeline_pair(
    *,
    model_id: str,
    pipeline_cls: type,
    model_key: str,
    samples: Sequence[EvaluationSample],
    spec: EvaluationSpec,
) -> EvaluationResult:
    """Generate BF16 references and optional quantized outputs."""

    sample_list = list(samples)
    if not sample_list:
        raise ValueError("evaluation requires at least one sample")

    bf16_outputs: list[EvaluationOutput] = []
    bf16_status = "skipped" if spec.skip_bf16 else "generated"
    if not spec.skip_bf16:
        logger.info("* Generating BF16 reference images")
        bf16_pipe = _load_pipeline(pipeline_cls, model_id, spec)
        bf16_outputs = generate_images(bf16_pipe, sample_list, spec.output_dir / "bf16", spec)

    quantized_outputs: list[EvaluationOutput] = []
    quantized_status = _quantized_status(spec)
    if quantized_status == "generated":
        logger.info("* Generating quantized images with %s", spec.runtime)
        quantized_pipe = _load_pipeline(pipeline_cls, model_id, spec)
        patch_quantized_pipeline(quantized_pipe, model_key=model_key, spec=spec)
        quantized_outputs = generate_images(quantized_pipe, sample_list, spec.output_dir / "quantized", spec)

    result = EvaluationResult(
        model_id=model_id,
        model_key=model_key,
        runtime=spec.runtime,
        checkpoint=None if spec.checkpoint is None else str(spec.checkpoint),
        settings=spec.settings(),
        samples=[sample.to_dict() for sample in sample_list],
        bf16=bf16_outputs,
        quantized=quantized_outputs,
        bf16_status=bf16_status,
        quantized_status=quantized_status,
    )
    result.save_json(spec.output_dir / "results.json")
    return result


def generate_images(
    pipe: Any,
    samples: Sequence[EvaluationSample],
    output_dir: str | Path,
    spec: EvaluationSpec,
) -> list[EvaluationOutput]:
    """Generate and save one image per sample."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[EvaluationOutput] = []
    for sample in samples:
        result = pipe(
            prompt=sample.prompt,
            height=spec.height,
            width=spec.width,
            num_inference_steps=spec.steps,
            guidance_scale=spec.guidance_scale,
            generator=_make_generator(sample.seed, spec.device),
        )
        images = getattr(result, "images", None)
        if images is None:
            raise ValueError("evaluation pipeline output must expose an images attribute")
        if len(images) != 1:
            raise ValueError(f"Expected one image for sample {sample.filename!r}, got {len(images)}")
        path = output_root / f"{sample.filename}.png"
        images[0].save(path)
        outputs.append(
            EvaluationOutput(
                filename=sample.filename,
                prompt=sample.prompt,
                seed=sample.seed,
                path=str(path),
            )
        )
    return outputs


def _load_pipeline(pipeline_cls: type, model_id: str, spec: EvaluationSpec):
    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=spec.torch_dtype)
    return pipe.to(spec.device) if hasattr(pipe, "to") else pipe


def _make_generator(seed: int, device: str) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def _quantized_status(spec: EvaluationSpec) -> str:
    if spec.skip_quantized:
        return "skipped"
    if spec.runtime == "none":
        return "skipped"
    return "generated"
