from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from diffuse_compressor.evaluation import EvaluationSample, EvaluationSpec, evaluate_pipeline_pair, generate_images
from diffuse_compressor.evaluation import runtime as runtime_module
from examples.evaluate_upstream_diffusion import build_parser


class FakeImage:
    def __init__(self, content: str):
        self.content = content
        self.path = None

    def save(self, path):
        self.path = path
        path.write_text(self.content, encoding="utf-8")


class FakePipeline:
    loads = []

    def __init__(self, model_id: str, torch_dtype: torch.dtype):
        self.model_id = model_id
        self.torch_dtype = torch_dtype
        self.device = None
        self.transformer = SimpleNamespace(patched=False)

    @classmethod
    def from_pretrained(cls, model_id: str, torch_dtype: torch.dtype):
        pipe = cls(model_id, torch_dtype)
        cls.loads.append(pipe)
        return pipe

    def to(self, device: str):
        self.device = device
        return self

    def __call__(self, *, prompt, height, width, num_inference_steps, guidance_scale, generator):
        image = FakeImage(f"{prompt}|{height}x{width}|{num_inference_steps}|{guidance_scale}|{self.device}")
        return SimpleNamespace(images=[image])


def test_generate_images_writes_outputs(tmp_path):
    pipe = FakePipeline("model", torch.bfloat16).to("cpu")
    spec = EvaluationSpec(output_dir=tmp_path, device="cpu", height=32, width=48, steps=2, guidance_scale=1.5)
    samples = [EvaluationSample(filename="0000-0", prompt="a prompt", seed=123)]

    outputs = generate_images(pipe, samples, tmp_path / "bf16", spec)

    assert outputs[0].filename == "0000-0"
    assert (tmp_path / "bf16" / "0000-0.png").read_text(encoding="utf-8") == "a prompt|32x48|2|1.5|cpu"


def test_evaluate_pipeline_pair_runtime_none_writes_manifest(tmp_path):
    FakePipeline.loads = []
    samples = [
        EvaluationSample(filename="0000-0", prompt="prompt 0", seed=0),
        EvaluationSample(filename="0001-0", prompt="prompt 1", seed=1),
    ]
    spec = EvaluationSpec(output_dir=tmp_path, runtime="none", device="cpu", height=16, width=16)

    result = evaluate_pipeline_pair(
        model_id="fake/model",
        pipeline_cls=FakePipeline,
        model_key="flux.1-schnell",
        samples=samples,
        spec=spec,
    )

    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert len(FakePipeline.loads) == 1
    assert result.quantized_status == "skipped"
    assert data["bf16"]["status"] == "generated"
    assert data["quantized"]["status"] == "skipped"
    assert data["samples"][1]["filename"] == "0001-0"
    assert (tmp_path / "bf16" / "0000-0.png").exists()
    assert not (tmp_path / "quantized").exists()


def test_evaluate_pipeline_pair_patches_quantized_runtime(monkeypatch, tmp_path):
    FakePipeline.loads = []
    calls = []

    def fake_patch_transformer(transformer, checkpoint, **kwargs):
        calls.append((transformer, checkpoint, kwargs))
        transformer.patched = True

    monkeypatch.setattr(runtime_module, "_load_nunchaku_lite_patch_transformer", lambda: fake_patch_transformer)
    samples = [EvaluationSample(filename="0000-0", prompt="prompt", seed=0)]
    spec = EvaluationSpec(
        output_dir=tmp_path,
        checkpoint=tmp_path / "checkpoint.safetensors",
        runtime="nunchaku-lite",
        device="cpu",
    )

    result = evaluate_pipeline_pair(
        model_id="fake/model",
        pipeline_cls=FakePipeline,
        model_key="flux.1-schnell",
        samples=samples,
        spec=spec,
    )

    assert len(FakePipeline.loads) == 2
    assert result.quantized_status == "generated"
    assert calls[0][2]["target"] == "flux"
    assert (tmp_path / "quantized" / "0000-0.png").exists()


def test_nunchaku_lite_runtime_requires_supported_model(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_module, "_load_nunchaku_lite_patch_transformer", lambda: lambda *args, **kwargs: None)
    spec = EvaluationSpec(output_dir=tmp_path, checkpoint=tmp_path / "checkpoint.safetensors", runtime="nunchaku-lite")

    with pytest.raises(RuntimeError, match="does not support"):
        runtime_module.patch_quantized_pipeline(SimpleNamespace(transformer=object()), model_key="pixart-sigma", spec=spec)


def test_evaluation_cli_parser_imports_without_diffusers():
    parser = build_parser()
    args = parser.parse_args(["--model-key", "flux.1-schnell", "--runtime", "none", "--num-samples", "2"])

    assert args.model_key == "flux.1-schnell"
    assert args.runtime == "none"
    assert args.num_samples == 2
