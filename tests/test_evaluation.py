from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

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


def test_torch_dequant_reconstructs_weight_low_rank_and_smoothing():
    qweight = _pack_nibbles(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    state = {
        "proj.qweight": qweight,
        "proj.wscales": torch.tensor([[2.0], [4.0]]),
        "proj.proj_down": torch.tensor([[1.0], [0.0], [0.0], [1.0]]),
        "proj.proj_up": torch.tensor([[2.0]]),
        "proj.smooth_factor": torch.tensor([1.0, 2.0, 1.0, 4.0]),
    }

    weight = runtime_module._reconstruct_target_weight(export_name="proj", state=state, precision="int4")

    expected_residual = torch.tensor([[2.0, 4.0, 12.0, 16.0]])
    expected_low_rank = torch.tensor([[2.0, 0.0, 0.0, 2.0]])
    expected = (expected_residual + expected_low_rank) / state["proj.smooth_factor"].view(1, -1)
    assert torch.allclose(weight, expected)


def test_torch_dequant_decodes_fp4_codebook():
    qweight = _pack_nibbles(torch.tensor([[1, 7, 9, 15]], dtype=torch.long))
    wscales = torch.tensor([[2.0], [4.0]])

    weight = runtime_module._dequantize_qweight(qweight, wscales, precision="fp4")

    assert torch.allclose(weight, torch.tensor([[1.0, 12.0, -2.0, -24.0]]))


def test_torch_dequant_runtime_patches_linear_weights_without_activation_hooks_by_default(tmp_path):
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 1, bias=True)

    transformer = TinyTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    checkpoint = tmp_path / "checkpoint.safetensors"
    qweight = _pack_nibbles(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    tensors = {
        "q_proj.qweight": qweight,
        "q_proj.wscales": torch.tensor([[1.0], [1.0]]),
        "q_proj.smooth_factor": torch.ones(4),
        "q_proj.bias": torch.tensor([0.5]),
    }
    metadata = {
        "weight": {"dtype": "int4", "group_size": 2},
        "targets": [
            {
                "name": "q",
                "export_name": "q_proj",
                "modules": ["q"],
                "roles": [],
                "precision": "int4",
                "group_size": 2,
            }
        ],
    }
    safetensors.torch.save_file(tensors, checkpoint, metadata={"quantization_config": json.dumps(metadata)})

    spec = EvaluationSpec(output_dir=tmp_path, checkpoint=checkpoint, runtime="torch-dequant", device="cpu")
    runtime_module.patch_quantized_pipeline(pipe, model_key="flux.1-schnell", spec=spec)

    assert torch.allclose(transformer.q.weight, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.allclose(transformer.q.bias, torch.tensor([0.5]))
    output = transformer.q(torch.tensor([[0.1, 0.1, 0.1, 0.1]]))
    assert torch.allclose(output, torch.tensor([[1.5]]))
    assert transformer._diffuse_compressor_torch_dequant_hooks == []


def test_torch_dequant_runtime_input_activation_mode_registers_pre_hook(tmp_path):
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 1, bias=True)

    transformer = TinyTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    checkpoint = tmp_path / "checkpoint.safetensors"
    tensors = {
        "q_proj.qweight": _pack_nibbles(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        "q_proj.wscales": torch.tensor([[1.0], [1.0]]),
        "q_proj.smooth_factor": torch.ones(4),
        "q_proj.bias": torch.tensor([0.5]),
    }
    metadata = {
        "weight": {"dtype": "int4", "group_size": 2},
        "targets": [
            {
                "name": "q",
                "export_name": "q_proj",
                "modules": ["q"],
                "roles": [],
                "precision": "int4",
                "group_size": 2,
                "activation_quant": {"enabled": True},
            }
        ],
    }
    safetensors.torch.save_file(tensors, checkpoint, metadata={"quantization_config": json.dumps(metadata)})

    spec = EvaluationSpec(
        output_dir=tmp_path,
        checkpoint=checkpoint,
        runtime="torch-dequant",
        device="cpu",
        torch_dequant_activation_mode="input",
    )
    runtime_module.patch_quantized_pipeline(pipe, model_key="flux.1-schnell", spec=spec)

    output = transformer.q(torch.tensor([[0.1, 0.1, 0.1, 0.1]]))
    assert torch.allclose(output, torch.tensor([[1.5]]))
    assert len(transformer._diffuse_compressor_torch_dequant_hooks) == 1


def test_torch_dequant_runtime_input_activation_mode_uses_dynamic_group_quantization(tmp_path):
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 1, bias=True)

    transformer = TinyTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    checkpoint = tmp_path / "checkpoint.safetensors"
    tensors = {
        "q_proj.qweight": _pack_nibbles(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        "q_proj.wscales": torch.tensor([[1.0], [1.0]]),
        "q_proj.smooth_factor": torch.ones(4),
        "q_proj.bias": torch.tensor([0.5]),
    }
    metadata = {
        "weight": {"dtype": "int4", "group_size": 2},
        "targets": [
            {
                "name": "q",
                "export_name": "q_proj",
                "modules": ["q"],
                "roles": [],
                "precision": "int4",
                "group_size": 2,
                "activation_quant": {"enabled": True},
            }
        ],
    }
    safetensors.torch.save_file(tensors, checkpoint, metadata={"quantization_config": json.dumps(metadata)})

    spec = EvaluationSpec(
        output_dir=tmp_path,
        checkpoint=checkpoint,
        runtime="torch-dequant",
        device="cpu",
        torch_dequant_activation_mode="input",
    )
    runtime_module.patch_quantized_pipeline(pipe, model_key="flux.1-schnell", spec=spec)

    output = transformer.q(torch.tensor([[0.1, 0.05, 0.1, 0.05]]))
    assert torch.allclose(output, torch.tensor([[1.2428572]]))
    assert len(transformer._diffuse_compressor_torch_dequant_hooks) == 1


def test_torch_dequant_runtime_input_activation_mode_applies_smoothing_before_quantization(tmp_path):
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 1, bias=True)

    transformer = TinyTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    checkpoint = tmp_path / "checkpoint.safetensors"
    tensors = {
        "q_proj.qweight": _pack_nibbles(torch.tensor([[2, 2, 2, 2]], dtype=torch.long)),
        "q_proj.wscales": torch.tensor([[1.0], [1.0]]),
        "q_proj.smooth_factor": torch.tensor([2.0, 2.0, 1.0, 1.0]),
        "q_proj.bias": torch.tensor([0.0]),
    }
    metadata = {
        "weight": {"dtype": "int4", "group_size": 2},
        "targets": [
            {
                "name": "q",
                "export_name": "q_proj",
                "modules": ["q"],
                "roles": [],
                "precision": "int4",
                "group_size": 2,
                "activation_quant": {"enabled": True},
            }
        ],
    }
    safetensors.torch.save_file(tensors, checkpoint, metadata={"quantization_config": json.dumps(metadata)})

    spec = EvaluationSpec(
        output_dir=tmp_path,
        checkpoint=checkpoint,
        runtime="torch-dequant",
        device="cpu",
        torch_dequant_activation_mode="input",
    )
    runtime_module.patch_quantized_pipeline(pipe, model_key="flux.1-schnell", spec=spec)

    output = transformer.q(torch.tensor([[0.2, 0.1, 0.2, 0.1]]))
    assert torch.allclose(output, torch.tensor([[0.94285715]]))
    assert len(transformer._diffuse_compressor_torch_dequant_hooks) == 1


def test_torch_dequant_runtime_skips_activation_hooks_for_w4a16_targets(tmp_path):
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.Linear(4, 1, bias=True)

    transformer = TinyTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    checkpoint = tmp_path / "checkpoint.safetensors"
    tensors = {
        "norm.qweight": _pack_nibbles(torch.tensor([[1, 2, 3, 4]], dtype=torch.long)),
        "norm.wscales": torch.tensor([[1.0], [1.0]]),
        "norm.smooth_factor": torch.ones(4),
        "norm.bias": torch.tensor([0.5]),
    }
    metadata = {
        "weight": {"dtype": "int4", "group_size": 2},
        "targets": [
            {
                "name": "norm",
                "export_name": "norm",
                "modules": ["norm"],
                "roles": [],
                "precision": "int4",
                "group_size": 2,
                "activation_quant": {"enabled": False},
            }
        ],
    }
    safetensors.torch.save_file(tensors, checkpoint, metadata={"quantization_config": json.dumps(metadata)})

    spec = EvaluationSpec(
        output_dir=tmp_path,
        checkpoint=checkpoint,
        runtime="torch-dequant",
        device="cpu",
        torch_dequant_activation_mode="input",
    )
    runtime_module.patch_quantized_pipeline(pipe, model_key="flux.1-schnell", spec=spec)

    output = transformer.norm(torch.tensor([[0.1, 0.1, 0.1, 0.1]]))
    assert torch.allclose(output, torch.tensor([[1.5]]))
    assert transformer._diffuse_compressor_torch_dequant_hooks == []


def test_torch_dequant_runtime_replays_activation_shift_wrappers(tmp_path):
    class TinyTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(2, 1, bias=True)
            self.q.weight.data.copy_(torch.tensor([[10.0, 20.0]]))
            self.q.bias.data.copy_(torch.tensor([3.0]))

    transformer = TinyTransformer()
    pipe = SimpleNamespace(transformer=transformer)
    checkpoint = tmp_path / "checkpoint.safetensors"
    tensors = {
        "q_proj.qweight": _pack_nibbles(torch.tensor([[1, 2]], dtype=torch.long)),
        "q_proj.wscales": torch.tensor([[1.0]]),
        "q_proj.smooth_factor": torch.ones(2),
        "q_proj.bias": torch.tensor([-147.0]),
    }
    metadata = {
        "weight": {"dtype": "int4", "group_size": 2},
        "calibration": {"activation_shifts": {"q": 5.0}},
        "targets": [
            {
                "name": "q",
                "export_name": "q_proj",
                "modules": ["q"],
                "roles": [],
                "precision": "int4",
                "group_size": 2,
            }
        ],
    }
    safetensors.torch.save_file(tensors, checkpoint, metadata={"quantization_config": json.dumps(metadata)})

    spec = EvaluationSpec(output_dir=tmp_path, checkpoint=checkpoint, runtime="torch-dequant", device="cpu")
    runtime_module.patch_quantized_pipeline(pipe, model_key="flux.1-schnell", spec=spec)

    assert isinstance(transformer.q, runtime_module.ShiftedLinear)
    assert torch.allclose(transformer.q.linear.bias, torch.tensor([-12.0]))
    assert torch.allclose(transformer.q(torch.tensor([[7.0, 11.0]])), torch.tensor([[32.0]]))


def test_nunchaku_lite_runtime_requires_supported_model(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_module, "_load_nunchaku_lite_patch_transformer", lambda: lambda *args, **kwargs: None)
    spec = EvaluationSpec(output_dir=tmp_path, checkpoint=tmp_path / "checkpoint.safetensors", runtime="nunchaku-lite")

    with pytest.raises(RuntimeError, match="does not support"):
        runtime_module.patch_quantized_pipeline(SimpleNamespace(transformer=object()), model_key="pixart-sigma", spec=spec)


def test_evaluation_cli_parser_imports_without_diffusers():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--model-key",
            "flux.1-schnell",
            "--runtime",
            "torch-dequant",
            "--torch-dequant-activation-mode",
            "input",
            "--num-samples",
            "2",
        ]
    )

    assert args.model_key == "flux.1-schnell"
    assert args.runtime == "torch-dequant"
    assert args.torch_dequant_activation_mode == "input"
    assert args.num_samples == 2


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    lo = codes[:, 0::2].bitwise_and(0xF)
    hi = codes[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    return lo.bitwise_or(hi).to(torch.uint8).view(torch.int8)
