from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from diffuse_compressor import runtime as runtime_module
from diffuse_compressor.runtime import (
    RuntimePipelineSpec,
    load_evaluation_pipeline,
)
from evaluation import evaluate_image_generation as image_generation
from evaluation.datasets import select_names


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


def test_load_evaluation_pipeline_original_from_class():
    FakePipeline.loads = []

    pipe = load_evaluation_pipeline(
        pipeline_cls=FakePipeline,
        model_id="fake/model",
        spec=RuntimePipelineSpec(mode="original", device="cpu", torch_dtype=torch.float32),
    )

    assert pipe is FakePipeline.loads[0]
    assert pipe.model_id == "fake/model"
    assert pipe.torch_dtype is torch.float32
    assert pipe.device == "cpu"
    assert pipe.transformer.patched is False


def test_load_evaluation_pipeline_from_existing_object():
    pipe = FakePipeline("existing", torch.bfloat16)

    loaded = load_evaluation_pipeline(
        pipeline=pipe,
        spec=RuntimePipelineSpec(mode="original", device="cpu"),
    )

    assert loaded is pipe
    assert pipe.device == "cpu"


def test_load_evaluation_pipeline_from_callable():
    calls = []

    def loader():
        calls.append("called")
        return FakePipeline("callable", torch.bfloat16)

    pipe = load_evaluation_pipeline(
        loader=loader,
        spec=RuntimePipelineSpec(mode="original", device="cpu"),
    )

    assert calls == ["called"]
    assert pipe.model_id == "callable"
    assert pipe.device == "cpu"


def test_load_evaluation_pipeline_rejects_ambiguous_sources():
    with pytest.raises(ValueError, match="exactly one pipeline source"):
        load_evaluation_pipeline(
            pipeline=FakePipeline("a", torch.bfloat16),
            loader=lambda: FakePipeline("b", torch.bfloat16),
            spec=RuntimePipelineSpec(mode="original", device="cpu"),
        )


def test_load_evaluation_pipeline_quantized_validates_required_fields(tmp_path):
    with pytest.raises(ValueError, match="runtime"):
        load_evaluation_pipeline(
            pipeline=FakePipeline("model", torch.bfloat16),
            spec=RuntimePipelineSpec(mode="quantized", device="cpu"),
        )
    with pytest.raises(ValueError, match="checkpoint"):
        load_evaluation_pipeline(
            pipeline=FakePipeline("model", torch.bfloat16),
            spec=RuntimePipelineSpec(mode="quantized", runtime="torch-dequant", model_key="flux.1-schnell", device="cpu"),
        )
    with pytest.raises(ValueError, match="model_key"):
        load_evaluation_pipeline(
            pipeline=FakePipeline("model", torch.bfloat16),
            spec=RuntimePipelineSpec(
                mode="quantized",
                runtime="torch-dequant",
                checkpoint=tmp_path / "checkpoint.safetensors",
                device="cpu",
            ),
        )


def test_load_evaluation_pipeline_quantized_patches_nunchaku(monkeypatch, tmp_path):
    calls = []

    def fake_patch_transformer(transformer, checkpoint, **kwargs):
        calls.append((transformer, checkpoint, kwargs))
        transformer.patched = True

    monkeypatch.setattr(runtime_module, "_load_nunchaku_lite_patch_transformer", lambda: fake_patch_transformer)
    pipe = load_evaluation_pipeline(
        pipeline_cls=FakePipeline,
        model_id="fake/model",
        spec=RuntimePipelineSpec(
            mode="quantized",
            runtime="nunchaku-lite",
            checkpoint=tmp_path / "checkpoint.safetensors",
            model_key="flux.1-schnell",
            device="cpu",
        ),
    )

    assert pipe.transformer.patched is True
    assert calls[0][2]["target"] == "flux"


def test_load_evaluation_pipeline_quantized_patches_flux2_nunchaku(monkeypatch, tmp_path):
    calls = []

    def fake_patch_transformer(transformer, checkpoint, **kwargs):
        calls.append((transformer, checkpoint, kwargs))
        transformer.patched = True

    monkeypatch.setattr(runtime_module, "_load_nunchaku_lite_patch_transformer", lambda: fake_patch_transformer)
    pipe = load_evaluation_pipeline(
        pipeline_cls=FakePipeline,
        model_id="fake/model",
        spec=RuntimePipelineSpec(
            mode="quantized",
            runtime="nunchaku-lite",
            checkpoint=tmp_path / "checkpoint.safetensors",
            model_key="flux.2-klein-9b",
            device="cpu",
        ),
    )

    assert pipe.transformer.patched is True
    assert calls[0][2]["target"] == "flux2"


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


def test_torch_dequant_decodes_nvfp4_split_scales():
    qweight = _pack_nibbles(torch.tensor([[1, 7, 9, 15]], dtype=torch.long))
    wscales = torch.tensor([[2.0], [4.0]]).to(dtype=torch.float8_e4m3fn)
    wcscales = torch.tensor([3.0])

    weight = runtime_module._dequantize_qweight(qweight, wscales, precision="fp4", wcscales=wcscales)

    assert torch.allclose(weight, torch.tensor([[3.0, 36.0, -6.0, -72.0]]))


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

    spec = RuntimePipelineSpec(mode="quantized", checkpoint=checkpoint, runtime="torch-dequant", device="cpu")
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

    spec = RuntimePipelineSpec(
        mode="quantized",
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

    spec = RuntimePipelineSpec(
        mode="quantized",
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

    spec = RuntimePipelineSpec(
        mode="quantized",
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

    spec = RuntimePipelineSpec(
        mode="quantized",
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

    spec = RuntimePipelineSpec(mode="quantized", checkpoint=checkpoint, runtime="torch-dequant", device="cpu")
    runtime_module.patch_quantized_pipeline(pipe, model_key="flux.1-schnell", spec=spec)

    assert isinstance(transformer.q, runtime_module.ShiftedLinear)
    assert torch.allclose(transformer.q.linear.bias, torch.tensor([-12.0]))
    assert torch.allclose(transformer.q(torch.tensor([[7.0, 11.0]])), torch.tensor([[32.0]]))


def test_nunchaku_lite_runtime_requires_supported_model(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_module, "_load_nunchaku_lite_patch_transformer", lambda: lambda *args, **kwargs: None)
    spec = RuntimePipelineSpec(mode="quantized", checkpoint=tmp_path / "checkpoint.safetensors", runtime="nunchaku-lite")

    with pytest.raises(RuntimeError, match="does not support"):
        runtime_module.patch_quantized_pipeline(SimpleNamespace(transformer=object()), model_key="pixart-sigma", spec=spec)


def test_image_generation_evaluation_cli_parser_imports_without_diffusers():
    parser = image_generation.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "quantized",
            "--runtime",
            "torch-dequant",
            "--checkpoint",
            "checkpoint.safetensors",
            "--output-dir",
            "outputs/eval/example",
            "--num-samples",
            "2",
            "--metrics",
            "psnr",
            "fid",
        ]
    )

    assert args.mode == "quantized"
    assert args.runtime == "torch-dequant"
    assert args.num_samples == 2
    assert args.metrics == ["psnr", "fid"]


def test_image_generation_evaluation_cli_accepts_mjhq_benchmark():
    parser = image_generation.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "original",
            "--benchmark",
            "MJHQ",
            "--output-dir",
            "outputs/eval/example",
            "--num-samples",
            "2",
            "--metrics",
            "fid",
        ]
    )

    assert args.benchmark == "MJHQ"
    assert args.prompt_file is None
    assert image_generation.MJHQDataset.sample_set_name == "MJHQ"


def test_image_generation_evaluation_cli_accepts_dci_benchmark():
    parser = image_generation.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "original",
            "--benchmark",
            "DCI",
            "--output-dir",
            "outputs/eval/example",
            "--num-samples",
            "2",
            "--metrics",
            "fid",
        ]
    )

    assert args.benchmark == "DCI"
    assert image_generation.DCIDataset.sample_set_name == "sDCI"


def test_image_generation_select_names_matches_deepcompressor_ordering():
    assert select_names(["b", "d", "a", "c"], 2) == ["a", "b"]


def test_image_generation_save_target_images(tmp_path):
    target_dir = image_generation._save_target_images(
        [{"filename": "sample", "target_image": FakeImage("target")}],
        tmp_path / "targets" / "MJHQ-1",
    )

    assert target_dir == tmp_path / "targets" / "MJHQ-1"
    assert (target_dir / "sample.png").read_text(encoding="utf-8") == "target"


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    lo = codes[:, 0::2].bitwise_and(0xF)
    hi = codes[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    return lo.bitwise_or(hi).to(torch.uint8).view(torch.int8)
