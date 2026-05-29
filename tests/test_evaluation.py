from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
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
import evaluation.datasets.image_edit as image_edit_dataset_module
from evaluation.datasets import select_names


class FakeImage:
    def __init__(self, content: str):
        self.content = content
        self.path = None
        self.size = (640, 512)

    def save(self, path):
        self.path = path
        path.write_text(self.content, encoding="utf-8")

    def crop(self, box):
        self.box = box
        return self

    def resize(self, size):
        self.size = size
        return self

    def convert(self, mode):
        self.mode = mode
        return self


class FakePipeline:
    loads = []

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.device = None
        self.offload = None
        self.transformer = SimpleNamespace(patched=False)

    @classmethod
    def from_pretrained(cls, model_id: str):
        pipe = cls(model_id)
        cls.loads.append(pipe)
        return pipe

    def to(self, device: str):
        self.device = device
        return self

    def enable_model_cpu_offload(self, *, device):
        self.offload = ("model", device)

    def enable_sequential_cpu_offload(self, *, device):
        self.offload = ("sequential", device)

    def __call__(self, *, prompt, height, width, num_inference_steps, guidance_scale, generator):
        image = FakeImage(f"{prompt}|{height}x{width}|{num_inference_steps}|{guidance_scale}|{self.device}")
        return SimpleNamespace(images=[image])


def _required_eval_cli_args(*, task: str = "text-to-image") -> list[str]:
    args = [
        "--model-id",
        "fake/model",
        "--steps",
        "4",
        "--guidance-scale",
        "1.0",
    ]
    if task != "text-to-image":
        args.extend(["--task", task])
    return args


def test_load_evaluation_pipeline_original_from_class():
    FakePipeline.loads = []

    pipe = load_evaluation_pipeline(
        pipeline_cls=FakePipeline,
        model_id="fake/model",
        spec=RuntimePipelineSpec(mode="original", device="cpu"),
    )

    assert pipe is FakePipeline.loads[0]
    assert pipe.model_id == "fake/model"
    assert pipe.device == "cpu"
    assert pipe.transformer.patched is False


def test_load_evaluation_pipeline_from_existing_object():
    pipe = FakePipeline("existing")

    loaded = load_evaluation_pipeline(
        pipeline=pipe,
        spec=RuntimePipelineSpec(mode="original", device="cpu"),
    )

    assert loaded is pipe
    assert pipe.device == "cpu"


def test_load_evaluation_pipeline_uses_requested_pipeline_offload():
    pipe = FakePipeline("existing")

    loaded = load_evaluation_pipeline(
        pipeline=pipe,
        spec=RuntimePipelineSpec(mode="original", device="cuda", pipeline_offload="model"),
    )

    assert loaded is pipe
    assert pipe.device is None
    assert pipe.offload == ("model", "cuda")


def test_load_evaluation_pipeline_from_callable():
    calls = []

    def loader():
        calls.append("called")
        return FakePipeline("callable")

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
            pipeline=FakePipeline("a"),
            loader=lambda: FakePipeline("b"),
            spec=RuntimePipelineSpec(mode="original", device="cpu"),
        )


def test_load_evaluation_pipeline_quantized_validates_required_fields(tmp_path):
    with pytest.raises(ValueError, match="runtime"):
        load_evaluation_pipeline(
            pipeline=FakePipeline("model"),
            spec=RuntimePipelineSpec(mode="quantized", device="cpu"),
        )
    with pytest.raises(ValueError, match="checkpoint"):
        load_evaluation_pipeline(
            pipeline=FakePipeline("model"),
            spec=RuntimePipelineSpec(mode="quantized", runtime="torch-dequant", device="cpu"),
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
            nunchaku_lite_target="flux",
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
            nunchaku_lite_target="flux2",
            device="cpu",
        ),
    )

    assert pipe.transformer.patched is True
    assert calls[0][2]["target"] == "flux2"


def test_load_evaluation_pipeline_quantized_patches_longcat_with_manifest(monkeypatch, tmp_path):
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
            nunchaku_lite_target="manifest",
            device="cpu",
        ),
    )

    assert pipe.transformer.patched is True
    assert calls[0][2]["target"] == "manifest"


def test_image_generation_evaluation_infers_nunchaku_lite_target():
    assert image_generation.infer_nunchaku_lite_target("black-forest-labs/FLUX.1-schnell") == "flux"
    assert image_generation.infer_nunchaku_lite_target("black-forest-labs/FLUX.2-klein-9B") == "flux2"
    assert image_generation.infer_nunchaku_lite_target("meituan-longcat/LongCat-Image-Edit-Turbo") == "manifest"
    assert image_generation.infer_nunchaku_lite_target("baidu/ERNIE-Image") == "manifest"
    assert image_generation.infer_nunchaku_lite_target("baidu/ERNIE-Image-Turbo") == "manifest"


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
    runtime_module.patch_quantized_pipeline(pipe, spec=spec)

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
    runtime_module.patch_quantized_pipeline(pipe, spec=spec)

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
    runtime_module.patch_quantized_pipeline(pipe, spec=spec)

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
    runtime_module.patch_quantized_pipeline(pipe, spec=spec)

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
    runtime_module.patch_quantized_pipeline(pipe, spec=spec)

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
    runtime_module.patch_quantized_pipeline(pipe, spec=spec)

    assert isinstance(transformer.q, runtime_module.ShiftedLinear)
    assert torch.allclose(transformer.q.linear.bias, torch.tensor([-12.0]))
    assert torch.allclose(transformer.q(torch.tensor([[7.0, 11.0]])), torch.tensor([[32.0]]))


def test_nunchaku_lite_runtime_requires_explicit_target(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_module, "_load_nunchaku_lite_patch_transformer", lambda: lambda *args, **kwargs: None)
    spec = RuntimePipelineSpec(mode="quantized", checkpoint=tmp_path / "checkpoint.safetensors", runtime="nunchaku-lite")

    with pytest.raises(RuntimeError, match="nunchaku_lite_target"):
        runtime_module.patch_quantized_pipeline(SimpleNamespace(transformer=object()), spec=spec)


def test_image_generation_evaluation_cli_parser_imports_without_diffusers():
    parser = image_generation.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "quantized",
            *_required_eval_cli_args(),
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
            *_required_eval_cli_args(),
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
            *_required_eval_cli_args(),
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


def test_image_generation_evaluation_cli_accepts_longcat_edit_benchmark():
    parser = image_generation.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "quantized",
            *_required_eval_cli_args(task="image-edit"),
            "--runtime",
            "nunchaku-lite",
            "--checkpoint",
            "checkpoint.safetensors",
            "--benchmark",
            "NHR-Edit-Change_Only",
            "--output-dir",
            "outputs/eval/example",
            "--num-samples",
            "2",
            "--metrics",
            "fid",
        ]
    )

    assert args.benchmark == "NHR-Edit-Change_Only"
    assert args.image_edit_dataset == "VyoJ/NHR-Edit-Change_Only"
    assert args.image_edit_split == "test"
    assert image_generation.LongCatImageEditDataset.sample_set_name == "NHR-Edit-Change_Only"


def test_image_generation_evaluation_cli_accepts_ernie_model_id():
    parser = image_generation.build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "quantized",
            *_required_eval_cli_args(),
            "--runtime",
            "nunchaku-lite",
            "--checkpoint",
            "checkpoint.safetensors",
            "--output-dir",
            "outputs/eval/example",
            "--num-samples",
            "2",
            "--metrics",
            "fid",
        ]
    )

    assert args.model_id == "fake/model"


def test_evaluation_scripts_do_not_import_examples_or_each_other():
    evaluation_dir = Path(image_generation.__file__).resolve().parent
    for script in ("evaluate_image_generation.py",):
        tree = ast.parse((evaluation_dir / script).read_text(encoding="utf-8"), filename=script)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append(node.module)

        assert not [module for module in imports if module == "examples" or module.startswith("examples.")]
        assert not [module for module in imports if module.startswith("evaluation.evaluate_image_")]


def test_longcat_image_edit_dataset_loads_test_split_and_targets(monkeypatch):
    calls = []
    rows = [
        {
            "sample_id": 17,
            "source": FakeImage("source"),
            "edited": FakeImage("target"),
            "edit_instruction": "make it brighter",
        }
    ]

    def fake_load_dataset(dataset, *, split):
        calls.append((dataset, split))
        return rows

    monkeypatch.setattr(image_edit_dataset_module, "require_benchmark_dependencies", lambda: None)
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))

    dataset = image_generation.LongCatImageEditDataset(1)
    sample = dataset[0]

    assert calls == [("VyoJ/NHR-Edit-Change_Only", "test")]
    assert sample["filename"] == "17"
    assert sample["prompt"] == "make it brighter"
    assert sample["image"].content == "source"
    assert sample["image"].size == (512, 512)
    assert dataset.records[0]["target_image"].content == "target"


def test_generate_images_uses_image_edit_pipeline_signature(tmp_path):
    class FakeImageEditPipeline:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            assert "height" not in kwargs
            assert "width" not in kwargs
            return SimpleNamespace(images=[FakeImage("generated")])

    pipe = FakeImageEditPipeline()
    image_generation._generate_images(
        pipe,
        [
            {
                "filename": ["sample"],
                "prompt": ["make it brighter"],
                "seed": [123],
                "image": [FakeImage("source")],
            }
        ],
        tmp_path,
        task="image-edit",
        height=512,
        width=512,
        steps=8,
        guidance_scale=1.0,
        device="cpu",
    )

    assert pipe.calls[0]["image"][0].content == "source"
    assert pipe.calls[0]["prompt"] == ["make it brighter"]
    assert pipe.calls[0]["negative_prompt"] == ""
    assert (tmp_path / "sample.png").read_text(encoding="utf-8") == "generated"


def test_generate_images_image_edit_can_use_pipeline_native_dimensions(monkeypatch, tmp_path):
    calls = []

    def fake_call_image_edit_pipeline(pipe, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=[FakeImage("generated")])

    monkeypatch.setattr(image_generation, "_call_image_edit_pipeline", fake_call_image_edit_pipeline)
    image_generation._generate_images(
        SimpleNamespace(),
        [
            {
                "filename": ["sample"],
                "prompt": ["make it brighter"],
                "seed": [123],
                "image": [FakeImage("source")],
            }
        ],
        tmp_path,
        task="image-edit",
        height=None,
        width=None,
        steps=8,
        guidance_scale=1.0,
        device="cpu",
    )

    assert calls[0]["height"] is None
    assert calls[0]["width"] is None


def test_generate_images_can_disable_ernie_prompt_enhancer(tmp_path):
    class FakeErniePipeline:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(images=[FakeImage("generated")])

    pipe = FakeErniePipeline()
    image_generation._generate_images(
        pipe,
        [
            {
                "filename": ["sample"],
                "prompt": ["a quiet studio"],
                "seed": [123],
            }
        ],
        tmp_path,
        task="text-to-image",
        height=1024,
        width=1024,
        steps=8,
        guidance_scale=1.0,
        device="cpu",
        use_pe=False,
    )

    assert pipe.calls[0]["use_pe"] is False
    assert pipe.calls[0]["prompt"] == ["a quiet studio"]
    assert (tmp_path / "sample.png").read_text(encoding="utf-8") == "generated"


def test_pair_metrics_resize_reference_to_generated_size(tmp_path):
    from PIL import Image

    ref_dir = tmp_path / "ref"
    gen_dir = tmp_path / "gen"
    ref_dir.mkdir()
    gen_dir.mkdir()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(ref_dir / "sample.png")
    Image.new("RGB", (4, 4), (255, 0, 0)).save(gen_dir / "sample.png")

    metrics = image_generation._compute_pair_metrics({"psnr"}, ref_dir, gen_dir, device="cpu")

    assert metrics["psnr"] == float("inf")


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
