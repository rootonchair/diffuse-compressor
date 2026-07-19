import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from examples.convert_nunchaku_lite_diffusers import (
    build_parser,
    build_diffusers_quantization_config,
    derive_output_dir,
    package_diffusers_pipeline,
    quantize_text_encoder_components,
)


def _manifest(*, precision="fp4", patches=None, op="svdq_w4a4"):
    rank = 32 if op == "svdq_w4a4" else 0
    return {
        "schema": "nunchaku_lite.runtime_manifest",
        "version": 1,
        "component": "transformer",
        "structural_patches": patches or [],
        "targets": [{
            "checkpoint_prefix": "proj",
            "source_modules": ["proj"],
            "nunchaku_op": op,
            "precision": precision,
            "group_size": 16 if precision == "fp4" else 64,
            "rank": rank,
            "has_bias": True,
        }],
    }


def _checkpoint(path: Path, manifest):
    target = manifest["targets"][0]
    tensors = {f"proj.{name}": torch.zeros(1) for name in ("qweight", "wscales", "bias")}
    if target["nunchaku_op"] == "svdq_w4a4":
        tensors.update({f"proj.{name}": torch.zeros(1) for name in ("smooth_factor", "proj_down", "proj_up")})
        if target["precision"] == "fp4":
            tensors.update({f"proj.{name}": torch.zeros(1) for name in ("wcscales", "wtscale")})
    else:
        tensors["proj.wzeros"] = torch.zeros(1)
    save_file(tensors, path, metadata={"quantization_config": json.dumps({"runtime_manifest": manifest})})


def test_build_diffusers_config_maps_fp4_to_nvfp4(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _checkpoint(checkpoint, _manifest())

    config = build_diffusers_quantization_config(checkpoint)

    assert config == {
        "quant_method": "nunchaku_lite",
        "compute_dtype": "bfloat16",
        "svdq_w4a4": {"precision": "nvfp4", "group_size": 16, "rank": 32, "targets": ["proj"]},
    }


def test_build_diffusers_config_rejects_structural_patches(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _checkpoint(checkpoint, _manifest(patches=[{"type": "split_linear_output"}]))
    with pytest.raises(ValueError, match="structural patches"):
        build_diffusers_quantization_config(checkpoint)


def test_build_diffusers_config_rejects_missing_tensor(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    manifest = _manifest(precision="int4", op="awq_w4a16")
    save_file(
        {"proj.qweight": torch.zeros(1), "proj.wscales": torch.zeros(1)}, checkpoint,
        metadata={"quantization_config": json.dumps({"runtime_manifest": manifest})},
    )
    with pytest.raises(ValueError, match="wzeros"):
        build_diffusers_quantization_config(checkpoint)


def test_package_full_local_pipeline(tmp_path):
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer" / "config.json").write_text(json.dumps({"_class_name": "TinyTransformer"}))
    (base / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"dense")
    (base / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text("{}")
    (base / "model_index.json").write_text(json.dumps({"_class_name": "TinyPipeline"}))
    (base / "model.safetensors").write_bytes(b"optional single-file weights")
    (base / "scheduler").mkdir()
    (base / "scheduler" / "scheduler_config.json").write_text("{}")
    checkpoint = tmp_path / "model.safetensors"
    _checkpoint(checkpoint, _manifest(precision="int4"))
    output = tmp_path / "output"

    package_diffusers_pipeline(checkpoint, base, output)

    config = json.loads((output / "transformer" / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "nunchaku_lite"
    assert (output / "transformer" / "diffusion_pytorch_model.safetensors").read_bytes() == checkpoint.read_bytes()
    assert not (output / "transformer" / "diffusion_pytorch_model-00001-of-00002.safetensors").exists()
    assert not (output / "transformer" / "diffusion_pytorch_model.safetensors.index.json").exists()
    assert not (output / "model.safetensors").exists()
    assert (output / "scheduler" / "scheduler_config.json").is_file()


def test_package_rejects_existing_output(tmp_path):
    checkpoint = tmp_path / "model.safetensors"
    _checkpoint(checkpoint, _manifest(precision="int4"))
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        package_diffusers_pipeline(checkpoint, tmp_path, output)


def _text_encoder_pipeline(tmp_path: Path) -> Path:
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text(
        json.dumps(
            {
                "text_encoder": ["transformers", "TinyTextEncoder"],
                "text_encoder_2": ["transformers", "TinyTextEncoder"],
                "transformer": ["diffusers", "TinyTransformer"],
            }
        )
    )
    for component in ("text_encoder", "text_encoder_2"):
        directory = pipeline / component
        directory.mkdir()
        (directory / "config.json").write_text("{}")
        (directory / "model.safetensors").write_bytes(f"dense-{component}".encode())
    return pipeline


def test_quantize_selected_text_encoders_as_bnb4(monkeypatch, tmp_path):
    import transformers

    pipeline = _text_encoder_pipeline(tmp_path)
    loads = []
    configs = []

    class FakeBitsAndBytesConfig:
        def __init__(self, **kwargs):
            configs.append(kwargs)

    class FakeModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            loads.append((Path(path).name, kwargs))
            return cls()

        def save_pretrained(self, path, *, safe_serialization):
            path = Path(path)
            path.mkdir()
            (path / "config.json").write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}))
            (path / "model.safetensors").write_bytes(b"bnb4")
            assert safe_serialization

    monkeypatch.setattr(transformers, "BitsAndBytesConfig", FakeBitsAndBytesConfig)
    monkeypatch.setattr(transformers, "TinyTextEncoder", FakeModel, raising=False)

    selected = quantize_text_encoder_components(
        pipeline, ("text_encoder_2", "text_encoder_2", "text_encoder")
    )

    assert selected == ("text_encoder_2", "text_encoder")
    assert [name for name, _ in loads] == ["text_encoder_2", "text_encoder"]
    assert all(call["device_map"] == "auto" and call["low_cpu_mem_usage"] for _, call in loads)
    assert all(config["load_in_4bit"] and config["bnb_4bit_quant_type"] == "nf4" for config in configs)
    assert all(config["bnb_4bit_use_double_quant"] is False for config in configs)
    assert (pipeline / "text_encoder" / "model.safetensors").read_bytes() == b"bnb4"
    assert (pipeline / "text_encoder_2" / "model.safetensors").read_bytes() == b"bnb4"


@pytest.mark.parametrize(
    ("component", "message"),
    [
        ("missing", "must be a text_encoder"),
        ("text_encoder_3", "does not declare"),
        ("transformer", "must be a text_encoder"),
    ],
)
def test_quantize_text_encoder_rejects_invalid_component(tmp_path, component, message):
    pipeline = _text_encoder_pipeline(tmp_path)
    with pytest.raises(ValueError, match=message):
        quantize_text_encoder_components(pipeline, (component,))


def test_quantize_text_encoder_rejects_non_transformers_component(tmp_path):
    pipeline = _text_encoder_pipeline(tmp_path)
    model_index = json.loads((pipeline / "model_index.json").read_text())
    model_index["text_encoder"] = ["diffusers", "TinyTextEncoder"]
    (pipeline / "model_index.json").write_text(json.dumps(model_index))
    with pytest.raises(ValueError, match="must be provided by Transformers"):
        quantize_text_encoder_components(pipeline, ("text_encoder",))


def test_converter_parser_accepts_repeated_bnb4_text_encoders():
    args = build_parser().parse_args(
        [
            "--checkpoint", "model.safetensors",
            "--model-id", "org/model",
            "--output-dir", "output",
            "--bnb4-text-encoder", "text_encoder",
            "--bnb4-text-encoder", "text_encoder_2",
        ]
    )
    assert args.bnb4_text_encoder == ["text_encoder", "text_encoder_2"]


def test_converter_parser_allows_derived_output_dir():
    args = build_parser().parse_args(
        ["--checkpoint", "model.safetensors", "--model-id", "org/model"]
    )
    assert args.output_dir is None


def test_derive_output_dir_from_outputs_checkpoint(tmp_path):
    checkpoint = tmp_path / "outputs" / "checkpoints" / "svdq-int4-model.safetensors"
    assert derive_output_dir(checkpoint, "baidu/ERNIE-Image-Turbo", "int4") == (
        tmp_path / "outputs" / "diffusers" / "ERNIE-Image-Turbo-nunchaku-lite-int4"
    )


def test_derive_output_dir_adds_one_bnb4_suffix(tmp_path):
    checkpoint = tmp_path / "outputs" / "checkpoints" / "unrelated-checkpoint-name.safetensors"
    assert derive_output_dir(
        checkpoint,
        "baidu/ERNIE-Image-Turbo",
        "nvfp4",
        has_bnb4_text_encoder=True,
    ) == (
        tmp_path
        / "outputs"
        / "diffusers"
        / "ERNIE-Image-Turbo-nunchaku-lite-nvfp4-bnb4-text-encoder"
    )


def test_derive_output_dir_for_arbitrary_checkpoint_parent(tmp_path):
    checkpoint = tmp_path / "models" / "model.safetensors"
    model = tmp_path / "local-models" / "My-Model"
    assert derive_output_dir(checkpoint, model, "int4") == (
        tmp_path / "models" / "diffusers" / "My-Model-nunchaku-lite-int4"
    )


def test_package_pipeline_forwards_bnb4_selection(monkeypatch, tmp_path):
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer" / "config.json").write_text(json.dumps({"_class_name": "TinyTransformer"}))
    (base / "model_index.json").write_text(json.dumps({"_class_name": "TinyPipeline"}))
    checkpoint = tmp_path / "model.safetensors"
    _checkpoint(checkpoint, _manifest(precision="int4"))
    output = tmp_path / "output"
    calls = []

    def fake_quantize(pipeline_dir, components):
        calls.append((Path(pipeline_dir), tuple(components)))
        (Path(pipeline_dir) / "bnb4-marker").write_text("ok")
        return tuple(components)

    monkeypatch.setattr(
        "examples.convert_nunchaku_lite_diffusers.quantize_text_encoder_components", fake_quantize
    )
    packaged = package_diffusers_pipeline(
        checkpoint, base, output, bnb4_text_encoders=("text_encoder", "text_encoder_2")
    )

    assert packaged == output
    assert len(calls) == 1
    assert calls[0][1] == ("text_encoder", "text_encoder_2")
    assert (output / "bnb4-marker").read_text() == "ok"


def test_package_pipeline_derives_bnb4_output_dir(monkeypatch, tmp_path):
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer" / "config.json").write_text(json.dumps({"_class_name": "TinyTransformer"}))
    (base / "model_index.json").write_text(json.dumps({"_class_name": "TinyPipeline"}))
    checkpoint = tmp_path / "outputs" / "checkpoints" / "model.safetensors"
    checkpoint.parent.mkdir(parents=True)
    _checkpoint(checkpoint, _manifest(precision="int4"))

    monkeypatch.setattr(
        "examples.convert_nunchaku_lite_diffusers.quantize_text_encoder_components",
        lambda pipeline_dir, components: tuple(components),
    )
    output = package_diffusers_pipeline(
        checkpoint, base, bnb4_text_encoders=("text_encoder", "text_encoder_2")
    )

    assert output == (
        tmp_path / "outputs" / "diffusers" / "base-nunchaku-lite-int4-bnb4-text-encoder"
    )
    assert output.is_dir()


def test_package_pipeline_cleans_up_after_invalid_bnb4_component(tmp_path):
    base = tmp_path / "base"
    (base / "transformer").mkdir(parents=True)
    (base / "transformer" / "config.json").write_text(json.dumps({"_class_name": "TinyTransformer"}))
    (base / "model_index.json").write_text(json.dumps({"_class_name": "TinyPipeline"}))
    checkpoint = tmp_path / "model.safetensors"
    _checkpoint(checkpoint, _manifest(precision="int4"))
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="does not declare"):
        package_diffusers_pipeline(checkpoint, base, output, bnb4_text_encoders=("text_encoder",))

    assert not output.exists()
    assert not list(tmp_path.glob(".output-*"))
