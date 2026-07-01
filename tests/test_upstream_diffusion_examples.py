import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import safetensors
import torch

from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    AwqW4A16Layout,
    DiffusionQuantSpec,
    ExportSpec,
    LoggingConfig,
    NunchakuSvdqLayout,
    AwqTargetQuant,
    TargetConfig,
    TargetRule,
    collect_quant_targets,
    prepare_model,
    quantize_and_export,
)
from diffuse_compressor.calibration.scope_rules import assign_calibration_scopes
from diffuse_compressor.runtime import RuntimePipelineSpec, patch_quantized_pipeline
import examples.text_to_image.quantize_ernie_image as ernie_example
import examples.text_to_image.quantize_ernie_image_turbo as cli_example
import examples.text_to_image.quantize_flux2_klein_4b as flux2_example
import examples.text_to_image.quantize_lens_turbo as lens_example
import examples.text_to_image.quantize_pixart_sigma as pixart_example
import examples.text_to_image.quantize_sana_1_6b as sana_example
import examples.image_to_image.quantize_longcat_image_edit as image_edit_example
from examples.text_to_image.quantize_ernie_image import ernie_image_target_config
from examples.text_to_image.quantize_flux1_schnell import flux1_target_config
from examples.text_to_image.quantize_flux2_klein_4b import flux2_klein_target_config
from examples.text_to_image.quantize_lens_turbo import lens_turbo_target_config
from examples.image_to_image.quantize_longcat_image_edit import (
    image_edit_forward_fn,
    image_edit_records,
    longcat_image_edit_target_config,
)
from examples.text_to_image.quantize_pixart_sigma import pixart_sigma_target_config
from examples.text_to_image.quantize_sana_1_6b import sana_target_config
from examples.text_to_image.quantize_ernie_image_turbo import (
    default_arg_parser,
    load_pipeline,
    pipeline_forward_fn,
    run_model_cli,
    svdquant_spec,
)


TEXT_TO_VIDEO_EXAMPLE_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "text_to_video"
)
sys.modules.pop("utils", None)
sys.path.insert(0, str(TEXT_TO_VIDEO_EXAMPLE_DIR))
try:
    import merge_ltx2_3_distilled_lora as ltx2_lora_merge_example
    import quantize_ltx2_3 as ltx2_example
    import quantize_ltx2_3_distilled as ltx2_distilled_example
    import quantize_ltx2_3_safetensors as ltx2_safetensors_example
    from quantize_ltx2_3 import ltx2_3_target_config
finally:
    sys.path.remove(str(TEXT_TO_VIDEO_EXAMPLE_DIR))
    sys.modules.pop("utils", None)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None, reason="diffusers is not installed"
)


def _config_metadata(checkpoint_path: str | Path) -> dict:
    return json.loads(
        Path(checkpoint_path).with_suffix(".config.yaml").read_text(encoding="utf-8")
    )


def _checkpoint_quantization_config(checkpoint_path: str | Path) -> dict | None:
    with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        metadata_blob = handle.metadata().get("quantization_config")
    return None if metadata_blob is None else json.loads(metadata_blob)


def _assert_checkpoint_quantization_config(
    checkpoint_metadata: dict,
    config_metadata: dict,
    *,
    has_runtime_manifest: bool = False,
) -> None:
    expected_keys = {"method", "rank", "weight", "activation"}
    if has_runtime_manifest:
        expected_keys.add("runtime_manifest")
    assert set(checkpoint_metadata) == expected_keys
    assert checkpoint_metadata["method"] == config_metadata["method"]
    assert checkpoint_metadata["rank"] == config_metadata["rank"]
    assert checkpoint_metadata["weight"] == config_metadata["weight"]
    assert checkpoint_metadata["activation"] == config_metadata["activation"]


def test_nvfp4_upstream_spec_uses_fp4_overlay():
    spec = svdquant_spec("nvfp4")

    assert spec.precision == "fp4"
    assert spec.group_size == 16


def test_upstream_parser_exposes_offload_flags():
    parser = default_arg_parser(
        "model",
        "output.safetensors",
        steps=4,
        guidance_scale=1.0,
        batch_size=2,
    )

    args = parser.parse_args(
        [
            "--offload-model",
            "--compute-device",
            "cuda",
            "--pipeline-offload",
            "model",
            "--sample-batch-size",
            "32",
            "--scope-capture-mode",
            "one-target",
            "--cache-num-samples",
            "64",
            "--log-dir",
            "run-logs",
            "--no-run-log",
        ]
    )

    assert args.offload_model is True
    assert args.compute_device == "cuda"
    assert args.pipeline_offload == "model"
    assert args.sample_batch_size == 32
    assert args.scope_capture_mode == "one-target"
    assert args.cache_num_samples == 64
    assert args.log_dir == "run-logs"
    assert args.no_run_log is True

    override = parser.parse_args([])
    assert override.sample_batch_size is None
    assert override.scope_capture_mode == "all-targets"
    assert override.cache_num_samples is None
    assert override.log_dir == "outputs/logs"
    assert override.no_run_log is False


def test_run_model_cli_passes_independent_sample_batch_size(monkeypatch, tmp_path):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["batch_size"] = calibration.batch_size
        captured["sample_batch_size"] = calibration.sample_batch_size
        captured["cache_num_samples"] = calibration.cache_num_samples
        captured["scope_capture_mode"] = calibration.scope_capture_mode
        captured["max_rows_per_target"] = calibration.max_rows_per_target
        captured["model"] = model
        captured["export"] = export.output
        captured["logging"] = logging

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--num-samples",
            "0",
            "--cache-num-samples",
            "4",
            "--batch-size",
            "1",
            "--sample-batch-size",
            "8",
            "--scope-capture-mode",
            "one-target",
            "--cache-mode",
            "disabled",
            "--output",
            str(tmp_path / "out.safetensors"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    monkeypatch.setattr(cli_example, "quantize_and_export", fake_quantize_and_export)
    pipe = type("FakePipe", (), {"transformer": torch.nn.Linear(1, 1)})()
    monkeypatch.setattr(cli_example, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(
        cli_example, "standard_prompt_records", lambda num_samples, prompt_file: []
    )
    monkeypatch.setattr(
        cli_example, "pipeline_forward_fn", lambda *args, **kwargs: lambda sample: None
    )

    run_model_cli()

    assert captured["batch_size"] == 1
    assert captured["sample_batch_size"] == 8
    assert captured["cache_num_samples"] == 4
    assert captured["scope_capture_mode"] == "one_target"
    assert captured["max_rows_per_target"] == 4096
    assert captured["model"] is pipe.transformer
    assert captured["export"] == tmp_path / "out.safetensors"
    assert captured["logging"].log_dir == str(tmp_path / "logs")
    assert captured["logging"].name == "out"


def test_ernie_image_run_model_cli_caps_rows_per_target(monkeypatch, tmp_path):
    captured = []

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured.append({"calibration": calibration, "export": export})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--num-samples",
            "1",
            "--cache-mode",
            "disabled",
            "--output",
            str(tmp_path / "ernie.safetensors"),
        ],
    )
    monkeypatch.setattr(ernie_example, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(
        ernie_example,
        "load_pipeline",
        lambda *args, **kwargs: type("FakePipe", (), {"transformer": object()})(),
    )
    monkeypatch.setattr(
        ernie_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": "0000-0", "prompt": "prompt", "seed": 0}
        ],
    )
    monkeypatch.setattr(
        ernie_example, "batched_samples", lambda records, batch_size: records
    )
    monkeypatch.setattr(
        ernie_example,
        "pipeline_forward_fn",
        lambda *args, **kwargs: lambda sample: None,
    )

    ernie_example.run_model_cli()

    assert captured[0]["calibration"].max_rows_per_target == 4096
    assert captured[0]["export"].output == tmp_path / "ernie.safetensors"


def test_run_model_cli_defaults_cache_num_samples_to_num_samples(monkeypatch):
    captured = []

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured.append({"calibration": calibration, "logging": logging})

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--num-samples", "9", "--batch-size", "3", "--cache-mode", "disabled"],
    )
    monkeypatch.setattr(cli_example, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(
        cli_example,
        "load_pipeline",
        lambda *args, **kwargs: type("FakePipe", (), {"transformer": object()})(),
    )
    monkeypatch.setattr(
        cli_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"prompt": str(i)} for i in range(num_samples)
        ],
    )
    monkeypatch.setattr(
        cli_example, "batched_samples", lambda records, batch_size: records
    )
    monkeypatch.setattr(
        cli_example, "pipeline_forward_fn", lambda *args, **kwargs: lambda sample: None
    )

    run_model_cli()

    assert captured[0]["calibration"].num_samples == 9
    assert captured[0]["calibration"].cache_num_samples == 9
    assert captured[0]["logging"] == LoggingConfig(
        log_dir="outputs/logs", name="svdq-int4_r32-ernie-image-turbo"
    )


def test_run_model_cli_preserves_all_cache_num_samples_sentinel(monkeypatch):
    captured = []

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured.append(calibration)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--num-samples",
            "9",
            "--cache-num-samples",
            "-1",
            "--cache-mode",
            "disabled",
        ],
    )
    monkeypatch.setattr(cli_example, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(
        cli_example,
        "load_pipeline",
        lambda *args, **kwargs: type("FakePipe", (), {"transformer": object()})(),
    )
    monkeypatch.setattr(
        cli_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"prompt": str(i)} for i in range(num_samples)
        ],
    )
    monkeypatch.setattr(
        cli_example, "batched_samples", lambda records, batch_size: records
    )
    monkeypatch.setattr(
        cli_example, "pipeline_forward_fn", lambda *args, **kwargs: lambda sample: None
    )

    run_model_cli()

    assert captured[0].cache_num_samples == -1


def test_run_model_cli_can_disable_default_run_logging(monkeypatch):
    captured = []

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured.append(logging)

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--num-samples", "1", "--cache-mode", "disabled", "--no-run-log"],
    )
    monkeypatch.setattr(cli_example, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(
        cli_example,
        "load_pipeline",
        lambda *args, **kwargs: type("FakePipe", (), {"transformer": object()})(),
    )
    monkeypatch.setattr(
        cli_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"prompt": str(i)} for i in range(num_samples)
        ],
    )
    monkeypatch.setattr(
        cli_example, "batched_samples", lambda records, batch_size: records
    )
    monkeypatch.setattr(
        cli_example, "pipeline_forward_fn", lambda *args, **kwargs: lambda sample: None
    )

    run_model_cli()

    assert captured[0].enabled is False


def test_run_model_cli_can_inspect_config_without_quantizing(monkeypatch, capsys):
    class FakeReport:
        def format_text(self):
            return "inspected target config"

    def fail_quantize_and_export(**_kwargs):
        raise AssertionError(
            "quantize_and_export should not run during config inspection"
        )

    pipe = type("FakePipe", (), {"transformer": torch.nn.Linear(1, 1)})()
    monkeypatch.setattr(
        sys, "argv", ["prog", "--inspect-config", "--cache-mode", "disabled"]
    )
    monkeypatch.setattr(cli_example, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(
        cli_example, "ernie_image_target_config", lambda precision: object()
    )
    monkeypatch.setattr(
        cli_example, "inspect_target_config", lambda model, target_config: FakeReport()
    )
    monkeypatch.setattr(cli_example, "quantize_and_export", fail_quantize_and_export)

    run_model_cli()

    assert "inspected target config" in capsys.readouterr().out


def test_image_edit_forward_fn_overrides_longcat_target_dimensions(monkeypatch):
    module_name = "tests.fake_longcat_pipeline"
    fake_module = ModuleType(module_name)

    def calculate_dimensions(target_area, ratio):
        return 1024, 1024

    fake_module.calculate_dimensions = calculate_dimensions
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    calls = []

    class FakePipe:
        def __call__(self, **kwargs):
            calls.append(fake_module.calculate_dimensions(1024 * 1024, 1.0))
            return kwargs

    FakePipe.__module__ = module_name
    forward = image_edit_forward_fn(
        FakePipe(), steps=4, guidance_scale=1.0, device="cpu", height=384, width=512
    )

    result = forward({"image": "image", "prompt": "prompt", "seed": 0})

    assert calls == [(512, 384)]
    assert result["num_inference_steps"] == 4
    assert result["guidance_scale"] == 1.0
    assert fake_module.calculate_dimensions is calculate_dimensions


def test_load_pipeline_uses_requested_diffusers_cpu_offload(monkeypatch):
    import diffusers

    calls = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("from_pretrained", model_id, kwargs))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

        def enable_model_cpu_offload(self, *, device):
            calls.append(("model_offload", device))

        def enable_sequential_cpu_offload(self, *, device):
            calls.append(("sequential_offload", device))

    monkeypatch.setattr(diffusers, "FakePipeline", FakePipeline, raising=False)

    pipe = load_pipeline(
        "FakePipeline", "fake/model", device="cuda", pipeline_offload="model"
    )

    assert isinstance(pipe, FakePipeline)
    assert calls == [
        ("from_pretrained", "fake/model", {}),
        ("model_offload", "cuda"),
    ]


def test_load_pipeline_passes_requested_torch_dtype(monkeypatch):
    import diffusers

    calls = []

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("from_pretrained", model_id, kwargs))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

    monkeypatch.setattr(diffusers, "FakePipeline", FakePipeline, raising=False)

    pipe = load_pipeline(
        "FakePipeline", "fake/model", device="cuda", dtype=torch.bfloat16
    )

    assert isinstance(pipe, FakePipeline)
    assert calls == [
        ("from_pretrained", "fake/model", {"torch_dtype": torch.bfloat16}),
        ("to", "cuda"),
    ]


def test_pipeline_forward_fn_can_disable_ernie_prompt_enhancer():
    calls = []

    class FakeErniePipe:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return object()

    forward = pipeline_forward_fn(
        FakeErniePipe(),
        height=1024,
        width=1024,
        steps=8,
        guidance_scale=1.0,
        device="cpu",
        use_pe=False,
    )
    forward({"prompt": "a quiet studio", "seed": 7})

    assert calls[0]["use_pe"] is False
    assert calls[0]["num_inference_steps"] == 8
    assert calls[0]["guidance_scale"] == 1.0


def test_lens_turbo_run_model_cli_wires_calibration(monkeypatch, tmp_path):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["spec"] = spec
        captured["target_config"] = target_config
        captured["calibration"] = calibration
        captured["export"] = export
        captured["logging"] = logging

    transformer = torch.nn.Linear(1, 1)

    class FakeLensPipe:
        def __init__(self, transformer):
            self.transformer = transformer

        def __call__(self, **kwargs):
            return kwargs

    pipe = FakeLensPipe(transformer)
    target_config = object()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--num-samples",
            "2",
            "--batch-size",
            "1",
            "--sample-batch-size",
            "4",
            "--cache-mode",
            "disabled",
            "--base-resolution",
            "1440",
            "--aspect-ratio",
            "16:9",
            "--output",
            str(tmp_path / "lens.safetensors"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    monkeypatch.setattr(lens_example, "load_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(
        lens_example, "lens_turbo_target_config", lambda precision: target_config
    )
    monkeypatch.setattr(
        lens_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": f"{index:04d}-0", "prompt": str(index), "seed": index}
            for index in range(num_samples)
        ],
    )
    monkeypatch.setattr(lens_example, "quantize_and_export", fake_quantize_and_export)

    lens_example.run_model_cli()

    assert captured["model"] is transformer
    assert captured["target_config"] is target_config
    assert captured["calibration"].num_samples == 2
    assert captured["calibration"].cache_num_samples == 2
    assert captured["calibration"].batch_size == 1
    assert captured["calibration"].sample_batch_size == 4
    assert captured["export"].output == tmp_path / "lens.safetensors"
    assert captured["logging"].log_dir == str(tmp_path / "logs")
    assert captured["logging"].name == "lens"
    result = captured["calibration"].forward_fn({"prompt": "lens prompt", "seed": 3})
    assert result["base_resolution"] == 1440
    assert result["aspect_ratio"] == "16:9"
    assert result["num_inference_steps"] == 4
    assert result["guidance_scale"] == 1.0


def test_lens_turbo_load_pipeline_requires_external_lens_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "lens", None)

    with pytest.raises(
        RuntimeError, match="requires Microsoft's Lens inference package"
    ):
        lens_example.load_pipeline("microsoft/Lens-Turbo", device="cpu")


def test_lens_turbo_load_pipeline_uses_external_lens_package(monkeypatch):
    fake_lens = ModuleType("lens")
    calls = []

    class FakeTextEncoder:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("text_encoder", model_id, kwargs))
            return "text-encoder"

    class FakeLensPipeline:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append(("pipeline", model_id, kwargs))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

    fake_lens.LensGptOssEncoder = FakeTextEncoder
    fake_lens.LensPipeline = FakeLensPipeline
    monkeypatch.setitem(sys.modules, "lens", fake_lens)

    pipe = lens_example.load_pipeline(
        "fake/lens", device="cpu", dtype=torch.float16, disable_mxfp4=True
    )

    assert isinstance(pipe, FakeLensPipeline)
    assert calls[0][0] == "text_encoder"
    assert calls[0][1] == "fake/lens"
    assert calls[0][2]["subfolder"] == "text_encoder"
    assert calls[0][2]["dtype"] == torch.float16
    assert calls[1] == (
        "pipeline",
        "fake/lens",
        {"text_encoder": "text-encoder", "torch_dtype": torch.float16},
    )
    assert calls[2] == ("to", "cpu")


def test_lens_turbo_target_config_resolves_fused_qkv_targets(monkeypatch):
    fake_lens = ModuleType("lens")
    fake_transformer = ModuleType("lens.transformer")
    fake_lens.__path__ = []

    class FakeLensJointAttention(torch.nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.img_qkv = torch.nn.Linear(dim, 3 * dim)
            self.txt_qkv = torch.nn.Linear(dim, 3 * dim)
            self.to_out = torch.nn.ModuleList(
                [torch.nn.Linear(dim, dim), torch.nn.Identity()]
            )
            self.to_add_out = torch.nn.Linear(dim, dim)

    class FakeGateMLP(torch.nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.w1 = torch.nn.Linear(dim, dim)
            self.w2 = torch.nn.Linear(dim, dim)
            self.w3 = torch.nn.Linear(dim, dim)

    class FakeLensTransformerBlock(torch.nn.Module):
        def __init__(self, dim=8):
            super().__init__()
            self.attn = FakeLensJointAttention(dim)
            self.img_mod = torch.nn.Sequential(
                torch.nn.SiLU(), torch.nn.Linear(dim, 6 * dim)
            )
            self.txt_mod = torch.nn.Sequential(
                torch.nn.SiLU(), torch.nn.Linear(dim, 6 * dim)
            )
            self.img_mlp = FakeGateMLP(dim)
            self.txt_mlp = FakeGateMLP(dim)

    class FakeLensTransformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList([FakeLensTransformerBlock()])

    fake_transformer.LensJointAttention = FakeLensJointAttention
    fake_transformer.LensTransformerBlock = FakeLensTransformerBlock
    fake_transformer.GateMLP = FakeGateMLP
    fake_lens.transformer = fake_transformer
    monkeypatch.setitem(sys.modules, "lens", fake_lens)
    monkeypatch.setitem(sys.modules, "lens.transformer", fake_transformer)

    model = FakeLensTransformer()
    target_config = lens_turbo_target_config("nvfp4", inner_dim=8)

    assert target_config.calibration_scopes[0].module_classes == (
        FakeLensTransformerBlock,
    )
    assert (
        target_config.calibration_scopes[0].prev_replay_transform
        is lens_example._lens_block_prev_replay_transform
    )
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert export_names == {
        "transformer_blocks.0.attn.img_qkv",
        "transformer_blocks.0.attn.txt_qkv",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.img_mod.1",
        "transformer_blocks.0.img_mlp.w1",
        "transformer_blocks.0.img_mlp.w2",
        "transformer_blocks.0.img_mlp.w3",
        "transformer_blocks.0.txt_mod.1",
        "transformer_blocks.0.txt_mlp.w1",
        "transformer_blocks.0.txt_mlp.w2",
        "transformer_blocks.0.txt_mlp.w3",
    }
    image_qkv = next(
        target
        for target in targets
        if target.export_name == "transformer_blocks.0.attn.img_qkv"
    )
    text_qkv = next(
        target
        for target in targets
        if target.export_name == "transformer_blocks.0.attn.txt_qkv"
    )
    assert tuple(image_qkv.module_names) == ("transformer_blocks.0.attn.img_qkv",)
    assert image_qkv.roles == ()
    assert text_qkv.roles == ()
    assert image_qkv.quant.bias == "auto"
    for name in ("transformer_blocks.0.img_mod.1", "transformer_blocks.0.txt_mod.1"):
        target = next(target for target in targets if target.export_name == name)
        assert isinstance(target.quant, AwqTargetQuant)
        assert isinstance(target.quant.layout, AdaNormAwqW4A16Layout)
        assert target.quant.layout.splits == 6


def test_flux1_upstream_target_config_matches_tiny_flux_int4_shift_scope():
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=64,
        pooled_projection_dim=64,
        guidance_embeds=True,
        axes_dims_rope=(8, 8),
    )
    target_config = flux1_target_config("int4")
    assert target_config.calibration_scopes[0].module_classes == (
        type(model.transformer_blocks[0]),
    )
    assert target_config.calibration_scopes[1].module_classes == (
        type(model.single_transformer_blocks[0]),
    )
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert "transformer_blocks.0.qkv_proj" in export_names
    assert "transformer_blocks.0.qkv_proj_context" in export_names
    assert "single_transformer_blocks.0.out_proj" in export_names
    assert "transformer_blocks.0.norm1.linear" in export_names
    assert "single_transformer_blocks.0.norm.linear" in export_names

    shifted_names = {
        target.export_name
        for target in targets
        if getattr(target.quant, "shift_activations", None) is True
    }
    assert shifted_names == {
        "transformer_blocks.0.mlp_fc2",
        "transformer_blocks.0.mlp_context_fc2",
        "single_transformer_blocks.0.mlp_fc2",
    }
    unshifted_names = {
        "transformer_blocks.0.qkv_proj",
        "transformer_blocks.0.qkv_proj_context",
        "transformer_blocks.0.out_proj",
        "transformer_blocks.0.out_proj_context",
        "transformer_blocks.0.mlp_fc1",
        "transformer_blocks.0.mlp_context_fc1",
        "single_transformer_blocks.0.qkv_proj",
        "single_transformer_blocks.0.out_proj",
        "single_transformer_blocks.0.mlp_fc1",
    }
    for name in unshifted_names:
        target = next(target for target in targets if target.export_name == name)
        assert getattr(target.quant, "shift_activations", None) is not True

    out_proj = next(
        target
        for target in targets
        if target.export_name == "single_transformer_blocks.0.out_proj"
    )
    assert out_proj.quant.bias == "zero"

    extra_names = {
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
    }
    for target in targets:
        if target.export_name not in extra_names:
            continue
        assert isinstance(target.quant, AwqTargetQuant)
        assert isinstance(target.quant.layout, AdaNormAwqW4A16Layout)
        assert target.quant.layout.splits == (
            3 if target.export_name.startswith("single_") else 6
        )


def test_flux1_upstream_target_config_matches_tiny_flux_nvfp4():
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=64,
        pooled_projection_dim=64,
        guidance_embeds=True,
        axes_dims_rope=(8, 8),
    )
    target_config = flux1_target_config("nvfp4")
    assert target_config.calibration_scopes[0].module_classes == (
        type(model.transformer_blocks[0]),
    )
    assert target_config.calibration_scopes[1].module_classes == (
        type(model.single_transformer_blocks[0]),
    )
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert "transformer_blocks.0.qkv_proj" in export_names
    assert "transformer_blocks.0.qkv_proj_context" in export_names
    assert "single_transformer_blocks.0.out_proj" in export_names
    assert "transformer_blocks.0.norm1.linear" in export_names
    assert "single_transformer_blocks.0.norm.linear" in export_names
    out_proj = next(
        target
        for target in targets
        if target.export_name == "single_transformer_blocks.0.out_proj"
    )
    assert out_proj.quant.bias == "zero"
    assert not any(getattr(target.quant, "shift_activations", None) for target in targets)


def test_flux2_klein_upstream_target_config_exports_nunchaku_lite_keys():
    from diffusers import Flux2Transformer2DModel

    model = Flux2Transformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 4, 4, 4),
        timestep_guidance_channels=32,
    )
    target_config = flux2_klein_target_config(
        single_qkv_features=96, single_attn_features=32
    )
    assert target_config.calibration_scopes[0].module_classes == (
        type(model.transformer_blocks[0]),
    )
    assert target_config.calibration_scopes[1].module_classes == (
        type(model.single_transformer_blocks[0]),
    )
    assert target_config.calibration_scopes[2].module_classes == (
        type(model.single_transformer_blocks[0]),
    )
    assert target_config.calibration_scopes[0].use_prev_scope_outputs is True
    assert target_config.calibration_scopes[1].use_prev_scope_outputs is False
    assert target_config.calibration_scopes[2].use_prev_scope_outputs is True
    assert (
        target_config.calibration_scopes[0].prev_replay_transform
        is flux2_example._flux2_block_prev_replay_transform
    )
    assert (
        target_config.calibration_scopes[2].prev_replay_transform
        is flux2_example._flux2_block_prev_replay_transform
    )
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert export_names == {
        "transformer_blocks.0.attn.to_qkv",
        "transformer_blocks.0.attn.to_added_qkv",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.ff.linear_in",
        "transformer_blocks.0.ff.linear_out",
        "transformer_blocks.0.ff_context.linear_in",
        "transformer_blocks.0.ff_context.linear_out",
        "single_transformer_blocks.0.attn.qkv_proj",
        "single_transformer_blocks.0.attn.mlp_fc1",
        "single_transformer_blocks.0.attn.out_proj",
        "single_transformer_blocks.0.attn.mlp_fc2",
    }


def test_flux2_klein_first_single_scope_recomputes_transition():
    from diffusers import Flux2Transformer2DModel

    model = Flux2Transformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=2,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 4, 4, 4),
        timestep_guidance_channels=32,
    )
    target_config = flux2_klein_target_config(
        single_qkv_features=96, single_attn_features=32
    )

    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    scopes = {
        scope.name: scope
        for scope in assign_calibration_scopes(model, targets, target_config)
    }

    assert scopes["single_transformer_blocks.0"].use_prev_scope_outputs is False
    assert scopes["single_transformer_blocks.0"].prev_replay_transform is None
    assert scopes["single_transformer_blocks.1"].use_prev_scope_outputs is True
    assert (
        scopes["single_transformer_blocks.1"].prev_replay_transform
        is flux2_example._flux2_block_prev_replay_transform
    )


def test_flux2_klein_model_variants_use_expected_split_sizes():
    config_4b = flux2_klein_target_config(
        single_qkv_features=9216, single_attn_features=3072
    )
    config_9b = flux2_klein_target_config(
        single_qkv_features=12288, single_attn_features=4096
    )

    assert config_4b.patches[0].args["splits"] == [9216]
    assert config_4b.patches[1].args["splits"] == [3072]
    assert config_9b.patches[0].args["splits"] == [12288]
    assert config_9b.patches[1].args["splits"] == [4096]
    assert isinstance(config_4b.targets[3].quant.weight_layout, NunchakuSvdqLayout)
    assert config_4b.targets[3].quant.weight_layout.outer_scale_splits == (
        3072,
        3072,
        3072,
    )
    assert isinstance(config_9b.targets[3].quant.weight_layout, NunchakuSvdqLayout)
    assert config_9b.targets[3].quant.weight_layout.outer_scale_splits == (
        4096,
        4096,
        4096,
    )


def test_longcat_image_edit_target_config_uses_manifest_exact_module_paths():
    from diffusers.models.transformers.transformer_longcat_image import (
        LongCatImageTransformer2DModel,
    )

    model = LongCatImageTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=64,
        num_attention_heads=2,
        joint_attention_dim=128,
        pooled_projection_dim=128,
        axes_dims_rope=[16, 56, 56],
    )
    target_config = longcat_image_edit_target_config("nvfp4")

    assert target_config.patches[0].type == "split_linear"
    assert target_config.patches[0].module == "single_transformer_blocks.*.proj_out"
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert "transformer_blocks.0.attn.to_q" in export_names
    assert "transformer_blocks.0.attn.to_k" in export_names
    assert "transformer_blocks.0.attn.to_v" in export_names
    assert "single_transformer_blocks.0.proj_out.linears.0" in export_names
    assert "single_transformer_blocks.0.proj_out.linears.1" in export_names
    assert not any(name.endswith("qkv_proj") for name in export_names)
    for target in targets:
        assert target.export_name == target.module_names[0]

    extra_names = {
        "transformer_blocks.0.norm1.linear": 6,
        "transformer_blocks.0.norm1_context.linear": 6,
        "single_transformer_blocks.0.norm.linear": 3,
    }
    for name, splits in extra_names.items():
        target = next(target for target in targets if target.export_name == name)
        assert isinstance(target.quant, AwqTargetQuant)
        assert isinstance(target.quant.layout, AdaNormAwqW4A16Layout)
        assert target.quant.layout.splits == splits


def test_image_edit_records_and_forward_use_source_image(monkeypatch):
    class FakeImage:
        size = (640, 512)

        def crop(self, box):
            self.box = box
            return self

        def resize(self, size):
            self.resized = size
            return self

        def convert(self, mode):
            self.mode = mode
            return self

    rows = [
        {"sample_id": 17, "source_image": FakeImage(), "prompt": "make it brighter"}
    ]

    def fake_load_dataset(dataset, **kwargs):
        assert dataset == "VyoJ/NHR-Edit-Change_Only"
        assert kwargs["split"] == "validation"
        return rows

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    records = image_edit_records(1)
    samples = image_edit_example.batched_samples(records, batch_size=1)

    assert records[0]["filename"] == "17"
    assert records[0]["prompt"] == "make it brighter"
    assert records[0]["image"].resized == (512, 512)
    assert samples[0]["image"] is records[0]["image"]

    calls = []

    class FakePipe:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return object()

    forward = image_edit_forward_fn(
        FakePipe(), steps=8, guidance_scale=1.0, device="cpu"
    )
    forward(samples[0])

    assert calls[0]["image"] is records[0]["image"]
    assert calls[0]["prompt"] == "make it brighter"
    assert calls[0]["negative_prompt"] == ""
    assert calls[0]["num_inference_steps"] == 8
    assert calls[0]["guidance_scale"] == 1.0


def test_longcat_image_edit_nvfp4_export_writes_manifest(tmp_path):
    from diffusers.models.transformers.transformer_longcat_image import (
        LongCatImageTransformer2DModel,
    )

    torch.manual_seed(0)
    model = LongCatImageTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=64,
        num_attention_heads=2,
        joint_attention_dim=128,
        pooled_projection_dim=128,
        axes_dims_rope=[16, 56, 56],
    ).to(torch.bfloat16)
    output = tmp_path / "longcat.safetensors"

    quantize_and_export(
        model,
        DiffusionQuantSpec(
            precision="fp4",
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
        ),
        longcat_image_edit_target_config("nvfp4"),
        calibration=None,
        export=ExportSpec(output=output),
    )

    config_metadata = _config_metadata(output)
    metadata = _checkpoint_quantization_config(output)
    _assert_checkpoint_quantization_config(
        metadata, config_metadata, has_runtime_manifest=True
    )
    manifest = metadata["runtime_manifest"]
    assert manifest["structural_patches"] == [
        {
            "type": "split_linear_input",
            "module": "single_transformer_blocks.*.proj_out",
            "args": {"splits": ["out_features"]},
        }
    ]
    assert manifest["targets"]
    for target in manifest["targets"]:
        assert target["checkpoint_prefix"] == target["source_modules"][0]
        assert len(target["source_modules"]) == 1
    assert any(
        target["checkpoint_prefix"] == "single_transformer_blocks.0.proj_out.linears.0"
        for target in manifest["targets"]
    )


def test_pixart_sigma_upstream_target_config_exports_int4(tmp_path):
    from diffusers import PixArtTransformer2DModel

    model = PixArtTransformer2DModel(
        num_attention_heads=4,
        attention_head_dim=32,
        in_channels=4,
        out_channels=8,
        num_layers=1,
        norm_num_groups=4,
        cross_attention_dim=128,
        sample_size=8,
        patch_size=2,
        caption_channels=128,
    )
    output = tmp_path / "pixart.safetensors"
    nvfp4_config = pixart_sigma_target_config("nvfp4")
    assert nvfp4_config.calibration_scopes[0].module_classes == (
        type(model.transformer_blocks[0]),
    )
    assert nvfp4_config.calibration_scopes[0].use_prev_scope_outputs is True
    assert (
        nvfp4_config.calibration_scopes[0].prev_replay_transform
        is pixart_example._hidden_states_prev_replay_transform
    )
    nvfp4_targets = collect_quant_targets(model, nvfp4_config)
    adaln_target = next(
        target
        for target in nvfp4_targets
        if target.export_name == "adaln_single.linear"
    )

    assert isinstance(adaln_target.quant, AwqTargetQuant)
    assert isinstance(adaln_target.quant.layout, AwqW4A16Layout)

    quantize_and_export(
        model,
        DiffusionQuantSpec(rank=16, group_size=64, smooth=False),
        pixart_sigma_target_config("int4"),
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    metadata = _config_metadata(output)

    assert "transformer_blocks.0.attn1.qkv_proj.qweight" in keys
    assert "transformer_blocks.0.attn2.kv_proj.qweight" in keys
    assert "transformer_blocks.0.mlp_fc1.qweight" in keys
    assert metadata["rank"] == 16
    _assert_checkpoint_quantization_config(
        _checkpoint_quantization_config(output), metadata
    )


@pytest.mark.skipif(
    importlib.util.find_spec("nunchaku_lite") is None,
    reason="nunchaku_lite is not installed",
)
def test_flux1_nvfp4_upstream_checkpoint_strict_loads_with_nunchaku_lite(tmp_path):
    from diffusers import FluxTransformer2DModel

    kwargs = dict(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=4,
        joint_attention_dim=128,
        pooled_projection_dim=128,
        guidance_embeds=False,
        axes_dims_rope=(8, 8),
    )
    source = FluxTransformer2DModel(**kwargs).to(torch.bfloat16)
    output = tmp_path / "flux1-nvfp4-lite-loadable.safetensors"
    target_config = flux1_target_config("nvfp4")

    quantize_and_export(
        source,
        DiffusionQuantSpec(
            precision="fp4",
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
        ),
        target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    metadata = _config_metadata(output)

    assert "transformer_blocks.0.norm1.linear.wzeros" in keys
    assert "transformer_blocks.0.norm1.linear.smooth_factor" not in keys
    norm_target = next(
        target
        for target in metadata["targets"]
        if target["export_name"] == "transformer_blocks.0.norm1.linear"
    )
    assert norm_target["weight_layout"] == {"name": "adanorm_awq_w4a16", "splits": 6}

    target = FluxTransformer2DModel(**kwargs)
    patch_quantized_pipeline(
        SimpleNamespace(transformer=target),
        spec=RuntimePipelineSpec(
            mode="quantized",
            runtime="nunchaku-lite",
            checkpoint=output,
            nunchaku_lite_target="flux",
            precision="fp4",
        ),
    )

    assert target._nunchaku_lite_patched


def test_sana_upstream_target_config_exports_pointwise_conv_nvfp4(tmp_path):
    from diffusers import SanaTransformer2DModel

    model = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=4,
        attention_head_dim=32,
        num_layers=1,
        num_cross_attention_heads=4,
        cross_attention_head_dim=32,
        cross_attention_dim=128,
        caption_channels=128,
        sample_size=8,
        patch_size=1,
        mlp_ratio=2.0,
    )
    target_config = sana_target_config("nvfp4")
    assert target_config.calibration_scopes[0].module_classes == (
        type(model.transformer_blocks[0]),
    )
    assert target_config.calibration_scopes[0].use_prev_scope_outputs is True
    assert (
        target_config.calibration_scopes[0].prev_replay_transform
        is sana_example._hidden_states_prev_replay_transform
    )
    targets = collect_quant_targets(model, target_config)
    output = tmp_path / "sana.safetensors"

    assert {
        target.kind
        for target in targets
        if target.export_name.endswith(("mlp_fc1", "mlp_fc2"))
    } == {"conv"}

    quantize_and_export(
        model,
        DiffusionQuantSpec(
            precision="fp4",
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
        ),
        target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    metadata = _config_metadata(output)

    assert "transformer_blocks.0.mlp_fc1.qweight" in keys
    assert "transformer_blocks.0.mlp_fc2.qweight" in keys
    assert metadata["weight"]["dtype"] == "fp4_e2m1_all"
    assert metadata["weight"]["scale_dtypes"] == [None, "sfp8_e4m3_nan"]
    _assert_checkpoint_quantization_config(
        _checkpoint_quantization_config(output), metadata
    )


def test_ernie_image_target_config_exports_manifest_nvfp4_with_dense_extras(tmp_path):
    from diffusers.models.transformers.transformer_ernie_image import (
        ErnieImageTransformer2DModel,
    )

    torch.manual_seed(0)
    model = ErnieImageTransformer2DModel(
        hidden_size=128,
        num_attention_heads=4,
        num_layers=1,
        ffn_hidden_size=256,
        in_channels=128,
        out_channels=128,
        patch_size=1,
        text_in_dim=64,
        rope_axes_dim=(8, 12, 12),
    ).to(torch.bfloat16)
    target_config = ernie_image_target_config("nvfp4")
    targets = collect_quant_targets(model, target_config)
    int4_targets = collect_quant_targets(model, ernie_image_target_config("int4"))
    output = tmp_path / "ernie.safetensors"
    export_names = {target.export_name for target in targets}
    int4_export_names = {target.export_name for target in int4_targets}
    extra_names = {
        "text_proj",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
        "adaLN_modulation.1",
        "final_norm.linear",
        "final_linear",
    }

    assert target_config.calibration_scopes[0].module_classes == (
        type(model.layers[0]),
    )
    assert target_config.calibration_scopes[0].use_prev_scope_outputs is True
    assert (
        target_config.calibration_scopes[0].prev_replay_transform
        is ernie_example._ernie_block_prev_replay_transform
    )
    assert len(targets) == 7
    assert len(int4_targets) == 7
    assert "layers.0.self_attention.to_q" in export_names
    assert "layers.0.mlp.linear_fc2" in export_names
    assert extra_names.isdisjoint(export_names)
    assert extra_names.isdisjoint(int4_export_names)
    assert all(not isinstance(target.quant, AwqTargetQuant) for target in targets)
    assert all(not isinstance(target.quant, AwqTargetQuant) for target in int4_targets)
    for target in targets:
        assert target.export_name == target.module_names[0]
        assert len(target.module_names) == 1
        assert target.roles == ()
        assert not isinstance(target.quant, AwqTargetQuant)

    quantize_and_export(
        model,
        DiffusionQuantSpec(
            precision="fp4",
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
        ),
        target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    config_metadata = _config_metadata(output)
    metadata = _checkpoint_quantization_config(output)
    _assert_checkpoint_quantization_config(
        metadata, config_metadata, has_runtime_manifest=True
    )

    manifest = metadata["runtime_manifest"]
    assert "layers.0.self_attention.to_q.qweight" in keys
    assert "text_proj.weight" in keys
    assert "text_proj.qweight" not in keys
    assert "final_linear.weight" in keys
    assert "final_linear.bias" in keys
    assert "final_linear.qweight" not in keys
    assert manifest["requirements"]["precision"] == "fp4"
    assert manifest["requirements"]["weight_dtype"] == "fp4_e2m1_all"
    assert manifest["structural_patches"] == []
    assert len(manifest["targets"]) == len(targets)
    ops = {
        target["checkpoint_prefix"]: target["nunchaku_op"]
        for target in manifest["targets"]
    }
    assert ops["layers.0.self_attention.to_q"] == "svdq_w4a4"
    assert "text_proj" not in ops
    assert "awq_w4a16" not in set(ops.values())
    assert "adanorm_awq_w4a16" not in set(ops.values())
    for target in manifest["targets"]:
        assert target["checkpoint_prefix"] == target["source_modules"][0]
        assert len(target["source_modules"]) == 1
        assert target["has_bias"] is False


def test_ltx2_3_target_config_matches_tiny_ltx2_transformer():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "LTX2VideoTransformer3DModel"):
        pytest.skip("diffusers does not provide LTX2VideoTransformer3DModel")

    model = diffusers.LTX2VideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        audio_in_channels=8,
        audio_out_channels=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        cross_attention_dim=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=16,
        caption_channels=16,
        gated_attn=True,
        audio_gated_attn=True,
        cross_attn_mod=True,
        audio_cross_attn_mod=True,
        use_prompt_embeddings=False,
    )
    target_config = ltx2_3_target_config()

    assert target_config.calibration_scopes[0].module_classes == (
        type(model.transformer_blocks[0]),
    )
    assert target_config.calibration_scopes[0].use_prev_scope_outputs is True
    assert (
        target_config.calibration_scopes[0].prev_replay_transform
        is ltx2_example._ltx2_block_prev_replay_transform
    )

    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert "transformer_blocks.0.attn1.to_q" in export_names
    assert "transformer_blocks.0.audio_attn1.to_q" in export_names
    assert "transformer_blocks.0.attn2.to_k" in export_names
    assert "transformer_blocks.0.audio_attn2.to_v" in export_names
    assert "transformer_blocks.0.audio_to_video_attn.to_out.0" in export_names
    assert "transformer_blocks.0.video_to_audio_attn.to_q" in export_names
    assert "transformer_blocks.0.ff.net.0.proj" in export_names
    assert "transformer_blocks.0.audio_ff.net.2" in export_names
    assert "transformer_blocks.0.attn1.to_gate_logits" not in export_names
    assert "transformer_blocks.0.audio_to_video_attn.to_gate_logits" not in export_names
    assert all("to_gate_logits" not in name for name in export_names)
    assert "proj_in" not in export_names
    assert "proj_out" not in export_names
    assert "time_embed.linear" not in export_names
    for target in targets:
        assert target.export_name == target.module_names[0]
        assert len(target.module_names) == 1
        assert target.roles == ()
        assert target.kind == "linear"

    fused_targets = collect_quant_targets(model, ltx2_3_target_config(fuse_qkv=True))
    fused_by_name = {target.export_name: target for target in fused_targets}
    fused_groups = {
        "transformer_blocks.0.attn1.qkv_proj": (
            (
                "transformer_blocks.0.attn1.to_q",
                "transformer_blocks.0.attn1.to_k",
                "transformer_blocks.0.attn1.to_v",
            ),
            ("q", "k", "v"),
        ),
        "transformer_blocks.0.audio_attn1.qkv_proj": (
            (
                "transformer_blocks.0.audio_attn1.to_q",
                "transformer_blocks.0.audio_attn1.to_k",
                "transformer_blocks.0.audio_attn1.to_v",
            ),
            ("q", "k", "v"),
        ),
        "transformer_blocks.0.attn2.kv_proj": (
            (
                "transformer_blocks.0.attn2.to_k",
                "transformer_blocks.0.attn2.to_v",
            ),
            ("k", "v"),
        ),
        "transformer_blocks.0.audio_attn2.kv_proj": (
            (
                "transformer_blocks.0.audio_attn2.to_k",
                "transformer_blocks.0.audio_attn2.to_v",
            ),
            ("k", "v"),
        ),
        "transformer_blocks.0.audio_to_video_attn.kv_proj": (
            (
                "transformer_blocks.0.audio_to_video_attn.to_k",
                "transformer_blocks.0.audio_to_video_attn.to_v",
            ),
            ("k", "v"),
        ),
        "transformer_blocks.0.video_to_audio_attn.kv_proj": (
            (
                "transformer_blocks.0.video_to_audio_attn.to_k",
                "transformer_blocks.0.video_to_audio_attn.to_v",
            ),
            ("k", "v"),
        ),
    }
    for name, (module_names, roles) in fused_groups.items():
        target = fused_by_name[name]
        assert target.module_names == module_names
        assert target.roles == roles
        assert target.kind == "linear"

    fused_export_names = set(fused_by_name)
    assert "transformer_blocks.0.attn2.to_q" in fused_export_names
    assert "transformer_blocks.0.audio_attn2.to_q" in fused_export_names
    assert "transformer_blocks.0.audio_to_video_attn.to_q" in fused_export_names
    assert "transformer_blocks.0.video_to_audio_attn.to_q" in fused_export_names
    assert "transformer_blocks.0.attn1.to_q" not in fused_export_names
    assert "transformer_blocks.0.audio_attn1.to_v" not in fused_export_names
    assert "transformer_blocks.0.attn2.to_k" not in fused_export_names
    assert "transformer_blocks.0.audio_to_video_attn.to_v" not in fused_export_names
    assert "transformer_blocks.0.attn1.to_gate_logits" not in fused_export_names
    assert all("to_gate_logits" not in name for name in fused_export_names)

    nvfp4_targets = collect_quant_targets(model, ltx2_3_target_config("nvfp4"))
    nvfp4_by_name = {target.export_name: target for target in nvfp4_targets}
    nvfp4_fused_targets = collect_quant_targets(
        model, ltx2_3_target_config("nvfp4", fuse_qkv=True)
    )
    nvfp4_fused_by_name = {target.export_name: target for target in nvfp4_fused_targets}
    dense_adaln_names = {
        "time_embed.linear",
        "audio_time_embed.linear",
        "av_cross_attn_video_scale_shift.linear",
        "av_cross_attn_audio_scale_shift.linear",
        "av_cross_attn_video_a2v_gate.linear",
        "av_cross_attn_audio_v2a_gate.linear",
        "prompt_adaln.linear",
        "audio_prompt_adaln.linear",
    }
    gate_names = {
        "transformer_blocks.0.attn1.to_gate_logits",
        "transformer_blocks.0.audio_attn1.to_gate_logits",
        "transformer_blocks.0.attn2.to_gate_logits",
        "transformer_blocks.0.audio_attn2.to_gate_logits",
        "transformer_blocks.0.audio_to_video_attn.to_gate_logits",
        "transformer_blocks.0.video_to_audio_attn.to_gate_logits",
    }

    assert dense_adaln_names.isdisjoint(nvfp4_by_name)
    assert dense_adaln_names.isdisjoint(nvfp4_fused_by_name)
    assert set(fused_groups) <= set(nvfp4_fused_by_name)
    assert gate_names.isdisjoint(nvfp4_by_name)
    assert gate_names.isdisjoint(nvfp4_fused_by_name)
    assert all("to_gate_logits" not in name for name in nvfp4_by_name)
    assert all("to_gate_logits" not in name for name in nvfp4_fused_by_name)


def test_ltx2_3_manifest_preflight_rejects_gate_logits_target():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "LTX2VideoTransformer3DModel"):
        pytest.skip("diffusers does not provide LTX2VideoTransformer3DModel")

    model = diffusers.LTX2VideoTransformer3DModel(
        in_channels=8,
        out_channels=8,
        audio_in_channels=8,
        audio_out_channels=8,
        num_layers=1,
        num_attention_heads=2,
        attention_head_dim=4,
        cross_attention_dim=16,
        audio_num_attention_heads=2,
        audio_attention_head_dim=4,
        audio_cross_attention_dim=16,
        caption_channels=16,
        gated_attn=True,
        audio_gated_attn=True,
        cross_attn_mod=True,
        audio_cross_attn_mod=True,
        use_prompt_embeddings=False,
    )
    safe_config = ltx2_3_target_config()
    nvfp4_config = ltx2_3_target_config("nvfp4")
    unsafe_config = TargetConfig(
        calibration_scopes=safe_config.calibration_scopes,
        targets=[
            TargetRule(
                scope_module_classes=type(model.transformer_blocks[0]),
                module_classes=torch.nn.Linear,
            )
        ],
    )

    ltx2_example.validate_ltx2_3_nunchaku_lite_manifest_targets(model, safe_config)
    ltx2_example.validate_ltx2_3_nunchaku_lite_manifest_targets(model, nvfp4_config)
    with pytest.raises(RuntimeError, match="to_gate_logits"):
        ltx2_example.validate_ltx2_3_nunchaku_lite_manifest_targets(model, unsafe_config)


def test_ltx2_3_safetensors_parser_uses_local_defaults():
    parser = ltx2_safetensors_example.default_arg_parser()
    args = parser.parse_args([])

    assert args.checkpoint_path == (
        "/mnt/disks/workspace/research/LTX-2/models/"
        "ltx-2.3-22b-distilled-1.1.safetensors"
    )
    assert args.gemma_root == (
        "/mnt/disks/workspace/research/LTX-2/models/"
        "gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert args.output == (
        "outputs/checkpoints/svdq-int4_r32-ltx2.3-distilled-sft.safetensors"
    )
    assert args.sigma_schedule == "distilled"
    assert args.sigmas is None
    assert not hasattr(args, "ltx2_root")
    assert not hasattr(args, "spatial_upsampler_path")


def test_ltx2_3_safetensors_target_config_matches_tiny_upstream_block():
    import inspect

    ltx_transformer = pytest.importorskip("ltx_core.model.transformer.transformer")
    BasicAVTransformerBlock = ltx_transformer.BasicAVTransformerBlock
    TransformerConfig = ltx_transformer.TransformerConfig
    block_kwargs = {"idx": 0} if "idx" in inspect.signature(BasicAVTransformerBlock).parameters else {}

    class TinyUpstreamLTX(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer_blocks = torch.nn.ModuleList(
                [
                    BasicAVTransformerBlock(
                        **block_kwargs,
                        video=TransformerConfig(
                            dim=8,
                            heads=2,
                            d_head=4,
                            context_dim=16,
                            apply_gated_attention=True,
                        ),
                        audio=TransformerConfig(
                            dim=8,
                            heads=2,
                            d_head=4,
                            context_dim=16,
                            apply_gated_attention=True,
                        ),
                    )
                ]
            )
            self.out = torch.nn.Linear(8, 8)

    model = TinyUpstreamLTX()
    target_config = ltx2_safetensors_example.ltx2_3_safetensors_target_config()

    assert target_config.calibration_scopes[0].module_classes == (
        BasicAVTransformerBlock,
    )
    assert target_config.calibration_scopes[0].use_prev_scope_outputs is True
    assert (
        target_config.calibration_scopes[0].prev_replay_transform
        is ltx2_safetensors_example._ltx2_block_prev_replay_transform
    )

    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert "transformer_blocks.0.attn1.to_q" in export_names
    assert "transformer_blocks.0.audio_attn1.to_q" in export_names
    assert "transformer_blocks.0.attn2.to_k" in export_names
    assert "transformer_blocks.0.audio_to_video_attn.to_out.0" in export_names
    assert "transformer_blocks.0.video_to_audio_attn.to_q" in export_names
    assert "transformer_blocks.0.ff.net.0.proj" in export_names
    assert "transformer_blocks.0.audio_ff.net.2" in export_names
    assert "transformer_blocks.0.attn1.to_gate_logits" not in export_names
    assert all("to_gate_logits" not in name for name in export_names)
    assert "out" not in export_names

    fused_targets = collect_quant_targets(
        model,
        ltx2_safetensors_example.ltx2_3_safetensors_target_config(fuse_qkv=True),
    )
    fused_by_name = {target.export_name: target for target in fused_targets}

    assert fused_by_name["transformer_blocks.0.attn1.qkv_proj"].module_names == (
        "transformer_blocks.0.attn1.to_q",
        "transformer_blocks.0.attn1.to_k",
        "transformer_blocks.0.attn1.to_v",
    )
    assert fused_by_name["transformer_blocks.0.attn1.qkv_proj"].roles == (
        "q",
        "k",
        "v",
    )
    assert fused_by_name["transformer_blocks.0.attn2.kv_proj"].module_names == (
        "transformer_blocks.0.attn2.to_k",
        "transformer_blocks.0.attn2.to_v",
    )
    assert fused_by_name["transformer_blocks.0.attn2.kv_proj"].roles == ("k", "v")
    assert "transformer_blocks.0.attn1.to_q" not in fused_by_name
    assert "transformer_blocks.0.attn2.to_k" not in fused_by_name
    assert all("to_gate_logits" not in name for name in fused_by_name)


def test_ltx2_3_safetensors_run_model_cli_wires_one_stage_calibration(
    monkeypatch, tmp_path
):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["spec"] = spec
        captured["target_config"] = target_config
        captured["calibration"] = calibration
        captured["export"] = export
        captured["logging"] = logging

    class FakeX0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.velocity_model = torch.nn.Linear(1, 1)

    class FakeModelContext:
        def __init__(self, transformer):
            self.transformer = transformer

        def __enter__(self):
            return self.transformer

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeStage:
        def __init__(self, transformer):
            self.transformer = transformer

        def model_context(self):
            return FakeModelContext(self.transformer)

    class FakePipe:
        def __init__(self, transformer):
            self.stage = FakeStage(transformer)

    transformer = FakeX0()
    pipe = FakePipe(transformer)
    target_config = object()
    forward_sentinel = object()

    def fake_forward_fn(pipe_arg, **kwargs):
        captured["forward_pipe"] = pipe_arg
        captured["forward_kwargs"] = kwargs
        return forward_sentinel

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--num-samples",
            "2",
            "--batch-size",
            "1",
            "--sample-batch-size",
            "4",
            "--cache-mode",
            "disabled",
            "--device",
            "cpu",
            "--height",
            "256",
            "--width",
            "384",
            "--num-frames",
            "17",
            "--frame-rate",
            "12",
            "--steps",
            "3",
            "--sigmas",
            "1.0,0.5,0.0",
            "--negative-prompt",
            "low quality",
            "--video-cfg-guidance-scale",
            "1.5",
            "--audio-cfg-guidance-scale",
            "2.5",
            "--output",
            str(tmp_path / "ltx2-sft.safetensors"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "load_ltx2_one_stage_pipeline",
        lambda *args, **kwargs: pipe,
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "ltx2_3_safetensors_target_config",
        lambda precision, *, fuse_qkv=False: target_config,
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "validate_ltx2_3_nunchaku_lite_manifest_targets",
        lambda transformer, target_config: None,
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": f"{index:04d}-0", "prompt": str(index), "seed": index}
            for index in range(num_samples)
        ],
    )
    monkeypatch.setattr(
        ltx2_safetensors_example, "ltx2_one_stage_forward_fn", fake_forward_fn
    )
    monkeypatch.setattr(
        ltx2_safetensors_example, "quantize_and_export", fake_quantize_and_export
    )

    ltx2_safetensors_example.run_model_cli()

    assert captured["model"] is transformer.velocity_model
    assert captured["target_config"] is target_config
    assert captured["calibration"].forward_fn is forward_sentinel
    assert captured["calibration"].num_samples == 2
    assert captured["calibration"].cache_num_samples == 2
    assert captured["calibration"].batch_size == 1
    assert captured["calibration"].sample_batch_size == 4
    assert captured["calibration"].max_rows_per_target == 4096
    assert captured["calibration"].output_dir == Path(
        "outputs/calibration/ltx2.3-distilled-sft/int4/inputs/samples"
    )
    save_fn = captured["calibration"].output_save_fn
    assert save_fn.func is ltx2_safetensors_example.save_ltx2_one_stage_video
    assert save_fn.keywords == {"frame_rate": 12.0}
    assert captured["export"].output == tmp_path / "ltx2-sft.safetensors"
    assert captured["logging"].log_dir == str(tmp_path / "logs")
    assert captured["logging"].name == "ltx2-sft"
    assert captured["forward_pipe"] is pipe
    assert captured["forward_kwargs"]["transformer"] is transformer
    assert captured["forward_kwargs"]["height"] == 256
    assert captured["forward_kwargs"]["width"] == 384
    assert captured["forward_kwargs"]["num_frames"] == 17
    assert captured["forward_kwargs"]["frame_rate"] == 12.0
    assert captured["forward_kwargs"]["steps"] == 3
    assert captured["forward_kwargs"]["sigma_schedule"] == "distilled"
    assert captured["forward_kwargs"]["sigmas"] == "1.0,0.5,0.0"
    assert captured["forward_kwargs"]["negative_prompt"] == "low quality"
    assert captured["forward_kwargs"]["video_cfg_guidance_scale"] == 1.5
    assert captured["forward_kwargs"]["audio_cfg_guidance_scale"] == 2.5


def test_ltx2_3_safetensors_rejects_model_offload(monkeypatch):
    def fail_load(*_args, **_kwargs):
        raise AssertionError("model should not load when --offload-model is rejected")

    monkeypatch.setattr(sys, "argv", ["prog", "--offload-model", "--compute-device", "cuda"])
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "load_ltx2_one_stage_pipeline",
        fail_load,
    )

    with pytest.raises(RuntimeError, match="--offload-model is not supported"):
        ltx2_safetensors_example.run_model_cli()


def test_ltx2_3_safetensors_run_model_cli_supports_model_ledger_pipeline(
    monkeypatch,
):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["target_config"] = target_config
        captured["calibration"] = calibration

    class FakeX0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.velocity_model = torch.nn.Linear(1, 1)

    class FakeModelLedger:
        def __init__(self, transformer):
            self.transformer_model = transformer

        def transformer(self):
            return self.transformer_model

    transformer = FakeX0()
    pipe = type("FakePipe", (), {"model_ledger": FakeModelLedger(transformer)})()
    target_config = object()
    forward_sentinel = object()

    def fake_forward_fn(pipe_arg, **kwargs):
        captured["forward_pipe"] = pipe_arg
        captured["forward_kwargs"] = kwargs
        return forward_sentinel

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--num-samples", "1", "--cache-mode", "disabled"],
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "load_ltx2_one_stage_pipeline",
        lambda *args, **kwargs: pipe,
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "ltx2_3_safetensors_target_config",
        lambda precision, *, fuse_qkv=False: target_config,
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "validate_ltx2_3_nunchaku_lite_manifest_targets",
        lambda transformer, target_config: None,
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": "0000-0", "prompt": "prompt", "seed": 0}
        ],
    )
    monkeypatch.setattr(
        ltx2_safetensors_example, "ltx2_one_stage_forward_fn", fake_forward_fn
    )
    monkeypatch.setattr(
        ltx2_safetensors_example, "quantize_and_export", fake_quantize_and_export
    )

    ltx2_safetensors_example.run_model_cli()

    assert captured["model"] is transformer.velocity_model
    assert captured["target_config"] is target_config
    assert captured["calibration"].forward_fn is forward_sentinel
    assert captured["forward_pipe"] is pipe
    assert captured["forward_kwargs"]["transformer"] is transformer


def test_ltx2_3_safetensors_legacy_forward_factory_avoids_new_denoiser_import(
    monkeypatch,
):
    calls = []

    class FakePipe:
        model_ledger = object()

    def fake_legacy_forward(pipe, **kwargs):
        calls.append((pipe, kwargs))
        return "legacy-result"

    monkeypatch.setattr(ltx2_safetensors_example, "_require_ltx2_imports", lambda: None)
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "_ltx2_legacy_one_stage_forward",
        fake_legacy_forward,
    )
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils.denoisers", None)

    forward = ltx2_safetensors_example.ltx2_one_stage_forward_fn(
        FakePipe(),
        transformer="transformer",
        height=256,
        width=384,
        num_frames=17,
        frame_rate=12.0,
        steps=3,
        sigma_schedule="distilled",
        sigmas=None,
        negative_prompt="bad",
        video_cfg_guidance_scale=1.0,
        video_stg_guidance_scale=0.0,
        video_rescale_scale=0.0,
        video_stg_blocks=[],
        a2v_guidance_scale=1.0,
        video_skip_step=0,
        audio_cfg_guidance_scale=1.0,
        audio_stg_guidance_scale=0.0,
        audio_rescale_scale=0.0,
        audio_stg_blocks=[],
        v2a_guidance_scale=1.0,
        audio_skip_step=0,
        max_batch_size=1,
        enhance_prompt=False,
    )

    assert forward({"prompt": "p", "seed": 0}) == "legacy-result"
    assert calls[0][1]["transformer"] == "transformer"
    assert calls[0][1]["sample"] == {"prompt": "p", "seed": 0}


def test_ltx2_3_safetensors_inspect_config_does_not_quantize(
    monkeypatch, capsys
):
    class FakeReport:
        def format_text(self):
            return "safetensors target config"

    class FakeX0(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.velocity_model = torch.nn.Linear(1, 1)

    class FakeModelContext:
        def __init__(self, transformer):
            self.transformer = transformer

        def __enter__(self):
            return self.transformer

        def __exit__(self, exc_type, exc, traceback):
            return False

    class FakeStage:
        def __init__(self, transformer):
            self.transformer = transformer

        def model_context(self):
            return FakeModelContext(self.transformer)

    pipe = type("FakePipe", (), {"stage": FakeStage(FakeX0())})()

    def fail_quantize_and_export(**_kwargs):
        raise AssertionError("quantize_and_export should not run")

    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--inspect-config", "--cache-mode", "disabled"],
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "load_ltx2_one_stage_pipeline",
        lambda *args, **kwargs: pipe,
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "ltx2_3_safetensors_target_config",
        lambda precision, *, fuse_qkv=False: object(),
    )
    monkeypatch.setattr(
        ltx2_safetensors_example,
        "inspect_target_config",
        lambda model, target_config: FakeReport(),
    )
    monkeypatch.setattr(
        ltx2_safetensors_example, "quantize_and_export", fail_quantize_and_export
    )

    ltx2_safetensors_example.run_model_cli()

    assert "safetensors target config" in capsys.readouterr().out


def test_ltx2_3_safetensors_calibration_sigmas_use_distilled_by_default(
    monkeypatch,
):
    fake_constants = ModuleType("ltx_pipelines.utils.constants")
    fake_constants.DISTILLED_SIGMAS = torch.tensor([1.0, 0.75, 0.0])
    monkeypatch.setattr(ltx2_safetensors_example, "_require_ltx2_imports", lambda: None)
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils.constants", fake_constants)

    class FakeScheduler:
        def execute(self, steps):
            return torch.tensor([9.0, 0.0])

    pipe = type("FakePipe", (), {"_scheduler": FakeScheduler()})()

    distilled = ltx2_safetensors_example._ltx2_calibration_sigmas(
        pipe, schedule="distilled", steps=5, sigmas=None
    )
    scheduler = ltx2_safetensors_example._ltx2_calibration_sigmas(
        pipe, schedule="scheduler", steps=5, sigmas=None
    )
    custom = ltx2_safetensors_example._ltx2_calibration_sigmas(
        pipe, schedule="distilled", steps=5, sigmas="1.0,0.5,0.0"
    )

    assert torch.equal(distilled, torch.tensor([1.0, 0.75, 0.0]))
    assert torch.equal(scheduler, torch.tensor([9.0, 0.0]))
    assert torch.equal(custom, torch.tensor([1.0, 0.5, 0.0]))


def test_ltx2_3_safetensors_rejects_single_custom_sigma():
    with pytest.raises(ValueError, match="at least two values"):
        ltx2_safetensors_example._parse_sigma_values("1.0")


def test_ltx2_3_distilled_lora_merge_cli_uses_defaults(monkeypatch):
    captured = {}

    def fake_merge_ltx2_lora(**kwargs):
        captured.update(kwargs)
        return Path(kwargs["output_dir"])

    monkeypatch.setattr(sys, "argv", ["prog"])
    monkeypatch.setattr(
        ltx2_lora_merge_example, "merge_ltx2_lora", fake_merge_ltx2_lora
    )

    ltx2_lora_merge_example.main()

    assert captured == {
        "base_model": "dg845/LTX-2.3-Diffusers",
        "lora_path": (
            "/mnt/disks/workspace/research/LTX-2/models/"
            "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        ),
        "output_dir": "outputs/models/ltx2.3-distilled-lora-384-diffusers",
        "lora_scale": 1.0,
        "adapter_name": "distilled_lora",
        "dtype": torch.bfloat16,
        "safe_fusing": False,
    }


def test_ltx2_3_distilled_lora_merge_saves_fused_pipeline(monkeypatch, tmp_path):
    calls = []

    class FakePipe:
        def load_lora_weights(self, lora_path, *, adapter_name):
            calls.append(("load_lora_weights", lora_path, adapter_name))

        def fuse_lora(
            self, *, components, lora_scale, safe_fusing, adapter_names
        ):
            calls.append(
                (
                    "fuse_lora",
                    components,
                    lora_scale,
                    safe_fusing,
                    adapter_names,
                )
            )

        def unload_lora_weights(self):
            calls.append(("unload_lora_weights",))

        def save_pretrained(self, output_dir, *, safe_serialization):
            calls.append(("save_pretrained", output_dir, safe_serialization))

    class FakeLTX2Pipeline:
        @classmethod
        def from_pretrained(cls, model_id, *, torch_dtype):
            calls.append(("from_pretrained", model_id, torch_dtype))
            return FakePipe()

    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(LTX2Pipeline=FakeLTX2Pipeline)
    )

    output_dir = tmp_path / "merged"
    result = ltx2_lora_merge_example.merge_ltx2_lora(
        base_model="base-ltx",
        lora_path=tmp_path / "adapter.safetensors",
        output_dir=output_dir,
        lora_scale=0.75,
        adapter_name="custom_adapter",
        dtype=torch.float16,
        safe_fusing=True,
    )

    assert result == output_dir
    assert calls == [
        ("from_pretrained", "base-ltx", torch.float16),
        (
            "load_lora_weights",
            str(tmp_path / "adapter.safetensors"),
            "custom_adapter",
        ),
        (
            "fuse_lora",
            ["transformer"],
            0.75,
            True,
            ["custom_adapter"],
        ),
        ("unload_lora_weights",),
        ("save_pretrained", output_dir, True),
    ]


def test_ltx2_3_run_model_cli_wires_calibration(monkeypatch, tmp_path):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["spec"] = spec
        captured["target_config"] = target_config
        captured["calibration"] = calibration
        captured["export"] = export
        captured["logging"] = logging

    transformer = torch.nn.Linear(1, 1)

    class FakeLTX2Pipe:
        def __init__(self, transformer):
            self.transformer = transformer
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            return kwargs

    pipe = FakeLTX2Pipe(transformer)
    target_config = object()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--num-samples",
            "2",
            "--batch-size",
            "1",
            "--sample-batch-size",
            "4",
            "--cache-mode",
            "disabled",
            "--device",
            "cpu",
            "--height",
            "256",
            "--width",
            "384",
            "--num-frames",
            "17",
            "--frame-rate",
            "12",
            "--steps",
            "3",
            "--guidance-scale",
            "1.5",
            "--negative-prompt",
            "low quality",
            "--stg-scale",
            "0.25",
            "--modality-scale",
            "1.25",
            "--guidance-rescale",
            "0.1",
            "--decode-timestep",
            "0.2",
            "--decode-noise-scale",
            "0.05",
            "--use-cross-timestep",
            "--max-sequence-length",
            "64",
            "--output",
            str(tmp_path / "ltx2.safetensors"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    monkeypatch.setattr(ltx2_example, "load_ltx2_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(
        ltx2_example,
        "ltx2_3_target_config",
        lambda precision, *, fuse_qkv=False: target_config,
    )
    monkeypatch.setattr(
        ltx2_example,
        "validate_ltx2_3_nunchaku_lite_manifest_targets",
        lambda transformer, target_config: None,
    )
    monkeypatch.setattr(
        ltx2_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": f"{index:04d}-0", "prompt": str(index), "seed": index}
            for index in range(num_samples)
        ],
    )
    monkeypatch.setattr(ltx2_example, "quantize_and_export", fake_quantize_and_export)

    ltx2_example.run_model_cli()

    assert captured["model"] is transformer
    assert captured["target_config"] is target_config
    assert captured["calibration"].num_samples == 2
    assert captured["calibration"].cache_num_samples == 2
    assert captured["calibration"].batch_size == 1
    assert captured["calibration"].sample_batch_size == 4
    assert captured["calibration"].max_rows_per_target == 4096
    assert captured["calibration"].output_dir == Path(
        "outputs/calibration/ltx2.3/int4/inputs/samples"
    )
    save_fn = captured["calibration"].output_save_fn
    assert save_fn.func is ltx2_example.save_ltx2_videos
    assert save_fn.keywords == {"frame_rate": 12.0, "audio_sample_rate": None}
    assert captured["export"].output == tmp_path / "ltx2.safetensors"
    assert captured["logging"].log_dir == str(tmp_path / "logs")
    assert captured["logging"].name == "ltx2"

    result = captured["calibration"].forward_fn({"prompt": "ltx prompt", "seed": 3})
    assert result["prompt"] == "ltx prompt"
    assert result["negative_prompt"] == "low quality"
    assert result["height"] == 256
    assert result["width"] == 384
    assert result["num_frames"] == 17
    assert result["frame_rate"] == 12.0
    assert result["num_inference_steps"] == 3
    assert result["guidance_scale"] == 1.5
    assert result["stg_scale"] == 0.25
    assert result["modality_scale"] == 1.25
    assert result["guidance_rescale"] == 0.1
    assert result["decode_timestep"] == 0.2
    assert result["decode_noise_scale"] == 0.05
    assert result["use_cross_timestep"] is True
    assert result["max_sequence_length"] == 64


def test_ltx2_3_run_model_cli_uses_non_distilled_defaults(monkeypatch):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["spec"] = spec
        captured["target_config"] = target_config
        captured["calibration"] = calibration
        captured["export"] = export
        captured["logging"] = logging

    transformer = torch.nn.Linear(1, 1)

    class FakeLTX2Pipe:
        def __init__(self, transformer):
            self.transformer = transformer

        def __call__(self, **kwargs):
            return kwargs

    pipe = FakeLTX2Pipe(transformer)
    target_config = object()

    def fake_load_ltx2_pipeline(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["load_kwargs"] = kwargs
        return pipe

    monkeypatch.setattr(sys, "argv", ["prog", "--num-samples", "1", "--device", "cpu"])
    monkeypatch.setattr(ltx2_example, "load_ltx2_pipeline", fake_load_ltx2_pipeline)
    monkeypatch.setattr(
        ltx2_example,
        "ltx2_3_target_config",
        lambda precision, *, fuse_qkv=False: target_config,
    )
    monkeypatch.setattr(
        ltx2_example,
        "validate_ltx2_3_nunchaku_lite_manifest_targets",
        lambda transformer, target_config: None,
    )
    monkeypatch.setattr(
        ltx2_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": "0000-0", "prompt": "prompt", "seed": 0}
        ],
    )
    monkeypatch.setattr(ltx2_example, "quantize_and_export", fake_quantize_and_export)

    ltx2_example.run_model_cli()

    assert captured["model_id"] == "dg845/LTX-2.3-Diffusers"
    assert captured["load_kwargs"]["device"] == "cpu"
    assert captured["model"] is transformer
    assert captured["target_config"] is target_config

    result = captured["calibration"].forward_fn({"prompt": "ltx", "seed": 3})
    assert result["prompt"] == "ltx"
    assert result["height"] == 512
    assert result["width"] == 768
    assert result["num_frames"] == 121
    assert result["frame_rate"] == 24.0
    assert result["num_inference_steps"] == 30
    assert result["guidance_scale"] == 3.0
    assert result["negative_prompt"] == ltx2_example._ltx2_default_negative_prompt()
    assert result["output_type"] == "np"
    assert result["return_dict"] is False


def test_ltx2_3_distilled_run_model_cli_uses_distilled_defaults(monkeypatch):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["spec"] = spec
        captured["target_config"] = target_config
        captured["calibration"] = calibration
        captured["export"] = export
        captured["logging"] = logging

    transformer = torch.nn.Linear(1, 1)

    class FakeLTX2Pipe:
        def __init__(self, transformer):
            self.transformer = transformer

        def __call__(self, **kwargs):
            return kwargs

    pipe = FakeLTX2Pipe(transformer)
    target_config = object()

    def fake_load_ltx2_pipeline(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["load_kwargs"] = kwargs
        return pipe

    monkeypatch.setattr(sys, "argv", ["prog", "--num-samples", "1", "--device", "cpu"])
    monkeypatch.setattr(ltx2_example, "load_ltx2_pipeline", fake_load_ltx2_pipeline)
    monkeypatch.setattr(
        ltx2_example,
        "ltx2_3_target_config",
        lambda precision, *, fuse_qkv=False: target_config,
    )
    monkeypatch.setattr(
        ltx2_example,
        "validate_ltx2_3_nunchaku_lite_manifest_targets",
        lambda transformer, target_config: None,
    )
    monkeypatch.setattr(
        ltx2_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": "0000-0", "prompt": "prompt", "seed": 0}
        ],
    )
    monkeypatch.setattr(ltx2_example, "quantize_and_export", fake_quantize_and_export)

    ltx2_distilled_example.run_model_cli()

    assert captured["model_id"] == "dg845/LTX-2.3-Distilled-Diffusers"
    assert captured["load_kwargs"]["device"] == "cpu"
    assert captured["model"] is transformer
    assert captured["target_config"] is target_config
    assert captured["calibration"].num_samples == 1
    assert captured["calibration"].cache_num_samples == 1
    assert captured["calibration"].batch_size == 1
    assert captured["calibration"].sample_batch_size == 1
    assert captured["calibration"].cache_dir == Path(
        "outputs/calibration/ltx2.3-distilled/int4/inputs"
    )
    assert captured["calibration"].artifact_cache.cache_dir == Path(
        "outputs/calibration/ltx2.3-distilled/int4/artifacts"
    )
    assert captured["calibration"].output_dir == Path(
        "outputs/calibration/ltx2.3-distilled/int4/inputs/samples"
    )
    save_fn = captured["calibration"].output_save_fn
    assert save_fn.func is ltx2_example.save_ltx2_videos
    assert save_fn.keywords == {"frame_rate": 24.0, "audio_sample_rate": None}
    assert captured["export"].output == Path(
        "outputs/checkpoints/svdq-int4_r32-ltx2.3-distilled.safetensors"
    )
    assert captured["logging"] == LoggingConfig(
        log_dir="outputs/logs", name="svdq-int4_r32-ltx2.3-distilled"
    )

    result = captured["calibration"].forward_fn({"prompt": "distilled", "seed": 3})
    assert result["prompt"] == "distilled"
    assert result["negative_prompt"] == ltx2_example._ltx2_default_negative_prompt()
    assert result["height"] == 512
    assert result["width"] == 768
    assert result["num_frames"] == 121
    assert result["frame_rate"] == 24.0
    assert result["num_inference_steps"] == 8
    assert result["guidance_scale"] == 1.0
    assert result["output_type"] == "np"
    assert result["return_dict"] is False


def test_ltx2_3_distilled_run_model_cli_wires_cli_overrides(monkeypatch, tmp_path):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["spec"] = spec
        captured["target_config"] = target_config
        captured["calibration"] = calibration
        captured["export"] = export
        captured["logging"] = logging

    transformer = torch.nn.Linear(1, 1)

    class FakeLTX2Pipe:
        def __init__(self, transformer):
            self.transformer = transformer

        def __call__(self, **kwargs):
            return kwargs

    pipe = FakeLTX2Pipe(transformer)
    target_config = object()

    def fake_load_ltx2_pipeline(model_id, **kwargs):
        captured["model_id"] = model_id
        captured["load_kwargs"] = kwargs
        return pipe

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-id",
            "local-ltx2-distilled",
            "--precision",
            "nvfp4",
            "--num-samples",
            "2",
            "--cache-num-samples",
            "4",
            "--batch-size",
            "1",
            "--sample-batch-size",
            "3",
            "--scope-capture-mode",
            "one-target",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--cache-mode",
            "refresh",
            "--device",
            "cpu",
            "--compute-device",
            "cuda",
            "--offload-model",
            "--height",
            "256",
            "--width",
            "384",
            "--num-frames",
            "17",
            "--frame-rate",
            "12",
            "--steps",
            "3",
            "--guidance-scale",
            "1.5",
            "--negative-prompt",
            "low quality",
            "--stg-scale",
            "0.25",
            "--modality-scale",
            "1.25",
            "--guidance-rescale",
            "0.1",
            "--decode-timestep",
            "0.2",
            "--decode-noise-scale",
            "0.05",
            "--use-cross-timestep",
            "--max-sequence-length",
            "64",
            "--output",
            str(tmp_path / "ltx2-distilled.safetensors"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    monkeypatch.setattr(ltx2_example, "load_ltx2_pipeline", fake_load_ltx2_pipeline)
    monkeypatch.setattr(
        ltx2_example,
        "ltx2_3_target_config",
        lambda precision, *, fuse_qkv=False: target_config,
    )
    monkeypatch.setattr(
        ltx2_example,
        "validate_ltx2_3_nunchaku_lite_manifest_targets",
        lambda transformer, target_config: None,
    )
    monkeypatch.setattr(
        ltx2_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": f"{index:04d}-0", "prompt": str(index), "seed": index}
            for index in range(num_samples)
        ],
    )
    monkeypatch.setattr(ltx2_example, "quantize_and_export", fake_quantize_and_export)

    ltx2_distilled_example.run_model_cli()

    assert captured["model_id"] == "local-ltx2-distilled"
    assert captured["load_kwargs"]["device"] == "cpu"
    assert captured["model"] is transformer
    assert captured["target_config"] is target_config
    assert captured["spec"].precision == "fp4"
    assert captured["spec"].compute_device == "cuda"
    assert captured["spec"].offload_model is True
    assert captured["calibration"].num_samples == 2
    assert captured["calibration"].cache_num_samples == 4
    assert captured["calibration"].batch_size == 1
    assert captured["calibration"].sample_batch_size == 3
    assert captured["calibration"].scope_capture_mode == "one_target"
    assert captured["calibration"].cache_dir == tmp_path / "cache" / "nvfp4" / "inputs"
    assert captured["calibration"].artifact_cache.cache_dir == (
        tmp_path / "cache" / "nvfp4" / "artifacts"
    )
    assert captured["calibration"].artifact_cache.cache_mode == "refresh"
    assert captured["calibration"].output_dir == (
        tmp_path / "cache" / "nvfp4" / "inputs" / "samples"
    )
    save_fn = captured["calibration"].output_save_fn
    assert save_fn.func is ltx2_example.save_ltx2_videos
    assert save_fn.keywords == {"frame_rate": 12.0, "audio_sample_rate": None}
    assert captured["calibration"].max_rows_per_target == 4096
    assert captured["export"].output == tmp_path / "ltx2-distilled.safetensors"
    assert captured["logging"].log_dir == str(tmp_path / "logs")
    assert captured["logging"].name == "ltx2-distilled"

    result = captured["calibration"].forward_fn({"prompt": "ltx prompt", "seed": 3})
    assert result["prompt"] == "ltx prompt"
    assert result["negative_prompt"] == "low quality"
    assert result["height"] == 256
    assert result["width"] == 384
    assert result["num_frames"] == 17
    assert result["frame_rate"] == 12.0
    assert result["num_inference_steps"] == 3
    assert result["guidance_scale"] == 1.5
    assert result["stg_scale"] == 0.25
    assert result["modality_scale"] == 1.25
    assert result["guidance_rescale"] == 0.1
    assert result["decode_timestep"] == 0.2
    assert result["decode_noise_scale"] == 0.05
    assert result["use_cross_timestep"] is True
    assert result["max_sequence_length"] == 64


def test_ltx2_3_run_model_cli_wires_calibration_video_saving(monkeypatch, tmp_path):
    captured = {}

    def fake_quantize_and_export(
        *, model, spec, target_config, calibration, export, logging=None
    ):
        captured["model"] = model
        captured["calibration"] = calibration

    transformer = torch.nn.Linear(1, 1)

    class FakeLTX2Pipe:
        def __init__(self, transformer):
            self.transformer = transformer
            self.vocoder = SimpleNamespace(
                config=SimpleNamespace(output_sampling_rate=44100)
            )

    pipe = FakeLTX2Pipe(transformer)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--num-samples",
            "1",
            "--cache-mode",
            "disabled",
            "--device",
            "cpu",
            "--frame-rate",
            "12",
            "--output",
            str(tmp_path / "ltx2.safetensors"),
        ],
    )
    monkeypatch.setattr(ltx2_example, "load_ltx2_pipeline", lambda *args, **kwargs: pipe)
    monkeypatch.setattr(
        ltx2_example,
        "ltx2_3_target_config",
        lambda precision, *, fuse_qkv=False: object(),
    )
    monkeypatch.setattr(
        ltx2_example,
        "validate_ltx2_3_nunchaku_lite_manifest_targets",
        lambda transformer, target_config: None,
    )
    monkeypatch.setattr(
        ltx2_example,
        "standard_prompt_records",
        lambda num_samples, prompt_file: [
            {"filename": "0000-0", "prompt": "0", "seed": 0}
        ],
    )
    monkeypatch.setattr(ltx2_example, "quantize_and_export", fake_quantize_and_export)

    ltx2_example.run_model_cli()

    assert captured["model"] is transformer
    assert captured["calibration"].output_dir == Path(
        "outputs/calibration/ltx2.3/int4/inputs/samples"
    )
    save_fn = captured["calibration"].output_save_fn
    assert save_fn.func is ltx2_example.save_ltx2_videos
    assert save_fn.keywords == {"frame_rate": 12.0, "audio_sample_rate": 44100}


def test_save_ltx2_videos_uses_sample_filenames(monkeypatch, tmp_path):
    calls = []

    def fake_encode(video, *, frame_rate, output_path, audio, audio_sample_rate):
        calls.append(
            {
                "video": video,
                "frame_rate": frame_rate,
                "output_path": output_path,
                "audio_shape": tuple(audio.shape),
                "audio_sample_rate": audio_sample_rate,
            }
        )

    monkeypatch.setattr(ltx2_example, "_encode_ltx2_video", fake_encode)
    result = SimpleNamespace(
        frames=[["a0", "a1"], ["b0", "b1"]],
        audio=torch.zeros(2, 1, 4),
    )

    ltx2_example.save_ltx2_videos(
        result,
        {"filename": ["sample-a", "sample-b"]},
        tmp_path,
        frame_rate=12.0,
        audio_sample_rate=44100,
    )

    assert calls == [
        {
            "video": ["a0", "a1"],
            "frame_rate": 12.0,
            "output_path": tmp_path / "sample-a.mp4",
            "audio_shape": (1, 4),
            "audio_sample_rate": 44100,
        },
        {
            "video": ["b0", "b1"],
            "frame_rate": 12.0,
            "output_path": tmp_path / "sample-b.mp4",
            "audio_shape": (1, 4),
            "audio_sample_rate": 44100,
        },
    ]
