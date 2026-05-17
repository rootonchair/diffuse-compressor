import json

import safetensors
import torch
from torch import nn

from diffuse_compressor import (
    ActivationQuantSpec,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    QuantizationCacheSpec,
    RangeCalibrationSpec,
    SmoothSpec,
    TargetConfig,
    TargetRule,
    WeightRangeCalibrationSpec,
    quantize_and_export,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.q = nn.Linear(64, 8, bias=True)
            block.k = nn.Linear(64, 8, bias=True)
            block.v = nn.Linear(64, 8, bias=True)
            block.out = nn.Linear(64, 8, bias=True)
            self.blocks.append(block)
        self.final = nn.Linear(8, 8)

    def forward(self, x):
        return self.blocks[0].q(x)


class TinyConvModel(nn.Module):
    def __init__(self, *, kernel_size=1, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(64, 16, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.proj(x)


def test_quantize_and_export_writes_nunchaku_safetensors(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    output = tmp_path / "tiny.safetensors"
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="qkv",
                modules=["blocks.*.q", "blocks.*.k", "blocks.*.v"],
                export_name="blocks.{0}.qkv_proj",
                roles=["q", "k", "v"],
            ),
            TargetRule(
                name="out",
                modules=["blocks.*.out"],
                export_name="blocks.{0}.out_proj",
            ),
        ],
        unquantized_patterns=["final.*"],
    )

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        target_config,
        CalibrationSpec(prompts=["a prompt"], num_samples=1),
        ExportSpec(output=output),
    )

    assert result.checkpoint_path == str(output)
    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])
        keys = set(handle.keys())

    assert metadata["method"] == "svdquant"
    assert metadata["rank"] == 4
    assert metadata["weight"]["dtype"] == "int4"
    assert "blocks.0.qkv_proj.qweight" in keys
    assert "blocks.0.qkv_proj.proj_down" in keys
    assert "blocks.1.out_proj.wscales" in keys
    assert "final.weight" in keys
    assert "blocks.0.q.weight" not in keys


def test_quantize_diffusion_captures_calibration_inputs():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.q"],
                export_name="blocks.0.q_proj",
            ),
        ],
    )
    targets = collect_quant_targets(model, target_config)
    samples = [{"x": torch.randn(4, 64, dtype=torch.bfloat16)} for _ in range(2)]

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        targets,
        calibration=CalibrationSpec(samples=samples, num_samples=2, max_rows_per_target=5),
        target_config=target_config,
    )

    assert artifact.metadata["calibration"]["captured_targets"] == ["blocks.0.q_proj"]
    assert artifact.quantized_targets[0].metadata["calibrated"] is True
    assert artifact.quantized_targets[0].state_dict["proj_down"].numel() > 0


def test_activation_and_weight_range_calibration_export_runtime_tensors(tmp_path):
    from diffuse_compressor import collect_quant_targets, export_checkpoint, quantize_diffusion

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")],
    )
    targets = collect_quant_targets(model, target_config)
    samples = [{"x": torch.rand(4, 64, dtype=torch.bfloat16)}]

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=64,
            smooth=False,
            activation_quant=ActivationQuantSpec(
                enabled=True,
                inputs=RangeCalibrationSpec(granularity="channel", symmetric=False, allow_unsigned=True),
                outputs=RangeCalibrationSpec(granularity="tensor"),
            ),
            weight_range_calibration=WeightRangeCalibrationSpec(
                enabled=True,
                range=RangeCalibrationSpec(granularity="group"),
            ),
        ),
        targets,
        calibration=CalibrationSpec(samples=samples),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert target.state_dict["input_scale"].shape == (64,)
    assert target.state_dict["input_zero"].shape == (64,)
    assert target.state_dict["output_scale"].shape == (1,)
    assert target.state_dict["weight_range_scale"].shape == (1,)
    assert target.metadata["activation_quant"]["inputs"]["calibrated"] is True
    assert target.metadata["activation_quant"]["inputs"]["qmin"] == 0

    output = tmp_path / "ranges.safetensors"
    export_checkpoint(artifact, ExportSpec(output=output))
    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = json.loads(handle.metadata()["quantization_config"])

    assert "blocks.0.q_proj.input_scale" in keys
    assert "blocks.0.q_proj.output_zero" in keys
    assert metadata["targets"][0]["export_name"] == "blocks.0.q_proj"


def test_explicit_activation_shift_patches_targets_and_records_metadata():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    from diffuse_compressor.patches import ShiftedLinear

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(targets=[TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")])
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=0, group_size=64, smooth=False, shift_activations=True),
        targets,
        calibration=CalibrationSpec(samples=[{"x": torch.randn(4, 64, dtype=torch.bfloat16) - 2}]),
        target_config=target_config,
    )

    assert isinstance(model.blocks[0].q, ShiftedLinear)
    assert artifact.metadata["calibration"]["activation_shifts"]["blocks.0.q"] > 0
    assert artifact.quantized_targets[0].target.modules[0] is model.blocks[0].q


def test_target_overrides_make_extra_weight_target_weight_only():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    from diffuse_compressor.patches import ShiftedLinear

    class TwoTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(64, 64, bias=True)
            self.extra = nn.Linear(64, 64, bias=True)

        def forward(self, x):
            return self.q(x) + self.extra(x)

    torch.manual_seed(0)
    model = TwoTargetModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["q"], "q"),
            TargetRule(
                "extra",
                ["extra"],
                "extra",
                precision="int4",
                group_size=64,
                rank=0,
                shared_low_rank=False,
                smooth=False,
                activation_quant=False,
                shift_activations=False,
            ),
        ]
    )
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            precision="fp4",
            rank=4,
            group_size=16,
            smooth=SmoothSpec(strategy="manual", alpha=0.5, beta=-1),
            activation_quant=ActivationQuantSpec(
                enabled=True,
                inputs=RangeCalibrationSpec(granularity="group", allow_unsigned=True),
                outputs=RangeCalibrationSpec(granularity="tensor"),
            ),
            shift_activations=True,
        ),
        collect_quant_targets(model, target_config),
        calibration=CalibrationSpec(samples=[{"x": torch.randn(4, 64, dtype=torch.bfloat16) - 2}]),
        target_config=target_config,
    )

    by_name = {target.target.export_name: target for target in artifact.quantized_targets}
    normal = by_name["q"]
    extra = by_name["extra"]

    assert isinstance(model.q, ShiftedLinear)
    assert not isinstance(model.extra, ShiftedLinear)
    assert "q" in artifact.metadata["calibration"]["activation_shifts"]
    assert "extra" not in artifact.metadata["calibration"]["activation_shifts"]

    assert normal.metadata["rank"] == 4
    assert "proj_down" in normal.state_dict
    assert normal.metadata["activation_quant"]["enabled"] is True

    assert extra.metadata["precision"] == "int4"
    assert extra.metadata["group_size"] == 64
    assert extra.metadata["rank"] == 0
    assert extra.metadata["smooth"]["enabled"] is False
    assert extra.metadata["activation_quant"]["enabled"] is False
    assert "proj_down" not in extra.state_dict
    assert "input_scale" not in extra.state_dict
    assert "output_scale" not in extra.state_dict


def test_quantization_artifact_cache_reuses_valid_model_cache(tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    torch.manual_seed(0)
    samples = [{"x": torch.randn(4, 64, dtype=torch.bfloat16)}]
    target_config = TargetConfig(targets=[TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")])
    cache = QuantizationCacheSpec(cache_dir=tmp_path / "artifacts", cache_mode="refresh")

    model = TinyModel().to(torch.bfloat16)
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        collect_quant_targets(model, target_config),
        calibration=CalibrationSpec(samples=samples, artifact_cache=cache),
        target_config=target_config,
    )
    assert (tmp_path / "artifacts" / "model.pt").exists()
    assert (tmp_path / "artifacts" / "smooth.pt").exists()

    reuse_model = TinyModel().to(torch.bfloat16)
    reused = quantize_diffusion(
        reuse_model,
        DiffusionQuantSpec(rank=4, group_size=64),
        collect_quant_targets(reuse_model, target_config),
        calibration=CalibrationSpec(samples=samples, artifact_cache=QuantizationCacheSpec(cache_dir=tmp_path / "artifacts")),
        target_config=target_config,
    )

    assert reused.metadata["artifact_cache"]["hit"] is True
    assert torch.equal(reused.quantized_targets[0].state_dict["qweight"], artifact.quantized_targets[0].state_dict["qweight"])


def test_scale_dtype_metadata_and_target_precision_override(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj", precision="fp4", group_size=64)]
    )
    output = tmp_path / "fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=64,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(enabled=True, scale_dtypes=("sfp8_e4m3_nan",)),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 64, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(result.checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])
        assert "blocks.0.q_proj.qweight" in set(handle.keys())

    assert metadata["weight"]["scale_dtypes"] == [None, "sfp8_e4m3_nan"]
    assert metadata["activation"]["scale_dtypes"] == ["sfp8_e4m3_nan"]
    assert metadata["targets"][0]["precision"] == "fp4"


def test_pointwise_conv_target_quantizes_and_exports_activation_ranges(tmp_path):
    torch.manual_seed(0)
    model = TinyConvModel().to(torch.bfloat16)
    output = tmp_path / "conv.safetensors"
    target_config = TargetConfig(targets=[TargetRule(name="proj", modules=["proj"], export_name="proj", kind="conv")])

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=4,
            group_size=64,
            smooth=False,
            activation_quant=ActivationQuantSpec(
                enabled=True,
                inputs=RangeCalibrationSpec(granularity="channel"),
                outputs=RangeCalibrationSpec(granularity="channel"),
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.randn(1, 64, 4, 4, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(result.checkpoint_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = json.loads(handle.metadata()["quantization_config"])

    assert "proj.qweight" in keys
    assert "proj.input_scale" in keys
    assert "proj.output_scale" in keys
    assert metadata["targets"][0]["modules"] == ["proj"]
    assert metadata["calibration"]["captured_targets"] == ["proj"]


def test_non_pointwise_conv_target_is_rejected(tmp_path):
    model = TinyConvModel(kernel_size=3, padding=1).to(torch.bfloat16)
    target_config = TargetConfig(targets=[TargetRule(name="proj", modules=["proj"], export_name="proj", kind="conv")])

    try:
        quantize_and_export(
            model,
            DiffusionQuantSpec(rank=4, group_size=64, smooth=False),
            target_config,
            calibration=None,
            export=ExportSpec(output=tmp_path / "conv3.safetensors"),
        )
    except NotImplementedError as exc:
        assert "kernel_size=(1, 1)" in str(exc)
    else:
        raise AssertionError("Expected non-pointwise Conv2d target to be rejected")
