import json

import pytest
import safetensors
import torch
from torch import nn

from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    ActivationQuantSpec,
    AwqW4A16Layout,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    NunchakuSvdqLayout,
    PatchRule,
    QuantizationCacheSpec,
    RangeCalibrationSpec,
    SmoothSpec,
    TargetConfig,
    TargetRule,
    WeightRangeCalibrationSpec,
    export_checkpoint,
    quantize_and_export,
)
from diffuse_compressor.artifact_cache import _jsonable


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


class AlignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128, bias=True)

    def forward(self, x):
        return self.proj(x)


class WideOutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 256, bias=True)

    def forward(self, x):
        return self.proj(x)


class TinyConvModel(nn.Module):
    def __init__(self, *, kernel_size=1, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(64, 16, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.proj(x)


def test_artifact_cache_serializes_module_classes_with_fully_qualified_names():
    first = type("RepeatedName", (nn.Linear,), {"__module__": "first_module"})
    second = type("RepeatedName", (nn.Linear,), {"__module__": "second_module"})

    first_config = _jsonable(TargetRule(module_classes=first))
    second_config = _jsonable(TargetRule(module_classes=second))

    assert first_config["module_classes"] == ["first_module.RepeatedName"]
    assert second_config["module_classes"] == ["second_module.RepeatedName"]
    assert first_config["module_classes"] != second_config["module_classes"]


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


def test_quantize_diffusion_can_offload_model_and_compute_on_cpu():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(targets=[TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")])
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=0, group_size=64, smooth=False, compute_device="cpu", offload_model=True),
        targets,
        calibration=None,
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert artifact.metadata["quantization"] == {"compute_device": "cpu", "offload_model": True}
    assert target.metadata["compute_device"] == "cpu"
    assert all(tensor.device.type == "cpu" for tensor in target.state_dict.values())
    assert next(model.parameters()).device.type == "cpu"


def test_cuda_compute_device_requires_cuda_when_unavailable():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(targets=[TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")])
    targets = collect_quant_targets(model, target_config)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        quantize_diffusion(
            model,
            DiffusionQuantSpec(rank=0, group_size=64, smooth=False, compute_device="cuda"),
            targets,
            calibration=None,
            target_config=target_config,
        )


def test_activation_range_metadata_and_weight_range_export_runtime_tensors(tmp_path):
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
    assert "input_scale" not in target.state_dict
    assert "input_zero" not in target.state_dict
    assert "output_scale" not in target.state_dict
    assert "output_zero" not in target.state_dict
    assert target.state_dict["weight_range_scale"].shape == (1,)
    assert target.metadata["activation_quant"]["inputs"]["calibrated"] is True
    assert target.metadata["activation_quant"]["inputs"]["qmin"] == 0
    assert target.metadata["activation_quant"]["inputs"]["num_scales"] == 64
    assert target.metadata["activation_quant"]["outputs"]["calibrated"] is True
    assert target.metadata["activation_quant"]["outputs"]["num_scales"] == 1

    output = tmp_path / "ranges.safetensors"
    export_checkpoint(artifact, ExportSpec(output=output))
    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = json.loads(handle.metadata()["quantization_config"])

    assert "blocks.0.q_proj.input_scale" not in keys
    assert "blocks.0.q_proj.output_zero" not in keys
    assert "blocks.0.q_proj.weight_range_scale" in keys
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


def test_awq_w4a16_target_layout_exports_nunchaku_lite_extra_weight_tensors(tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    from diffuse_compressor.runtime import _reconstruct_target_weight

    class ExtraWeightModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.extra = nn.Linear(64, 64, bias=True)

        def forward(self, x):
            return self.extra(x)

    torch.manual_seed(0)
    model = ExtraWeightModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
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
                weight_layout=AwqW4A16Layout(),
            )
        ]
    )
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(precision="fp4", rank=4, group_size=16, smooth=False),
        collect_quant_targets(model, target_config),
        calibration=None,
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    state = target.state_dict

    assert target.metadata["weight_layout"]["name"] == "awq_w4a16"
    assert state["qweight"].shape == (16, 32)
    assert state["qweight"].dtype == torch.int32
    assert state["wscales"].shape == (1, 64)
    assert state["wzeros"].shape == (1, 64)
    assert "smooth_factor" not in state
    assert "smooth_factor_orig" not in state
    assert "proj_down" not in state

    flat_state = {f"extra.{key}": value for key, value in state.items()}
    reconstructed = _reconstruct_target_weight(
        export_name="extra",
        state=flat_state,
        precision="int4",
        weight_layout={"name": "awq_w4a16"},
    )

    assert reconstructed.shape == model.extra.weight.shape
    assert torch.equal(state["wzeros"], (-7 * state["wscales"].float()).to(dtype=state["wscales"].dtype))

    output = tmp_path / "awq.safetensors"
    export_checkpoint(artifact, ExportSpec(output=output))
    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])
    manifest = metadata["runtime_manifest"]
    assert manifest["schema"] == "nunchaku_lite.runtime_manifest"
    assert manifest["version"] == 1
    assert manifest["nunchaku_format_version"] == 1
    assert manifest["targets"][0]["checkpoint_prefix"] == "extra"
    assert manifest["targets"][0]["nunchaku_op"] == "awq_w4a16"
    assert manifest["targets"][0]["op_options"] == {}


@pytest.mark.parametrize("splits", [3, 6])
def test_adanorm_awq_w4a16_layout_reorders_outputs_and_bias(splits, tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    class AdaNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.Linear(64, 12, bias=True)

        def forward(self, x):
            return self.norm(x)

    model = AdaNormModel().to(torch.bfloat16)
    with torch.no_grad():
        model.norm.weight.copy_(torch.arange(12 * 64, dtype=torch.bfloat16).view(12, 64).mul_(0.001))
        model.norm.bias.copy_(torch.arange(12, dtype=torch.bfloat16))
    target_config = TargetConfig(
        targets=[
            TargetRule(
                "norm",
                ["norm"],
                "norm",
                precision="int4",
                group_size=64,
                rank=0,
                shared_low_rank=False,
                smooth=False,
                activation_quant=False,
                shift_activations=False,
                weight_layout=AdaNormAwqW4A16Layout(splits=splits),
            )
        ]
    )
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(precision="fp4", rank=4, group_size=16, smooth=False),
        collect_quant_targets(model, target_config),
        calibration=None,
        target_config=target_config,
    )
    state = artifact.quantized_targets[0].state_dict
    metadata = artifact.quantized_targets[0].metadata["weight_layout"]

    expected_bias = torch.arange(12, dtype=torch.bfloat16).view(splits, 12 // splits).transpose(0, 1).contiguous()
    delta = torch.zeros(splits, dtype=torch.bfloat16)
    delta[1] = 1
    delta[-2] = 1
    expected_bias = expected_bias.add(delta.view(1, splits)).reshape(12)

    assert metadata == {"name": "adanorm_awq_w4a16", "splits": splits}
    assert torch.equal(state["bias"], expected_bias)
    assert torch.equal(state["wzeros"], (-7 * state["wscales"].float()).to(dtype=state["wscales"].dtype))

    output = tmp_path / f"adanorm_{splits}.safetensors"
    export_checkpoint(artifact, ExportSpec(output=output))
    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        export_metadata = json.loads(handle.metadata()["quantization_config"])
    target = export_metadata["runtime_manifest"]["targets"][0]
    assert target["nunchaku_op"] == "adanorm_awq_w4a16"
    assert target["op_options"] == {"adanorm_splits": splits}


def test_target_export_bias_zero_synthesizes_bias_for_biasless_linear():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    class BiaslessModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(0)
    model = BiaslessModel().to(torch.bfloat16)
    target_config = TargetConfig(targets=[TargetRule("proj", ["proj"], "proj", export_bias="zero")])
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(precision="fp4", rank=4, group_size=16, smooth=False),
        collect_quant_targets(model, target_config),
        calibration=None,
        target_config=target_config,
    )

    bias = artifact.quantized_targets[0].state_dict["bias"]

    assert bias.shape == (32,)
    assert bias.dtype == torch.bfloat16
    assert torch.count_nonzero(bias) == 0


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


def test_nvfp4_export_writes_deepcompressor_split_scales(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj", precision="fp4")]
    )
    output = tmp_path / "fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=16,
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
        keys = set(handle.keys())
        wscales = handle.get_tensor("blocks.0.q_proj.wscales")
        wcscales = handle.get_tensor("blocks.0.q_proj.wcscales")

    assert "blocks.0.q_proj.qweight" in keys
    assert "blocks.0.q_proj.wscales" in keys
    assert "blocks.0.q_proj.wcscales" in keys
    assert "blocks.0.q_proj.wtscale" not in keys
    assert wscales.dtype == torch.float8_e4m3fn
    assert wscales.shape == (4, 8)
    assert wcscales.shape == (8,)
    assert metadata["weight"]["scale_dtypes"] == [None, "sfp8_e4m3_nan"]
    assert metadata["activation"]["scale_dtypes"] == ["sfp8_e4m3_nan"]
    assert metadata["targets"][0]["precision"] == "fp4"
    assert metadata["targets"][0]["group_size"] == 16
    assert metadata["targets"][0]["weight_scale_layout"] == "nvfp4_deepcompressor"
    assert metadata["targets"][0]["runtime_tensor_layout"] == "logical"
    assert "runtime_manifest" not in metadata


def test_nunchaku_svdq_layout_fails_when_target_cannot_pack(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.q"],
                export_name="blocks.0.q_proj",
                precision="fp4",
                weight_layout=NunchakuSvdqLayout(),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="NunchakuSvdqLayout"):
        quantize_and_export(
            model,
            DiffusionQuantSpec(
                rank=0,
                group_size=16,
                smooth=False,
                weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
                activation_quant=ActivationQuantSpec(enabled=True, scale_dtypes=("sfp8_e4m3_nan",)),
            ),
            target_config,
            CalibrationSpec(samples=[{"x": torch.rand(4, 64, dtype=torch.bfloat16)}]),
            ExportSpec(output=tmp_path / "fp4.safetensors"),
        )


def test_aligned_nvfp4_export_writes_nunchaku_packed_svdq_tensors(tmp_path):
    torch.manual_seed(0)
    model = AlignedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[TargetRule(name="proj", modules=["proj"], export_name="proj", precision="fp4")]
    )
    output = tmp_path / "aligned_fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(enabled=True, scale_dtypes=("sfp8_e4m3_nan",)),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(result.checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])
        qweight = handle.get_tensor("proj.qweight")
        wscales = handle.get_tensor("proj.wscales")
        wcscales = handle.get_tensor("proj.wcscales")
        wtscale = handle.get_tensor("proj.wtscale")
        smooth = handle.get_tensor("proj.smooth_factor")
        bias = handle.get_tensor("proj.bias")
        proj_down = handle.get_tensor("proj.proj_down")
        proj_up = handle.get_tensor("proj.proj_up")

    assert qweight.shape == (128, 64)
    assert qweight.dtype == torch.int8
    assert wscales.shape == (8, 128)
    assert wscales.dtype == torch.float8_e4m3fn
    assert wcscales.shape == (128,)
    assert torch.all(wcscales.float() == 1)
    assert wtscale.shape == (1,)
    assert smooth.shape == (128,)
    assert bias.shape == (128,)
    assert proj_down.shape == (128, 16)
    assert proj_up.shape == (128, 16)
    assert metadata["targets"][0]["weight_scale_layout"] == "nvfp4_deepcompressor"
    assert metadata["targets"][0]["runtime_tensor_layout"] == "nunchaku_packed"
    manifest = metadata["runtime_manifest"]
    assert manifest["schema"] == "nunchaku_lite.runtime_manifest"
    assert manifest["version"] == 1
    assert manifest["nunchaku_format_version"] == 1
    assert manifest["requirements"]["precision"] == "fp4"
    assert manifest["requirements"]["weight_dtype"] == "fp4_e2m1_all"
    assert manifest["structural_patches"] == []
    assert manifest["targets"][0]["checkpoint_prefix"] == "proj"
    assert manifest["targets"][0]["nunchaku_op"] == "svdq_w4a4"
    assert manifest["targets"][0]["precision"] == "fp4"
    assert manifest["targets"][0]["rank"] == 16
    assert manifest["targets"][0]["has_bias"] is True


def test_aligned_nvfp4_export_respects_nunchaku_svdq_layout_outer_scale_splits(tmp_path):
    torch.manual_seed(0)
    model = AlignedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="proj",
                modules=["proj"],
                export_name="proj",
                precision="fp4",
                weight_layout=NunchakuSvdqLayout(outer_scale_splits=(64, 64)),
            )
        ]
    )
    output = tmp_path / "aligned_fp4_split_scales.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(enabled=True, scale_dtypes=("sfp8_e4m3_nan",)),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(result.checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])
        keys = set(handle.keys())
        wcscales = handle.get_tensor("proj.wcscales")

    assert "proj.wcscales" in keys
    assert "proj.wtscale" not in keys
    assert wcscales.shape == (128,)
    assert torch.unique(wcscales[:64].float()).numel() == 1
    assert torch.unique(wcscales[64:].float()).numel() == 1
    assert metadata["targets"][0]["weight_layout"] == {
        "name": "nunchaku_svdq",
        "outer_scale_splits": [64, 64],
    }
    manifest_target = metadata["runtime_manifest"]["targets"][0]
    assert manifest_target["nunchaku_op"] == "svdq_w4a4"
    assert manifest_target["op_options"] == {"outer_scale_splits": [64, 64]}


def test_runtime_manifest_records_structural_patches_for_packed_targets(tmp_path):
    torch.manual_seed(0)
    model = WideOutModel().to(torch.bfloat16)
    target_config = TargetConfig(
        patches=[PatchRule(type="split_linear_output", module="proj", args={"splits": [128]})],
        targets=[
            TargetRule(
                name="proj0",
                modules=["proj.linears.0"],
                export_name="proj.linears.0",
                precision="fp4",
            )
        ],
    )
    output = tmp_path / "split_manifest.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(enabled=True, scale_dtypes=("sfp8_e4m3_nan",)),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(result.checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])
        keys = set(handle.keys())

    assert "proj.linears.0.qweight" in keys
    assert metadata["runtime_manifest"]["structural_patches"] == [
        {"type": "split_linear_output", "module": "proj", "args": {"splits": [128]}}
    ]
    assert metadata["runtime_manifest"]["targets"][0]["checkpoint_prefix"] == "proj.linears.0"
    assert metadata["runtime_manifest"]["targets"][0]["source_modules"] == ["proj.linears.0"]


def test_runtime_manifest_omits_grouped_synthetic_targets(tmp_path):
    torch.manual_seed(0)
    model = WideOutModel().to(torch.bfloat16)
    target_config = TargetConfig(
        patches=[PatchRule(type="split_linear_output", module="proj", args={"splits": [128]})],
        targets=[
            TargetRule(
                name="proj",
                modules=["proj.linears.0", "proj.linears.1"],
                export_name="proj",
                precision="fp4",
            )
        ],
    )
    output = tmp_path / "split_synthetic_manifest.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(enabled=True, scale_dtypes=("sfp8_e4m3_nan",)),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(result.checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])

    assert metadata["targets"][0]["export_name"] == "proj"
    assert metadata["targets"][0]["modules"] == ["proj.linears.0", "proj.linears.1"]
    assert "runtime_manifest" not in metadata


def test_shifted_aligned_nvfp4_export_stays_nunchaku_packed(tmp_path):
    torch.manual_seed(0)
    model = AlignedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[TargetRule(name="proj", modules=["proj"], export_name="proj", precision="fp4")]
    )
    output = tmp_path / "shifted_aligned_fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            shift_activations=True,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(enabled=True, scale_dtypes=("sfp8_e4m3_nan",)),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.randn(4, 128, dtype=torch.bfloat16) - 4}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(result.checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = json.loads(handle.metadata()["quantization_config"])
        qweight = handle.get_tensor("proj.qweight")
        bias = handle.get_tensor("proj.bias")
        proj_down = handle.get_tensor("proj.proj_down")

    shifts = metadata["calibration"]["activation_shifts"]
    assert shifts["proj"] > 0
    assert qweight.shape == (128, 64)
    assert bias.shape == (128,)
    assert proj_down.shape == (128, 16)
    assert metadata["targets"][0]["runtime_tensor_layout"] == "nunchaku_packed"


def test_pointwise_conv_target_quantizes_and_records_activation_range_metadata(tmp_path):
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
    assert "proj.input_scale" not in keys
    assert "proj.output_scale" not in keys
    assert metadata["targets"][0]["modules"] == ["proj"]
    assert metadata["targets"][0]["activation_quant"]["inputs"]["calibrated"] is True
    assert metadata["targets"][0]["activation_quant"]["outputs"]["calibrated"] is True
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
