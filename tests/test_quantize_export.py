import json

import safetensors
import torch
from torch import nn

from diffuse_compressor import (
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    TargetConfig,
    TargetRule,
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
