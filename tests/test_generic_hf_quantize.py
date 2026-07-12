from types import SimpleNamespace

import torch.nn as nn

from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    AwqTargetQuant,
    AwqW4A16Layout,
    DiffusionQuantSpec,
    ExportSpec,
    quantize_and_export,
)
from examples.quantize_hf import discover_denoiser, scan_linear_targets


class AdaNormBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 384)


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128)
        self.norm = AdaNormBlock()
        self.img_mod = nn.Sequential(nn.Identity(), nn.Linear(128, 384))
        self.norm_only_by_name = nn.Linear(128, 128)
        self.bad = nn.Linear(63, 127)


def test_generic_scan_uses_only_plain_awq_layout():
    result = scan_linear_targets(TinyDenoiser())
    assert result.awq_targets == ("norm.linear", "img_mod.1")
    assert "proj" in result.svdq_targets
    assert "norm_only_by_name" in result.svdq_targets
    assert "norm_only_by_name" in result.ambiguous
    assert any(name == "bad" for name, _ in result.skipped)
    awq_rules = [rule for rule in result.target_config.targets if isinstance(rule.quant, AwqTargetQuant)]
    assert len(awq_rules) == 2
    assert all(isinstance(rule.quant.layout, AwqW4A16Layout) for rule in awq_rules)
    assert all(not isinstance(rule.quant.layout, AdaNormAwqW4A16Layout) for rule in awq_rules)


def test_generic_scan_honors_include_and_skip_patterns():
    result = scan_linear_targets(TinyDenoiser(), include=("proj", "norm.*"), skip=("norm.*",))
    assert result.svdq_targets == ("proj",)
    assert result.awq_targets == ()


def test_discover_denoiser_prefers_transformer_then_unet():
    transformer = TinyDenoiser()
    unet = TinyDenoiser()
    assert discover_denoiser(SimpleNamespace(transformer=transformer, unet=unet)) == ("transformer", transformer)
    assert discover_denoiser(SimpleNamespace(unet=unet)) == ("unet", unet)


def test_generic_mixed_checkpoint_exports_plain_awq(tmp_path):
    model = TinyDenoiser()
    result = scan_linear_targets(model, include=("proj", "norm.linear"))
    output = tmp_path / "generic.safetensors"
    quantize_and_export(
        model,
        DiffusionQuantSpec(rank=32),
        result.target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )
    from safetensors import safe_open

    with safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    assert "proj.proj_down" in keys
    assert "norm.linear.wzeros" in keys
    assert "norm.linear.proj_down" not in keys
