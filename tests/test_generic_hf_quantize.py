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
from examples.image_to_image.quantize_hf import build_parser as build_image_parser
from examples.image_to_image.quantize_hf import scan_linear_targets as scan_image_targets
from examples.text_to_image.quantize_hf import build_parser as build_image_generation_parser
from examples.text_to_image.quantize_hf import discover_denoiser, scan_linear_targets
from examples.text_to_video.quantize_hf import build_parser as build_video_parser
from examples.text_to_video.quantize_hf import save_diffusers_videos, scan_linear_targets as scan_video_targets


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


class RepeatedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128)
        self.ff = nn.Linear(128, 128)


class RepeatedDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_embedder = nn.Linear(128, 128)
        self.blocks = nn.ModuleList([RepeatedBlock(), RepeatedBlock()])
        self.other_blocks = nn.ModuleList([RepeatedBlock(), RepeatedBlock(), RepeatedBlock()])
        self.output_projection = nn.Linear(128, 128)
        self.img_mod = nn.Sequential(nn.Identity(), nn.Linear(128, 384))


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


def test_generic_scan_honors_skip_patterns():
    result = scan_linear_targets(TinyDenoiser(), skip=("norm.*", "img_mod.*"))
    assert result.svdq_targets == ("proj", "norm_only_by_name")
    assert result.awq_targets == ()


def test_task_specific_generic_scripts_have_independent_parsers_and_scanners():
    text_args = {action.dest for action in build_image_generation_parser()._actions}
    image_args = {action.dest for action in build_image_parser()._actions}
    video_args = {action.dest for action in build_video_parser()._actions}

    assert "dataset" not in text_args and "num_frames" not in text_args
    assert {"dataset", "dataset_config", "dataset_split", "image_column", "prompt_column"} <= image_args
    assert "num_frames" not in image_args
    assert {"num_frames", "fps"} <= video_args
    assert "dataset" not in video_args

    model = TinyDenoiser()
    expected = scan_linear_targets(model)
    assert scan_image_targets(model).svdq_targets == expected.svdq_targets
    assert scan_video_targets(model).awq_targets == expected.awq_targets


def test_text_to_video_generic_saver_exports_each_sample(monkeypatch, tmp_path):
    calls = []

    def fake_export(frames, path, *, fps):
        calls.append((frames, path, fps))

    monkeypatch.setattr("diffusers.utils.export_to_video", fake_export)
    result = SimpleNamespace(frames=[["a0", "a1"], ["b0", "b1"]])
    save_diffusers_videos(result, {"filename": ["a", "b"]}, tmp_path, fps=12)

    assert calls == [
        (["a0", "a1"], str(tmp_path / "a.mp4"), 12),
        (["b0", "b1"], str(tmp_path / "b.mp4"), 12),
    ]


def test_generic_scan_limits_svdq_to_repeated_module_list_blocks():
    result = scan_linear_targets(RepeatedDenoiser())

    assert result.svdq_targets == (
        "blocks.0.proj",
        "blocks.0.ff",
        "blocks.1.proj",
        "blocks.1.ff",
        "other_blocks.0.proj",
        "other_blocks.0.ff",
        "other_blocks.1.proj",
        "other_blocks.1.ff",
        "other_blocks.2.proj",
        "other_blocks.2.ff",
    )
    assert result.awq_targets == ("img_mod.1",)
    assert ("input_embedder", "outside repeated block stacks") in result.skipped
    assert ("output_projection", "outside repeated block stacks") in result.skipped
    assert tuple(rule.modules[0] for rule in result.target_config.calibration_scopes) == (
        "blocks.0",
        "blocks.1",
        "other_blocks.0",
        "other_blocks.1",
        "other_blocks.2",
    )
    assert all(not rule.use_prev_scope_outputs for rule in result.target_config.calibration_scopes)


def test_generic_scan_repeated_blocks_still_honors_skip_patterns():
    result = scan_linear_targets(RepeatedDenoiser(), skip=("blocks.0.*", "img_mod.*"))

    assert not any(name.startswith("blocks.0.") for name in result.svdq_targets)
    assert result.awq_targets == ()
    assert tuple(rule.modules[0] for rule in result.target_config.calibration_scopes) == (
        "blocks.1",
        "other_blocks.0",
        "other_blocks.1",
        "other_blocks.2",
    )


def test_discover_denoiser_prefers_transformer_then_unet():
    transformer = TinyDenoiser()
    unet = TinyDenoiser()
    assert discover_denoiser(SimpleNamespace(transformer=transformer, unet=unet)) == ("transformer", transformer)
    assert discover_denoiser(SimpleNamespace(unet=unet)) == ("unet", unet)


def test_generic_mixed_checkpoint_exports_plain_awq(tmp_path):
    model = TinyDenoiser()
    result = scan_linear_targets(model, skip=("img_mod.*", "norm_only_by_name"))
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
