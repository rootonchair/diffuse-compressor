import pytest
import torch
from torch import nn

from diffuse_compressor import ActivationQuantSpec, PatchRule, TargetConfig, TargetRule, collect_quant_targets, prepare_model


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(64, 8)
        self.attn.to_k = nn.Linear(64, 8)
        self.attn.to_v = nn.Linear(64, 8)
        self.proj_out = nn.Linear(16, 8)

    def forward(self, x):
        return self.proj_out(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])


class SpecialLinear(nn.Linear):
    pass


def test_collect_quant_targets_groups_by_wildcard_index():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                name="qkv",
                modules=["blocks.*.attn.to_q", "blocks.*.attn.to_k", "blocks.*.attn.to_v"],
                export_name="blocks.{0}.qkv_proj",
                roles=["q", "k", "v"],
            )
        ]
    )

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == ["blocks.0.qkv_proj", "blocks.1.qkv_proj"]
    assert targets[0].module_names == ("blocks.0.attn.to_q", "blocks.0.attn.to_k", "blocks.0.attn.to_v")
    assert targets[0].roles == ("q", "k", "v")


def test_collect_quant_targets_can_match_module_classes_without_patterns():
    model = TinyModel()
    config = TargetConfig(targets=[TargetRule(module_classes=nn.Linear)])

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == [
        "blocks.0.attn.to_k",
        "blocks.0.attn.to_q",
        "blocks.0.attn.to_v",
        "blocks.0.proj_out",
        "blocks.1.attn.to_k",
        "blocks.1.attn.to_q",
        "blocks.1.attn.to_v",
        "blocks.1.proj_out",
    ]
    assert targets[0].name == "blocks.0.attn.to_k"
    assert targets[0].module_names == ("blocks.0.attn.to_k",)


def test_collect_quant_targets_filters_patterns_by_module_class():
    model = TinyModel()
    model.blocks[1].attn.to_q = SpecialLinear(64, 8)
    config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.*.attn.to_q"],
                export_name="blocks.{0}.q_proj",
                module_classes=SpecialLinear,
            )
        ]
    )

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == ["blocks.1.q_proj"]
    assert targets[0].module_names == ("blocks.1.attn.to_q",)


def test_target_rule_resolves_quantization_overrides():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.*.attn.to_q"],
                export_name="blocks.{0}.q_proj",
                precision="int4",
                group_size=64,
                rank=0,
                smooth=False,
                activation_quant=ActivationQuantSpec(enabled=False),
                shift_activations=False,
            )
        ]
    )

    target = collect_quant_targets(model, config)[0]

    assert target.precision == "int4"
    assert target.group_size == 64
    assert target.rank == 0
    assert target.smooth is False
    assert isinstance(target.activation_quant, ActivationQuantSpec)
    assert target.activation_quant.enabled is False
    assert target.shift_activations is False


def test_target_rule_rejects_invalid_override_values():
    with pytest.raises(ValueError, match="modules or module_classes"):
        TargetRule()
    with pytest.raises(TypeError, match="module_classes"):
        TargetRule(module_classes=("not-a-class",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank"):
        TargetRule("q", ["blocks.*.attn.to_q"], rank=-1)
    with pytest.raises(TypeError, match="activation_quant"):
        TargetRule("q", ["blocks.*.attn.to_q"], activation_quant="disabled")  # type: ignore[arg-type]


def test_split_linear_patch_preserves_output_and_exposes_children():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.randn(2, 16)
    expected = model.blocks[0](x)

    prepare_model(
        model,
        [PatchRule(type="split_linear", module="blocks.*.proj_out", args={"splits": [8]})],
    )

    actual = model.blocks[0](x)
    assert torch.allclose(actual, expected, atol=1e-6)
    modules = dict(model.named_modules())
    assert "blocks.0.proj_out.linears.0" in modules
    assert "blocks.0.proj_out.linears.1" in modules


def test_split_linear_output_patch_preserves_output_and_exposes_children():
    torch.manual_seed(0)
    linear = nn.Linear(16, 12)
    model = nn.Sequential(linear)
    x = torch.randn(2, 16)
    expected = model(x)

    prepare_model(
        model,
        [PatchRule(type="split_linear_output", module="0", args={"splits": [4, 4]})],
    )

    actual = model(x)
    assert torch.allclose(actual, expected, atol=1e-6)
    modules = dict(model.named_modules())
    assert "0.linears.0" in modules
    assert "0.linears.1" in modules
    assert "0.linears.2" in modules


def test_split_conv_patch_preserves_output_and_exposes_children():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(4, 3, kernel_size=1, bias=True))
    x = torch.randn(2, 4, 5, 5)
    expected = model(x)

    prepare_model(model, [PatchRule(type="split_conv", module="0", args={"splits": [2]})])

    actual = model(x)
    assert torch.allclose(actual, expected, atol=1e-6)
    modules = dict(model.named_modules())
    assert "0.convs.0" in modules
    assert "0.convs.1" in modules
