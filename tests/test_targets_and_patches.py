import torch
from torch import nn

from diffuse_compressor import PatchRule, TargetConfig, TargetRule, collect_quant_targets, prepare_model


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
