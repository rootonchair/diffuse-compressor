import pytest
import torch
from torch import nn

from diffuse_compressor import (
    ActivationQuantSpec,
    AdaNormAwqW4A16Layout,
    DiffusionQuantSpec,
    NaiveSvdqLayout,
    NunchakuSvdqLayout,
    PatchRule,
    SkipRule,
    SvdqTargetQuant,
    SvdqLayout,
    TargetConfig,
    TargetRule,
    AwqTargetQuant,
    collect_quant_targets,
    prepare_model,
)


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(64, 8)
        self.to_k = nn.Linear(64, 8)
        self.to_v = nn.Linear(64, 8)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()
        self.proj_out = nn.Linear(16, 8)

    def forward(self, x):
        return self.proj_out(x)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.tail = nn.Linear(8, 8)


class SpecialLinear(nn.Linear):
    pass


def test_collect_quant_targets_groups_by_wildcard_index():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                name="qkv",
                modules=[
                    "blocks.*.attn.to_q",
                    "blocks.*.attn.to_k",
                    "blocks.*.attn.to_v",
                ],
                export_name="blocks.{0}.qkv_proj",
                roles=["q", "k", "v"],
            )
        ]
    )

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == [
        "blocks.0.qkv_proj",
        "blocks.1.qkv_proj",
    ]
    assert targets[0].module_names == (
        "blocks.0.attn.to_q",
        "blocks.0.attn.to_k",
        "blocks.0.attn.to_v",
    )
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
        "tail",
    ]
    assert targets[0].name == "blocks.0.attn.to_k"
    assert targets[0].module_names == ("blocks.0.attn.to_k",)


def test_collect_quant_targets_can_scan_module_classes_inside_scope_classes():
    model = TinyModel()
    config = TargetConfig(
        targets=[TargetRule(scope_module_classes=TinyBlock, module_classes=nn.Linear)]
    )

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


def test_collect_quant_targets_can_group_members_with_callable_selector():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=TinyAttention,
                member_selector=lambda attn: {
                    "q": attn.to_q,
                    "k": attn.to_k,
                    "v": attn.to_v,
                },
                export_name="{parent_path}.qkv_proj",
            )
        ]
    )

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == [
        "blocks.0.attn.qkv_proj",
        "blocks.1.attn.qkv_proj",
    ]
    assert targets[0].name == "blocks.0.attn"
    assert targets[0].module_names == (
        "blocks.0.attn.to_q",
        "blocks.0.attn.to_k",
        "blocks.0.attn.to_v",
    )
    assert targets[0].roles == ("q", "k", "v")


def test_collect_quant_targets_uses_child_path_for_single_callable_member_default():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=TinyAttention,
                member_selector=lambda attn: {"q": attn.to_q},
            )
        ]
    )

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == [
        "blocks.0.attn.to_q",
        "blocks.1.attn.to_q",
    ]
    assert targets[0].name == "blocks.0.attn.to_q"
    assert targets[0].module_names == ("blocks.0.attn.to_q",)
    assert targets[0].roles == ("q",)


def test_collect_quant_targets_omits_callable_group_members_from_later_scans():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=TinyAttention,
                member_selector=lambda attn: {
                    "q": attn.to_q,
                    "k": attn.to_k,
                    "v": attn.to_v,
                },
                export_name="{parent_path}.qkv_proj",
            ),
            TargetRule(scope_module_classes=TinyBlock, module_classes=nn.Linear),
        ]
    )

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == [
        "blocks.0.attn.qkv_proj",
        "blocks.1.attn.qkv_proj",
        "blocks.0.proj_out",
        "blocks.1.proj_out",
    ]


def test_collect_quant_targets_applies_skip_rules_to_scans():
    model = TinyModel()
    config = TargetConfig(
        targets=[TargetRule(scope_module_classes=TinyBlock, module_classes=nn.Linear)],
        skips=[SkipRule(modules=["blocks.*.proj_out"])],
    )

    targets = collect_quant_targets(model, config)

    assert [target.export_name for target in targets] == [
        "blocks.0.attn.to_k",
        "blocks.0.attn.to_q",
        "blocks.0.attn.to_v",
        "blocks.1.attn.to_k",
        "blocks.1.attn.to_q",
        "blocks.1.attn.to_v",
    ]


def test_collect_quant_targets_rejects_explicit_rules_for_skipped_or_grouped_modules():
    model = TinyModel()
    skipped_config = TargetConfig(
        targets=[TargetRule(modules=["blocks.*.proj_out"])],
        skips=[SkipRule(modules=["blocks.*.proj_out"])],
    )

    with pytest.raises(ValueError, match="explicitly selects skipped modules"):
        collect_quant_targets(model, skipped_config)

    grouped_config = TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=TinyAttention,
                member_selector=lambda attn: {
                    "q": attn.to_q,
                    "k": attn.to_k,
                    "v": attn.to_v,
                },
                export_name="{parent_path}.qkv_proj",
            ),
            TargetRule(modules=["blocks.*.attn.to_q"]),
        ]
    )

    with pytest.raises(ValueError, match="explicitly selects grouped modules"):
        collect_quant_targets(model, grouped_config)


def test_collect_quant_targets_rejects_callable_selector_modules_outside_model():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=TinyAttention,
                member_selector=lambda attn: {"q": nn.Linear(64, 8)},
                export_name="{parent_path}.qkv_proj",
            )
        ]
    )

    with pytest.raises(ValueError, match="not present in model.named_modules"):
        collect_quant_targets(model, config)


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
                quant=SvdqTargetQuant(
                    precision="int4",
                    group_size=64,
                    rank=0,
                    smooth=False,
                    activation_quant=ActivationQuantSpec(enabled=False),
                    shift_activations=False,
                ),
            )
        ]
    )

    target = collect_quant_targets(model, config)[0]

    assert isinstance(target.quant, SvdqTargetQuant)
    assert target.quant.precision == "int4"
    assert target.quant.group_size == 64
    assert target.quant.rank == 0
    assert target.quant.smooth is False
    assert isinstance(target.quant.activation_quant, ActivationQuantSpec)
    assert target.quant.activation_quant.enabled is False
    assert target.quant.shift_activations is False


def test_target_rule_accepts_naive_svdq_layout():
    rule = TargetRule(
        "q",
        ["blocks.*.attn.to_q"],
        quant=SvdqTargetQuant(weight_layout=NaiveSvdqLayout()),
    )

    assert rule.quant.weight_layout.name == "naive_svdq"


def test_svdq_target_quant_defaults_to_nunchaku_svdq_layout():
    quant = SvdqTargetQuant()

    assert isinstance(quant.weight_layout, NunchakuSvdqLayout)


def test_collect_quant_targets_rejects_incompatible_default_nunchaku_layout_with_spec():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.attn.to_q"],
                export_name="blocks.0.q_proj",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="NunchakuSvdqLayout.*output rows"):
        collect_quant_targets(model, config, spec=DiffusionQuantSpec(rank=0))


def test_collect_quant_targets_accepts_aligned_default_nunchaku_layout_with_spec():
    model = nn.Module()
    model.proj = nn.Linear(128, 128)
    config = TargetConfig(targets=[TargetRule("proj", ["proj"])])

    targets = collect_quant_targets(model, config, spec=DiffusionQuantSpec(rank=16))

    assert [target.export_name for target in targets] == ["proj"]


def test_collect_quant_targets_rejects_incompatible_nunchaku_low_rank_geometry_with_spec():
    model = nn.Module()
    model.proj = nn.Linear(128, 128)
    config = TargetConfig(targets=[TargetRule("proj", ["proj"])])

    with pytest.raises(RuntimeError, match="low-rank geometry.*rank"):
        collect_quant_targets(model, config, spec=DiffusionQuantSpec(rank=8))


def test_collect_quant_targets_allows_explicit_auto_svdq_layout_with_spec():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.attn.to_q"],
                export_name="blocks.0.q_proj",
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            )
        ]
    )

    targets = collect_quant_targets(model, config, spec=DiffusionQuantSpec(rank=8))

    assert [target.export_name for target in targets] == ["blocks.0.q_proj"]


def test_collect_quant_targets_allows_explicit_naive_svdq_layout_with_spec():
    model = TinyModel()
    config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.attn.to_q"],
                export_name="blocks.0.q_proj",
                quant=SvdqTargetQuant(weight_layout=NaiveSvdqLayout()),
            )
        ]
    )

    targets = collect_quant_targets(model, config, spec=DiffusionQuantSpec(rank=8))

    assert [target.export_name for target in targets] == ["blocks.0.q_proj"]


def test_target_rule_rejects_invalid_override_values():
    with pytest.raises(ValueError, match="modules, module_classes, or member_selector"):
        TargetRule()
    with pytest.raises(TypeError, match="module_classes"):
        TargetRule(module_classes=("not-a-class",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rank"):
        SvdqTargetQuant(rank=-1)
    with pytest.raises(TypeError, match="activation_quant"):
        SvdqTargetQuant(activation_quant="disabled")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bias"):
        SvdqTargetQuant(bias="always")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="weight_layout"):
        SvdqTargetQuant(weight_layout="awq")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layout"):
        AwqTargetQuant(layout=NaiveSvdqLayout())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="split count"):
        AdaNormAwqW4A16Layout(splits=4)  # type: ignore[arg-type]
    with pytest.raises(
        ValueError, match="member_selector cannot be combined with module_classes"
    ):
        TargetRule(
            parent_module_classes=TinyAttention,
            module_classes=nn.Linear,
            member_selector=lambda attn: {"q": attn.to_q},
        )


def test_skip_rule_rejects_missing_selector():
    with pytest.raises(ValueError, match="SkipRule requires"):
        SkipRule()


def test_split_linear_patch_preserves_output_and_exposes_children():
    torch.manual_seed(0)
    model = TinyModel()
    x = torch.randn(2, 16)
    expected = model.blocks[0](x)

    prepare_model(
        model,
        [
            PatchRule(
                type="split_linear", module="blocks.*.proj_out", args={"splits": [8]}
            )
        ],
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

    prepare_model(
        model, [PatchRule(type="split_conv", module="0", args={"splits": [2]})]
    )

    actual = model(x)
    assert torch.allclose(actual, expected, atol=1e-6)
    modules = dict(model.named_modules())
    assert "0.convs.0" in modules
    assert "0.convs.1" in modules
