import runpy
from pathlib import Path

import torch.nn as nn

from diffuse_compressor import (
    CalibrationCaptureRule,
    CalibrationScopeRule,
    PatchRule,
    SkipRule,
    TargetConfig,
    TargetRule,
    inspect_target_config,
)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 8)
        self.k = nn.Linear(16, 8)
        self.v = nn.Linear(16, 8)
        self.out = nn.Linear(8, 16)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.mlp = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final = nn.Linear(16, 16)

    def forward(self, x):
        return self.final(self.blocks[0].mlp(x))


def test_inspect_target_config_reports_grouped_targets_and_scopes():
    model = TinyModel()
    target_config = TargetConfig(
        targets=[
            TargetRule(
                modules=["blocks.*.attn.q", "blocks.*.attn.k", "blocks.*.attn.v"],
                export_name="blocks.{0}.attn.qkv",
                roles=("q", "k", "v"),
            )
        ],
        calibration_scopes=[
            CalibrationScopeRule(
                name="blocks.{0}",
                modules=["blocks.*"],
                eval_module="blocks.*.attn",
                replay_module="blocks.*",
                cache_aliases={
                    "blocks.{0}.attn.k": "blocks.{0}.attn.q",
                    "blocks.{0}.attn.v": "blocks.{0}.attn.q",
                },
                capture_modules=[
                    CalibrationCaptureRule(
                        name="blocks.{0}.attn_io",
                        modules=["blocks.*.attn"],
                        inputs=True,
                        outputs=True,
                        input_keys=("hidden_states",),
                    )
                ],
            )
        ],
    )

    report = inspect_target_config(model, target_config)

    assert report.ok
    assert [target.export_name for target in report.targets] == [
        "blocks.0.attn.qkv",
        "blocks.1.attn.qkv",
    ]
    assert report.targets[0].modules == (
        "blocks.0.attn.q",
        "blocks.0.attn.k",
        "blocks.0.attn.v",
    )
    assert report.targets[0].roles == ("q", "k", "v")
    assert [scope.name for scope in report.calibration_scopes] == [
        "blocks.0",
        "blocks.1",
    ]
    assert report.calibration_scopes[0].targets == ("blocks.0.attn.qkv",)
    assert report.calibration_scopes[0].captures[0].name == "blocks.0.attn_io"
    assert (
        report.calibration_scopes[0].cache_aliases["blocks.0.attn.k"]
        == "blocks.0.attn.q"
    )


def test_inspect_target_config_reports_class_scan_and_skips():
    model = TinyModel()
    target_config = TargetConfig(
        targets=[TargetRule(scope_module_classes=Block, module_classes=nn.Linear)],
        skips=[SkipRule(modules=["blocks.*.attn.q"])],
        unquantized_patterns=["final.*"],
    )

    report = inspect_target_config(model, target_config)

    assert report.ok
    assert "blocks.0.attn.q" in report.skipped_modules
    assert "blocks.1.attn.q" in report.skipped_modules
    assert "final.weight" in report.unquantized_keys
    assert "blocks.0.attn.q" not in {
        module for target in report.targets for module in target.modules
    }


def test_inspect_target_config_reports_callable_group_targets():
    model = TinyModel()
    target_config = TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=Attention,
                member_selector=lambda attn: {"q": attn.q, "k": attn.k, "v": attn.v},
                export_name="{parent_path}.qkv",
            )
        ]
    )

    report = inspect_target_config(model, target_config)

    assert report.ok
    assert report.targets[0].export_name == "blocks.0.attn.qkv"
    assert report.targets[0].roles == ("q", "k", "v")


def test_inspect_target_config_reports_errors_without_raising():
    model = TinyModel()
    target_config = TargetConfig(
        targets=[
            TargetRule(modules=["blocks.*.attn.q"], export_name="duplicate"),
            TargetRule(modules=["blocks.*.attn.k"], export_name="duplicate"),
        ]
    )

    report = inspect_target_config(model, target_config)

    assert not report.ok
    assert [error.code for error in report.errors] == ["target_collection_failed"]
    assert "Duplicate export_name" in report.errors[0].message


def test_inspect_target_config_reports_unmatched_patterns_as_warnings_and_errors():
    model = TinyModel()
    target_config = TargetConfig(targets=[TargetRule(modules=["missing.*.q"])])

    report = inspect_target_config(model, target_config)

    assert not report.ok
    assert any(message.code == "target_rule_unmatched" for message in report.warnings)
    assert any(message.code == "target_collection_failed" for message in report.errors)
    assert "missing.*.q" in report.format_text()


def test_inspect_target_config_applies_structural_patches_before_matching():
    class ModelWithFusedLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.fused = nn.Linear(16, 16)

    model = ModelWithFusedLinear()
    target_config = TargetConfig(
        patches=[
            PatchRule(type="split_linear_output", module="fused", args={"splits": [8]})
        ],
        targets=[
            TargetRule(modules=["fused.linears.0"]),
            TargetRule(modules=["fused.linears.1"]),
        ],
    )

    report = inspect_target_config(model, target_config)

    assert report.ok
    assert [target.export_name for target in report.targets] == [
        "fused.linears.0",
        "fused.linears.1",
    ]
    assert hasattr(model.fused, "linears")


def test_inspect_target_config_report_to_dict_is_serializable():
    model = TinyModel()
    report = inspect_target_config(
        model, TargetConfig(targets=[TargetRule(modules=["final"])])
    )

    payload = report.to_dict()

    assert payload["targets"][0]["export_name"] == "final"
    assert payload["errors"] == ()


def test_recipe_scripts_run_without_model_downloads():
    recipe_dir = Path(__file__).resolve().parents[1] / "examples" / "recipes"
    for path in sorted(recipe_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        runpy.run_path(str(path), run_name="__main__")
