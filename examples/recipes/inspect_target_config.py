"""Inspect common target-config patterns on tiny local models.

Run:

    python examples/recipes/inspect_target_config.py

The examples are intentionally local-only: no Diffusers pipeline, model
download, calibration, quantization, or checkpoint export is required.
"""

from __future__ import annotations

import torch.nn as nn

from diffuse_compressor import (
    AwqW4A16Layout,
    CalibrationCaptureRule,
    CalibrationScopeRule,
    SkipRule,
    TargetConfig,
    TargetRule,
    AwqTargetQuant,
    inspect_target_config,
)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(16, 8)
        self.k = nn.Linear(16, 8)
        self.v = nn.Linear(16, 8)
        self.out = nn.Linear(8, 16)


class FusedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(16, 8)
        self.to_k = nn.Linear(16, 8)
        self.to_v = nn.Linear(16, 8)
        self.to_out = nn.Linear(8, 16)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.mlp = nn.Linear(16, 16)
        self.norm = nn.LayerNorm(16)
        self.norm_modulation = nn.Linear(16, 48)


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.extra_attn = FusedAttention()
        self.final = nn.Linear(16, 16)


def path_and_scope_config() -> TargetConfig:
    """Path-based targets with grouped QKV and block calibration scopes."""

    return TargetConfig(
        targets=[
            TargetRule(
                modules=["blocks.*.attn.q", "blocks.*.attn.k", "blocks.*.attn.v"],
                export_name="blocks.{0}.attn.qkv",
                roles=("q", "k", "v"),
            ),
            TargetRule(
                modules=["blocks.*.attn.out"], export_name="blocks.{0}.attn.out"
            ),
            TargetRule(modules=["blocks.*.mlp"], export_name="blocks.{0}.mlp"),
        ],
        skips=[SkipRule(modules=["final"])],
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
                        outputs=True,
                        input_keys=("hidden_states",),
                    )
                ],
            )
        ],
        unquantized_patterns=["final.*"],
    )


def callable_and_class_scan_config() -> TargetConfig:
    """Callable QKV grouping plus a broad class scan for the remaining linears."""

    return TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=FusedAttention,
                member_selector=lambda attn: {
                    "q": attn.to_q,
                    "k": attn.to_k,
                    "v": attn.to_v,
                },
                export_name="{parent_path}.to_qkv",
            ),
            TargetRule(scope_module_classes=Block, module_classes=nn.Linear),
        ],
        skips=[
            SkipRule(modules=["blocks.*.attn.q", "blocks.*.attn.k", "blocks.*.attn.v"]),
            SkipRule(modules=["blocks.*.norm_modulation"]),
        ],
    )


def awq_target_config() -> TargetConfig:
    """AWQ target policy for runtime-specific tensors."""

    return TargetConfig(
        targets=[
            TargetRule(
                modules=["blocks.*.norm_modulation"],
                export_name="blocks.{0}.norm_modulation",
                quant=AwqTargetQuant(layout=AwqW4A16Layout()),
            )
        ]
    )


def broken_config() -> TargetConfig:
    """Intentionally broken config that demonstrates diagnostics messages."""

    return TargetConfig(
        targets=[
            TargetRule(modules=["blocks.*.missing_q"], export_name="broken.{0}.q"),
            TargetRule(modules=["blocks.*.attn.out"], export_name="duplicate"),
            TargetRule(modules=["blocks.*.mlp"], export_name="duplicate"),
        ],
        calibration_scopes=[
            CalibrationScopeRule(name="missing.{0}", modules=["missing_blocks.*"]),
        ],
        unquantized_patterns=["does_not_exist.*"],
    )


def print_report(title: str, config: TargetConfig) -> None:
    print(f"\n## {title}")
    print(inspect_target_config(TinyTransformer(), config).format_text())


def main() -> None:
    print_report("Path targets and calibration scopes", path_and_scope_config())
    print_report("Callable groups and class scans", callable_and_class_scan_config())
    print_report("AWQ target policy", awq_target_config())
    print_report("Broken config diagnostics", broken_config())


if __name__ == "__main__":
    main()
