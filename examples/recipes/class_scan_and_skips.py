"""Class-based target scans constrained by block scope classes."""

from __future__ import annotations

import torch.nn as nn

from diffuse_compressor import SkipRule, TargetConfig, TargetRule, inspect_target_config


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_in = nn.Linear(16, 32)
        self.norm = nn.LayerNorm(32)
        self.proj_out = nn.Linear(32, 16)
        self.debug_head = nn.Linear(16, 4)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])
        self.final = nn.Linear(16, 16)


def build_target_config() -> TargetConfig:
    return TargetConfig(
        targets=[TargetRule(scope_module_classes=Block, module_classes=nn.Linear)],
        skips=[
            SkipRule(modules=["blocks.*.debug_head"]),
            SkipRule(modules=["final"]),
        ],
        unquantized_patterns=["final.*"],
    )


def main() -> None:
    print(inspect_target_config(TinyModel(), build_target_config()).format_text())


if __name__ == "__main__":
    main()
