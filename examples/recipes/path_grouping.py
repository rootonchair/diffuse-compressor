"""Path-based target rules with wildcard grouping."""

from __future__ import annotations

import torch.nn as nn

from diffuse_compressor import TargetConfig, TargetRule, inspect_target_config


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(32, 16)
        self.to_k = nn.Linear(32, 16)
        self.to_v = nn.Linear(32, 16)
        self.to_out = nn.Linear(16, 32)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ff = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def build_target_config() -> TargetConfig:
    return TargetConfig(
        targets=[
            TargetRule(
                modules=[
                    "blocks.*.attn.to_q",
                    "blocks.*.attn.to_k",
                    "blocks.*.attn.to_v",
                ],
                export_name="blocks.{0}.attn.to_qkv",
                roles=("q", "k", "v"),
            ),
            TargetRule(
                modules=["blocks.*.attn.to_out"], export_name="blocks.{0}.attn.to_out"
            ),
            TargetRule(modules=["blocks.*.ff.0"], export_name="blocks.{0}.mlp_fc1"),
            TargetRule(modules=["blocks.*.ff.2"], export_name="blocks.{0}.mlp_fc2"),
        ]
    )


def main() -> None:
    print(inspect_target_config(TinyModel(), build_target_config()).format_text())


if __name__ == "__main__":
    main()
