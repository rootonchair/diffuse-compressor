"""Callable grouped targets for stable parent classes with variable paths."""

from __future__ import annotations

import torch.nn as nn

from diffuse_compressor import TargetConfig, TargetRule, inspect_target_config


class Attention(nn.Module):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix
        self.query = nn.Linear(24, 8)
        self.key = nn.Linear(24, 8)
        self.value = nn.Linear(24, 8)
        self.output = nn.Linear(8, 24)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = Attention("first")
        self.second = Attention("second")


def build_target_config() -> TargetConfig:
    return TargetConfig(
        targets=[
            TargetRule(
                parent_module_classes=Attention,
                member_selector=lambda attn: {"q": attn.query, "k": attn.key, "v": attn.value},
                export_name="{parent_path}.qkv",
            ),
            TargetRule(modules=["*.output"], export_name="{0}.out"),
        ]
    )


def main() -> None:
    print(inspect_target_config(TinyModel(), build_target_config()).format_text())


if __name__ == "__main__":
    main()
