"""Target quant policies for standalone W4A16 linear targets."""

from __future__ import annotations

import torch.nn as nn

from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    AwqW4A16Layout,
    TargetConfig,
    TargetRule,
    W4A16TargetQuant,
    inspect_target_config,
)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_modulation = nn.Linear(32, 96)
        self.context_modulation = nn.Linear(32, 192)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block()])


def build_target_config() -> TargetConfig:
    return TargetConfig(
        targets=[
            TargetRule(
                modules=["blocks.*.norm_modulation"],
                export_name="blocks.{0}.norm_modulation",
                quant=W4A16TargetQuant(layout=AwqW4A16Layout()),
            ),
            TargetRule(
                modules=["blocks.*.context_modulation"],
                export_name="blocks.{0}.context_modulation",
                quant=W4A16TargetQuant(layout=AdaNormAwqW4A16Layout(splits=6)),
            ),
        ]
    )


def main() -> None:
    print(inspect_target_config(TinyModel(), build_target_config()).format_text())


if __name__ == "__main__":
    main()
