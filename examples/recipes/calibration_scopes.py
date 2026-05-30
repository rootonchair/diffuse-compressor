"""Calibration scopes with replay modules, captures, and cache aliases."""

from __future__ import annotations

import torch.nn as nn

from diffuse_compressor import (
    CalibrationCaptureRule,
    CalibrationScopeRule,
    TargetConfig,
    TargetRule,
    inspect_target_config,
)


class Attention(nn.Module):
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


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([Block(), Block(), Block()])


def build_target_config() -> TargetConfig:
    return TargetConfig(
        targets=[
            TargetRule(
                modules=["blocks.*.attn.to_q", "blocks.*.attn.to_k", "blocks.*.attn.to_v"],
                export_name="blocks.{0}.attn.qkv",
                roles=("q", "k", "v"),
            ),
            TargetRule(modules=["blocks.*.mlp"], export_name="blocks.{0}.mlp"),
        ],
        calibration_scopes=[
            CalibrationScopeRule(
                name="blocks.{0}",
                modules=["blocks.*"],
                replay_module="blocks.*",
                eval_module="blocks.*.attn",
                cache_aliases={
                    "blocks.{0}.attn.to_k": "blocks.{0}.attn.to_q",
                    "blocks.{0}.attn.to_v": "blocks.{0}.attn.to_q",
                },
                replay_kwarg_keys=("hidden_states", "encoder_hidden_states"),
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


def main() -> None:
    print(inspect_target_config(TinyModel(), build_target_config()).format_text())


if __name__ == "__main__":
    main()
