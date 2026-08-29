from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from flux2_synthetic import synthetic_denoise_forward_fn, synthetic_prompt_embeds


class StubFlux2Transformer(nn.Module):
    def __init__(self, in_channels: int, joint_attention_dim: int):
        super().__init__()
        self.config = SimpleNamespace(
            in_channels=in_channels, joint_attention_dim=joint_attention_dim
        )
        self.proj = nn.Linear(in_channels, in_channels)
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        hidden_states,
        timestep=None,
        guidance=None,
        encoder_hidden_states=None,
        img_ids=None,
        txt_ids=None,
        return_dict=True,
    ):
        self.calls.append(
            {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "guidance": guidance,
                "encoder_hidden_states": encoder_hidden_states,
                "img_ids": img_ids,
                "txt_ids": txt_ids,
            }
        )
        assert not return_dict
        return (self.proj(hidden_states),)


def _make_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)


def test_gaussian_prompt_embeds_shape_and_determinism():
    config = SimpleNamespace(joint_attention_dim=24)
    first = synthetic_prompt_embeds(config, 3, seed=7, max_sequence_length=16)
    second = synthetic_prompt_embeds(config, 3, seed=7, max_sequence_length=16)
    assert first.shape == (3, 16, 24)
    assert first.dtype == torch.bfloat16
    assert torch.equal(first, second)
    assert not torch.equal(first, synthetic_prompt_embeds(config, 3, seed=8, max_sequence_length=16))


def test_denoise_forward_fn_drives_expected_transformer_calls():
    steps = 3
    height = width = 64  # 4x4 latent grid
    model = StubFlux2Transformer(in_channels=8, joint_attention_dim=24)
    embeds = synthetic_prompt_embeds(model.config, 2, seed=0, max_sequence_length=16)
    forward = synthetic_denoise_forward_fn(
        model,
        model_id="unused",
        height=height,
        width=width,
        steps=steps,
        prompt_embeds=embeds,
        device="cpu",
        scheduler=_make_scheduler(),
    )

    forward({"seed": 1})
    assert len(model.calls) == steps
    seq = (height // 16) * (width // 16)
    for call in model.calls:
        assert call["hidden_states"].shape == (1, seq, 8)
        assert call["encoder_hidden_states"].shape == (1, 16, 24)
        assert call["guidance"] is None
        assert call["timestep"].shape == (1,)
        assert 0.0 <= float(call["timestep"][0]) <= 1.0
        assert call["img_ids"].shape == (1, seq, 4)
        assert call["txt_ids"].shape == (1, 16, 4)
    # trajectory advances: latents fed to later steps differ from the initial noise
    assert not torch.equal(
        model.calls[0]["hidden_states"], model.calls[-1]["hidden_states"]
    )

    # batched sample (batch collation turns seeds into a list)
    model.calls.clear()
    forward({"seed": [0, 1]})
    assert model.calls[0]["hidden_states"].shape == (2, seq, 8)
    # per-seed latents differ within the batch
    assert not torch.equal(
        model.calls[0]["hidden_states"][0], model.calls[0]["hidden_states"][1]
    )


def test_denoise_forward_fn_seed_determinism():
    model = StubFlux2Transformer(in_channels=8, joint_attention_dim=24)
    embeds = synthetic_prompt_embeds(model.config, 2, seed=0, max_sequence_length=16)

    def run(seed):
        local = StubFlux2Transformer(in_channels=8, joint_attention_dim=24)
        local.load_state_dict(model.state_dict())
        forward = synthetic_denoise_forward_fn(
            local,
            model_id="unused",
            height=64,
            width=64,
            steps=2,
            prompt_embeds=embeds,
            device="cpu",
            scheduler=_make_scheduler(),
        )
        forward({"seed": seed})
        return local.calls[0]["hidden_states"]

    assert torch.equal(run(1), run(1))
    assert not torch.equal(run(1), run(2))
