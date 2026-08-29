"""Synthetic calibration inputs for FLUX.2 Klein transformers.

Drives the transformer alone through a short flow-match denoising loop on
Gaussian latents and fabricated text embeddings, so activation-based
calibration (smoothing search, activation-weighted SVD) runs without any
dataset, VAE, or — in the default Gaussian mode — text encoder.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
import torch
import torch.nn as nn

SyntheticTextMode = Literal["gaussian", "encoder"]

_TEXT_SEQ_LEN = 512


def synthetic_prompt_embeds(
    model_config: Any,
    num_samples: int,
    *,
    mode: SyntheticTextMode = "gaussian",
    model_id: str | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    max_sequence_length: int = _TEXT_SEQ_LEN,
    prompt_file: str | None = None,
) -> torch.Tensor:
    """Build per-sample text embeddings of shape (num_samples, L, joint_attention_dim)."""

    joint_attention_dim = int(model_config.joint_attention_dim)
    if mode == "gaussian":
        generator = torch.Generator("cpu").manual_seed(seed)
        return torch.randn(
            (num_samples, max_sequence_length, joint_attention_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(dtype)
    if mode != "encoder":
        raise ValueError(f"Unsupported synthetic text mode: {mode!r}")
    if model_id is None:
        raise ValueError("mode='encoder' requires model_id")

    from diffusers.pipelines.flux2.pipeline_flux2_klein import Flux2KleinPipeline
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    from utils import standard_prompt_records

    records = standard_prompt_records(num_samples, prompt_file=prompt_file)
    prompts = [record["prompt"] for record in records]
    if len(prompts) < num_samples:
        prompts = [prompts[i % len(prompts)] for i in range(num_samples)]
    tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        model_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device)
    try:
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(prompts), 4):
                chunks.append(
                    Flux2KleinPipeline._get_qwen3_prompt_embeds(
                        text_encoder,
                        tokenizer,
                        prompts[start : start + 4],
                        dtype=dtype,
                        device=device,
                        max_sequence_length=max_sequence_length,
                    ).cpu()
                )
    finally:
        del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return torch.cat(chunks, dim=0)


def synthetic_denoise_forward_fn(
    model: nn.Module,
    *,
    model_id: str,
    height: int,
    width: int,
    steps: int,
    prompt_embeds: torch.Tensor,
    device: str = "cuda",
    scheduler: Any | None = None,
) -> Callable[[dict[str, Any]], Any]:
    """Return a forward_fn running a flow-match denoising loop on synthetic inputs.

    Each sample is ``{"seed": int}``; the loop mirrors ``Flux2KleinPipeline``
    (sigmas, empirical mu, timestep/1000, guidance=None) so the root pre-hook of
    the calibration cache records ``steps`` realistic transformer input sets per
    trajectory.
    """

    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.flux2.pipeline_flux2_klein import (
        compute_empirical_mu,
        retrieve_timesteps,
    )

    if scheduler is None:
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    config = model.config
    in_channels = int(config.in_channels)
    grid_h, grid_w = height // 16, width // 16
    image_seq_len = grid_h * grid_w
    dtype = next(model.parameters()).dtype

    def forward(sample: dict[str, Any]) -> Any:
        seeds = sample["seed"] if isinstance(sample["seed"], list) else [sample["seed"]]
        batch = len(seeds)
        latents = torch.stack(
            [
                torch.randn(
                    (image_seq_len, in_channels),
                    generator=torch.Generator("cpu").manual_seed(int(seed)),
                    dtype=torch.float32,
                )
                for seed in seeds
            ]
        ).to(device=device, dtype=dtype)
        embeds = torch.stack(
            [prompt_embeds[int(seed) % prompt_embeds.shape[0]] for seed in seeds]
        ).to(device=device, dtype=dtype)
        coords = torch.cartesian_prod(
            torch.arange(1), torch.arange(grid_h), torch.arange(grid_w), torch.arange(1)
        )
        img_ids = coords.unsqueeze(0).expand(batch, -1, -1).to(device)
        text_seq_len = embeds.shape[1]
        txt_coords = torch.cartesian_prod(
            torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(text_seq_len)
        )
        txt_ids = txt_coords.unsqueeze(0).expand(batch, -1, -1).to(device)

        sigmas = np.linspace(1.0, 1 / steps, steps)
        if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
            sigmas = None
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=steps)
        timesteps, _ = retrieve_timesteps(scheduler, steps, device, sigmas=sigmas, mu=mu)
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)
        for t in timesteps:
            timestep = t.expand(latents.shape[0]).to(dtype) / 1000
            noise_pred = model(
                hidden_states=latents,
                timestep=timestep,
                guidance=None,
                encoder_hidden_states=embeds,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]
            latents = scheduler.step(noise_pred[:, : latents.shape[1]], t, latents, return_dict=False)[0]
        return latents

    return forward
