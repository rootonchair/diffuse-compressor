"""Shared DeepCompressor-style diffusion SVDQuant example helpers.

The library core is model-agnostic. This module contains user-side target
configs for the diffusion model families represented by upstream
DeepCompressor's SVDQuant example configs.

Use these functions as architecture templates when adding a new diffusion
transformer:

1. Inspect ``dict(model.named_modules())`` and list every Linear or pointwise
   Conv2d projection that should become a quantized runtime projection.
2. Add broad class-scan ``TargetRule`` entries for plain projections whose
   module path is already the export name.
3. Add grouped ``TargetRule`` entries for fused runtime projection families.
   Grouped rules can use path patterns or callable selectors on parent classes
   such as attention modules.
4. Group patterns only when the modules share the same input activation, such
   as self-attention Q/K/V or cross-attention K/V projections. The grouping key
   is the wildcard capture tuple shared by all patterns, or the matched parent
   module for callable selectors.
5. Use ``*`` for the repeated block index. Every pattern in a grouped rule must
   capture the same wildcard values, and ``export_name`` can reuse those values
   as ``{0}``, ``{1}``, and so on.
6. Add ``PatchRule`` entries only for generic rewrites needed before matching,
   such as splitting a fused projection into children that can be targeted.
7. Add ``CalibrationScopeRule`` entries at the block granularity you want to
   replay and clear from RAM. A simple transformer stack usually needs one
   scope rule for each repeated block collection.
8. Use ``module_classes`` as a guard when broad path patterns should only match
   a specific module implementation, or omit ``name``/``modules`` for a
   class-only selector that uses matched module paths as names.
9. Keep architecture-specific names here, not in ``diffuse_compressor`` core.
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Sequence

import torch
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
from diffusers.models.transformers.transformer_flux import FluxSingleTransformerBlock, FluxTransformerBlock
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Attention,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)

from diffuse_compressor import (
    ActivationQuantSpec,
    CalibrationScopeRule,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    LowRankSolverSpec,
    PatchRule,
    QuantizationCacheSpec,
    RangeCalibrationSpec,
    SmoothSpec,
    TargetConfig,
    TargetRule,
    quantize_and_export,
)


def configure_logging() -> None:
    """Configure concise progress logging for example CLIs."""

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")


Precision = Literal["int4", "nvfp4"]
SvdBackend = Literal["full", "svd_lowrank"]
ModelKey = Literal["flux.1-schnell", "flux.1-dev", "flux.2-klein-4b", "flux.2-klein-9b", "pixart-sigma", "sana-1.6b"]
PromptRecord = dict[str, str | int]
UPSTREAM_DEEPCOMPRESSOR_COMMIT = "69f3473f5e1c1504bae35cc50c7858ef900a9b17"
UPSTREAM_QDIFF_PROMPT_SOURCE = (
    "https://raw.githubusercontent.com/nunchaku-ai/deepcompressor/"
    f"{UPSTREAM_DEEPCOMPRESSOR_COMMIT}/examples/diffusion/prompts/qdiff.yaml"
)


@dataclass(frozen=True)
class ModelDefaults:
    """Default upstream model settings for one diffusion SVDQuant example.

    Args:
        model_id: Hugging Face model id used by the upstream example config.
        output_prefix: Filename-safe model name used for exported checkpoints.
        pipeline_name: Diffusers pipeline class name.
        target_config_fn: Function that returns the model target config.
        steps: Default denoising step count.
        guidance_scale: Default classifier-free guidance scale.
        batch_size: Default calibration batch size.
        torch_dtype: Default model dtype string.
        shared_input_keys: Input keys preserved during cache replay batching.
    """

    model_id: str
    output_prefix: str
    pipeline_name: str
    target_config_fn: Callable[[Precision], TargetConfig]
    steps: int
    guidance_scale: float
    batch_size: int
    torch_dtype: str = "bfloat16"
    shared_input_keys: tuple[str, ...] = ()


MODEL_DEFAULTS: dict[ModelKey, ModelDefaults]


def module_class_selector_config(
    *,
    block_class: type[torch.nn.Module],
    projection_class: type[torch.nn.Module] = torch.nn.Linear,
) -> TargetConfig:
    """Return a compact class-only selector example for new architectures.

    The returned config creates one target per named child module matching
    ``projection_class`` inside ``block_class`` scopes, and one calibration
    scope per named child module matching ``block_class``. Because ``name`` and
    ``modules`` are omitted, the matched module paths become the target/export
    names and scope names.

    For production runtime exports, prefer explicit path-pattern rules when the
    checkpoint loader expects renamed or grouped projections.
    """

    return TargetConfig(
        calibration_scopes=[CalibrationScopeRule(module_classes=block_class)],
        targets=[TargetRule(scope_module_classes=block_class, module_classes=projection_class)],
    )


def svdquant_spec(
    precision: Precision,
    *,
    svd_backend: SvdBackend = "full",
    svd_lowrank_oversample: int = 10,
    svd_lowrank_niter: int = 4,
) -> DiffusionQuantSpec:
    """Build an upstream-style SVDQuant spec for one precision overlay.

    Args:
        precision: Precision overlay name, either ``"int4"`` or ``"nvfp4"``.
        svd_backend: Low-rank SVD backend, ``"full"`` or ``"svd_lowrank"``.
        svd_lowrank_oversample: Extra rank for ``torch.svd_lowrank``.
        svd_lowrank_niter: Power iterations for ``torch.svd_lowrank``.

    Returns:
        Quantization spec matching the selected precision.
    """

    if precision == "int4":
        return DiffusionQuantSpec(
            precision="int4",
            rank=32,
            group_size=64,
            shift_activations=True,
            low_rank_solver=_low_rank_solver(
                svd_backend=svd_backend,
                svd_lowrank_oversample=svd_lowrank_oversample,
                svd_lowrank_niter=svd_lowrank_niter,
            ),
            smooth=_smooth_spec(),
            activation_quant=ActivationQuantSpec(
                enabled=True,
                static=False,
                scale_dtypes=(None,),
                inputs=RangeCalibrationSpec(granularity="group", allow_unsigned=True),
                outputs=RangeCalibrationSpec(granularity="tensor"),
            ),
        )
    if precision == "nvfp4":
        return DiffusionQuantSpec(
            precision="fp4",
            rank=32,
            group_size=16,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            shift_activations=True,
            low_rank_solver=_low_rank_solver(
                svd_backend=svd_backend,
                svd_lowrank_oversample=svd_lowrank_oversample,
                svd_lowrank_niter=svd_lowrank_niter,
            ),
            smooth=_smooth_spec(),
            activation_quant=ActivationQuantSpec(
                enabled=True,
                static=False,
                scale_dtypes=("sfp8_e4m3_nan",),
                inputs=RangeCalibrationSpec(granularity="group", allow_unsigned=True),
                outputs=RangeCalibrationSpec(granularity="tensor"),
            ),
        )
    raise ValueError(f"Unsupported precision: {precision!r}")


def flux1_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return a Flux.1 target config for upstream SVDQuant examples.

    This is the template for architectures with two repeated block families:
    ``transformer_blocks`` and ``single_transformer_blocks``. The important
    adaptation points are:

    - grouped QKV rules concatenate weights from modules that consume the same
      hidden-state input and should share one low-rank branch;
    - context/add-QKV rules are separate because they consume a different
      context stream from the self-attention QKV path;
    - ``single_transformer_blocks.*.proj_out`` is split first because the
      original fused output contains both attention-output and MLP-output
      slices, and the runtime expects them as two export targets;
    - NVFP4 adds target-level INT4 overrides for norm linears because upstream
      DeepCompressor treats those as extra weights, not FP4 residual weights.

    To adapt this to another Flux-like model, keep the grouping semantics but
    replace the module path strings and export names with the runtime names
    expected by the checkpoint loader.

    Args:
        precision: Precision overlay; NVFP4 adds extra INT4 norm targets.

    Returns:
        Target config for Flux.1 Dev/Schnell transformers.
    """

    # Each TargetRule is one exported runtime projection tensor family. Rules
    # with several module patterns are grouped by shared wildcard captures.
    targets = [
        # Group double-block self-attention Q/K/V into one runtime QKV target
        # because all three projections consume the hidden-state stream. For
        # every shared block-index capture, the three matched modules become
        # one target in q/k/v order.
        TargetRule(
            modules=[
                "transformer_blocks.*.attn.to_q",
                "transformer_blocks.*.attn.to_k",
                "transformer_blocks.*.attn.to_v",
            ],
            export_name="transformer_blocks.{0}.qkv_proj",
            roles=["q", "k", "v"],
        ),
        # Group double-block context/add-Q/K/V separately from self QKV because
        # these projections consume the encoder/context stream.
        TargetRule(
            modules=[
                "transformer_blocks.*.attn.add_q_proj",
                "transformer_blocks.*.attn.add_k_proj",
                "transformer_blocks.*.attn.add_v_proj",
            ],
            export_name="transformer_blocks.{0}.qkv_proj_context",
            roles=["add_q", "add_k", "add_v"],
        ),
        # Quantize the double-block self-attention output projection.
        TargetRule(modules=["transformer_blocks.*.attn.to_out.0"], export_name="transformer_blocks.{0}.out_proj"),
        # Quantize the double-block context/add-attention output projection.
        TargetRule(modules=["transformer_blocks.*.attn.to_add_out"], export_name="transformer_blocks.{0}.out_proj_context"),
        # Quantize the first double-block hidden-state MLP projection.
        TargetRule(modules=["transformer_blocks.*.ff.net.0.proj"], export_name="transformer_blocks.{0}.mlp_fc1"),
        # Quantize the second double-block hidden-state MLP projection.
        TargetRule(modules=["transformer_blocks.*.ff.net.2"], export_name="transformer_blocks.{0}.mlp_fc2"),
        # Quantize the first double-block context-stream MLP projection.
        TargetRule(
            modules=["transformer_blocks.*.ff_context.net.0.proj"],
            export_name="transformer_blocks.{0}.mlp_context_fc1",
        ),
        # Quantize the second double-block context-stream MLP projection.
        TargetRule(
            modules=["transformer_blocks.*.ff_context.net.2"],
            export_name="transformer_blocks.{0}.mlp_context_fc2",
        ),
        # Group single-block self-attention Q/K/V into one runtime QKV target.
        TargetRule(
            modules=[
                "single_transformer_blocks.*.attn.to_q",
                "single_transformer_blocks.*.attn.to_k",
                "single_transformer_blocks.*.attn.to_v",
            ],
            export_name="single_transformer_blocks.{0}.qkv_proj",
            roles=["q", "k", "v"],
        ),
        # Quantize the attention-output slice exposed by splitting proj_out.
        TargetRule(
            modules=["single_transformer_blocks.*.proj_out.linears.0"],
            export_name="single_transformer_blocks.{0}.out_proj",
        ),
        # Quantize the single-block MLP input projection.
        TargetRule(modules=["single_transformer_blocks.*.proj_mlp"], export_name="single_transformer_blocks.{0}.mlp_fc1"),
        # Quantize the MLP-output slice exposed by splitting proj_out.
        TargetRule(
            modules=["single_transformer_blocks.*.proj_out.linears.1"],
            export_name="single_transformer_blocks.{0}.mlp_fc2",
        ),
    ]
    if precision == "nvfp4":
        targets.extend(_flux_extra_weight_targets())
    return TargetConfig(
        # Split the fused single-block output into child linears so each child
        # can be matched by an ordinary TargetRule below.
        patches=[PatchRule(type="split_linear", module="single_transformer_blocks.*.proj_out", args={"splits": ["out_features"]})],
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=FluxTransformerBlock,
                use_prev_scope_outputs=True,
                prev_replay_transform=_flux_block_prev_replay_transform,
            ),
            CalibrationScopeRule(
                module_classes=FluxSingleTransformerBlock,
                use_prev_scope_outputs=True,
                prev_replay_transform=_flux_block_prev_replay_transform,
            ),
        ],
        targets=targets,
    )


def _flux_block_prev_replay_transform(replay) -> tuple[tuple, dict]:
    """Build the next Flux block input from the previous block replay."""

    if not isinstance(replay.output, tuple) or len(replay.output) != 2:
        raise TypeError("Flux block replay output must be (encoder_hidden_states, hidden_states)")
    encoder_hidden_states, hidden_states = replay.output
    if replay.kwargs:
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = hidden_states
        kwargs["encoder_hidden_states"] = encoder_hidden_states
        return (), kwargs
    args = list(replay.args)
    if len(args) < 2:
        raise TypeError("Flux block replay args must include hidden_states and encoder_hidden_states")
    args[0] = hidden_states
    args[1] = encoder_hidden_states
    return tuple(args), {}


def _flux2_attention_qkv_members(attn: Flux2Attention) -> dict[str, torch.nn.Module]:
    return {"q": attn.to_q, "k": attn.to_k, "v": attn.to_v}


def _flux2_attention_added_qkv_members(attn: Flux2Attention) -> dict[str, torch.nn.Module]:
    return {"add_q": attn.add_q_proj, "add_k": attn.add_k_proj, "add_v": attn.add_v_proj}


def flux2_klein_target_config(
    precision: Precision = "int4",
    *,
    single_qkv_features: int = 9216,
    single_attn_features: int = 3072,
) -> TargetConfig:
    """Return a FLUX.2 Klein target config for upstream SVDQuant examples.

    FLUX.2 Klein uses Flux-like double/single block families but the single
    block contains fused projections that must be split before targeting:

    - ``attn.to_qkv_mlp_proj`` is split by output features into QKV and MLP-in
      child linears;
    - ``attn.to_out`` is split by input features into attention-out and MLP-out
      child linears;
    - double-block Q/K/V and added-Q/K/V are grouped into runtime QKV targets;
    - the split single-block children are exported under the Nunchaku Lite
      FLUX.2 names.

    Args:
        precision: Precision overlay. The target layout is shared by INT4 and
            NVFP4.
        single_qkv_features: Output-feature width of the single-block fused QKV
            slice. FLUX.2 Klein 4B uses ``9216`` and 9B uses ``12288``.
        single_attn_features: Input-feature width of the single-block attention
            output slice. FLUX.2 Klein 4B uses ``3072`` and 9B uses ``4096``.

    Returns:
        Target config for FLUX.2 Klein transformers.
    """

    return TargetConfig(
        patches=[
            PatchRule(
                type="split_linear_output",
                module="single_transformer_blocks.*.attn.to_qkv_mlp_proj",
                args={"splits": [single_qkv_features]},
            ),
            PatchRule(
                type="split_linear",
                module="single_transformer_blocks.*.attn.to_out",
                args={"splits": [single_attn_features]},
            ),
        ],
        calibration_scopes=[
            CalibrationScopeRule(module_classes=Flux2TransformerBlock),
            CalibrationScopeRule(module_classes=Flux2SingleTransformerBlock),
        ],
        targets=[
            TargetRule(
                parent_module_classes=Flux2Attention,
                member_selector=_flux2_attention_qkv_members,
                export_name="{parent_path}.to_qkv",
            ),
            TargetRule(
                parent_module_classes=Flux2Attention,
                member_selector=_flux2_attention_added_qkv_members,
                export_name="{parent_path}.to_added_qkv",
            ),
            TargetRule(scope_module_classes=Flux2TransformerBlock, module_classes=torch.nn.Linear),
            TargetRule(
                modules=["single_transformer_blocks.*.attn.to_qkv_mlp_proj.linears.0"],
                export_name="single_transformer_blocks.{0}.attn.qkv_proj",
            ),
            TargetRule(
                modules=["single_transformer_blocks.*.attn.to_qkv_mlp_proj.linears.1"],
                export_name="single_transformer_blocks.{0}.attn.mlp_fc1",
            ),
            TargetRule(
                modules=["single_transformer_blocks.*.attn.to_out.linears.0"],
                export_name="single_transformer_blocks.{0}.attn.out_proj",
            ),
            TargetRule(
                modules=["single_transformer_blocks.*.attn.to_out.linears.1"],
                export_name="single_transformer_blocks.{0}.attn.mlp_fc2",
            ),
        ],
    )


def flux2_klein_4b_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return the FLUX.2 Klein 4B upstream target config."""

    return flux2_klein_target_config(precision, single_qkv_features=9216, single_attn_features=3072)


def flux2_klein_9b_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return the FLUX.2 Klein 9B upstream target config."""

    return flux2_klein_target_config(precision, single_qkv_features=12288, single_attn_features=4096)


def pixart_sigma_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return a PixArt Sigma target config for upstream SVDQuant examples.

    This is the template for architectures with one repeated transformer block
    family and separate self-attention/cross-attention modules:

    - self-attention Q/K/V are grouped into one exported QKV projection because
      they share hidden-state inputs;
    - cross-attention Q is kept separate from cross-attention K/V because Q
      consumes hidden states while K/V consume encoder states;
    - K/V are grouped together because they share the same encoder-state input;
    - MLP up/down projections are ordinary one-module targets;
    - the NVFP4 overlay adds ``adaln_single.linear`` as an extra INT4 target,
      mirroring DeepCompressor's extra-weight precision policy.

    To adapt this shape to a new model, first identify which attention
    projections share the same activation tensor. Group only those projections,
    then set ``export_name`` to the name your runtime converter/loader expects.

    Args:
        precision: Precision overlay; NVFP4 adds an extra INT4 AdaLN target.

    Returns:
        Target config for PixArt Sigma transformers.
    """

    # The wildcard captures the block index. Each grouped rule must resolve to
    # one concrete module tuple per block, for example block 0 q/k/v together.
    targets = [
        # Group self-attention Q/K/V because they consume the same hidden-state
        # activation and export as one runtime QKV projection.
        TargetRule(
            modules=["transformer_blocks.*.attn1.to_q", "transformer_blocks.*.attn1.to_k", "transformer_blocks.*.attn1.to_v"],
            export_name="transformer_blocks.{0}.attn1.qkv_proj",
            roles=["q", "k", "v"],
        ),
        # Quantize the self-attention output projection.
        TargetRule(modules=["transformer_blocks.*.attn1.to_out.0"], export_name="transformer_blocks.{0}.attn1.out_proj"),
        # Keep cross-attention Q separate because it consumes hidden states.
        TargetRule(modules=["transformer_blocks.*.attn2.to_q"], export_name="transformer_blocks.{0}.attn2.q_proj"),
        # Group cross-attention K/V because both consume encoder states.
        TargetRule(
            modules=["transformer_blocks.*.attn2.to_k", "transformer_blocks.*.attn2.to_v"],
            export_name="transformer_blocks.{0}.attn2.kv_proj",
            roles=["k", "v"],
        ),
        # Quantize the cross-attention output projection.
        TargetRule(modules=["transformer_blocks.*.attn2.to_out.0"], export_name="transformer_blocks.{0}.attn2.out_proj"),
        # Quantize the first PixArt feed-forward projection.
        TargetRule(modules=["transformer_blocks.*.ff.net.0.proj"], export_name="transformer_blocks.{0}.mlp_fc1"),
        # Quantize the second PixArt feed-forward projection.
        TargetRule(modules=["transformer_blocks.*.ff.net.2"], export_name="transformer_blocks.{0}.mlp_fc2"),
    ]
    if precision == "nvfp4":
        targets.append(
            # NVFP4 upstream config keeps this AdaLN modulation linear as an
            # extra INT4 weight target instead of FP4 residual weight.
            TargetRule(
                modules=["adaln_single.linear"],
                shared_low_rank=False,
                precision="int4",
                group_size=64,
                rank=0,
                smooth=False,
                activation_quant=False,
                shift_activations=False,
            )
        )
    return TargetConfig(
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=BasicTransformerBlock,
            )
        ],
        targets=targets,
    )


def sana_target_config(precision: Precision = "int4") -> TargetConfig:
    """Return a Sana target config for upstream SVDQuant examples.

    This extends the PixArt-style attention layout with convolutional feed
    forward projections:

    - attention grouping follows the same rule as PixArt: self QKV together,
      cross Q alone, cross KV together;
    - ``ff.conv_inverted`` and ``ff.conv_point`` are pointwise Conv2d
      projections and are marked with ``kind="conv"`` so the quantizer flattens
      weights as ``[out_channels, in_channels]`` for export;
    - ``ff.conv_depth`` is intentionally skipped because depthwise/spatial
      convolutions do not match the linear-projector SVDQuant layout;
    - if another model has pointwise Conv1x1 projections, use ``kind="conv"``;
      for depthwise or larger kernels, leave them unquantized or add a new
      quantization method explicitly.

    To adapt this to another convolutional transformer, separate the pointwise
    projection convs from spatial/depthwise convs and only include the
    pointwise ones as Conv2d targets.

    Args:
        precision: Precision overlay.

    Returns:
        Target config for Sana transformers, including pointwise Conv2d FFNs.
    """

    return TargetConfig(
        calibration_scopes=[
            CalibrationScopeRule(
                module_classes=SanaTransformerBlock,
            )
        ],
        targets=[
            # Group self-attention Q/K/V because they share hidden-state inputs
            # and should export as one runtime QKV projection.
            TargetRule(
                modules=["transformer_blocks.*.attn1.to_q", "transformer_blocks.*.attn1.to_k", "transformer_blocks.*.attn1.to_v"],
                export_name="transformer_blocks.{0}.attn1.qkv_proj",
                roles=["q", "k", "v"],
            ),
            # Quantize the self-attention output projection.
            TargetRule(modules=["transformer_blocks.*.attn1.to_out.0"], export_name="transformer_blocks.{0}.attn1.out_proj"),
            # Keep cross-attention Q separate because it consumes hidden states.
            TargetRule(modules=["transformer_blocks.*.attn2.to_q"], export_name="transformer_blocks.{0}.attn2.q_proj"),
            # Group cross-attention K/V because both consume encoder states.
            TargetRule(
                modules=["transformer_blocks.*.attn2.to_k", "transformer_blocks.*.attn2.to_v"],
                export_name="transformer_blocks.{0}.attn2.kv_proj",
                roles=["k", "v"],
            ),
            # Quantize the cross-attention output projection.
            TargetRule(modules=["transformer_blocks.*.attn2.to_out.0"], export_name="transformer_blocks.{0}.attn2.out_proj"),
            # Quantize the pointwise Conv2d expansion projection in Sana's
            # convolutional feed-forward block.
            TargetRule(
                modules=["transformer_blocks.*.ff.conv_inverted"],
                export_name="transformer_blocks.{0}.mlp_fc1",
                kind="conv",
            ),
            # Quantize the pointwise Conv2d output projection and intentionally
            # leave the depthwise convolution unquantized.
            TargetRule(
                modules=["transformer_blocks.*.ff.conv_point"],
                export_name="transformer_blocks.{0}.mlp_fc2",
                kind="conv",
            ),
        ],
    )


MODEL_DEFAULTS = {
    "flux.1-schnell": ModelDefaults(
        model_id="black-forest-labs/FLUX.1-schnell",
        output_prefix="flux.1-schnell",
        pipeline_name="FluxPipeline",
        target_config_fn=flux1_target_config,
        steps=4,
        guidance_scale=0.0,
        batch_size=16,
        shared_input_keys=("txt_ids", "img_ids"),
    ),
    "flux.1-dev": ModelDefaults(
        model_id="black-forest-labs/FLUX.1-dev",
        output_prefix="flux.1-dev",
        pipeline_name="FluxPipeline",
        target_config_fn=flux1_target_config,
        steps=50,
        guidance_scale=3.5,
        batch_size=16,
        shared_input_keys=("txt_ids", "img_ids"),
    ),
    "flux.2-klein-4b": ModelDefaults(
        model_id="black-forest-labs/FLUX.2-klein-4B",
        output_prefix="flux2-klein-4b",
        pipeline_name="Flux2KleinPipeline",
        target_config_fn=flux2_klein_4b_target_config,
        steps=4,
        guidance_scale=1.0,
        batch_size=1,
    ),
    "flux.2-klein-9b": ModelDefaults(
        model_id="black-forest-labs/FLUX.2-klein-9B",
        output_prefix="flux2-klein-9b",
        pipeline_name="Flux2KleinPipeline",
        target_config_fn=flux2_klein_9b_target_config,
        steps=4,
        guidance_scale=1.0,
        batch_size=1,
    ),
    "pixart-sigma": ModelDefaults(
        model_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        output_prefix="pixart-sigma",
        pipeline_name="PixArtSigmaPipeline",
        target_config_fn=pixart_sigma_target_config,
        steps=20,
        guidance_scale=4.5,
        batch_size=256,
    ),
    "sana-1.6b": ModelDefaults(
        model_id="Lawrence-cj/Sana_1600M_1024px_BF16_diffusers_ch5632",
        output_prefix="sana-1.6b",
        pipeline_name="SanaPipeline",
        target_config_fn=sana_target_config,
        steps=20,
        guidance_scale=4.5,
        batch_size=256,
    ),
}


def default_arg_parser(
    model_id: str,
    output: str,
    *,
    steps: int,
    guidance_scale: float,
    batch_size: int,
    torch_dtype: str = "bfloat16",
) -> argparse.ArgumentParser:
    """Create a shared CLI parser for upstream diffusion examples.

    Args:
        model_id: Default Hugging Face model id.
        output: Default output checkpoint path.
        steps: Default inference steps for calibration/eval sample forwards.
        guidance_scale: Default guidance scale.
        batch_size: Default calibration batch size.
        torch_dtype: Default model dtype.

    Returns:
        Configured argument parser.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("int4", "nvfp4"), default="int4")
    parser.add_argument("--model-id", default=model_id)
    parser.add_argument("--output", default=output)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--cache-mode", choices=("reuse", "refresh", "disabled"), default="reuse")
    parser.add_argument("--prompt-file", default=UPSTREAM_QDIFF_PROMPT_SOURCE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", choices=("float16", "bfloat16", "float32"), default=torch_dtype)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--guidance-scale", type=float, default=guidance_scale)
    parser.add_argument("--svd-backend", choices=("full", "svd_lowrank"), default="full")
    parser.add_argument("--svd-lowrank-oversample", type=int, default=10)
    parser.add_argument("--svd-lowrank-niter", type=int, default=4)
    return parser


def run_model_cli(model_key: ModelKey) -> None:
    """Load a supported upstream diffusion pipeline and run quantization.

    Args:
        model_key: Key in :data:`MODEL_DEFAULTS`.
    """

    configure_logging()
    defaults = MODEL_DEFAULTS[model_key]
    output = f"outputs/checkpoints/svdq-int4_r32-{defaults.output_prefix}.safetensors"
    parser = default_arg_parser(
        defaults.model_id,
        output,
        steps=defaults.steps,
        guidance_scale=defaults.guidance_scale,
        batch_size=defaults.batch_size,
        torch_dtype=defaults.torch_dtype,
    )
    args = parser.parse_args()
    if args.output == output:
        args.output = f"outputs/checkpoints/svdq-{args.precision}_r32-{defaults.output_prefix}.safetensors"
    cache_dir = args.cache_dir or f"outputs/calibration/{defaults.output_prefix}"
    pipe = load_pipeline(
        defaults.pipeline_name,
        args.model_id,
        torch_dtype=_resolve_torch_dtype(args.torch_dtype),
        device=args.device,
    )
    records = standard_prompt_records(args.num_samples, prompt_file=args.prompt_file)
    samples = batched_samples(records, args.batch_size)
    forward_fn = pipeline_forward_fn(
        pipe,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
    )
    run_quantization(
        pipe=pipe,
        target_config=defaults.target_config_fn(args.precision),
        precision=args.precision,
        output=args.output,
        cache_dir=cache_dir,
        cache_mode=args.cache_mode,
        samples=samples,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        forward_fn=forward_fn,
        shared_input_keys=defaults.shared_input_keys,
        svd_backend=args.svd_backend,
        svd_lowrank_oversample=args.svd_lowrank_oversample,
        svd_lowrank_niter=args.svd_lowrank_niter,
    )


def run_quantization(
    *,
    pipe,
    target_config: TargetConfig,
    precision: Precision,
    output: str | Path,
    cache_dir: str | Path | None,
    cache_mode: Literal["reuse", "refresh", "disabled"],
    samples: list[dict],
    batch_size: int,
    num_samples: int,
    forward_fn: Callable[[dict], object],
    shared_input_keys: Sequence[str] = (),
    svd_backend: SvdBackend = "full",
    svd_lowrank_oversample: int = 10,
    svd_lowrank_niter: int = 4,
) -> None:
    """Run quantization and export for one example script.

    Args:
        pipe: Diffusers pipeline containing a ``transformer``.
        target_config: Model-specific target configuration.
        precision: Precision overlay.
        output: Output checkpoint path.
        cache_dir: Optional calibration cache directory.
        cache_mode: Calibration and artifact cache mode.
        samples: Calibration sample dictionaries.
        batch_size: Calibration DataLoader batch size.
        num_samples: Calibration sample limit.
        forward_fn: Callable that runs one calibration sample through the
            pipeline.
        shared_input_keys: Input keys preserved during cache replay batching.
        svd_backend: Low-rank SVD backend, ``"full"`` or ``"svd_lowrank"``.
        svd_lowrank_oversample: Extra rank for ``torch.svd_lowrank``.
        svd_lowrank_niter: Power iterations for ``torch.svd_lowrank``.
    """

    artifact_cache = None
    if cache_dir is not None and cache_mode != "disabled":
        artifact_cache = QuantizationCacheSpec(cache_dir=Path(cache_dir) / precision / "artifacts", cache_mode=cache_mode)
    output_dir = None if cache_dir is None else Path(cache_dir) / precision / "inputs" / "samples"
    quantize_and_export(
        model=pipe.transformer,
        spec=svdquant_spec(
            precision,
            svd_backend=svd_backend,
            svd_lowrank_oversample=svd_lowrank_oversample,
            svd_lowrank_niter=svd_lowrank_niter,
        ),
        target_config=target_config,
        calibration=CalibrationSpec(
            samples=samples,
            num_samples=num_samples,
            batch_size=batch_size,
            cache_dir=None if cache_dir is None else Path(cache_dir) / precision / "inputs",
            cache_mode=cache_mode,
            seed=0,
            forward_fn=forward_fn,
            output_dir=output_dir,
            output_save_fn=save_diffusers_images,
            shared_input_keys=shared_input_keys,
            max_rows_per_target=4096,
            sample_batch_size=batch_size,
            element_batch_size=64,
            element_size=512,
            artifact_cache=artifact_cache,
        ),
        export=ExportSpec(output=Path(output)),
    )


def load_pipeline(pipeline_name: str, model_id: str, *, torch_dtype: torch.dtype, device: str):
    """Load a diffusers pipeline by class name.

    Args:
        pipeline_name: Name exported by the ``diffusers`` package.
        model_id: Hugging Face model id or local directory.
        torch_dtype: Dtype used for pipeline loading.
        device: Target device for the pipeline.

    Returns:
        Loaded and device-moved diffusers pipeline.
    """

    import diffusers

    pipeline_cls = getattr(diffusers, pipeline_name)
    pipe = pipeline_cls.from_pretrained(model_id, torch_dtype=torch_dtype)
    return pipe.to(device)


def pipeline_forward_fn(
    pipe,
    *,
    height: int,
    width: int,
    steps: int,
    guidance_scale: float,
    device: str,
) -> Callable[[dict], object]:
    """Create a calibration forward function for a diffusers pipeline.

    Args:
        pipe: Diffusers pipeline whose transformer is being quantized.
        height: Generated image height used while collecting calibration.
        width: Generated image width used while collecting calibration.
        steps: Number of denoising steps.
        guidance_scale: Classifier-free guidance scale.
        device: Device for deterministic generators.

    Returns:
        Callable accepted by ``CalibrationSpec.forward_fn``.
    """

    def forward(sample: dict) -> object:
        """Run one prompt sample through the pipeline.

        Args:
            sample: Calibration sample containing ``prompt`` and ``seed``.

        Returns:
            Pipeline output.
        """

        return pipe(
            prompt=sample["prompt"],
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=make_generator(sample.get("seed", 0), device=device),
        )

    return forward


def save_diffusers_images(result: object, sample: dict, output_dir: Path) -> None:
    """Save generated Diffusers images using calibration sample filenames."""

    images = getattr(result, "images", None)
    if images is None:
        raise ValueError("Diffusers calibration output must expose an images attribute")
    filenames = _as_list(sample.get("filename"))
    if not filenames:
        filenames = [f"{int(seed):04d}-0" for seed in _as_list(sample.get("seed"))]
    if len(filenames) != len(images):
        raise ValueError(f"Expected {len(filenames)} image filenames, got {len(images)} images")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, image in zip(filenames, images, strict=True):
        image.save(output_dir / f"{filename}.png")


def standard_prompt_records(
    num_samples: int,
    prompt_file: str | Path = UPSTREAM_QDIFF_PROMPT_SOURCE,
) -> list[PromptRecord]:
    """Return upstream qdiff calibration prompt records.

    Args:
        num_samples: Number of prompts to produce.
        prompt_file: Local qdiff YAML path or URL. By default this points to
            upstream DeepCompressor's qdiff prompt file at the pinned commit.

    Returns:
        Prompt records with upstream-compatible filenames, prompts, and seeds.
    """

    meta = _load_qdiff_prompts(prompt_file)
    names = list(meta)
    if num_samples > 0:
        random.Random(0).shuffle(names)
        names = sorted(names[:num_samples])
    records = []
    for name in names:
        filename = f"{name}-0"
        records.append(
            {
                "filename": filename,
                "prompt": meta[name],
                "seed": _hash_str_to_int(filename),
            }
        )
    return records


def standard_prompts(num_samples: int, prompt_file: str | Path = UPSTREAM_QDIFF_PROMPT_SOURCE) -> list[str]:
    """Return upstream qdiff calibration prompt strings."""

    return [str(record["prompt"]) for record in standard_prompt_records(num_samples, prompt_file=prompt_file)]


def batched_samples(prompts: list[str] | list[PromptRecord], batch_size: int) -> list[dict]:
    """Pack prompts and seeds into calibration sample dictionaries.

    Args:
        prompts: Prompt strings or upstream prompt records.
        batch_size: Number of prompts per sample.

    Returns:
        Calibration samples.
    """

    samples = []
    for start in range(0, len(prompts), batch_size):
        end = min(start + batch_size, len(prompts))
        batch = prompts[start:end]
        if batch and isinstance(batch[0], dict):
            filenames = [str(item["filename"]) for item in batch]  # type: ignore[index]
            prompt_batch = [str(item["prompt"]) for item in batch]  # type: ignore[index]
            seeds = [int(item["seed"]) for item in batch]  # type: ignore[index]
        else:
            filenames = [f"{index:04d}-0" for index in range(start, end)]
            prompt_batch = [str(item) for item in batch]
            seeds = list(range(start, end))
        samples.append(
            {
                "filename": filenames[0] if len(filenames) == 1 else filenames,
                "prompt": prompt_batch[0] if len(prompt_batch) == 1 else prompt_batch,
                "seed": seeds[0] if len(seeds) == 1 else seeds,
            }
        )
    return samples


def _as_list(value: object) -> list:
    """Return a scalar or sequence value as a list."""

    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _load_qdiff_prompts(prompt_file: str | Path) -> dict[str, str]:
    """Load upstream qdiff prompt YAML from a path or URL."""

    source = str(prompt_file)
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as response:
            text = response.read().decode("utf-8")
    else:
        text = Path(source).read_text(encoding="utf-8")
    return _parse_qdiff_prompt_yaml(text)


def _parse_qdiff_prompt_yaml(text: str) -> dict[str, str]:
    """Parse the simple key/value qdiff prompt YAML without adding PyYAML."""

    prompts: dict[str, str] = {}
    current_key: str | None = None
    current_value: list[str] = []
    entry_pattern = re.compile(r"^'?(?P<key>\d{4})'?:\s*(?P<value>.*)$")

    def flush() -> None:
        if current_key is not None:
            prompts[current_key] = _normalize_qdiff_value(" ".join(current_value))

    for line in text.splitlines():
        if not line.strip():
            continue
        match = entry_pattern.match(line)
        if match:
            flush()
            current_key = match.group("key")
            current_value = [match.group("value").strip()]
        elif current_key is not None and line[0].isspace():
            current_value.append(line.strip())
        else:
            raise ValueError(f"Unsupported qdiff prompt line: {line!r}")
    flush()
    return prompts


def _normalize_qdiff_value(value: str) -> str:
    """Normalize a qdiff YAML scalar."""

    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.replace("''", "'")


def _hash_str_to_int(value: str) -> int:
    """Hash a string the same way upstream DeepCompressor seeds samples."""

    modulus = 10**9 + 7
    hash_int = 0
    for char in value:
        hash_int = (hash_int * 31 + ord(char)) % modulus
    return hash_int


def make_generator(seed: int | list[int], device: str = "cuda"):
    """Create one or more deterministic torch generators.

    Args:
        seed: Seed integer or list of seeds.
        device: Generator device.

    Returns:
        Torch generator or list of generators.
    """

    if isinstance(seed, list):
        return [torch.Generator(device=device).manual_seed(int(item)) for item in seed]
    return torch.Generator(device=device).manual_seed(int(seed))


def _resolve_torch_dtype(name: str) -> torch.dtype:
    """Resolve a CLI dtype name to ``torch.dtype``.

    Args:
        name: CLI dtype value.

    Returns:
        Torch dtype object.
    """

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _low_rank_solver(
    *,
    svd_backend: SvdBackend = "full",
    svd_lowrank_oversample: int = 10,
    svd_lowrank_niter: int = 4,
) -> LowRankSolverSpec:
    """Return the upstream-style low-rank search spec."""

    return LowRankSolverSpec(
        mode="search",
        num_iters=100,
        early_stop=True,
        degree=2,
        eval_replay=True,
        svd_backend=svd_backend,
        svd_lowrank_oversample=svd_lowrank_oversample,
        svd_lowrank_niter=svd_lowrank_niter,
    )


def _smooth_spec() -> SmoothSpec:
    """Return the upstream-style projection smoothing spec."""

    return SmoothSpec(
        enabled=True,
        strategy="grid_search",
        objective="outputs_error",
        alpha=0.5,
        beta=-2,
        num_grids=20,
        spans=(("absmax", "absmax"),),
    )


def _flux_extra_weight_targets() -> list[TargetRule]:
    """Return NVFP4 Flux extra INT4 target rules."""

    return [
        # NVFP4 extra-weight rule for double-block hidden-state norm modulation.
        TargetRule(
            modules=["transformer_blocks.*.norm1.linear"],
            shared_low_rank=False,
            precision="int4",
            group_size=64,
            rank=0,
            smooth=False,
            activation_quant=False,
            shift_activations=False,
        ),
        # NVFP4 extra-weight rule for double-block context norm modulation.
        TargetRule(
            modules=["transformer_blocks.*.norm1_context.linear"],
            shared_low_rank=False,
            precision="int4",
            group_size=64,
            rank=0,
            smooth=False,
            activation_quant=False,
            shift_activations=False,
        ),
        # NVFP4 extra-weight rule for single-block norm modulation.
        TargetRule(
            modules=["single_transformer_blocks.*.norm.linear"],
            shared_low_rank=False,
            precision="int4",
            group_size=64,
            rank=0,
            smooth=False,
            activation_quant=False,
            shift_activations=False,
        ),
    ]
