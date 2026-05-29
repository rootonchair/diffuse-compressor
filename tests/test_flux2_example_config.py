import importlib.util

import pytest
import safetensors
import torch

from diffuse_compressor import DiffusionQuantSpec, ExportSpec, collect_quant_targets, prepare_model, quantize_and_export
from examples.text_to_image.quantize_flux2_klein_4b import flux2_klein_target_config


@pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="diffusers is not installed")
def test_flux2_example_config_exports_nunchaku_lite_keys(tmp_path):
    from diffusers import Flux2Transformer2DModel

    transformer = Flux2Transformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 4, 4, 4),
        timestep_guidance_channels=32,
    ).to(torch.bfloat16)
    target_config = flux2_klein_target_config(single_qkv_features=96, single_attn_features=32, use_nunchaku_layout=False)
    output = tmp_path / "flux2-lite.safetensors"

    quantize_and_export(
        transformer,
        DiffusionQuantSpec(rank=4, group_size=32),
        target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())

    assert "transformer_blocks.0.attn.to_qkv.qweight" in keys
    assert "transformer_blocks.0.attn.to_added_qkv.qweight" in keys
    assert "single_transformer_blocks.0.attn.qkv_proj.qweight" in keys
    assert "single_transformer_blocks.0.attn.mlp_fc1.qweight" in keys
    assert "single_transformer_blocks.0.attn.out_proj.qweight" in keys
    assert "single_transformer_blocks.0.attn.mlp_fc2.qweight" in keys


@pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="diffusers is not installed")
def test_flux2_example_config_resolves_targets_after_patching():
    from diffusers import Flux2Transformer2DModel

    transformer = Flux2Transformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 4, 4, 4),
        timestep_guidance_channels=32,
    )
    target_config = flux2_klein_target_config(single_qkv_features=96, single_attn_features=32)

    prepare_model(transformer, target_config.patches)
    targets = collect_quant_targets(transformer, target_config)

    assert {target.export_name for target in targets} >= {
        "single_transformer_blocks.0.attn.qkv_proj",
        "single_transformer_blocks.0.attn.mlp_fc1",
        "single_transformer_blocks.0.attn.out_proj",
        "single_transformer_blocks.0.attn.mlp_fc2",
    }


@pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="diffusers is not installed")
@pytest.mark.skipif(importlib.util.find_spec("nunchaku_lite") is None, reason="nunchaku_lite is not installed")
def test_flux2_example_checkpoint_strict_loads_with_nunchaku_lite(tmp_path):
    from diffusers import Flux2Transformer2DModel
    from nunchaku_lite import patch_transformer

    kwargs = dict(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=64,
        guidance_embeds=False,
        axes_dims_rope=(8, 8, 8, 8),
        timestep_guidance_channels=64,
    )
    source = Flux2Transformer2DModel(**kwargs).to(torch.bfloat16)
    output = tmp_path / "flux2-lite-loadable.safetensors"
    target_config = flux2_klein_target_config(single_qkv_features=192, single_attn_features=64, use_nunchaku_layout=False)
    quantize_and_export(
        source,
        DiffusionQuantSpec(rank=4, group_size=64),
        target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )

    target = Flux2Transformer2DModel(**kwargs)
    patch_transformer(target, output, target="flux2", precision="int4")

    assert target._nunchaku_lite_patched
