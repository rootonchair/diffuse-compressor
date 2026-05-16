import importlib.util
import json

import pytest
import safetensors

from diffuse_compressor import DiffusionQuantSpec, ExportSpec, collect_quant_targets, prepare_model, quantize_and_export
from examples.upstream_diffusion_svdquant import (
    flux1_target_config,
    pixart_sigma_target_config,
    sana_target_config,
)


pytestmark = pytest.mark.skipif(importlib.util.find_spec("diffusers") is None, reason="diffusers is not installed")


def test_flux1_upstream_target_config_matches_tiny_flux_nvfp4():
    from diffusers import FluxTransformer2DModel

    model = FluxTransformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=64,
        pooled_projection_dim=64,
        guidance_embeds=True,
        axes_dims_rope=(8, 8),
    )
    target_config = flux1_target_config("nvfp4")
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert "transformer_blocks.0.qkv_proj" in export_names
    assert "transformer_blocks.0.qkv_proj_context" in export_names
    assert "single_transformer_blocks.0.out_proj" in export_names
    assert "transformer_blocks.0.norm1.linear" in export_names
    assert "single_transformer_blocks.0.norm.linear" in export_names


def test_pixart_sigma_upstream_target_config_exports_int4(tmp_path):
    from diffusers import PixArtTransformer2DModel

    model = PixArtTransformer2DModel(
        num_attention_heads=2,
        attention_head_dim=32,
        in_channels=4,
        out_channels=8,
        num_layers=1,
        norm_num_groups=4,
        cross_attention_dim=64,
        sample_size=8,
        patch_size=2,
        caption_channels=64,
    )
    output = tmp_path / "pixart.safetensors"

    quantize_and_export(
        model,
        DiffusionQuantSpec(rank=4, group_size=64, smooth=False),
        pixart_sigma_target_config("int4"),
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = json.loads(handle.metadata()["quantization_config"])

    assert "transformer_blocks.0.attn1.qkv_proj.qweight" in keys
    assert "transformer_blocks.0.attn2.kv_proj.qweight" in keys
    assert "transformer_blocks.0.mlp_fc1.qweight" in keys
    assert metadata["rank"] == 4


def test_sana_upstream_target_config_exports_pointwise_conv_nvfp4(tmp_path):
    from diffusers import SanaTransformer2DModel

    model = SanaTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_attention_heads=2,
        attention_head_dim=32,
        num_layers=1,
        num_cross_attention_heads=2,
        cross_attention_head_dim=32,
        cross_attention_dim=64,
        caption_channels=64,
        sample_size=8,
        patch_size=1,
        mlp_ratio=2.0,
    )
    target_config = sana_target_config("nvfp4")
    targets = collect_quant_targets(model, target_config)
    output = tmp_path / "sana.safetensors"

    assert {target.kind for target in targets if target.export_name.endswith(("mlp_fc1", "mlp_fc2"))} == {"conv"}

    quantize_and_export(
        model,
        DiffusionQuantSpec(precision="fp4", rank=4, group_size=16, smooth=False, weight_scale_dtypes=(None, "sfp8_e4m3_nan")),
        target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = json.loads(handle.metadata()["quantization_config"])

    assert "transformer_blocks.0.mlp_fc1.qweight" in keys
    assert "transformer_blocks.0.mlp_fc2.qweight" in keys
    assert metadata["weight"]["dtype"] == "fp4_e2m1_all"
    assert metadata["weight"]["scale_dtypes"] == [None, "sfp8_e4m3_nan"]
