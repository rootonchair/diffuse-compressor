import importlib.util
import json

import pytest
import safetensors
import torch

from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    DiffusionQuantSpec,
    ExportSpec,
    collect_quant_targets,
    prepare_model,
    quantize_and_export,
)
from examples.upstream_diffusion_svdquant import (
    MODEL_DEFAULTS,
    flux1_target_config,
    flux2_klein_4b_target_config,
    flux2_klein_9b_target_config,
    flux2_klein_target_config,
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
    assert target_config.calibration_scopes[0].module_classes == (type(model.transformer_blocks[0]),)
    assert target_config.calibration_scopes[1].module_classes == (type(model.single_transformer_blocks[0]),)
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert "transformer_blocks.0.qkv_proj" in export_names
    assert "transformer_blocks.0.qkv_proj_context" in export_names
    assert "single_transformer_blocks.0.out_proj" in export_names
    assert "transformer_blocks.0.norm1.linear" in export_names
    assert "single_transformer_blocks.0.norm.linear" in export_names
    out_proj = next(target for target in targets if target.export_name == "single_transformer_blocks.0.out_proj")
    assert out_proj.export_bias == "zero"

    extra_names = {
        "transformer_blocks.0.norm1.linear",
        "transformer_blocks.0.norm1_context.linear",
        "single_transformer_blocks.0.norm.linear",
    }
    for target in targets:
        if target.export_name not in extra_names:
            continue
        assert target.precision == "int4"
        assert target.group_size == 64
        assert target.rank == 0
        assert target.shared_low_rank is False
        assert target.smooth is False
        assert target.activation_quant is False
        assert target.shift_activations is False
        assert isinstance(target.weight_layout, AdaNormAwqW4A16Layout)
        assert target.weight_layout.splits == (3 if target.export_name.startswith("single_") else 6)


def test_flux2_klein_upstream_target_config_exports_nunchaku_lite_keys():
    from diffusers import Flux2Transformer2DModel

    model = Flux2Transformer2DModel(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=32,
        guidance_embeds=False,
        axes_dims_rope=(4, 4, 4, 4),
        timestep_guidance_channels=32,
    )
    target_config = flux2_klein_target_config(single_qkv_features=96, single_attn_features=32)
    assert target_config.calibration_scopes[0].module_classes == (type(model.transformer_blocks[0]),)
    assert target_config.calibration_scopes[1].module_classes == (type(model.single_transformer_blocks[0]),)
    prepare_model(model, target_config.patches)
    targets = collect_quant_targets(model, target_config)
    export_names = {target.export_name for target in targets}

    assert export_names == {
        "transformer_blocks.0.attn.to_qkv",
        "transformer_blocks.0.attn.to_added_qkv",
        "transformer_blocks.0.attn.to_out.0",
        "transformer_blocks.0.attn.to_add_out",
        "transformer_blocks.0.ff.linear_in",
        "transformer_blocks.0.ff.linear_out",
        "transformer_blocks.0.ff_context.linear_in",
        "transformer_blocks.0.ff_context.linear_out",
        "single_transformer_blocks.0.attn.qkv_proj",
        "single_transformer_blocks.0.attn.mlp_fc1",
        "single_transformer_blocks.0.attn.out_proj",
        "single_transformer_blocks.0.attn.mlp_fc2",
    }


def test_flux2_klein_model_variants_use_expected_split_sizes():
    config_4b = flux2_klein_4b_target_config()
    config_9b = flux2_klein_9b_target_config()

    assert MODEL_DEFAULTS["flux.2-klein-4b"].model_id == "black-forest-labs/FLUX.2-klein-4B"
    assert MODEL_DEFAULTS["flux.2-klein-9b"].model_id == "black-forest-labs/FLUX.2-klein-9B"
    assert config_4b.patches[0].args["splits"] == [9216]
    assert config_4b.patches[1].args["splits"] == [3072]
    assert config_9b.patches[0].args["splits"] == [12288]
    assert config_9b.patches[1].args["splits"] == [4096]


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
    nvfp4_config = pixart_sigma_target_config("nvfp4")
    assert nvfp4_config.calibration_scopes[0].module_classes == (type(model.transformer_blocks[0]),)
    nvfp4_targets = collect_quant_targets(model, nvfp4_config)
    adaln_target = next(target for target in nvfp4_targets if target.export_name == "adaln_single.linear")

    assert adaln_target.precision == "int4"
    assert adaln_target.group_size == 64
    assert adaln_target.rank == 0
    assert adaln_target.shared_low_rank is False
    assert adaln_target.smooth is False
    assert adaln_target.activation_quant is False
    assert adaln_target.shift_activations is False

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


@pytest.mark.skipif(importlib.util.find_spec("nunchaku_lite") is None, reason="nunchaku_lite is not installed")
def test_flux1_nvfp4_upstream_checkpoint_strict_loads_with_nunchaku_lite(tmp_path):
    from diffusers import FluxTransformer2DModel
    from nunchaku_lite import patch_transformer

    kwargs = dict(
        in_channels=16,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=32,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=64,
        guidance_embeds=False,
        axes_dims_rope=(8, 8),
    )
    source = FluxTransformer2DModel(**kwargs).to(torch.bfloat16)
    output = tmp_path / "flux1-nvfp4-lite-loadable.safetensors"
    target_config = flux1_target_config("nvfp4")

    quantize_and_export(
        source,
        DiffusionQuantSpec(
            precision="fp4",
            rank=4,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
        ),
        target_config,
        calibration=None,
        export=ExportSpec(output=output),
    )

    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        metadata = json.loads(handle.metadata()["quantization_config"])

    assert "transformer_blocks.0.norm1.linear.wzeros" in keys
    assert "transformer_blocks.0.norm1.linear.smooth_factor" not in keys
    norm_target = next(target for target in metadata["targets"] if target["export_name"] == "transformer_blocks.0.norm1.linear")
    assert norm_target["weight_layout"] == {"name": "adanorm_awq_w4a16", "splits": 6}

    target = FluxTransformer2DModel(**kwargs)
    patch_transformer(target, output, target="flux", precision="fp4", torch_dtype=torch.bfloat16)

    assert target._nunchaku_lite_patched


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
    assert target_config.calibration_scopes[0].module_classes == (type(model.transformer_blocks[0]),)
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
