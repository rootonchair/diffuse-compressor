"""Dedicated data-free scripts: FLUX.2 Klein 4B/9B and LTX-2.3 (+ Distilled)."""

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from torch import nn


def _load_script(monkeypatch, folder: str, name: str) -> ModuleType:
    """Import ``examples/<folder>/data_free/<name>.py`` as a package module."""

    return importlib.import_module(f"examples.{folder}.data_free.{name}")


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128)


def _fake_run(monkeypatch, namespace: dict, loader_name, tmp_path):
    """Stub the loader and export in ``namespace`` (a module's globals)."""

    calls = {}
    model = _Tiny()

    def fake_load(model_id, *, device, dtype):
        calls["load"] = (model_id, device, dtype)
        return model

    def fake_quantize_and_export(*, model, spec, target_config, calibration, export, logging):
        calls.update(model=model, spec=spec, target_config=target_config, calibration=calibration, output=export.output)

    monkeypatch.setitem(namespace, loader_name, fake_load)
    monkeypatch.setitem(namespace, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.chdir(tmp_path)
    return calls, model


# --- FLUX.2 Klein -------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, model_id, sizes, model_name",
    [
        ("quantize_flux2_klein_4b", "black-forest-labs/FLUX.2-klein-4B", (9216, 3072), "flux2-klein-4b"),
        ("quantize_flux2_klein_9b", "black-forest-labs/FLUX.2-klein-9B", (12288, 4096), "flux2-klein-9b"),
    ],
)
def test_klein_scripts_load_the_transformer_and_export_the_klein_layout(
    monkeypatch, tmp_path, name, model_id, sizes, model_name
):
    import torch

    klein = _load_script(monkeypatch, "text_to_image", name)
    calls, model = _fake_run(monkeypatch, vars(klein), "load_flux2_transformer", tmp_path)
    seen = {}

    def fake_config(precision, *, single_qkv_features, single_attn_features):
        seen.update(precision=precision, sizes=(single_qkv_features, single_attn_features))
        return "klein-config"

    monkeypatch.setattr(klein, "flux2_klein_target_config", fake_config)
    klein.run_model_cli(["--device", "cuda", "--offload-model", "--hadamard", "--no-run-log"])

    assert calls["load"] == (model_id, "cpu", torch.bfloat16)
    assert seen == {"precision": "int4", "sizes": sizes}
    assert calls["target_config"] == "klein-config" and calls["model"] is model
    spec = calls["spec"]
    assert spec.rotation is not None and not spec.smooth.enabled
    assert spec.low_rank_solver.mode == "weighted_svd"
    assert spec.offload_model and spec.compute_device == "cuda"
    assert calls["output"] == Path(f"outputs/checkpoints/svdq-int4_r32-{model_name}-datafree-hadamard.safetensors")
    cache = calls["calibration"].artifact_cache
    assert Path(cache.cache_dir) == Path(f"outputs/calibration/{model_name}/int4/artifacts-datafree")


@pytest.mark.parametrize("name", ["quantize_flux2_klein_4b", "quantize_flux2_klein_9b"])
@pytest.mark.parametrize(
    "argv, message",
    [
        (["--precision", "nvfp4", "--hadamard"], "int4 only"),
        (["--smooth-exponent", "2"], "--smooth-exponent must be in [0, 1]"),
        (["--num-samples", "4"], "unrecognized arguments"),
        (["--synthetic-calibration"], "unrecognized arguments"),
        (["--rank", "24"], "--rank must be a positive multiple of 16"),
        (["--rank", "0"], "--rank must be a positive multiple of 16"),
        (["org/a", "--model-id", "org/b"], "not both"),
    ],
)
def test_klein_scripts_reject_bad_flags_before_loading(monkeypatch, capsys, name, argv, message):
    klein = _load_script(monkeypatch, "text_to_image", name)
    monkeypatch.setattr(klein, "load_flux2_transformer", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(SystemExit):
        klein.run_model_cli(argv)
    assert message in capsys.readouterr().err


# --- LTX-2.3 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, model_id, model_name",
    [
        ("quantize_ltx2_3", "dg845/LTX-2.3-Diffusers", "ltx2.3"),
        ("quantize_ltx2_3_distilled", "dg845/LTX-2.3-Distilled-Diffusers", "ltx2.3-distilled"),
    ],
)
def test_ltx_scripts_validate_after_inspect_and_export_the_ltx_layout(monkeypatch, tmp_path, capsys, name, model_id, model_name):
    script = _load_script(monkeypatch, "text_to_video", name)
    # The distilled script delegates to data_free/quantize_ltx2_3.py; patch the module that runs.
    runner = vars(importlib.import_module("examples.text_to_video.data_free.quantize_ltx2_3"))
    calls, model = _fake_run(monkeypatch, runner, "load_ltx2_transformer", tmp_path)
    validated = []
    monkeypatch.setitem(runner, "ltx2_3_target_config", lambda precision, fuse_qkv: ("ltx-config", precision, fuse_qkv))
    monkeypatch.setitem(runner, "validate_ltx2_3_nunchaku_lite_manifest_targets", lambda m, c: validated.append(c))
    monkeypatch.setitem(
        runner, "inspect_target_config", lambda m, c: SimpleNamespace(format_text=lambda: "inspected")
    )

    script.run_model_cli(["--device", "cpu", "--fuse-qkv", "--inspect-config"])
    assert "inspected" in capsys.readouterr().out
    assert validated == [] and "spec" not in calls

    script.run_model_cli(["--device", "cpu", "--precision", "nvfp4", "--fuse-qkv", "--cache-mode", "disabled", "--no-run-log"])
    assert calls["load"][:2] == (model_id, "cpu")
    assert validated == [("ltx-config", "nvfp4", True)]
    assert calls["target_config"] == ("ltx-config", "nvfp4", True)
    assert calls["calibration"] is None
    assert calls["spec"].smooth.enabled and calls["spec"].smooth.beta == 0.5
    assert calls["output"] == Path(f"outputs/checkpoints/svdq-nvfp4_r32-{model_name}-datafree.safetensors")


def test_ltx_data_free_cache_folder(monkeypatch, tmp_path):
    ltx = _load_script(monkeypatch, "text_to_video", "quantize_ltx2_3")
    calls, _ = _fake_run(monkeypatch, vars(ltx), "load_ltx2_transformer", tmp_path)
    monkeypatch.setattr(ltx, "ltx2_3_target_config", lambda precision, fuse_qkv: "ltx-config")
    monkeypatch.setattr(ltx, "validate_ltx2_3_nunchaku_lite_manifest_targets", lambda m, c: None)
    ltx.run_model_cli(["--device", "cpu", "--no-run-log"])
    cache = calls["calibration"].artifact_cache
    assert Path(cache.cache_dir) == Path("outputs/calibration/ltx2.3/int4/artifacts-datafree")


# --- real tiny diffusers models -----------------------------------------------------


def test_klein_script_quantizes_a_real_flux2_model(monkeypatch, tmp_path):
    """The real FLUX.2 Klein layout on a tiny real Flux2 transformer (needs diffusers)."""

    diffusers = pytest.importorskip("diffusers")
    from safetensors import safe_open

    model = diffusers.Flux2Transformer2DModel(
        in_channels=64,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=64,
        num_attention_heads=2,
        joint_attention_dim=128,
        timestep_guidance_channels=64,
        axes_dims_rope=(16, 16, 16, 16),
        guidance_embeds=False,
    )
    klein = _load_script(monkeypatch, "text_to_image", "quantize_flux2_klein_4b")
    # The script's sizes are for the real 4B model; this tiny model has inner dim 128.
    real_config = klein.flux2_klein_target_config
    monkeypatch.setattr(
        klein,
        "flux2_klein_target_config",
        lambda precision, **_: real_config(precision, single_qkv_features=384, single_attn_features=128),
    )
    monkeypatch.setattr(klein, "load_flux2_transformer", lambda model_id, *, device, dtype: model.to(dtype))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "klein.safetensors"
    klein.run_model_cli(["--device", "cpu", "--no-run-log", "--svd-backend", "full", "--rank", "16", "--output", str(output)])
    with safe_open(str(output), framework="pt") as handle:
        keys = set(handle.keys())
    assert any(key.startswith("single_transformer_blocks.0.attn.qkv_proj") for key in keys)
    assert any(key.startswith("transformer_blocks.0.attn.to_qkv") for key in keys)


def test_ltx_script_quantizes_a_real_ltx2_model(monkeypatch, tmp_path):
    """The real LTX-2.3 layout and manifest check on a tiny real LTX2 transformer (needs diffusers)."""

    diffusers = pytest.importorskip("diffusers")
    from safetensors import safe_open

    model = diffusers.LTX2VideoTransformer3DModel(
        in_channels=64,
        out_channels=64,
        num_attention_heads=2,
        attention_head_dim=64,
        cross_attention_dim=128,
        gated_attn=True,  # LTX-2.3 attention gate logits must stay dense/AWQ
        audio_in_channels=64,
        audio_out_channels=64,
        audio_num_attention_heads=2,
        audio_attention_head_dim=64,
        audio_cross_attention_dim=128,
        audio_gated_attn=True,
        num_layers=1,
        caption_channels=128,
        use_prompt_embeddings=False,
    )
    ltx = _load_script(monkeypatch, "text_to_video", "quantize_ltx2_3")
    monkeypatch.setattr(ltx, "load_ltx2_transformer", lambda model_id, *, device, dtype: model.to(dtype))
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "ltx.safetensors"
    ltx.run_model_cli(["--device", "cpu", "--no-run-log", "--svd-backend", "full", "--rank", "16", "--output", str(output)])
    with safe_open(str(output), framework="pt") as handle:
        keys = set(handle.keys())
    assert any(key.startswith("transformer_blocks.0") and key.endswith("proj_down") for key in keys)


@pytest.mark.parametrize("argv", [["--model-id", "org/custom"], ["org/custom"]])
def test_model_id_works_as_flag_or_positional(monkeypatch, tmp_path, argv):
    klein = _load_script(monkeypatch, "text_to_image", "quantize_flux2_klein_4b")
    calls, _ = _fake_run(monkeypatch, vars(klein), "load_flux2_transformer", tmp_path)
    monkeypatch.setattr(klein, "flux2_klein_target_config", lambda *a, **k: "klein-config")
    klein.run_model_cli([*argv, "--device", "cpu", "--no-run-log"])
    assert calls["load"][0] == "org/custom"
