"""Device placement, cache separation and in-place fake-quant for the data-free path."""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

import examples.text_to_image.utils as image_utils
import examples.text_to_video.utils as video_utils
from diffuse_compressor.backends.nunchaku.packing import dynamic_fake_quantize_activation

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_example(relative: str, name: str, utils_module, extra_modules=None):
    """Import an example script with its bare ``utils`` (and friends) bound explicitly."""

    bound = {"utils": utils_module, **(extra_modules or {})}
    previous = {key: sys.modules.get(key) for key in bound}
    sys.modules.update(bound)
    try:
        spec = importlib.util.spec_from_file_location(name, EXAMPLES / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for key, value in previous.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return module


# --- in-place activation fake-quant -------------------------------------------------


@pytest.mark.parametrize("float_point,group_size", [(True, 16), (False, 64)])
def test_inplace_fake_quant_matches_out_of_place(float_point, group_size):
    torch.manual_seed(0)
    values = torch.randn(37, 4 * group_size) * 3
    expected = dynamic_fake_quantize_activation(values, group_size=group_size, float_point=float_point)
    actual = dynamic_fake_quantize_activation(values.clone(), group_size=group_size, float_point=float_point, inplace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("float_point,group_size", [(True, 16), (False, 64)])
def test_inplace_fake_quant_handles_strided_input(float_point, group_size):
    torch.manual_seed(0)
    big = torch.randn(64, 2 * group_size) * 3
    strided = big[::2]
    assert not strided.is_contiguous()
    expected = dynamic_fake_quantize_activation(strided.contiguous(), group_size=group_size, float_point=float_point)
    actual = dynamic_fake_quantize_activation(strided, group_size=group_size, float_point=float_point, inplace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_inplace_fp4_fake_quant_crosses_chunk_boundary():
    torch.manual_seed(0)
    values = torch.randn(8_000_000 // 64 + 3, 64)  # just over one 8M-element chunk
    values[0, 0] = float("inf")
    values[5, 7] = float("nan")
    expected = dynamic_fake_quantize_activation(values, group_size=16, float_point=True)
    actual = dynamic_fake_quantize_activation(values.clone(), group_size=16, float_point=True, inplace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


# --- generic image/video scripts ----------------------------------------------------


def test_generic_scripts_keep_data_free_cache_separate():
    from examples.text_to_image.quantize_hf import _artifact_dir as image_dir
    from examples.text_to_video.quantize_hf import _artifact_dir as video_dir

    for artifact_dir in (image_dir, video_dir):
        assert artifact_dir(False) == "artifacts"
        assert artifact_dir(True) == "artifacts-datafree"


# --- LTX-2.3 ------------------------------------------------------------------------


def _run_ltx(monkeypatch, tmp_path, *flags):
    ltx = _load_example("text_to_video/quantize_ltx2_3.py", "_ltx2_3_wiring_example", video_utils)
    calls = {}
    transformer = nn.Linear(64, 64)

    def fake_load_transformer(model_id, *, device, dtype):
        calls["device"] = device
        return transformer

    def fake_quantize_and_export(*, model, spec, target_config, calibration, export, logging):
        calls.update(spec=spec, calibration=calibration)

    monkeypatch.setattr(ltx, "_ltx2_default_negative_prompt", lambda: "")
    monkeypatch.setattr(ltx, "load_ltx2_pipeline", lambda *a, **k: pytest.fail("pipeline loaded"))
    monkeypatch.setattr(ltx, "load_ltx2_transformer", fake_load_transformer)
    monkeypatch.setattr(ltx, "ltx2_3_target_config", lambda precision, fuse_qkv=False: object())
    monkeypatch.setattr(ltx, "validate_ltx2_3_nunchaku_lite_manifest_targets", lambda *args: None)
    monkeypatch.setattr(ltx, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(sys, "argv", ["quantize_ltx2_3.py", "--data-free", "--no-run-log", *flags])
    ltx.run_ltx2_3_cli(
        model_id="org/ltx",
        output_template=str(tmp_path / "svdq-{precision}_r32-ltx.safetensors"),
        default_cache_dir=str(tmp_path / "cache"),
        steps=8,
        guidance_scale=1.0,
        batch_size=1,
        height=512,
        width=768,
        num_frames=121,
        frame_rate=24.0,
    )
    return calls


def test_ltx_data_free_pipeline_offload_keeps_transformer_on_cpu(monkeypatch, tmp_path):
    calls = _run_ltx(monkeypatch, tmp_path, "--device", "cuda", "--pipeline-offload", "sequential")
    assert calls["device"] == "cpu"
    assert calls["spec"].offload_model
    assert calls["spec"].compute_device == "cuda"


def test_ltx_data_free_uses_its_own_artifact_cache(monkeypatch, tmp_path):
    calls = _run_ltx(monkeypatch, tmp_path, "--device", "cpu")
    cache = calls["calibration"].artifact_cache
    assert Path(cache.cache_dir) == tmp_path / "cache" / "int4" / "artifacts-datafree"


# --- FLUX.2 klein-9B ----------------------------------------------------------------


def _run_klein9b(monkeypatch, tmp_path, *flags):
    calls = {}
    fake_quantize_hf = types.ModuleType("quantize_hf")

    def fake_load_denoiser_only(model_id, *, device):
        calls["device"] = device
        return nn.Linear(64, 64)

    fake_quantize_hf.load_denoiser_only = fake_load_denoiser_only
    klein = _load_example("text_to_image/quantize_flux2_klein_9b.py", "_klein9b_wiring_example", image_utils)

    def fake_quantize_and_export(*, model, spec, target_config, calibration, export, logging):
        calls.update(spec=spec, calibration=calibration, output=export.output)

    monkeypatch.setitem(sys.modules, "quantize_hf", fake_quantize_hf)
    monkeypatch.setattr(klein, "load_pipeline", lambda *a, **k: pytest.fail("pipeline loaded"))
    monkeypatch.setattr(klein, "flux2_klein_target_config", lambda *a, **k: object())
    monkeypatch.setattr(klein, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(
        sys,
        "argv",
        ["quantize_flux2_klein_9b.py", "--data-free", "--no-run-log", "--cache-dir", str(tmp_path), *flags],
    )
    klein.run_model_cli()
    return calls


def test_klein9b_data_free_offload_model_loads_on_cpu(monkeypatch, tmp_path):
    calls = _run_klein9b(monkeypatch, tmp_path, "--device", "cuda", "--offload-model", "--compute-device", "cuda")
    assert calls["device"] == "cpu"
    assert calls["spec"].offload_model and calls["spec"].compute_device == "cuda"
    assert Path(calls["calibration"].artifact_cache.cache_dir) == tmp_path / "int4" / "artifacts-datafree"
    assert calls["output"].name == "svdq-datafree-int4_r32-flux2-klein-9b.safetensors"


def test_klein9b_data_free_pipeline_offload_implies_cpu_loading(monkeypatch, tmp_path):
    calls = _run_klein9b(monkeypatch, tmp_path, "--device", "cuda", "--pipeline-offload", "model")
    assert calls["device"] == "cpu"
    assert calls["spec"].offload_model and calls["spec"].compute_device == "cuda"


def test_klein9b_data_free_default_stays_on_device(monkeypatch, tmp_path):
    calls = _run_klein9b(monkeypatch, tmp_path, "--device", "cuda")
    assert calls["device"] == "cuda"
    assert not calls["spec"].offload_model
