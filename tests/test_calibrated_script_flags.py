"""Calibrated model scripts: data-free flags are gone, --hadamard and klein-9B synthetic calibration remain."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from torch import nn

import examples.text_to_image.calibrated.quantize_flux2_klein_9b as klein9b_module
import examples.text_to_video.calibrated.quantize_ltx2_3 as ltx_module
from examples.text_to_video.calibrated.quantize_hf import build_parser as build_video_parser

LTX_KWARGS = dict(
    model_id="org/ltx",
    output_template="outputs/checkpoints/svdq-{precision}_r32-ltx.safetensors",
    default_cache_dir="outputs/calibration/ltx",
    steps=8,
    guidance_scale=1.0,
    batch_size=1,
    height=512,
    width=768,
    num_frames=121,
    frame_rate=24.0,
)


@pytest.fixture
def ltx(monkeypatch):
    module = ltx_module
    monkeypatch.setattr(module, "_ltx2_default_negative_prompt", lambda: "")  # needs diffusers otherwise
    monkeypatch.setattr(module, "load_ltx2_pipeline", lambda *a, **k: pytest.fail("pipeline loaded"))
    return module


@pytest.fixture
def klein9b():
    return klein9b_module


@pytest.mark.parametrize("flag", ["--data-free", "--smooth-exponent=0.5"])
def test_generic_video_parser_rejects_data_free_flags(flag):
    with pytest.raises(SystemExit):
        build_video_parser().parse_args(["model/id", flag])
    assert build_video_parser().parse_args(["model/id", "--hadamard"]).hadamard


@pytest.mark.parametrize("flag", ["--data-free", "--smooth-exponent=0.5"])
def test_ltx_rejects_data_free_flags(monkeypatch, ltx, flag):
    monkeypatch.setattr(sys, "argv", ["quantize_ltx2_3.py", flag])
    with pytest.raises(SystemExit):
        ltx.run_ltx2_3_cli(**LTX_KWARGS)


def test_ltx_rejects_nvfp4_hadamard_before_loading_the_pipeline(monkeypatch, ltx):
    monkeypatch.setattr(sys, "argv", ["quantize_ltx2_3.py", "--precision", "nvfp4", "--hadamard", "--device", "cpu"])
    with pytest.raises(ValueError, match="INT4"):
        ltx.run_ltx2_3_cli(**LTX_KWARGS)


@pytest.mark.parametrize("flag", ["--data-free", "--smooth-exponent=0.5"])
def test_klein9b_rejects_data_free_flags(monkeypatch, klein9b, flag):
    monkeypatch.setattr(sys, "argv", ["quantize_flux2_klein_9b.py", flag])
    with pytest.raises(SystemExit):
        klein9b.run_model_cli()


def test_klein9b_synthetic_calibration_runs_without_data_free(monkeypatch, tmp_path, klein9b):
    calls = {}
    transformer = nn.Linear(128, 128)
    transformer.config = SimpleNamespace()

    class FakeFlux2Transformer:
        @staticmethod
        def from_pretrained(model_id, *, subfolder, torch_dtype):
            calls["load"] = (model_id, subfolder)
            return SimpleNamespace(to=lambda device: calls.setdefault("device", device) and transformer)

    fake_diffusers = ModuleType("diffusers")
    fake_diffusers.Flux2Transformer2DModel = FakeFlux2Transformer
    fake_synthetic = ModuleType("examples.text_to_image.calibrated.flux2_synthetic")
    fake_synthetic.synthetic_prompt_embeds = lambda *a, **k: "embeds"
    fake_synthetic.synthetic_denoise_forward_fn = lambda model, **k: calls.setdefault("forward_model", model)

    def fake_quantize_and_export(*, model, spec, target_config, calibration, export, logging):
        calls.update(model=model, spec=spec, calibration=calibration, output=export.output)

    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setitem(sys.modules, "examples.text_to_image.calibrated.flux2_synthetic", fake_synthetic)
    monkeypatch.setattr(klein9b, "load_pipeline", lambda *a, **k: pytest.fail("pipeline loaded"))
    monkeypatch.setattr(klein9b, "flux2_klein_target_config", lambda *a, **k: object())
    monkeypatch.setattr(klein9b, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize_flux2_klein_9b.py",
            "--synthetic-calibration",
            "--device",
            "cpu",
            "--cache-dir",
            str(tmp_path),
            "--no-run-log",
        ],
    )
    klein9b.run_model_cli()

    assert calls["load"] == ("black-forest-labs/FLUX.2-klein-9B", "transformer")
    assert calls["device"] == "cpu"
    assert calls["model"] is transformer and calls["forward_model"] is transformer
    assert calls["output"].name == "svdq-synthcal-gaussian-int4_r32-flux2-klein-9b.safetensors"
    cache = calls["calibration"].artifact_cache
    assert Path(cache.cache_dir) == tmp_path / "int4" / "artifacts-synthetic-gaussian"
    # Synthetic calibration uses the calibrated smoothing search, not data-free smoothing.
    assert calls["spec"].smooth.strategy != "manual"
    assert calls["spec"].low_rank_solver.mode == "search"
