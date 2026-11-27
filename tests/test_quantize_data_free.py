"""examples/quantize_data_free.py (generic scanner): flags, defaults, loading, end-to-end export."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

import examples.quantize_data_free as data_free


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(128, 128)
        self.to_out = nn.Linear(128, 128)


class _TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])
        self.config = SimpleNamespace(num_attention_heads=32, attention_head_dim=128)


@pytest.fixture
def run_script(monkeypatch, tmp_path):
    """Run the script with a tiny model; return what was loaded and quantized."""

    calls = {}
    model = _TinyDenoiser()

    def fake_load(model_id, *, device, dtype=None):
        calls["load"] = (model_id, device)
        calls["dtype"] = dtype
        return model

    def fake_quantize_and_export(*, model, spec, target_config, calibration, export, logging):
        calls.update(model=model, spec=spec, target_config=target_config, calibration=calibration, output=export.output)

    monkeypatch.setattr(data_free, "load_denoiser", fake_load)
    monkeypatch.setattr(data_free, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.chdir(tmp_path)

    def run(*argv):
        data_free.run(["org/Some-Model", "--device", "cpu", "--no-run-log", *argv])
        return calls

    return run


def test_defaults_build_the_data_free_spec_and_output_names(run_script):
    calls = run_script()
    spec = calls["spec"]
    assert spec.smooth.enabled and spec.smooth.strategy == "manual"
    assert (spec.smooth.alpha, spec.smooth.beta) == (0.0, 0.5)
    assert spec.low_rank_solver.mode == "weighted_svd"
    assert spec.rotation is None and spec.rank == 32
    assert calls["load"] == ("org/Some-Model", "cpu")
    assert calls["output"] == Path("outputs/checkpoints/svdq-int4_r32-some-model-datafree.safetensors")
    cache = calls["calibration"].artifact_cache
    assert Path(cache.cache_dir) == Path("outputs/calibration/some-model/int4/artifacts-datafree")
    assert calls["calibration"].cache_dir is None  # no calibration capture, only the artifact cache


def test_hadamard_smoothing_exponent_rank_and_cache_flags(run_script):
    calls = run_script("--hadamard", "--rank", "64", "--cache-mode", "disabled")
    assert calls["spec"].rotation is not None
    assert not calls["spec"].smooth.enabled  # 0 by default under rotation
    assert calls["spec"].rank == 64
    assert calls["calibration"] is None
    assert calls["output"].name == "svdq-int4_r64-some-model-datafree-hadamard.safetensors"
    explicit = run_script("--hadamard", "--smooth-exponent", "0.25")["spec"].smooth
    assert explicit.enabled and explicit.beta == 0.25


def test_offload_model_loads_on_cpu_and_quantizes_on_the_device(run_script):
    calls = run_script("--device", "cuda", "--offload-model")
    assert calls["load"] == ("org/Some-Model", "cpu")
    assert calls["spec"].offload_model and calls["spec"].compute_device == "cuda"


def test_generic_scan_finds_the_linear_layers(run_script):
    calls = run_script("--skip", "*to_out*")
    config = calls["target_config"]
    assert config is not None
    from diffuse_compressor import inspect_target_config

    text = inspect_target_config(calls["model"], config).format_text()
    assert "to_q" in text


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--precision", "nvfp4", "--hadamard"], "int4 only"),
        (["--smooth-exponent", "1.5"], "--smooth-exponent must be in [0, 1]"),
    ],
)
def test_bad_flag_combinations_fail_before_loading(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(data_free, "load_denoiser", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(SystemExit):
        data_free.run(["org/model", *argv])
    assert message in capsys.readouterr().err


def test_dtype_flag_sets_the_load_dtype(run_script):
    import torch

    assert run_script()["dtype"] is torch.bfloat16
    assert run_script("--dtype", "float32")["dtype"] is torch.float32


@pytest.mark.parametrize("flag", ["--num-samples", "--prompt-file", "--pipeline-offload", "--steps", "--target-config"])
def test_calibration_and_model_specific_flags_are_not_accepted(monkeypatch, flag):
    monkeypatch.setattr(data_free, "load_denoiser", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises(SystemExit):
        data_free.run(["org/model", flag, "1"])


@pytest.mark.parametrize("extra", [[], ["--hadamard"]])
def test_end_to_end_writes_a_checkpoint(monkeypatch, tmp_path, extra):
    from safetensors import safe_open

    monkeypatch.setattr(data_free, "load_denoiser", lambda model_id, *, device, dtype=None: _TinyDenoiser())
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "tiny.safetensors"
    data_free.run(
        ["org/tiny", "--device", "cpu", "--no-run-log", "--svd-backend", "full", "--output", str(output), *extra]
    )
    with safe_open(str(output), framework="pt") as handle:
        keys = set(handle.keys())
    assert any(key.endswith("proj_down") for key in keys)
    assert (tmp_path / "outputs/calibration/tiny/int4/artifacts-datafree").is_dir()
