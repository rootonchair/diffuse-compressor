"""examples/quantize_data_free.py: flags, defaults, loading and target selection."""

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


def test_auto_target_config_scans_the_linear_layers(run_script):
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
        (["--target-config", "flux2-klein", "--skip", "x"], "--skip"),
        (["--fuse-qkv"], "--fuse-qkv"),
        (["--smooth-exponent", "1.5"], None),
    ],
)
def test_bad_flag_combinations_fail_before_loading(monkeypatch, capsys, argv, message):
    monkeypatch.setattr(data_free, "load_denoiser", lambda *a, **k: pytest.fail("model loaded"))
    with pytest.raises((SystemExit, ValueError)):
        data_free.run(["org/model", *argv])
    if message:
        assert message in capsys.readouterr().err


def test_calibration_flags_are_not_accepted():
    for flag in ("--num-samples", "--prompt-file", "--pipeline-offload", "--steps"):
        with pytest.raises(SystemExit):
            data_free.build_parser().parse_args(["org/model", flag, "1"])


def test_flux2_klein_config_is_sized_from_the_model(monkeypatch):
    seen = {}

    def fake_config(precision, *, single_qkv_features, single_attn_features):
        seen.update(precision=precision, qkv=single_qkv_features, attn=single_attn_features)
        return "klein-config"

    monkeypatch.setattr(
        data_free, "_load_example_module", lambda *a: SimpleNamespace(flux2_klein_target_config=fake_config)
    )
    assert data_free.flux2_klein_targets(_TinyDenoiser(), "nvfp4") == "klein-config"
    assert seen == {"precision": "nvfp4", "qkv": 3 * 4096, "attn": 4096}  # klein-9B sizes


def test_ltx_config_runs_the_nunchaku_lite_manifest_check(monkeypatch):
    seen = {}
    fake = SimpleNamespace(
        ltx2_3_target_config=lambda precision, fuse_qkv: ("ltx-config", precision, fuse_qkv),
        validate_ltx2_3_nunchaku_lite_manifest_targets=lambda model, config: seen.setdefault("validated", config),
    )
    monkeypatch.setattr(data_free, "_load_example_module", lambda *a: fake)
    config = data_free.ltx2_3_targets(_TinyDenoiser(), "int4", fuse_qkv=True)
    assert config == ("ltx-config", "int4", True)
    assert seen["validated"] == config


@pytest.mark.parametrize(
    "relative, utils_module, attrs",
    [
        ("text_to_image/quantize_flux2_klein_4b.py", "examples.text_to_image.utils", ("flux2_klein_target_config",)),
        (
            "text_to_video/quantize_ltx2_3.py",
            "examples.text_to_video.utils",
            ("ltx2_3_target_config", "validate_ltx2_3_nunchaku_lite_manifest_targets"),
        ),
    ],
)
def test_model_scripts_import_with_the_right_utils(relative, utils_module, attrs):
    previous = sys.modules.get("utils")
    module = data_free._load_example_module(relative, utils_module)
    assert all(callable(getattr(module, attr)) for attr in attrs)
    assert sys.modules.get("utils") is previous  # binding restored


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
