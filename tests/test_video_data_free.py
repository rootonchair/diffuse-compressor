import importlib.util
import sys
from pathlib import Path

import pytest
from torch import nn

import examples.text_to_video.utils as video_utils
from examples.text_to_video.quantize_hf import build_parser as build_video_parser

video_svdquant_spec = video_utils.svdquant_spec


def _load_ltx_example():
    """Import quantize_ltx2_3 with its bare ``utils`` import bound to the video utils.

    Other example tests may already have put a different ``utils`` module in
    ``sys.modules``, so bind it explicitly for the import and restore it after.
    """

    path = Path(__file__).resolve().parents[1] / "examples" / "text_to_video" / "quantize_ltx2_3.py"
    previous = sys.modules.get("utils")
    sys.modules["utils"] = video_utils
    try:
        spec = importlib.util.spec_from_file_location("_ltx2_3_data_free_example", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = previous
    return module


ltx_example = _load_ltx_example()


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_video_spec_data_free_uses_weight_span_smoothing_and_plain_svd(precision):
    spec = video_svdquant_spec(precision, data_free=True)
    assert spec.smooth.enabled and spec.smooth.strategy == "manual"
    assert (spec.smooth.alpha, spec.smooth.beta) == (0.0, 0.5)
    assert spec.low_rank_solver.mode == "weighted_svd"
    assert spec.rotation is None
    assert video_svdquant_spec(precision, data_free=True, smooth_exponent=0.25).smooth.beta == 0.25
    assert not video_svdquant_spec(precision, data_free=True, smooth_exponent=0.0).smooth.enabled


def test_video_spec_calibrated_path_is_unchanged():
    spec = video_svdquant_spec("int4")
    assert spec.smooth.strategy == "grid_search" and spec.smooth.num_grids == 10
    assert spec.low_rank_solver.mode == "search"
    assert spec.rotation is None


def test_video_spec_hadamard_defaults_smoothing_off_but_respects_explicit_exponent():
    rotated = video_svdquant_spec("int4", data_free=True, hadamard=True)
    assert rotated.rotation is not None
    assert not rotated.smooth.enabled
    explicit = video_svdquant_spec("int4", data_free=True, hadamard=True, smooth_exponent=0.25)
    assert explicit.smooth.enabled and explicit.smooth.beta == 0.25
    with pytest.raises(ValueError, match="smooth_exponent"):
        video_svdquant_spec("int4", data_free=True, smooth_exponent=-0.1)


def test_generic_video_parser_accepts_data_free_flags():
    args = build_video_parser().parse_args(["model/id", "--data-free", "--hadamard", "--smooth-exponent", "0.25"])
    assert args.data_free and args.hadamard and args.smooth_exponent == 0.25
    defaults = build_video_parser().parse_args(["model/id"])
    assert not defaults.data_free and not defaults.hadamard and defaults.smooth_exponent is None


def test_ltx_variant_output_names():
    base = "outputs/checkpoints/svdq-int4_r32-ltx2.3.safetensors"
    assert ltx_example._variant_output(base, data_free=False, hadamard=False) == base
    assert ltx_example._variant_output(base, data_free=True, hadamard=False).endswith("ltx2.3-datafree.safetensors")
    assert ltx_example._variant_output(base, data_free=True, hadamard=True).endswith(
        "ltx2.3-datafree-hadamard.safetensors"
    )


def test_ltx_data_free_loads_only_the_transformer_and_skips_calibration(monkeypatch, tmp_path):
    transformer = nn.Linear(64, 64)
    calls = {}

    def fail_pipeline(*args, **kwargs):
        raise AssertionError("data-free must not load the full LTX2 pipeline")

    def fake_load_transformer(model_id, *, device, dtype):
        calls["load"] = (model_id, device)
        return transformer

    def fake_quantize_and_export(*, model, spec, target_config, calibration, export, logging):
        calls["quantize"] = dict(model=model, spec=spec, calibration=calibration, output=export.output)

    # The parser reads the upstream default negative prompt from diffusers.
    monkeypatch.setattr(ltx_example, "_ltx2_default_negative_prompt", lambda: "")
    monkeypatch.setattr(ltx_example, "load_ltx2_pipeline", fail_pipeline)
    monkeypatch.setattr(ltx_example, "load_ltx2_transformer", fake_load_transformer)
    monkeypatch.setattr(ltx_example, "ltx2_3_target_config", lambda precision, fuse_qkv=False: object())
    monkeypatch.setattr(ltx_example, "validate_ltx2_3_nunchaku_lite_manifest_targets", lambda *args: None)
    monkeypatch.setattr(ltx_example, "quantize_and_export", fake_quantize_and_export)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "quantize_ltx2_3.py",
            "--data-free",
            "--hadamard",
            "--cache-mode",
            "disabled",
            "--device",
            "cpu",
            "--no-run-log",
        ],
    )

    ltx_example.run_ltx2_3_cli(
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

    assert calls["load"] == ("org/ltx", "cpu")
    quantized = calls["quantize"]
    assert quantized["model"] is transformer
    assert quantized["calibration"] is None
    assert quantized["spec"].rotation is not None
    assert not quantized["spec"].smooth.enabled
    assert quantized["spec"].low_rank_solver.mode == "weighted_svd"
    assert quantized["output"].name == "svdq-int4_r32-ltx-datafree-hadamard.safetensors"
