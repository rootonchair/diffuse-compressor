"""Every entry point should agree with ``CalibrationSpec`` on the capture mode.

``CalibrationSpec.scope_capture_mode`` and all five model-specific example CLIs
default to ``all_targets``. The three generic ``quantize_hf.py`` scanners used to
default to ``one-target``, which replays each calibration scope once per target
instead of once per scope. On a model with 8 quantized linears per block that is
8x the replay passes, and replay is the dominant cost for deep denoisers, so the
divergence was a large silent slowdown for anyone using the generic path.
"""

import pytest

from diffuse_compressor.config import CalibrationSpec


def test_library_default_is_all_targets():
    assert CalibrationSpec(num_samples=1).scope_capture_mode == "all_targets"


@pytest.mark.parametrize(
    ("module_path", "argv"),
    [
        ("examples.text_to_image.quantize_hf", ["some/model"]),
        ("examples.text_to_video.quantize_hf", ["some/model"]),
        # image-to-image additionally requires a calibration dataset.
        ("examples.image_to_image.quantize_hf", ["some/model", "--dataset", "some/dataset"]),
    ],
)
def test_generic_scanners_default_to_all_targets(module_path, argv):
    module = pytest.importorskip(module_path)
    args = module.build_parser().parse_args(argv)
    assert args.scope_capture_mode == "all-targets"


@pytest.mark.parametrize(
    ("module_path", "argv"),
    [
        ("examples.text_to_image.quantize_hf", ["some/model"]),
        ("examples.text_to_video.quantize_hf", ["some/model"]),
        ("examples.image_to_image.quantize_hf", ["some/model", "--dataset", "some/dataset"]),
    ],
)
def test_one_target_remains_available_for_low_memory_runs(module_path, argv):
    module = pytest.importorskip(module_path)
    args = module.build_parser().parse_args([*argv, "--scope-capture-mode", "one-target"])
    assert args.scope_capture_mode == "one-target"


def test_model_specific_parsers_already_agree():
    from examples.text_to_image.utils import default_arg_parser

    parser = default_arg_parser(
        "some/model", "out.safetensors", steps=4, guidance_scale=1.0, batch_size=1
    )
    assert parser.parse_args([]).scope_capture_mode == "all-targets"
