"""The Qwen-Image 2.1 script must keep the generic scanner's targets and calibrate safely.

Matching the scanner keeps the checkpoint ABI and runtime manifest stable. The
forward function must disable the pipeline's prefix KV cache, because cached
calls skip text tokens and carry a stateful cache object into replay records.
"""

from types import SimpleNamespace

import pytest
import torch

transformer_qwenimage21 = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")

from examples.text_to_image import quantize_qwen_image_2_1 as qwen_example  # noqa: E402
from examples.text_to_image.quantize_hf import scan_linear_targets  # noqa: E402


def _tiny_model():
    return transformer_qwenimage21.QwenImage21Transformer2DModel(
        attention_head_dim=128,
        num_attention_heads=2,
        num_layers=2,
        in_channels=64,
        out_channels=64,
        context_in_dim=256,
        axes_dims_rope=[16, 56, 56],
        mlp_ratio=3,
        causal_condition=True,
    )


def test_target_set_is_identical_to_the_generic_scanner():
    model = _tiny_model()
    generic = scan_linear_targets(model, precision="int4", rank=32)
    chained, scan = qwen_example.qwen_image_21_target_config(model, precision="int4", rank=32)

    assert chained.targets == generic.target_config.targets
    assert scan.awq_targets == ("modulation.1",)
    assert set(scan.svdq_targets) == {
        f"transformer_blocks.{index}.{name}"
        for index in range(2)
        for name in (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_out.0",
            "img_mlp.proj",
            "img_mlp.gate_layer",
            "img_mlp.out",
        )
    }


def test_first_block_recomputes_and_the_rest_chain():
    first, rest = qwen_example.qwen_image_21_calibration_scopes()
    block_class = transformer_qwenimage21.QwenImage21TransformerBlock

    assert first.modules == ("transformer_blocks.0",)
    assert first.use_prev_scope_outputs is False
    assert first.module_classes == (block_class,)
    assert rest.modules == ("transformer_blocks.*",)
    assert rest.prev_replay_transform is qwen_example._qwen_image_21_block_prev_replay_transform
    assert rest.module_classes == (block_class,)


def test_prev_replay_transform_replaces_only_hidden_states():
    output = torch.ones(1, 4, 8)
    shared = {"hidden_states": torch.zeros(1, 4, 8), "modulation": torch.randn(2, 16), "segments": [(0, 4, True)]}

    args, kwargs = qwen_example._qwen_image_21_block_prev_replay_transform(
        SimpleNamespace(args=(), kwargs=shared, output=output)
    )
    assert args == ()
    assert kwargs["hidden_states"] is output
    assert kwargs["modulation"] is shared["modulation"]
    assert kwargs["segments"] is shared["segments"]
    assert shared["hidden_states"] is not output

    args, kwargs = qwen_example._qwen_image_21_block_prev_replay_transform(
        SimpleNamespace(args=(torch.zeros(1), "mod"), kwargs={}, output=output)
    )
    assert args == (output, "mod")
    assert kwargs == {}


def _forward_kwargs(*cli):
    calls = []
    args = qwen_example.build_parser().parse_args(["--device", "cpu", *cli])
    forward = qwen_example.qwen_image_21_forward_fn(lambda **kwargs: calls.append(kwargs), args)
    forward({"prompt": "a red train", "seed": 3})
    return calls[0]


def test_forward_disables_kv_cache_and_skips_cfg_by_default():
    kwargs = _forward_kwargs()

    assert kwargs["use_kv_cache"] is False
    assert (kwargs["height"], kwargs["width"], kwargs["num_inference_steps"]) == (1024, 1024, 40)
    assert "guidance_scale" not in kwargs
    assert "true_cfg_scale" not in kwargs
    assert "negative_prompt" not in kwargs


def test_forward_runs_negative_prompt_passes_when_cfg_is_enabled():
    kwargs = _forward_kwargs("--true-cfg-scale", "4.0", "--negative-prompt", "blurry")

    assert kwargs["true_cfg_scale"] == 4.0
    assert kwargs["negative_prompt"] == "blurry"
    assert kwargs["use_kv_cache"] is False


def test_run_rejects_batched_calibration(monkeypatch):
    def fail_load(*_args, **_kwargs):
        raise AssertionError("pipeline must not load when batch size is invalid")

    monkeypatch.setattr(qwen_example, "load_auto_pipeline", fail_load)
    monkeypatch.setattr("sys.argv", ["quantize_qwen_image_2_1.py", "--batch-size", "2"])

    with pytest.raises(ValueError, match="batch-size 1"):
        qwen_example.run()


def test_row_limit_defaults_above_widest_target_and_gptq_is_opt_in():
    args = qwen_example.build_parser().parse_args(["--device", "cpu"])

    assert args.max_rows_per_target == qwen_example.DEFAULT_MAX_ROWS_PER_TARGET
    assert args.max_rows_per_target > 12288  # img_mlp.out input width
    assert qwen_example.qwen_image_21_spec(args).gptq.enabled is False


def test_gptq_flags_build_an_enabled_gptq_spec():
    args = qwen_example.build_parser().parse_args(
        ["--device", "cpu", "--gptq", "--gptq-damp-percentage", "0.02", "--gptq-block-size", "64", "--rank", "64"]
    )
    spec = qwen_example.qwen_image_21_spec(args)

    assert spec.gptq.enabled is True
    assert spec.gptq.damp_percentage == 0.02
    assert spec.gptq.block_size == 64
    assert spec.rank == 64
