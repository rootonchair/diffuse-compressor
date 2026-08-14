from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffuse_compressor import (
    AwqTargetQuant,
    SvdqTargetQuant,
    collect_quant_targets,
    inspect_target_config,
)
from diffuse_compressor.calibration import assign_calibration_scopes
from examples.text_to_video import quantize_minimax_h3
from examples.text_to_video.quantize_minimax_h3 import (
    _h3_block_prev_replay_transform,
    build_parser,
    format_minimax_h3_prompt,
    load_minimax_h3_pipeline,
    minimax_h3_forward_fn,
    minimax_h3_prompt_records,
    minimax_h3_target_config,
    run,
    save_minimax_h3_videos,
    validate_minimax_h3_args,
)


class TinyAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(128, 128)
        self.to_k = nn.Linear(128, 128)
        self.to_v = nn.Linear(128, 128)
        self.to_out = nn.ModuleList([nn.Linear(128, 128)])


class TinySwiGLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(128, 256)


class TinyFeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.ModuleList([TinySwiGLU(), nn.Identity(), nn.Linear(128, 128)])


class TinyAdaLN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(128, 384)


class TinyBlock(nn.Module):
    def __init__(self, *, with_adaln: bool) -> None:
        super().__init__()
        self.attn = TinyAttention()
        self.ff = TinyFeedForward()
        if with_adaln:
            self.adaln_proj = TinyAdaLN()


class TinyRefiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [TinyBlock(with_adaln=False), TinyBlock(with_adaln=False)]
        )


class TinyMiniMaxH3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_in = nn.Linear(128, 128)
        self.token_refiner = TinyRefiner()
        self.transformer_blocks = nn.ModuleList(
            [TinyBlock(with_adaln=True), TinyBlock(with_adaln=True)]
        )
        self.proj_out = nn.Linear(128, 128)


def test_minimax_h3_target_config_selects_expected_layers() -> None:
    model = TinyMiniMaxH3()
    config = minimax_h3_target_config()

    report = inspect_target_config(model, config)
    targets = collect_quant_targets(model, config)
    scopes = assign_calibration_scopes(model, targets, config)

    assert report.ok
    assert sum(isinstance(target.quant, SvdqTargetQuant) for target in targets) == 16
    assert sum(isinstance(target.quant, AwqTargetQuant) for target in targets) == 2
    assert "proj_in" not in {target.name for target in targets}
    assert "proj_out" not in {target.name for target in targets}
    assert {
        target.export_name
        for target in targets
        if isinstance(target.quant, AwqTargetQuant)
    } == {
        "blocks.0.adaln_proj.linear",
        "blocks.1.adaln_proj.linear",
    }
    qkv = next(
        target for target in targets if target.export_name == "blocks.0.attn.qkv_proj"
    )
    assert qkv.module_names == (
        "transformer_blocks.0.attn.to_q",
        "transformer_blocks.0.attn.to_k",
        "transformer_blocks.0.attn.to_v",
    )
    assert qkv.roles == ("q", "k", "v")
    assert [scope.module_name for scope in scopes] == [
        "token_refiner.refiner_blocks.0",
        "token_refiner.refiner_blocks.1",
        "transformer_blocks.0",
        "transformer_blocks.1",
    ]
    assert [scope.use_prev_scope_outputs for scope in scopes] == [
        True,
        True,
        False,
        True,
    ]
    assert scopes[-1].prev_replay_transform is _h3_block_prev_replay_transform


def test_minimax_h3_forward_fn_builds_pipeline_kwargs(monkeypatch) -> None:
    generator = torch.Generator().manual_seed(7)
    monkeypatch.setattr(
        quantize_minimax_h3,
        "make_generator",
        lambda seed, device: generator,
    )
    calls = []

    def pipeline(**kwargs):
        calls.append(kwargs)
        return "pipeline-state"

    args = SimpleNamespace(
        height=544,
        width=960,
        num_frames=124,
        steps=4,
        device="cpu",
        save_calibration_videos=False,
    )

    result = minimax_h3_forward_fn(pipeline, args)(
        {"prompt": "a test prompt", "seed": 7}
    )

    assert result == "pipeline-state"
    assert calls == [
        {
            "prompt": "a test prompt",
            "height": 544,
            "width": 960,
            "num_frames": 124,
            "num_inference_steps": 4,
            "generator": generator,
            "output_type": "latent",
            "output": ["videos", "audio", "sampling_rate"],
        }
    ]


def test_h3_block_prev_replay_transform_updates_hidden_states() -> None:
    output = torch.randn(1, 4, 8)
    temb = torch.randn(3, 8)

    args, kwargs = _h3_block_prev_replay_transform(
        SimpleNamespace(output=output, args=(torch.zeros_like(output), temb), kwargs={})
    )
    assert args == (output, temb)
    assert kwargs == {}

    args, kwargs = _h3_block_prev_replay_transform(
        SimpleNamespace(
            output=output,
            args=(),
            kwargs={"hidden_states": torch.zeros_like(output), "temb": temb},
        )
    )
    assert args == ()
    assert kwargs == {"hidden_states": output, "temb": temb}


def test_h3_block_prev_replay_transform_requires_tensor_output() -> None:
    with pytest.raises(TypeError, match="hidden-state tensor"):
        _h3_block_prev_replay_transform(
            SimpleNamespace(output=(torch.zeros(1),), args=(), kwargs={})
        )


def test_minimax_h3_latent_loader_skips_vae_weights(monkeypatch, tmp_path) -> None:
    config_calls = []

    class ConfigLoader:
        @classmethod
        def load_config(cls, model_id, *, subfolder, revision):
            config_calls.append((model_id, subfolder, revision))
            return {
                "latent_channels": 24,
                "latents_mean": [0.0],
                "latents_std": [1.0],
                "sampling_rate": 32_000,
            }

    class FakePipe:
        pretrained_component_names = [
            "transformer",
            "text_encoder",
            "tokenizer",
            "processor",
            "vae",
            "audio_vae",
        ]

        def __init__(self):
            self.loaded = None
            self.moved_to = None
            self.specs = {
                name: SimpleNamespace(
                    type_hint=ConfigLoader,
                    pretrained_model_name_or_path="MiniMaxAI/MiniMax-H3",
                    subfolder=name,
                    revision=None,
                )
                for name in self.pretrained_component_names
            }

        def load_components(self, *, names, **kwargs):
            self.loaded = (names, kwargs)

        def get_component_spec(self, name):
            return self.specs[name]

        def to(self, device):
            self.moved_to = device

    fake_pipe = FakePipe()
    monkeypatch.setattr(
        "diffusers.ModularPipeline.from_pretrained",
        lambda model_id, components_manager: fake_pipe,
    )

    result = load_minimax_h3_pipeline(
        "MiniMaxAI/MiniMax-H3",
        device="cpu",
        pipeline_offload="none",
        memory_reserve_margin="12GB",
        text_encoder_path="/models/qwen3-vl",
    )

    assert result is fake_pipe
    assert result.loaded == (
        ["transformer", "text_encoder", "tokenizer", "processor"],
        {
            "dtype": torch.bfloat16,
            "pretrained_model_name_or_path": {"text_encoder": "/models/qwen3-vl"},
            "subfolder": {"text_encoder": ""},
        },
    )
    assert result.moved_to == "cpu"
    assert result.vae.spatial_compression_ratio == 16
    assert result.vae.config.latent_channels == 24
    assert result.audio_vae.config.sampling_rate == 32_000
    assert config_calls == [
        ("MiniMaxAI/MiniMax-H3", "vae", None),
        ("MiniMaxAI/MiniMax-H3", "audio_vae", None),
    ]

    local_model_dir = tmp_path / "minimax-h3"
    local_model_dir.mkdir()
    load_minimax_h3_pipeline(
        str(local_model_dir),
        device="cpu",
        pipeline_offload="none",
        memory_reserve_margin="12GB",
        text_encoder_path="/models/qwen3-vl",
    )

    assert fake_pipe.loaded == (
        ["transformer", "text_encoder", "tokenizer", "processor"],
        {
            "dtype": torch.bfloat16,
            "pretrained_model_name_or_path": {
                "transformer": str(local_model_dir),
                "text_encoder": "/models/qwen3-vl",
                "tokenizer": str(local_model_dir),
                "processor": str(local_model_dir),
            },
            "subfolder": {
                "transformer": "transformer",
                "text_encoder": "",
                "tokenizer": "tokenizer",
                "processor": "processor",
            },
        },
    )
    assert config_calls[-2:] == [
        (str(local_model_dir), "vae", None),
        (str(local_model_dir), "audio_vae", None),
    ]


def test_minimax_h3_parser_defaults() -> None:
    args = build_parser().parse_args([])

    assert args.model_id == "MiniMaxAI/MiniMax-H3"
    assert args.precision == "nvfp4"
    assert args.rank == 32
    assert args.num_samples == 1
    assert (args.height, args.width) == (768, 1344)
    assert args.num_frames == 124
    assert args.fps == 24
    assert args.pipeline_offload == "none"
    assert args.scope_capture_mode == "all-targets"
    assert args.sample_batch_size == -1
    assert args.row_sampling == "reservoir"
    assert args.max_eval_replays == 4
    assert args.save_model_cache is False
    assert args.gptq is False
    assert args.gptq_damp_percentage == 0.01
    assert args.gptq_block_size == 128
    assert args.gptq_num_inv_tries == 250
    assert args.gptq_hessian_block_size == 512


def test_minimax_h3_run_wires_gptq(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        quantize_minimax_h3,
        "load_minimax_h3_pipeline",
        lambda *args, **kwargs: SimpleNamespace(transformer=TinyMiniMaxH3()),
    )
    monkeypatch.setattr(
        quantize_minimax_h3,
        "quantize_and_export",
        lambda **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "quantize_minimax_h3.py",
            "--gptq",
            "--gptq-damp-percentage",
            "0.02",
            "--gptq-block-size",
            "64",
            "--gptq-num-inv-tries",
            "17",
            "--gptq-hessian-block-size",
            "256",
            "--device",
            "cpu",
        ],
    )

    run()

    gptq = captured["spec"].gptq
    assert gptq.enabled is True
    assert gptq.damp_percentage == 0.02
    assert gptq.block_size == 64
    assert gptq.num_inv_tries == 17
    assert gptq.hessian_block_size == 256
    assert captured["export"].output.name == (
        "svdq-gptq-fp4_r32-minimax-h3-fl2va.safetensors"
    )
    calibration = captured["calibration"]
    assert calibration.row_sampling == "reservoir"
    assert calibration.max_eval_replays == 4
    assert calibration.artifact_cache.cache_dir.name == (
        "gptq-reservoir-n1-replay4-artifacts"
    )
    assert calibration.artifact_cache.save_model is False


def test_minimax_h3_prompt_records_use_native_audiovisual_format(tmp_path) -> None:
    prompt_file = tmp_path / "prompts.yaml"
    prompt_file.write_text("'0000': A dog catches a frisbee.\n", encoding="utf-8")

    records = minimax_h3_prompt_records(1, prompt_file)

    assert records[0]["filename"] == "0000-0"
    assert records[0]["prompt"] == (
        "integrated_multimodal_description: [Shot 1] A dog catches a frisbee.\n\n"
        "overall_soundscape: Natural ambient sound synchronized with the visible action.\n\n"
        "non_diegetic_music: N/A"
    )


def test_minimax_h3_prompt_format_preserves_native_prompt() -> None:
    prompt = (
        "integrated_multimodal_description: [Shot 1] A train passes.\n\n"
        "overall_soundscape: Rhythmic rail noise.\n\n"
        "non_diegetic_music: N/A"
    )

    assert format_minimax_h3_prompt(prompt) == prompt


def test_validate_minimax_h3_args_rejects_invalid_geometry() -> None:
    args = build_parser().parse_args([])
    args.width = 1340

    with pytest.raises(ValueError, match="multiples of 32"):
        validate_minimax_h3_args(args)


def test_save_minimax_h3_videos_muxes_audio(monkeypatch, tmp_path) -> None:
    calls = []
    monkeypatch.setattr(
        "diffusers.utils.export_utils.encode_video",
        lambda video, **kwargs: calls.append((video, kwargs)),
    )
    state = {
        "videos": [torch.zeros(1, 3, 4, 4)],
        "audio": [torch.zeros(1, 16)],
        "sampling_rate": 44_100,
    }

    save_minimax_h3_videos(
        state,
        {"filename": ["sample"]},
        tmp_path,
        fps=24,
    )

    assert len(calls) == 1
    assert calls[0][1] == {
        "fps": 24,
        "output_path": str(tmp_path / "sample.mp4"),
        "audio": state["audio"][0],
        "audio_sample_rate": 44_100,
    }
