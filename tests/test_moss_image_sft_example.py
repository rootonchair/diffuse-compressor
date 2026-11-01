import json
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.text_to_image import quantize_moss_image_sft as moss
from examples.text_to_image.quantize_moss_image_sft import MossReplaySamples, compatible_source, moss_spec

ROOT = Path(__file__).resolve().parents[1]


def test_native_dtype_compatibility_keeps_dense_and_packed_compute_dtypes():
    original = """class _RMSNorm(nn.Module): pass
def example(self):
    return self.proj['q'].weight.dtype, self.shared_block_modulation.weight.dtype, self.video_head_modulation.weight.dtype
"""
    patched = compatible_source(original)
    assert compatible_source(patched) == patched
    namespace = {"nn": torch.nn}
    exec(patched, namespace)
    resolve = namespace["_linear_compute_dtype"]
    assert resolve(torch.nn.Linear(128, 128).bfloat16()) == torch.bfloat16

    class NativeSVDQ:
        torch_dtype = torch.bfloat16

    class NativeAWQ:
        wscales = torch.ones(1, dtype=torch.bfloat16)

    assert resolve(NativeSVDQ()) == resolve(NativeAWQ()) == torch.bfloat16
    with pytest.raises(ValueError, match="Unexpected MOSS pipeline source"):
        compatible_source("class _RMSNorm(nn.Module): pass\n")


def test_memmapped_capture_preserves_both_guidance_calls_and_layer_skip(tmp_path):
    (tmp_path / "calibration").mkdir()
    (tmp_path / "embeddings").mkdir()
    files = []
    for i in range(2):
        file = f"embeddings/calib-{i:04d}.pt"
        files.append(file)
        torch.save({"positive_embeds": torch.full((1, 5, 128), float(i), dtype=torch.bfloat16),
                    "positive_mask": torch.ones(1, 5, dtype=torch.bool),
                    "negative_embeds": torch.ones(1, 3, 128, dtype=torch.bfloat16),
                    "negative_mask": torch.ones(1, 3, dtype=torch.bool)}, tmp_path / file)
    latents = np.arange(4 * 16 * 2 * 2, dtype=np.float32).reshape(4, 16, 2, 2)
    np.save(tmp_path / "calibration/latents.npy", latents)
    records = [{"latent_index": i // 2, "prompt_index": i // 4, "branch": "positive" if i % 2 == 0 else "negative",
                "skip_layers": [] if i % 2 == 0 else [22], "timestep": 1000 - i // 2 * 10.,
                "timestep_dtype": "float32", "timestep_shape": []} for i in range(8)]
    meta = {"complete": True, "records": records, "prompts": 2, "steps": 2, "embedding_files": files}
    moss._write(tmp_path / "calibration/records.json", meta)
    samples = MossReplaySamples(tmp_path / "calibration")
    assert len(samples) == 8
    assert torch.equal(samples[0]["hidden_states"], samples[1]["hidden_states"])
    assert samples[0]["hidden_states"].data_ptr() == samples[1]["hidden_states"].data_ptr()
    assert samples[0]["skip_layers"] == () and samples[1]["skip_layers"] == (22,)
    assert samples[0]["encoder_hidden_states"].shape[1] == 5
    assert samples[1]["encoder_hidden_states"].shape[1] == 3
    assert samples[1]["timestep"].shape == torch.Size([])
    meta["records"][1]["skip_layers"] = []
    moss._write(tmp_path / "calibration/records.json", meta)
    with pytest.raises(ValueError, match="skipping changed"):
        MossReplaySamples(tmp_path / "calibration")


@pytest.mark.parametrize("precision", ("nvfp4", "int4"))
@pytest.mark.parametrize("rank", (32, 64, 128))
def test_gptq_and_rank_match_the_published_recipe(precision, rank):
    spec = moss_spec(precision, rank)
    assert spec.rank == rank and spec.gptq.enabled
    assert spec.gptq.damp_percentage == .01 and spec.gptq.block_size == 128
    assert spec.gptq.hessian_block_size == 512 and spec.gptq.num_inv_tries == 250
    assert not spec.offload_model


def test_profile_matches_pinned_moss_defaults():
    assert moss.PROFILE == {"steps": 50, "guidance_scale": 3.0, "shift": 5.0, "negative_prompt": "",
                            "uncond_skip_layers": [22], "slg_scale": 0.0}
    assert moss.RESOLUTION == 1024


def test_shipped_prompt_files_load_and_bad_files_are_rejected(tmp_path):
    for name, count in (("moss_sft_calibration8.json", 8), ("moss_sft_calibration32.json", 32)):
        assert len(moss.load_prompt_records(ROOT / "examples/prompts" / name)) == count
    bad = tmp_path / "bad.json"
    for payload, match in (([{"prompt": "a", "seed": 1}], "records"), ({"records": []}, "records"),
                           ({"records": [{"prompt": "a"}]}, "seed"),
                           ({"records": [{"prompt": "a", "seed": 1}, {"prompt": "a", "seed": 2}]}, "duplicate")):
        bad.write_text(json.dumps(payload))
        with pytest.raises(ValueError, match=match):
            moss.load_prompt_records(bad)


def test_recipe_fingerprint_tracks_prompts_and_pipeline():
    records = [{"filename": "a", "prompt": "a cat", "seed": 1}]
    recipe = moss.build_recipe(records, "abc")
    assert recipe["calibration_calls"] == 100 and recipe["revision"] == moss.REVISION
    assert moss.fingerprint(recipe) == moss.fingerprint(moss.build_recipe(records, "abc"))
    assert moss.fingerprint(recipe) != moss.fingerprint(moss.build_recipe(records, "abd"))
    assert moss.fingerprint(recipe) != moss.fingerprint(moss.build_recipe([{**records[0], "seed": 2}], "abc"))


def test_cli_stages_and_missing_preparation(tmp_path):
    parser = moss.build_parser()
    args = parser.parse_args(["prepare", "--run-dir", str(tmp_path)])
    assert args.prompt_file.name == "moss_sft_calibration8.json" and args.device == "cuda"
    args = parser.parse_args(["quantize", "--run-dir", str(tmp_path), "--precision", "int4",
                              "--cuda-memory-limit-gib", "22", "--artifact-cache", "disabled"])
    assert (args.rank, args.cuda_memory_limit_gib, args.artifact_cache, args.preflight) == (32, 22.0, "disabled", False)
    with pytest.raises(SystemExit):
        parser.parse_args(["quantize", "--run-dir", str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="prepare"):
        moss.quantize(tmp_path, "nvfp4", device="cpu")


def test_call_recorder_requires_shared_latent_and_negative_skip():
    latents = np.zeros((2, 16, 2, 2), dtype=np.float32)
    recorder = moss.CallRecorder(latents, prompt_index=1, steps=1)
    x = torch.randn(1, 16, 2, 2)
    t = torch.tensor(1000.)
    recorder(None, (x, t), {})
    assert np.array_equal(latents[1], x[0].numpy())
    with pytest.raises(ValueError, match="different latent"):
        recorder(None, (x + 1, t), {"skip_layers": [22]})
    with pytest.raises(ValueError, match="skip_layers"):
        recorder(None, (x, t), {})
    assert len(recorder.records) == 1
    recorder(None, (x, t), {"skip_layers": (22,)})
    assert [r["branch"] for r in recorder.records] == ["positive", "negative"]
    assert recorder.records[1]["skip_layers"] == [22] and recorder.records[1]["latent_index"] == 1
    with pytest.raises(RuntimeError, match="More transformer calls"):
        recorder(None, (x, t), {})


def test_capture_on_a_tiny_transformer_replays_through_the_samples(tmp_path, monkeypatch):
    """End-to-end capture with the real pinned pipeline code and a tiny random transformer."""
    from test_moss_replay_equivalence import moss_compat_pipeline
    compat = tmp_path / "compat/pipeline_moss_image.py"
    compat.parent.mkdir()
    compat.write_text(moss_compat_pipeline(tmp_path).read_text())
    module = moss.load_moss_module(compat, name="moss_capture_unit_fixture")
    torch.manual_seed(0)
    module.MOSSImageTransformer(dim=128, ffn_dim=256, num_heads=8, num_layers=24, text_dim=256, freq_dim=32,
                                latent_channels=16).bfloat16().save_pretrained(tmp_path / "source/transformer")
    monkeypatch.setattr(moss, "PROFILE", {**moss.PROFILE, "steps": 2})
    monkeypatch.setattr(moss, "RESOLUTION", 32)
    records = [{"filename": f"p{i}", "prompt": f"prompt {i}", "seed": i} for i in range(2)]
    files = []
    for i in range(2):
        files.append(f"embeddings/calib-{i:04d}.pt")
        (tmp_path / "embeddings").mkdir(exist_ok=True)
        torch.save({"positive_embeds": torch.randn(1, 5, 256).bfloat16(), "positive_mask": torch.ones(1, 5, dtype=torch.bool),
                    "negative_embeds": torch.randn(1, 3, 256).bfloat16(), "negative_mask": torch.ones(1, 3, dtype=torch.bool)},
                   tmp_path / files[-1])
    moss.capture(tmp_path, tmp_path / "source", records, files, "cpu")
    samples = MossReplaySamples(tmp_path / "calibration")
    recipe = moss.build_recipe(records, "unused")
    assert moss.validate_calibration_recipe(samples, recipe) == 8
    assert samples.latents.shape == (4, 16, 2, 2)
    assert [s["skip_layers"] for s in (samples[0], samples[1])] == [(), (22,)]
    assert samples[2]["timestep"] < samples[0]["timestep"]
    # Rerunning on a finished capture is a no-op; a different prompt count is rejected.
    moss.capture(tmp_path, tmp_path / "source", records, files, "cpu")
    with pytest.raises(ValueError, match="different capture"):
        moss.capture(tmp_path, tmp_path / "source", records[:1], files[:1], "cpu")


def test_encode_prompts_uses_the_pinned_pipeline_without_a_transformer(tmp_path, monkeypatch):
    """Exercise the text-encoder-only MOSSImagePipeline path with tiny stand-ins."""
    import transformers
    from test_moss_replay_equivalence import moss_compat_pipeline
    compat = tmp_path / "compat/pipeline_moss_image.py"
    compat.parent.mkdir()
    compat.write_text(moss_compat_pipeline(tmp_path).read_text())

    class Inputs(dict):
        def to(self, device):
            return Inputs({k: v.to(device) for k, v in self.items()})

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<user>" + messages[0]["content"]

        def __call__(self, text, **kwargs):
            ids = torch.tensor([[ord(c) % 97 for c in text]])
            return Inputs(input_ids=ids, attention_mask=torch.ones_like(ids))

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(97, 8)

        @property
        def device(self):  # transformers models expose this; diffusers relies on it
            return self.embed.weight.device

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return transformers.modeling_outputs.BaseModelOutput(last_hidden_state=self.embed(input_ids))

    encoder = Encoder()
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: Tokenizer()))
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", classmethod(lambda cls, *a, **k: encoder))
    records = [{"prompt": "a cat", "seed": 1}, {"prompt": "a much longer dog", "seed": 2}]
    files = moss.encode_prompts(tmp_path, tmp_path / "source", records, "cpu")
    assert files == ["embeddings/calib-0000.pt", "embeddings/calib-0001.pt"]
    first, second = (torch.load(tmp_path / name, weights_only=True) for name in files)
    assert first["positive_embeds"].shape == (1, len("<user>a cat"), 8) and first["positive_mask"].all()
    assert second["positive_embeds"].shape[1] == len("<user>a much longer dog")
    assert torch.equal(first["negative_embeds"], second["negative_embeds"])
    assert first["negative_embeds"].shape[1] == len("<user>")
    assert not list(tmp_path.glob("embeddings/*.tmp"))


def test_quantize_defaults_smoothing_budget_and_rejects_a_changed_pipeline(tmp_path, monkeypatch):
    monkeypatch.delenv("DIFFUSE_COMPRESSOR_SMOOTH_CANDIDATE_BUDGET_GIB", raising=False)
    (tmp_path / "compat").mkdir()
    (tmp_path / "compat/pipeline_moss_image.py").write_text("")
    moss._write(tmp_path / "manifest.json", {"fingerprint": "f", "recipe": {"custom_pipeline_compatible_sha256": "old"}})
    with pytest.raises(ValueError, match="changed after prepare"):
        moss.quantize(tmp_path, "nvfp4", device="cpu")
    import os
    assert os.environ["DIFFUSE_COMPRESSOR_SMOOTH_CANDIDATE_BUDGET_GIB"] == "0.125"
