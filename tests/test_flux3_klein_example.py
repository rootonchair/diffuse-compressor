"""The FLUX.3 [klein] script must target every block linear and calibrate the reference-image blocks.

Records without reference images never reach ``mode_blocks.image_cond``; root
replay with early stop must still calibrate those blocks from the records that
do, and leave the text-to-image calls untouched.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

transformer_flux3 = pytest.importorskip("diffusers.models.transformers.transformer_flux3")

from examples.text_to_image import quantize_flux3_klein as flux3_example  # noqa: E402

BLOCK_LINEARS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.linear_in", "ff.linear_out")


def _tiny_model():
    torch.manual_seed(0)
    return transformer_flux3.Flux3Transformer2DModel(
        in_channels=64,
        cond_in_channels=64,
        num_layers=2,
        num_single_layers=2,
        attention_head_dim=128,
        num_attention_heads=2,
        joint_attention_dim=256,
        axes_dims_rope=(32, 32, 32, 32),
        flow_map=True,
    )


def _block_names():
    return (*(f"{stack}.{index}" for stack in ("mode_blocks.image", "mode_blocks.image_cond", "txt_mode_blocks")
              for index in range(2)), "single_transformer_blocks.0", "single_transformer_blocks.1")


def test_targets_cover_every_block_linear_and_keep_outer_layers_dense():
    model = _tiny_model()
    config, scan = flux3_example.flux3_target_config(model, precision="int4", rank=32)

    assert set(scan.svdq_targets) == {f"{block}.{name}" for block in _block_names() for name in BLOCK_LINEARS}
    assert scan.awq_targets == ()
    skipped = {name for name, _ in scan.skipped}
    assert {"context_embedder", "proj_out", "norm_out.linear", "x_embedders.image",
            "single_stream_modulations.image.linear", "early_stream_modulations.txt.linear"} <= skipped
    assert [scope.modules for scope in config.calibration_scopes] == [(name,) for name in _block_names()]
    assert all(scope.use_prev_scope_outputs is False for scope in config.calibration_scopes)


def test_image_cond_blocks_can_be_skipped():
    _, scan = flux3_example.flux3_target_config(
        _tiny_model(), precision="int4", rank=32, skip=(flux3_example.IMAGE_COND_BLOCKS,)
    )
    assert not any(name.startswith("mode_blocks.image_cond") for name in scan.svdq_targets)


def test_json_records_resolve_reference_images(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs/a.png").write_bytes(b"")
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"records": [
        {"filename": "t2i", "prompt": "a cat", "seed": 1},
        {"filename": "edit", "prompt": "make it red", "seed": 2, "images": ["refs/a.png"]},
    ]}))

    records = flux3_example.load_calibration_records(path, -1)
    assert records[0]["images"] == []
    assert records[1]["images"] == [str((tmp_path / "refs/a.png").resolve())]
    assert len(flux3_example.load_calibration_records(path, 1)) == 1

    (tmp_path / "refs/a.png").unlink()
    with pytest.raises(FileNotFoundError):
        flux3_example.load_calibration_records(path, -1)


def test_forward_passes_embeddings_geometry_and_references(tmp_path):
    from PIL import Image

    Image.new("RGB", (32, 32), "red").save(tmp_path / "ref.png")
    calls = []
    pipe = lambda **kwargs: calls.append(kwargs) or ["image"]  # noqa: E731
    embeds = {"edit": torch.ones(1, 80, 8)}
    args = SimpleNamespace(device="cpu", height=1024, width=1024, steps=16)
    forward = flux3_example.flux3_forward_fn(pipe, embeds, args)

    forward({"filename": "edit", "prompt": "unused", "seed": 3, "width": 768, "images": [str(tmp_path / "ref.png")]})
    kwargs = calls[0]
    assert torch.equal(kwargs["prompt_embeds"], embeds["edit"])
    assert (kwargs["width"], kwargs["height"], kwargs["num_inference_steps"]) == (768, 1024, 16)
    assert kwargs["output"] == "images" and "prompt" not in kwargs
    assert kwargs["image"][0].size == (32, 32)
    assert "guidance_scale" not in kwargs


def test_upsampler_is_removed_from_the_blocks():
    blocks = flux3_example.flux3_blocks()
    assert list(blocks.sub_blocks) == ["text_encoder", "vae_encoder", "denoise", "decode"]
    assert list(flux3_example.flux3_blocks(keep=("text_encoder",)).sub_blocks) == ["text_encoder"]


def test_run_refuses_calibration_without_reference_images(tmp_path, monkeypatch):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"records": [{"filename": "t2i", "prompt": "a cat", "seed": 1}]}))
    monkeypatch.setattr("sys.argv", ["quantize_flux3_klein.py", "--prompt-file", str(path)])
    with pytest.raises(ValueError, match="image_cond"):
        flux3_example.run()



def test_specification_file_and_duplicate_filenames_are_rejected(tmp_path):
    spec = Path(__file__).resolve().parents[1] / "examples/prompts/flux3_klein_calibration.json"
    with pytest.raises(ValueError, match="no 'records' list"):
        flux3_example.load_calibration_records(spec, -1)

    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"records": [{"filename": "a", "prompt": "x", "seed": 1},
                                            {"filename": "a", "prompt": "y", "seed": 2}]}))
    with pytest.raises(ValueError, match="unique"):
        flux3_example.load_calibration_records(path, -1)


@pytest.mark.parametrize("offload", [False, True])
def test_cached_run_passes_seed_and_keeps_offload_on_cpu(tmp_path, monkeypatch, offload):
    import diffusers

    model = _tiny_model()
    moves = []
    monkeypatch.setattr(model, "to", lambda *a, **kw: moves.append(a) or model)
    monkeypatch.setattr(diffusers, "Flux3Transformer2DModel", SimpleNamespace(from_pretrained=lambda *a, **kw: model))
    (tmp_path / "cache/inputs/caches").mkdir(parents=True)
    (tmp_path / "cache/inputs/caches/0.pt").write_bytes(b"")
    captured = {}
    monkeypatch.setattr(flux3_example, "quantize_and_export", lambda m, spec, targets, calibration, *a: captured.update(
        calibration=calibration))
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"records": [{"filename": "t2i", "prompt": "a cat", "seed": 1}]}))
    monkeypatch.setattr("sys.argv", ["quantize_flux3_klein.py", "--device", "cpu", "--precision", "int4",
        "--prompt-file", str(path), "--skip", flux3_example.IMAGE_COND_BLOCKS, "--seed", "7",
        "--cache-dir", str(tmp_path / "cache"), "--output", str(tmp_path / "w.safetensors"),
        *(["--offload-model"] if offload else [])])
    flux3_example.run()

    assert captured["calibration"].seed == 7
    assert moves == ([] if offload else [("cpu",)])

def _root_call(model, *, with_reference: bool, seed: int):
    generator = torch.Generator().manual_seed(seed)
    side, text = 4, 8
    ids = torch.stack(torch.meshgrid(torch.zeros(1), torch.arange(side * 1.0), torch.arange(side * 1.0), torch.zeros(1),
                                     indexing="ij"), dim=-1).reshape(-1, 4)
    call = {
        "hidden_states": torch.randn(1, side * side, 64, generator=generator, dtype=torch.bfloat16),
        "encoder_hidden_states": torch.randn(1, text, 256, generator=generator, dtype=torch.bfloat16),
        "timestep": torch.tensor([0.5], dtype=torch.bfloat16),
        "target_timestep": torch.tensor([0.25], dtype=torch.bfloat16),
        "img_ids": ids,
        "txt_ids": torch.zeros(text, 4),
        "return_dict": False,
    }
    if with_reference:
        call["image_cond_hidden_states"] = torch.randn(1, side * side, 64, generator=generator, dtype=torch.bfloat16)
        call["image_cond_ids"] = ids + torch.tensor([1.0, 0, 0, 0])
    return call


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SVDQuant export needs CUDA")
def test_tiny_quantization_calibrates_image_cond_blocks_from_mixed_records(tmp_path):
    from diffuse_compressor import CalibrationSpec, ExportSpec, LoggingConfig, quantize_and_export
    from diffuse_compressor.calibration.data import prepare_calibration_cache
    from safetensors import safe_open

    model = _tiny_model().to(torch.bfloat16)
    for name, module in model.named_modules():  # what from_pretrained(torch_dtype=bf16) keeps in FP32
        if name.endswith(("norm_q", "norm_k")):
            module.float()
    config, scan = flux3_example.flux3_target_config(model, precision="int4", rank=32)
    samples = [_root_call(model, with_reference=index % 2 == 1, seed=index) for index in range(4)]
    calibration = CalibrationSpec(samples=samples, num_samples=4, cache_num_samples=-1, batch_size=1,
                                  cache_dir=tmp_path / "inputs", cache_mode="reuse", max_rows_per_target=4096)
    assert len(prepare_calibration_cache(model, calibration)) == 4

    args = SimpleNamespace(precision="int4", svd_backend="svd_lowrank", svd_lowrank_oversample=10,
                           svd_lowrank_niter=4, compute_device="cuda", device="cuda", offload_model=False,
                           rank=32, gptq=True, gptq_damp_percentage=0.01, gptq_block_size=128,
                           gptq_num_inv_tries=250, gptq_hessian_block_size=512)
    output = tmp_path / "flux3.safetensors"
    quantize_and_export(model, flux3_example.qwen_image_21_spec(args), config, calibration,
                        ExportSpec(output=output), LoggingConfig(enabled=False))

    with safe_open(output, framework="pt") as handle:
        keys = set(handle.keys())
        manifest = json.loads(handle.metadata()["quantization_config"])["runtime_manifest"]
    prefixes = {target["checkpoint_prefix"] for target in manifest["targets"]}
    assert prefixes == set(scan.svdq_targets)
    for prefix in ("mode_blocks.image_cond.0.attn.to_q", "single_transformer_blocks.1.ff.linear_out"):
        assert {f"{prefix}.{name}" for name in ("qweight", "wscales", "smooth_factor", "proj_down", "proj_up")} <= keys
        assert torch.isfinite(safe_open(output, framework="pt").get_tensor(f"{prefix}.smooth_factor").float()).all()
    assert "context_embedder.weight" in keys and "single_stream_modulations.image.linear.weight" in keys
