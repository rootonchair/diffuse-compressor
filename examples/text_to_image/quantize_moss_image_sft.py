"""MOSS adapter using the existing SVDQuant/GPTQ scanner, quantizer and exporter."""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.util
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


def compatible_source(source: str) -> str:
    """Keep dense arithmetic unchanged while supporting native packed linear dtypes."""
    helper = '''def _linear_compute_dtype(layer):
    dtype = getattr(layer, 'torch_dtype', None)
    if dtype is not None:
        return dtype
    weight = getattr(layer, 'weight', None)
    if weight is not None:
        return weight.dtype
    return layer.wscales.dtype


'''
    if "def _linear_compute_dtype(" in source:
        return source
    assert source.count("class _RMSNorm(nn.Module):") == 1
    source = source.replace("class _RMSNorm(nn.Module):", helper + "class _RMSNorm(nn.Module):")
    for original, replacement in (
        ("self.proj['q'].weight.dtype", "_linear_compute_dtype(self.proj['q'])"),
        ("self.shared_block_modulation.weight.dtype", "_linear_compute_dtype(self.shared_block_modulation)"),
        ("self.video_head_modulation.weight.dtype", "_linear_compute_dtype(self.video_head_modulation)"),
    ):
        assert source.count(original) == 1, original
        source = source.replace(original, replacement)
    return source


def load_moss_module(path: Path, *, name="pipeline_moss_image"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MossReplaySamples:
    """Reference both CFG calls to one immutable FP32 pre-step latent on disk."""

    def __init__(self, cache_dir: Path):
        import numpy as np
        import torch
        self.directory = Path(cache_dir)
        self.metadata = json.loads((self.directory / "records.json").read_text())
        if not self.metadata.get("complete"):
            raise ValueError("Incomplete MOSS calibration capture")
        self.records = self.metadata["records"]
        self.latents = np.load(self.directory / "latents.npy", mmap_mode="c")
        self.embeddings = [torch.load(self.directory.parent / name, map_location="cpu", weights_only=True)
                           for name in self.metadata["embedding_files"]]
        expected = self.metadata["prompts"] * self.metadata["steps"] * 2
        if len(self.records) != expected or len(self.latents) != expected // 2:
            raise ValueError("Incomplete MOSS calibration record count")
        for index, record in enumerate(self.records):
            if record["latent_index"] != index // 2 or record["branch"] != ("positive" if index % 2 == 0 else "negative"):
                raise ValueError("MOSS calibration branch/latent pairing changed")
            if record["skip_layers"] != ([] if index % 2 == 0 else [22]):
                raise ValueError("MOSS negative-branch layer skipping changed")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import torch
        if index >= len(self):
            raise IndexError(index)
        record = self.records[index]
        embedding = self.embeddings[record["prompt_index"]]
        branch = record["branch"]
        return {"hidden_states": torch.from_numpy(self.latents[record["latent_index"]:record["latent_index"] + 1]),
                "timestep": torch.tensor(record["timestep"], dtype=getattr(torch, record["timestep_dtype"])).reshape(record["timestep_shape"]),
                "encoder_hidden_states": embedding[branch + "_embeds"],
                "encoder_attention_mask": embedding[branch + "_mask"],
                "skip_layers": tuple(record["skip_layers"])}


def moss_spec(precision, rank=32):
    from diffuse_compressor import GptqSpec
    from examples.text_to_image.utils import svdquant_spec
    return replace(svdquant_spec(precision, svd_backend="svd_lowrank", svd_lowrank_oversample=10,
                                svd_lowrank_niter=4, compute_device="cuda", offload_model=False),
                   rank=rank, gptq=GptqSpec(enabled=True, damp_percentage=.01, block_size=128,
                                         hessian_block_size=512, num_inv_tries=250))


def validate_calibration_recipe(samples, recipe):
    """Reject captures that do not cover the frozen prompt set and both CFG branches."""
    prompts = len(recipe["calibration_prompts"])
    steps = recipe["profile"]["steps"]
    expected = prompts * steps * 2
    if (prompts <= 0 or samples.metadata["prompts"] != prompts or samples.metadata["steps"] != steps
            or samples.metadata["profile"] != recipe["profile"]
            or len(samples) != expected or recipe["calibration_calls"] != expected):
        raise ValueError("Calibration capture differs from the frozen prompt count, sampling profile or CFG calls")
    return expected


def resident_replay_forward(model, sample):
    """Undo generic sequential offload before calling a model that fits on GPU.

    This runs outside Module.__call__, before its wrapped forward is selected.
    PyTorch capture/early-stop hooks remain installed; only Accelerate placement
    hooks are detached. All root records and quantization math are unchanged.
    """
    from diffuse_compressor.calibration.utils import has_accelerate_hooks, remove_accelerate_hooks
    if has_accelerate_hooks(model):
        remove_accelerate_hooks(model)
        # Earlier scoped replay can leave individual blocks on CPU even while
        # the model's first parameter is still on CUDA.
        model.to("cuda")
    elif next(model.parameters()).device.type != "cuda":
        model.to("cuda")
    return model(**sample)


def moss_previous_block_input(replay):
    args = list(replay.args)
    if not args:
        raise TypeError("MOSS block replay requires positional x")
    args[0] = replay.output
    return tuple(args), dict(replay.kwargs)


def moss_target_config(model, *, precision, rank=32):
    """Bounded chaining with root restarts across the negative skip boundary."""
    from diffuse_compressor import CalibrationScopeRule
    from examples.text_to_image.quantize_hf import scan_linear_targets
    scan = scan_linear_targets(model, precision=precision, rank=rank)
    # Rules are expanded in declaration order; enumerate the actual execution
    # order so boundary restarts never move layers 22/23 ahead of layers 1..21.
    scopes = []
    for prefix, blocks in (("text_embedder.blocks", model.text_embedder.blocks), ("layers", model.layers)):
        for index, block in enumerate(blocks):
            restart = index == 0 or (prefix == "layers" and index in (22, 23))
            scopes.append(CalibrationScopeRule(modules=(f"{prefix}.{index}",), module_classes=type(block),
                use_prev_scope_outputs=not restart,
                prev_replay_transform=None if restart else moss_previous_block_input))
    return replace(scan.target_config, calibration_scopes=scopes), scan


def run(run_dir, precision, preflight=False, rank=32, artifact_cache="reuse"):
    import torch
    from functools import partial
    from diffuse_compressor import CalibrationSpec, ExportSpec, LoggingConfig, QuantizationCacheSpec, TargetConfig, quantize_and_export
    from examples.text_to_image.quantize_hf import scan_linear_targets
    from examples.text_to_image.quantize_qwen_image_2_1 import configure_cuda_memory_limit
    torch.set_num_threads(8)
    torch.manual_seed(0)
    configure_cuda_memory_limit("cuda", 22)
    module = load_moss_module(run_dir / "compat/pipeline_moss_image.py")
    source = run_dir / ("preflight/dense" if preflight else "source/transformer")
    # The 9.5-GiB dense MOSS transformer fits inside the 22-GiB allocator cap.
    # Keep it resident during root replay, avoiding repeated PCIe weight transfers.
    model = module.MOSSImageTransformer.from_pretrained(source, torch_dtype=torch.bfloat16, local_files_only=True).eval().to("cuda")
    target_config, scan = moss_target_config(model, precision=precision, rank=rank)
    if preflight:
        wanted = {"layers.0.proj.q", "text_embedder.blocks.0.qkv", "shared_block_modulation", "video_head_modulation"}
        rules = [rule for rule in target_config.targets if rule.modules[0] in wanted]
        assert len(rules) == 4
        target_config = TargetConfig(targets=rules)
        generator = torch.Generator().manual_seed(7)
        samples = [{"hidden_states": torch.randn(1, 16, 2, 2, generator=generator),
                    "timestep": torch.tensor([1000 - i * 100.]),
                    "encoder_hidden_states": torch.randn(1, 5, 256, generator=generator).bfloat16(),
                    "encoder_attention_mask": torch.ones(1, 5, dtype=torch.bool)} for i in range(8)]
        output = run_dir / f"preflight/{precision}.safetensors"
        calibration = CalibrationSpec(samples=samples, cache_mode="disabled", max_rows_per_target=512,
                                      sample_batch_size=128, max_score_records=2, seed=0)
    else:
        assert len(scan.svdq_targets) == 262 and len(scan.awq_targets) == 2
        samples = MossReplaySamples(run_dir / "calibration")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        expected_calls = validate_calibration_recipe(samples, manifest["recipe"])
        output = run_dir / f"checkpoints/svdq-gptq-{precision}_r{rank}-moss-image-sft.safetensors"
        calibration = CalibrationSpec(samples=samples, num_samples=expected_calls, cache_mode="disabled", batch_size=1,
                                      max_rows_per_target=32768, sample_batch_size=4096,
                                      shared_input_keys=("skip_layers",), seed=0)
    calibration = replace(calibration, forward_fn=partial(resident_replay_forward, model),
                          keep_replay_on_device=True)
    if not preflight:
        manifest = json.loads((run_dir / "manifest.json").read_text())
        assert manifest["recipe"]["rank"] == rank, "Rank differs from the frozen recipe"
        declared_cache = manifest["recipe"].get("target_artifact_cache", {}).get("mode", "reuse")
        assert declared_cache == artifact_cache, "Artifact cache differs from the frozen recipe"
        fingerprint = manifest["fingerprint"]
        calibration = replace(calibration, artifact_cache=QuantizationCacheSpec(
            cache_dir=run_dir / "target-artifacts" / fingerprint / precision,
            cache_mode=artifact_cache, save_model=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    print(scan.format_text(), flush=True)
    original_parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()} if preflight else None
    quantize_and_export(model, moss_spec(precision, rank), target_config, calibration,
                        ExportSpec(output=output), LoggingConfig(name=output.stem))
    if original_parameters is not None:
        assert all(torch.equal(p.detach().cpu(), original_parameters[n]) for n, p in model.named_parameters()), "Quantization changed teacher parameters"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("nvfp4", "int4"), required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--rank", type=int, choices=(32, 64, 128), default=32)
    parser.add_argument("--artifact-cache", choices=("reuse", "disabled"), default="reuse",
                        help="Disable optional per-target disk recovery artifacts when disk space is limited.")
    args = parser.parse_args()
    run(args.run_dir, args.precision, args.preflight, args.rank, args.artifact_cache)
