"""Quantize MOSS-Image SFT to Nunchaku Lite SVDQuant W4A4 with GPTQ.

This is the recipe behind ``lite-infer/moss-image-sft-nunchaku-lite-{nvfp4,int4}-r32``.
It runs in two stages that share one run directory:

``prepare``
    Download the pinned MOSS-Image-SFT revision, write a copy of its custom
    pipeline that also accepts packed linear layers (``compat/``), encode the
    calibration prompts, then sample each prompt at 1024x1024 with the default
    profile (50 steps, CFG 3, layer 22 skipped on the negative branch) and record
    every transformer call. Both guidance calls of a step share one FP32 latent
    in ``calibration/latents.npy``; ``calibration/records.json`` describes the
    calls and ``manifest.json`` freezes the prompt set and sampling profile.

``quantize``
    Replay the recorded calls block by block (restarting from the root around
    the skipped layer 22, so both CFG branches stay exact) and export a Nunchaku
    Lite checkpoint to ``checkpoints/``. The dense transformer (about 9.5 GiB)
    stays resident on the GPU during replay.

Example::

    python examples/text_to_image/quantize_moss_image_sft.py prepare \\
        --run-dir outputs/moss-image-sft --prompt-file examples/prompts/moss_sft_calibration8.json
    python examples/text_to_image/quantize_moss_image_sft.py quantize \\
        --run-dir outputs/moss-image-sft --precision nvfp4

Resources: ``prepare`` downloads about 27 GB of source weights, then holds the
bf16 text encoder (about 17 GB) and later the transformer on the GPU; the
captured latents take 0.84 GB for 8 prompts (3.4 GB for 32). Resuming a
finished ``prepare`` does not touch the network.

The published checkpoints also used a 0.125 GiB smoothing candidate budget
(``DIFFUSE_COMPRESSOR_SMOOTH_CANDIDATE_BUDGET_GIB``); ``quantize`` sets it unless
the variable is already set. Low-rank SVD draws unseeded CUDA random numbers
per scoring batch, so the budget changes which smoothing scale may win.
Captures made with different torch/transformers versions are equivalent but
not guaranteed bit-identical.

``quantize --preflight`` runs the same code path on a tiny random MOSS
transformer (four targets, eight random calls) and only needs ``compat/``.

The exported transformer still needs the patched ``compat/pipeline_moss_image.py``
at inference time, because upstream reads ``.weight.dtype`` from layers that
Nunchaku packs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

MODEL_ID = "OpenMOSS-Team/MOSS-Image-SFT"
REVISION = "f972bd4b4b9e3852133cc5da954d87bc32cd8fb9"
PROFILE = {"steps": 50, "guidance_scale": 3.0, "shift": 5.0, "negative_prompt": "",
           "uncond_skip_layers": [22], "slg_scale": 0.0}
RESOLUTION = 1024
SMOOTH_CANDIDATE_BUDGET_GIB = "0.125"
EXPECTED_TARGETS = {"svdq": 262, "awq": 2}
SOURCE_PATTERNS = ("model_index.json", "pipeline_moss_image.py", "LICENSE", "README.md",
                   "transformer/*", "text_encoder/*", "tokenizer/*", "scheduler/*")


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
    replacements = (
        ("class _RMSNorm(nn.Module):", helper + "class _RMSNorm(nn.Module):"),
        ("self.proj['q'].weight.dtype", "_linear_compute_dtype(self.proj['q'])"),
        ("self.shared_block_modulation.weight.dtype", "_linear_compute_dtype(self.shared_block_modulation)"),
        ("self.video_head_modulation.weight.dtype", "_linear_compute_dtype(self.video_head_modulation)"),
    )
    for original, replacement in replacements:
        if source.count(original) != 1:
            raise ValueError(f"Unexpected MOSS pipeline source; expected exactly one {original!r}")
        source = source.replace(original, replacement)
    return source


def load_moss_module(path: Path, *, name="pipeline_moss_image"):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _read(path: Path):
    return json.loads(Path(path).read_text())


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


class MossReplaySamples:
    """Reference both CFG calls to one immutable FP32 pre-step latent on disk."""

    def __init__(self, cache_dir: Path):
        import numpy as np
        import torch
        self.directory = Path(cache_dir)
        self.metadata = _read(self.directory / "records.json")
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


# --------------------------------------------------------------------------- prepare


def load_prompt_records(path: Path) -> list[dict]:
    """Read ``{"records": [{"filename", "prompt", "seed"}, ...]}``."""
    data = _read(path)
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} must contain a non-empty 'records' list")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str) or not isinstance(record.get("seed"), int):
            raise ValueError(f"Every record in {path} needs a string 'prompt' and an integer 'seed'")
    if len({r["prompt"] for r in records}) != len(records):
        raise ValueError(f"{path} contains duplicate prompts")
    return records


def build_recipe(records: list[dict], compat_sha256: str) -> dict:
    return {"source": MODEL_ID, "revision": REVISION, "profile": PROFILE, "resolution": RESOLUTION,
            "calibration_prompts": records, "calibration_calls": len(records) * PROFILE["steps"] * 2,
            "custom_pipeline_compatible_sha256": compat_sha256}


def fingerprint(recipe: dict) -> str:
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def call_kwargs(seed: int, embedding: dict) -> dict:
    import torch
    return {"height": RESOLUTION, "width": RESOLUTION, "num_inference_steps": PROFILE["steps"],
            "guidance_scale": PROFILE["guidance_scale"], "negative_prompt": PROFILE["negative_prompt"],
            "uncond_skip_layers": PROFILE["uncond_skip_layers"], "slg_scale": PROFILE["slg_scale"],
            "generator": torch.Generator("cpu").manual_seed(seed),
            "prompt_embeds": embedding["positive_embeds"], "prompt_attention_mask": embedding["positive_mask"],
            "negative_prompt_embeds": embedding["negative_embeds"],
            "negative_prompt_attention_mask": embedding["negative_mask"]}


class CallRecorder:
    """Forward pre-hook that stores both CFG calls of every step of one prompt."""

    def __init__(self, latents, prompt_index: int, steps: int):
        self.latents = latents
        self.prompt_index = prompt_index
        self.steps = steps
        self.records: list[dict] = []

    def __call__(self, module, args, kwargs):
        import numpy as np
        call = len(self.records)
        if call >= 2 * self.steps:
            raise RuntimeError("More transformer calls than expected for one prompt")
        slot = self.prompt_index * self.steps + call // 2
        value = args[0].detach().float().cpu().numpy()
        if value.shape != (1, *self.latents.shape[1:]) or not np.isfinite(value).all():
            raise ValueError(f"Unexpected calibration latent with shape {value.shape}")
        positive = call % 2 == 0
        if positive:
            self.latents[slot] = value[0]
        elif not np.array_equal(self.latents[slot], value[0]):
            raise ValueError("The negative CFG call saw a different latent than the positive call")
        skipped = list(kwargs.get("skip_layers") or ())
        if skipped != ([] if positive else PROFILE["uncond_skip_layers"]):
            raise ValueError(f"Unexpected skip_layers {skipped} on call {call}")
        timestep = args[1].detach().cpu()
        self.records.append({"prompt_index": self.prompt_index, "latent_index": slot,
                             "branch": "positive" if positive else "negative", "skip_layers": skipped,
                             "timestep": timestep.tolist(), "timestep_dtype": str(timestep.dtype).split(".")[-1],
                             "timestep_shape": list(timestep.shape)})


def download_source(run_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download
    source = run_dir / "source"
    upstream = Path(hf_hub_download(MODEL_ID, "pipeline_moss_image.py", revision=REVISION, local_dir=source))
    compat = run_dir / "compat/pipeline_moss_image.py"
    patched = compatible_source(upstream.read_text())
    if compat.exists() and compat.read_text() != patched:
        raise ValueError(f"{compat} differs from the patched pinned pipeline; use a fresh --run-dir")
    compat.parent.mkdir(parents=True, exist_ok=True)
    compat.write_text(patched)
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=source, allow_patterns=list(SOURCE_PATTERNS))
    return source


def encode_prompts(run_dir: Path, source: Path, records: list[dict], device: str) -> list[str]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    files = [f"embeddings/calib-{index:04d}.pt" for index in range(len(records))]
    if all((run_dir / name).exists() for name in files):
        return files
    module = load_moss_module(run_dir / "compat/pipeline_moss_image.py")
    tokenizer = AutoTokenizer.from_pretrained(source / "tokenizer", local_files_only=True)
    encoder = AutoModel.from_pretrained(source / "text_encoder", dtype=torch.bfloat16,
                                        local_files_only=True).eval().to(device)
    pipe = module.MOSSImagePipeline(None, module.MOSSImageScheduler(), text_encoder=encoder, tokenizer=tokenizer)
    negative, negative_mask = pipe.encode_prompt(PROFILE["negative_prompt"])
    for name, record in zip(files, records):
        path = run_dir / name
        if path.exists():
            continue
        positive, mask = pipe.encode_prompt(record["prompt"])
        value = {"positive_embeds": positive.cpu(), "positive_mask": mask.cpu(),
                 "negative_embeds": negative.cpu(), "negative_mask": negative_mask.cpu()}
        if not (torch.isfinite(value["positive_embeds"]).all() and torch.isfinite(value["negative_embeds"]).all()):
            raise ValueError(f"Non-finite text embedding for {record['prompt']!r}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.tmp")
        torch.save(value, temporary)
        temporary.replace(path)
    del pipe, encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return files


def capture(run_dir: Path, source: Path, records: list[dict], embedding_files: list[str], device: str) -> None:
    import numpy as np
    import torch
    directory = run_dir / "calibration"
    if (directory / "records.json").exists():
        if len(MossReplaySamples(directory)) != len(records) * PROFILE["steps"] * 2:
            raise ValueError(f"{directory} holds a different capture; use a fresh --run-dir")
        return
    module = load_moss_module(run_dir / "compat/pipeline_moss_image.py")
    transformer = module.MOSSImageTransformer.from_pretrained(source / "transformer", torch_dtype=torch.bfloat16,
                                                              local_files_only=True).eval()
    pipe = module.MOSSImagePipeline(transformer, module.MOSSImageScheduler(shift=PROFILE["shift"]))
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    steps = PROFILE["steps"]
    latent_size = RESOLUTION // 16
    directory.mkdir(parents=True, exist_ok=True)
    latents = np.lib.format.open_memmap(directory / "latents.npy", mode="w+", dtype=np.float32,
        shape=(len(records) * steps, transformer.config.latent_channels, latent_size, latent_size))
    calls = []
    for index, (record, name) in enumerate(zip(records, embedding_files)):
        embedding = torch.load(run_dir / name, map_location="cpu", weights_only=True)
        recorder = CallRecorder(latents, index, steps)
        handle = transformer.register_forward_pre_hook(recorder, with_kwargs=True)
        try:
            result = pipe(**call_kwargs(record["seed"], embedding), output_type="latent").images
        finally:
            handle.remove()
        if not torch.isfinite(result).all() or len(recorder.records) != 2 * steps:
            raise RuntimeError(f"Calibration sampling failed for prompt {index}")
        calls.extend(recorder.records)
        latents.flush()
        logging.info("captured prompt %d/%d", index + 1, len(records))
    _write(directory / "records.json", {"complete": True, "prompts": len(records), "steps": steps, "profile": PROFILE,
           "records": calls, "embedding_files": embedding_files,
           "latent_file_sha256": _sha256(directory / "latents.npy"), "source_revision": REVISION})


def prepare(run_dir: Path, prompt_file: Path, device: str = "cuda", cuda_memory_limit_gib: float | None = None) -> dict:
    """Build (or resume) a run directory and return its manifest."""
    from examples.text_to_image.quantize_qwen_image_2_1 import configure_cuda_memory_limit
    run_dir = Path(run_dir)
    records = load_prompt_records(prompt_file)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = _read(manifest_path)
        if manifest["recipe"]["calibration_prompts"] != records:
            raise ValueError(f"{run_dir} was prepared with different prompts; use a fresh --run-dir")
        if (run_dir / "calibration/records.json").exists():
            validate_calibration_recipe(MossReplaySamples(run_dir / "calibration"), manifest["recipe"])
            return manifest
    configure_cuda_memory_limit(device, cuda_memory_limit_gib)
    source = download_source(run_dir)
    recipe = build_recipe(records, _sha256(run_dir / "compat/pipeline_moss_image.py"))
    manifest = {"fingerprint": fingerprint(recipe), "recipe": recipe}
    if manifest_path.exists() and _read(manifest_path)["fingerprint"] != manifest["fingerprint"]:
        raise ValueError(f"{manifest_path} differs from this recipe; use a fresh --run-dir")
    _write(manifest_path, manifest)
    embedding_files = encode_prompts(run_dir, source, records, device)
    capture(run_dir, source, records, embedding_files, device)
    return manifest


# --------------------------------------------------------------------------- quantize


def moss_spec(precision, rank=32, device="cuda"):
    from diffuse_compressor import GptqSpec
    from examples.text_to_image.utils import svdquant_spec
    return replace(svdquant_spec(precision, svd_backend="svd_lowrank", svd_lowrank_oversample=10,
                                 svd_lowrank_niter=4, compute_device=device, offload_model=False),
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


def resident_replay_forward(model, sample, device="cuda"):
    """Undo generic sequential offload before calling a model that fits on the GPU.

    This runs outside Module.__call__, before its wrapped forward is selected.
    PyTorch capture/early-stop hooks remain installed; only Accelerate placement
    hooks are detached. All root records and quantization math are unchanged.
    """
    import torch
    from diffuse_compressor.calibration.utils import has_accelerate_hooks, remove_accelerate_hooks
    if has_accelerate_hooks(model):
        remove_accelerate_hooks(model)
        # Earlier scoped replay can leave individual blocks on the CPU even while
        # the model's first parameter is still on the GPU.
        model.to(device)
    elif next(model.parameters()).device != torch.device(device):
        model.to(device)
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


def preflight_model_and_samples(module, device):
    """Tiny random transformer, four targets and eight random calls."""
    import torch
    from diffuse_compressor import TargetConfig
    config = dict(dim=128, ffn_dim=256, num_heads=8, num_layers=2, text_dim=256, freq_dim=32, latent_channels=16)
    model = module.MOSSImageTransformer(**config).bfloat16().eval().to(device)
    generator = torch.Generator().manual_seed(7)
    samples = [{"hidden_states": torch.randn(1, 16, 2, 2, generator=generator),
                "timestep": torch.tensor([1000 - i * 100.]),
                "encoder_hidden_states": torch.randn(1, 5, 256, generator=generator).bfloat16(),
                "encoder_attention_mask": torch.ones(1, 5, dtype=torch.bool)} for i in range(8)]
    wanted = {"layers.0.proj.q", "text_embedder.blocks.0.qkv", "shared_block_modulation", "video_head_modulation"}
    return model, samples, lambda config: TargetConfig(targets=[r for r in config.targets if r.modules[0] in wanted])


def quantize(run_dir: Path, precision: str, *, rank: int = 32, preflight: bool = False, artifact_cache: str = "reuse",
             device: str = "cuda", cuda_memory_limit_gib: float | None = None) -> Path:
    import torch
    from functools import partial
    from diffuse_compressor import CalibrationSpec, ExportSpec, LoggingConfig, QuantizationCacheSpec, quantize_and_export
    from examples.text_to_image.quantize_qwen_image_2_1 import configure_cuda_memory_limit
    run_dir = Path(run_dir)
    compat = run_dir / "compat/pipeline_moss_image.py"
    if not compat.exists():
        raise FileNotFoundError(f"{compat} is missing; run the prepare stage first")
    os.environ.setdefault("DIFFUSE_COMPRESSOR_SMOOTH_CANDIDATE_BUDGET_GIB", SMOOTH_CANDIDATE_BUDGET_GIB)
    torch.set_num_threads(8)
    torch.manual_seed(0)
    configure_cuda_memory_limit(device, cuda_memory_limit_gib)
    module = load_moss_module(compat)
    if preflight:
        model, samples, select = preflight_model_and_samples(module, device)
        target_config, scan = moss_target_config(model, precision=precision, rank=rank)
        target_config = select(target_config)
        output = run_dir / f"preflight/{precision}_r{rank}.safetensors"
        calibration = CalibrationSpec(samples=samples, cache_mode="disabled", max_rows_per_target=512,
                                      sample_batch_size=128, max_score_records=2, seed=0)
    else:
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"{manifest_path} is missing; run the prepare stage first")
        manifest = _read(manifest_path)
        if _sha256(compat) != manifest["recipe"]["custom_pipeline_compatible_sha256"]:
            raise ValueError(f"{compat} changed after prepare; use a fresh --run-dir")
        samples = MossReplaySamples(run_dir / "calibration")
        expected_calls = validate_calibration_recipe(samples, manifest["recipe"])
        model = module.MOSSImageTransformer.from_pretrained(run_dir / "source/transformer", torch_dtype=torch.bfloat16,
                                                            local_files_only=True).eval().to(device)
        target_config, scan = moss_target_config(model, precision=precision, rank=rank)
        found = {"svdq": len(scan.svdq_targets), "awq": len(scan.awq_targets)}
        if found != EXPECTED_TARGETS:
            raise ValueError(f"Expected {EXPECTED_TARGETS} quantization targets in MOSS-Image SFT, found {found}")
        output = run_dir / f"checkpoints/svdq-gptq-{precision}_r{rank}-moss-image-sft.safetensors"
        calibration = CalibrationSpec(samples=samples, num_samples=expected_calls, cache_mode="disabled", batch_size=1,
                                      max_rows_per_target=32768, sample_batch_size=4096,
                                      shared_input_keys=("skip_layers",), seed=0,
                                      artifact_cache=QuantizationCacheSpec(
                                          cache_dir=run_dir / "target-artifacts" / manifest["fingerprint"] / f"{precision}_r{rank}",
                                          cache_mode=artifact_cache, save_model=False))
    calibration = replace(calibration, forward_fn=partial(resident_replay_forward, model, device=device),
                          keep_replay_on_device=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    print(scan.format_text(), flush=True)
    original_parameters = {n: p.detach().cpu().clone() for n, p in model.named_parameters()} if preflight else None
    quantize_and_export(model, moss_spec(precision, rank, device), target_config, calibration,
                        ExportSpec(output=output), LoggingConfig(name=output.stem))
    if original_parameters is not None and not all(
            torch.equal(p.detach().cpu(), original_parameters[n]) for n, p in model.named_parameters()):
        raise RuntimeError("Quantization changed the teacher parameters")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "quantize"):
        stage = stages.add_parser(name)
        stage.add_argument("--run-dir", type=Path, required=True)
        stage.add_argument("--device", default="cuda")
        stage.add_argument("--cuda-memory-limit-gib", type=float,
                           help="Cap this process's CUDA allocator, e.g. when sharing the GPU.")
        if name == "prepare":
            stage.add_argument("--prompt-file", type=Path, default=ROOT / "examples/prompts/moss_sft_calibration8.json",
                               help='JSON with a "records" list of {"filename", "prompt", "seed"}.')
        else:
            stage.add_argument("--precision", choices=("nvfp4", "int4"), required=True)
            stage.add_argument("--rank", type=int, choices=(32, 64, 128), default=32)
            stage.add_argument("--preflight", action="store_true",
                               help="Quantize a tiny random transformer to check the toolchain.")
            stage.add_argument("--artifact-cache", choices=("reuse", "disabled"), default="reuse",
                               help="Disable per-target recovery artifacts when disk space is limited.")
    return parser


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    if args.stage == "prepare":
        prepare(args.run_dir, args.prompt_file, args.device, args.cuda_memory_limit_gib)
    else:
        quantize(args.run_dir, args.precision, rank=args.rank, preflight=args.preflight,
                 artifact_cache=args.artifact_cache, device=args.device,
                 cuda_memory_limit_gib=args.cuda_memory_limit_gib)


if __name__ == "__main__":
    main()
