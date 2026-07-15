"""Package a Nunchaku Lite pipeline, optionally serializing selected text encoders as BNB4."""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from huggingface_hub import list_repo_files, snapshot_download
from safetensors import safe_open


_DENSE_TRANSFORMER_PATTERNS = (
    "diffusion_pytorch_model*.bin",
    "diffusion_pytorch_model*.safetensors",
    "model*.safetensors",
    "pytorch_model*.bin",
)


def build_diffusers_quantization_config(
    checkpoint: str | Path, *, compute_dtype: str = "bfloat16"
) -> dict[str, Any]:
    """Convert an embedded runtime manifest to Diffusers' compact Nunchaku config."""
    checkpoint = Path(checkpoint)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
    raw_config = metadata.get("quantization_config")
    if raw_config is None:
        raise ValueError("Checkpoint does not contain quantization_config metadata")
    try:
        manifest = json.loads(raw_config).get("runtime_manifest")
    except json.JSONDecodeError as exc:
        raise ValueError("Checkpoint quantization_config metadata is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Checkpoint does not contain quantization_config.runtime_manifest")
    if manifest.get("schema") != "nunchaku_lite.runtime_manifest" or manifest.get("version") != 1:
        raise ValueError("Converter requires nunchaku_lite.runtime_manifest version 1")
    if manifest.get("component") != "transformer":
        raise ValueError("Diffusers packaging requires runtime_manifest component='transformer'")
    if manifest.get("structural_patches"):
        raise ValueError("Diffusers Nunchaku loading does not support runtime manifest structural patches")
    targets = manifest.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Runtime manifest must contain at least one target")

    grouped: dict[str, list[dict[str, Any]]] = {"svdq_w4a4": [], "awq_w4a16": []}
    seen: set[str] = set()
    for target in targets:
        if not isinstance(target, dict):
            raise ValueError("Runtime manifest targets must be JSON objects")
        op = target.get("nunchaku_op")
        if op not in grouped:
            raise ValueError(f"Diffusers Nunchaku loading does not support operation {op!r}")
        prefix = target.get("checkpoint_prefix")
        sources = target.get("source_modules")
        if not isinstance(prefix, str) or sources != [prefix]:
            raise ValueError(f"Manifest target {prefix!r} must map one source module to the same checkpoint prefix")
        if prefix in seen:
            raise ValueError(f"Duplicate runtime manifest target {prefix!r}")
        seen.add(prefix)
        grouped[op].append(target)

        required = {"qweight", "wscales"}
        if op == "svdq_w4a4":
            required.update({"smooth_factor", "proj_down", "proj_up"})
            if target.get("precision") == "fp4":
                required.update({"wcscales", "wtscale"})
        else:
            required.add("wzeros")
        if target.get("has_bias"):
            required.add("bias")
        missing = sorted(name for name in required if f"{prefix}.{name}" not in keys)
        if missing:
            raise ValueError(f"Checkpoint target {prefix!r} is missing required tensors: {missing}")

    output: dict[str, Any] = {"quant_method": "nunchaku_lite", "compute_dtype": compute_dtype}
    for op, entries in grouped.items():
        if not entries:
            continue
        settings = {(entry.get("precision"), entry.get("group_size"), entry.get("rank")) for entry in entries}
        if len(settings) != 1:
            raise ValueError(f"Diffusers compact config requires uniform settings for {op} targets")
        precision, group_size, rank = next(iter(settings))
        if op == "svdq_w4a4":
            if precision not in {"int4", "fp4"}:
                raise ValueError(f"Unsupported SVDQ precision {precision!r}")
            output[op] = {
                "precision": "nvfp4" if precision == "fp4" else precision,
                "group_size": group_size,
                "rank": rank,
                "targets": [entry["checkpoint_prefix"] for entry in entries],
            }
        else:
            if precision != "int4" or group_size != 64:
                raise ValueError("Diffusers AWQ targets require INT4 precision and group size 64")
            output[op] = {
                "precision": "int4",
                "group_size": 64,
                "targets": [entry["checkpoint_prefix"] for entry in entries],
            }
    return output


def quantize_text_encoder_components(pipeline_dir: str | Path, components: Sequence[str]) -> tuple[str, ...]:
    """Replace selected Transformers text encoders with serialized BNB4 NF4 models."""

    selected = tuple(dict.fromkeys(components))
    if not selected:
        return ()

    pipeline_dir = Path(pipeline_dir)
    model_index_path = pipeline_dir / "model_index.json"
    if not model_index_path.is_file():
        raise FileNotFoundError("Base pipeline does not contain model_index.json")
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))

    declarations: list[tuple[str, str]] = []
    for component in selected:
        if not component.startswith("text_encoder"):
            raise ValueError(f"BNB4 component {component!r} must be a text_encoder component")
        declaration = model_index.get(component)
        if not isinstance(declaration, list) or len(declaration) != 2:
            raise ValueError(f"Pipeline does not declare text encoder component {component!r}")
        library, class_name = declaration
        if library != "transformers" or not isinstance(class_name, str):
            raise ValueError(f"Text encoder component {component!r} must be provided by Transformers")
        component_dir = pipeline_dir / component
        if not (component_dir / "config.json").is_file():
            raise FileNotFoundError(f"Text encoder component {component!r} does not contain config.json")
        declarations.append((component, class_name))

    try:
        import torch
        import transformers
    except ImportError as exc:
        raise ImportError(
            "BNB4 text encoder conversion requires torch, transformers, accelerate, and bitsandbytes"
        ) from exc

    for component, class_name in declarations:
        model_class = getattr(transformers, class_name, None)
        if model_class is None or not hasattr(model_class, "from_pretrained"):
            raise ValueError(f"Transformers does not expose loadable class {class_name!r} for {component!r}")
        quantization_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        component_dir = pipeline_dir / component
        replacement_dir = pipeline_dir / f".{component}.bnb4"
        shutil.rmtree(replacement_dir, ignore_errors=True)
        model = None
        try:
            model = model_class.from_pretrained(
                component_dir,
                quantization_config=quantization_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True,
            )
            model.save_pretrained(replacement_dir, safe_serialization=True)
            shutil.rmtree(component_dir)
            replacement_dir.rename(component_dir)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            shutil.rmtree(replacement_dir, ignore_errors=True)
    return selected


def package_diffusers_pipeline(
    checkpoint: str | Path,
    model_id: str | Path,
    output_dir: str | Path,
    *,
    revision: str | None = None,
    compute_dtype: str = "bfloat16",
    bnb4_text_encoders: Sequence[str] = (),
) -> Path:
    """Create a complete Diffusers pipeline with Nunchaku Lite transformer weights."""
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    quantization_config = build_diffusers_quantization_config(checkpoint, compute_dtype=compute_dtype)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        source = Path(model_id).expanduser()
        if source.is_dir():
            shutil.copytree(source, temporary, dirs_exist_ok=True)
        else:
            ignore_patterns = [f"transformer/{pattern}" for pattern in _DENSE_TRANSFORMER_PATTERNS]
            ignore_patterns.extend(
                filename
                for filename in list_repo_files(str(model_id), revision=revision)
                if "/" not in filename and Path(filename).suffix in {".bin", ".ckpt", ".safetensors"}
            )
            snapshot_download(
                repo_id=str(model_id), revision=revision, local_dir=temporary, ignore_patterns=ignore_patterns
            )

        transformer_dir = temporary / "transformer"
        transformer_config_path = transformer_dir / "config.json"
        if not transformer_config_path.is_file():
            raise FileNotFoundError("Base pipeline does not contain transformer/config.json")
        for pattern in _DENSE_TRANSFORMER_PATTERNS:
            for dense_weight in transformer_dir.glob(pattern):
                dense_weight.unlink()
        for suffix in ("*.bin", "*.ckpt", "*.safetensors"):
            for single_file_weight in temporary.glob(suffix):
                single_file_weight.unlink()

        transformer_config = json.loads(transformer_config_path.read_text(encoding="utf-8"))
        transformer_config["quantization_config"] = quantization_config
        transformer_config_path.write_text(
            json.dumps(transformer_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(checkpoint, transformer_dir / "diffusion_pytorch_model.safetensors")
        quantize_text_encoder_components(temporary, bnb4_text_encoders)
        temporary.rename(output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-id", required=True, help="Hugging Face model id or local Diffusers pipeline")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--compute-dtype", default="bfloat16")
    parser.add_argument(
        "--bnb4-text-encoder",
        action="append",
        default=[],
        metavar="COMPONENT",
        help="Transformers text encoder component to serialize as BNB4 NF4; repeat for multiple encoders",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    output = package_diffusers_pipeline(
        args.checkpoint,
        args.model_id,
        args.output_dir,
        revision=args.revision,
        compute_dtype=args.compute_dtype,
        bnb4_text_encoders=args.bnb4_text_encoder,
    )
    print(f"Packaged Nunchaku Lite Diffusers pipeline at {output}")


if __name__ == "__main__":
    main()
