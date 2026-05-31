from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import torch

from .artifact import QuantizedArtifact, QuantizedTarget
from .config import CalibrationSpec, DiffusionQuantSpec, QuantizationCacheSpec, TargetConfig, target_quant_metadata
from .targets import QuantTarget


logger = logging.getLogger(__name__)


def resolve_quantization_cache(calibration: CalibrationSpec | None) -> QuantizationCacheSpec | None:
    """Resolve quantization artifact cache settings from calibration config.

    Args:
        calibration: Optional calibration settings.

    Returns:
        Cache settings, or ``None`` when artifact caching is disabled.
    """

    if calibration is None or calibration.artifact_cache is None:
        return None
    cache = calibration.artifact_cache
    if cache.cache_mode == "disabled":
        return None
    cache_dir = cache.cache_dir
    if cache_dir is None:
        if calibration.cache_dir is None:
            return None
        cache_dir = Path(calibration.cache_dir) / "artifacts"
    return QuantizationCacheSpec(cache_dir=cache_dir, cache_mode=cache.cache_mode, save_model=cache.save_model)


def load_quantization_cache(
    spec: DiffusionQuantSpec,
    target_config: TargetConfig | None,
    targets: list[QuantTarget],
    unquantized_state_dict: dict[str, torch.Tensor],
    calibration: CalibrationSpec | None,
) -> QuantizedArtifact | None:
    """Load a valid cached quantized artifact.

    Args:
        spec: Requested quantization settings.
        target_config: Target configuration used for the request.
        targets: Concrete targets selected for this request.
        unquantized_state_dict: Current unquantized model tensors.
        calibration: Calibration settings containing artifact cache options.

    Returns:
        Cached artifact, or ``None`` when no valid cache is available.
    """

    cache = resolve_quantization_cache(calibration)
    if cache is None or cache.cache_mode != "reuse":
        return None
    root = Path(cache.cache_dir)
    metadata_path = root / "metadata.json"
    model_path = root / "model.pt"
    if not metadata_path.exists() or not model_path.exists():
        logger.info("- Quantization artifact cache miss at %s", root)
        return None
    metadata = json.loads(metadata_path.read_text())
    expected_key = cache_key(spec, target_config, targets)
    if metadata.get("cache_key") != expected_key:
        logger.info("- Quantization artifact cache key mismatch at %s", root)
        return None
    logger.info("- Loading quantization artifact cache from %s", root)
    target_states = torch.load(model_path, map_location="cpu", weights_only=False)
    target_metadata = metadata.get("target_metadata", {})
    quantized_targets = []
    for target in targets:
        state = target_states.get(target.export_name)
        if state is None:
            logger.info("- Quantization artifact cache missing target %s", target.export_name)
            return None
        quantized_targets.append(
            QuantizedTarget(
                target=target,
                state_dict={key: value.cpu() for key, value in state.items()},
                metadata=target_metadata.get(target.export_name, {}),
            )
        )
    artifact_metadata = metadata.get("artifact_metadata", {})
    artifact_metadata["artifact_cache"] = {
        "cache_dir": str(root),
        "cache_mode": cache.cache_mode,
        "hit": True,
        "cache_key": expected_key,
    }
    return QuantizedArtifact(
        spec=spec,
        target_config=target_config,
        targets=targets,
        quantized_targets=quantized_targets,
        unquantized_state_dict=unquantized_state_dict,
        metadata=artifact_metadata,
    )


def load_target_quantization_caches(
    spec: DiffusionQuantSpec,
    target_config: TargetConfig | None,
    targets: list[QuantTarget],
    calibration: CalibrationSpec | None,
) -> dict[str, QuantizedTarget]:
    """Load reusable cached targets for an incomplete quantization run."""

    cache = resolve_quantization_cache(calibration)
    if cache is None or cache.cache_mode != "reuse":
        return {}
    root = Path(cache.cache_dir)
    target_root = root / "targets"
    if not target_root.exists():
        return {}
    expected_key = cache_key(spec, target_config, targets)
    cached: dict[str, QuantizedTarget] = {}
    for target in targets:
        path = _target_cache_path(root, target.export_name)
        if not path.exists():
            continue
        payload = _load_target_cache_payload(path)
        if payload is None:
            continue
        if payload.get("cache_key") != expected_key:
            logger.info("- Ignoring target cache with stale key for %s", target.export_name)
            continue
        if payload.get("export_name") != target.export_name:
            logger.info("- Ignoring target cache with mismatched export name for %s", target.export_name)
            continue
        state = payload.get("state_dict")
        metadata = payload.get("metadata", {})
        if (
            not isinstance(state, dict)
            or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items())
            or not isinstance(metadata, dict)
        ):
            logger.info("- Ignoring malformed target cache for %s", target.export_name)
            continue
        cached[target.export_name] = QuantizedTarget(
            target=target,
            state_dict={key: value.cpu() for key, value in state.items()},
            metadata=metadata,
        )
    if cached:
        logger.info("- Reusing %d/%d target artifact caches from %s", len(cached), len(targets), target_root)
    return cached


def save_target_quantization_cache(
    quantized: QuantizedTarget,
    spec: DiffusionQuantSpec,
    target_config: TargetConfig | None,
    targets: list[QuantTarget],
    calibration: CalibrationSpec | None,
) -> None:
    """Persist one completed quantized target for resume."""

    cache = resolve_quantization_cache(calibration)
    if cache is None:
        return
    root = Path(cache.cache_dir)
    target_root = root / "targets"
    target_root.mkdir(parents=True, exist_ok=True)
    path = _target_cache_path(root, quantized.target.export_name)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "cache_key": cache_key(spec, target_config, targets),
        "export_name": quantized.target.export_name,
        "state_dict": {key: value.cpu() for key, value in quantized.state_dict.items()},
        "metadata": quantized.metadata,
    }
    try:
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def save_quantization_cache(artifact: QuantizedArtifact, calibration: CalibrationSpec | None) -> None:
    """Persist quantized artifact tensors in DeepCompressor-style cache files.

    Args:
        artifact: Quantized artifact to store.
        calibration: Calibration settings containing artifact cache options.
    """

    cache = resolve_quantization_cache(calibration)
    if cache is None:
        return
    root = Path(cache.cache_dir)
    logger.info("- Saving quantization artifact cache to %s", root)
    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(artifact.spec, artifact.target_config, artifact.targets)
    artifact.metadata["artifact_cache"] = {
        "cache_dir": str(root),
        "cache_mode": cache.cache_mode,
        "hit": False,
        "cache_key": key,
    }
    target_states = {target.target.export_name: target.state_dict for target in artifact.quantized_targets}
    if cache.save_model:
        torch.save(target_states, root / "model.pt")
    torch.save(_select_suffixes(artifact, ("smooth_factor", "smooth_factor_orig")), root / "smooth.pt")
    torch.save(_select_suffixes(artifact, ("proj_down", "proj_up")), root / "branch.pt")
    torch.save(
        _select_suffixes(
            artifact,
            ("qweight", "wscales", "wcscales", "wtscale", "wzeros", "bias", "weight_range_scale", "weight_range_zero"),
        ),
        root / "wgts.pt",
    )
    torch.save(_select_prefixes(artifact, ("input_", "output_")), root / "acts.pt")
    torch.save(_select_suffixes(artifact, ("wscales", "wcscales", "wtscale", "weight_range_scale")), root / "scale.pt")
    metadata = {
        "cache_key": key,
        "artifact_metadata": artifact.metadata,
        "target_metadata": {target.target.export_name: target.metadata for target in artifact.quantized_targets},
        "files": ["smooth.pt", "branch.pt", "wgts.pt", "acts.pt", "scale.pt"] + (["model.pt"] if cache.save_model else []),
    }
    (root / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True))


def cache_key(spec: DiffusionQuantSpec, target_config: TargetConfig | None, targets: list[QuantTarget]) -> str:
    """Build a deterministic cache key for quantization artifacts.

    Args:
        spec: Quantization settings.
        target_config: Target configuration.
        targets: Concrete quantization targets.

    Returns:
        SHA256 digest for the cacheable request.
    """

    payload = {
        "spec": _jsonable(spec),
        "target_config": _jsonable(target_config),
        "targets": [
            {
                "name": target.name,
                "export_name": target.export_name,
                "modules": list(target.module_names),
                "roles": list(target.roles),
                "kind": target.kind,
                "quant": target_quant_metadata(target.quant),
            }
            for target in targets
        ],
    }
    blob = json.dumps(_jsonable(payload), sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()


def _target_cache_path(root: Path, export_name: str) -> Path:
    digest = hashlib.sha256(export_name.encode()).hexdigest()
    return root / "targets" / f"{digest}.pt"


def _load_target_cache_payload(path: Path) -> dict[str, Any] | None:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (EOFError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        logger.info("- Ignoring unreadable target cache %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _select_suffixes(artifact: QuantizedArtifact, suffixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Select cached tensors by exact target-local suffix.

    Args:
        artifact: Quantized artifact to inspect.
        suffixes: Target-local state dict suffixes to keep.

    Returns:
        Flat tensor mapping keyed by exported checkpoint name.
    """

    result = {}
    for quantized in artifact.quantized_targets:
        prefix = quantized.target.export_name
        for suffix, tensor in quantized.state_dict.items():
            if suffix in suffixes:
                result[f"{prefix}.{suffix}"] = tensor.cpu()
    return result


def _select_prefixes(artifact: QuantizedArtifact, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """Select cached tensors by target-local prefix.

    Args:
        artifact: Quantized artifact to inspect.
        prefixes: Target-local state dict prefixes to keep.

    Returns:
        Flat tensor mapping keyed by exported checkpoint name.
    """

    result = {}
    for quantized in artifact.quantized_targets:
        prefix = quantized.target.export_name
        for suffix, tensor in quantized.state_dict.items():
            if suffix.startswith(prefixes):
                result[f"{prefix}.{suffix}"] = tensor.cpu()
    return result


def _jsonable(value: Any) -> Any:
    """Convert a value into a JSON-stable representation.

    Args:
        value: Arbitrary Python value.

    Returns:
        JSON-compatible value.
    """

    if dataclasses.is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module is not None and qualname is not None:
            return f"{module}.{qualname}"
        return repr(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
