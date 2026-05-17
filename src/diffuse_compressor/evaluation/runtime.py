from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .core import EvaluationSpec


def patch_quantized_pipeline(pipe: Any, *, model_key: str, spec: "EvaluationSpec") -> Any:
    """Patch a pipeline with an exported quantized checkpoint."""

    if spec.runtime == "none":
        return pipe
    if spec.runtime != "nunchaku-lite":
        raise RuntimeError(f"Unsupported quantized runtime: {spec.runtime!r}")
    if spec.checkpoint is None:
        raise RuntimeError("runtime='nunchaku-lite' requires EvaluationSpec.checkpoint")

    patch_transformer = _load_nunchaku_lite_patch_transformer()
    target = _nunchaku_lite_target(model_key)
    if not hasattr(pipe, "transformer"):
        raise RuntimeError("nunchaku-lite evaluation requires the pipeline to expose a transformer")
    patch_transformer(
        pipe.transformer,
        spec.checkpoint,
        target=target,
        precision=spec.precision,
        torch_dtype=spec.torch_dtype or torch.bfloat16,
    )
    return pipe


def _load_nunchaku_lite_patch_transformer():
    try:
        from nunchaku_lite import patch_transformer
    except ImportError as exc:
        raise RuntimeError("runtime='nunchaku-lite' requires the optional nunchaku_lite package") from exc
    return patch_transformer


def _nunchaku_lite_target(model_key: str) -> str:
    normalized = model_key.lower()
    if normalized.startswith("flux2") or "flux2" in normalized:
        return "flux2"
    if normalized.startswith("flux.1") or normalized.startswith("flux1") or normalized.startswith("flux"):
        return "flux"
    raise RuntimeError(f"nunchaku-lite evaluation does not support model_key={model_key!r}")
