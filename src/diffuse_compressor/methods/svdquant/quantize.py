from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from ...artifact import QuantizedTarget
from ...calibration import EvalReplayBatch
from ...config import DiffusionQuantSpec
from ...targets import QuantTarget
from .lowrank_search import search_low_rank_branch
from .smoothing import SmoothCandidate, iter_smooth_candidates, resolve_smooth_spec


@torch.inference_mode()
def quantize_targets(
    targets: Iterable[QuantTarget],
    spec: DiffusionQuantSpec,
    calibration_inputs: dict[str, torch.Tensor] | None = None,
    eval_replay: EvalReplayBatch | None = None,
) -> list[QuantizedTarget]:
    quantized: list[QuantizedTarget] = []
    for target in targets:
        if target.kind != "linear":
            raise NotImplementedError(f"SVDQuant currently supports linear targets only, got {target.kind!r}")
        inputs = calibration_inputs.get(target.export_name) if calibration_inputs is not None else None
        quantized.append(_quantize_linear_target(target, spec, inputs, eval_replay))
    return quantized


def _quantize_linear_target(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    calibration_inputs: torch.Tensor | None = None,
    eval_replay: EvalReplayBatch | None = None,
) -> QuantizedTarget:
    modules = [module for module in target.modules if isinstance(module, nn.Linear)]
    weight = torch.cat([module.weight.detach() for module in modules], dim=0)
    bias = _concat_bias(modules, weight.device, weight.dtype)
    export_dtype = torch.bfloat16 if weight.dtype not in (torch.float16, torch.bfloat16) else weight.dtype
    weight = weight.to(dtype=export_dtype)
    if bias is not None:
        bias = bias.to(dtype=export_dtype)

    smooth, smooth_metadata = _select_smooth_scale(target, spec, weight, bias, calibration_inputs)
    quant_inputs = _smooth_inputs(calibration_inputs, smooth) if calibration_inputs is not None else None
    smooth_weight = weight * smooth.view(1, -1)

    low_rank_metadata: dict[str, object] = {"mode": spec.low_rank_solver.mode}
    if spec.rank > 0 and target.shared_low_rank and spec.low_rank_solver.mode == "search":
        search = search_low_rank_branch(
            target=target,
            weight=smooth_weight,
            bias=bias,
            inputs=quant_inputs,
            spec=spec,
            eval_replay=eval_replay,
            low_rank_fn=_low_rank_branch,
            weight_scales_fn=_weight_scales,
            fake_quant_weight_fn=_fake_quantize_weight,
        )
        low_rank = search.low_rank
        quant_weight = search.residual
        low_rank_metadata = search.metadata
    else:
        low_rank = (
            _low_rank_branch(smooth_weight, rank=spec.rank, inputs=quant_inputs)
            if spec.rank > 0 and target.shared_low_rank
            else None
        )
        quant_weight = smooth_weight
        if low_rank is not None:
            quant_weight = smooth_weight - low_rank[1] @ low_rank[0]
        low_rank_metadata = {
            "mode": "weighted_svd",
            "iterations": 1 if low_rank is not None else 0,
            "eval_replay": False,
        }

    scale = _weight_scales(quant_weight, group_size=spec.group_size, float_point=spec.precision == "fp4")
    qweight, wscales = _pack_lite_linear_weight(quant_weight, scale, float_point=spec.precision == "fp4")
    state_dict = {
        "qweight": qweight,
        "wscales": wscales,
        "smooth_factor": smooth.detach().cpu(),
        "smooth_factor_orig": smooth.detach().cpu().clone(),
    }
    if bias is not None:
        state_dict["bias"] = bias.detach().cpu()
    if low_rank is not None:
        state_dict["proj_down"] = low_rank[0].t().contiguous().cpu()
        state_dict["proj_up"] = low_rank[1].contiguous().cpu()
    return QuantizedTarget(
        target=target,
        state_dict=state_dict,
        metadata={
            "source_modules": list(target.module_names),
            "roles": list(target.roles),
            "rank": spec.rank,
            "precision": spec.precision,
            "calibrated": calibration_inputs is not None,
            "low_rank_solver": low_rank_metadata,
            "smooth": smooth_metadata,
        },
    )


def _concat_bias(modules: list[nn.Linear], device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if all(module.bias is None for module in modules):
        return None
    return torch.cat(
        [
            module.bias.detach() if module.bias is not None else torch.zeros(module.out_features, device=device, dtype=dtype)
            for module in modules
        ],
        dim=0,
    )


def _weight_scales(weight: torch.Tensor, group_size: int, float_point: bool) -> torch.Tensor:
    oc, ic = weight.shape
    if ic % group_size != 0:
        raise ValueError(f"Input features ({ic}) must be divisible by group_size ({group_size}) for Nunchaku export")
    groups = ic // group_size
    max_q = 6 if float_point else 7
    scale = weight.float().view(oc, groups, group_size).abs().amax(dim=2).clamp_min(1e-6) / max_q
    return scale.to(dtype=weight.dtype).view(oc, 1, groups, 1)


def _pack_lite_linear_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    float_point: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    oc, ic = weight.shape
    groups = scale.shape[2]
    group_size = ic // groups
    scaled = weight.float().view(oc, groups, group_size) / scale.float().view(oc, groups, 1)
    if float_point:
        raise NotImplementedError("Nunchaku Lite FP4 packing is not implemented yet")
    qweight = scaled.round_().clamp_(-8, 7).to(torch.int16).view(oc, ic)
    lo = qweight[:, 0::2].bitwise_and(0xF)
    hi = qweight[:, 1::2].bitwise_and(0xF).bitwise_left_shift(4)
    packed = lo.bitwise_or(hi).to(torch.uint8).view(torch.int8).contiguous()
    wscales = scale.view(oc, groups).t().contiguous().cpu()
    return packed.cpu(), wscales


def _select_smooth_scale(
    target: QuantTarget,
    spec: DiffusionQuantSpec,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    calibration_inputs: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, object]]:
    smooth_spec = resolve_smooth_spec(spec.smooth)
    identity = torch.ones(weight.shape[1], dtype=weight.dtype, device=weight.device)
    if not smooth_spec.enabled:
        return identity, {"enabled": False, "searched": False, "reason": "disabled"}
    if calibration_inputs is None:
        return identity, {"enabled": True, "searched": False, "reason": "missing_calibration"}

    inputs = calibration_inputs.to(device=weight.device, dtype=torch.float32).reshape(-1, weight.shape[1])
    search_weight = weight.to(dtype=torch.float32)
    search_bias = None if bias is None else bias.to(device=weight.device, dtype=torch.float32)
    expected = _linear_output(inputs, search_weight, search_bias)
    best_error = torch.tensor(float("inf"), device=weight.device)
    best_scale = identity
    best_candidate: SmoothCandidate | None = None
    num_candidates = 0
    for candidate in iter_smooth_candidates(inputs, weight, smooth_spec):
        num_candidates += 1
        error = _candidate_output_error(
            candidate.scale.to(device=weight.device, dtype=weight.dtype),
            inputs,
            expected,
            search_weight,
            search_bias,
            spec,
            target.shared_low_rank,
        )
        if error < best_error:
            best_error = error
            best_scale = candidate.scale.to(device=weight.device, dtype=weight.dtype)
            best_candidate = candidate
    metadata: dict[str, object] = {
        "enabled": True,
        "searched": True,
        "strategy": smooth_spec.strategy,
        "num_candidates": num_candidates,
        "error": float(best_error.cpu()),
    }
    if best_candidate is not None:
        metadata.update(
            {
                "alpha": best_candidate.alpha,
                "beta": best_candidate.beta,
                "span": list(best_candidate.span),
            }
        )
    return best_scale, metadata


def _candidate_output_error(
    smooth: torch.Tensor,
    inputs: torch.Tensor,
    expected: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    spec: DiffusionQuantSpec,
    shared_low_rank: bool,
) -> torch.Tensor:
    smoothed_inputs = _smooth_inputs(inputs, smooth)
    smoothed_weight = weight * smooth.view(1, -1)
    low_rank = _low_rank_branch(smoothed_weight, rank=spec.rank, inputs=smoothed_inputs) if spec.rank > 0 and shared_low_rank else None
    residual = smoothed_weight
    if low_rank is not None:
        residual = smoothed_weight - low_rank[1] @ low_rank[0]
    scale = _weight_scales(residual, group_size=spec.group_size, float_point=spec.precision == "fp4")
    approx_residual = _fake_quantize_weight(residual, scale, float_point=spec.precision == "fp4")
    approx_weight = approx_residual
    if low_rank is not None:
        approx_weight = approx_weight + low_rank[1] @ low_rank[0]
    actual = _linear_output(smoothed_inputs, approx_weight, bias)
    return (actual.float() - expected.float()).pow(2).mean()


def _smooth_inputs(inputs: torch.Tensor | None, smooth: torch.Tensor) -> torch.Tensor | None:
    if inputs is None:
        return None
    return inputs / smooth.to(device=inputs.device, dtype=inputs.dtype).view(1, -1)


def _linear_output(inputs: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    output = inputs @ weight.t()
    if bias is not None:
        output = output + bias.view(1, -1)
    return output


def _fake_quantize_weight(weight: torch.Tensor, scale: torch.Tensor, float_point: bool) -> torch.Tensor:
    groups = scale.shape[2]
    group_size = weight.shape[1] // groups
    max_q = 6 if float_point else 7
    qweight = weight.float().view(weight.shape[0], groups, group_size) / scale.float().view(weight.shape[0], groups, 1)
    if float_point:
        raise NotImplementedError("Nunchaku Lite FP4 smoothing search is not implemented yet")
    qweight = qweight.round_().clamp_(-8, max_q)
    return (qweight * scale.float().view(weight.shape[0], groups, 1)).view_as(weight).to(dtype=weight.dtype)


def _low_rank_branch(
    weight: torch.Tensor,
    rank: int,
    inputs: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rank = min(rank, min(weight.shape))
    if rank == 0:
        return torch.empty(0, weight.shape[1], dtype=weight.dtype, device=weight.device), torch.empty(
            weight.shape[0], 0, dtype=weight.dtype, device=weight.device
        )
    svd_dtype = torch.float32
    if inputs is None:
        svd_dtype = torch.float64
        u, s, vh = torch.linalg.svd(weight.to(svd_dtype), full_matrices=False)
        down = vh[:rank].to(dtype=weight.dtype, device=weight.device)
    else:
        rms = inputs.to(device=weight.device, dtype=svd_dtype).pow(2).mean(dim=0).sqrt().clamp_min(1e-6)
        weighted = weight.to(svd_dtype) * rms.view(1, -1)
        u, s, vh = torch.linalg.svd(weighted, full_matrices=False)
        down = (vh[:rank] / rms.view(1, -1)).to(dtype=weight.dtype, device=weight.device)
    up = (u[:, :rank] * s[:rank]).to(dtype=weight.dtype, device=weight.device)
    return down, up
