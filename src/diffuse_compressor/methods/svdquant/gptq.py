from __future__ import annotations

import math

import torch

from ...backends.nunchaku.layouts import fp4_e2m1_codebook, fp_quantize
from ...config import DiffusionQuantSpec
from ...logging import QuantizationLogger


@torch.inference_mode()
def gptq_residual_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    input_partitions: tuple[torch.Tensor, ...],
    spec: DiffusionQuantSpec,
    *,
    logger: QuantizationLogger | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Apply GPTQ column-error compensation to a residual weight matrix."""

    if not input_partitions:
        raise ValueError("GPTQ requires calibration input partitions")

    log = _resolve_logger(logger)
    cfg = spec.gptq
    work_weight = weight.float()
    out_features, in_features = work_weight.shape
    groups = scale.shape[2]
    if in_features % groups != 0:
        raise ValueError("GPTQ scale groups must divide the input feature count")
    group_size = in_features // groups
    work_scale = scale.float().view(out_features, groups)

    log.info(
        "    - Applying GPTQ residual rounding: block_size=%d, damp=%s, hessian_block_size=%d",
        cfg.block_size,
        cfg.damp_percentage,
        cfg.hessian_block_size,
    )

    hessian, num_samples = _hessian(input_partitions, in_features, cfg.hessian_block_size, work_weight.device)
    dead = hessian.diagonal() == 0
    hessian[dead, dead] = 1
    work_weight[:, dead] = 0

    importance = torch.diag(hessian)
    permute = torch.argsort(importance, descending=True)
    inverse_permute = torch.argsort(permute)
    hessian = hessian[permute][:, permute]
    work_weight = work_weight[:, permute]

    hessian_diag = hessian.diagonal()
    hessian_diag_mean = hessian_diag.mean()
    hessian_diag += cfg.damp_percentage * hessian_diag_mean

    hessian_inv = None
    num_inv_tries = 0
    for num_inv_tries in range(1, cfg.num_inv_tries + 1):
        try:
            chol = torch.linalg.cholesky(hessian)
            hessian_inv = torch.cholesky_inverse(chol)
            hessian_inv = torch.linalg.cholesky(hessian_inv, upper=True)
            break
        except RuntimeError:
            hessian_diag += (cfg.damp_percentage * 0.1) * hessian_diag_mean
    if hessian_inv is None:
        raise RuntimeError(f"GPTQ Hessian inversion failed after {cfg.num_inv_tries} attempts")
    if torch.isnan(hessian_inv).any() or torch.isinf(hessian_inv).any():
        raise RuntimeError("GPTQ Hessian inverse contains NaN or Inf")

    quantized = torch.empty_like(work_weight)
    for c_start in range(0, in_features, cfg.block_size):
        c_end = min(c_start + cfg.block_size, in_features)
        block_weight = work_weight[:, c_start:c_end].clone()
        block_quantized = quantized[:, c_start:c_end]
        block_hessian_inv = hessian_inv[c_start:c_end, c_start:c_end]
        block_error = torch.zeros_like(block_weight)
        for offset in range(c_end - c_start):
            column_index = c_start + offset
            column = block_weight[:, offset]
            original_column = int(permute[column_index].item())
            column_group = original_column // group_size
            column_scale = work_scale[:, column_group]
            qcolumn = _quantize_dequantize_column(
                column,
                column_scale,
                float_point=spec.precision == "fp4",
            )
            block_quantized[:, offset] = qcolumn
            pos_diag = block_hessian_inv[offset, offset].clamp_min(1e-12)
            column_error = (column - qcolumn) / pos_diag
            block_error[:, offset] = column_error
            block_weight[:, offset:] -= column_error.view(-1, 1) @ block_hessian_inv[offset, offset:].view(1, -1)
        if c_end < in_features:
            work_weight[:, c_end:] -= block_error @ hessian_inv[c_start:c_end, c_end:]

    quantized = quantized[:, inverse_permute].to(dtype=weight.dtype)
    if torch.isnan(quantized).any() or torch.isinf(quantized).any():
        raise RuntimeError("GPTQ quantized residual contains NaN or Inf")
    return quantized, {
        "enabled": True,
        "damp_percentage": cfg.damp_percentage,
        "block_size": cfg.block_size,
        "num_inv_tries": cfg.num_inv_tries,
        "hessian_block_size": cfg.hessian_block_size,
        "in_features": in_features,
        "out_features": out_features,
        "num_samples": num_samples,
        "inverse_tries": num_inv_tries,
    }


def _hessian(
    input_partitions: tuple[torch.Tensor, ...],
    in_features: int,
    hessian_block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    rows: list[torch.Tensor] = []
    num_samples = 0
    for partition in input_partitions:
        x = partition.reshape(-1, in_features)
        if x.numel() == 0:
            continue
        num_samples += x.shape[0]
        rows.append(x)
    if num_samples == 0:
        raise ValueError("GPTQ received no calibration rows")

    hessian = torch.zeros((in_features, in_features), device=device, dtype=torch.float32)
    factor = math.sqrt(2.0 / num_samples)
    for x in rows:
        x = x.to(device=device, dtype=torch.float32)
        if hessian_block_size > 0 and x.shape[0] > hessian_block_size:
            for start in range(0, x.shape[0], hessian_block_size):
                block = factor * x[start : start + hessian_block_size]
                hessian += block.t() @ block
        else:
            x = factor * x
            hessian += x.t() @ x
    return hessian, num_samples


def _quantize_dequantize_column(column: torch.Tensor, scale: torch.Tensor, *, float_point: bool) -> torch.Tensor:
    scaled = column / scale.clamp_min(1e-12)
    if float_point:
        codebook = fp4_e2m1_codebook(device=column.device, dtype=torch.float32)
        codes = fp_quantize(scaled.view(1, -1), codebook=codebook).view(-1)
        return codebook[codes.long()] * scale
    return scaled.round().clamp(-8, 7) * scale


def _resolve_logger(logger: QuantizationLogger | None) -> QuantizationLogger:
    if logger is None:
        return QuantizationLogger.get_logger(__name__)
    return logger.for_name(__name__)
