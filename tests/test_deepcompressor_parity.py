import os
import sys
import ast
from pathlib import Path

import pytest
import torch

from diffuse_compressor import SmoothSpec
from diffuse_compressor.methods.svdquant.quantize import (
    _apply_adanorm_awq_w4a16_layout,
    _low_rank_branch,
    _pack_awq_w4a16_weight,
)
from diffuse_compressor.methods.svdquant.smoothing import iter_smooth_candidates


DEEP_COMPRESSOR_ROOT = os.environ.get("DEEP_COMPRESSOR_ROOT")
pytestmark = pytest.mark.skipif(
    not DEEP_COMPRESSOR_ROOT,
    reason="set DEEP_COMPRESSOR_ROOT to run optional DeepCompressor parity tests",
)


def test_unweighted_low_rank_branch_matches_original_deepcompressor_lowrank_branch():
    assert DEEP_COMPRESSOR_ROOT is not None
    sys.path.insert(0, str(Path(DEEP_COMPRESSOR_ROOT)))
    try:
        from deepcompressor.nn.patch.lowrank import LowRankBranch
    finally:
        sys.path.remove(str(Path(DEEP_COMPRESSOR_ROOT)))

    torch.manual_seed(0)
    weight = torch.randn(7, 5, dtype=torch.float32)
    rank = 3

    down, up = _low_rank_branch(weight, rank=rank, inputs=None)
    original = LowRankBranch(in_features=weight.shape[1], out_features=weight.shape[0], rank=rank, weight=weight)

    actual_effective = up @ down
    expected_effective = original.get_effective_weight()
    assert expected_effective is not None
    assert torch.allclose(actual_effective, expected_effective, atol=1e-5, rtol=1e-5)


def test_weighted_svd_intentionally_differs_from_original_branch_initialization():
    assert DEEP_COMPRESSOR_ROOT is not None
    sys.path.insert(0, str(Path(DEEP_COMPRESSOR_ROOT)))
    try:
        from deepcompressor.nn.patch.lowrank import LowRankBranch
    finally:
        sys.path.remove(str(Path(DEEP_COMPRESSOR_ROOT)))

    torch.manual_seed(1)
    weight = torch.randn(7, 5, dtype=torch.float32)
    inputs = torch.randn(11, 5, dtype=torch.float32).mul(torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0]))
    rank = 3

    down, up = _low_rank_branch(weight, rank=rank, inputs=inputs)
    original = LowRankBranch(in_features=weight.shape[1], out_features=weight.shape[0], rank=rank, weight=weight)

    actual_effective = up @ down
    expected_effective = original.get_effective_weight()
    assert expected_effective is not None
    assert not torch.allclose(actual_effective, expected_effective, atol=1e-5, rtol=1e-5)


def test_smoothing_scale_matches_original_deepcompressor_get_smooth_scale():
    original_get_smooth_scale = _load_original_get_smooth_scale()
    inputs = torch.tensor([[1.0, 2.0, 8.0], [4.0, 1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([[2.0, 4.0, 16.0], [8.0, 2.0, 4.0]], dtype=torch.float32)
    spec = SmoothSpec(strategy="manual", alpha=0.25, beta=0.75, spans=(("absmax", "absmax"),))

    candidate = next(iter_smooth_candidates(inputs, weight, spec))
    alpha_base = inputs.abs().amax(dim=0).clamp_min(spec.eps)
    beta_base = weight.abs().amax(dim=0).clamp_min(spec.eps)
    expected = original_get_smooth_scale(alpha_base=alpha_base, beta_base=beta_base, alpha=0.25, beta=0.75)

    assert torch.allclose(candidate.scale, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("splits", [1, 3, 6])
def test_awq_w4a16_export_matches_original_deepcompressor_nunchaku_converter(splits):
    assert DEEP_COMPRESSOR_ROOT is not None
    sys.path.insert(0, str(Path(DEEP_COMPRESSOR_ROOT)))
    try:
        from deepcompressor.backend.nunchaku.utils import convert_to_nunchaku_w4x16_linear_weight
    finally:
        sys.path.remove(str(Path(DEEP_COMPRESSOR_ROOT)))

    torch.manual_seed(splits)
    weight = torch.randn(24, 1024, dtype=torch.bfloat16)
    bias = torch.randn(24, dtype=torch.bfloat16)
    scale = weight.float().view(24, 16, 64).abs().amax(dim=2).clamp_min(1e-6) / 7
    expected_qweight, expected_wscales, expected_wzeros, expected_bias = convert_to_nunchaku_w4x16_linear_weight(
        weight,
        scale=scale.to(weight.dtype).view(24, 1, 16, 1),
        bias=bias.clone(),
        adanorm_splits=splits,
    )

    actual_weight, actual_bias = (
        (weight, bias) if splits == 1 else _apply_adanorm_awq_w4a16_layout(weight, bias, splits=splits)
    )
    actual_qweight, actual_wscales, actual_wzeros = _pack_awq_w4a16_weight(actual_weight, group_size=64)

    assert torch.equal(actual_qweight, expected_qweight.cpu())
    assert torch.equal(actual_wscales, expected_wscales.cpu())
    assert torch.equal(actual_wzeros, expected_wzeros.cpu())
    assert torch.equal(actual_bias.cpu(), expected_bias.cpu())


def _load_original_get_smooth_scale():
    assert DEEP_COMPRESSOR_ROOT is not None
    source_path = Path(DEEP_COMPRESSOR_ROOT) / "deepcompressor" / "calib" / "smooth.py"
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_smooth_scale"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["get_smooth_scale"]
