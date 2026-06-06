import os
import sys
import ast
from pathlib import Path

import pytest
import torch

from diffuse_compressor import SmoothSpec
from diffuse_compressor.backends.nunchaku.layouts import (
    apply_adanorm_awq_w4a16_layout,
    nvfp4_scale_leaves,
    pack_awq_w4a16_weight,
    pack_nunchaku_w4a4_state,
)
from diffuse_compressor.methods.svdquant.factorization import low_rank_branch
from diffuse_compressor.methods.svdquant.smoothing import (
    ManualSmoothSearchStrategy,
    SmoothEvaluation,
    build_smooth_span_contexts,
)


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

    down, up = low_rank_branch(weight, rank=rank, inputs=None)
    original = LowRankBranch(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        rank=rank,
        weight=weight,
    )

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
    inputs = torch.randn(11, 5, dtype=torch.float32).mul(
        torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0])
    )
    rank = 3

    down, up = low_rank_branch(weight, rank=rank, inputs=inputs)
    original = LowRankBranch(
        in_features=weight.shape[1],
        out_features=weight.shape[0],
        rank=rank,
        weight=weight,
    )

    actual_effective = up @ down
    expected_effective = original.get_effective_weight()
    assert expected_effective is not None
    assert not torch.allclose(
        actual_effective, expected_effective, atol=1e-5, rtol=1e-5
    )


def test_smoothing_scale_matches_original_deepcompressor_get_smooth_scale():
    original_get_smooth_scale = _load_original_get_smooth_scale()
    inputs = torch.tensor([[1.0, 2.0, 8.0], [4.0, 1.0, 2.0]], dtype=torch.float32)
    weight = torch.tensor([[2.0, 4.0, 16.0], [8.0, 2.0, 4.0]], dtype=torch.float32)
    spec = SmoothSpec(
        strategy="manual", alpha=0.25, beta=0.75, spans=(("absmax", "absmax"),)
    )
    evaluated = []

    def evaluate_candidates(candidates):
        evaluated.extend(candidates)
        return tuple(
            SmoothEvaluation(candidate=candidate, error=torch.tensor(float(index)))
            for index, candidate in enumerate(candidates)
        )

    ManualSmoothSearchStrategy().search(
        spec, build_smooth_span_contexts(inputs, weight, spec), evaluate_candidates
    )
    candidate = evaluated[0]
    alpha_base = inputs.abs().amax(dim=0).clamp_min(spec.eps)
    beta_base = weight.abs().amax(dim=0).clamp_min(spec.eps)
    expected = original_get_smooth_scale(
        alpha_base=alpha_base, beta_base=beta_base, alpha=0.25, beta=0.75
    )

    assert torch.allclose(candidate.scale, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("splits", [1, 3, 6])
def test_awq_w4a16_export_matches_original_deepcompressor_nunchaku_converter(splits):
    assert DEEP_COMPRESSOR_ROOT is not None
    sys.path.insert(0, str(Path(DEEP_COMPRESSOR_ROOT)))
    try:
        from deepcompressor.backend.nunchaku.utils import (
            convert_to_nunchaku_w4x16_linear_weight,
        )
    finally:
        sys.path.remove(str(Path(DEEP_COMPRESSOR_ROOT)))

    torch.manual_seed(splits)
    weight = torch.randn(24, 1024, dtype=torch.bfloat16)
    bias = torch.randn(24, dtype=torch.bfloat16)
    scale = weight.float().view(24, 16, 64).abs().amax(dim=2).clamp_min(1e-6) / 7
    expected_qweight, expected_wscales, expected_wzeros, expected_bias = (
        convert_to_nunchaku_w4x16_linear_weight(
            weight,
            scale=scale.to(weight.dtype).view(24, 1, 16, 1),
            bias=bias.clone(),
            adanorm_splits=splits,
        )
    )

    actual_weight, actual_bias = (
        (weight, bias)
        if splits == 1
        else apply_adanorm_awq_w4a16_layout(weight, bias, splits=splits)
    )
    actual_qweight, actual_wscales, actual_wzeros = pack_awq_w4a16_weight(
        actual_weight, group_size=64
    )

    assert torch.equal(actual_qweight, expected_qweight.cpu())
    assert torch.equal(actual_wscales, expected_wscales.cpu())
    assert torch.equal(actual_wzeros, expected_wzeros.cpu())
    assert torch.equal(actual_bias.cpu(), expected_bias.cpu())


def test_nvfp4_svdq_export_matches_original_deepcompressor_nunchaku_converter():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required for NVFP4 scale parity")
    assert DEEP_COMPRESSOR_ROOT is not None
    sys.path.insert(0, str(Path(DEEP_COMPRESSOR_ROOT)))
    try:
        from deepcompressor.backend.nunchaku.utils import (
            convert_to_nunchaku_w4x4y16_linear_weight,
        )
    finally:
        sys.path.remove(str(Path(DEEP_COMPRESSOR_ROOT)))

    torch.manual_seed(0)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    bias = torch.randn(128, dtype=torch.bfloat16)
    smooth = torch.rand(128, dtype=torch.bfloat16).add_(0.5)
    low_rank = (
        torch.randn(16, 128, dtype=torch.bfloat16),
        torch.randn(128, 16, dtype=torch.bfloat16),
    )
    effective_scale = (
        weight.float().view(128, 8, 16).abs().amax(dim=2).clamp_min(1e-6) / 6
    )
    scale, subscale = nvfp4_scale_leaves(
        weight, effective_scale.to(weight.dtype).view(128, 1, 8, 1)
    )

    (
        expected_weight,
        expected_scale,
        expected_bias,
        expected_smooth,
        expected_lora,
        expected_subscale,
    ) = convert_to_nunchaku_w4x4y16_linear_weight(
        weight,
        scale=scale,
        bias=bias,
        smooth=smooth,
        lora=(
            (low_rank[0].to(torch.float64) / smooth.to(torch.float64).view(1, -1)).to(
                weight.dtype
            ),
            low_rank[1],
        ),
        float_point=True,
        subscale=subscale,
    )
    actual = pack_nunchaku_w4a4_state(
        weight,
        scale,
        smooth,
        bias,
        low_rank,
        float_point=True,
        subscale=subscale,
    )

    assert expected_lora is not None
    assert expected_subscale is not None
    assert torch.equal(actual["qweight"], expected_weight.cpu())
    assert torch.equal(actual["wtscale"], expected_scale.cpu())
    assert torch.equal(actual["bias"], expected_bias.cpu())
    assert torch.equal(actual["smooth_factor"], expected_smooth.cpu())
    assert torch.equal(actual["proj_down"], expected_lora[0].cpu())
    assert torch.equal(actual["proj_up"], expected_lora[1].cpu())
    assert torch.equal(actual["wscales"], expected_subscale.cpu())


def test_shifted_nvfp4_svdq_export_matches_original_deepcompressor_nunchaku_converter():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("torch.float8_e4m3fn is required for NVFP4 scale parity")
    assert DEEP_COMPRESSOR_ROOT is not None
    sys.path.insert(0, str(Path(DEEP_COMPRESSOR_ROOT)))
    try:
        from deepcompressor.backend.nunchaku.convert import (
            convert_to_nunchaku_w4x4y16_linear_state_dict,
        )
    finally:
        sys.path.remove(str(Path(DEEP_COMPRESSOR_ROOT)))

    torch.manual_seed(0)
    weight = torch.randn(128, 128, dtype=torch.bfloat16)
    bias = torch.randn(128, dtype=torch.bfloat16)
    smooth = torch.rand(128, dtype=torch.bfloat16).add_(0.5)
    shift = torch.full((128,), 0.25, dtype=torch.bfloat16)
    low_rank = (
        torch.randn(16, 128, dtype=torch.bfloat16),
        torch.randn(128, 16, dtype=torch.bfloat16),
    )
    effective_scale = (
        weight.float().view(128, 8, 16).abs().amax(dim=2).clamp_min(1e-6) / 6
    )
    scale, subscale = nvfp4_scale_leaves(
        weight, effective_scale.to(weight.dtype).view(128, 1, 8, 1)
    )

    expected = convert_to_nunchaku_w4x4y16_linear_state_dict(
        weight,
        scale=scale,
        bias=bias,
        smooth=smooth,
        lora=low_rank,
        shift=shift,
        float_point=True,
        subscale=subscale,
    )
    actual = pack_nunchaku_w4a4_state(
        weight,
        scale,
        smooth,
        bias,
        low_rank,
        float_point=True,
        subscale=subscale,
        shift=shift,
    )

    assert torch.equal(actual["qweight"], expected["qweight"].cpu())
    assert torch.equal(actual["wtscale"], expected["wtscale"].cpu())
    assert torch.equal(actual["bias"], expected["bias"].cpu())
    assert torch.equal(actual["smooth_factor"], expected["smooth"].cpu())
    assert torch.equal(actual["smooth_factor_orig"], expected["smooth_orig"].cpu())
    assert torch.equal(actual["proj_down"], expected["lora_down"].cpu())
    assert torch.equal(actual["proj_up"], expected["lora_up"].cpu())
    assert torch.equal(actual["wscales"], expected["wscales"].cpu())


def _load_original_get_smooth_scale():
    assert DEEP_COMPRESSOR_ROOT is not None
    source_path = Path(DEEP_COMPRESSOR_ROOT) / "deepcompressor" / "calib" / "smooth.py"
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_smooth_scale"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["get_smooth_scale"]
