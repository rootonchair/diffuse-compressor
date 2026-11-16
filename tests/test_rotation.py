import json
from dataclasses import replace

import pytest
import safetensors
import torch
from torch import nn

import diffuse_compressor.methods.svdquant.factorization as factorization_module
import diffuse_compressor.methods.svdquant.lowrank_search as lowrank_search_module
from diffuse_compressor import (
    ActivationQuantSpec,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    GptqSpec,
    LowRankSolverSpec,
    NaiveSvdqLayout,
    RotationSpec,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    collect_quant_targets,
    quantize_and_export,
    quantize_diffusion,
)
from diffuse_compressor.artifact_cache import cache_key
from diffuse_compressor.backends.nunchaku.layouts import (
    fake_quantize_weight,
    quantize_to_codes,
    weight_scales,
)
from diffuse_compressor.backends.nunchaku.packing import (
    dynamic_fake_quantize_activation,
)
from diffuse_compressor.methods.svdquant.gptq import _hessian
from diffuse_compressor.methods.svdquant.rotation import (
    IDENTITY,
    BlockHadamardRotation,
    hadamard_matrix,
    input_rotation_from_metadata,
    resolve_input_rotation,
)

HADAMARD = BlockHadamardRotation(64)
FLAG = {"type": "block_matrix", "block_size": 64, "shared": True, "generator": "hadamard"}


class Proj(nn.Module):
    def __init__(self, in_features=128, out_features=128):
        super().__init__()
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.proj(x)


def _blockdiag_hadamard(in_features):
    return torch.block_diag(*[hadamard_matrix(64)] * (in_features // 64))


def _fwht_butterfly(x):
    """In-place style 64-point FWHT in ``x.dtype``, as the kernel computes it."""

    x = x.reshape(*x.shape[:-1], -1, 64).clone()
    h = 1
    while h < 64:
        x = x.reshape(*x.shape[:-1], 64 // (2 * h), 2, h)
        a, b = x[..., 0, :], x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape[:-3], 64)
        h *= 2
    return (x * 0.125).reshape(*x.shape[:-2], -1)


def test_hadamard_matches_scipy_sylvester_and_is_orthonormal_and_symmetric():
    scipy_linalg = pytest.importorskip("scipy.linalg")
    matrix = hadamard_matrix(64)
    expected = torch.from_numpy(scipy_linalg.hadamard(64)).float() / 8
    assert torch.equal(matrix, expected)
    assert torch.equal(matrix, matrix.t())
    torch.testing.assert_close(matrix @ matrix.t(), torch.eye(64), rtol=0, atol=1e-6)
    # +-1/8 is exact in half precision, and the matrix is cached per device/dtype.
    assert torch.equal(hadamard_matrix(64, dtype=torch.bfloat16).float(), expected)
    assert hadamard_matrix(64) is hadamard_matrix(64)


def test_block_rotation_matches_blockdiag_and_kernel_butterfly():
    torch.manual_seed(0)
    x = torch.randn(5, 192)
    torch.testing.assert_close(
        HADAMARD.rotate_activations(x), x @ _blockdiag_hadamard(192)
    )
    # The kernel's fp16 butterfly differs from the fp32 reference by rounding only.
    kernel = _fwht_butterfly(x.half()).float()
    torch.testing.assert_close(
        kernel, HADAMARD.rotate_activations(x), rtol=1e-2, atol=1e-2
    )
    torch.testing.assert_close(
        HADAMARD.rotate_input_dim(HADAMARD.rotate_input_dim(x)), x
    )


def test_rotated_residual_linear_matches_original_without_quantization():
    torch.manual_seed(0)
    n, k, r = 48, 128, 8
    x, weight = torch.randn(16, k), torch.randn(n, k)
    smooth = torch.rand(k) + 0.5
    down, up = torch.randn(r, k), torch.randn(n, r)
    x_s = x / smooth
    residual = weight * smooth - up @ down

    # Kernel order: LoRA-down on the unrotated x/s, residual GEMM in the rotated basis.
    rotated = HADAMARD.rotate_activations(x_s) @ HADAMARD.rotate_input_dim(residual).t()
    actual = rotated + x_s @ down.t() @ up.t()
    torch.testing.assert_close(actual, x @ weight.t(), rtol=1e-4, atol=1e-4)


def test_rotation_reduces_int4_w4a4_error_with_outlier_channels():
    # INT4 with 64-wide groups: an outlier channel inflates its whole group's
    # scale; H64 spreads it across the block. Ratio is 0.66-0.75 over 20 seeds.
    # (NVFP4's 16-wide groups already isolate outliers; there H64 can hurt.)
    torch.manual_seed(0)
    x, weight = torch.randn(256, 256), torch.randn(128, 256) / 16
    x[:, [3, 70, 130, 200]] *= 50  # one outlier channel per 64-wide group
    expected = x @ weight.t()

    def w4a4_error(rotation):
        def quantize_weight(w):
            return fake_quantize_weight(w, weight_scales(w, 64, False), False)

        def quantize_input(t):
            return dynamic_fake_quantize_activation(t, group_size=64, float_point=False)

        qx = rotation.quantize_in_basis(x, quantize_input)
        qw = rotation.quantize_in_basis(weight, quantize_weight)
        return ((qx @ qw.t() - expected).norm() / expected.norm()).item()

    plain, rotated = w4a4_error(IDENTITY), w4a4_error(HADAMARD)
    assert rotated < 0.85 * plain, (plain, rotated)


def test_identity_rotation_is_a_passthrough():
    x = torch.randn(4, 64)
    fn = torch.tanh
    assert IDENTITY.rotate_input_dim(x) is x
    assert IDENTITY.rotate_activations(x) is x
    assert IDENTITY.wrap(fn) is fn
    assert IDENTITY.metadata() is None
    assert resolve_input_rotation(None, in_features=96) is IDENTITY
    assert input_rotation_from_metadata(None) is IDENTITY
    # A spec without rotation keys caches exactly as before the field existed.
    targets = collect_quant_targets(
        Proj(), TargetConfig(targets=[TargetRule(modules=["proj"])])
    )
    spec = DiffusionQuantSpec()
    assert cache_key(spec, None, targets) != cache_key(
        replace(spec, rotation=RotationSpec()), None, targets
    )


def _search(spec, inputs, weight, targets, input_rotation=None):
    kwargs = {} if input_rotation is None else {"input_rotation": input_rotation}
    torch.manual_seed(1)  # torch.svd_lowrank is randomized
    return lowrank_search_module.search_low_rank_branch(
        target=targets[0],
        weight=weight,
        bias=None,
        inputs=inputs,
        input_partitions=(inputs,),
        spec=spec,
        eval_replay=None,
        low_rank_fn=lambda w, rank, _inputs: factorization_module.low_rank_branch(
            w, rank, None, solver=spec.low_rank_solver
        ),
        weight_scales_fn=weight_scales,
        fake_quant_weight_fn=fake_quantize_weight,
        **kwargs,
    )


def test_search_without_rotation_is_bit_identical():
    torch.manual_seed(0)
    targets = collect_quant_targets(
        Proj(), TargetConfig(targets=[TargetRule(modules=["proj"])])
    )
    weight, inputs = torch.randn(32, 128), torch.randn(16, 128)
    solver = LowRankSolverSpec(mode="search", num_iters=3, activation_quant=True)
    spec = DiffusionQuantSpec(
        rank=4, group_size=64, smooth=False, low_rank_solver=solver
    )

    default = _search(spec, inputs, weight, targets)
    explicit = _search(spec, inputs, weight, targets, input_rotation=IDENTITY)
    rotated = _search(spec, inputs, weight, targets, input_rotation=HADAMARD)

    assert torch.equal(default.residual, explicit.residual)
    assert default.metadata == explicit.metadata
    # The rotated search scores a different (rotated-basis) rounding, but still
    # returns an unrotated residual of its own branch.
    assert default.metadata["errors"] != rotated.metadata["errors"]
    down, up = rotated.low_rank
    torch.testing.assert_close(rotated.residual, weight - up @ down)
    plain_q = lowrank_search_module._quantized_weight(
        weight, spec, weight_scales, fake_quantize_weight
    )
    assert torch.equal(
        plain_q, fake_quantize_weight(weight, weight_scales(weight, 64, False), False)
    )


def test_eval_replay_matches_kernel_rotated_w4a4():
    torch.manual_seed(0)
    n, k, r = 32, 128, 4
    x = torch.randn(6, k)
    smooth = torch.rand(k) + 0.5
    residual, branch = torch.randn(n, k), torch.randn(n, r) @ torch.randn(r, k)

    def quantize_input(t):
        return dynamic_fake_quantize_activation(t, group_size=64, float_point=False)

    def quantize_weight(w):
        return fake_quantize_weight(w, weight_scales(w, 64, False), False)

    # The search patches Q_rot(R) into the module and wraps the activation quantizer.
    module = nn.Linear(k, n, bias=False)
    module.weight.data = HADAMARD.quantize_in_basis(residual, quantize_weight)
    pre_hook, post_hook = lowrank_search_module._replay_projector_hooks(
        module, branch, LowRankSolverSpec(), HADAMARD.wrap(quantize_input), smooth
    )
    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(post_hook)

    x_s = x / smooth
    kernel = (
        quantize_input(HADAMARD.rotate_activations(x_s))
        @ quantize_weight(HADAMARD.rotate_input_dim(residual)).t()
        + x_s @ branch.t()
    )
    torch.testing.assert_close(module(x), kernel, rtol=1e-4, atol=1e-4)


def test_gptq_hessian_is_built_in_rotated_basis():
    torch.manual_seed(0)
    x = torch.randn(40, 128)
    plain, _ = _hessian((x,), 128, -1, torch.device("cpu"))
    rotated, _ = _hessian((x,), 128, -1, torch.device("cpu"), HADAMARD)
    blockdiag = _blockdiag_hadamard(128)
    torch.testing.assert_close(
        rotated, blockdiag.t() @ plain @ blockdiag, rtol=1e-4, atol=1e-4
    )


def _rotated_spec(**kwargs):
    defaults = dict(
        rank=0,
        group_size=64,
        smooth=False,
        rotation=RotationSpec(),
        activation_quant=ActivationQuantSpec(enabled=False),
    )
    return DiffusionQuantSpec(**(defaults | kwargs))


def test_export_writes_rotation_flag_and_rotated_residual(tmp_path):
    from diffuse_compressor.runtime import (
        _load_checkpoint,
        _load_dequantized_transformer_state,
    )

    torch.manual_seed(0)
    model = Proj().to(torch.bfloat16)
    weight = model.proj.weight.detach().clone()
    output = tmp_path / "rotated.safetensors"
    rule = TargetRule(
        modules=["proj"], quant=SvdqTargetQuant(weight_layout=NaiveSvdqLayout())
    )
    quantize_and_export(
        model,
        _rotated_spec(),
        TargetConfig(targets=[rule]),
        CalibrationSpec(samples=[{"x": torch.randn(8, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    flag = FLAG
    state, metadata = _load_checkpoint(output)
    assert metadata["targets"][0]["rotation"] == flag
    assert metadata["rotation"] == flag
    # The transform is stored as data: exactly the Sylvester H64 / 8 blocks.
    assert torch.equal(state["proj.input_transform"], hadamard_matrix(64))
    with safetensors.safe_open(output, framework="pt") as handle:
        assert json.loads(handle.metadata()["quantization_config"])["rotation"] == flag

    # rank=0 and no smoothing: the packed residual is exactly Q(W @ H_blk).
    rotated = HADAMARD.rotate_input_dim(weight)
    codes = quantize_to_codes(
        rotated, weight_scales(rotated, 64, False), None, float_point=False
    )
    lo, hi = codes[:, 0::2] & 0xF, (codes[:, 1::2] & 0xF) << 4
    assert torch.equal(
        state["proj.qweight"], (lo | hi).to(torch.uint8).view(torch.int8)
    )

    # torch-dequant undoes the rotation, recovering W up to 4-bit error.
    restored = Proj().to(torch.bfloat16)
    _load_dequantized_transformer_state(restored, state=state, metadata=metadata)
    relative = (
        restored.proj.weight.float() - weight.float()
    ).norm() / weight.float().norm()
    assert relative < 0.15


def test_full_rotated_flow_exports_rotation_op_option(tmp_path):
    torch.manual_seed(0)
    output = tmp_path / "packed.safetensors"
    quantize_and_export(
        Proj().to(torch.bfloat16),
        _rotated_spec(
            rank=16,
            smooth=True,
            activation_quant=ActivationQuantSpec(enabled=True),
            low_rank_solver=LowRankSolverSpec(mode="search", num_iters=2),
            gptq=GptqSpec(enabled=True, block_size=32),
        ),
        TargetConfig(targets=[TargetRule(modules=["proj"])]),
        CalibrationSpec(samples=[{"x": torch.randn(160, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )
    with safetensors.safe_open(output, framework="pt") as handle:
        config = json.loads(handle.metadata()["quantization_config"])
    target = config["runtime_manifest"]["targets"][0]
    assert target["op_options"]["rotation"] == FLAG
    # Rotated checkpoints declare format v2 so v1-only loaders refuse them.
    assert config["runtime_manifest"]["nunchaku_format_version"] == 2


def test_unrotated_export_has_no_rotation_keys(tmp_path):
    output = tmp_path / "plain.safetensors"
    quantize_and_export(
        Proj().to(torch.bfloat16),
        replace(_rotated_spec(), rotation=None),
        TargetConfig(targets=[TargetRule(modules=["proj"])]),
        None,
        ExportSpec(output=output),
    )
    with safetensors.safe_open(output, framework="pt") as handle:
        config = json.loads(handle.metadata()["quantization_config"])
    assert "rotation" not in config
    assert "rotation" not in config["runtime_manifest"]["targets"][0]["op_options"]


def test_rotation_rejects_in_features_not_divisible_by_block():
    model = Proj(in_features=96, out_features=64).to(torch.bfloat16)
    config = TargetConfig(targets=[TargetRule(modules=["proj"])])
    with pytest.raises(ValueError, match="in_features divisible by 64.*in_features=96"):
        quantize_diffusion(
            model,
            _rotated_spec(),
            collect_quant_targets(model, config),
            target_config=config,
        )


def test_rotation_spec_and_conv_targets_are_validated():
    with pytest.raises(ValueError, match="block_size must be 64"):
        RotationSpec(block_size=32)
    with pytest.raises(ValueError, match="linear targets only"):
        resolve_input_rotation(RotationSpec(), in_features=128, kind="conv")
    with pytest.raises(ValueError, match="Unsupported input rotation"):
        input_rotation_from_metadata({"type": "random_hadamard"})


def test_rotation_refuses_nvfp4_unless_explicitly_allowed():
    with pytest.raises(ValueError, match="INT4 only"):
        DiffusionQuantSpec(precision="fp4", group_size=16, rotation=RotationSpec())
    spec = DiffusionQuantSpec(precision="fp4", group_size=16, rotation=RotationSpec(allow_fp4=True))
    assert spec.rotation.allow_fp4
    with pytest.raises(ValueError, match="INT4 only"):
        resolve_input_rotation(RotationSpec(), in_features=128, precision="fp4")
    assert resolve_input_rotation(RotationSpec(allow_fp4=True), in_features=128, precision="fp4").enabled
    with pytest.raises(TypeError, match="allow_fp4"):
        RotationSpec(allow_fp4="yes")


def test_rotation_refuses_per_target_nvfp4_override(tmp_path):
    torch.manual_seed(0)
    model = Proj().to(torch.bfloat16)
    rule = TargetRule(
        modules=["proj"],
        quant=SvdqTargetQuant(precision="fp4", group_size=16, weight_layout=NaiveSvdqLayout()),
    )
    with pytest.raises(ValueError, match="INT4 only"):
        quantize_and_export(
            model,
            _rotated_spec(),
            TargetConfig(targets=[rule]),
            CalibrationSpec(samples=[{"x": torch.randn(8, 128, dtype=torch.bfloat16)}]),
            ExportSpec(output=tmp_path / "fp4.safetensors"),
        )


def test_torch_dequant_input_mode_rotated_low_rank_branch_reads_raw_input(tmp_path):
    """With rotation, Q(x) still reaches the output only through the rotated residual."""
    from diffuse_compressor import SmoothSpec
    from diffuse_compressor.runtime import (
        _activation_input_quantizer_from_state,
        _load_checkpoint,
        _load_dequantized_transformer_state,
    )

    torch.manual_seed(0)
    model = Proj()
    x = torch.randn(256, 128)
    x[:, 3] *= 40.0
    output = tmp_path / "rotated_lowrank.safetensors"
    spec = DiffusionQuantSpec(
        rank=16,
        group_size=64,
        rotation=RotationSpec(),
        smooth=SmoothSpec(num_grids=4),
        activation_quant=ActivationQuantSpec(enabled=True, static=False),
    )
    rule = TargetRule(modules=["proj"], quant=SvdqTargetQuant(weight_layout=NaiveSvdqLayout()))
    quantize_and_export(
        model, spec, TargetConfig(targets=[rule]), CalibrationSpec(samples=[{"x": x}]), ExportSpec(output=output)
    )
    state, metadata = _load_checkpoint(output)
    target = metadata["targets"][0]
    assert target["rotation"] == FLAG
    merged, split = Proj(), Proj()
    _load_dequantized_transformer_state(merged, state=state, metadata=metadata, activation_mode="none")
    _load_dequantized_transformer_state(split, state=state, metadata=metadata, activation_mode="input")
    quant = _activation_input_quantizer_from_state(
        "proj", target, state, input_rotation_from_metadata(target["rotation"], state["proj.input_transform"])
    )
    probe = x[:64]
    with torch.no_grad():
        noise = split(probe) - merged(probe)
        residual_only = (quant(probe) - probe) @ split.proj.weight.t()
        through_merged = (quant(probe) - probe) @ merged.proj.weight.t()
    torch.testing.assert_close(noise, residual_only, rtol=1e-3, atol=1e-3)
    assert (noise - through_merged).abs().max() > 1e-3



def _random_orthogonal(*shape):
    q, _ = torch.linalg.qr(torch.randn(*shape, dtype=torch.float64))
    return q.float()


def test_block_matrix_rotation_is_exact_for_any_orthogonal_blocks():
    from diffuse_compressor.methods.svdquant.rotation import BlockMatrixRotation

    torch.manual_seed(0)
    x, w = torch.randn(32, 256), torch.randn(48, 256)
    for blocks in (_random_orthogonal(64, 64), _random_orthogonal(4, 64, 64), _random_orthogonal(16, 16)):
        rotation = BlockMatrixRotation(blocks)
        torch.testing.assert_close(
            rotation.rotate_activations(x) @ rotation.rotate_input_dim(w).t(), x @ w.t(), rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(rotation.unrotate_input_dim(rotation.rotate_input_dim(x)), x, rtol=1e-5, atol=1e-5)
        assert rotation.metadata()["type"] == "block_matrix"
        assert rotation.metadata()["shared"] == (blocks.dim() == 2)


def test_block_matrix_rotation_validates_shape_and_orthogonality():
    from diffuse_compressor.methods.svdquant.rotation import BlockMatrixRotation

    with pytest.raises(ValueError, match="orthogonal"):
        BlockMatrixRotation(torch.randn(64, 64))
    with pytest.raises(ValueError, match=r"\[b, b\]"):
        BlockMatrixRotation(torch.randn(64, 32))
    with pytest.raises(ValueError, match="blocks"):
        BlockMatrixRotation(_random_orthogonal(3, 64, 64)).rotate_input_dim(torch.randn(2, 256))


def test_rotation_metadata_round_trips_through_stored_matrix():
    from diffuse_compressor.methods.svdquant.rotation import BlockMatrixRotation

    blocks = _random_orthogonal(64, 64)
    stored = BlockMatrixRotation(blocks, generator="learned")
    rebuilt = input_rotation_from_metadata(stored.metadata(), stored.matrix())
    x = torch.randn(8, 128)
    torch.testing.assert_close(rebuilt.rotate_activations(x), stored.rotate_activations(x))
    with pytest.raises(ValueError, match="input_transform"):
        input_rotation_from_metadata(stored.metadata(), None)
    # Hadamard is just data; the pre-release {"type": "hadamard"} form (no tensor) is rejected.
    assert torch.equal(HADAMARD.matrix(), hadamard_matrix(64))
    with pytest.raises(ValueError, match="Unsupported input rotation metadata"):
        input_rotation_from_metadata({"type": "hadamard", "block_size": 64})



class TwoProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(128, 128)
        self.b = nn.Linear(128, 128)

    def forward(self, x):
        return self.b(self.a(x))


def test_rotation_refuses_unsigned_shifted_targets_before_calibration(monkeypatch, tmp_path):
    import diffuse_compressor.api as api_module

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("activation-shift calibration ran before the rotation check")

    monkeypatch.setattr(api_module, "_apply_calibrated_activation_shifts", fail_if_called)
    rules = [
        TargetRule(modules=["a"]),
        TargetRule(modules=["b"], quant=SvdqTargetQuant(shift_activations=True)),
    ]
    with pytest.raises(ValueError, match=r"signed activation path.*\['b'\].*rotation=False"):
        quantize_and_export(
            TwoProj(),
            _rotated_spec(),
            TargetConfig(targets=rules),
            CalibrationSpec(samples=[{"x": torch.randn(32, 128)}]),
            ExportSpec(output=tmp_path / "refused.safetensors"),
        )
    with pytest.raises(ValueError, match="signed activation path"):
        resolve_input_rotation(RotationSpec(), in_features=128, signed_activations=False)


def test_per_target_rotation_opt_out_exports_that_target_unrotated(tmp_path):
    from diffuse_compressor.runtime import _load_checkpoint

    torch.manual_seed(0)
    output = tmp_path / "mixed.safetensors"
    rules = [
        TargetRule(modules=["a"], quant=SvdqTargetQuant(weight_layout=NaiveSvdqLayout())),
        TargetRule(
            modules=["b"],
            quant=SvdqTargetQuant(weight_layout=NaiveSvdqLayout(), shift_activations=True, rotation=False),
        ),
    ]
    quantize_and_export(
        TwoProj(),
        _rotated_spec(),
        TargetConfig(targets=rules),
        CalibrationSpec(samples=[{"x": torch.randn(32, 128)}]),
        ExportSpec(output=output),
    )
    state, metadata = _load_checkpoint(output)
    by_name = {t["export_name"]: t for t in metadata["targets"]}
    assert by_name["a"]["rotation"] == FLAG and "a.input_transform" in state
    assert "rotation" not in by_name["b"] and "b.input_transform" not in state
    assert by_name["b"]["quant"]["rotation"] is False
    with pytest.raises(ValueError, match="None .*or False"):
        SvdqTargetQuant(rotation=True)


def test_unrotated_manifest_stays_format_v1(tmp_path):
    output = tmp_path / "v1.safetensors"
    quantize_and_export(
        Proj().to(torch.bfloat16),
        _rotated_spec(rotation=None),
        TargetConfig(targets=[TargetRule(modules=["proj"])]),
        CalibrationSpec(samples=[{"x": torch.randn(32, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )
    with safetensors.safe_open(output, framework="pt") as handle:
        config = json.loads(handle.metadata()["quantization_config"])
    assert config["runtime_manifest"]["nunchaku_format_version"] == 1
