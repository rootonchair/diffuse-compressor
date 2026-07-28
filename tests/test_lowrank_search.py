from dataclasses import dataclass

import pytest
import torch
from torch import nn

import diffuse_compressor.methods.svdquant.factorization as factorization_module
import diffuse_compressor.methods.svdquant.lowrank_search as lowrank_search_module
from diffuse_compressor import (
    ActivationQuantSpec,
    CalibrationScopeRule,
    CalibrationSpec,
    DiffusionQuantSpec,
    LowRankSolverSpec,
    SvdqLayout,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    collect_quant_targets,
    quantize_diffusion,
)


class ReplayBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4, bias=True)

    def forward(self, x):
        return self.q(x).tanh()


class ReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = ReplayBlock()

    def forward(self, x):
        return self.block(x)


class GroupedReplayBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 4, bias=False)
        self.k = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return (self.q(x) + self.k(x)).tanh()


class GroupedReplayModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = GroupedReplayBlock()

    def forward(self, x):
        return self.block(x)


@dataclass(frozen=True)
class DataclassReplayPayload:
    x: torch.Tensor
    aux: dict[str, torch.Tensor]


def _target_config(eval_module: str | None = "block"):
    return TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["block.q"],
                export_name="block.q_proj",
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            )
        ],
        calibration_scopes=[
            CalibrationScopeRule("block", ["block"], eval_module=eval_module)
        ],
    )


def test_low_rank_solver_spec_validates_mode():
    with pytest.raises(ValueError, match="Unsupported low-rank solver mode"):
        LowRankSolverSpec(mode="bad")  # type: ignore[arg-type]


def test_low_rank_solver_spec_validates_degree():
    with pytest.raises(ValueError, match="degree must be positive"):
        LowRankSolverSpec(degree=0)


def test_low_rank_solver_spec_validates_svd_backend():
    with pytest.raises(ValueError, match="Unsupported low-rank SVD backend"):
        LowRankSolverSpec(svd_backend="bad")  # type: ignore[arg-type]


def test_weighted_svd_remains_default_solver():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    target_config = _target_config(eval_module=None)
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=2, group_size=4, smooth=False),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(2, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    metadata = artifact.quantized_targets[0].metadata["low_rank_solver"]
    assert metadata["mode"] == "weighted_svd"
    assert metadata["svd_backend"] == "svd_lowrank"
    assert "proj_down" in artifact.quantized_targets[0].state_dict


def test_weighted_svd_can_use_torch_svd_lowrank(monkeypatch):
    calls = []

    def fake_svd_lowrank(weight, q, niter):
        calls.append((tuple(weight.shape), q, niter))
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        return u[:, :q], s[:q], vh[:q].t()

    monkeypatch.setattr(factorization_module.torch, "svd_lowrank", fake_svd_lowrank)
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    target_config = _target_config(eval_module=None)
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=2,
            group_size=4,
            smooth=False,
            low_rank_solver=LowRankSolverSpec(
                svd_backend="svd_lowrank",
                svd_lowrank_oversample=3,
                svd_lowrank_niter=2,
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(2, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    metadata = artifact.quantized_targets[0].metadata["low_rank_solver"]
    assert calls == [((4, 4), 4, 2)]
    assert metadata["svd_backend"] == "svd_lowrank"
    assert metadata["svd_lowrank_oversample"] == 3
    assert metadata["svd_lowrank_niter"] == 2


def test_search_solver_uses_eval_replay_and_exports_low_rank_metadata():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    original_weight = model.block.q.weight.detach().clone()
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=2,
            group_size=4,
            smooth=False,
            low_rank_solver=LowRankSolverSpec(
                mode="search", num_iters=2, eval_replay=True
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(3, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    metadata = target.metadata["low_rank_solver"]
    assert metadata["mode"] == "search"
    assert metadata["iterations"] == 2
    assert metadata["best_error"] == min(metadata["errors"])
    assert metadata["errors"][metadata["best_iteration"]] == metadata["best_error"]
    assert metadata["eval_replay"] is True
    assert artifact.metadata["calibration"]["eval_replay_scopes"] == ["block"]
    assert target.state_dict["proj_down"].shape[-1] == 2
    assert torch.allclose(model.block.q.weight, original_weight)


def test_search_solver_eval_replay_supports_compute_device_offload():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    original_weight = model.block.q.weight.detach().clone()
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=2,
            group_size=4,
            smooth=False,
            compute_device="cpu",
            offload_model=True,
            low_rank_solver=LowRankSolverSpec(
                mode="search", num_iters=1, eval_replay=True
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(3, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert target.metadata["compute_device"] == "cpu"
    assert target.metadata["low_rank_solver"]["eval_replay"] is True
    assert next(model.parameters()).device.type == "cpu"
    assert torch.allclose(model.block.q.weight, original_weight)


def test_eval_replay_device_move_traverses_dataclass_inputs():
    payload = DataclassReplayPayload(
        x=torch.randn(2, 4),
        aux={"context": torch.randn(2, 4)},
    )

    moved = lowrank_search_module._to_device(payload, torch.device("cpu"))

    assert moved is not payload
    assert moved.x.device.type == "cpu"
    assert moved.aux["context"].device.type == "cpu"


def test_eval_replay_tree_error_flattens_dataclass_outputs():
    actual = DataclassReplayPayload(
        x=torch.tensor([[2.0, 4.0]]),
        aux={"context": torch.tensor([[1.0, 5.0]])},
    )
    expected = DataclassReplayPayload(
        x=torch.tensor([[1.0, 2.0]]),
        aux={"context": torch.tensor([[1.0, 3.0]])},
    )

    error = lowrank_search_module._tree_error(actual, expected, degree=2)

    assert error.item() == pytest.approx(2.25)


def test_fp4_search_solver_scores_candidates():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            precision="fp4",
            rank=2,
            group_size=4,
            smooth=False,
            low_rank_solver=LowRankSolverSpec(
                mode="search", num_iters=1, eval_replay=True
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(3, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert target.metadata["precision"] == "fp4"
    assert target.metadata["low_rank_solver"]["mode"] == "search"
    assert target.metadata["low_rank_solver"]["iterations"] == 1
    assert "proj_down" in target.state_dict


def test_search_solver_scores_all_eval_replays():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=2,
            group_size=4,
            smooth=False,
            low_rank_solver=LowRankSolverSpec(
                mode="search", num_iters=1, eval_replay=True, degree=1
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[
                {"x": torch.randn(2, 4, dtype=torch.bfloat16)},
                {"x": torch.randn(2, 4, dtype=torch.bfloat16)},
            ],
            batch_size=1,
        ),
        target_config=target_config,
    )

    metadata = artifact.quantized_targets[0].metadata["low_rank_solver"]
    assert metadata["degree"] == 1
    assert metadata["eval_replay_count"] == 2


def test_search_solver_handles_grouped_targets_with_shared_branch():
    torch.manual_seed(0)
    model = GroupedReplayModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="qk",
                modules=["block.q", "block.k"],
                export_name="block.qk_proj",
                roles=("q", "k"),
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            )
        ],
        calibration_scopes=[
            CalibrationScopeRule("block", ["block"], eval_module="block")
        ],
    )
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=2,
            group_size=4,
            smooth=False,
            low_rank_solver=LowRankSolverSpec(
                mode="search", num_iters=2, eval_replay=True
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(3, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    metadata = target.metadata["low_rank_solver"]
    assert metadata["mode"] == "search"
    assert metadata["best_error"] == min(metadata["errors"])
    assert target.metadata["source_modules"] == ["block.q", "block.k"]
    assert target.state_dict["proj_down"].shape == (4, 2)
    assert target.state_dict["proj_up"].shape == (8, 2)


def test_search_solver_early_stop_on_non_improvement():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=4,
            group_size=4,
            smooth=False,
            low_rank_solver=LowRankSolverSpec(
                mode="search", num_iters=5, early_stop=True
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(3, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    metadata = artifact.quantized_targets[0].metadata["low_rank_solver"]
    assert metadata["stopped_early"] is True
    assert metadata["iterations"] < 5


def test_search_solver_compensate_and_activation_quant_are_recorded():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    target_config = _target_config(eval_module=None)
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=2,
            group_size=4,
            smooth=False,
            low_rank_solver=LowRankSolverSpec(
                mode="search",
                num_iters=2,
                compensate=True,
                activation_quant=True,
                eval_replay=False,
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(3, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    metadata = artifact.quantized_targets[0].metadata["low_rank_solver"]
    assert metadata["compensate"] is True
    assert metadata["activation_quant"] is True
    assert metadata["eval_replay"] is False


def test_search_solver_uses_calibrated_activation_quant_from_quant_spec():
    torch.manual_seed(0)
    model = ReplayModel().to(torch.bfloat16)
    target_config = _target_config(eval_module=None)
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=2,
            group_size=4,
            smooth=False,
            activation_quant=ActivationQuantSpec(enabled=True),
            low_rank_solver=LowRankSolverSpec(
                mode="search", num_iters=1, activation_quant=False, eval_replay=False
            ),
        ),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(3, 4, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert target.metadata["low_rank_solver"]["activation_quant"] is True
    assert target.metadata["activation_quant"]["inputs"]["calibrated"] is True
    assert "input_scale" not in target.state_dict
