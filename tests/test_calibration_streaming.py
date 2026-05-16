import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffuse_compressor import (
    CalibrationCaptureRule,
    CalibrationScopeRule,
    CalibrationSpec,
    DiffusionQuantSpec,
    TargetConfig,
    TargetRule,
    collect_quant_targets,
    quantize_diffusion,
)
from diffuse_compressor.calibration import IOTensorsCache, assign_calibration_scopes, iter_calibration_scopes, prepare_calibration_cache, _check_ram


class ScopedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.q = nn.Linear(64, 8)
            self.blocks.append(block)

    def forward(self, x):
        self.calls += 1
        return self.blocks[0].q(x) + self.blocks[1].q(x)


def _target_config():
    return TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.*.q"],
                export_name="blocks.{0}.q_proj",
            )
        ],
        calibration_scopes=[
            CalibrationScopeRule("blocks.{0}", ["blocks.*"]),
        ],
    )


def test_prepare_calibration_cache_creates_reuses_and_refreshes(tmp_path):
    sample = {"x": torch.randn(2, 64)}
    cache_dir = tmp_path / "calib"

    model = ScopedModel()
    paths = prepare_calibration_cache(
        model,
        CalibrationSpec(samples=[sample], cache_dir=cache_dir, cache_mode="reuse"),
    )
    assert len(paths) == 1
    assert model.calls == 1

    model = ScopedModel()
    paths = prepare_calibration_cache(
        model,
        CalibrationSpec(samples=[sample], cache_dir=cache_dir, cache_mode="reuse"),
    )
    assert len(paths) == 1
    assert model.calls == 0

    model = ScopedModel()
    paths = prepare_calibration_cache(
        model,
        CalibrationSpec(samples=[sample], cache_dir=cache_dir, cache_mode="refresh"),
    )
    assert len(paths) == 1
    assert model.calls == 1


def test_quantize_diffusion_streams_scopes_and_records_metadata(tmp_path):
    torch.manual_seed(0)
    model = ScopedModel().to(torch.bfloat16)
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)
    samples = [{"x": torch.randn(2, 64, dtype=torch.bfloat16)}]

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        targets,
        calibration=CalibrationSpec(samples=samples, cache_dir=tmp_path / "calib", cache_mode="refresh"),
        target_config=target_config,
    )

    metadata = artifact.metadata["calibration"]
    assert metadata["captured_scopes"] == ["blocks.0", "blocks.1"]
    assert metadata["scope_target_counts"] == {"blocks.0": 1, "blocks.1": 1}
    assert metadata["captured_targets"] == ["blocks.0.q_proj", "blocks.1.q_proj"]
    assert [target.metadata["calibrated"] for target in artifact.quantized_targets] == [True, True]


def test_assign_calibration_scopes_falls_back_to_target_scopes():
    model = ScopedModel()
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["blocks.*.q"], "blocks.{0}.q_proj"),
        ]
    )
    targets = collect_quant_targets(model, target_config)

    scopes = assign_calibration_scopes(model, targets, target_config)

    assert [scope.name for scope in scopes] == ["blocks.0.q_proj", "blocks.1.q_proj"]
    assert [len(scope.targets) for scope in scopes] == [1, 1]


def test_disabled_cache_mode_does_not_write_cache_files(tmp_path):
    torch.manual_seed(0)
    model = ScopedModel().to(torch.bfloat16)
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)

    quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(2, 64, dtype=torch.bfloat16)}],
            cache_dir=tmp_path / "calib",
            cache_mode="disabled",
        ),
        target_config=target_config,
    )

    assert not (tmp_path / "calib" / "caches").exists()


def test_ram_usage_limit_raises(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "psutil",
        SimpleNamespace(virtual_memory=lambda: SimpleNamespace(percent=95.0)),
    )

    with pytest.raises(RuntimeError, match="ram_usage_limit"):
        _check_ram(CalibrationSpec(samples=[], ram_usage_limit=0.90))


def test_tensor_and_io_cache_capture_cpu_rows_and_clear():
    cache = IOTensorsCache()
    cache.inputs.add(torch.randn(2, 3), max_rows=4)
    cache.outputs.add(torch.randn(5, 3), max_rows=4)

    assert cache.inputs.tensor().device.type == "cpu"
    assert cache.inputs.tensor().shape == (2, 3)
    assert cache.outputs.tensor().shape == (4, 3)

    cache.clear()
    assert cache.inputs.tensor() is None
    assert cache.outputs.tensor() is None


def test_calibration_scope_capture_modules_inputs_and_outputs():
    torch.manual_seed(0)
    model = ScopedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*.q"], "blocks.{0}.q_proj")],
        calibration_scopes=[
            CalibrationScopeRule(
                "blocks.{0}",
                ["blocks.*"],
                capture_modules=[
                    CalibrationCaptureRule(
                        name="block.{0}.q_io",
                        modules=["blocks.*.q"],
                        inputs=True,
                        outputs=True,
                    )
                ],
            )
        ],
    )
    targets = collect_quant_targets(model, target_config)
    iterator = iter_calibration_scopes(
        model,
        targets,
        target_config,
        CalibrationSpec(samples=[{"x": torch.randn(2, 64, dtype=torch.bfloat16)}]),
    )
    batch = next(iterator)

    assert "block.0.q_io" in batch.layer_cache
    assert batch.layer_cache["block.0.q_io"].inputs.tensor().shape[-1] == 64
    assert batch.layer_cache["block.0.q_io"].outputs.tensor().shape[-1] == 8
    assert batch.inputs["blocks.0.q_proj"].shape[-1] == 64


class SequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x):
        self.calls += 1
        for block in self.blocks:
            x = block(x)
        return x


def test_use_prev_scope_outputs_replays_next_scope_without_root_recompute():
    torch.manual_seed(0)
    model = SequentialModel()
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["blocks.*"], "blocks.{0}"),
        ],
        calibration_scopes=[
            CalibrationScopeRule("blocks.{0}", ["blocks.*"], use_prev_scope_outputs=True),
        ],
    )
    targets = collect_quant_targets(model, target_config)
    batches = list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(samples=[{"x": torch.randn(2, 4)}]),
        )
    )

    assert [batch.scope.name for batch in batches] == ["blocks.0", "blocks.1"]
    assert model.calls == 1
    assert all(batch.eval_replay is not None for batch in batches)
