import random
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
    SvdqLayout,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    collect_quant_targets,
    quantize_diffusion,
)
from diffuse_compressor.calibration import (
    IOTensorsCache,
    assign_calibration_scopes,
    iter_calibration_scopes,
    prepare_calibration_cache,
    _check_ram,
)
from diffuse_compressor.calibration.data import (
    ModuleForwardInput,
    iter_calibration_forward_inputs,
    resolve_samples,
    run_forward_input,
)
from diffuse_compressor.calibration.utils import model_device, remove_accelerate_hooks


class ScopedBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 8)


class ScopedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.blocks = nn.ModuleList()
        for _ in range(2):
            self.blocks.append(ScopedBlock())

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
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            )
        ],
        calibration_scopes=[
            CalibrationScopeRule(
                "blocks.{0}", ["blocks.*"], use_prev_scope_outputs=False
            ),
        ],
    )


def test_model_device_prefers_accelerate_execution_device_for_offloaded_models():
    model = ScopedModel()
    model._hf_hook = SimpleNamespace(execution_device=torch.device("cuda:0"))

    assert model_device(model) == torch.device("cuda:0")


def test_model_device_finds_chained_accelerate_execution_device():
    model = ScopedModel()
    model._hf_hook = SimpleNamespace(
        hooks=(
            SimpleNamespace(execution_device=None),
            SimpleNamespace(execution_device=torch.device("cuda:1")),
        )
    )

    assert model_device(model) == torch.device("cuda:1")


def test_remove_accelerate_hooks_logs_when_hooks_are_removed(monkeypatch, caplog):
    model = ScopedModel()
    model._hf_hook = SimpleNamespace(detach_hook=lambda module: None)

    def fake_remove_hook_from_submodules(module):
        delattr(module, "_hf_hook")

    monkeypatch.setitem(
        sys.modules,
        "accelerate.hooks",
        SimpleNamespace(remove_hook_from_submodules=fake_remove_hook_from_submodules),
    )

    with caplog.at_level("INFO", logger="diffuse_compressor.calibration.utils"):
        removed = remove_accelerate_hooks(model)

    assert removed is True
    assert not hasattr(model, "_hf_hook")
    assert "- Removed Accelerate hooks from model" in caplog.text


def test_remove_accelerate_hooks_restores_inference_tensors_inside_inference_mode(monkeypatch):
    model = ScopedModel()
    model._hf_hook = SimpleNamespace(detach_hook=lambda module: None)
    with torch.inference_mode():
        inference_weight = model.blocks[0].q.weight.detach().clone()
    inference_mode_states = []

    def fake_remove_hook_from_submodules(module):
        inference_mode_states.append(torch.is_inference_mode_enabled())
        module.blocks[0].q.weight = nn.Parameter(inference_weight, requires_grad=True)
        delattr(module, "_hf_hook")

    monkeypatch.setitem(
        sys.modules,
        "accelerate.hooks",
        SimpleNamespace(remove_hook_from_submodules=fake_remove_hook_from_submodules),
    )

    removed = remove_accelerate_hooks(model)

    assert removed is True
    assert inference_mode_states == [True]
    assert model.blocks[0].q.weight.requires_grad is True
    assert not hasattr(model, "_hf_hook")


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


def test_prepare_calibration_cache_limits_reused_records_with_seed(tmp_path):
    samples = [{"x": torch.randn(2, 64)} for _ in range(5)]
    cache_dir = tmp_path / "calib"
    all_paths = prepare_calibration_cache(
        ScopedModel(),
        CalibrationSpec(
            samples=samples, num_samples=-1, cache_dir=cache_dir, cache_mode="refresh"
        ),
    )
    expected = list(all_paths)
    random.Random(7).shuffle(expected)
    expected = sorted(expected[:3])

    model = ScopedModel()
    selected = prepare_calibration_cache(
        model,
        CalibrationSpec(
            samples=samples,
            cache_num_samples=3,
            seed=7,
            cache_dir=cache_dir,
            cache_mode="reuse",
        ),
    )

    assert selected == expected
    assert model.calls == 0


def test_prepare_calibration_cache_does_not_limit_reused_records_with_num_samples(
    tmp_path,
):
    samples = [{"x": torch.randn(2, 64)} for _ in range(5)]
    cache_dir = tmp_path / "calib"
    all_paths = prepare_calibration_cache(
        ScopedModel(),
        CalibrationSpec(
            samples=samples, num_samples=-1, cache_dir=cache_dir, cache_mode="refresh"
        ),
    )

    selected = prepare_calibration_cache(
        ScopedModel(),
        CalibrationSpec(
            samples=samples, num_samples=3, cache_dir=cache_dir, cache_mode="reuse"
        ),
    )

    assert selected == all_paths


def test_prepare_calibration_cache_allows_all_reused_records(tmp_path):
    samples = [{"x": torch.randn(2, 64)} for _ in range(4)]
    cache_dir = tmp_path / "calib"
    all_paths = prepare_calibration_cache(
        ScopedModel(),
        CalibrationSpec(
            samples=samples, num_samples=-1, cache_dir=cache_dir, cache_mode="refresh"
        ),
    )

    assert (
        prepare_calibration_cache(
            ScopedModel(),
            CalibrationSpec(
                samples=samples,
                cache_num_samples=-1,
                cache_dir=cache_dir,
                cache_mode="reuse",
            ),
        )
        == all_paths
    )
    assert (
        prepare_calibration_cache(
            ScopedModel(),
            CalibrationSpec(
                samples=samples,
                cache_num_samples=None,
                cache_dir=cache_dir,
                cache_mode="reuse",
            ),
        )
        == all_paths
    )


def test_resolve_samples_all_sentinel_keeps_every_sample():
    samples = [{"x": index} for index in range(4)]

    assert resolve_samples(CalibrationSpec(samples=samples, num_samples=-1)) == samples
    assert (
        resolve_samples(CalibrationSpec(samples=samples, num_samples=2)) == samples[:2]
    )


def test_calibration_cache_records_samples_then_replay_batches(tmp_path):
    class SingleTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.q = nn.Linear(4, 4)

        def forward(self, x):
            self.calls += 1
            return self.q(x)

    samples = [{"x": torch.randn(1, 4)} for _ in range(3)]
    cache_dir = tmp_path / "calib"
    model = SingleTargetModel()
    paths = prepare_calibration_cache(
        model,
        CalibrationSpec(
            samples=samples, cache_dir=cache_dir, cache_mode="refresh", batch_size=2
        ),
    )

    assert len(paths) == 3
    assert model.calls == 3

    replay_model = SingleTargetModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["q"], "q")],
        calibration_scopes=[CalibrationScopeRule("q", ["q"])],
    )
    targets = collect_quant_targets(replay_model, target_config)
    batches = list(
        iter_calibration_scopes(
            replay_model,
            targets,
            target_config,
            CalibrationSpec(
                cache_dir=cache_dir,
                cache_mode="reuse",
                batch_size=2,
                sample_batch_size=2,
            ),
        )
    )

    assert len(batches) == 1
    assert replay_model.calls == 2
    assert batches[0].inputs["q"].shape == (3, 4)
    assert [chunk.shape[0] for chunk in batches[0].input_partitions["q"]] == [2, 1]


def test_cached_replay_allows_sample_batch_size_different_from_replay_batch_size(
    tmp_path, caplog
):
    class SingleTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.q = nn.Linear(4, 4)

        def forward(self, x):
            self.calls += 1
            return self.q(x)

    samples = [{"x": torch.randn(1, 4)} for _ in range(3)]
    cache_dir = tmp_path / "calib"
    prepare_calibration_cache(
        SingleTargetModel(),
        CalibrationSpec(
            samples=samples, cache_dir=cache_dir, cache_mode="refresh", batch_size=2
        ),
    )
    replay_model = SingleTargetModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["q"], "q")],
        calibration_scopes=[CalibrationScopeRule("q", ["q"])],
    )
    targets = collect_quant_targets(replay_model, target_config)

    with caplog.at_level("INFO", logger="diffuse_compressor.calibration.scopes"):
        batch = next(
            iter_calibration_scopes(
                replay_model,
                targets,
                target_config,
                CalibrationSpec(
                    cache_dir=cache_dir,
                    cache_mode="reuse",
                    batch_size=2,
                    sample_batch_size=1,
                ),
            )
        )

    assert replay_model.calls == 2
    assert batch.inputs["q"].shape == (3, 4)
    assert [chunk.shape[0] for chunk in batch.input_partitions["q"]] == [1, 1, 1]
    assert "Captured 1 input caches (3 rows, 3 partition rows)" in caplog.text


def test_cached_replay_preserves_none_kwargs_when_batching(tmp_path):
    class OptionalGuidanceModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.guidance_values = []
            self.q = nn.Linear(4, 4)

        def forward(self, x, guidance=None):
            self.calls += 1
            self.guidance_values.append(guidance)
            if guidance is not None:
                guidance = guidance.to(x.dtype)
                x = x + guidance.reshape(-1, 1)
            return self.q(x)

    samples = [{"x": torch.randn(1, 4), "guidance": None} for _ in range(3)]
    cache_dir = tmp_path / "calib"
    prepare_calibration_cache(
        OptionalGuidanceModel(),
        CalibrationSpec(samples=samples, cache_dir=cache_dir, cache_mode="refresh"),
    )
    replay_model = OptionalGuidanceModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["q"], "q")],
        calibration_scopes=[CalibrationScopeRule("q", ["q"])],
    )
    targets = collect_quant_targets(replay_model, target_config)

    batches = list(
        iter_calibration_scopes(
            replay_model,
            targets,
            target_config,
            CalibrationSpec(cache_dir=cache_dir, cache_mode="reuse", batch_size=2),
        )
    )

    assert len(batches) == 1
    assert replay_model.calls == 2
    assert replay_model.guidance_values == [None, None]


def test_cached_replay_preserves_configured_shared_inputs_when_batching(tmp_path):
    cache_paths = []
    txt_ids = torch.arange(6).reshape(2, 3)
    img_ids = torch.arange(12).reshape(4, 3)
    for index in range(2):
        path = tmp_path / f"record-{index}.pt"
        torch.save(
            {
                "args": (),
                "kwargs": {
                    "hidden_states": torch.full((2, 4, 3), float(index)),
                    "txt_ids": txt_ids,
                    "img_ids": img_ids,
                },
            },
            path,
        )
        cache_paths.append(path)

    batch = next(
        iter_calibration_forward_inputs(
            CalibrationSpec(batch_size=2, shared_input_keys=("txt_ids", "img_ids")),
            cache_paths=cache_paths,
        )
    )

    assert batch.kwargs["hidden_states"].shape == (4, 4, 3)
    assert torch.equal(batch.kwargs["txt_ids"], txt_ids)
    assert torch.equal(batch.kwargs["img_ids"], img_ids)


def test_cached_replay_rejects_inconsistent_shared_inputs(tmp_path):
    cache_paths = []
    for index in range(2):
        path = tmp_path / f"record-{index}.pt"
        torch.save(
            {
                "args": (),
                "kwargs": {
                    "hidden_states": torch.full((2, 4, 3), float(index)),
                    "img_ids": torch.arange(12).reshape(4, 3) + index,
                },
            },
            path,
        )
        cache_paths.append(path)

    with pytest.raises(ValueError, match="img_ids"):
        next(
            iter_calibration_forward_inputs(
                CalibrationSpec(batch_size=2, shared_input_keys=("img_ids",)),
                cache_paths=cache_paths,
            )
        )


def test_run_forward_input_saves_custom_forward_outputs(tmp_path):
    calls = []

    def forward_fn(sample):
        return {"prompt": sample["prompt"]}

    def save_fn(result, sample, output_dir):
        calls.append((result, dict(sample), output_dir))

    run_forward_input(
        nn.Identity(),
        CalibrationSpec(
            forward_fn=forward_fn,
            output_dir=tmp_path / "samples",
            output_save_fn=save_fn,
        ),
        ModuleForwardInput(kwargs={"prompt": "a prompt"}),
    )

    assert calls == [
        ({"prompt": "a prompt"}, {"prompt": "a prompt"}, tmp_path / "samples")
    ]


def test_run_forward_input_ignores_output_saving_without_complete_config(tmp_path):
    calls = []

    run_forward_input(
        nn.Identity(),
        CalibrationSpec(
            forward_fn=lambda sample: "result", output_dir=tmp_path / "samples"
        ),
        ModuleForwardInput(kwargs={"prompt": "a prompt"}),
    )
    run_forward_input(
        nn.Identity(),
        CalibrationSpec(
            forward_fn=lambda sample: "result",
            output_save_fn=lambda *args: calls.append(args),
        ),
        ModuleForwardInput(kwargs={"prompt": "a prompt"}),
    )

    assert calls == []


def test_calibration_dataloader_honors_drop_last_and_seeded_shuffle(tmp_path):
    class SingleTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(1, 1, bias=False)

        def forward(self, x):
            return self.q(x)

    samples = [{"x": torch.tensor([[float(index)]])} for index in range(5)]
    cache_dir = tmp_path / "calib"
    prepare_calibration_cache(
        SingleTargetModel(),
        CalibrationSpec(samples=samples, cache_dir=cache_dir, cache_mode="refresh"),
    )
    target_config = TargetConfig(
        targets=[TargetRule("q", ["q"], "q")],
        calibration_scopes=[CalibrationScopeRule("q", ["q"])],
    )

    def captured(seed):
        model = SingleTargetModel()
        targets = collect_quant_targets(model, target_config)
        batch = next(
            iter_calibration_scopes(
                model,
                targets,
                target_config,
                CalibrationSpec(
                    cache_dir=cache_dir,
                    cache_mode="reuse",
                    batch_size=2,
                    drop_last=True,
                    shuffle=True,
                    seed=seed,
                ),
            )
        )
        return batch.inputs["q"].flatten()

    first = captured(123)
    second = captured(123)
    assert first.shape == (4,)
    assert torch.equal(first, second)


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
        calibration=CalibrationSpec(
            samples=samples, cache_dir=tmp_path / "calib", cache_mode="refresh"
        ),
        target_config=target_config,
    )

    metadata = artifact.metadata["calibration"]
    assert metadata["scope_capture_mode"] == "all_targets"
    assert metadata["cache_records"] == {"selected": 1, "total": 1}
    assert metadata["captured_scopes"] == ["blocks.0", "blocks.1"]
    assert metadata["scope_target_counts"] == {"blocks.0": 1, "blocks.1": 1}
    assert metadata["captured_targets"] == ["blocks.0.q_proj", "blocks.1.q_proj"]
    assert [target.metadata["calibrated"] for target in artifact.quantized_targets] == [
        True,
        True,
    ]


def test_quantize_diffusion_records_selected_cache_metadata(tmp_path):
    torch.manual_seed(0)
    cache_dir = tmp_path / "calib"
    samples = [{"x": torch.randn(2, 64, dtype=torch.bfloat16)} for _ in range(5)]
    prepare_calibration_cache(
        ScopedModel().to(torch.bfloat16),
        CalibrationSpec(
            samples=samples, num_samples=-1, cache_dir=cache_dir, cache_mode="refresh"
        ),
    )
    model = ScopedModel().to(torch.bfloat16)
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        targets,
        calibration=CalibrationSpec(
            cache_dir=cache_dir, cache_mode="reuse", cache_num_samples=3, seed=7
        ),
        target_config=target_config,
    )

    assert artifact.metadata["calibration"]["cache_records"] == {
        "selected": 3,
        "total": 5,
    }
    assert artifact.metadata["calibration"]["cache_num_samples"] == 3


def test_one_target_scope_capture_yields_target_local_batches():
    class MultiTargetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)
            self.k = nn.Linear(4, 4)

        def forward(self, x):
            return self.q(x) + self.k(x)

    class MultiTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.block = MultiTargetBlock()

        def forward(self, x):
            self.calls += 1
            return self.block(x)

    torch.manual_seed(0)
    model = MultiTargetModel()
    target_config = TargetConfig(
        targets=[
            TargetRule(
                "q",
                ["block.q"],
                "block.q",
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            ),
            TargetRule(
                "k",
                ["block.k"],
                "block.k",
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            ),
        ],
        calibration_scopes=[CalibrationScopeRule("block", ["block"])],
    )
    targets = collect_quant_targets(model, target_config)

    batches = list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(
                samples=[{"x": torch.randn(2, 4)}], scope_capture_mode="one_target"
            ),
        )
    )

    assert model.calls == 2
    assert [batch.scope.name for batch in batches] == ["block", "block"]
    assert [
        [target.export_name for target in batch.scope.targets] for batch in batches
    ] == [["block.q"], ["block.k"]]
    assert [set(batch.inputs) for batch in batches] == [{"block.q"}, {"block.k"}]
    assert all(batch.scope_target_count == 2 for batch in batches)
    assert batches[0].eval_replays is batches[1].eval_replays


def test_quantize_diffusion_one_target_capture_records_scope_metadata():
    class MultiTargetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)
            self.k = nn.Linear(4, 4)

        def forward(self, x):
            return self.q(x) + self.k(x)

    class MultiTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = MultiTargetBlock()

        def forward(self, x):
            return self.block(x)

    torch.manual_seed(0)
    model = MultiTargetModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                "q",
                ["block.q"],
                "block.q",
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            ),
            TargetRule(
                "k",
                ["block.k"],
                "block.k",
                quant=SvdqTargetQuant(weight_layout=SvdqLayout()),
            ),
        ],
        calibration_scopes=[CalibrationScopeRule("block", ["block"])],
    )
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=0, group_size=4, smooth=False),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(2, 4, dtype=torch.bfloat16)}],
            scope_capture_mode="one_target",
        ),
        target_config=target_config,
    )

    metadata = artifact.metadata["calibration"]
    assert metadata["scope_capture_mode"] == "one_target"
    assert metadata["captured_scopes"] == ["block"]
    assert metadata["scope_target_counts"] == {"block": 2}
    assert metadata["captured_targets"] == ["block.k", "block.q"]
    assert [target.target.export_name for target in artifact.quantized_targets] == [
        "block.q",
        "block.k",
    ]


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


def test_calibration_scope_name_defaults_to_matched_module_path():
    model = ScopedModel()
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["blocks.*.q"], "blocks.{0}.q_proj"),
        ],
        calibration_scopes=[
            CalibrationScopeRule(["blocks.*"]),
        ],
    )
    targets = collect_quant_targets(model, target_config)

    scopes = assign_calibration_scopes(model, targets, target_config)

    assert [scope.name for scope in scopes] == ["blocks.0", "blocks.1"]
    assert [scope.module_name for scope in scopes] == ["blocks.0", "blocks.1"]


def test_calibration_scope_can_match_module_classes_without_patterns():
    model = ScopedModel()
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["blocks.*.q"], "blocks.{0}.q_proj"),
        ],
        calibration_scopes=[
            CalibrationScopeRule(module_classes=ScopedBlock),
        ],
    )
    targets = collect_quant_targets(model, target_config)

    scopes = assign_calibration_scopes(model, targets, target_config)

    assert [scope.name for scope in scopes] == ["blocks.0", "blocks.1"]
    assert [scope.module_name for scope in scopes] == ["blocks.0", "blocks.1"]
    assert [target.export_name for target in scopes[0].targets] == ["blocks.0.q_proj"]


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


def test_scope_capture_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="scope_capture_mode"):
        CalibrationSpec(scope_capture_mode="target")  # type: ignore[arg-type]


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


def test_tensor_cache_can_capture_without_row_cap():
    cache = IOTensorsCache()
    cache.inputs.add(torch.randn(5, 3), max_rows=None)

    assert cache.inputs.tensor().shape == (5, 3)


def test_io_cache_stores_keyed_tensors_and_repartitions():
    cache = IOTensorsCache()
    args = (torch.randn(3, 4),)
    kwargs = {"encoder_hidden_states": torch.randn(2, 6)}

    cache.inputs.add((args, kwargs), max_rows=8, keys=("arg0", "encoder_hidden_states"))

    keyed = cache.inputs.keyed_tensors()
    assert set(keyed) == {"arg0", "encoder_hidden_states"}
    assert keyed["arg0"].shape == (3, 4)
    assert keyed["encoder_hidden_states"].shape == (2, 6)
    assert [
        chunk.shape[0]
        for chunk in cache.inputs.repartition("arg0", sample_batch_size=2)
    ] == [2, 1]


def test_input_stats_only_captures_min_without_target_row_or_output_caches():
    torch.manual_seed(0)
    model = ScopedModel()
    target_config = _target_config()
    targets = collect_quant_targets(model, target_config)
    x = torch.tensor([[-3.0] + [1.0] * 63, [2.0] * 64])

    iterator = iter_calibration_scopes(
        model,
        targets,
        target_config,
        CalibrationSpec(samples=[{"x": x}]),
        input_stats_only=True,
        capture_target_outputs=False,
    )
    batch = next(iterator)

    assert batch.inputs["blocks.0.q_proj"].item() == -3.0
    for cache in batch.layer_cache.values():
        assert not cache.inputs.tensors
        assert not cache.outputs.tensors
        assert cache.input_min == -3.0
    iterator.close()


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


def test_calibration_scope_cache_aliases_reuse_grouped_inputs():
    torch.manual_seed(0)
    model = ScopedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["blocks.*.q"], "blocks.{0}.q_proj"),
            TargetRule("k", ["blocks.*.q"], "blocks.{0}.k_proj"),
        ],
        calibration_scopes=[
            CalibrationScopeRule(
                "blocks.{0}",
                ["blocks.*"],
                cache_aliases={"blocks.{0}.k_proj": "blocks.{0}.q_proj"},
            )
        ],
    )
    targets = collect_quant_targets(model, target_config)
    batch = next(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(samples=[{"x": torch.randn(2, 64, dtype=torch.bfloat16)}]),
        )
    )

    assert torch.equal(batch.inputs["blocks.0.k_proj"], batch.inputs["blocks.0.q_proj"])


def test_eval_replay_filters_kwargs_for_attention_like_blocks():
    class AttentionLike(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)

        def forward(
            self, hidden_states, encoder_hidden_states=None, attention_mask=None
        ):
            if encoder_hidden_states is not None:
                hidden_states = hidden_states + encoder_hidden_states
            if attention_mask is not None:
                hidden_states = hidden_states + attention_mask
            return self.q(hidden_states)

    class AttentionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = AttentionLike()

        def forward(self, x, context, mask):
            return self.attn(x, encoder_hidden_states=context, attention_mask=mask)

    model = AttentionModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["attn.q"], "attn.q")],
        calibration_scopes=[
            CalibrationScopeRule(
                "attn",
                ["attn"],
                eval_module="attn",
                replay_kwarg_keys=("encoder_hidden_states",),
            )
        ],
    )
    targets = collect_quant_targets(model, target_config)
    batch = next(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(
                samples=[
                    {
                        "x": torch.randn(2, 4),
                        "context": torch.randn(2, 4),
                        "mask": torch.randn(2, 4),
                    }
                ]
            ),
        )
    )

    assert batch.eval_replay is not None
    assert set(batch.eval_replay.kwargs) == {"encoder_hidden_states"}


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


class TrackingLinear(nn.Linear):
    def __init__(self):
        super().__init__(4, 4)
        self.to_calls: list[str] = []

    def to(self, *args, **kwargs):
        if args:
            self.to_calls.append(str(torch.device(args[0])))
        elif "device" in kwargs:
            self.to_calls.append(str(torch.device(kwargs["device"])))
        else:
            self.to_calls.append("")
        return super().to(*args, **kwargs)


class TrackingSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.to_calls: list[str] = []
        self.blocks = nn.ModuleList([TrackingLinear(), TrackingLinear()])

    def forward(self, x):
        self.calls += 1
        for block in self.blocks:
            x = block(x)
        return x

    def to(self, *args, **kwargs):
        if args:
            self.to_calls.append(str(torch.device(args[0])))
        elif "device" in kwargs:
            self.to_calls.append(str(torch.device(kwargs["device"])))
        else:
            self.to_calls.append("")
        return super().to(*args, **kwargs)


def test_use_prev_scope_outputs_replays_next_scope_without_root_recompute():
    torch.manual_seed(0)
    model = SequentialModel()
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["blocks.*"], "blocks.{0}"),
        ],
        calibration_scopes=[
            CalibrationScopeRule("blocks.{0}", ["blocks.*"]),
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


def test_offload_model_prev_scope_replay_moves_only_scoped_module():
    torch.manual_seed(0)
    model = TrackingSequentialModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*"], "blocks.{0}")],
        calibration_scopes=[CalibrationScopeRule("blocks.{0}", ["blocks.*"])],
    )
    targets = collect_quant_targets(model, target_config)

    batches = list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(samples=[{"x": torch.randn(2, 4)}]),
            offload_model=True,
        )
    )

    assert [batch.scope.name for batch in batches] == ["blocks.0", "blocks.1"]
    assert model.calls == 1
    assert model.to_calls == ["cpu"]
    assert model.blocks[0].to_calls == []
    assert model.blocks[1].to_calls == ["cpu", "cpu"]


def test_offload_model_recompute_warns_and_restores_full_model(caplog):
    torch.manual_seed(0)
    model = TrackingSequentialModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*"], "blocks.{0}")],
        calibration_scopes=[
            CalibrationScopeRule("blocks.{0}", ["blocks.*"], recompute=True),
        ],
    )
    targets = collect_quant_targets(model, target_config)

    with caplog.at_level("WARNING", logger="diffuse_compressor.calibration.scopes"):
        batches = list(
            iter_calibration_scopes(
                model,
                targets,
                target_config,
                CalibrationSpec(samples=[{"x": torch.randn(2, 4)}]),
                offload_model=True,
            )
        )

    assert [batch.scope.name for batch in batches] == ["blocks.0", "blocks.1"]
    assert model.calls == 2
    assert model.to_calls == ["cpu", "cpu"]
    assert model.blocks[0].to_calls == []
    assert model.blocks[1].to_calls == []
    assert "Scoped replay offload unavailable for blocks.1" in caplog.text
    assert "recompute=True" in caplog.text


def test_full_model_replay_uses_accelerate_cpu_offload_for_cuda(monkeypatch):
    from diffuse_compressor.calibration import replay as replay_module

    model = TrackingSequentialModel()
    calls: list[tuple[nn.Module, torch.device]] = []

    def fake_cpu_offload(module, *, execution_device):
        calls.append((module, torch.device(execution_device)))

    monkeypatch.setitem(
        sys.modules, "accelerate", SimpleNamespace(cpu_offload=fake_cpu_offload)
    )

    replay_module._restore_model_for_full_replay(
        model,
        torch.device("cuda:0"),
        offload_model=True,
        skip_moves=False,
    )

    assert calls == [(model, torch.device("cuda:0"))]
    assert model.to_calls == []


def test_offload_model_accelerate_hooks_skip_manual_scoped_moves():
    torch.manual_seed(0)
    model = TrackingSequentialModel()
    model._hf_hook = SimpleNamespace(execution_device=torch.device("cpu"))
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*"], "blocks.{0}")],
        calibration_scopes=[CalibrationScopeRule("blocks.{0}", ["blocks.*"])],
    )
    targets = collect_quant_targets(model, target_config)

    batches = list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(samples=[{"x": torch.randn(2, 4)}]),
            offload_model=True,
        )
    )

    assert [batch.scope.name for batch in batches] == ["blocks.0", "blocks.1"]
    assert model.calls == 1
    assert model.to_calls == []
    assert model.blocks[0].to_calls == []
    assert model.blocks[1].to_calls == []


def test_offload_model_keeps_accelerate_hooks_for_cached_replay(tmp_path):
    torch.manual_seed(0)
    cache_dir = tmp_path / "calib"
    prepare_calibration_cache(
        TrackingSequentialModel(),
        CalibrationSpec(
            samples=[{"x": torch.randn(2, 4)}],
            cache_dir=cache_dir,
            cache_mode="refresh",
        ),
    )
    detach_calls: list[str] = []

    class FakeAccelerateHook:
        execution_device = torch.device("cpu")

        def detach_hook(self, module):
            detach_calls.append(type(module).__name__)

    model = TrackingSequentialModel()
    model._hf_hook = FakeAccelerateHook()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*"], "blocks.{0}")],
        calibration_scopes=[CalibrationScopeRule("blocks.{0}", ["blocks.*"])],
    )
    targets = collect_quant_targets(model, target_config)

    batches = list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(cache_dir=cache_dir, cache_mode="reuse"),
            offload_model=True,
        )
    )

    assert [batch.scope.name for batch in batches] == ["blocks.0", "blocks.1"]
    assert detach_calls == []
    assert hasattr(model, "_hf_hook")
    assert model.to_calls == []


def test_use_prev_scope_outputs_replays_all_batches_without_root_recompute():
    torch.manual_seed(0)
    model = SequentialModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*"], "blocks.{0}")],
        calibration_scopes=[CalibrationScopeRule("blocks.{0}", ["blocks.*"])],
    )
    targets = collect_quant_targets(model, target_config)
    batches = list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(
                samples=[{"x": torch.randn(1, 4)} for _ in range(3)], batch_size=2
            ),
        )
    )

    assert model.calls == 2
    assert [len(batch.eval_replays) for batch in batches] == [2, 2]


def test_prev_output_transform_can_repack_dict_output():
    class DictBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)

        def forward(self, x):
            return {"x": self.q(x)}

    class DictSequential(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.blocks = nn.ModuleList([DictBlock(), DictBlock()])

        def forward(self, x):
            self.calls += 1
            value = self.blocks[0](x)
            return self.blocks[1](value["x"])

    model = DictSequential()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*.q"], "blocks.{0}.q")],
        calibration_scopes=[
            CalibrationScopeRule(
                "blocks.{0}",
                ["blocks.*"],
                prev_output_transform=lambda output: ((output["x"],), {}),
            )
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

    assert len(batches) == 2
    assert model.calls == 1


def test_prev_replay_transform_replays_flux_like_blocks_without_root_recompute():
    class FluxLikeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.q = nn.Linear(4, 4)

        def forward(
            self,
            *,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            self.calls += 1
            hidden_states = self.q(hidden_states + temb)
            encoder_hidden_states = encoder_hidden_states + temb
            return encoder_hidden_states, hidden_states

    class FluxLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0
            self.blocks = nn.ModuleList(
                [FluxLikeBlock(), FluxLikeBlock(), FluxLikeBlock()]
            )

        def forward(
            self,
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            self.calls += 1
            for block in self.blocks:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )
            return encoder_hidden_states, hidden_states

    def prev_replay_to_flux_kwargs(replay):
        encoder_hidden_states, hidden_states = replay.output
        kwargs = dict(replay.kwargs)
        kwargs["hidden_states"] = hidden_states
        kwargs["encoder_hidden_states"] = encoder_hidden_states
        return (), kwargs

    torch.manual_seed(0)
    model = FluxLikeModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*.q"], "blocks.{0}.q")],
        calibration_scopes=[
            CalibrationScopeRule(
                "blocks.{0}",
                ["blocks.*"],
                prev_replay_transform=prev_replay_to_flux_kwargs,
            )
        ],
    )
    targets = collect_quant_targets(model, target_config)

    batches = list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(
                samples=[
                    {
                        "hidden_states": torch.randn(2, 4),
                        "encoder_hidden_states": torch.randn(2, 4),
                        "temb": torch.randn(2, 4),
                        "image_rotary_emb": None,
                        "joint_attention_kwargs": {"scale": 1.0},
                    }
                ]
            ),
        )
    )

    assert [batch.scope.name for batch in batches] == [
        "blocks.0",
        "blocks.1",
        "blocks.2",
    ]
    assert model.calls == 1
    assert [block.calls for block in model.blocks] == [1, 1, 1]
    assert all(
        batch.inputs[f"blocks.{index}.q"].shape == (2, 4)
        for index, batch in enumerate(batches)
    )
    assert all(batch.eval_replay is not None for batch in batches)


def test_recompute_bypasses_previous_scope_outputs():
    torch.manual_seed(0)
    model = SequentialModel()
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.*"], "blocks.{0}")],
        calibration_scopes=[
            CalibrationScopeRule("blocks.{0}", ["blocks.*"], recompute=True),
        ],
    )
    targets = collect_quant_targets(model, target_config)
    list(
        iter_calibration_scopes(
            model,
            targets,
            target_config,
            CalibrationSpec(samples=[{"x": torch.randn(2, 4)}]),
        )
    )

    assert model.calls == 2
