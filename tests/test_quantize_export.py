import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors
import torch
from torch import nn

from diffuse_compressor import (
    AdaNormAwqW4A16Layout,
    ActivationQuantSpec,
    AwqW4A16Layout,
    CalibrationSpec,
    DiffusionQuantSpec,
    ExportSpec,
    LoggingConfig,
    NaiveSvdqLayout,
    NunchakuSvdqLayout,
    PatchRule,
    QuantizationCacheSpec,
    RangeCalibrationSpec,
    SmoothSpec,
    SvdqTargetQuant,
    TargetConfig,
    TargetRule,
    AwqTargetQuant,
    WeightRangeCalibrationSpec,
    export_checkpoint,
    quantize_and_export,
)
from diffuse_compressor.artifact_cache import _jsonable, _target_cache_path


def _config_metadata(checkpoint_path: str | Path) -> dict:
    return json.loads(
        Path(checkpoint_path).with_suffix(".config.yaml").read_text(encoding="utf-8")
    )


def _checkpoint_quantization_config(checkpoint_path: str | Path) -> dict | None:
    with safetensors.safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        metadata_blob = handle.metadata().get("quantization_config")
    return None if metadata_blob is None else json.loads(metadata_blob)


def _assert_checkpoint_quantization_config(
    checkpoint_metadata: dict,
    config_metadata: dict,
    *,
    has_runtime_manifest: bool = False,
) -> None:
    expected_keys = {"method", "rank", "weight", "activation"}
    if has_runtime_manifest:
        expected_keys.add("runtime_manifest")
    assert set(checkpoint_metadata) == expected_keys
    assert checkpoint_metadata["method"] == config_metadata["method"]
    assert checkpoint_metadata["rank"] == config_metadata["rank"]
    assert checkpoint_metadata["weight"] == config_metadata["weight"]
    assert checkpoint_metadata["activation"] == config_metadata["activation"]


def _run_logged_tiny_quantize_and_export(
    tmp_path: Path, logging_config: LoggingConfig
) -> None:
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    output = tmp_path / f"{logging_config.name or 'tiny'}.safetensors"
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.0.q"], "blocks.0.q_proj")]
    )

    quantize_and_export(
        model,
        DiffusionQuantSpec(rank=0, group_size=64, smooth=False),
        target_config,
        CalibrationSpec(samples=[{"x": torch.randn(2, 64, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
        logging=logging_config,
    )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(2):
            block = nn.Module()
            block.q = nn.Linear(64, 8, bias=True)
            block.k = nn.Linear(64, 8, bias=True)
            block.v = nn.Linear(64, 8, bias=True)
            block.out = nn.Linear(64, 8, bias=True)
            self.blocks.append(block)
        self.final = nn.Linear(8, 8)

    def forward(self, x):
        return self.blocks[0].q(x)


class AlignedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 128, bias=True)

    def forward(self, x):
        return self.proj(x)


class WideOutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 256, bias=True)

    def forward(self, x):
        return self.proj(x)


class TinyConvModel(nn.Module):
    def __init__(self, *, kernel_size=1, padding=0):
        super().__init__()
        self.proj = nn.Conv2d(64, 16, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.proj(x)


def test_artifact_cache_serializes_module_classes_with_fully_qualified_names():
    first = type("RepeatedName", (nn.Linear,), {"__module__": "first_module"})
    second = type("RepeatedName", (nn.Linear,), {"__module__": "second_module"})

    first_config = _jsonable(TargetRule(module_classes=first))
    second_config = _jsonable(TargetRule(module_classes=second))

    assert first_config["module_classes"] == ["first_module.RepeatedName"]
    assert second_config["module_classes"] == ["second_module.RepeatedName"]
    assert first_config["module_classes"] != second_config["module_classes"]


def test_quantize_and_export_writes_nunchaku_safetensors(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    output = tmp_path / "tiny.safetensors"
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="qkv",
                modules=["blocks.*.q", "blocks.*.k", "blocks.*.v"],
                export_name="blocks.{0}.qkv_proj",
                roles=["q", "k", "v"],
            ),
            TargetRule(
                name="out",
                modules=["blocks.*.out"],
                export_name="blocks.{0}.out_proj",
            ),
        ],
        unquantized_patterns=["final.*"],
    )

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        target_config,
        CalibrationSpec(prompts=["a prompt"], num_samples=1),
        ExportSpec(output=output),
    )

    assert result.checkpoint_path == str(output)
    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    metadata = _config_metadata(output)

    assert metadata["method"] == "svdquant"
    assert metadata["rank"] == 4
    assert metadata["weight"]["dtype"] == "int4"
    _assert_checkpoint_quantization_config(
        _checkpoint_quantization_config(output), metadata
    )
    assert "blocks.0.qkv_proj.qweight" in keys
    assert "blocks.0.qkv_proj.proj_down" in keys
    assert "blocks.1.out_proj.wscales" in keys
    assert "final.weight" in keys
    assert "blocks.0.q.weight" not in keys


def test_quantize_and_export_logging_writes_text_and_target_records(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    output = tmp_path / "tiny.safetensors"
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.0.q"], "blocks.0.q_proj")]
    )

    quantize_and_export(
        model,
        DiffusionQuantSpec(rank=2, group_size=64),
        target_config,
        CalibrationSpec(samples=[{"x": torch.randn(2, 64, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
        logging=LoggingConfig(log_dir=tmp_path / "logs", name="run"),
    )

    text_log = tmp_path / "logs" / "run.txt"
    target_log = tmp_path / "logs" / "run.targets.jsonl"
    assert text_log.exists()
    assert target_log.exists()
    assert "Finished target blocks.0.q_proj" in text_log.read_text()

    records = [json.loads(line) for line in target_log.read_text().splitlines()]
    assert len(records) == 1
    record = records[0]
    assert record["target"] == "blocks.0.q_proj"
    assert record["modules"] == ["blocks.0.q"]
    assert record["checkpoint_path"] == str(output)
    assert record["elapsed_sec"] is not None
    assert record["low_rank_mode"] == "weighted_svd"
    assert record["iterations"] == 1


def test_quantize_diffusion_logging_writes_target_records_without_checkpoint(tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.0.q"], "blocks.0.q_proj")]
    )
    targets = collect_quant_targets(model, target_config)

    quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=0, group_size=64, smooth=False),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(2, 64, dtype=torch.bfloat16)}]
        ),
        target_config=target_config,
        logging=LoggingConfig(log_dir=tmp_path / "logs", name="diffusion"),
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "diffusion.targets.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records == [
        {
            "best_error": None,
            "calibrated": True,
            "checkpoint_path": None,
            "elapsed_sec": records[0]["elapsed_sec"],
            "errors": None,
            "group_size": 64,
            "iterations": 0,
            "low_rank_mode": "weighted_svd",
            "modules": ["blocks.0.q"],
            "precision": "int4",
            "rank": 0,
            "stopped_early": None,
            "target": "blocks.0.q_proj",
        }
    ]
    assert records[0]["elapsed_sec"] is not None


def test_quantize_and_export_without_logging_writes_no_run_logs(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    output = tmp_path / "tiny.safetensors"
    target_config = TargetConfig(
        targets=[TargetRule("q", ["blocks.0.q"], "blocks.0.q_proj")]
    )

    quantize_and_export(
        model,
        DiffusionQuantSpec(rank=0, group_size=64, smooth=False),
        target_config,
        CalibrationSpec(samples=[{"x": torch.randn(2, 64, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    assert not (tmp_path / "outputs").exists()


def test_quantize_and_export_logging_does_not_replace_process_streams_or_root_handlers(
    tmp_path,
):
    stdout = sys.stdout
    stderr = sys.stderr
    root_handlers = tuple(logging.getLogger().handlers)

    _run_logged_tiny_quantize_and_export(
        tmp_path,
        LoggingConfig(log_dir=tmp_path / "logs", name="run"),
    )

    assert sys.stdout is stdout
    assert sys.stderr is stderr
    assert tuple(logging.getLogger().handlers) == root_handlers


def test_quantize_and_export_logging_repeated_runs_use_available_paths(tmp_path):
    log_dir = tmp_path / "logs"

    _run_logged_tiny_quantize_and_export(
        tmp_path, LoggingConfig(log_dir=log_dir, name="run")
    )
    _run_logged_tiny_quantize_and_export(
        tmp_path, LoggingConfig(log_dir=log_dir, name="run")
    )

    text_logs = sorted(path.name for path in log_dir.glob("run*.txt"))
    target_logs = sorted(path.name for path in log_dir.glob("run*.targets.jsonl"))
    assert len(text_logs) == 2
    assert len(target_logs) == 2
    assert "run.txt" in text_logs
    assert "run.targets.jsonl" in target_logs


def test_quantize_and_export_logging_can_write_only_target_records(tmp_path):
    log_dir = tmp_path / "logs"

    _run_logged_tiny_quantize_and_export(
        tmp_path,
        LoggingConfig(
            log_dir=log_dir, name="targets-only", text_output=False, target_records=True
        ),
    )

    assert not (log_dir / "targets-only.txt").exists()
    assert (log_dir / "targets-only.targets.jsonl").exists()


def test_quantize_and_export_logging_can_write_only_text_log(tmp_path):
    log_dir = tmp_path / "logs"

    _run_logged_tiny_quantize_and_export(
        tmp_path,
        LoggingConfig(
            log_dir=log_dir, name="text-only", text_output=True, target_records=False
        ),
    )

    text_log = log_dir / "text-only.txt"
    assert text_log.exists()
    assert "Finished target blocks.0.q_proj" in text_log.read_text()
    assert not (log_dir / "text-only.targets.jsonl").exists()


def test_quantize_diffusion_captures_calibration_inputs(caplog):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.q"],
                export_name="blocks.0.q_proj",
            ),
        ],
    )
    targets = collect_quant_targets(model, target_config)
    samples = [{"x": torch.randn(4, 64, dtype=torch.bfloat16)} for _ in range(2)]

    with caplog.at_level("INFO"):
        artifact = quantize_diffusion(
            model,
            DiffusionQuantSpec(rank=4, group_size=64),
            targets,
            calibration=CalibrationSpec(
                samples=samples, num_samples=2, max_rows_per_target=5
            ),
            target_config=target_config,
        )

    assert artifact.metadata["calibration"]["captured_targets"] == ["blocks.0.q_proj"]
    assert artifact.quantized_targets[0].metadata["calibrated"] is True
    assert artifact.quantized_targets[0].state_dict["proj_down"].numel() > 0
    assert "Captured 1 input caches (5 rows, 5 partition rows)" in caplog.text
    assert "Calibrating input activation range from 5 rows across 1 partition" in caplog.text


def test_quantize_diffusion_can_offload_model_and_compute_on_cpu():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")
        ]
    )
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=64,
            smooth=False,
            compute_device="cpu",
            offload_model=True,
        ),
        targets,
        calibration=None,
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert artifact.metadata["quantization"] == {
        "compute_device": "cpu",
        "offload_model": True,
    }
    assert target.metadata["compute_device"] == "cpu"
    assert all(tensor.device.type == "cpu" for tensor in target.state_dict.values())
    assert next(model.parameters()).device.type == "cpu"


def test_quantize_diffusion_removes_accelerate_hooks_after_replay(monkeypatch):
    import diffuse_compressor.api as api
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    from diffuse_compressor.artifact import QuantizedTarget

    class HookTrackingTinyModel(TinyModel):
        def __init__(self):
            super().__init__()
            self.hook_states_during_forward: list[bool] = []
            self.to_calls: list[str] = []

        def forward(self, x):
            self.hook_states_during_forward.append(hasattr(self, "_hf_hook"))
            return super().forward(x)

        def to(self, *args, **kwargs):
            if args:
                self.to_calls.append(str(torch.device(args[0])))
            elif "device" in kwargs:
                self.to_calls.append(str(torch.device(kwargs["device"])))
            else:
                self.to_calls.append("")
            return super().to(*args, **kwargs)

    model = HookTrackingTinyModel()
    model._hf_hook = SimpleNamespace(execution_device=torch.device("cpu"))
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")
        ]
    )
    targets = collect_quant_targets(model, target_config)
    removed_hooks: list[bool] = []
    quantize_hook_states: list[bool] = []

    def fake_remove_accelerate_hooks(_model):
        removed_hooks.append(hasattr(_model, "_hf_hook"))
        delattr(_model, "_hf_hook")
        return True

    def fake_quantize_targets(iter_targets, *_args, **_kwargs):
        quantize_hook_states.append(hasattr(model, "_hf_hook"))
        return [
            QuantizedTarget(
                target=target,
                state_dict={"packed": torch.zeros(1)},
                metadata={"calibrated": True},
            )
            for target in iter_targets
        ]

    monkeypatch.setattr(api, "_remove_accelerate_hooks", fake_remove_accelerate_hooks)
    monkeypatch.setattr(api, "quantize_targets", fake_quantize_targets)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=64,
            smooth=False,
            compute_device="cpu",
            offload_model=False,
        ),
        targets,
        calibration=CalibrationSpec(samples=[{"x": torch.randn(4, 64)}]),
        target_config=target_config,
    )

    assert model.hook_states_during_forward == [True]
    assert removed_hooks == [True]
    assert quantize_hook_states == [False]
    assert model.to_calls == []
    assert artifact.quantized_targets[0].state_dict["packed"].shape == (1,)


def test_cuda_compute_device_requires_cuda_when_unavailable():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    if torch.cuda.is_available():
        pytest.skip("CUDA is available")
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")
        ]
    )
    targets = collect_quant_targets(model, target_config)

    with pytest.raises(RuntimeError, match="requires CUDA"):
        quantize_diffusion(
            model,
            DiffusionQuantSpec(
                rank=0, group_size=64, smooth=False, compute_device="cuda"
            ),
            targets,
            calibration=None,
            target_config=target_config,
        )


def test_activation_range_metadata_and_weight_range_export_runtime_tensors(tmp_path):
    from diffuse_compressor import (
        collect_quant_targets,
        export_checkpoint,
        quantize_diffusion,
    )

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")
        ],
    )
    targets = collect_quant_targets(model, target_config)
    samples = [{"x": torch.rand(4, 64, dtype=torch.bfloat16)}]

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=64,
            smooth=False,
            activation_quant=ActivationQuantSpec(
                enabled=True,
                inputs=RangeCalibrationSpec(
                    granularity="channel", symmetric=False, allow_unsigned=True
                ),
                outputs=RangeCalibrationSpec(granularity="tensor"),
            ),
            weight_range_calibration=WeightRangeCalibrationSpec(
                enabled=True,
                range=RangeCalibrationSpec(granularity="group"),
            ),
        ),
        targets,
        calibration=CalibrationSpec(samples=samples),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert "input_scale" not in target.state_dict
    assert "input_zero" not in target.state_dict
    assert "output_scale" not in target.state_dict
    assert "output_zero" not in target.state_dict
    assert target.state_dict["weight_range_scale"].shape == (1,)
    assert target.metadata["activation_quant"]["inputs"]["calibrated"] is True
    assert target.metadata["activation_quant"]["inputs"]["qmin"] == 0
    assert target.metadata["activation_quant"]["inputs"]["num_scales"] == 64
    assert target.metadata["activation_quant"]["outputs"]["calibrated"] is True
    assert target.metadata["activation_quant"]["outputs"]["num_scales"] == 1

    output = tmp_path / "ranges.safetensors"
    export_checkpoint(artifact, ExportSpec(output=output))
    with safetensors.safe_open(output, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
    metadata = _config_metadata(output)

    assert "blocks.0.q_proj.input_scale" not in keys
    assert "blocks.0.q_proj.output_zero" not in keys
    assert "blocks.0.q_proj.weight_range_scale" in keys
    assert metadata["targets"][0]["export_name"] == "blocks.0.q_proj"
    _assert_checkpoint_quantization_config(
        _checkpoint_quantization_config(output), metadata
    )


def test_explicit_activation_shift_patches_targets_and_records_metadata():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    from diffuse_compressor.patches import ShiftedLinear

    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")
        ]
    )
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=0, group_size=64, smooth=False, shift_activations=True),
        targets,
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(4, 64, dtype=torch.bfloat16) - 2}]
        ),
        target_config=target_config,
    )

    assert isinstance(model.blocks[0].q, ShiftedLinear)
    assert artifact.metadata["calibration"]["activation_shifts"]["blocks.0.q"] > 0
    assert artifact.quantized_targets[0].target.modules[0] is model.blocks[0].q


def test_activation_shift_calibration_honors_offload_model(monkeypatch):
    import diffuse_compressor.api as api
    from diffuse_compressor import collect_quant_targets

    class TrackingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(4, 4)
            self.to_calls: list[str] = []

        def forward(self, x):
            return self.q(x)

        def to(self, *args, **kwargs):
            if args:
                self.to_calls.append(str(torch.device(args[0])))
            elif "device" in kwargs:
                self.to_calls.append(str(torch.device(kwargs["device"])))
            else:
                self.to_calls.append("")
            return super().to(*args, **kwargs)

    model = TrackingModel()
    model._hf_hook = SimpleNamespace(execution_device=torch.device("cpu"))
    target_config = TargetConfig(targets=[TargetRule("q", ["q"], "q")])
    targets = collect_quant_targets(model, target_config)
    calls: list[tuple[bool, bool, bool]] = []
    removed_hooks: list[bool] = []
    prepare_hook_states: list[bool] = []

    def fake_iter_calibration_scopes(
        _model,
        iter_targets,
        _target_config,
        _calibration,
        *,
        offload_model=False,
        input_stats_only=False,
        capture_target_outputs=True,
    ):
        calls.append((offload_model, input_stats_only, capture_target_outputs))
        scope = type("Scope", (), {"name": "q", "targets": tuple(iter_targets)})()
        yield type(
            "Batch", (), {"scope": scope, "inputs": {"q": torch.tensor([[-1.0, 0.5]])}}
        )()

    monkeypatch.setattr(api, "iter_calibration_scopes", fake_iter_calibration_scopes)
    original_prepare_model = api.prepare_model

    def fake_remove_accelerate_hooks(_model):
        removed_hooks.append(hasattr(_model, "_hf_hook"))
        delattr(_model, "_hf_hook")
        return True

    def fake_prepare_model(_model, rules):
        prepare_hook_states.append(hasattr(_model, "_hf_hook"))
        return original_prepare_model(_model, rules)

    monkeypatch.setattr(api, "_remove_accelerate_hooks", fake_remove_accelerate_hooks)
    monkeypatch.setattr(api, "prepare_model", fake_prepare_model)

    refreshed, shifts = api._apply_calibrated_activation_shifts(
        model,
        targets,
        CalibrationSpec(samples=[{"x": torch.randn(1, 4)}]),
        target_config,
        DiffusionQuantSpec(
            rank=0,
            group_size=4,
            smooth=False,
            shift_activations=True,
            compute_device="cpu",
            offload_model=False,
        ),
    )

    assert calls == [(False, True, False)]
    assert removed_hooks == [True]
    assert prepare_hook_states == [False]
    assert model.to_calls == []
    assert shifts == {"q": 1.0}
    assert refreshed[0].modules[0] is model.q


def test_target_overrides_make_extra_weight_target_weight_only():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    from diffuse_compressor.patches import ShiftedLinear

    class TwoTargetModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(64, 64, bias=True)
            self.extra = nn.Linear(64, 64, bias=True)

        def forward(self, x):
            return self.q(x) + self.extra(x)

    torch.manual_seed(0)
    model = TwoTargetModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["q"], "q"),
            TargetRule(
                "extra",
                ["extra"],
                "extra",
                quant=SvdqTargetQuant(
                    precision="int4",
                    group_size=64,
                    rank=0,
                    shared_low_rank=False,
                    smooth=False,
                    activation_quant=False,
                    shift_activations=False,
                ),
            ),
        ]
    )
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            precision="fp4",
            rank=4,
            group_size=16,
            smooth=SmoothSpec(strategy="manual", alpha=0.5, beta=-1),
            activation_quant=ActivationQuantSpec(
                enabled=True,
                inputs=RangeCalibrationSpec(granularity="group", allow_unsigned=True),
                outputs=RangeCalibrationSpec(granularity="tensor"),
            ),
            shift_activations=True,
        ),
        collect_quant_targets(model, target_config),
        calibration=CalibrationSpec(
            samples=[{"x": torch.randn(4, 64, dtype=torch.bfloat16) - 2}]
        ),
        target_config=target_config,
    )

    by_name = {
        target.target.export_name: target for target in artifact.quantized_targets
    }
    normal = by_name["q"]
    extra = by_name["extra"]

    assert isinstance(model.q, ShiftedLinear)
    assert not isinstance(model.extra, ShiftedLinear)
    assert "q" in artifact.metadata["calibration"]["activation_shifts"]
    assert "extra" not in artifact.metadata["calibration"]["activation_shifts"]

    assert normal.metadata["rank"] == 4
    assert "proj_down" in normal.state_dict
    assert normal.metadata["activation_quant"]["enabled"] is True

    assert extra.metadata["precision"] == "int4"
    assert extra.metadata["group_size"] == 64
    assert extra.metadata["rank"] == 0
    assert extra.metadata["smooth"]["enabled"] is False
    assert extra.metadata["activation_quant"]["enabled"] is False
    assert "proj_down" not in extra.state_dict
    assert "input_scale" not in extra.state_dict
    assert "output_scale" not in extra.state_dict


def test_awq_w4a16_target_layout_exports_nunchaku_lite_extra_weight_tensors(tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    from diffuse_compressor.runtime import _reconstruct_target_weight

    class ExtraWeightModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.extra = nn.Linear(64, 64, bias=True)

        def forward(self, x):
            return self.extra(x)

    torch.manual_seed(0)
    model = ExtraWeightModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                "extra",
                ["extra"],
                "extra",
                quant=AwqTargetQuant(layout=AwqW4A16Layout()),
            )
        ]
    )
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(precision="fp4", rank=4, group_size=16, smooth=False),
        collect_quant_targets(model, target_config),
        calibration=None,
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    state = target.state_dict

    assert target.metadata["weight_layout"]["name"] == "awq_w4a16"
    assert state["qweight"].shape == (16, 32)
    assert state["qweight"].dtype == torch.int32
    assert state["wscales"].shape == (1, 64)
    assert state["wzeros"].shape == (1, 64)
    assert "smooth_factor" not in state
    assert "smooth_factor_orig" not in state
    assert "proj_down" not in state

    flat_state = {f"extra.{key}": value for key, value in state.items()}
    reconstructed = _reconstruct_target_weight(
        export_name="extra",
        state=flat_state,
        precision="int4",
        weight_layout={"name": "awq_w4a16"},
    )

    assert reconstructed.shape == model.extra.weight.shape
    assert torch.equal(
        state["wzeros"],
        (-7 * state["wscales"].float()).to(dtype=state["wscales"].dtype),
    )

    output = tmp_path / "awq.safetensors"
    export_checkpoint(artifact, ExportSpec(output=output))
    config_metadata = _config_metadata(output)
    metadata = _checkpoint_quantization_config(output)
    assert config_metadata["targets"][0]["quant"]["type"] == "awq"
    manifest = metadata["runtime_manifest"]
    assert manifest["schema"] == "nunchaku_lite.runtime_manifest"
    assert manifest["version"] == 1
    assert manifest["nunchaku_format_version"] == 1
    assert manifest["targets"][0]["checkpoint_prefix"] == "extra"
    assert manifest["targets"][0]["nunchaku_op"] == "awq_w4a16"
    assert manifest["targets"][0]["op_options"] == {}


def test_quantize_diffusion_skips_calibration_replay_for_awq_targets(monkeypatch):
    from diffuse_compressor import api
    from diffuse_compressor import (
        CalibrationScopeRule,
        collect_quant_targets,
        quantize_diffusion,
    )
    from diffuse_compressor.calibration import CalibrationScope, CalibrationScopeBatch

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(64, 64)

        def forward(self, x):
            return self.q(x)

    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = Block()
            self.extra = nn.Linear(64, 64)

        def forward(self, x):
            return self.extra(self.block(x))

    model = MixedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule("q", ["block.q"], "block.q"),
            TargetRule(
                "extra",
                ["extra"],
                "extra",
                quant=AwqTargetQuant(layout=AwqW4A16Layout()),
            ),
        ],
        calibration_scopes=[CalibrationScopeRule("block", ["block"])],
    )
    targets = collect_quant_targets(model, target_config)
    replay_target_names = []

    def fake_iter_calibration_scopes(
        _model,
        iter_targets,
        _target_config,
        _calibration,
        *,
        offload_model=False,
        input_stats_only=False,
        capture_target_outputs=True,
    ):
        del offload_model, input_stats_only, capture_target_outputs
        iter_targets = list(iter_targets)
        replay_target_names.extend(target.export_name for target in iter_targets)
        assert [target.export_name for target in iter_targets] == ["block.q"]
        yield CalibrationScopeBatch(
            scope=CalibrationScope(name="block", targets=tuple(iter_targets)),
            inputs={"block.q": torch.randn(2, 64, dtype=torch.bfloat16)},
        )

    monkeypatch.setattr(api, "iter_calibration_scopes", fake_iter_calibration_scopes)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=0, group_size=64, smooth=False),
        targets,
        calibration=CalibrationSpec(samples=[{"x": torch.randn(1, 64)}]),
        target_config=target_config,
    )

    assert replay_target_names == ["block.q"]
    assert [target.target.export_name for target in artifact.quantized_targets] == [
        "block.q",
        "extra",
    ]
    assert artifact.metadata["calibration"]["captured_targets"] == ["block.q"]
    assert artifact.quantized_targets[1].metadata["calibrated"] is False


@pytest.mark.parametrize("splits", [3, 6])
def test_adanorm_awq_w4a16_layout_reorders_outputs_and_bias(splits, tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    class AdaNormModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.Linear(64, 12, bias=True)

        def forward(self, x):
            return self.norm(x)

    model = AdaNormModel().to(torch.bfloat16)
    with torch.no_grad():
        model.norm.weight.copy_(
            torch.arange(12 * 64, dtype=torch.bfloat16).view(12, 64).mul_(0.001)
        )
        model.norm.bias.copy_(torch.arange(12, dtype=torch.bfloat16))
    target_config = TargetConfig(
        targets=[
            TargetRule(
                "norm",
                ["norm"],
                "norm",
                quant=AwqTargetQuant(layout=AdaNormAwqW4A16Layout(splits=splits)),
            )
        ]
    )
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(precision="fp4", rank=4, group_size=16, smooth=False),
        collect_quant_targets(model, target_config),
        calibration=None,
        target_config=target_config,
    )
    state = artifact.quantized_targets[0].state_dict
    metadata = artifact.quantized_targets[0].metadata["weight_layout"]

    expected_bias = (
        torch.arange(12, dtype=torch.bfloat16)
        .view(splits, 12 // splits)
        .transpose(0, 1)
        .contiguous()
    )
    delta = torch.zeros(splits, dtype=torch.bfloat16)
    delta[1] = 1
    delta[-2] = 1
    expected_bias = expected_bias.add(delta.view(1, splits)).reshape(12)

    assert metadata == {"name": "adanorm_awq_w4a16", "splits": splits}
    assert torch.equal(state["bias"], expected_bias)
    assert torch.equal(
        state["wzeros"],
        (-7 * state["wscales"].float()).to(dtype=state["wscales"].dtype),
    )

    output = tmp_path / f"adanorm_{splits}.safetensors"
    export_checkpoint(artifact, ExportSpec(output=output))
    export_metadata = _checkpoint_quantization_config(output)
    target = export_metadata["runtime_manifest"]["targets"][0]
    assert target["nunchaku_op"] == "adanorm_awq_w4a16"
    assert target["op_options"] == {"adanorm_splits": splits}


def test_target_export_bias_zero_synthesizes_bias_for_biasless_linear():
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    class BiaslessModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 32, bias=False)

        def forward(self, x):
            return self.proj(x)

    torch.manual_seed(0)
    model = BiaslessModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule("proj", ["proj"], "proj", quant=SvdqTargetQuant(bias="zero"))
        ]
    )
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(precision="fp4", rank=4, group_size=16, smooth=False),
        collect_quant_targets(model, target_config),
        calibration=None,
        target_config=target_config,
    )

    bias = artifact.quantized_targets[0].state_dict["bias"]

    assert bias.shape == (32,)
    assert bias.dtype == torch.bfloat16
    assert torch.count_nonzero(bias) == 0


def test_quantization_artifact_cache_reuses_valid_model_cache(tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion

    torch.manual_seed(0)
    samples = [{"x": torch.randn(4, 64, dtype=torch.bfloat16)}]
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")
        ]
    )
    cache = QuantizationCacheSpec(
        cache_dir=tmp_path / "artifacts", cache_mode="refresh"
    )

    model = TinyModel().to(torch.bfloat16)
    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=4, group_size=64),
        collect_quant_targets(model, target_config),
        calibration=CalibrationSpec(samples=samples, artifact_cache=cache),
        target_config=target_config,
    )
    assert (tmp_path / "artifacts" / "model.pt").exists()
    assert (tmp_path / "artifacts" / "smooth.pt").exists()

    reuse_model = TinyModel().to(torch.bfloat16)
    reused = quantize_diffusion(
        reuse_model,
        DiffusionQuantSpec(rank=4, group_size=64),
        collect_quant_targets(reuse_model, target_config),
        calibration=CalibrationSpec(
            samples=samples,
            artifact_cache=QuantizationCacheSpec(cache_dir=tmp_path / "artifacts"),
        ),
        target_config=target_config,
    )

    assert reused.metadata["artifact_cache"]["hit"] is True
    assert torch.equal(
        reused.quantized_targets[0].state_dict["qweight"],
        artifact.quantized_targets[0].state_dict["qweight"],
    )


def test_quantization_artifact_cache_resumes_completed_targets(monkeypatch, tmp_path):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    import diffuse_compressor.methods.svdquant.quantize as quantize_module

    torch.manual_seed(0)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj"),
            TargetRule(name="k", modules=["blocks.0.k"], export_name="blocks.0.k_proj"),
        ]
    )
    spec = DiffusionQuantSpec(rank=0, group_size=64, smooth=False)
    cache_root = tmp_path / "artifacts"

    model = TinyModel().to(torch.bfloat16)
    artifact = quantize_diffusion(
        model,
        spec,
        collect_quant_targets(model, target_config),
        calibration=CalibrationSpec(
            artifact_cache=QuantizationCacheSpec(
                cache_dir=cache_root, cache_mode="refresh"
            )
        ),
        target_config=target_config,
    )
    assert _target_cache_path(cache_root, "blocks.0.q_proj").exists()
    assert _target_cache_path(cache_root, "blocks.0.k_proj").exists()
    _target_cache_path(cache_root, "blocks.0.k_proj").unlink()
    (cache_root / "metadata.json").unlink()
    (cache_root / "model.pt").unlink()

    calls = []
    original = quantize_module._quantize_projector_target

    def wrapped_quantize_target(target, *args, **kwargs):
        calls.append(target.export_name)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(
        quantize_module, "_quantize_projector_target", wrapped_quantize_target
    )
    reuse_model = TinyModel().to(torch.bfloat16)
    reused = quantize_diffusion(
        reuse_model,
        spec,
        collect_quant_targets(reuse_model, target_config),
        calibration=CalibrationSpec(
            artifact_cache=QuantizationCacheSpec(cache_dir=cache_root)
        ),
        target_config=target_config,
    )

    assert calls == ["blocks.0.k_proj"]
    assert [target.target.export_name for target in reused.quantized_targets] == [
        "blocks.0.q_proj",
        "blocks.0.k_proj",
    ]
    assert torch.equal(
        reused.quantized_targets[0].state_dict["qweight"],
        artifact.quantized_targets[0].state_dict["qweight"],
    )
    assert _target_cache_path(cache_root, "blocks.0.k_proj").exists()


def test_quantization_artifact_cache_refresh_rewrites_completed_targets(
    monkeypatch, tmp_path
):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    import diffuse_compressor.methods.svdquant.quantize as quantize_module

    torch.manual_seed(0)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj"),
            TargetRule(name="k", modules=["blocks.0.k"], export_name="blocks.0.k_proj"),
        ]
    )
    spec = DiffusionQuantSpec(rank=0, group_size=64, smooth=False)
    cache_root = tmp_path / "artifacts"
    model = TinyModel().to(torch.bfloat16)
    quantize_diffusion(
        model,
        spec,
        collect_quant_targets(model, target_config),
        calibration=CalibrationSpec(
            artifact_cache=QuantizationCacheSpec(
                cache_dir=cache_root, cache_mode="refresh"
            )
        ),
        target_config=target_config,
    )

    calls = []
    original = quantize_module._quantize_projector_target

    def wrapped_quantize_target(target, *args, **kwargs):
        calls.append(target.export_name)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(
        quantize_module, "_quantize_projector_target", wrapped_quantize_target
    )
    refresh_model = TinyModel().to(torch.bfloat16)
    quantize_diffusion(
        refresh_model,
        spec,
        collect_quant_targets(refresh_model, target_config),
        calibration=CalibrationSpec(
            artifact_cache=QuantizationCacheSpec(
                cache_dir=cache_root, cache_mode="refresh"
            )
        ),
        target_config=target_config,
    )

    assert calls == ["blocks.0.q_proj", "blocks.0.k_proj"]


def test_quantization_artifact_cache_ignores_invalid_target_records(
    monkeypatch, tmp_path
):
    from diffuse_compressor import collect_quant_targets, quantize_diffusion
    import diffuse_compressor.methods.svdquant.quantize as quantize_module

    torch.manual_seed(0)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="q", modules=["blocks.0.q"], export_name="blocks.0.q_proj")
        ]
    )
    spec = DiffusionQuantSpec(rank=0, group_size=64, smooth=False)
    cache_root = tmp_path / "artifacts"
    model = TinyModel().to(torch.bfloat16)
    quantize_diffusion(
        model,
        spec,
        collect_quant_targets(model, target_config),
        calibration=CalibrationSpec(
            artifact_cache=QuantizationCacheSpec(
                cache_dir=cache_root, cache_mode="refresh"
            )
        ),
        target_config=target_config,
    )
    (cache_root / "metadata.json").unlink()
    (cache_root / "model.pt").unlink()
    target_path = _target_cache_path(cache_root, "blocks.0.q_proj")
    payload = torch.load(target_path, map_location="cpu", weights_only=False)
    payload["cache_key"] = "stale"
    torch.save(payload, target_path)
    torch.save(payload, target_path.with_name(f".{target_path.name}.tmp"))

    calls = []
    original = quantize_module._quantize_projector_target

    def wrapped_quantize_target(target, *args, **kwargs):
        calls.append(target.export_name)
        return original(target, *args, **kwargs)

    monkeypatch.setattr(
        quantize_module, "_quantize_projector_target", wrapped_quantize_target
    )
    reuse_model = TinyModel().to(torch.bfloat16)
    quantize_diffusion(
        reuse_model,
        spec,
        collect_quant_targets(reuse_model, target_config),
        calibration=CalibrationSpec(
            artifact_cache=QuantizationCacheSpec(cache_dir=cache_root)
        ),
        target_config=target_config,
    )

    assert calls == ["blocks.0.q_proj"]


def test_nvfp4_export_writes_deepcompressor_split_scales(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.q"],
                export_name="blocks.0.q_proj",
                quant=SvdqTargetQuant(precision="fp4"),
            )
        ]
    )
    output = tmp_path / "fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(
                enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 64, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(
        result.checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
        wscales = handle.get_tensor("blocks.0.q_proj.wscales")
        wcscales = handle.get_tensor("blocks.0.q_proj.wcscales")
    metadata = _config_metadata(result.checkpoint_path)

    assert "blocks.0.q_proj.qweight" in keys
    assert "blocks.0.q_proj.wscales" in keys
    assert "blocks.0.q_proj.wcscales" in keys
    assert "blocks.0.q_proj.wtscale" not in keys
    assert wscales.dtype == torch.float8_e4m3fn
    assert wscales.shape == (4, 8)
    assert wcscales.shape == (8,)
    assert metadata["weight"]["scale_dtypes"] == [None, "sfp8_e4m3_nan"]
    assert metadata["activation"]["scale_dtypes"] == ["sfp8_e4m3_nan"]
    assert metadata["targets"][0]["precision"] == "fp4"
    assert metadata["targets"][0]["group_size"] == 16
    assert metadata["targets"][0]["weight_scale_layout"] == "nvfp4_deepcompressor"
    assert metadata["targets"][0]["runtime_tensor_layout"] == "logical"
    _assert_checkpoint_quantization_config(
        _checkpoint_quantization_config(result.checkpoint_path), metadata
    )


def test_nunchaku_svdq_layout_fails_when_target_cannot_pack(tmp_path):
    torch.manual_seed(0)
    model = TinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="q",
                modules=["blocks.0.q"],
                export_name="blocks.0.q_proj",
                quant=SvdqTargetQuant(
                    precision="fp4", weight_layout=NunchakuSvdqLayout()
                ),
            )
        ]
    )

    with pytest.raises(RuntimeError, match="NunchakuSvdqLayout"):
        quantize_and_export(
            model,
            DiffusionQuantSpec(
                rank=0,
                group_size=16,
                smooth=False,
                weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
                activation_quant=ActivationQuantSpec(
                    enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
                ),
            ),
            target_config,
            CalibrationSpec(samples=[{"x": torch.rand(4, 64, dtype=torch.bfloat16)}]),
            ExportSpec(output=tmp_path / "fp4.safetensors"),
        )


def test_aligned_nvfp4_export_writes_nunchaku_packed_svdq_tensors(tmp_path):
    torch.manual_seed(0)
    model = AlignedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="proj",
                modules=["proj"],
                export_name="proj",
                quant=SvdqTargetQuant(precision="fp4"),
            )
        ]
    )
    output = tmp_path / "aligned_fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(
                enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(
        result.checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        qweight = handle.get_tensor("proj.qweight")
        wscales = handle.get_tensor("proj.wscales")
        wcscales = handle.get_tensor("proj.wcscales")
        wtscale = handle.get_tensor("proj.wtscale")
        smooth = handle.get_tensor("proj.smooth_factor")
        bias = handle.get_tensor("proj.bias")
        proj_down = handle.get_tensor("proj.proj_down")
        proj_up = handle.get_tensor("proj.proj_up")
    metadata = _config_metadata(result.checkpoint_path)
    checkpoint_metadata = _checkpoint_quantization_config(result.checkpoint_path)

    assert qweight.shape == (128, 64)
    assert qweight.dtype == torch.int8
    assert wscales.shape == (8, 128)
    assert wscales.dtype == torch.float8_e4m3fn
    assert wcscales.shape == (128,)
    assert torch.all(wcscales.float() == 1)
    assert wtscale.shape == (1,)
    assert smooth.shape == (128,)
    assert bias.shape == (128,)
    assert proj_down.shape == (128, 16)
    assert proj_up.shape == (128, 16)
    assert metadata["targets"][0]["weight_scale_layout"] == "nvfp4_deepcompressor"
    assert metadata["targets"][0]["runtime_tensor_layout"] == "nunchaku_packed"
    assert metadata["runtime_manifest_diagnostics"] == {"emitted": True, "reasons": []}
    _assert_checkpoint_quantization_config(
        checkpoint_metadata, metadata, has_runtime_manifest=True
    )
    manifest = checkpoint_metadata["runtime_manifest"]
    assert manifest["schema"] == "nunchaku_lite.runtime_manifest"
    assert manifest["version"] == 1
    assert manifest["nunchaku_format_version"] == 1
    assert manifest["requirements"]["precision"] == "fp4"
    assert manifest["requirements"]["weight_dtype"] == "fp4_e2m1_all"
    assert manifest["structural_patches"] == []
    assert manifest["targets"][0]["checkpoint_prefix"] == "proj"
    assert manifest["targets"][0]["nunchaku_op"] == "svdq_w4a4"
    assert manifest["targets"][0]["precision"] == "fp4"
    assert manifest["targets"][0]["rank"] == 16
    assert manifest["targets"][0]["has_bias"] is True


def test_naive_svdq_layout_forces_logical_tensors_for_aligned_target(tmp_path, caplog):
    torch.manual_seed(0)
    caplog.set_level(logging.WARNING, logger="diffuse_compressor.exporters.nunchaku")
    model = AlignedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="proj",
                modules=["proj"],
                export_name="proj",
                quant=SvdqTargetQuant(precision="fp4", weight_layout=NaiveSvdqLayout()),
            )
        ]
    )
    output = tmp_path / "naive_fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(
                enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(
        result.checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        qweight = handle.get_tensor("proj.qweight")
        wscales = handle.get_tensor("proj.wscales")
    metadata = _config_metadata(result.checkpoint_path)
    checkpoint_metadata = _checkpoint_quantization_config(result.checkpoint_path)

    assert qweight.shape == (128, 64)
    assert wscales.shape == (8, 128)
    assert metadata["targets"][0]["weight_layout"] == {"name": "naive_svdq"}
    assert metadata["targets"][0]["runtime_tensor_layout"] == "logical"
    assert "runtime_manifest" not in metadata
    diagnostics = metadata["runtime_manifest_diagnostics"]
    assert diagnostics["emitted"] is False
    assert any(
        "requires Nunchaku-packed SVDQ tensor layout" in reason["reason"]
        for reason in diagnostics["reasons"]
    )
    assert any(
        "requires Nunchaku-packed SVDQ tensor layout" in record.message
        for record in caplog.records
    )
    _assert_checkpoint_quantization_config(checkpoint_metadata, metadata)


def test_aligned_nvfp4_export_respects_nunchaku_svdq_layout_outer_scale_splits(
    tmp_path,
):
    torch.manual_seed(0)
    model = AlignedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="proj",
                modules=["proj"],
                export_name="proj",
                quant=SvdqTargetQuant(
                    precision="fp4",
                    weight_layout=NunchakuSvdqLayout(outer_scale_splits=(64, 64)),
                ),
            )
        ]
    )
    output = tmp_path / "aligned_fp4_split_scales.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(
                enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(
        result.checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
        wcscales = handle.get_tensor("proj.wcscales")
    metadata = _config_metadata(result.checkpoint_path)
    checkpoint_metadata = _checkpoint_quantization_config(result.checkpoint_path)

    assert "proj.wcscales" in keys
    assert "proj.wtscale" not in keys
    assert wcscales.shape == (128,)
    assert torch.unique(wcscales[:64].float()).numel() == 1
    assert torch.unique(wcscales[64:].float()).numel() == 1
    assert metadata["targets"][0]["weight_layout"] == {
        "name": "nunchaku_svdq",
        "outer_scale_splits": [64, 64],
    }
    manifest_target = checkpoint_metadata["runtime_manifest"]["targets"][0]
    assert manifest_target["nunchaku_op"] == "svdq_w4a4"
    assert manifest_target["op_options"] == {"outer_scale_splits": [64, 64]}


def test_runtime_manifest_records_structural_patches_for_packed_targets(tmp_path):
    torch.manual_seed(0)
    model = WideOutModel().to(torch.bfloat16)
    target_config = TargetConfig(
        patches=[
            PatchRule(type="split_linear_output", module="proj", args={"splits": [128]})
        ],
        targets=[
            TargetRule(
                name="proj0",
                modules=["proj.linears.0"],
                export_name="proj.linears.0",
                quant=SvdqTargetQuant(precision="fp4"),
            )
        ],
    )
    output = tmp_path / "split_manifest.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(
                enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(
        result.checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
    metadata = _config_metadata(result.checkpoint_path)
    checkpoint_metadata = _checkpoint_quantization_config(result.checkpoint_path)

    assert "proj.linears.0.qweight" in keys
    assert metadata["structural_patches"] == [
        {"type": "split_linear_output", "module": "proj", "args": {"splits": [128]}}
    ]
    _assert_checkpoint_quantization_config(
        checkpoint_metadata, metadata, has_runtime_manifest=True
    )
    assert checkpoint_metadata["runtime_manifest"]["structural_patches"] == [
        {"type": "split_linear_output", "module": "proj", "args": {"splits": [128]}}
    ]
    assert (
        checkpoint_metadata["runtime_manifest"]["targets"][0]["checkpoint_prefix"]
        == "proj.linears.0"
    )
    assert checkpoint_metadata["runtime_manifest"]["targets"][0]["source_modules"] == [
        "proj.linears.0"
    ]


def test_runtime_manifest_omits_grouped_synthetic_targets(tmp_path, caplog):
    torch.manual_seed(0)
    caplog.set_level(logging.WARNING, logger="diffuse_compressor.exporters.nunchaku")
    model = WideOutModel().to(torch.bfloat16)
    target_config = TargetConfig(
        patches=[
            PatchRule(type="split_linear_output", module="proj", args={"splits": [128]})
        ],
        targets=[
            TargetRule(
                name="proj",
                modules=["proj.linears.0", "proj.linears.1"],
                export_name="proj",
                quant=SvdqTargetQuant(precision="fp4"),
            )
        ],
    )
    output = tmp_path / "split_synthetic_manifest.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(
                enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.rand(4, 128, dtype=torch.bfloat16)}]),
        ExportSpec(output=output),
    )

    metadata = _config_metadata(result.checkpoint_path)

    assert metadata["targets"][0]["export_name"] == "proj"
    assert metadata["targets"][0]["modules"] == ["proj.linears.0", "proj.linears.1"]
    assert metadata["structural_patches"] == [
        {"type": "split_linear_output", "module": "proj", "args": {"splits": [128]}}
    ]
    assert "runtime_manifest" not in metadata
    diagnostics = metadata["runtime_manifest_diagnostics"]
    assert diagnostics["emitted"] is False
    assert any(
        "grouped target has 2 source modules" in reason["reason"]
        for reason in diagnostics["reasons"]
    )
    assert any(
        "grouped target has 2 source modules" in record.message
        for record in caplog.records
    )
    _assert_checkpoint_quantization_config(
        _checkpoint_quantization_config(result.checkpoint_path), metadata
    )


def test_shifted_aligned_nvfp4_export_stays_nunchaku_packed(tmp_path):
    torch.manual_seed(0)
    model = AlignedModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="proj",
                modules=["proj"],
                export_name="proj",
                quant=SvdqTargetQuant(precision="fp4"),
            )
        ]
    )
    output = tmp_path / "shifted_aligned_fp4.safetensors"

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=16,
            group_size=16,
            smooth=False,
            shift_activations=True,
            weight_scale_dtypes=(None, "sfp8_e4m3_nan"),
            activation_quant=ActivationQuantSpec(
                enabled=True, scale_dtypes=("sfp8_e4m3_nan",)
            ),
        ),
        target_config,
        CalibrationSpec(samples=[{"x": torch.randn(4, 128, dtype=torch.bfloat16) - 4}]),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(
        result.checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        qweight = handle.get_tensor("proj.qweight")
        bias = handle.get_tensor("proj.bias")
        proj_down = handle.get_tensor("proj.proj_down")
    metadata = _config_metadata(result.checkpoint_path)
    checkpoint_metadata = _checkpoint_quantization_config(result.checkpoint_path)

    shifts = metadata["calibration"]["activation_shifts"]
    assert shifts["proj"] > 0
    assert set(metadata["calibration"]) == {"activation_shifts"}
    assert qweight.shape == (128, 64)
    assert bias.shape == (128,)
    assert proj_down.shape == (128, 16)
    assert metadata["targets"][0]["runtime_tensor_layout"] == "nunchaku_packed"
    assert metadata["runtime_manifest_diagnostics"] == {"emitted": True, "reasons": []}
    _assert_checkpoint_quantization_config(
        checkpoint_metadata, metadata, has_runtime_manifest=True
    )


def test_pointwise_conv_target_quantizes_and_records_activation_range_metadata(
    tmp_path, caplog
):
    torch.manual_seed(0)
    caplog.set_level(logging.WARNING, logger="diffuse_compressor.exporters.nunchaku")
    model = TinyConvModel().to(torch.bfloat16)
    output = tmp_path / "conv.safetensors"
    target_config = TargetConfig(
        targets=[
            TargetRule(name="proj", modules=["proj"], export_name="proj", kind="conv")
        ]
    )

    result = quantize_and_export(
        model,
        DiffusionQuantSpec(
            rank=4,
            group_size=64,
            smooth=False,
            activation_quant=ActivationQuantSpec(
                enabled=True,
                inputs=RangeCalibrationSpec(granularity="channel"),
                outputs=RangeCalibrationSpec(granularity="channel"),
            ),
        ),
        target_config,
        CalibrationSpec(
            samples=[{"x": torch.randn(1, 64, 4, 4, dtype=torch.bfloat16)}]
        ),
        ExportSpec(output=output),
    )

    with safetensors.safe_open(
        result.checkpoint_path, framework="pt", device="cpu"
    ) as handle:
        keys = set(handle.keys())
    metadata = _config_metadata(result.checkpoint_path)

    assert "proj.qweight" in keys
    assert "proj.input_scale" not in keys
    assert "proj.output_scale" not in keys
    assert metadata["targets"][0]["modules"] == ["proj"]
    assert metadata["targets"][0]["activation_quant"]["inputs"]["calibrated"] is True
    assert metadata["targets"][0]["activation_quant"]["outputs"]["calibrated"] is True
    assert metadata["calibration"] == {"activation_shifts": {}}
    diagnostics = metadata["runtime_manifest_diagnostics"]
    assert diagnostics["emitted"] is False
    assert any(
        "manifest v1 supports only linear targets" in reason["reason"]
        for reason in diagnostics["reasons"]
    )
    assert any(
        "manifest v1 supports only linear targets" in record.message
        for record in caplog.records
    )
    _assert_checkpoint_quantization_config(
        _checkpoint_quantization_config(result.checkpoint_path), metadata
    )


def test_non_pointwise_conv_target_is_rejected(tmp_path):
    model = TinyConvModel(kernel_size=3, padding=1).to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(name="proj", modules=["proj"], export_name="proj", kind="conv")
        ]
    )

    try:
        quantize_and_export(
            model,
            DiffusionQuantSpec(rank=4, group_size=64, smooth=False),
            target_config,
            calibration=None,
            export=ExportSpec(output=tmp_path / "conv3.safetensors"),
        )
    except NotImplementedError as exc:
        assert "kernel_size=(1, 1)" in str(exc)
    else:
        raise AssertionError("Expected non-pointwise Conv2d target to be rejected")
