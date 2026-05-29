import pytest
import torch
from torch import nn

from diffuse_compressor import CalibrationSpec, DiffusionQuantSpec, SmoothSpec, TargetConfig, TargetRule
from diffuse_compressor.api import collect_quant_targets, quantize_diffusion
import diffuse_compressor.methods.svdquant.smoothing as smoothing_module
from diffuse_compressor.methods.svdquant.smoothing import (
    GridSmoothSearchStrategy,
    ManualSmoothSearchStrategy,
    RandomSmoothSearchStrategy,
    SmoothCandidate,
    SmoothEvaluation,
    SmoothSearchResult,
    build_smooth_span_contexts,
    build_smooth_span_contexts_from_partitions,
    resolve_smooth_search_strategy,
)
from diffuse_compressor.targets import QuantTarget


class SmoothTinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 3, bias=False)
        self.k = nn.Linear(4, 3, bias=False)
        self.v = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.q(x)


def test_grid_smooth_search_strategy_matches_deepcompressor_beta_minus_two():
    spec = SmoothSpec(strategy="grid_search", alpha=0.5, beta=-2, num_grids=4)
    inputs = torch.ones(2, 4)
    weight = torch.ones(3, 4)
    span_contexts = build_smooth_span_contexts(inputs, weight, spec)
    evaluated = []

    def evaluate_candidates(candidates):
        evaluated.extend(candidates)
        return tuple(
            SmoothEvaluation(candidate=candidate, error=torch.tensor(float(index)))
            for index, candidate in enumerate(candidates)
        )

    result = GridSmoothSearchStrategy().search(spec, span_contexts, evaluate_candidates)
    pairs = [(candidate.alpha, candidate.beta) for candidate in evaluated]

    assert pairs == [
        (0, 0),
        (0.25, 0),
        (0.5, 0),
        (0.75, 0),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
    ]
    assert result.num_candidates == 7
    assert result.best_candidate is evaluated[0]


def test_smooth_spec_validates_objective():
    with pytest.raises(ValueError, match="Unsupported smoothing objective"):
        SmoothSpec(objective="tensor_error")  # type: ignore[arg-type]


def test_smooth_spec_validates_random_search_options():
    spec = SmoothSpec(strategy="random_search", strategy_options={"random_samples": 3})

    assert spec.strategy == "random_search"
    assert spec.strategy_options["random_samples"] == 3

    with pytest.raises(ValueError, match="Unsupported smooth strategy_options"):
        SmoothSpec(strategy="grid_search", strategy_options={"random_samples": 3})
    with pytest.raises(ValueError, match="random_samples"):
        SmoothSpec(strategy="random_search", strategy_options={"random_samples": 0})


def test_manual_smooth_search_strategy_supports_absmax_and_rms_spans():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([[2.0, 8.0], [4.0, 16.0]])
    spec = SmoothSpec(strategy="manual", alpha=0.5, beta=0.5, spans=(("rms", "absmax"),))
    span_contexts = build_smooth_span_contexts(inputs, weight, spec)
    evaluated = []

    def evaluate_candidates(candidates):
        evaluated.extend(candidates)
        return tuple(
            SmoothEvaluation(candidate=candidate, error=torch.tensor(float(index)))
            for index, candidate in enumerate(candidates)
        )

    result = ManualSmoothSearchStrategy().search(spec, span_contexts, evaluate_candidates)
    candidate = evaluated[0]

    expected_input = inputs.pow(2).mean(dim=0).sqrt()
    expected_weight = weight.abs().amax(dim=0)
    assert torch.allclose(candidate.scale, expected_input.sqrt() / expected_weight.sqrt())
    assert candidate.span == ("rms", "absmax")
    assert result.best_candidate is candidate


def test_partitioned_smooth_span_contexts_match_full_inputs():
    inputs = torch.tensor([[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]])
    partitions = (inputs[:1], inputs[1:])
    weight = torch.tensor([[2.0, 8.0], [4.0, 16.0]])
    spec = SmoothSpec(strategy="manual", alpha=0.5, beta=0.5, spans=(("absmax", "absmax"), ("rms", "rms")))

    full = build_smooth_span_contexts(inputs, weight, spec)
    partitioned = build_smooth_span_contexts_from_partitions(partitions, weight, spec)

    assert len(partitioned) == len(full)
    for actual, expected in zip(partitioned, full, strict=True):
        assert actual.span == expected.span
        assert torch.allclose(actual.alpha_span, expected.alpha_span)
        assert torch.allclose(actual.beta_span, expected.beta_span)


def test_smooth_search_strategy_resolver():
    assert isinstance(resolve_smooth_search_strategy(SmoothSpec(strategy="manual")), ManualSmoothSearchStrategy)
    assert isinstance(resolve_smooth_search_strategy(SmoothSpec(strategy="grid_search")), GridSmoothSearchStrategy)
    assert isinstance(resolve_smooth_search_strategy(SmoothSpec(strategy="random_search")), RandomSmoothSearchStrategy)


def test_smooth_search_strategy_uses_supplied_evaluator():
    spec = SmoothSpec(strategy="grid_search", alpha=0.5, beta=0, num_grids=3)
    span_contexts = build_smooth_span_contexts(torch.ones(2, 4), torch.ones(3, 4), spec)
    seen_candidate_counts = []

    def evaluate_candidates(candidates):
        seen_candidate_counts.append(len(candidates))
        return tuple(
            SmoothEvaluation(candidate=candidate, error=torch.tensor(0.0 if index == 2 else float(index + 1)))
            for index, candidate in enumerate(candidates)
        )

    result = GridSmoothSearchStrategy().search(spec, span_contexts, evaluate_candidates)

    assert seen_candidate_counts == [3]
    assert result.num_candidates == 3
    assert result.best_candidate is not None
    assert result.best_candidate.alpha == 2 / 3


def test_random_smooth_search_strategy_is_deterministic_and_keeps_identity():
    spec = SmoothSpec(strategy="random_search", alpha=0.5, beta=-2, num_grids=6, strategy_options={"random_samples": 4})
    span_contexts = build_smooth_span_contexts(torch.ones(2, 4), torch.ones(3, 4), spec)
    evaluated_runs = []

    def capture_run(seed):
        evaluated = []

        def evaluate_candidates(candidates):
            evaluated.extend(candidates)
            return tuple(
                SmoothEvaluation(candidate=candidate, error=torch.tensor(float(index)))
                for index, candidate in enumerate(candidates)
            )

        result = RandomSmoothSearchStrategy(seed=seed).search(spec, span_contexts, evaluate_candidates)
        evaluated_runs.append([(candidate.alpha, candidate.beta) for candidate in evaluated])
        return result

    first = capture_run(11)
    second = capture_run(11)

    assert evaluated_runs[0] == evaluated_runs[1]
    assert evaluated_runs[0][0] == (0, 0)
    assert len(evaluated_runs[0]) == 4
    assert first.metadata == {"samples": 4, "actual_samples": 4, "seed": 11}
    assert second.metadata == first.metadata


def test_random_smooth_search_caps_samples_to_candidate_pool():
    spec = SmoothSpec(strategy="random_search", alpha=0.5, beta=-2, num_grids=4, strategy_options={"random_samples": 100})
    span_contexts = build_smooth_span_contexts(torch.ones(2, 4), torch.ones(3, 4), spec)
    evaluated = []

    def evaluate_candidates(candidates):
        evaluated.extend(candidates)
        return tuple(
            SmoothEvaluation(candidate=candidate, error=torch.tensor(float(index)))
            for index, candidate in enumerate(candidates)
        )

    result = RandomSmoothSearchStrategy(seed=0).search(spec, span_contexts, evaluate_candidates)

    assert len(evaluated) == 7
    assert result.metadata == {"samples": 100, "actual_samples": 7, "seed": 0}


def test_smoothing_search_selects_lowest_output_error_candidate(monkeypatch):
    target = QuantTarget(
        name="q",
        modules=(),
        module_names=(),
        export_name="q_proj",
        shared_low_rank=False,
    )
    spec = DiffusionQuantSpec(rank=0, group_size=4, smooth=SmoothSpec(strategy="grid_search"))
    weight = torch.ones(2, 4)
    inputs = torch.ones(3, 4)
    candidates = [
        SmoothCandidate(scale=torch.ones(4), alpha=0.0, beta=0.0, span=("absmax", "absmax")),
        SmoothCandidate(scale=torch.full((4,), 2.0), alpha=0.5, beta=0.0, span=("absmax", "absmax")),
    ]

    class FakeStrategy:
        def search(self, _spec, _span_contexts, evaluate_candidates):
            evaluations = evaluate_candidates(candidates)
            best = min(evaluations, key=lambda item: float(item.error))
            return SmoothSearchResult(
                best_candidate=best.candidate,
                best_error=best.error,
                num_candidates=len(evaluations),
            )

    def fake_error(smooth, *_args):
        return torch.tensor(2.0 if float(smooth[0]) == 1.0 else 0.5)

    monkeypatch.setattr(smoothing_module, "resolve_smooth_search_strategy", lambda _spec, **_kwargs: FakeStrategy())
    monkeypatch.setattr(smoothing_module, "_candidate_output_error", fake_error)

    smooth, metadata = smoothing_module._select_smooth_scale(
        target,
        spec,
        weight,
        bias=None,
        calibration_inputs=inputs,
        calibration_input_partitions=(inputs,),
    )

    assert torch.allclose(smooth, torch.full((4,), 2.0))
    assert metadata["objective"] == "outputs_error"
    assert metadata["num_candidates"] == 2
    assert metadata["error"] == 0.5
    assert metadata["alpha"] == 0.5


def test_random_smoothing_search_records_metadata(monkeypatch):
    target = QuantTarget(
        name="q",
        modules=(),
        module_names=(),
        export_name="q_proj",
        shared_low_rank=False,
    )
    spec = DiffusionQuantSpec(
        rank=0,
        group_size=4,
        smooth=SmoothSpec(strategy="random_search", num_grids=4, strategy_options={"random_samples": 3}),
    )
    weight = torch.ones(2, 4)
    inputs = torch.ones(3, 4)

    monkeypatch.setattr(smoothing_module, "_candidate_output_error", lambda smooth, *_args: smooth.float().mean())

    smooth, metadata = smoothing_module._select_smooth_scale(
        target,
        spec,
        weight,
        bias=None,
        calibration_inputs=inputs,
        calibration_input_partitions=(inputs,),
        seed=7,
    )

    assert smooth.shape == (4,)
    assert metadata["strategy"] == "random_search"
    assert metadata["num_candidates"] == 3
    assert metadata["search"] == {"samples": 3, "actual_samples": 3, "seed": 7}


def test_calibrated_smoothing_exports_non_identity_scale():
    torch.manual_seed(0)
    model = SmoothTinyModel().to(torch.bfloat16)
    with torch.no_grad():
        model.q.weight.fill_(1)
    target_config = TargetConfig(
        targets=[TargetRule(name="q", modules=["q"], export_name="q_proj")],
    )
    targets = collect_quant_targets(model, target_config)
    samples = [{"x": torch.tensor([[16.0, 4.0, 1.0, 0.25]], dtype=torch.bfloat16)}]

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=4,
            smooth=SmoothSpec(strategy="manual", alpha=0.5, beta=-1),
        ),
        targets,
        calibration=CalibrationSpec(samples=samples),
        target_config=target_config,
    )

    smooth = artifact.quantized_targets[0].state_dict["smooth_factor"]
    metadata = artifact.quantized_targets[0].metadata["smooth"]
    assert metadata["searched"] is True
    assert metadata["objective"] == "outputs_error"
    assert not torch.allclose(smooth, torch.ones_like(smooth))


def test_fp4_smoothing_search_uses_fake_quantization():
    torch.manual_seed(0)
    model = SmoothTinyModel().to(torch.bfloat16)
    target_config = TargetConfig(targets=[TargetRule(name="q", modules=["q"], export_name="q_proj")])
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            precision="fp4",
            rank=0,
            group_size=4,
            smooth=SmoothSpec(strategy="manual", alpha=0.5, beta=-1),
        ),
        targets,
        calibration=CalibrationSpec(samples=[{"x": torch.randn(2, 4, dtype=torch.bfloat16)}]),
        target_config=target_config,
    )

    target = artifact.quantized_targets[0]
    assert target.metadata["precision"] == "fp4"
    assert target.metadata["smooth"]["searched"] is True
    assert "qweight" in target.state_dict


def test_disabled_smoothing_exports_identity_scale():
    model = SmoothTinyModel()
    target_config = TargetConfig(targets=[TargetRule(name="q", modules=["q"], export_name="q_proj")])
    targets = collect_quant_targets(model, target_config)

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(rank=0, group_size=4, smooth=False),
        targets,
        calibration=None,
        target_config=target_config,
    )

    smooth = artifact.quantized_targets[0].state_dict["smooth_factor"]
    assert torch.allclose(smooth, torch.ones_like(smooth))
    assert artifact.quantized_targets[0].metadata["smooth"]["enabled"] is False


def test_grouped_qkv_target_uses_one_shared_smooth_vector():
    model = SmoothTinyModel().to(torch.bfloat16)
    target_config = TargetConfig(
        targets=[
            TargetRule(
                name="qkv",
                modules=["q", "k", "v"],
                export_name="qkv_proj",
                roles=["q", "k", "v"],
            )
        ]
    )
    targets = collect_quant_targets(model, target_config)
    samples = [{"x": torch.tensor([[8.0, 2.0, 1.0, 0.5]], dtype=torch.bfloat16)}]

    artifact = quantize_diffusion(
        model,
        DiffusionQuantSpec(
            rank=0,
            group_size=4,
            smooth=SmoothSpec(strategy="manual", alpha=0.5, beta=-1),
        ),
        targets,
        calibration=CalibrationSpec(samples=samples),
        target_config=target_config,
    )

    smooth = artifact.quantized_targets[0].state_dict["smooth_factor"]
    assert smooth.shape == (4,)
    assert artifact.quantized_targets[0].metadata["source_modules"] == ["q", "k", "v"]
