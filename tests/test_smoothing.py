import torch
from torch import nn

from diffuse_compressor import CalibrationSpec, DiffusionQuantSpec, SmoothSpec, TargetConfig, TargetRule
from diffuse_compressor.api import collect_quant_targets, quantize_diffusion
from diffuse_compressor.methods.svdquant.smoothing import iter_smooth_candidates, smooth_alpha_beta_pairs


class SmoothTinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4, 3, bias=False)
        self.k = nn.Linear(4, 3, bias=False)
        self.v = nn.Linear(4, 3, bias=False)

    def forward(self, x):
        return self.q(x)


def test_smooth_alpha_beta_pairs_match_deepcompressor_beta_minus_two():
    spec = SmoothSpec(strategy="grid_search", alpha=0.5, beta=-2, num_grids=4)

    pairs = smooth_alpha_beta_pairs(spec)

    assert pairs == [
        (0, 0),
        (0.25, 0),
        (0.5, 0),
        (0.75, 0),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
    ]


def test_iter_smooth_candidates_supports_absmax_and_rms_spans():
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weight = torch.tensor([[2.0, 8.0], [4.0, 16.0]])
    spec = SmoothSpec(strategy="manual", alpha=0.5, beta=0.5, spans=(("rms", "absmax"),))

    candidate = next(iter_smooth_candidates(inputs, weight, spec))

    expected_input = inputs.pow(2).mean(dim=0).sqrt()
    expected_weight = weight.abs().amax(dim=0)
    assert torch.allclose(candidate.scale, expected_input.sqrt() / expected_weight.sqrt())
    assert candidate.span == ("rms", "absmax")


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
    assert artifact.quantized_targets[0].metadata["smooth"]["searched"] is True
    assert not torch.allclose(smooth, torch.ones_like(smooth))


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
