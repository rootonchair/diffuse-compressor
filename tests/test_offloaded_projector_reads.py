"""Direct projector weight reads must work while Accelerate offload hooks stay attached."""

import pytest
import torch
from torch import nn

from diffuse_compressor import AwqTargetQuant, TargetConfig, TargetRule, collect_quant_targets
from diffuse_compressor.config import DiffusionQuantSpec
from diffuse_compressor.logging import QuantizationLogger
from diffuse_compressor.quantize import _projector_context


class OffloadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(64, 16, bias=True)

    def forward(self, x):
        return self.proj(x)


def _target_config() -> TargetConfig:
    return TargetConfig(targets=[TargetRule(modules=["proj"], quant=AwqTargetQuant())])


def test_projector_context_reads_accelerate_offloaded_weights():
    """Reproduces the 'Cannot copy out of meta tensor' crash on an offloaded model."""

    cpu_offload = pytest.importorskip("accelerate").cpu_offload

    model = OffloadModel()
    expected_weight = model.proj.weight.detach().clone()
    expected_bias = model.proj.bias.detach().clone()

    cpu_offload(model, torch.device("cpu"))
    # Between forward passes an offloaded parameter reads as a meta tensor.
    assert model.proj.weight.device.type == "meta"

    targets = collect_quant_targets(model, _target_config())
    assert len(targets) == 1

    context = _projector_context(
        targets[0],
        DiffusionQuantSpec(),
        None,
        None,
        None,
        None,
        None,
        compute_device=None,
        logger=QuantizationLogger(),
    )

    assert context.weight.device.type != "meta"
    torch.testing.assert_close(context.weight, expected_weight.to(context.weight.dtype))
    assert context.bias is not None
    torch.testing.assert_close(context.bias, expected_bias.to(context.bias.dtype))

    # The hooks must survive so calibration replay can still run forward passes.
    assert getattr(model.proj, "_hf_hook", None) is not None
    assert model.proj.weight.device.type == "meta"
    torch.testing.assert_close(model(torch.zeros(2, 64)), torch.zeros(2, 16) + expected_bias)


def test_projector_context_reads_plain_model_weights():
    """The no-offload path stays unchanged."""

    model = OffloadModel()
    expected_weight = model.proj.weight.detach().clone()

    targets = collect_quant_targets(model, _target_config())
    context = _projector_context(
        targets[0],
        DiffusionQuantSpec(),
        None,
        None,
        None,
        None,
        None,
        compute_device=None,
        logger=QuantizationLogger(),
    )

    torch.testing.assert_close(context.weight, expected_weight.to(context.weight.dtype))


def test_materialized_state_dict_falls_back_when_a_module_stays_on_meta(monkeypatch):
    """A tensor that cannot be materialized must fall back, never export meta."""

    from diffuse_compressor.calibration import utils

    model = OffloadModel()
    # Sequential offload can leave a submodule on meta while the hook owning its
    # weights sits on an ancestor, so aligning that submodule materializes nothing.
    model.proj.weight = nn.Parameter(
        torch.empty_like(model.proj.weight, device="meta"), requires_grad=False
    )
    model._hf_hook = object()  # look offloaded to has_accelerate_hooks()

    monkeypatch.setattr(utils, "has_accelerate_hooks", lambda _m: True)

    assert utils.materialized_state_dict(model) is None


def test_materialized_state_dict_returns_none_without_hooks():
    from diffuse_compressor.calibration.utils import materialized_state_dict

    assert materialized_state_dict(OffloadModel()) is None
