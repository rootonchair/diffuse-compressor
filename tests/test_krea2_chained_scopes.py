"""The Krea2 script may change calibration scopes but never the exported targets.

Changing the target set would change the checkpoint ABI and break runtime-manifest
compatibility, so that equality is the invariant worth pinning.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from examples.text_to_image.quantize_hf import scan_linear_targets
from examples.text_to_image.quantize_krea2_turbo import (
    _krea2_block_prev_replay_transform,
    krea2_calibration_scopes,
    krea2_target_config,
)


class Block(nn.Module):
    """Stands in for a Krea2 block: two quantizable linears, single-tensor output."""

    def __init__(self, dim=128):
        super().__init__()
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(dim, dim, bias=False)
        self.ff = nn.Module()
        self.ff.up = nn.Linear(dim, dim, bias=False)

    def forward(self, hidden_states, **_):
        return self.ff.up(self.attn.to_q(hidden_states))


class Stack(nn.Module):
    def __init__(self, n=4, dim=128):
        super().__init__()
        self.transformer_blocks = nn.ModuleList(Block(dim) for _ in range(n))

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        return x


def test_target_set_is_identical_to_the_generic_scanner():
    model = Stack()
    generic = scan_linear_targets(model, precision="int4", rank=32)
    chained, scan = krea2_target_config(model, precision="int4", rank=32)

    generic_names = {rule.export_name for rule in generic.target_config.targets}
    chained_names = {rule.export_name for rule in chained.targets}
    assert chained_names == generic_names
    assert scan.svdq_targets == generic.svdq_targets
    # Only the scopes may differ.
    assert chained.calibration_scopes != generic.target_config.calibration_scopes
    assert not chained.patches


def test_each_stack_head_starts_a_fresh_chain():
    scopes = krea2_calibration_scopes()
    heads = [s for s in scopes if not s.use_prev_scope_outputs]
    chained = [s for s in scopes if s.prev_replay_transform is not None]

    # One unchained head and one chained rule per stack; the stacks carry
    # different tensors, so a head must never consume the previous stack.
    assert len(heads) == 3
    assert len(chained) == 3
    assert {s.modules[0] for s in heads} == {
        "text_fusion.layerwise_blocks.0",
        "text_fusion.refiner_blocks.0",
        "transformer_blocks.0",
    }
    assert all(s.use_prev_scope_outputs for s in chained)


def test_prev_replay_transform_substitutes_only_hidden_states():
    temb, rotary = torch.zeros(1, 6), ("cos", "sin")
    out = torch.ones(2, 4)

    replay = SimpleNamespace(args=(), kwargs={"hidden_states": torch.zeros(2, 4), "temb": temb,
                                             "image_rotary_emb": rotary}, output=out)
    args, kwargs = _krea2_block_prev_replay_transform(replay)
    assert args == ()
    assert kwargs["hidden_states"] is out
    assert kwargs["temb"] is temb and kwargs["image_rotary_emb"] is rotary

    positional = SimpleNamespace(args=(torch.zeros(2, 4), temb, rotary), kwargs={}, output=out)
    args, kwargs = _krea2_block_prev_replay_transform(positional)
    assert args[0] is out and args[1] is temb and args[2] is rotary
    assert kwargs == {}


def test_prev_replay_transform_rejects_an_empty_record():
    with pytest.raises(TypeError, match="hidden_states"):
        _krea2_block_prev_replay_transform(SimpleNamespace(args=(), kwargs={}, output=torch.zeros(1)))


def test_default_model_id_makes_the_positional_argument_optional():
    """The script runs argument-free; the generic scanner still requires a model id."""

    from examples.text_to_image.quantize_hf import build_parser
    from examples.text_to_image.quantize_krea2_turbo import MODEL_ID

    assert build_parser(MODEL_ID).parse_args([]).model_id == MODEL_ID
    assert build_parser(MODEL_ID).parse_args(["other/model"]).model_id == "other/model"
    assert build_parser().parse_args(["some/model"]).model_id == "some/model"
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
