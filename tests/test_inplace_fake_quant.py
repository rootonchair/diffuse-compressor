"""In-place activation fake-quant (torch-dequant runtime) matches the out-of-place path."""

import pytest
import torch

from diffuse_compressor.backends.nunchaku.packing import dynamic_fake_quantize_activation


@pytest.mark.parametrize("float_point,group_size", [(True, 16), (False, 64)])
def test_inplace_fake_quant_matches_out_of_place(float_point, group_size):
    torch.manual_seed(0)
    values = torch.randn(37, 4 * group_size) * 3
    expected = dynamic_fake_quantize_activation(values, group_size=group_size, float_point=float_point)
    actual = dynamic_fake_quantize_activation(values.clone(), group_size=group_size, float_point=float_point, inplace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("float_point,group_size", [(True, 16), (False, 64)])
def test_inplace_fake_quant_handles_strided_input(float_point, group_size):
    torch.manual_seed(0)
    big = torch.randn(64, 2 * group_size) * 3
    strided = big[::2]
    assert not strided.is_contiguous()
    expected = dynamic_fake_quantize_activation(strided.contiguous(), group_size=group_size, float_point=float_point)
    actual = dynamic_fake_quantize_activation(strided, group_size=group_size, float_point=float_point, inplace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_inplace_fp4_fake_quant_crosses_chunk_boundary():
    torch.manual_seed(0)
    values = torch.randn(8_000_000 // 64 + 3, 64)  # just over one 8M-element chunk
    values[0, 0] = float("inf")
    values[5, 7] = float("nan")
    expected = dynamic_fake_quantize_activation(values, group_size=16, float_point=True)
    actual = dynamic_fake_quantize_activation(values.clone(), group_size=16, float_point=True, inplace=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
