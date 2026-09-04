"""Correctness tests for the Transformer Engine native Full-NVFP4 backend."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from Metis.Metis.native_nvfp4 import (
    NativeActivationGroup,
    NativeFullNVFP4Linear,
    PackedMeanActivation,
    PackedNVFP4Weight,
    _dequantize_columnwise_transpose,
    _get_quantizer,
    _native_mean_backward,
    _native_mean_fprop,
    register_native_nvfp4_optimizer_hook,
    require_native_nvfp4,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Native NVFP4 tests require CUDA",
)

ATOL = 1e-2
RTOL = 1e-2
BACKWARD_RELATIVE_L2_LIMIT = 5e-3


def relative_l2_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = torch.linalg.vector_norm((actual - expected).float())
    reference = torch.linalg.vector_norm(expected.float())
    return float(difference / reference)


def make_packed_weight(weight: torch.Tensor) -> PackedNVFP4Weight:
    quantizer = _get_quantizer(weight.device, stochastic_rounding=False)
    return PackedNVFP4Weight(
        quantized=quantizer.quantize(weight),
        source_version=weight._version,
    )


def test_native_mean_backward_matches_dequantized_reference():
    require_native_nvfp4()
    torch.manual_seed(11)
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)

    packed_x = PackedMeanActivation.from_tensor(x)
    packed_weight = make_packed_weight(weight)
    actual_dx, actual_dw = _native_mean_backward(
        grad,
        packed_x,
        packed_weight,
        stochastic_rounding=False,
    )

    grad_mean = grad.mean(dim=0, keepdim=True)
    grad_residual = grad - grad_mean
    grad_quantizer = _get_quantizer(grad.device, stochastic_rounding=False)
    quantized_grad = grad_quantizer.quantize(grad_residual)
    reconstructed_grad_rowwise = grad_mean + quantized_grad.dequantize(
        dtype=torch.bfloat16
    )
    reconstructed_weight_columnwise = (
        packed_weight.dequantized_transpose().T
    )
    reconstructed_grad_columnwise = grad_mean + (
        _dequantize_columnwise_transpose(quantized_grad).T
    )
    reconstructed_x_columnwise = packed_x.mean + (
        _dequantize_columnwise_transpose(packed_x.quantized_residual).T
    )

    expected_dx = (
        reconstructed_grad_rowwise @ reconstructed_weight_columnwise
    )
    expected_dw = (
        reconstructed_grad_columnwise.T @ reconstructed_x_columnwise
    )
    # The SM120 block-scaled Tensor Core and torch BF16 matmul use different
    # scale-application/reduction orders, so an elementwise comparison is not
    # stable near zero. The aggregate error stays below 0.5% for both native
    # backward GEMMs.
    assert (
        relative_l2_error(actual_dx, expected_dx)
        < BACKWARD_RELATIVE_L2_LIMIT
    )
    assert (
        relative_l2_error(actual_dw, expected_dw)
        < BACKWARD_RELATIVE_L2_LIMIT
    )


def test_native_mean_backward_pads_grpo_rows_without_changing_shape():
    require_native_nvfp4()
    torch.manual_seed(12)
    rows = 48
    x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(rows, 256, device="cuda", dtype=torch.bfloat16)

    packed_x = PackedMeanActivation.from_tensor(x)
    packed_weight = make_packed_weight(weight)
    actual_output = _native_mean_fprop(packed_x, packed_weight)
    actual_dx, actual_dw = _native_mean_backward(
        grad,
        packed_x,
        packed_weight,
        stochastic_rounding=False,
    )

    assert packed_x.rows == rows
    assert packed_x.packed_rows == 64
    assert actual_output.shape == (rows, 256)
    assert actual_dx.shape == x.shape
    assert actual_dw.shape == weight.shape
    assert torch.isfinite(actual_output).all()
    assert torch.isfinite(actual_dx).all()
    assert torch.isfinite(actual_dw).all()


def test_native_full_forward_matches_dequantized_reference():
    require_native_nvfp4()
    torch.manual_seed(13)
    source = torch.nn.Linear(
        128,
        256,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer = NativeFullNVFP4Linear.from_linear(
        source,
        rank=64,
        stochastic_rounding=False,
    )
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    actual = layer(x)

    packed_x = layer.activation_group._packed
    assert packed_x is not None
    packed_residual = layer._weight_cache["residual"][1]
    packed_v = layer._weight_cache["v"][1]
    reconstructed_x = packed_x.mean + packed_x.quantized_residual.dequantize(
        dtype=torch.bfloat16
    )
    residual_output = F.linear(
        reconstructed_x,
        packed_residual.dequantized(),
    )
    v_output = F.linear(reconstructed_x, packed_v.dequantized())
    expected = residual_output + F.linear(
        v_output * layer.singular_values,
        layer.u_weight,
    )
    expected = expected + layer.bias
    expected = expected.reshape_as(actual)
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def test_rank64_svd_reconstructs_source_weight():
    require_native_nvfp4()
    torch.manual_seed(15)
    source = torch.nn.Linear(
        128,
        256,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    expected = source.weight.detach().clone()
    layer = NativeFullNVFP4Linear.from_linear(
        source,
        rank=64,
        stochastic_rounding=False,
    )
    torch.testing.assert_close(
        layer.reconstructed_weight(),
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_native_full_backward_wires_all_parameter_paths():
    require_native_nvfp4()
    torch.manual_seed(16)
    source = torch.nn.Linear(
        128,
        256,
        bias=True,
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer = NativeFullNVFP4Linear.from_linear(
        source,
        rank=64,
        stochastic_rounding=False,
    )
    x = torch.randn(
        2,
        16,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_output = torch.randn(
        2,
        16,
        256,
        device="cuda",
        dtype=torch.bfloat16,
    )
    output = layer(x)

    packed_x = layer.activation_group._packed
    assert packed_x is not None
    packed_residual = layer._weight_cache["residual"][1]
    packed_v = layer._weight_cache["v"][1]
    v_output = _native_mean_fprop(packed_x, packed_v)
    scaled_v_output = v_output * layer.singular_values
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    expected_grad_u = grad_flat.T @ scaled_v_output
    expected_grad_scaled_v = grad_flat @ layer.u_weight
    expected_grad_s = (expected_grad_scaled_v * v_output).sum(dim=0)
    expected_grad_v_output = (
        expected_grad_scaled_v * layer.singular_values
    )
    expected_dx_residual, expected_grad_residual = _native_mean_backward(
        grad_flat,
        packed_x,
        packed_residual,
        stochastic_rounding=False,
    )
    expected_dx_v, expected_grad_v = _native_mean_backward(
        expected_grad_v_output,
        packed_x,
        packed_v,
        stochastic_rounding=False,
    )

    output.backward(grad_output)
    torch.testing.assert_close(
        x.grad,
        (expected_dx_residual + expected_dx_v).reshape_as(x),
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        layer.residual_weight.grad,
        expected_grad_residual,
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        layer.v_weight.grad,
        expected_grad_v,
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        layer.u_weight.grad,
        expected_grad_u,
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        layer.singular_values.grad,
        expected_grad_s,
        atol=ATOL,
        rtol=RTOL,
    )
    torch.testing.assert_close(
        layer.bias.grad,
        grad_flat.sum(dim=0),
        atol=ATOL,
        rtol=RTOL,
    )


def test_shared_activation_group_reuses_one_packed_tensor():
    require_native_nvfp4()
    torch.manual_seed(17)
    group = NativeActivationGroup("shared")
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
    first = group.get(x)
    second = group.get(x)
    assert first is second


def test_native_full_nvfp4_lowrank_nvfp4_is_inference_only():
    require_native_nvfp4()
    torch.manual_seed(18)
    source = torch.nn.Linear(
        128,
        256,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer = NativeFullNVFP4Linear.from_linear(
        source,
        rank=64,
        stochastic_rounding=False,
        lowrank_compute="nvfp4",
    )
    x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="inference-only"):
        layer(x)

    with torch.inference_mode():
        output = layer(x)
    assert output.shape == (2, 16, 256)
    assert torch.isfinite(output).all()
    assert "u" in layer._weight_cache


def test_optimizer_hook_invalidates_fused_adam_weight_cache():
    require_native_nvfp4()
    torch.manual_seed(19)
    source = torch.nn.Linear(
        128,
        256,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    layer = NativeFullNVFP4Linear.from_linear(
        source,
        rank=64,
        stochastic_rounding=False,
    )
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-4, fused=True)
    hook = register_native_nvfp4_optimizer_hook(optimizer, layer)
    try:
        x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
        layer(x).float().square().mean().backward()
        cached_before = layer._weight_cache["residual"][1]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert not layer._weight_cache

        layer(x)
        cached_after = layer._weight_cache["residual"][1]
        assert cached_after is not cached_before
    finally:
        hook.remove()
