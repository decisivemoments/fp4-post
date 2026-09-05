"""Validation for the post-quantization, single-D-store experiment."""

from __future__ import annotations

import pytest
import torch
from transformer_engine.pytorch.cpp_extensions import general_gemm

from Metis.Metis.fused_residual_lowrank_bf16 import (
    fused_residual_lowrank_bf16,
    is_available,
)
from Metis.Metis.native_nvfp4 import _get_quantizer


pytestmark = pytest.mark.skipif(
    not is_available(), reason="SM120 RTX 50 GPU is required"
)


def test_fused_residual_lowrank_matches_separate_post_quant_forward():
    torch.manual_seed(20260904)
    m, n, k, rank = 128, 128, 128, 64
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(m, rank, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16)
    quantizer = _get_quantizer(x.device, stochastic_rounding=False, rowwise=True, columnwise=False)
    actual_x, actual_residual = quantizer.quantize(x), quantizer.quantize(residual)
    ref_x, ref_residual = quantizer.quantize(x), quantizer.quantize(residual)

    expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    general_gemm(ref_residual, ref_x, out_dtype=torch.bfloat16, layout="TN", out=expected)
    expected.addmm_(z, u.T)

    actual = torch.empty_like(expected)
    fused_residual_lowrank_bf16(actual_x, actual_residual, z, u, actual)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)


def test_fused_epilogue_preserves_residual_fp4_gemm_when_lowrank_is_zero():
    """Separates FP4 scaling/epilogue correctness from coordinate mapping."""
    torch.manual_seed(20260905)
    m, n, k = 128, 128, 128
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    z = torch.zeros(m, 64, device="cuda", dtype=torch.bfloat16)
    u = torch.zeros(n, 64, device="cuda", dtype=torch.bfloat16)
    quantizer = _get_quantizer(x.device, stochastic_rounding=False, rowwise=True, columnwise=False)
    actual_x, actual_residual = quantizer.quantize(x), quantizer.quantize(residual)
    ref_x, ref_residual = quantizer.quantize(x), quantizer.quantize(residual)
    expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    general_gemm(ref_residual, ref_x, out_dtype=torch.bfloat16, layout="TN", out=expected)
    actual = torch.empty_like(expected)
    fused_residual_lowrank_bf16(actual_x, actual_residual, z, u, actual)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1.25e-1)


def test_fused_epilogue_lowrank_coordinate_mapping():
    """Checks the register fragment to output-coordinate mapping in isolation."""
    m, n, k = 128, 128, 128
    x = torch.zeros(m, k, device="cuda", dtype=torch.bfloat16)
    residual = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    z = torch.arange(m, device="cuda", dtype=torch.float32).to(torch.bfloat16)[:, None].repeat(1, 64)
    u = torch.arange(n, device="cuda", dtype=torch.float32).to(torch.bfloat16)[:, None].repeat(1, 64)
    quantizer = _get_quantizer(x.device, stochastic_rounding=False, rowwise=True, columnwise=False)
    packed_x, packed_residual = quantizer.quantize(x), quantizer.quantize(residual)
    expected = z @ u.T
    actual = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    fused_residual_lowrank_bf16(packed_x, packed_residual, z, u, actual)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2.0)


def test_fused_epilogue_multiple_cta_tiles():
    """Exercise tile_m/tile_n offsets, not just the first 128x128 CTA."""
    torch.manual_seed(20260906)
    m, n, k = 256, 256, 128
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(m, 64, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(n, 64, device="cuda", dtype=torch.bfloat16)
    quantizer = _get_quantizer(x.device, stochastic_rounding=False, rowwise=True, columnwise=False)
    actual_x, actual_residual = quantizer.quantize(x), quantizer.quantize(residual)
    ref_x, ref_residual = quantizer.quantize(x), quantizer.quantize(residual)
    expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    general_gemm(ref_residual, ref_x, out_dtype=torch.bfloat16, layout="TN", out=expected)
    expected.addmm_(z, u.T)
    actual = torch.empty_like(expected)
    fused_residual_lowrank_bf16(actual_x, actual_residual, z, u, actual)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2.5e-1)
