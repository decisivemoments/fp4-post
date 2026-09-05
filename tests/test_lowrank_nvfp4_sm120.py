"""Same-precision validation for the project-local SM120 NVFP4 operator."""

from __future__ import annotations

import pytest
import torch
from transformer_engine.pytorch.cpp_extensions import general_gemm

from Metis.Metis.lowrank_nvfp4_sm120 import (
    is_available,
    lowrank_nvfp4_add_sm120,
)
from Metis.Metis.native_nvfp4 import _get_quantizer


pytestmark = pytest.mark.skipif(
    not is_available(), reason="SM120 RTX 50 GPU is required"
)


def test_lowrank_nvfp4_sm120_matches_te_nvfp4_beta_one():
    """Compare the custom path to TE, not BF16 addmm."""
    torch.manual_seed(20260904)
    rows, columns, rank = 128, 128, 64
    scaled_v = torch.randn(rows, rank, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(columns, rank, device="cuda", dtype=torch.bfloat16)
    quantizer = _get_quantizer(
        scaled_v.device,
        stochastic_rounding=False,
        rowwise=True,
        columnwise=False,
    )
    custom_a = quantizer.quantize(scaled_v)
    custom_b = quantizer.quantize(u)
    te_a = quantizer.quantize(scaled_v)
    te_b = quantizer.quantize(u)
    actual = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    expected = actual.clone()

    lowrank_nvfp4_add_sm120(custom_a, custom_b, actual)
    general_gemm(
        te_b,
        te_a,
        out_dtype=torch.bfloat16,
        layout="TN",
        out=expected,
        beta=1.0,
        accumulate=True,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1.25e-1)
