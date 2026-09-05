"""Correctness tests for the experimental rank-64 BF16 Tensor-Core kernel."""

from __future__ import annotations

import pytest
import torch

from Metis.Metis.lowrank_bf16_fused import lowrank_add_bf16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("rows,columns", [(128, 128), (256, 896)])
def test_lowrank_add_bf16_matches_addmm(rows: int, columns: int):
    torch.manual_seed(rows + columns)
    scaled_v = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
    u_transpose = torch.randn(64, columns, device="cuda", dtype=torch.bfloat16)
    actual = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    expected = actual.clone()
    expected.addmm_(scaled_v, u_transpose)

    lowrank_add_bf16(scaled_v, u_transpose, actual)

    # WMMA uses FP32 accumulation, but the accumulator layout and BF16 output
    # rounding are not bitwise identical to cuBLASLt's BF16 addmm path.
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1.25e-1)
