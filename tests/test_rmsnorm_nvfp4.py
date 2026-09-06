"""RMSNorm + centered NVFP4 packed-storage compatibility tests."""
from __future__ import annotations

import pytest
import torch

from Metis.Metis.centered_nvfp4 import (
    fused_mean_and_quantize_centered_rowwise,
)
from Metis.Metis.native_nvfp4 import _get_quantizer
from Metis.Metis.rmsnorm_nvfp4 import (
    fused_rmsnorm_mean_and_quantize_centered_rowwise,
    is_available,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="RMSNorm NVFP4 CUDA extension and CUDA are required",
)


def qwen2_rmsnorm_reference(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    """Exact operation ordering in the installed Qwen2RMSNorm.forward."""
    normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return weight * normalized.to(x.dtype)


@pytest.mark.parametrize("rows,columns", [(512, 896), (1024, 1536), (512, 3584)])
def test_rmsnorm_centered_nvfp4_storage_matches_reference(rows: int, columns: int):
    torch.manual_seed(rows + columns)
    x = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(columns, device="cuda", dtype=torch.bfloat16)
    eps = 1e-6
    quantizer = _get_quantizer(
        x.device, stochastic_rounding=False, rowwise=True, columnwise=False
    )
    reference_y = qwen2_rmsnorm_reference(x, weight, eps)
    reference_mean, reference = fused_mean_and_quantize_centered_rowwise(
        reference_y, quantizer
    )
    actual_mean, actual = fused_rmsnorm_mean_and_quantize_centered_rowwise(
        x, weight, eps, quantizer
    )
    torch.testing.assert_close(actual_mean, reference_mean, rtol=0, atol=0)
    expected_meta = reference.get_metadata()
    actual_meta = actual.get_metadata()
    for key in ("rowwise_data", "rowwise_scale_inv", "amax_rowwise"):
        torch.testing.assert_close(actual_meta[key], expected_meta[key], rtol=0, atol=0)
