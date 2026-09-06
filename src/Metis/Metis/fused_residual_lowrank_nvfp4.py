"""SM120 dual-FP4 kernel consuming four independently quantized operands.

Z = (X @ V.T) * singular_values is materialized and quantized by the caller.
This post-quantization operator writes Y once and performs no global C update.
"""
from __future__ import annotations

from typing import Any

import torch
from transformer_engine.pytorch import cpp_extensions as tex

_EXTENSION: Any | None = None


def _extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from . import _fused_residual_lowrank_nvfp4_cuda
        _EXTENSION = _fused_residual_lowrank_nvfp4_cuda
    return _EXTENSION


def is_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@torch.no_grad()
def fused_residual_lowrank_nvfp4(
    packed_x: Any, packed_residual: Any, packed_z: Any, packed_u: Any,
    residual_mean_correction: torch.Tensor, output: torch.Tensor,
) -> torch.Tensor:
    """Y = dequant(Xq) @ dequant(Rq).T + dequant(Zq) @ dequant(Uq).T + residual_mean_correction.

    Requires M,N divisible by 128, K divisible by 64, rank=64, and a
    contiguous BF16 output. Each operand retains its own scales and amax.
    First-use TE scale swizzling mutates the packed objects; later calls
    reuse the swizzled storage. The explicit BF16 [N] mean correction is added in FP32 before final conversion.
    """
    metadata = []
    for packed in (packed_x, packed_residual, packed_z, packed_u):
        if not packed.get_metadata()["with_gemm_swizzled_scales"]:
            tex.swizzle_scales_for_gemm_(packed)
        metadata.append(packed.get_metadata())
    x, r, z, u = metadata
    _extension().fused_residual_lowrank_nvfp4(
        x["rowwise_data"], x["rowwise_scale_inv"],
        r["rowwise_data"], r["rowwise_scale_inv"],
        x["amax_rowwise"], r["amax_rowwise"],
        z["rowwise_data"], z["rowwise_scale_inv"],
        u["rowwise_data"], u["rowwise_scale_inv"],
        z["amax_rowwise"], u["amax_rowwise"], residual_mean_correction, output,
    )
    return output
