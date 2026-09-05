"""Experimental single-store residual FP4 + BF16 low-rank forward path.

This is deliberately a standalone experiment: callers provide already-packed
NVFP4 activation/residual operands and the already-computed ``Z = (X @ V.T) *
s``.  It therefore measures only the post-quantization forward path and is
not yet enabled from :mod:`native_nvfp4`.
"""

from __future__ import annotations

from typing import Any

import torch
from transformer_engine.pytorch import cpp_extensions as tex


_EXTENSION: Any | None = None


def _extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from . import _fused_residual_lowrank_bf16_cuda

        _EXTENSION = _fused_residual_lowrank_bf16_cuda
    return _EXTENSION


def is_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@torch.no_grad()
def fused_residual_lowrank_bf16(
    packed_x: Any,
    packed_residual: Any,
    scaled_v_output: torch.Tensor,
    u_weight: torch.Tensor,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write ``FP4(X) @ FP4(R).T + scaled_v_output @ U.T`` once to ``output``.

    ``scaled_v_output`` and ``u_weight`` are BF16, have shapes ``[M,64]`` and
    ``[N,64]``, respectively.  Mean corrections and module bias are excluded
    from this first dataflow validation kernel and remain explicit next steps.
    """
    for packed in (packed_x, packed_residual):
        if not packed.get_metadata()["with_gemm_swizzled_scales"]:
            tex.swizzle_scales_for_gemm_(packed)
    x = packed_x.get_metadata()
    residual = packed_residual.get_metadata()
    _extension().fused_residual_lowrank_bf16(
        x["rowwise_data"], x["rowwise_scale_inv"],
        residual["rowwise_data"], residual["rowwise_scale_inv"],
        x["amax_rowwise"], residual["amax_rowwise"],
        scaled_v_output, u_weight, output,
    )
    return output
