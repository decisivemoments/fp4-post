"""SM120 CUTLASS NVFP4 rank-64 low-rank GEMM experiment.

This module consumes Transformer Engine NVFP4 storage directly.  It is not
wired into ``NativeFullNVFP4Linear`` until its accuracy and end-to-end timing
are verified against TE's current low-rank GEMM.
"""

from __future__ import annotations

from typing import Any

import torch
from transformer_engine.pytorch import cpp_extensions as tex


_EXTENSION: Any | None = None


def _extension() -> Any:
    global _EXTENSION
    if _EXTENSION is None:
        from . import _lowrank_nvfp4_sm120_cuda

        _EXTENSION = _lowrank_nvfp4_sm120_cuda
    return _EXTENSION


def is_available() -> bool:
    """Whether this process can execute an SM120a extension."""
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@torch.no_grad()
def lowrank_nvfp4_add_sm120(packed_a: Any, packed_b: Any, output: torch.Tensor) -> torch.Tensor:
    """In-place ``output += dequant(packed_a) @ dequant(packed_b).T``.

    ``packed_a`` has logical shape ``[M,64]`` and ``packed_b`` has logical
    shape ``[N,64]``.  Their block scales are independently swizzled through
    Transformer Engine's supported API; no singular values are folded into U.
    """
    for packed in (packed_a, packed_b):
        metadata = packed.get_metadata()
        if metadata["with_gemm_swizzled_scales"] is False:
            tex.swizzle_scales_for_gemm_(packed)
    a = packed_a.get_metadata()
    b = packed_b.get_metadata()
    # Compute TE's global-amax alpha on the CUDA stream.  No .item() host
    # synchronization is permitted here: scaled_v is newly quantized per
    # forward and its amax is only ready on device.
    _extension().lowrank_nvfp4_add_sm120(
        a["rowwise_data"], a["rowwise_scale_inv"],
        b["rowwise_data"], b["rowwise_scale_inv"],
        a["amax_rowwise"], b["amax_rowwise"], output,
    )
    return output
