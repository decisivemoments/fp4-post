"""Experimental BF16 rank-64 residual epilogue kernel.

This module is intentionally not used by ``NativeFullNVFP4Linear`` yet.  It
provides a correctness- and performance-isolated path for
``residual += scaled_v @ u_transpose`` before attempting the dual-GEMM fusion.
"""

from __future__ import annotations

from typing import Any

import torch


_EXTENSION: Any | None = None
_LOAD_ERROR: Exception | None = None


def _extension() -> Any:
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise RuntimeError("Experimental low-rank BF16 extension could not be loaded") from _LOAD_ERROR
    try:
        from . import _lowrank_bf16_fused_cuda
    except Exception as error:  # pragma: no cover - build dependent
        _LOAD_ERROR = error
        raise RuntimeError(
            "Build csrc/lowrank_bf16_fused before using the experimental kernel."
        ) from error
    _EXTENSION = _lowrank_bf16_fused_cuda
    return _EXTENSION


@torch.no_grad()
def lowrank_add_bf16(
    scaled_v: torch.Tensor,
    u_transpose: torch.Tensor,
    residual_output: torch.Tensor,
) -> torch.Tensor:
    """In-place ``residual_output += scaled_v @ u_transpose`` for rank 64."""
    _extension().lowrank_add_bf16(scaled_v, u_transpose, residual_output)
    return residual_output


@torch.no_grad()
def lowrank_add_bf16_cutlass(
    scaled_v: torch.Tensor,
    u_transpose: torch.Tensor,
    residual_output: torch.Tensor,
) -> torch.Tensor:
    """Run the legacy CUTLASS BF16 candidate, when the target supports it.

    This remains a diagnostic baseline only.  CUTLASS's pre-Blackwell BF16
    device GEMM is not a valid SM120 implementation and currently rejects
    execution on GeForce Blackwell; keeping it explicit prevents accidental
    use in the native inference path.
    """
    _extension().lowrank_add_bf16_cutlass(
        scaled_v, u_transpose, residual_output
    )
    return residual_output
