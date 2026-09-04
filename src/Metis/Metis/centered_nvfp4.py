"""Project-local centered NVFP4 activation-packing backend.

The extension writes directly into storage allocated by Transformer Engine's
NVFP4Quantizer. Keeping allocation and the resulting Python NVFP4Tensor in
TE avoids constructing or guessing private storage metadata in Metis.
"""

from __future__ import annotations

from typing import Any

import torch


_EXTENSION: Any | None = None
_LOAD_ERROR: Exception | None = None


def _extension() -> Any:
    """Import the compiled extension with a useful error if it is absent."""
    global _EXTENSION, _LOAD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _LOAD_ERROR is not None:
        raise RuntimeError(
            "The centered NVFP4 extension could not be loaded"
        ) from _LOAD_ERROR
    try:
        from . import _centered_nvfp4_cuda
    except Exception as error:  # pragma: no cover - CUDA build dependent
        _LOAD_ERROR = error
        raise RuntimeError(
            "Centered NVFP4 packing requires the project-local CUDA "
            "extension. Build csrc/centered_nvfp4 before selecting "
            "activation_pack_backend='centered_cuda'."
        ) from error
    _EXTENSION = _centered_nvfp4_cuda
    return _EXTENSION


def is_available() -> bool:
    try:
        _extension()
    except RuntimeError:
        return False
    return True


@torch.no_grad()
def quantize_centered_rowwise(
    value: torch.Tensor,
    mean: torch.Tensor,
    quantizer: Any,
):
    """Pack ``BF16Round(value - mean)`` into TE-owned NVFP4 storage.

    The compiled kernel receives raw storage tensors, but its returned object
    is the standard TE NVFP4Tensor, directly accepted by ``general_gemm``.
    """
    if value.dtype != torch.bfloat16 or mean.dtype != torch.bfloat16:
        raise ValueError("Centered NVFP4 packing requires BF16 value and mean")
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    if mean.shape != (1, flat.shape[1]):
        raise ValueError(
            f"mean shape must be (1, {flat.shape[1]}), got {tuple(mean.shape)}"
        )
    rows, columns = flat.shape
    if rows % 32 != 0 or columns % 32 != 0:
        raise ValueError(
            "Centered NVFP4 v1 requires M and K divisible by 32; "
            f"got ({rows}, {columns})"
        )
    packed = quantizer.make_empty(
        flat.shape,
        dtype=torch.bfloat16,
        device=flat.device,
    )
    metadata = packed.get_metadata()
    _extension().quantize_centered_rowwise(
        flat,
        mean,
        metadata["rowwise_data"],
        metadata["rowwise_scale_inv"],
        metadata["amax_rowwise"],
    )
    return packed


@torch.no_grad()
def fused_mean_and_quantize_centered_rowwise(
    value: torch.Tensor,
    quantizer: Any,
) -> tuple[torch.Tensor, Any]:
    """Compute BF16 column mean, centered global amax, and NVFP4 pack.

    The CUDA implementation makes one coalesced full read of ``value`` for
    column partial sum/min/max, then derives ``amax(abs(value - mean))`` from
    the extrema.  It therefore removes the separate full-tensor centered-amax
    pass and its atomic reduction.
    """
    if value.dtype != torch.bfloat16:
        raise ValueError("Centered NVFP4 packing requires BF16 value")
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    rows, columns = flat.shape
    if rows % 32 != 0 or columns % 32 != 0:
        raise ValueError(
            "Centered NVFP4 v1 requires M and K divisible by 32; "
            f"got ({rows}, {columns})"
        )
    mean = torch.empty((1, columns), device=flat.device, dtype=torch.bfloat16)
    packed = quantizer.make_empty(
        flat.shape,
        dtype=torch.bfloat16,
        device=flat.device,
    )
    metadata = packed.get_metadata()
    _extension().fused_mean_and_quantize_centered_rowwise(
        flat,
        mean,
        metadata["rowwise_data"],
        metadata["rowwise_scale_inv"],
        metadata["amax_rowwise"],
    )
    return mean, packed
