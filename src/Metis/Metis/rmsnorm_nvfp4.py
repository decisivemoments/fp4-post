"""RMSNorm fused with mean-centered rowwise NVFP4 activation packing."""
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
        raise RuntimeError("RMSNorm NVFP4 extension could not be loaded") from _LOAD_ERROR
    try:
        from . import _rmsnorm_nvfp4_cuda
    except Exception as error:  # pragma: no cover - CUDA build dependent
        _LOAD_ERROR = error
        raise RuntimeError(
            "RMSNorm NVFP4 packing requires csrc/rmsnorm_nvfp4 to be built"
        ) from error
    _EXTENSION = _rmsnorm_nvfp4_cuda
    return _EXTENSION


def is_available() -> bool:
    try:
        _extension()
    except RuntimeError:
        return False
    return True


@torch.no_grad()
def fused_rmsnorm_mean_and_quantize_centered_rowwise(
    value: torch.Tensor,
    rms_weight: torch.Tensor,
    eps: float,
    quantizer: Any,
) -> tuple[torch.Tensor, Any]:
    """Return Qwen2-RMSNorm's mean and TE-compatible packed centered output.

    The extension never materializes the normalized BF16 activation.  It
    preserves the existing centered rowwise NVFP4 storage contract, so the
    result can be used directly by Transformer Engine ``general_gemm``.
    """
    if value.dtype != torch.bfloat16:
        raise ValueError("RMSNorm NVFP4 packing requires BF16 activation")
    if rms_weight.dtype != torch.bfloat16:
        raise ValueError("RMSNorm NVFP4 packing requires BF16 RMS weight")
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    rows, columns = flat.shape
    if rows % 32 != 0 or columns % 32 != 0 or columns > 4096:
        raise ValueError(
            "RMSNorm NVFP4 v1 requires M/H divisible by 32 and H <= 4096; "
            f"got ({rows}, {columns})"
        )
    if rms_weight.device != flat.device or rms_weight.shape != (columns,):
        raise ValueError(
            f"rms_weight must be BF16 [{columns}] on activation device"
        )
    packed = quantizer.make_empty(
        flat.shape, dtype=torch.bfloat16, device=flat.device
    )
    mean = torch.empty((1, columns), device=flat.device, dtype=torch.bfloat16)
    metadata = packed.get_metadata()
    _extension().fused_rmsnorm_mean_and_quantize_centered_rowwise(
        flat,
        rms_weight.contiguous(),
        float(eps),
        mean,
        metadata["rowwise_data"],
        metadata["rowwise_scale_inv"],
        metadata["amax_rowwise"],
    )
    return mean, packed
