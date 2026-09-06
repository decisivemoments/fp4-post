"""Accuracy-free direct NVFP4 linear used only as a speed upper-bound."""
from __future__ import annotations

import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions import general_gemm


def _quantizer(device: torch.device, *, columnwise: bool) -> NVFP4Quantizer:
    device_index = device.index if device.index is not None else torch.cuda.current_device()
    with torch.cuda.device(device_index):
        return NVFP4Quantizer(
            rowwise=True,
            columnwise=columnwise,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
        )


class DirectNVFP4Linear(nn.Module):
    """TE full-weight NVFP4 GEMM; intentionally not an accuracy-preserving layer."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        available, reason = te.is_nvfp4_available(return_reason=True)
        if not available:
            raise RuntimeError(f"Transformer Engine native NVFP4 is unavailable: {reason}")
        if weight.dtype != torch.bfloat16:
            raise ValueError("DirectNVFP4Linear requires BF16 source weights")
        if weight.shape[0] % 16 or weight.shape[1] % 16:
            raise ValueError("DirectNVFP4Linear requires 16-aligned weight shape")
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self._activation_quantizer = _quantizer(weight.device, columnwise=False)
        with torch.no_grad():
            self._packed_weight = _quantizer(weight.device, columnwise=True).quantize(
                weight.detach().contiguous()
            )

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "DirectNVFP4Linear":
        return cls(linear.weight)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            raise RuntimeError("DirectNVFP4Linear is inference-only")
        flat = input_.reshape(-1, input_.shape[-1]).contiguous()
        if flat.dtype != torch.bfloat16:
            flat = flat.to(torch.bfloat16)
        packed_input = self._activation_quantizer.quantize(flat)
        output = general_gemm(
            self._packed_weight,
            packed_input,
            out_dtype=torch.bfloat16,
            layout="TN",
        )[0]
        # Bias is deliberately ignored: this module is a speed-only full-FP4
        # reference, not a numerically valid replacement.
        return output.reshape(*input_.shape[:-1], self.out_features)


def replace_linear_with_direct_nvfp4(
    model: nn.Module,
    target_modules: set[str],
) -> list[str]:
    replaced: list[str] = []
    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] not in target_modules or not isinstance(module, nn.Linear):
            continue
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, DirectNVFP4Linear.from_linear(module))
        replaced.append(name)
    if not replaced:
        raise RuntimeError("No target linear layers were replaced")
    return replaced
