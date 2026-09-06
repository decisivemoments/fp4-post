"""Native Transformer Engine NVFP4 backend for the full Metis linear path."""

from __future__ import annotations

import math
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.function import once_differentiable
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor.storage.nvfp4_tensor_storage import (
    NVFP4TensorStorage,
)

from .centered_nvfp4 import (
    fused_mean_and_quantize_centered_rowwise,
    quantize_centered_rowwise,
)
from .rmsnorm_nvfp4 import (
    fused_rmsnorm_mean_and_quantize_centered_rowwise,
)


NVFP4_BLOCK_SIZE = 16
# TE 2.15's SM120 native Wgrad path needs the reduction dimension aligned
# more strictly than the public NVFP4 quantizer contract. GRPO sequence
# padding can otherwise produce valid 16-aligned, but unsupported, row counts
# such as 1872.
NVFP4_WGRAD_ROW_ALIGNMENT = 32
DEFAULT_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}

_QUANTIZER_CACHE: dict[tuple[int, bool, bool, bool], NVFP4Quantizer] = {}
_NVTX_PROFILING_ENABLED = False


def set_native_nvfp4_profiling(enabled: bool) -> None:
    """Enable named NVTX ranges for a dedicated profiler run."""
    global _NVTX_PROFILING_ENABLED
    _NVTX_PROFILING_ENABLED = enabled


def _native_profile_range(name: str):
    if _NVTX_PROFILING_ENABLED:
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def require_native_nvfp4() -> None:
    """Raise with Transformer Engine's reason if native NVFP4 is unavailable."""
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        raise RuntimeError(f"Transformer Engine native NVFP4 is unavailable: {reason}")


def _get_quantizer(
    device: torch.device,
    *,
    stochastic_rounding: bool,
    rowwise: bool = True,
    columnwise: bool = True,
) -> NVFP4Quantizer:
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (device_index, stochastic_rounding, rowwise, columnwise)
    quantizer = _QUANTIZER_CACHE.get(key)
    if quantizer is None:
        with torch.cuda.device(device_index):
            quantizer = NVFP4Quantizer(
                rowwise=rowwise,
                columnwise=columnwise,
                with_amax_reduction=False,
                amax_reduction_group=None,
                with_rht=False,
                with_post_rht_amax=False,
                with_2d_quantization=False,
                stochastic_rounding=stochastic_rounding,
            )
        _QUANTIZER_CACHE[key] = quantizer
    return quantizer


def _validate_native_shape(rows: int, columns: int, *, label: str) -> None:
    if rows % NVFP4_BLOCK_SIZE != 0 or columns % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"{label} shape ({rows}, {columns}) must be divisible by "
            f"{NVFP4_BLOCK_SIZE} in both dimensions for native NVFP4"
        )


def _columnwise_transpose_view(quantized) -> NVFP4TensorStorage:
    """Expose TE's columnwise packing as a rowwise-packed logical transpose.

    Transformer Engine does not currently expose a public columnwise
    ``dequantize`` method. Its columnwise representation is exactly the
    rowwise representation of the logical transpose, so this metadata-only
    view lets the BF16 mean correction use the same values as the NN/NT GEMMs.
    """
    metadata = quantized.get_metadata()
    return NVFP4TensorStorage(
        rowwise_data=metadata["columnwise_data"],
        rowwise_scale_inv=metadata["columnwise_scale_inv"],
        columnwise_data=None,
        columnwise_scale_inv=None,
        amax_rowwise=metadata["amax_columnwise"],
        amax_columnwise=None,
        fp4_dtype=metadata["fp4_dtype"],
        quantizer=metadata["quantizer"],
        with_gemm_swizzled_scales=metadata["with_gemm_swizzled_scales"],
        fake_dtype=torch.bfloat16,
    )


def _dequantize_columnwise_transpose(quantized) -> torch.Tensor:
    """Dequantize TE's columnwise representation in transposed shape."""
    return _columnwise_transpose_view(quantized).dequantize(
        dtype=torch.bfloat16
    )


@dataclass
class PackedNVFP4Weight:
    """Packed weight plus a lazily materialized BF16 correction view."""

    quantized: object
    source_version: int
    _dequantized: Optional[torch.Tensor] = None
    _dequantized_transpose: Optional[torch.Tensor] = None

    def dequantized(self) -> torch.Tensor:
        if self._dequantized is None:
            self._dequantized = self.quantized.dequantize(dtype=torch.bfloat16)
        return self._dequantized

    def dequantized_transpose(self) -> torch.Tensor:
        if self._dequantized_transpose is None:
            self._dequantized_transpose = _dequantize_columnwise_transpose(
                self.quantized
            )
        return self._dequantized_transpose

    def release_dequantized(self) -> None:
        """Release transient BF16 correction views, retaining packed FP4."""
        self._dequantized = None
        self._dequantized_transpose = None


class PackedMeanActivation:
    """Mean plus packed NVFP4 residual shared by related projections."""

    def __init__(
        self,
        *,
        mean: torch.Tensor,
        quantized_residual,
        rows: int,
        packed_rows: int,
        columns: int,
    ) -> None:
        self.mean = mean
        self.quantized_residual = quantized_residual
        self.rows = rows
        self.packed_rows = packed_rows
        self.columns = columns
        self._columnwise_residual_sum: Optional[torch.Tensor] = None

    @classmethod
    @torch.no_grad()
    def from_tensor(
        cls,
        value: torch.Tensor,
        *,
        columnwise: bool = True,
        pack_backend: str = "te",
    ) -> "PackedMeanActivation":
        flat = value.reshape(-1, value.shape[-1]).contiguous()
        rows, columns = flat.shape
        packed_rows = math.ceil(rows / NVFP4_WGRAD_ROW_ALIGNMENT)
        packed_rows *= NVFP4_WGRAD_ROW_ALIGNMENT
        _validate_native_shape(packed_rows, columns, label="activation")
        with _native_profile_range("native_nvfp4.activation_mean_and_pack"):
            quantizer = _get_quantizer(
                value.device,
                stochastic_rounding=False,
                rowwise=True,
                columnwise=columnwise,
            )
            if pack_backend == "te":
                with _native_profile_range("native_nvfp4.activation_mean"):
                    mean = flat.mean(dim=0, keepdim=True)
                with _native_profile_range("native_nvfp4.activation_center_and_pack"):
                    residual = (flat - mean).contiguous()
                    if packed_rows != rows:
                        residual = F.pad(
                            residual,
                            (0, 0, 0, packed_rows - rows),
                        )
                    quantized = quantizer.quantize(residual)
            elif pack_backend == "centered_cuda":
                if columnwise:
                    raise ValueError(
                        "centered_cuda is rowwise-only and inference-only"
                    )
                if packed_rows != rows:
                    raise ValueError(
                        "centered_cuda v1 does not support activation padding"
                    )
                with _native_profile_range(
                    "native_nvfp4.activation_fused_mean_and_amax"
                ):
                    with _native_profile_range(
                        "native_nvfp4.activation_centered_cuda_pack"
                    ):
                        mean, quantized = fused_mean_and_quantize_centered_rowwise(
                            flat,
                            quantizer,
                        )
            else:
                raise ValueError(f"Unknown activation pack backend: {pack_backend}")
        return cls(
            mean=mean,
            quantized_residual=quantized,
            rows=rows,
            packed_rows=packed_rows,
            columns=columns,
        )

    @classmethod
    @torch.no_grad()
    def from_rmsnorm_tensor(
        cls,
        value: torch.Tensor,
        rms_weight: torch.Tensor,
        eps: float,
    ) -> "PackedMeanActivation":
        """Pack Qwen RMSNorm(value) without materializing its BF16 output."""
        flat = value.reshape(-1, value.shape[-1]).contiguous()
        rows, columns = flat.shape
        packed_rows = math.ceil(rows / NVFP4_WGRAD_ROW_ALIGNMENT)
        packed_rows *= NVFP4_WGRAD_ROW_ALIGNMENT
        _validate_native_shape(packed_rows, columns, label="RMSNorm activation")
        if packed_rows != rows:
            raise ValueError(
                "fused RMSNorm NVFP4 v1 does not support activation padding"
            )
        with _native_profile_range("native_nvfp4.rmsnorm_mean_and_pack"):
            quantizer = _get_quantizer(
                value.device,
                stochastic_rounding=False,
                rowwise=True,
                columnwise=False,
            )
            mean, quantized = fused_rmsnorm_mean_and_quantize_centered_rowwise(
                flat, rms_weight, eps, quantizer
            )
        return cls(
            mean=mean,
            quantized_residual=quantized,
            rows=rows,
            packed_rows=packed_rows,
            columns=columns,
        )

    @torch.no_grad()
    def columnwise_residual_sum(self) -> torch.Tensor:
        metadata = self.quantized_residual.get_metadata()
        if metadata["columnwise_data"] is None:
            raise RuntimeError(
                "Columnwise activation packing is required for backward; "
                "rowwise-only activation packing is inference-only"
            )
        if self._columnwise_residual_sum is None:
            residual_transpose = _dequantize_columnwise_transpose(
                self.quantized_residual
            )
            self._columnwise_residual_sum = residual_transpose.sum(
                dim=1,
            ).unsqueeze(0)
        return self._columnwise_residual_sum


class NativeActivationGroup:
    """Weak-reference cache for Q/K/V and Gate/Up activation packing."""

    def __init__(
        self,
        name: str,
        *,
        columnwise: bool = True,
        pack_backend: str = "te",
    ) -> None:
        self.name = name
        self.columnwise = columnwise
        self.pack_backend = pack_backend
        self._source_ref: Optional[weakref.ReferenceType] = None
        self._packed: Optional[PackedMeanActivation] = None

    def clear(self) -> None:
        self._source_ref = None
        self._packed = None

    def get(self, value: torch.Tensor) -> PackedMeanActivation:
        if self._source_ref is not None and self._source_ref() is value:
            if self._packed is None:
                raise RuntimeError("Activation cache reference exists without packed data")
            return self._packed

        packed = PackedMeanActivation.from_tensor(
            value,
            columnwise=self.columnwise,
            pack_backend=self.pack_backend,
        )

        def clear_cached_input(ref) -> None:
            if self._source_ref is ref:
                self.clear()

        self._source_ref = weakref.ref(value, clear_cached_input)
        self._packed = packed
        return packed


def _native_mean_fprop(
    activation: PackedMeanActivation,
    weight: PackedNVFP4Weight,
) -> torch.Tensor:
    """Compute (mean + FP4 residual) @ FP4(weight).T."""
    with _native_profile_range("native_nvfp4.fprop_mean_correction_bf16"):
        dequantized_weight = weight.dequantized()
        mean_correction = F.linear(activation.mean, dequantized_weight)
        weight.release_dequantized()
        del dequantized_weight
    # The correction is a per-output-channel value.  Passing it as GEMM bias
    # makes TE apply it in the FP4 GEMM epilogue and removes a separate
    # MxN BF16 broadcast-add kernel.
    with _native_profile_range("native_nvfp4.fprop_native_gemm"):
        output = general_gemm(
            weight.quantized,
            activation.quantized_residual,
            out_dtype=torch.bfloat16,
            layout="TN",
            bias=mean_correction.squeeze(0),
        )[0]
    return output[: activation.rows]


def _native_mean_backward(
    grad_output: torch.Tensor,
    activation: PackedMeanActivation,
    weight: PackedNVFP4Weight,
    *,
    stochastic_rounding: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Native residual dgrad/Wgrad plus exact BF16 mean correction terms."""
    grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
    rows, output_features = grad_flat.shape
    if rows != activation.rows:
        raise ValueError(
            f"Gradient rows {rows} do not match activation rows "
            f"{activation.rows}"
        )
    _validate_native_shape(
        activation.packed_rows,
        output_features,
        label="gradient",
    )

    with _native_profile_range("native_nvfp4.gradient_mean_and_pack"):
        grad_mean = grad_flat.mean(dim=0, keepdim=True)
        grad_residual = (grad_flat - grad_mean).contiguous()
        if activation.packed_rows != rows:
            grad_residual = F.pad(
                grad_residual,
                (0, 0, 0, activation.packed_rows - rows),
            )
        grad_quantizer = _get_quantizer(
            grad_output.device,
            stochastic_rounding=stochastic_rounding,
        )
        quantized_grad = grad_quantizer.quantize(grad_residual)

    with _native_profile_range("native_nvfp4.dgrad_native_gemm"):
        grad_input_residual = general_gemm(
            weight.quantized,
            quantized_grad,
            out_dtype=torch.bfloat16,
            layout="NN",
            grad=True,
        )[0]
    with _native_profile_range("native_nvfp4.dgrad_mean_correction_bf16"):
        dequantized_weight_transpose = weight.dequantized_transpose()
        grad_input_mean = F.linear(grad_mean, dequantized_weight_transpose)
        weight.release_dequantized()
        del dequantized_weight_transpose
    grad_input = grad_input_residual[:rows] + grad_input_mean

    with _native_profile_range("native_nvfp4.wgrad_native_gemm"):
        try:
            grad_weight_residual = general_gemm(
                activation.quantized_residual,
                quantized_grad,
                out_dtype=torch.bfloat16,
                layout="NT",
                grad=True,
            )[0]
        except RuntimeError as error:
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                grad_output.device
            )
            raise RuntimeError(
                "Native NVFP4 Wgrad GEMM failed for "
                f"activation=({activation.rows}, {activation.columns}), "
                f"packed_rows={activation.packed_rows}, "
                f"gradient=({rows}, {output_features}), "
                f"free_cuda_gib={free_bytes / 1024**3:.3f}, "
                f"total_cuda_gib={total_bytes / 1024**3:.3f}"
            ) from error

    with _native_profile_range("native_nvfp4.wgrad_mean_correction_bf16"):
        quantized_grad_transpose = _dequantize_columnwise_transpose(
            quantized_grad
        )
        grad_residual_sum = quantized_grad_transpose.sum(
            dim=1
        ).unsqueeze(0)
        activation_residual_sum = activation.columnwise_residual_sum()

        correction = grad_residual_sum.T @ activation.mean
        correction = correction + grad_mean.T @ activation_residual_sum
        correction = correction + rows * (grad_mean.T @ activation.mean)
        grad_weight = grad_weight_residual + correction
    return grad_input, grad_weight


class _NativeFullNVFP4Function(torch.autograd.Function):
    """Autograd for W-residual + U diag(s) V with native mean NVFP4 GEMMs."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        residual_weight: torch.Tensor,
        v_weight: torch.Tensor,
        u_weight: torch.Tensor,
        singular_values: torch.Tensor,
        bias: Optional[torch.Tensor],
        packed_activation: PackedMeanActivation,
        packed_residual_weight: PackedNVFP4Weight,
        packed_v_weight: PackedNVFP4Weight,
        packed_u_weight: Optional[PackedNVFP4Weight],
        stochastic_rounding: bool,
        lowrank_compute: str,
        enable_dual_fp4_fusion: bool,
    ) -> torch.Tensor:
        input_shape = tuple(input_.shape)
        # The dual-FP4 fusion consumes the centered activation residual and
        # therefore cannot use _native_mean_fprop's GEMM bias directly. Keep
        # its mathematically required per-output mean correction explicit.
        # The correction vector is broadcast in the fused epilogue.
        # Unsupported shapes retain the existing TE beta=1 path below.
        use_dual_fp4_fusion = (
            lowrank_compute == "nvfp4"
            and enable_dual_fp4_fusion
            # The custom collective is an SM120 rank-64 kernel with fixed
            # 128x128 output tiles and a 64-wide K granularity.
            and torch.cuda.get_device_capability(input_.device) == (12, 0)
            and packed_activation.packed_rows % 128 == 0
            and residual_weight.shape[0] % 128 == 0
            and residual_weight.shape[1] % 64 == 0
            and u_weight.shape[1] == 64
        )
        if use_dual_fp4_fusion:
            with _native_profile_range(
                "native_nvfp4.fused_residual_mean_correction"
            ):
                residual_dequantized = packed_residual_weight.dequantized()
                residual_mean_correction = F.linear(
                    packed_activation.mean, residual_dequantized
                )
                if bias is not None:
                    # Fold module bias into the small vector as well, so a
                    # biased layer does not reintroduce an MxN output add.
                    residual_mean_correction = residual_mean_correction + bias
                packed_residual_weight.release_dequantized()
                del residual_dequantized
        else:
            residual_output = _native_mean_fprop(
                packed_activation,
                packed_residual_weight,
            )
        v_output = _native_mean_fprop(
            packed_activation,
            packed_v_weight,
        )
        with _native_profile_range("native_nvfp4.lowrank_scale_bf16"):
            scaled_v_output = v_output * singular_values
        if lowrank_compute == "bf16":
            with _native_profile_range("native_nvfp4.lowrank_fprop_bf16"):
                # Accumulate in-place: out-of-place torch.addmm copies its
                # input C matrix before invoking GEMM. residual_output is a
                # forward-only temporary and is not used by this Function's
                # backward implementation.
                with _native_profile_range(
                    "native_nvfp4.lowrank_addmm_inplace"
                ):
                    residual_output.addmm_(scaled_v_output, u_weight.T)
                output = residual_output
        elif lowrank_compute == "nvfp4":
            if packed_u_weight is None:
                raise RuntimeError("NVFP4 low-rank path requires packed U")
            with _native_profile_range("native_nvfp4.lowrank_scaled_v_pack"):
                scaled_v_quantizer = _get_quantizer(
                    scaled_v_output.device,
                    stochastic_rounding=False,
                    rowwise=True,
                    columnwise=False,
                )
                # _native_mean_fprop returns only logical rows.  In the
                # fused path, Z must instead have the same padded M extent as
                # X because both operands are consumed by the same 128-row
                # collective tiles.  Padding is zero and is cropped away
                # before this Function returns.
                scaled_v_for_nvfp4 = scaled_v_output
                if use_dual_fp4_fusion and (
                    packed_activation.packed_rows != packed_activation.rows
                ):
                    scaled_v_for_nvfp4 = F.pad(
                        scaled_v_output,
                        (
                            0,
                            0,
                            0,
                            packed_activation.packed_rows
                            - packed_activation.rows,
                        ),
                    )
                packed_scaled_v = scaled_v_quantizer.quantize(
                    scaled_v_for_nvfp4
                )
            with _native_profile_range("native_nvfp4.lowrank_nvfp4_gemm"):
                if use_dual_fp4_fusion:
                    from .fused_residual_lowrank_nvfp4 import (
                        fused_residual_lowrank_nvfp4,
                    )

                    # The packed activation can be row-padded for native
                    # NVFP4 alignment.  The fused kernel must cover those
                    # rows too, then we crop to the logical activation rows.
                    output = torch.empty(
                        packed_activation.packed_rows,
                        residual_weight.shape[0],
                        device=input_.device,
                        dtype=torch.bfloat16,
                    )
                    fused_residual_lowrank_nvfp4(
                        packed_activation.quantized_residual,
                        packed_residual_weight.quantized,
                        packed_scaled_v,
                        packed_u_weight.quantized,
                        residual_mean_correction.squeeze(0).contiguous(),
                        output,
                    )
                    output = output[: packed_activation.rows]
                else:
                    # Write directly into residual_output with beta=1,
                    # avoiding a separate MxN BF16 output add. U and
                    # scaled_v are separately quantized; singular values are
                    # intentionally not folded into U.
                    output = general_gemm(
                        packed_u_weight.quantized,
                        packed_scaled_v,
                        out_dtype=torch.bfloat16,
                        layout="TN",
                        out=residual_output,
                        beta=1.0,
                        accumulate=True,
                    )[0]
                    if output is None:
                        output = residual_output
        else:
            raise ValueError(f"Unsupported low-rank compute mode: {lowrank_compute}")
        if bias is not None and not use_dual_fp4_fusion:
            with _native_profile_range("native_nvfp4.fprop_bias_add"):
                output = output + bias

        ctx.input_shape = input_shape
        ctx.has_bias = bias is not None
        ctx.packed_activation = packed_activation
        ctx.packed_residual_weight = packed_residual_weight
        ctx.packed_v_weight = packed_v_weight
        ctx.stochastic_rounding = stochastic_rounding
        ctx.save_for_backward(
            u_weight,
            singular_values,
            v_output,
            scaled_v_output,
        )
        return output.reshape(*input_shape[:-1], residual_weight.shape[0])

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: torch.Tensor):
        u_weight, singular_values, v_output, scaled_v_output = ctx.saved_tensors
        grad_flat = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()

        with _native_profile_range("native_nvfp4.lowrank_backward_bf16"):
            grad_u_weight = grad_flat.T @ scaled_v_output
            grad_scaled_v = grad_flat @ u_weight
            grad_singular_values = (grad_scaled_v * v_output).sum(dim=0)
            grad_v_output = grad_scaled_v * singular_values

        grad_input_residual, grad_residual_weight = _native_mean_backward(
            grad_flat,
            ctx.packed_activation,
            ctx.packed_residual_weight,
            stochastic_rounding=ctx.stochastic_rounding,
        )
        grad_input_v, grad_v_weight = _native_mean_backward(
            grad_v_output,
            ctx.packed_activation,
            ctx.packed_v_weight,
            stochastic_rounding=ctx.stochastic_rounding,
        )
        grad_input = grad_input_residual + grad_input_v
        grad_bias = grad_flat.sum(dim=0) if ctx.has_bias else None

        return (
            grad_input.reshape(ctx.input_shape),
            grad_residual_weight,
            grad_v_weight,
            grad_u_weight,
            grad_singular_values,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class NativeFullNVFP4Linear(nn.Module):
    """Full Metis linear with rank-r W-SVD and native NVFP4 mean residuals."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int = 64,
        bias: bool = False,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.bfloat16,
        stochastic_rounding: bool = True,
        activation_columnwise: bool = True,
        activation_pack_backend: str = "te",
        lowrank_compute: str = "bf16",
        enable_dual_fp4_fusion: bool = True,
    ) -> None:
        super().__init__()
        require_native_nvfp4()
        if dtype != torch.bfloat16:
            raise ValueError("Transformer Engine NVFP4 training requires BF16 parameters")
        if rank <= 0 or rank > min(in_features, out_features):
            raise ValueError(
                f"rank must be in [1, {min(in_features, out_features)}], got {rank}"
            )
        if lowrank_compute not in {"bf16", "nvfp4"}:
            raise ValueError(
                "lowrank_compute must be either 'bf16' or 'nvfp4'"
            )
        if activation_pack_backend not in {"te", "centered_cuda"}:
            raise ValueError(
                "activation_pack_backend must be 'te' or 'centered_cuda'"
            )
        if activation_pack_backend == "centered_cuda" and activation_columnwise:
            raise ValueError(
                "centered_cuda activation packing is rowwise-only"
            )
        _validate_native_shape(out_features, in_features, label="weight")
        _validate_native_shape(rank, in_features, label="V weight")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.stochastic_rounding = stochastic_rounding
        self.lowrank_compute = lowrank_compute
        self.enable_dual_fp4_fusion = enable_dual_fp4_fusion
        self.residual_weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.v_weight = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.u_weight = nn.Parameter(
            torch.empty(out_features, rank, device=device, dtype=dtype)
        )
        self.singular_values = nn.Parameter(
            torch.empty(rank, device=device, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
            if bias
            else None
        )
        self.activation_group = NativeActivationGroup(
            "unshared",
            columnwise=activation_columnwise,
            pack_backend=activation_pack_backend,
        )
        self.layer_name = ""
        self._weight_cache: dict[str, tuple[tuple, PackedNVFP4Weight]] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.residual_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.u_weight, a=math.sqrt(5))
        nn.init.ones_(self.singular_values)
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int = 64,
        stochastic_rounding: bool = True,
        activation_columnwise: bool = True,
        activation_pack_backend: str = "te",
        lowrank_compute: str = "bf16",
        enable_dual_fp4_fusion: bool = True,
    ) -> "NativeFullNVFP4Linear":
        module = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
            stochastic_rounding=stochastic_rounding,
            activation_columnwise=activation_columnwise,
            activation_pack_backend=activation_pack_backend,
            lowrank_compute=lowrank_compute,
            enable_dual_fp4_fusion=enable_dual_fp4_fusion,
        )
        weight_fp32 = linear.weight.detach().to(torch.float32)
        u, singular_values, vh = torch.linalg.svd(
            weight_fp32,
            full_matrices=False,
        )
        u_rank = u[:, :rank]
        s_rank = singular_values[:rank]
        v_rank = vh[:rank, :]
        residual = weight_fp32 - (u_rank * s_rank.unsqueeze(0)) @ v_rank

        module.residual_weight.copy_(residual.to(linear.weight.dtype))
        module.u_weight.copy_(u_rank.to(linear.weight.dtype))
        module.singular_values.copy_(s_rank.to(linear.weight.dtype))
        module.v_weight.copy_(v_rank.to(linear.weight.dtype))
        if linear.bias is not None:
            module.bias.copy_(linear.bias)
        module._weight_cache.clear()
        return module

    def _get_packed_weight(
        self,
        name: str,
        weight: torch.Tensor,
    ) -> PackedNVFP4Weight:
        key = (
            weight._version,
            weight.device,
            weight.dtype,
            weight.data_ptr(),
        )
        cached = self._weight_cache.get(name)
        if cached is not None and cached[0] == key:
            return cached[1]

        quantizer = _get_quantizer(weight.device, stochastic_rounding=False)
        with torch.no_grad():
            with _native_profile_range("native_nvfp4.weight_pack"):
                quantized = quantizer.quantize(weight.detach())
        packed = PackedNVFP4Weight(
            quantized=quantized,
            source_version=weight._version,
        )
        self._weight_cache[name] = (key, packed)
        return packed

    def clear_native_cache(self) -> None:
        self._weight_cache.clear()
        self.activation_group.clear()

    def _apply(self, fn, recurse: bool = True):
        self.clear_native_cache()
        return super()._apply(fn, recurse=recurse)

    def _forward_with_packed_activation(
        self,
        input_: torch.Tensor,
        packed_activation: PackedMeanActivation,
    ) -> torch.Tensor:
        """Run the native projection using a caller-owned packed activation."""
        if packed_activation.columns != self.in_features:
            raise ValueError(
                "packed activation columns do not match linear in_features"
            )
        if packed_activation.rows != input_.reshape(-1, input_.shape[-1]).shape[0]:
            raise ValueError("packed activation rows do not match input")
        if self.lowrank_compute == "nvfp4" and torch.is_grad_enabled():
            raise RuntimeError(
                "lowrank_compute='nvfp4' is inference-only; use "
                "torch.inference_mode() or select lowrank_compute='bf16'"
            )
        packed_residual = self._get_packed_weight(
            "residual",
            self.residual_weight,
        )
        packed_v = self._get_packed_weight("v", self.v_weight)
        packed_u = (
            self._get_packed_weight("u", self.u_weight)
            if self.lowrank_compute == "nvfp4"
            else None
        )
        return _NativeFullNVFP4Function.apply(
            input_,
            self.residual_weight,
            self.v_weight,
            self.u_weight,
            self.singular_values,
            self.bias,
            packed_activation,
            packed_residual,
            packed_v,
            packed_u,
            self.stochastic_rounding,
            self.lowrank_compute,
            self.enable_dual_fp4_fusion,
        )

    def forward_from_packed_activation(
        self,
        input_: torch.Tensor,
        packed_activation: PackedMeanActivation,
    ) -> torch.Tensor:
        """Inference-only group entry point that avoids activation repacking."""
        if torch.is_grad_enabled():
            raise RuntimeError(
                "forward_from_packed_activation is inference-only; use "
                "torch.inference_mode()"
            )
        if input_.dtype != torch.bfloat16:
            raise ValueError("packed-activation input shape carrier must be BF16")
        return self._forward_with_packed_activation(input_, packed_activation)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if input_.dtype != torch.bfloat16:
            input_ = input_.to(torch.bfloat16)
        packed_activation = self.activation_group.get(input_)
        return self._forward_with_packed_activation(input_, packed_activation)

    @torch.no_grad()
    def reconstructed_weight(self) -> torch.Tensor:
        return self.residual_weight + (
            self.u_weight * self.singular_values.unsqueeze(0)
        ) @ self.v_weight

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, native_nvfp4=True, "
            f"lowrank_compute={self.lowrank_compute}, "
            f"dual_fp4_fusion={self.enable_dual_fp4_fusion}"
        )


def replace_linear_with_native_full_nvfp4(
    model: nn.Module,
    *,
    rank: int = 64,
    target_modules: Optional[Iterable[str]] = None,
    stochastic_rounding: bool = True,
    activation_columnwise: bool = True,
    activation_pack_backend: str = "te",
    lowrank_compute: str = "bf16",
    enable_dual_fp4_fusion: bool = True,
) -> list[str]:
    """Replace selected Qwen projections and return their fully qualified names."""
    targets = set(target_modules or DEFAULT_TARGET_MODULES)
    activation_groups: dict[str, NativeActivationGroup] = {}
    replaced: list[str] = []

    for name, module in list(model.named_modules()):
        projection_name = name.rsplit(".", 1)[-1]
        if projection_name not in targets or not isinstance(module, nn.Linear):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        native_layer = NativeFullNVFP4Linear.from_linear(
            module,
            rank=rank,
            stochastic_rounding=stochastic_rounding,
            activation_columnwise=activation_columnwise,
            activation_pack_backend=activation_pack_backend,
            lowrank_compute=lowrank_compute,
            enable_dual_fp4_fusion=enable_dual_fp4_fusion,
        )
        native_layer.layer_name = name

        if projection_name in {"q_proj", "k_proj", "v_proj"}:
            group_name = f"{parent_name}.qkv"
        elif projection_name in {"gate_proj", "up_proj"}:
            group_name = f"{parent_name}.gate_up"
        else:
            group_name = f"{name}.input"
        native_layer.activation_group = activation_groups.setdefault(
            group_name,
            NativeActivationGroup(
                group_name,
                columnwise=activation_columnwise,
                pack_backend=activation_pack_backend,
            ),
        )

        setattr(parent, child_name, native_layer)
        replaced.append(name)

    if not replaced:
        raise RuntimeError("No target linear layers were replaced")
    return replaced


def clear_native_nvfp4_caches(model: nn.Module) -> None:
    """Clear packed weights and activations after an optimizer update."""
    for module in model.modules():
        if isinstance(module, NativeFullNVFP4Linear):
            module.clear_native_cache()


def register_native_nvfp4_optimizer_hook(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
):
    """Invalidate native caches after every optimizer step.

    Fused optimizers may update parameter storage without incrementing
    ``Parameter._version``, so version-keyed cache checks are not sufficient.
    """

    def clear_after_step(_optimizer, _args, _kwargs) -> None:
        clear_native_nvfp4_caches(model)

    return optimizer.register_step_post_hook(clear_after_step)
