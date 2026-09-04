"""Format and numerical tests for project-local centered NVFP4 packing."""

from __future__ import annotations

import pytest
import torch
from transformer_engine.pytorch.cpp_extensions import general_gemm

from Metis.Metis.centered_nvfp4 import is_available, quantize_centered_rowwise
from Metis.Metis.native_nvfp4 import (
    PackedMeanActivation,
    _get_quantizer,
    require_native_nvfp4,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_available(),
    reason="Centered NVFP4 CUDA extension and CUDA are required",
)


@pytest.mark.parametrize("rows,columns", [(512, 896), (1024, 4864)])
def test_centered_nvfp4_raw_storage_matches_te_reference(rows: int, columns: int):
    require_native_nvfp4()
    torch.manual_seed(rows + columns)
    x = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    mean = x.mean(dim=0, keepdim=True)
    quantizer = _get_quantizer(
        x.device,
        stochastic_rounding=False,
        rowwise=True,
        columnwise=False,
    )
    reference = quantizer.quantize((x - mean).contiguous())
    actual = quantize_centered_rowwise(x, mean, quantizer)
    reference_metadata = reference.get_metadata()
    actual_metadata = actual.get_metadata()

    torch.testing.assert_close(actual_metadata["rowwise_data"], reference_metadata["rowwise_data"], rtol=0, atol=0)
    torch.testing.assert_close(actual_metadata["rowwise_scale_inv"], reference_metadata["rowwise_scale_inv"], rtol=0, atol=0)
    torch.testing.assert_close(actual_metadata["amax_rowwise"], reference_metadata["amax_rowwise"], rtol=0, atol=0)


def test_centered_nvfp4_dequantization_and_native_gemm_match_te_reference():
    require_native_nvfp4()
    torch.manual_seed(20260903)
    rows, columns, output_features = 512, 896, 896
    x = torch.randn(rows, columns, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(output_features, columns, device="cuda", dtype=torch.bfloat16)
    mean = x.mean(dim=0, keepdim=True)
    activation_quantizer = _get_quantizer(
        x.device,
        stochastic_rounding=False,
        rowwise=True,
        columnwise=False,
    )
    weight_quantizer = _get_quantizer(
        weight.device,
        stochastic_rounding=False,
        rowwise=True,
        columnwise=True,
    )
    reference_x = activation_quantizer.quantize((x - mean).contiguous())
    actual_x = quantize_centered_rowwise(x, mean, activation_quantizer)
    packed_weight = weight_quantizer.quantize(weight)

    torch.testing.assert_close(actual_x.dequantize(dtype=torch.bfloat16), reference_x.dequantize(dtype=torch.bfloat16), rtol=0, atol=0)
    reference_output = general_gemm(packed_weight, reference_x, out_dtype=torch.bfloat16, layout="TN")[0]
    actual_output = general_gemm(packed_weight, actual_x, out_dtype=torch.bfloat16, layout="TN")[0]
    torch.testing.assert_close(actual_output, reference_output, rtol=0, atol=0)


def test_packed_mean_activation_uses_centered_cuda_backend():
    """The native activation wrapper returns a normal TE-consumable tensor."""
    require_native_nvfp4()
    torch.manual_seed(90210)
    x = torch.randn(512, 896, device="cuda", dtype=torch.bfloat16)
    expected_mean = x.mean(dim=0, keepdim=True)
    quantizer = _get_quantizer(
        x.device,
        stochastic_rounding=False,
        rowwise=True,
        columnwise=False,
    )
    expected = quantizer.quantize((x - expected_mean).contiguous())

    packed = PackedMeanActivation.from_tensor(
        x,
        columnwise=False,
        pack_backend="centered_cuda",
    )

    torch.testing.assert_close(packed.mean, expected_mean, rtol=0, atol=0)
    actual_metadata = packed.quantized_residual.get_metadata()
    expected_metadata = expected.get_metadata()
    for key in ("rowwise_data", "rowwise_scale_inv", "amax_rowwise"):
        torch.testing.assert_close(actual_metadata[key], expected_metadata[key], rtol=0, atol=0)
