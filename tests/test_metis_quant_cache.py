from types import SimpleNamespace

import pytest
import torch

from Metis.Metis.bitlinear import LinearLowbit, LinearLowbitFunction
from Metis.Metis.quant import (
    Cast2Fp32,
    Cast2NVFp4e2m1BlockNOSR,
    nvfp4_nosr_qdq_compile_status,
)


def test_quantize_dequantize_matches_existing_sequence():
    x = torch.randn(4, 32)
    scalar = Cast2NVFp4e2m1BlockNOSR.get_scalar(x)
    expected = Cast2NVFp4e2m1BlockNOSR.rquant(
        Cast2NVFp4e2m1BlockNOSR.quant(x, scalar), scalar
    )

    actual = Cast2NVFp4e2m1BlockNOSR.quantize_dequantize(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torch.compile NVFP4 test requires CUDA")
def test_compiled_qdq_matches_eager_fused_qdq():
    x = torch.randn(8, 16, 128, device="cuda", dtype=torch.bfloat16)
    Cast2NVFp4e2m1BlockNOSR.compile_quantize_dequantize = False
    expected = Cast2NVFp4e2m1BlockNOSR.quantize_dequantize(x)

    Cast2NVFp4e2m1BlockNOSR.compile_quantize_dequantize = True
    try:
        actual = Cast2NVFp4e2m1BlockNOSR.quantize_dequantize(x)
    finally:
        Cast2NVFp4e2m1BlockNOSR.compile_quantize_dequantize = False

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert nvfp4_nosr_qdq_compile_status() == "validated"


def test_weight_cache_reuses_then_invalidates_after_parameter_update():
    class CountingQuant(Cast2Fp32):
        calls = 0

        @classmethod
        def quantize_dequantize(cls, value):
            cls.calls += 1
            return value.detach().clone()

    args = SimpleNamespace(device="cpu", cache_quantized_weight=True)
    layer = LinearLowbit(16, 8, bias=False, args=args)
    LinearLowbitFunction.compute_dtype = torch.float32
    LinearLowbitFunction.q_forward_input = Cast2Fp32
    LinearLowbitFunction.q_forward_weight = CountingQuant
    LinearLowbitFunction.enable_activation_svd = False
    LinearLowbitFunction.enable_nv_recipe = False

    x = torch.randn(2, 3, 16, requires_grad=True)
    layer(x).sum().backward()
    assert layer.weight.grad is not None

    layer(x.detach())
    assert CountingQuant.calls == 1

    with torch.no_grad():
        layer.weight.add_(0.1)
    layer(x.detach())
    assert CountingQuant.calls == 2


def test_mean_activation_path_uses_quantize_dequantize():
    class CountingQuant(Cast2Fp32):
        calls = 0

        @classmethod
        def quantize_dequantize(cls, value):
            cls.calls += 1
            return value

        @classmethod
        def get_scalar(cls, value):
            raise AssertionError("mean activation path must use fused QDQ")

    x = torch.randn(2, 3, 16)
    output = LinearLowbitFunction.svd_quant(
        x,
        quant_func=CountingQuant,
        metis_mode="mean",
        mean_cache={},
        cache_key="test",
    )

    assert output.shape == x.shape
    assert CountingQuant.calls == 1
