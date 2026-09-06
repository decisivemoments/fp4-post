"""Dual-FP4 correctness, phase reuse, independent scaling and stream tests."""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="SM120 GPU required")


def _case(m, n, k, gains=(1.0, 1.0, 1.0, 1.0), stream=None, bias_gain=1.0):
    from transformer_engine.pytorch.cpp_extensions import general_gemm
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    from Metis.Metis.fused_residual_lowrank_nvfp4 import fused_residual_lowrank_nvfp4

    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU required")
    torch.manual_seed(20260906)
    with torch.cuda.stream(stream or torch.cuda.current_stream()):
        tensors = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16) * gain
                   for shape, gain in zip(((m, k), (n, k), (m, 64), (n, 64)), gains)]
        # Separate quantizers for every operand and separate TE/custom objects:
        # swizzling one pair cannot accidentally supply the other pair's scales.
        packed, reference_packed = [], []
        for t in tensors:
            q = NVFP4Quantizer(rowwise=True, columnwise=False, with_amax_reduction=False,
                               with_rht=False, with_post_rht_amax=False,
                               with_2d_quantization=False, stochastic_rounding=False)
            packed.append(q.quantize(t))
            reference_packed.append(q.quantize(t))
        x, r, z, u = reference_packed
        expected = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        general_gemm(r, x, out_dtype=torch.bfloat16, layout="TN", out=expected)
        general_gemm(u, z, out_dtype=torch.bfloat16, layout="TN", out=expected,
                     beta=1.0, accumulate=True)
        bias = torch.linspace(-1, 1, n, device="cuda", dtype=torch.bfloat16) * bias_gain
        expected.add_(bias)
        actual = torch.full_like(expected, float("nan"))
        for _ in range(3):
            result = fused_residual_lowrank_nvfp4(*packed, bias, actual)
            assert result.data_ptr() == actual.data_ptr()
        # TE rounds the residual to BF16 before addition; fused rounds only Y.
        for start in range(0, m, 2048):
            a, e = actual[start:start+2048], expected[start:start+2048]
            torch.testing.assert_close(a, e, rtol=2e-2, atol=1.25e-1)
            assert torch.isfinite(a).all()
        if gains[0] == 0 and gains[2] == 0:
            torch.testing.assert_close(actual, bias.expand_as(actual), rtol=0, atol=0)


@pytest.mark.parametrize("m,n,k", [(128,128,128), (256,256,128),
                                   (256,384,896), (8192,256,896), (128,128,64)])
def test_dual_fp4_tiles(m, n, k):
    _case(m, n, k)


@pytest.mark.parametrize("gains", [(0,1,1,1), (1,0,1,1), (1,1,0,1),
                                   (1,1,1,0), (0,1,0,1), (0.125,2,4,0.25)])
def test_dual_fp4_independent_alpha_and_zero(gains):
    _case(256, 256, 128, gains)


def test_dual_fp4_nondefault_stream():
    _case(256, 256, 896, stream=torch.cuda.Stream())


@pytest.mark.parametrize("m", [8192, 131072])
def test_dual_fp4_target_prefill(m):
    _case(m, 4864, 896)


def test_dual_fp4_zero_bias():
    _case(256, 256, 128, bias_gain=0.0)


@pytest.mark.parametrize("invalid", ["dtype", "shape", "device", "stride", "alias"])
def test_dual_fp4_rejects_invalid_correction(invalid):
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
    from Metis.Metis.fused_residual_lowrank_nvfp4 import fused_residual_lowrank_nvfp4
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    q = NVFP4Quantizer(rowwise=True, columnwise=False, with_amax_reduction=False,
        with_rht=False, with_post_rht_amax=False, with_2d_quantization=False,
        stochastic_rounding=False)
    packed = [q.quantize(torch.randn(128,k,device="cuda",dtype=torch.bfloat16))
              for k in (128,128,64,64)]
    output = torch.empty(128,128,device="cuda",dtype=torch.bfloat16)
    candidates = {
        "dtype": torch.zeros(128,device="cuda",dtype=torch.float32),
        "shape": torch.zeros(1,128,device="cuda",dtype=torch.bfloat16),
        "device": torch.zeros(128,dtype=torch.bfloat16),
        "stride": torch.zeros(256,device="cuda",dtype=torch.bfloat16)[::2],
        "alias": output[0],
    }
    with pytest.raises(RuntimeError, match="residual_mean_correction"):
        fused_residual_lowrank_nvfp4(*packed,candidates[invalid],output)
