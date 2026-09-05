#!/usr/bin/env python3
"""Compare project SM120 NVFP4 kernel with TE NVFP4 GEMM at beta=1.

Quantization and TE scale swizzling are outside CUDA events. Both candidates
therefore receive identical, separately quantized rank-64 operands and the
same BF16 residual C matrix.
"""

from __future__ import annotations

import argparse
import statistics

import torch
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm

from Metis.Metis.lowrank_nvfp4_sm120 import lowrank_nvfp4_add_sm120
from Metis.Metis.native_nvfp4 import _get_quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--columns", type=int, default=4864)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def time_cuda(fn, output: torch.Tensor, base: torch.Tensor, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        output.copy_(base)
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        output.copy_(base)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main() -> None:
    args = parse_args()
    if args.rows % 128 or args.columns % 128:
        raise ValueError("SM120 reference tile requires rows and columns divisible by 128")
    torch.manual_seed(20260904)
    scaled_v = torch.randn(args.rows, 64, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(args.columns, 64, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(args.rows, args.columns, device="cuda", dtype=torch.bfloat16)
    quantizer = _get_quantizer(
        scaled_v.device,
        stochastic_rounding=False,
        rowwise=True,
        columnwise=False,
    )

    custom_a, custom_b = quantizer.quantize(scaled_v), quantizer.quantize(u)
    custom_output = base.clone()
    custom_ms = time_cuda(
        lambda: lowrank_nvfp4_add_sm120(custom_a, custom_b, custom_output),
        custom_output, base, args.warmup, args.iterations,
    )

    te_a, te_b = quantizer.quantize(scaled_v), quantizer.quantize(u)
    tex.swizzle_scales_for_gemm_(te_a)
    tex.swizzle_scales_for_gemm_(te_b)
    te_output = base.clone()
    te_ms = time_cuda(
        lambda: general_gemm(
            te_b, te_a, out_dtype=torch.bfloat16, layout="TN",
            out=te_output, beta=1.0, accumulate=True,
        ),
        te_output, base, args.warmup, args.iterations,
    )
    print({
        "shape": [args.rows, 64, args.columns],
        "custom_nvfp4_ms": custom_ms,
        "te_nvfp4_ms": te_ms,
        "te_over_custom": te_ms / custom_ms,
    })


if __name__ == "__main__":
    main()
