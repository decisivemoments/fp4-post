#!/usr/bin/env python3
"""Baseline for the future fused residual-FP4 + low-rank-BF16 kernel.

This intentionally times only post-quantization forward work.  ``X``, ``R``
and ``V`` are packed once, then the measured sequence exactly mirrors the
current BF16 low-rank forward path:

    residual_fp4(X, R) + (fp4(X, V) * s) @ U.T

The future fused kernel must replace the residual GEMM plus ``addmm_`` without
materializing the residual [M,N] matrix.  Keeping this baseline independent of
model loading makes each performance delta reproducible.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from transformer_engine.pytorch.cpp_extensions import general_gemm

from Metis.Metis.native_nvfp4 import _get_quantizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--input-features", type=int, default=896)
    parser.add_argument("--output-features", type=int, default=4864)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def time_cuda(fn: Callable[[], None], warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def time_cuda_with_reset(
    reset: Callable[[], None], fn: Callable[[], None], warmup: int, iterations: int
) -> float:
    """Run reset outside CUDA events, then time only the target operation."""
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iterations):
        reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    return statistics.median(timings)


def main() -> None:
    args = parse_args()
    if args.rows % 16 or args.input_features % 16 or args.output_features % 16:
        raise ValueError("NVFP4 M/K/N dimensions must be divisible by 16")
    torch.manual_seed(20260904)
    device = torch.device("cuda")
    x = torch.randn(args.rows, args.input_features, device=device, dtype=torch.bfloat16)
    residual = torch.randn(args.output_features, args.input_features, device=device, dtype=torch.bfloat16)
    v = torch.randn(args.rank, args.input_features, device=device, dtype=torch.bfloat16)
    u = torch.randn(args.output_features, args.rank, device=device, dtype=torch.bfloat16)
    singular_values = torch.randn(args.rank, device=device, dtype=torch.bfloat16)
    mean_r = torch.randn(args.output_features, device=device, dtype=torch.bfloat16)
    mean_v = torch.randn(args.rank, device=device, dtype=torch.bfloat16)

    quantizer = _get_quantizer(
        device, stochastic_rounding=False, rowwise=True, columnwise=False
    )
    xq = quantizer.quantize(x)
    residual_q = quantizer.quantize(residual)
    v_q = quantizer.quantize(v)
    residual_output = torch.empty(
        args.rows, args.output_features, device=device, dtype=torch.bfloat16
    )
    v_output = torch.empty(args.rows, args.rank, device=device, dtype=torch.bfloat16)

    def run_residual() -> None:
        general_gemm(
            residual_q, xq, out_dtype=torch.bfloat16, layout="TN",
            bias=mean_r, out=residual_output,
        )

    def run_v() -> None:
        general_gemm(v_q, xq, out_dtype=torch.bfloat16, layout="TN", bias=mean_v, out=v_output)

    def run_scale_addmm() -> None:
        residual_output.addmm_(v_output * singular_values, u.T)

    # The total must restore C because addmm_ is in-place.
    def run_total() -> None:
        run_residual()
        run_v()
        run_scale_addmm()

    residual_ms = time_cuda(run_residual, args.warmup, args.iterations)
    v_ms = time_cuda(run_v, args.warmup, args.iterations)
    run_residual()
    torch.cuda.synchronize()
    residual_seed = residual_output.clone()
    scale_addmm_ms = time_cuda_with_reset(
        lambda: residual_output.copy_(residual_seed),
        run_scale_addmm,
        args.warmup,
        args.iterations,
    )
    total_ms = time_cuda(run_total, args.warmup, args.iterations)
    print({
        "shape": [args.rows, args.input_features, args.output_features],
        "rank": args.rank,
        "residual_fp4_ms": residual_ms,
        "v_fp4_ms": v_ms,
        "scale_plus_addmm_estimated_ms": scale_addmm_ms,
        "baseline_total_ms": total_ms,
    })


if __name__ == "__main__":
    main()
