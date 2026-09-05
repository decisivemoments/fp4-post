"""Time the post-quantization single-D-store fusion experiment.

This intentionally excludes activation quantization and V projection.  It
compares exactly the part being fused:
``TE residual FP4 GEMM + BF16 addmm`` versus the custom single-D-store kernel.
The custom implementation uses eight cooperative BF16 WMMA tiles for the
rank-64 correction.  It remains an experimental post-quantization kernel,
not yet the native linear dispatch path.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformer_engine.pytorch.cpp_extensions import general_gemm

from Metis.Metis.fused_residual_lowrank_bf16 import fused_residual_lowrank_bf16
from Metis.Metis.native_nvfp4 import _get_quantizer


def measure(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--input-features", type=int, default=896)
    parser.add_argument("--output-features", type=int, default=4864)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    assert args.rows % 128 == 0 and args.output_features % 128 == 0
    torch.manual_seed(20260906)
    x = torch.randn(args.rows, args.input_features, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(args.output_features, args.input_features, device="cuda", dtype=torch.bfloat16)
    z = torch.randn(args.rows, args.rank, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(args.output_features, args.rank, device="cuda", dtype=torch.bfloat16)
    quantizer = _get_quantizer(x.device, stochastic_rounding=False, rowwise=True, columnwise=False)
    packed_x, packed_residual = quantizer.quantize(x), quantizer.quantize(residual)
    output = torch.empty(args.rows, args.output_features, device="cuda", dtype=torch.bfloat16)
    reference = torch.empty_like(output)

    def te_baseline() -> None:
        general_gemm(packed_residual, packed_x, out_dtype=torch.bfloat16, layout="TN", out=reference)
        reference.addmm_(z, u.T)

    def fused() -> None:
        fused_residual_lowrank_bf16(packed_x, packed_residual, z, u, output)

    baseline_ms = measure(te_baseline, args.warmup, args.iterations)
    fused_ms = measure(fused, args.warmup, args.iterations)
    print({
        "rows": args.rows,
        "n": args.output_features,
        "k": args.input_features,
        "rank": args.rank,
        "te_residual_plus_addmm_ms": baseline_ms,
        "fused_single_d_store_ms": fused_ms,
        "speedup": baseline_ms / fused_ms,
    })


if __name__ == "__main__":
    main()
