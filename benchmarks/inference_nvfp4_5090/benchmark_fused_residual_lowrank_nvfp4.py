"""Compare the three post-quantization paths using CUDA-event medians.

Projection, Z materialization, and independent quantization are outside timing.
Use --only dual for NCU so its kernel filter cannot capture a baseline kernel.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

from Metis.Metis.fused_residual_lowrank_nvfp4 import fused_residual_lowrank_nvfp4
from Metis.Metis.fused_residual_lowrank_bf16 import fused_residual_lowrank_bf16


def measure(fn, warmup, iterations):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8192)
    parser.add_argument("--input-features", type=int, default=896)
    parser.add_argument("--output-features", type=int, default=4864)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--only", choices=("all", "dual"), default="all")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rank != 64 or args.warmup < 0 or args.iterations < 1:
        parser.error("rank must be 64, warmup >= 0, iterations >= 1")
    torch.manual_seed(20260906)
    m, n, k = args.rows, args.output_features, args.input_features
    tensors = [torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
               for shape in ((m,k), (n,k), (m,64), (n,64))]
    packed = []
    for t in tensors:
        q = NVFP4Quantizer(rowwise=True, columnwise=False, with_amax_reduction=False,
                           with_rht=False, with_post_rht_amax=False,
                           with_2d_quantization=False, stochastic_rounding=False)
        p = q.quantize(t)
        tex.swizzle_scales_for_gemm_(p)
        packed.append(p)
    x, r, z, u = packed
    output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

    correction = torch.linspace(-1, 1, n, device="cuda", dtype=torch.bfloat16)

    def te_pair():
        general_gemm(r, x, out_dtype=torch.bfloat16, layout="TN", out=output)
        general_gemm(u, z, out_dtype=torch.bfloat16, layout="TN", out=output,
                     beta=1.0, accumulate=True)
        output.add_(correction)

    def bf16():
        fused_residual_lowrank_bf16(x, r, tensors[2], tensors[3], output)
        output.add_(correction)

    def dual():
        fused_residual_lowrank_nvfp4(x, r, z, u, correction, output)

    paths = {"dual_fp4_single_store": dual}
    if args.only == "all":
        paths = {"te_fp4_plus_te_nvfp4": te_pair, "bf16_single_store": bf16, **paths}
    result = {"m": m, "n": n, "k": k, "rank": 64,
              "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "scope": "post-quantization including device alpha setup and launch",
              "timings": {name: measure(fn, args.warmup, args.iterations)
                          for name, fn in paths.items()}}
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
