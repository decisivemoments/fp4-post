#!/usr/bin/env python3
"""Compare experimental rank-64 BF16 kernel with torch.addmm_."""

from __future__ import annotations

import argparse
import statistics

import torch

from Metis.Metis.lowrank_bf16_fused import (
    lowrank_add_bf16,
    lowrank_add_bf16_cutlass,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--columns", type=int, default=4864)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--implementation",
        choices=("wmma", "cutlass"),
        default="wmma",
        help="Experimental kernel to compare; CUTLASS BF16 is diagnostic only on SM120.",
    )
    return parser.parse_args()


def time_cuda(fn, output: torch.Tensor, base: torch.Tensor, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        output.copy_(base)
        fn(output)
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        output.copy_(base)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(output)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)


def main() -> None:
    args = parse_args()
    if args.rows % 16 or args.columns % 128:
        raise ValueError("rows must be divisible by 16 and columns by 128")
    torch.manual_seed(20260904)
    scaled_v = torch.randn(args.rows, 64, device="cuda", dtype=torch.bfloat16)
    u_transpose = torch.randn(64, args.columns, device="cuda", dtype=torch.bfloat16)
    base = torch.randn(args.rows, args.columns, device="cuda", dtype=torch.bfloat16)
    custom_output = torch.empty_like(base)
    addmm_output = torch.empty_like(base)

    implementations = {
        "wmma": lowrank_add_bf16,
        "cutlass": lowrank_add_bf16_cutlass,
    }
    custom_ms = time_cuda(
        lambda output: implementations[args.implementation](
            scaled_v, u_transpose, output
        ),
        custom_output, base, args.warmup, args.iterations,
    )
    addmm_ms = time_cuda(
        lambda output: output.addmm_(scaled_v, u_transpose),
        addmm_output, base, args.warmup, args.iterations,
    )
    print({
        "shape": [args.rows, 64, args.columns],
        "implementation": args.implementation,
        "custom_ms": custom_ms,
        "addmm_ms": addmm_ms,
        "addmm_over_custom": addmm_ms / custom_ms,
        "max_abs_error": (custom_output.float() - addmm_output.float()).abs().max().item(),
    })


if __name__ == "__main__":
    main()
