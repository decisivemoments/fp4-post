#!/usr/bin/env python3
"""Microbenchmark native Transformer Engine NVFP4 GEMMs on Qwen shapes."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import transformer_engine
import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions import general_gemm


CASES = {
    "qwen05_q": (512, 896, 896),
    "qwen05_up": (512, 896, 4864),
    "qwen05_down": (512, 4864, 896),
    "qwen15_q": (512, 1536, 1536),
    "qwen15_up": (512, 1536, 8960),
    "qwen15_down": (512, 8960, 1536),
    "qwen05_v_rank64": (512, 896, 64),
    "qwen15_v_rank64": (512, 1536, 64),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        default=list(CASES),
    )
    parser.add_argument(
        "--operations",
        nargs="+",
        choices=["fprop", "dgrad", "wgrad"],
        default=["fprop", "dgrad", "wgrad"],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


def quantizer(*, stochastic_rounding: bool = False) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=stochastic_rounding,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
    }


def time_cuda(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return times


def native_gemm(operation: str, x4, w4, g4) -> torch.Tensor:
    if operation == "fprop":
        return general_gemm(
            w4,
            x4,
            out_dtype=torch.bfloat16,
            layout="TN",
        )[0]
    if operation == "dgrad":
        return general_gemm(
            w4,
            g4,
            out_dtype=torch.bfloat16,
            layout="NN",
            grad=True,
        )[0]
    if operation == "wgrad":
        return general_gemm(
            x4,
            g4,
            out_dtype=torch.bfloat16,
            layout="NT",
            grad=True,
        )[0]
    raise ValueError(operation)


def bf16_gemm(operation: str, x: torch.Tensor, w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    if operation == "fprop":
        return F.linear(x, w)
    if operation == "dgrad":
        return g @ w
    if operation == "wgrad":
        return g.T @ x
    raise ValueError(operation)


def reference_gemm(operation: str, x4, w4, g4) -> torch.Tensor:
    x = x4.dequantize(dtype=torch.bfloat16)
    w = w4.dequantize(dtype=torch.bfloat16)
    g = g4.dequantize(dtype=torch.bfloat16)
    return bf16_gemm(operation, x, w, g)


def run_case(
    name: str,
    operation: str,
    *,
    warmup: int,
    iterations: int,
    seed: int,
    profile_only: bool,
) -> dict:
    m, k, n = CASES[name]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(m, n, dtype=torch.bfloat16, device="cuda")
    x_quantizer = quantizer()
    weight_quantizer = quantizer()
    grad_quantizer = quantizer(stochastic_rounding=False)
    x4 = x_quantizer.quantize(x)
    w4 = weight_quantizer.quantize(w)
    g4 = grad_quantizer.quantize(g)

    native = native_gemm(operation, x4, w4, g4)
    reference = reference_gemm(operation, x4, w4, g4)
    error = (native.float() - reference.float()).abs()

    if profile_only:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"native_nvfp4_{name}_{operation}")
        for _ in range(iterations):
            native_gemm(operation, x4, w4, g4)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        return {
            "case": name,
            "operation": operation,
            "shape_mkn": [m, k, n],
            "profile_only": True,
        }

    bf16_times = time_cuda(
        lambda: bf16_gemm(operation, x, w, g),
        warmup,
        iterations,
    )
    native_times = time_cuda(
        lambda: native_gemm(operation, x4, w4, g4),
        warmup,
        iterations,
    )

    if operation == "fprop":
        pack_activation_times = time_cuda(
            lambda: general_gemm(
                w4,
                x_quantizer.quantize(x),
                out_dtype=torch.bfloat16,
                layout="TN",
            )[0],
            warmup,
            iterations,
        )
        pack_all_times = time_cuda(
            lambda: general_gemm(
                weight_quantizer.quantize(w),
                x_quantizer.quantize(x),
                out_dtype=torch.bfloat16,
                layout="TN",
            )[0],
            warmup,
            iterations,
        )
    else:
        pack_activation_times = []
        pack_all_times = []

    bf16_summary = summarize(bf16_times)
    native_summary = summarize(native_times)
    result = {
        "case": name,
        "operation": operation,
        "shape_mkn": [m, k, n],
        "bf16_ms": bf16_summary,
        "native_nvfp4_gemm_ms": native_summary,
        "gemm_speedup": bf16_summary["median"] / native_summary["median"],
        "absolute_error": {
            "max": float(error.max().item()),
            "mean": float(error.mean().item()),
        },
        "packed_types": {
            "activation": type(x4).__name__,
            "weight": type(w4).__name__,
            "gradient": type(g4).__name__,
        },
    }
    if pack_activation_times:
        result["pack_activation_plus_gemm_ms"] = summarize(pack_activation_times)
        result["pack_all_plus_gemm_ms"] = summarize(pack_all_times)
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    available, reason = te.is_nvfp4_available(return_reason=True)
    if not available:
        raise RuntimeError(f"Transformer Engine NVFP4 unavailable: {reason}")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")

    results = []
    for case in args.cases:
        for operation in args.operations:
            result = run_case(
                case,
                operation,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=args.seed,
                profile_only=args.profile_only,
            )
            results.append(result)
            if args.profile_only:
                print(f"profiled {case} {operation}", flush=True)
            else:
                print(
                    f"{case:20s} {operation:5s} "
                    f"BF16={result['bf16_ms']['median']:.4f} ms "
                    f"NVFP4={result['native_nvfp4_gemm_ms']['median']:.4f} ms "
                    f"speedup={result['gemm_speedup']:.3f}x",
                    flush=True,
                )

    payload = {
        "schema_version": 1,
        "configuration": {
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
            "nvfp4": {
                "block_size": 16,
                "two_dimensional_weight_scaling": False,
                "random_hadamard_transform": False,
                "stochastic_rounding": False,
            },
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformer_engine": transformer_engine.__version__,
            "gpu": torch.cuda.get_device_name(),
            "gpu_capability": list(torch.cuda.get_device_capability()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
