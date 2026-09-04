#!/usr/bin/env python3
"""Benchmark the U @ scaled_v low-rank path in BF16 and native NVFP4.

U is obtained from the actual rank-r SVD decomposition of the selected Qwen
projection.  U is packed once because it is static at inference; each fresh
scaled_v activation is multiplied by its singular values and packed in the
timed NVFP4 path.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Metis.Metis.native_nvfp4 import NativeFullNVFP4Linear, require_native_nvfp4  # noqa: E402


MODEL_IDS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BF16_PEAK_TFLOPS = 419.0
NVFP4_PEAK_TFLOPS = 1676.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_IDS, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--projection", choices=PROJECTIONS, default="up_proj")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(16, 32, 64, 128))
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-nvtx", action="store_true")
    parser.add_argument("--bf16-peak-tflops", type=float, default=BF16_PEAK_TFLOPS)
    parser.add_argument("--nvfp4-peak-tflops", type=float, default=NVFP4_PEAK_TFLOPS)
    return parser.parse_args()


def profile_range(enabled: bool, name: str):
    return torch.cuda.nvtx.range(name) if enabled else nullcontext()


def median_ms(fn: Callable[[], torch.Tensor], warmup: int, iterations: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        values = []
        for _ in range(iterations):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            values.append(start.elapsed_time(end))
    return statistics.median(values)


def quantizer(*, columnwise: bool) -> NVFP4Quantizer:
    return NVFP4Quantizer(
        rowwise=True,
        columnwise=columnwise,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=False,
        stochastic_rounding=False,
    )


def find_projection(model: torch.nn.Module, layer_index: int, name: str) -> torch.nn.Linear:
    layers = model.model.layers
    layer = layers[layer_index]
    for module_name, module in layer.named_modules():
        if module_name.rsplit(".", 1)[-1] == name and isinstance(module, torch.nn.Linear):
            return module
    raise RuntimeError(f"Projection {name!r} was not found in decoder layer {layer_index}")


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm((actual - expected).float())
        / torch.linalg.vector_norm(expected.float())
    )


def tflops(logical_flops: int, latency_ms: float) -> float:
    """Logical dense-equivalent TFLOP/s for a latency in milliseconds."""
    return logical_flops / (latency_ms * 1e9)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    require_native_nvfp4()
    if any(batch <= 0 for batch in args.batch_sizes) or args.seq_length <= 0:
        raise ValueError("batch sizes and sequence length must be positive")

    source = args.model_path or MODEL_IDS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(args.device).eval()
    projection = find_projection(model, args.layer_index, args.projection)
    lowrank = NativeFullNVFP4Linear.from_linear(
        projection,
        rank=args.rank,
        stochastic_rounding=False,
    )
    # U and singular_values remain separate: U is quantized as requested and
    # scaled_v is quantized after its runtime singular-value multiplication.
    u_weight = lowrank.u_weight.detach().contiguous()
    singular_values = lowrank.singular_values.detach()
    packed_u = quantizer(columnwise=True).quantize(u_weight)
    dynamic_quantizer = quantizer(columnwise=False)

    results = []
    for batch in args.batch_sizes:
        rows = batch * args.seq_length
        v_outputs = [
            torch.randn(rows, args.rank, device=args.device, dtype=torch.bfloat16)
            for _ in range(2)
        ]
        index = 0

        def next_v_output() -> torch.Tensor:
            nonlocal index
            value = v_outputs[index]
            index = (index + 1) % len(v_outputs)
            return value

        def bf16_path() -> torch.Tensor:
            with profile_range(args.profile_nvtx, "lowrank.bf16_scale_and_gemm"):
                return F.linear(next_v_output() * singular_values, u_weight)

        def nvfp4_path() -> torch.Tensor:
            scaled = next_v_output() * singular_values
            with profile_range(args.profile_nvtx, "lowrank.nvfp4_scaled_v_pack"):
                packed_scaled = dynamic_quantizer.quantize(scaled)
            with profile_range(args.profile_nvtx, "lowrank.nvfp4_gemm"):
                return general_gemm(
                    packed_u,
                    packed_scaled,
                    out_dtype=torch.bfloat16,
                    layout="TN",
                )[0]

        fixed_scaled = v_outputs[0] * singular_values
        packed_fixed_scaled = dynamic_quantizer.quantize(fixed_scaled)

        def packed_gemm() -> torch.Tensor:
            with profile_range(args.profile_nvtx, "lowrank.nvfp4_packed_gemm"):
                return general_gemm(
                    packed_u,
                    packed_fixed_scaled,
                    out_dtype=torch.bfloat16,
                    layout="TN",
                )[0]

        reference = F.linear(fixed_scaled, u_weight)
        actual = packed_gemm()
        bf16_ms = median_ms(bf16_path, args.warmup, args.iterations)
        nvfp4_ms = median_ms(nvfp4_path, args.warmup, args.iterations)
        packed_gemm_ms = median_ms(packed_gemm, args.warmup, args.iterations)
        flops = 2 * rows * args.rank * lowrank.out_features
        bf16_tflops = tflops(flops, bf16_ms)
        nvfp4_e2e_tflops = tflops(flops, nvfp4_ms)
        nvfp4_packed_gemm_tflops = tflops(flops, packed_gemm_ms)
        results.append(
            {
                "batch_size": batch,
                "rows": rows,
                "rank": args.rank,
                "u_shape": list(u_weight.shape),
                "logical_flops": flops,
                "bf16_scale_and_gemm_ms": bf16_ms,
                "nvfp4_scaled_v_pack_and_gemm_ms": nvfp4_ms,
                "nvfp4_packed_gemm_only_ms": packed_gemm_ms,
                "bf16_logical_tflops": bf16_tflops,
                "bf16_logical_peak_utilization": bf16_tflops / args.bf16_peak_tflops,
                "nvfp4_end_to_end_logical_tflops": nvfp4_e2e_tflops,
                "nvfp4_end_to_end_logical_peak_utilization": nvfp4_e2e_tflops / args.nvfp4_peak_tflops,
                "nvfp4_packed_gemm_logical_tflops": nvfp4_packed_gemm_tflops,
                "nvfp4_packed_gemm_logical_peak_utilization": nvfp4_packed_gemm_tflops / args.nvfp4_peak_tflops,
                "speedup_bf16_over_nvfp4_end_to_end": bf16_ms / nvfp4_ms,
                "speedup_bf16_over_nvfp4_packed_gemm": bf16_ms / packed_gemm_ms,
                "nvfp4_relative_l2_vs_bf16": relative_l2(actual, reference),
            }
        )

    payload = {
        "benchmark": "lowrank_u_scaled_v_bf16_vs_nvfp4",
        "model": args.model,
        "model_source": source,
        "projection": args.projection,
        "layer_index": args.layer_index,
        "peak_tflops": {
            "bf16_dense": args.bf16_peak_tflops,
            "nvfp4_dense": args.nvfp4_peak_tflops,
        },
        "note": "U and scaled_v are separately quantized; singular values are not folded into U.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
