#!/usr/bin/env python3
"""Measure one Qwen projection or one Qwen decoder block on one CUDA GPU.

The benchmark deliberately measures steady-state *inference* latency.  The
NVFP4 setup (SVD and packed-weight creation) happens before warmup, whereas
activation packing remains in the timed interval because it is needed for
every new request.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Metis.Metis.native_nvfp4 import (  # noqa: E402
    NativeFullNVFP4Linear,
    replace_linear_with_native_full_nvfp4,
    require_native_nvfp4,
    set_native_nvfp4_profiling,
)
from Metis.Metis.qwen2_block_nvfp4 import (  # noqa: E402
    fuse_qwen2_decoder_layer_norms,
)
from Metis.Metis.direct_nvfp4 import (  # noqa: E402
    DirectNVFP4Linear,
    replace_linear_with_direct_nvfp4,
)


MODEL_IDS = {
    "qwen0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
}
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")
BF16_PEAK_TFLOPS = 419.0
NVFP4_PEAK_TFLOPS = 1676.0


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("linear", "transformer"), required=True)
    parser.add_argument("--model", choices=tuple(MODEL_IDS), required=True)
    parser.add_argument("--model-path", help="Local path overrides the Hugging Face model id.")
    parser.add_argument("--modes", nargs="+", choices=("bf16", "nvfp4", "direct_nvfp4"), default=("bf16", "nvfp4"))
    parser.add_argument(
        "--projections",
        nargs="+",
        choices=PROJECTIONS,
        help="Only benchmark these projections; valid only with --scope linear.",
    )
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument(
        "--lowrank-compute",
        choices=("bf16", "nvfp4"),
        default="bf16",
        help="Low-rank U @ scaled_v implementation for native NVFP4 inference.",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--bf16-peak-tflops", type=float, default=BF16_PEAK_TFLOPS)
    parser.add_argument("--nvfp4-peak-tflops", type=float, default=NVFP4_PEAK_TFLOPS)
    parser.add_argument(
        "--activation-mode",
        choices=("cached", "fresh"),
        default="fresh",
        help=(
            "cached reuses one activation (pure backend microbenchmark); "
            "fresh alternates preallocated tensors so every native forward "
            "packs its activation, without timing input copies."
        ),
    )
    parser.add_argument(
        "--activation-layout",
        choices=("full", "rowwise"),
        default="full",
        help=(
            "full emits rowwise and columnwise NVFP4 activation layouts; "
            "rowwise is an inference-only experiment that omits the "
            "backward-only columnwise layout."
        ),
    )
    parser.add_argument(
        "--activation-pack-backend",
        choices=("te", "centered_cuda"),
        default="te",
        help="TE reference pack or the project-local centered CUDA pack.",
    )
    parser.add_argument(
        "--activation-pack-backends",
        nargs="+",
        choices=("te", "centered_cuda"),
        help=(
            "Benchmark multiple NVFP4 activation-pack backends in one linear "
            "run. BF16 is measured once; every NVFP4 row carries its backend "
            "and comparison fields report each backend against BF16."
        ),
    )
    parser.add_argument(
        "--profile-native-ranges",
        action="store_true",
        help="Emit NVTX ranges for native activation pack, FP4 GEMM, and BF16 correction paths.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket measured iterations with cudaProfilerStart/Stop for Nsight Systems.",
    )
    parser.add_argument(
        "--fuse-rmsnorm-quant",
        action="store_true",
        help=(
            "For NVFP4 transformer scope, route Qwen RMSNorm directly into "
            "the shared Q/K/V and Gate/Up centered activation packs."
        ),
    )
    parser.add_argument(
        "--disable-dual-fp4-fusion",
        action="store_true",
        help=(
            "Keep the legacy TE beta=1 residual/low-rank two-GEMM path. "
            "This is a controlled baseline for dual-FP4 fusion."
        ),
    )
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def time_cuda(
    fn: Callable[[], Any],
    warmup: int,
    iterations: int,
    *,
    cuda_profiler_range: bool = False,
) -> dict[str, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        values = []
        if cuda_profiler_range:
            torch.cuda.cudart().cudaProfilerStart()
        try:
            for _ in range(iterations):
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
        finally:
            if cuda_profiler_range:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
    return {"min": min(values), "p10": percentile(values, .10), "median": statistics.median(values), "mean": statistics.fmean(values), "p90": percentile(values, .90)}


def decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    base = getattr(model, "model", model)
    layers = getattr(base, "layers", None)
    if layers is None:
        raise RuntimeError("Expected a Qwen-style model.model.layers decoder stack")
    return layers


def projection_modules(layer: torch.nn.Module) -> dict[str, torch.nn.Module]:
    found = {
        name.rsplit(".", 1)[-1]: module
        for name, module in layer.named_modules()
        if name.rsplit(".", 1)[-1] in PROJECTIONS
        and isinstance(module, (torch.nn.Linear, NativeFullNVFP4Linear, DirectNVFP4Linear))
    }
    missing = set(PROJECTIONS) - set(found)
    if missing:
        raise RuntimeError(f"Qwen decoder layer lacks projections: {sorted(missing)}")
    return found


def load_layer(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.nn.Module, Any]:
    source = args.model_path or MODEL_IDS[args.model]
    model = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=args.trust_remote_code
    ).to(args.device).eval()
    layers = decoder_layers(model)
    if not 0 <= args.layer_index < len(layers):
        raise ValueError(f"--layer-index must be in [0, {len(layers) - 1}]")
    return model, layers[args.layer_index], model.config


def block_inputs(model: torch.nn.Module, config: Any, batch: int, seq: int, device: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    hidden = torch.randn(batch, seq, config.hidden_size, device=device, dtype=torch.bfloat16)
    position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
    # Qwen SDPA accepts the conventional additive causal mask when a decoder
    # layer is invoked directly (rather than through QwenModel.forward).
    mask = torch.full((seq, seq), torch.finfo(torch.bfloat16).min, device=device, dtype=torch.bfloat16).triu(1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch, 1, seq, seq)
    base = getattr(model, "model", model)
    rotary = getattr(base, "rotary_emb", None)
    kwargs: dict[str, Any] = {"attention_mask": mask, "position_ids": position_ids, "cache_position": torch.arange(seq, device=device)}
    if rotary is not None:
        kwargs["position_embeddings"] = rotary(hidden, position_ids)
    return hidden, kwargs


def call_layer(layer: torch.nn.Module, hidden: torch.Tensor, kwargs: dict[str, Any]) -> Any:
    accepted = inspect.signature(layer.forward).parameters
    output = layer(hidden, **{key: value for key, value in kwargs.items() if key in accepted})
    return output[0] if isinstance(output, tuple) else output


def logical_projection_flops(batch: int, seq: int, projection: torch.nn.Module) -> int:
    return 2 * batch * seq * projection.in_features * projection.out_features


def decoder_gemm_flops(layer: torch.nn.Module, batch: int, seq: int) -> int:
    return sum(logical_projection_flops(batch, seq, module) for module in projection_modules(layer).values())


def measured_rate(flops: int, ms: float) -> float:
    return flops / (ms * 1e9)  # FLOP / ms -> TFLOP / s


def metric(flops: int, timing: dict[str, float], peak: float) -> dict[str, float]:
    tflops = measured_rate(flops, timing["median"])
    return {"logical_flops": flops, "logical_tflops": tflops, "peak_tflops": peak, "logical_peak_utilization": tflops / peak}


def replace_block_with_nvfp4(
    layer: torch.nn.Module,
    rank: int,
    *,
    activation_layout: str,
    lowrank_compute: str,
    activation_pack_backend: str,
    enable_dual_fp4_fusion: bool,
) -> list[str]:
    require_native_nvfp4()
    return replace_linear_with_native_full_nvfp4(
        layer,
        rank=rank,
        stochastic_rounding=False,
        activation_columnwise=activation_layout == "full",
        lowrank_compute=lowrank_compute,
        activation_pack_backend=activation_pack_backend,
        enable_dual_fp4_fusion=enable_dual_fp4_fusion,
    )


def activation_supplier(
    batch: int,
    seq: int,
    features: int,
    args: argparse.Namespace,
) -> Callable[[], torch.Tensor]:
    """Return cached or alternating preallocated activation tensors.

    NativeActivationGroup keeps only the immediately previous tensor identity.
    Alternating two live tensors therefore exercises a new activation pack on
    every call without charging a tensor clone/copy to the measured linear.
    """
    tensors = [
        torch.randn(batch, seq, features, device=args.device, dtype=torch.bfloat16)
        for _ in range(1 if args.activation_mode == "cached" else 2)
    ]
    index = 0

    def next_tensor() -> torch.Tensor:
        nonlocal index
        value = tensors[index]
        index = (index + 1) % len(tensors)
        return value

    return next_tensor


def benchmark_projection(
    layer: torch.nn.Module,
    batch: int,
    seq: int,
    mode: str,
    args: argparse.Namespace,
    *,
    activation_pack_backend: str | None = None,
) -> list[dict[str, Any]]:
    measured_layer = copy.deepcopy(layer) if mode != "bf16" else layer
    if mode == "nvfp4":
        replace_block_with_nvfp4(
            measured_layer,
            args.rank,
            activation_layout=args.activation_layout,
            lowrank_compute=args.lowrank_compute,
            activation_pack_backend=activation_pack_backend or args.activation_pack_backend,
            enable_dual_fp4_fusion=not args.disable_dual_fp4_fusion,
        )
    elif mode == "direct_nvfp4":
        replace_linear_with_direct_nvfp4(measured_layer, set(PROJECTIONS))
    rows = []
    for name, projection in projection_modules(measured_layer).items():
        if args.projections is not None and name not in args.projections:
            continue
        next_activation = activation_supplier(batch, seq, projection.in_features, args)
        timing = time_cuda(
            lambda: projection(next_activation()),
            args.warmup,
            args.iterations,
            cuda_profiler_range=args.cuda_profiler_range,
        )
        flops = logical_projection_flops(batch, seq, projection)
        rows.append({"projection": name, "mode": mode, "activation_mode": args.activation_mode, "activation_layout": args.activation_layout, "activation_pack_backend": activation_pack_backend if mode == "nvfp4" else ("te_direct" if mode == "direct_nvfp4" else None), "batch_size": batch, "seq_length": seq, "shape_mkn": [batch * seq, projection.in_features, projection.out_features], "latency_ms": timing, **metric(flops, timing, args.nvfp4_peak_tflops if mode != "bf16" else args.bf16_peak_tflops)})
    return rows


def benchmark_block(model: torch.nn.Module, layer: torch.nn.Module, batch: int, seq: int, mode: str, args: argparse.Namespace) -> dict[str, Any]:
    measured_layer = copy.deepcopy(layer) if mode != "bf16" else layer
    replaced = (
        replace_block_with_nvfp4(
            measured_layer,
            args.rank,
            activation_layout=args.activation_layout,
            lowrank_compute=args.lowrank_compute,
            activation_pack_backend=args.activation_pack_backend,
            enable_dual_fp4_fusion=not args.disable_dual_fp4_fusion,
        )
        if mode == "nvfp4"
        else []
    )
    if mode == "direct_nvfp4":
        replaced = replace_linear_with_direct_nvfp4(measured_layer, set(PROJECTIONS))
    if mode == "nvfp4" and args.fuse_rmsnorm_quant:
        if args.activation_layout != "rowwise":
            raise ValueError("--fuse-rmsnorm-quant requires --activation-layout rowwise")
        if args.activation_pack_backend != "centered_cuda":
            raise ValueError(
                "--fuse-rmsnorm-quant requires --activation-pack-backend centered_cuda"
            )
        measured_layer = fuse_qwen2_decoder_layer_norms(measured_layer)
    hidden, kwargs = block_inputs(model, model.config, batch, seq, args.device)
    timing = time_cuda(
        lambda: call_layer(measured_layer, hidden, kwargs),
        args.warmup,
        args.iterations,
        cuda_profiler_range=args.cuda_profiler_range,
    )
    flops = decoder_gemm_flops(measured_layer, batch, seq)
    # Keep the result schema compatible with benchmark_projection().  The
    # shared paired_speedups()/backend_comparisons() helpers run for both
    # scopes and index every NVFP4 row by this field.
    return {
        "scope": "transformer",
        "mode": mode,
        "activation_pack_backend": (
            args.activation_pack_backend if mode == "nvfp4" else ("te_direct" if mode == "direct_nvfp4" else None)
        ),
        "fuse_rmsnorm_quant": mode == "nvfp4" and args.fuse_rmsnorm_quant,
        "dual_fp4_fusion": mode == "nvfp4" and not args.disable_dual_fp4_fusion,
        "batch_size": batch,
        "seq_length": seq,
        "layer_index": args.layer_index,
        "replaced_modules": replaced,
        "latency_ms": timing,
        **metric(
            flops,
            timing,
            args.nvfp4_peak_tflops if mode != "bf16" else args.bf16_peak_tflops,
        ),
    }


def paired_speedups(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report each NVFP4 backend's median speedup over the BF16 baseline."""
    pairs: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in results:
        key = (result.get("projection"), result["batch_size"], result["seq_length"])
        pair = pairs.setdefault(key, {"bf16": None, "nvfp4": {}})
        if result["mode"] == "bf16":
            pair["bf16"] = result
        else:
            pair["nvfp4"][result["activation_pack_backend"]] = result
    rows = []
    for key, modes in pairs.items():
        if modes["bf16"] is None:
            continue
        bf16_ms = modes["bf16"]["latency_ms"]["median"]
        for backend, nvfp4 in modes["nvfp4"].items():
            nvfp4_ms = nvfp4["latency_ms"]["median"]
            rows.append({
                "projection": key[0],
                "batch_size": key[1],
                "seq_length": key[2],
                "activation_pack_backend": backend,
                "bf16_median_ms": bf16_ms,
                "nvfp4_median_ms": nvfp4_ms,
                "bf16_over_nvfp4": bf16_ms / nvfp4_ms,
            })
    return rows


def backend_comparisons(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join BF16, TE, and centered CUDA rows for convenient A/B reporting."""
    pairs: dict[tuple[Any, ...], dict[str, Any]] = {}
    for result in results:
        key = (result.get("projection"), result["batch_size"], result["seq_length"])
        pair = pairs.setdefault(key, {"bf16": None, "nvfp4": {}})
        if result["mode"] == "bf16":
            pair["bf16"] = result
        else:
            pair["nvfp4"][result["activation_pack_backend"]] = result
    rows = []
    for key, modes in pairs.items():
        bf16 = modes["bf16"]
        te = modes["nvfp4"].get("te")
        centered = modes["nvfp4"].get("centered_cuda")
        if bf16 is None:
            continue
        row: dict[str, Any] = {
            "projection": key[0], "batch_size": key[1], "seq_length": key[2],
            "bf16_median_ms": bf16["latency_ms"]["median"],
        }
        for label, value in (("te", te), ("centered_cuda", centered)):
            if value is not None:
                latency = value["latency_ms"]["median"]
                row[f"{label}_median_ms"] = latency
                row[f"bf16_over_{label}"] = row["bf16_median_ms"] / latency
        if te is not None and centered is not None:
            row["te_over_centered_cuda"] = te["latency_ms"]["median"] / centered["latency_ms"]["median"]
        rows.append(row)
    return rows


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.seq_length <= 0 or any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("sequence length and batch sizes must be positive")
    if args.scope == "transformer" and args.projections is not None:
        raise ValueError("--projections is valid only with --scope linear")
    if args.scope != "linear" and args.activation_pack_backends is not None:
        raise ValueError("--activation-pack-backends is valid only with --scope linear")
    if args.fuse_rmsnorm_quant and args.scope != "transformer":
        raise ValueError("--fuse-rmsnorm-quant is valid only with --scope transformer")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(20260902)
    set_native_nvfp4_profiling(args.profile_native_ranges)
    model, layer, config = load_layer(args)
    results: list[dict[str, Any]] = []
    for batch in args.batch_sizes:
        for mode in args.modes:
            if args.scope == "linear":
                if mode == "nvfp4":
                    backends = args.activation_pack_backends or (args.activation_pack_backend,)
                    for backend in backends:
                        results.extend(benchmark_projection(
                            layer, batch, args.seq_length, mode, args,
                            activation_pack_backend=backend,
                        ))
                else:
                    results.extend(benchmark_projection(layer, batch, args.seq_length, mode, args))
            else:
                results.append(benchmark_block(model, layer, batch, args.seq_length, mode, args))
    payload = {"benchmark": "qwen_one_layer_inference", "scope": args.scope, "model": args.model, "model_source": str(args.model_path or MODEL_IDS[args.model]), "config": {"hidden_size": config.hidden_size, "intermediate_size": config.intermediate_size, "num_attention_heads": config.num_attention_heads, "num_key_value_heads": config.num_key_value_heads}, "peak_tflops": {"bf16_dense": args.bf16_peak_tflops, "nvfp4_dense": args.nvfp4_peak_tflops}, "rank": args.rank, "lowrank_compute": args.lowrank_compute, "activation_mode": args.activation_mode, "activation_layout": args.activation_layout, "activation_pack_backends": args.activation_pack_backends or [args.activation_pack_backend], "fuse_rmsnorm_quant": args.fuse_rmsnorm_quant, "dual_fp4_fusion": not args.disable_dual_fp4_fusion, "profile_native_ranges": args.profile_native_ranges, "cuda_profiler_range": args.cuda_profiler_range, "results": results, "paired_speedups": paired_speedups(results), "backend_comparisons": backend_comparisons(results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
