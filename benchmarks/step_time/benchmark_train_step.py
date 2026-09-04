#!/usr/bin/env python3
"""Benchmark one full BF16 or current direct_fp4 training step."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import transformers
from torch import nn
from transformers import AutoModelForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Metis.Metis.bitlinear import BitLinear, LinearLowbitFunction  # noqa: E402
from Metis.Metis.quant import nvfp4_nosr_qdq_compile_status  # noqa: E402


TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass
class DirectNvfp4Args:
    device: torch.device
    cache_quantized_weight: bool = False
    compile_qdq: bool = False

    enable_forward_svd: bool = False
    enable_lowbit: bool = True
    enable_te: bool = False
    q_forward_input: str = "nvfp4e2m1bnosr"
    q_forward_weight: str = "nvfp4e2m1bnosr"
    q_backward_input: str = "nvfp4e2m1b"
    q_backward_weight: str = "nvfp4e2m1b"
    q_backward_outputgrad: str = "nvfp4e2m1b"
    enable_backward_svd: bool = False
    backward_lowrank_svd: int = 0
    backward_lowrank_niter: int = 0
    enable_activation_svd: bool = False
    activation_lowrank_svd: int = 0
    activation_lowrank_niter: int = 0
    activation_broadcast_dim: int = 0
    backward_broadcast_dim: int = -1
    enable_nv_recipe: bool = False
    hadamard_workspace_mb: int = 128
    hadamard_tile_size: int = 16
    hadamard_backend: str = "auto"
    tp_simulation: bool = False
    tp_parts: int = 1
    metis_mode: str = "mean"
    forward_svd_warmup_steps: int = 50
    forward_svd_rank: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument("--mode", choices=["bf16", "nvfp4"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--compile-qdq", action="store_true")
    parser.add_argument("--cache-quantized-weight", action="store_true")
    return parser.parse_args()


def replace_with_direct_nvfp4(
    model: nn.Module,
    *,
    device: torch.device,
    compile_qdq: bool,
    cache_quantized_weight: bool,
) -> list[str]:
    metis_args = DirectNvfp4Args(
        device=device,
        compile_qdq=compile_qdq,
        cache_quantized_weight=cache_quantized_weight,
    )
    replaced: list[str] = []

    for name, module in list(model.named_modules()):
        if name.rsplit(".", 1)[-1] not in TARGET_MODULES:
            continue
        if not isinstance(module, nn.Linear):
            continue

        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        new_layer = BitLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            args=metis_args,
            bias=module.bias is not None,
            dtype=torch.bfloat16,
            compute_dtype=torch.bfloat16,
        )
        with torch.no_grad():
            new_layer.warmup_linear.weight.copy_(module.weight)
            if module.bias is not None:
                new_layer.warmup_linear.bias.copy_(module.bias)
        new_layer.layer_name = name
        setattr(parent, child_name, new_layer)
        replaced.append(name)

    if not replaced:
        raise RuntimeError("No Qwen projection layers were replaced with NVFP4")
    return replaced


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot summarize an empty list")
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
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def make_batch(
    cache: dict[str, torch.Tensor],
    *,
    batch_size: int,
    seq_length: int,
    step: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample_count = cache["input_ids"].shape[0]
    start = (step * batch_size) % sample_count
    indices = [(start + offset) % sample_count for offset in range(batch_size)]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    batch = {
        key: value.index_select(0, index_tensor)[:, :seq_length].to(
            device, non_blocking=False
        )
        for key, value in cache.items()
    }
    return batch


def run_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    *,
    measure: bool,
) -> dict[str, float]:
    if measure:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        events[0].record()

    outputs = model(**batch)
    loss = outputs.loss
    if measure:
        events[1].record()

    loss.backward()
    if measure:
        events[2].record()

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if measure:
        events[3].record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
    else:
        torch.cuda.synchronize()
        wall_ms = 0.0

    loss_value = float(loss.detach().float().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(f"Non-finite loss: {loss_value}")

    if not measure:
        return {"loss": loss_value}

    return {
        "loss": loss_value,
        "wall_ms": wall_ms,
        "gpu_total_ms": events[0].elapsed_time(events[3]),
        "forward_ms": events[0].elapsed_time(events[1]),
        "backward_ms": events[1].elapsed_time(events[2]),
        "optimizer_ms": events[2].elapsed_time(events[3]),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0 or args.seq_length <= 1 or args.steps <= 0:
        raise ValueError("batch size and steps must be positive; seq length > 1")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    cache = torch.load(args.data_cache, map_location="cpu", weights_only=True)
    expected_keys = {"input_ids", "attention_mask", "labels"}
    if set(cache) != expected_keys:
        raise ValueError(f"Unexpected cache keys: {sorted(cache)}")
    if cache["input_ids"].shape[1] < args.seq_length:
        raise ValueError(
            f"Cache seq length {cache['input_ids'].shape[1]} < {args.seq_length}"
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.to(device)

    replaced_modules: list[str] = []
    if args.mode == "nvfp4":
        replaced_modules = replace_with_direct_nvfp4(
            model,
            device=device,
            compile_qdq=args.compile_qdq,
            cache_quantized_weight=args.cache_quantized_weight,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)

    for warmup_step in range(args.warmup_steps):
        batch = make_batch(
            cache,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            step=warmup_step,
            device=device,
        )
        run_step(model, optimizer, batch, measure=False)

    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, float]] = []
    for measured_step in range(args.steps):
        batch = make_batch(
            cache,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            step=args.warmup_steps + measured_step,
            device=device,
        )
        record = run_step(model, optimizer, batch, measure=True)
        record["step"] = measured_step
        records.append(record)
        print(
            f"{args.mode} step {measured_step + 1}/{args.steps}: "
            f"{record['wall_ms']:.3f} ms loss={record['loss']:.6f}",
            flush=True,
        )

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    padded_tokens_per_step = args.batch_size * args.seq_length
    nonpadding_tokens = []
    for measured_step in range(args.steps):
        batch = make_batch(
            cache,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            step=args.warmup_steps + measured_step,
            device=torch.device("cpu"),
        )
        nonpadding_tokens.append(int(batch["attention_mask"].sum().item()))

    timing_keys = [
        "wall_ms",
        "gpu_total_ms",
        "forward_ms",
        "backward_ms",
        "optimizer_ms",
    ]
    timing_summary = {
        key: summarize([float(record[key]) for record in records])
        for key in timing_keys
    }
    median_wall_seconds = timing_summary["wall_ms"]["median"] / 1000

    result: dict[str, Any] = {
        "schema_version": 1,
        "mode": args.mode,
        "model_path": str(args.model_path.resolve()),
        "data_cache": str(args.data_cache.resolve()),
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attention_implementation": args.attn_implementation,
        "optimizer": "torch.optim.AdamW(fused=True)",
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "padded_tokens_per_step": padded_tokens_per_step,
        "nonpadding_tokens_per_step": summarize(
            [float(value) for value in nonpadding_tokens]
        ),
        "padded_tokens_per_second_at_median": (
            padded_tokens_per_step / median_wall_seconds
        ),
        "parameter_count": total_parameters,
        "replaced_linear_count": len(replaced_modules),
        "replaced_linear_modules": replaced_modules,
        "nvfp4": {
            "implementation": (
                "Metis fake quantize-dequantize + BF16 torch.matmul"
                if args.mode == "nvfp4"
                else None
            ),
            "native_fp4_gemm": False,
            "matmul_compute_dtype": "torch.bfloat16",
            "compile_qdq_requested": args.compile_qdq,
            "compile_qdq_status": (
                nvfp4_nosr_qdq_compile_status()
                if args.mode == "nvfp4"
                else "not applicable"
            ),
            "cache_quantized_weight": args.cache_quantized_weight,
        },
        "timing_ms": timing_summary,
        "peak_memory_bytes": {
            "allocated": peak_allocated,
            "reserved": peak_reserved,
        },
        "loss": summarize([float(record["loss"]) for record in records]),
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote result: {args.output}")


if __name__ == "__main__":
    main()
