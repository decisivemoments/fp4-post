#!/usr/bin/env python3
"""Benchmark BF16 against the native full-NVFP4 training path."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import transformer_engine
import transformers
from torch import nn
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Metis.Metis.native_nvfp4 import (  # noqa: E402
    NativeFullNVFP4Linear,
    register_native_nvfp4_optimizer_hook,
    replace_linear_with_native_full_nvfp4,
    require_native_nvfp4,
    set_native_nvfp4_profiling,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-cache", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["bf16", "native_full_nvfp4"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--no-optimizer",
        action="store_true",
        help=(
            "Benchmark forward + loss + backward only. Do not construct an "
            "optimizer or allocate optimizer state; gradient clearing happens "
            "after the timed region."
        ),
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help=(
            "Capture fixed-shape forward/loss and autograd backward with "
            "torch.cuda.make_graphed_callables. Currently supported only "
            "together with --no-optimizer."
        ),
    )
    parser.add_argument(
        "--cuda-graph-capture-warmup-steps",
        type=int,
        default=3,
    )
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--stochastic-gradient-quantization",
        action="store_true",
        help=(
            "Enable stochastic gradient rounding. The installed TE build "
            "must contain architecture-specific SM120a conversion kernels."
        ),
    )
    parser.add_argument(
        "--validate-first-step",
        action="store_true",
        help="Run an untimed finite-gradient and parameter-update check.",
    )
    parser.add_argument("--cuda-profiler-range", action="store_true")
    parser.add_argument("--profile-native-ranges", action="store_true")
    return parser.parse_args()


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
    if not values:
        raise ValueError("Cannot summarize an empty list")
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
    indices = [
        (start + offset) % sample_count for offset in range(batch_size)
    ]
    index_tensor = torch.tensor(indices, dtype=torch.long)
    return {
        key: value.index_select(0, index_tensor)[:, :seq_length].to(
            device,
            non_blocking=False,
        )
        for key, value in cache.items()
    }


def copy_batch_to_static(
    cache: dict[str, torch.Tensor],
    static_batch: dict[str, torch.Tensor],
    *,
    batch_size: int,
    seq_length: int,
    step: int,
    causal_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Update fixed-address CUDA Graph inputs outside the timed region."""
    cpu_batch = make_batch(
        cache,
        batch_size=batch_size,
        seq_length=seq_length,
        step=step,
        device=torch.device("cpu"),
    )
    for key in ("input_ids", "labels"):
        static_batch[key].copy_(cpu_batch[key])
    if causal_mask is None:
        static_batch["attention_mask"].copy_(cpu_batch["attention_mask"])
    else:
        padding_mask = cpu_batch["attention_mask"].to(
            device=static_batch["attention_mask"].device,
            dtype=torch.bool,
        )
        static_batch["attention_mask"].copy_(
            padding_mask[:, None, None, :]
        )
        static_batch["attention_mask"].logical_and_(causal_mask)
    return static_batch


class CausalLMLossWrapper(nn.Module):
    """Tensor-only positional interface for make_graphed_callables."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss


def run_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    batch: dict[str, torch.Tensor],
    *,
    measure: bool,
    validate_gradients: bool = False,
    loss_callable: nn.Module | None = None,
) -> dict[str, float | int | bool]:
    if measure:
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        torch.cuda.synchronize()
        allocated_before_forward = torch.cuda.memory_allocated()
        wall_start = time.perf_counter()
        events[0].record()

    if loss_callable is None:
        outputs = model(**batch)
        loss = outputs.loss
    else:
        loss = loss_callable(
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
    if measure:
        events[1].record()
        allocated_after_forward = torch.cuda.memory_allocated()

    loss.backward()
    if measure:
        events[2].record()
        allocated_after_backward = torch.cuda.memory_allocated()

    finite_gradient_tensors = 0
    gradient_tensors = 0
    if validate_gradients:
        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            gradient_tensors += 1
            finite_gradient_tensors += int(
                bool(torch.isfinite(parameter.grad).all().item())
            )

    if optimizer is not None:
        optimizer.step()
        if measure:
            events[3].record()
    elif measure:
        events[3].record()

    if measure:
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - wall_start) * 1000
    else:
        torch.cuda.synchronize()
        wall_ms = 0.0

    loss_value = float(loss.detach().float().item())
    if not math.isfinite(loss_value):
        raise FloatingPointError(f"Non-finite loss: {loss_value}")
    if validate_gradients and finite_gradient_tensors != gradient_tensors:
        raise FloatingPointError(
            f"Only {finite_gradient_tensors}/{gradient_tensors} gradient "
            "tensors are finite"
        )

    # Keep gradient clearing out of the forward/backward timing window. With
    # optimizer=None this is the only post-backward mutation, and it does not
    # allocate optimizer state.
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)

    result: dict[str, float | int | bool] = {"loss": loss_value}
    if validate_gradients:
        result.update(
            {
                "gradient_tensors": gradient_tensors,
                "finite_gradient_tensors": finite_gradient_tensors,
                "all_gradients_finite": True,
            }
        )
    if measure:
        result.update(
            {
                "wall_ms": wall_ms,
                "gpu_total_ms": (
                    events[0].elapsed_time(events[3])
                    if optimizer is not None
                    else events[0].elapsed_time(events[2])
                ),
                "forward_ms": events[0].elapsed_time(events[1]),
                "backward_ms": events[1].elapsed_time(events[2]),
                "forward_backward_ms": events[0].elapsed_time(events[2]),
                "optimizer_ms": (
                    events[2].elapsed_time(events[3])
                    if optimizer is not None
                    else 0.0
                ),
                "allocated_before_forward_bytes": allocated_before_forward,
                "allocated_after_forward_bytes": allocated_after_forward,
                "allocated_after_backward_bytes": allocated_after_backward,
            }
        )
    return result


def find_probe_parameter(model: nn.Module) -> torch.Tensor:
    for module in model.modules():
        if isinstance(module, NativeFullNVFP4Linear):
            return module.residual_weight
    for parameter in model.parameters():
        if parameter.requires_grad:
            return parameter
    raise RuntimeError("Model has no trainable parameter")


def read_gpu_state() -> dict[str, Any]:
    """Capture physical-GPU state outside the timed region."""
    fields = [
        "index",
        "uuid",
        "temperature.gpu",
        "pstate",
        "clocks.sm",
        "power.draw",
        "memory.used",
        "utilization.gpu",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
    rows = []
    for line in completed.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) == len(fields):
            rows.append(dict(zip(fields, values)))
    for row in rows:
        if row["index"] == visible or row["uuid"] == visible:
            return row
    return {
        "error": f"Could not resolve CUDA_VISIBLE_DEVICES={visible}",
        "all_gpu_rows": rows,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if (
        args.batch_size <= 0
        or args.seq_length <= 1
        or args.steps <= 0
        or args.warmup_steps < 0
        or args.repeat <= 0
        or args.cuda_graph_capture_warmup_steps <= 0
    ):
        raise ValueError("Invalid batch size, sequence length, or step count")
    if (args.batch_size * args.seq_length) % 16 != 0:
        raise ValueError("Flattened token count must be divisible by 16")
    if args.cuda_graph and not args.no_optimizer:
        raise ValueError("--cuda-graph currently requires --no-optimizer")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    set_native_nvfp4_profiling(args.profile_native_ranges)

    cache = torch.load(args.data_cache, map_location="cpu", weights_only=True)
    expected_keys = {"input_ids", "attention_mask", "labels"}
    if set(cache) != expected_keys:
        raise ValueError(f"Unexpected cache keys: {sorted(cache)}")
    if cache["input_ids"].shape[1] < args.seq_length:
        raise ValueError(
            f"Cache seq length {cache['input_ids'].shape[1]} "
            f"< requested {args.seq_length}"
        )

    print(f"Loading model from {args.model_path}", flush=True)
    setup_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.to(device)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - setup_start

    replaced_modules: list[str] = []
    decomposition_seconds = 0.0
    if args.mode == "native_full_nvfp4":
        require_native_nvfp4()
        print(
            f"Replacing Qwen projections with rank-{args.rank} native "
            "full-NVFP4 layers",
            flush=True,
        )
        torch.cuda.synchronize()
        decomposition_start = time.perf_counter()
        replaced_modules = replace_linear_with_native_full_nvfp4(
            model,
            rank=args.rank,
            stochastic_rounding=args.stochastic_gradient_quantization,
        )
        torch.cuda.synchronize()
        decomposition_seconds = time.perf_counter() - decomposition_start
        print(
            f"Replaced {len(replaced_modules)} projections in "
            f"{decomposition_seconds:.2f} s",
            flush=True,
        )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.train()

    optimizer: torch.optim.Optimizer | None = None
    cache_hook = None
    if not args.no_optimizer:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=0.01,
            fused=True,
        )
        if args.mode == "native_full_nvfp4":
            cache_hook = register_native_nvfp4_optimizer_hook(
                optimizer,
                model,
            )
        optimizer.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)

    graph_loss_callable: nn.Module | None = None
    static_batch: dict[str, torch.Tensor] | None = None
    graph_causal_mask: torch.Tensor | None = None
    cuda_graph_capture_seconds = 0.0
    if args.cuda_graph:
        static_batch = make_batch(
            cache,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            step=0,
            device=device,
        )
        # Transformers' dynamic SDPA mask optimization reads
        # ``padding_mask.all()`` back to the host, which is illegal during
        # CUDA Graph capture. A prebuilt 4D mask is an officially supported
        # early-exit path and preserves the same causal + key-padding
        # semantics for this fixed-shape training workload.
        graph_causal_mask = torch.ones(
            (1, 1, args.seq_length, args.seq_length),
            device=device,
            dtype=torch.bool,
        ).tril_()
        padding_mask = static_batch["attention_mask"].to(torch.bool)
        static_batch["attention_mask"] = (
            graph_causal_mask
            & padding_mask[:, None, None, :]
        ).contiguous()
        loss_wrapper = CausalLMLossWrapper(model)
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        capture_start = time.perf_counter()
        graph_loss_callable = torch.cuda.make_graphed_callables(
            loss_wrapper,
            (
                static_batch["input_ids"],
                static_batch["attention_mask"],
                static_batch["labels"],
            ),
            num_warmup_iters=args.cuda_graph_capture_warmup_steps,
        )
        torch.cuda.synchronize()
        cuda_graph_capture_seconds = time.perf_counter() - capture_start
        model.zero_grad(set_to_none=True)
        print(
            "Captured CUDA Graph forward/backward in "
            f"{cuda_graph_capture_seconds:.2f} s",
            flush=True,
        )

    validation: dict[str, Any] | None = None
    next_step = 0
    if args.validate_first_step:
        probe_parameter = None
        probe_before = None
        if optimizer is not None:
            probe_parameter = find_probe_parameter(model)
            probe_before = probe_parameter.detach().clone()
        if static_batch is None:
            validation_batch = make_batch(
                cache,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                step=next_step,
                device=device,
            )
        else:
            validation_batch = copy_batch_to_static(
                cache,
                static_batch,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                step=next_step,
                causal_mask=graph_causal_mask,
            )
        validation = run_step(
            model,
            optimizer,
            validation_batch,
            measure=False,
            validate_gradients=True,
            loss_callable=graph_loss_callable,
        )
        if optimizer is not None:
            assert probe_parameter is not None
            assert probe_before is not None
            validation["probe_parameter_changed"] = bool(
                torch.any(
                    probe_parameter.detach() != probe_before
                ).item()
            )
            validation["probe_parameter_name"] = (
                "first native residual_weight"
                if args.mode == "native_full_nvfp4"
                else "first trainable parameter"
            )
            del probe_before
            if not validation["probe_parameter_changed"]:
                raise RuntimeError(
                    "Optimizer step did not change probe parameter"
                )
        else:
            validation["probe_parameter_changed"] = None
            validation["probe_parameter_name"] = None
        print(
            "Validation step passed: finite loss/gradients"
            + (
                " and parameter update"
                if optimizer is not None
                else "; optimizer intentionally absent"
            ),
            flush=True,
        )
        next_step += 1

    print(f"Running {args.warmup_steps} warmup steps", flush=True)
    for warmup_step in range(args.warmup_steps):
        if static_batch is None:
            batch = make_batch(
                cache,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                step=next_step + warmup_step,
                device=device,
            )
        else:
            batch = copy_batch_to_static(
                cache,
                static_batch,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                step=next_step + warmup_step,
                causal_mask=graph_causal_mask,
            )
        record = run_step(
            model,
            optimizer,
            batch,
            measure=False,
            loss_callable=graph_loss_callable,
        )
        print(
            f"warmup {warmup_step + 1}/{args.warmup_steps}: "
            f"loss={record['loss']:.6f}",
            flush=True,
        )
    next_step += args.warmup_steps

    gpu_state_before_measurement = read_gpu_state()
    torch.cuda.reset_peak_memory_stats(device)
    records: list[dict[str, float | int | bool]] = []
    if args.cuda_profiler_range:
        torch.cuda.cudart().cudaProfilerStart()
    for measured_step in range(args.steps):
        if static_batch is None:
            batch = make_batch(
                cache,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                step=next_step + measured_step,
                device=device,
            )
        else:
            batch = copy_batch_to_static(
                cache,
                static_batch,
                batch_size=args.batch_size,
                seq_length=args.seq_length,
                step=next_step + measured_step,
                causal_mask=graph_causal_mask,
            )
        record = run_step(
            model,
            optimizer,
            batch,
            measure=True,
            loss_callable=graph_loss_callable,
        )
        record["step"] = measured_step
        records.append(record)
        print(
            f"{args.mode} step {measured_step + 1}/{args.steps}: "
            f"{record['wall_ms']:.3f} ms loss={record['loss']:.6f}",
            flush=True,
        )
    if args.cuda_profiler_range:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    gpu_state_after_measurement = read_gpu_state()

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    padded_tokens_per_step = args.batch_size * args.seq_length
    nonpadding_tokens = []
    for measured_step in range(args.steps):
        cpu_batch = make_batch(
            cache,
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            step=next_step + measured_step,
            device=torch.device("cpu"),
        )
        nonpadding_tokens.append(
            int(cpu_batch["attention_mask"].sum().item())
        )

    timing_keys = [
        "wall_ms",
        "gpu_total_ms",
        "forward_ms",
        "backward_ms",
        "forward_backward_ms",
        "optimizer_ms",
    ]
    timing_summary = {
        key: summarize([float(record[key]) for record in records])
        for key in timing_keys
    }
    median_wall_seconds = timing_summary["wall_ms"]["median"] / 1000

    result: dict[str, Any] = {
        "schema_version": 3,
        "benchmark_scope": (
            "forward_loss_backward_no_optimizer"
            if args.no_optimizer
            else "forward_loss_backward_optimizer"
        ),
        "mode": args.mode,
        "repeat": args.repeat,
        "model_path": str(args.model_path.resolve()),
        "data_cache": str(args.data_cache.resolve()),
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "attention_implementation": args.attn_implementation,
        "optimizer": (
            None if args.no_optimizer else "torch.optim.AdamW(fused=True)"
        ),
        "optimizer_constructed": not args.no_optimizer,
        "optimizer_state_bytes": 0 if args.no_optimizer else None,
        "learning_rate": None if args.no_optimizer else args.learning_rate,
        "cuda_graph": {
            "enabled": args.cuda_graph,
            "api": (
                "torch.cuda.make_graphed_callables"
                if args.cuda_graph
                else None
            ),
            "capture_warmup_steps": (
                args.cuda_graph_capture_warmup_steps
                if args.cuda_graph
                else 0
            ),
            "capture_seconds": cuda_graph_capture_seconds,
            "static_input_addresses": args.cuda_graph,
            "input_copy_inside_timed_region": False,
            "gradient_clear_inside_timed_region": False,
            "attention_mask": (
                "static_4d_causal_and_padding"
                if args.cuda_graph
                else None
            ),
        },
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
        "native_full_nvfp4": {
            "enabled": args.mode == "native_full_nvfp4",
            "native_fp4_gemm": args.mode == "native_full_nvfp4",
            "backend": (
                "transformer_engine.pytorch.cpp_extensions.general_gemm"
                if args.mode == "native_full_nvfp4"
                else None
            ),
            "weight_svd_rank": (
                args.rank if args.mode == "native_full_nvfp4" else None
            ),
            "activation_mode": (
                "mean-residual" if args.mode == "native_full_nvfp4" else None
            ),
            "gradient_mode": (
                "mean-residual" if args.mode == "native_full_nvfp4" else None
            ),
            "gradient_stochastic_rounding": (
                args.stochastic_gradient_quantization
                if args.mode == "native_full_nvfp4"
                else None
            ),
            "nvfp4_block_size": (
                16 if args.mode == "native_full_nvfp4" else None
            ),
            "random_hadamard_transform": False,
            "two_dimensional_scaling": False,
        },
        "setup_seconds": {
            "model_load": model_load_seconds,
            "weight_svd_and_replacement": decomposition_seconds,
        },
        "validation": validation,
        "profiling": {
            "cuda_profiler_range": args.cuda_profiler_range,
            "native_nvtx_ranges": args.profile_native_ranges,
        },
        "timing_ms": timing_summary,
        "peak_memory_bytes": {
            "allocated": peak_allocated,
            "reserved": peak_reserved,
        },
        "allocation_bytes": {
            "before_forward": summarize(
                [
                    float(record["allocated_before_forward_bytes"])
                    for record in records
                ]
            ),
            "after_forward": summarize(
                [
                    float(record["allocated_after_forward_bytes"])
                    for record in records
                ]
            ),
            "after_backward": summarize(
                [
                    float(record["allocated_after_backward_bytes"])
                    for record in records
                ]
            ),
        },
        "loss": summarize(
            [float(record["loss"]) for record in records]
        ),
        "records": records,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "transformer_engine": transformer_engine.__version__,
            "cuda": torch.version.cuda,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
            "gpu_state_before_measurement": gpu_state_before_measurement,
            "gpu_state_after_measurement": gpu_state_after_measurement,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote result: {args.output}", flush=True)

    if cache_hook is not None:
        cache_hook.remove()


if __name__ == "__main__":
    main()
