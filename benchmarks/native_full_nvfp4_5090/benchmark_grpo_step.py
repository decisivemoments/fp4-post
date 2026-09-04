#!/usr/bin/env python3
"""Benchmark complete GRPO optimizer steps in BF16 and native Full-NVFP4."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import accelerate
import datasets
import torch
import transformer_engine
import transformers
import trl
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import GRPOConfig, GRPOTrainer


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Metis.Metis.native_nvfp4 import (  # noqa: E402
    NativeFullNVFP4Linear,
    register_native_nvfp4_optimizer_hook,
    replace_linear_with_native_full_nvfp4,
    require_native_nvfp4,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["bf16", "native_full_nvfp4"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measured-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--expected-gpu-uuid", default="")
    parser.add_argument(
        "--stochastic-gradient-quantization",
        action="store_true",
    )
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
        return {}
    return {
        "min": min(values),
        "p10": percentile(values, 0.10),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "p90": percentile(values, 0.90),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def custom_accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **_: Any,
) -> list[float]:
    """Repository-equivalent DeepMath accuracy reward used by GRPO."""
    rewards: list[float] = []
    for completion, gold in zip(completions, solution, strict=True):
        content = completion[0].get("content", "")
        if not isinstance(content, str) or not isinstance(gold, str):
            rewards.append(0.0)
            continue
        try:
            gold_parsed = parse(gold.strip())
            if gold_parsed:
                answer_parsed = parse(
                    content.strip(),
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                units=True
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                rewards.append(float(verify(gold_parsed, answer_parsed)))
                continue
            words = content.strip().split()
            last_word = (
                words[-1].rstrip(".,!?;:").lower() if words else ""
            )
            rewards.append(
                float(
                    gold.strip() in {"True", "False", "Yes", "No"}
                    and last_word == gold.strip().lower()
                )
            )
        except Exception:
            rewards.append(0.0)
    return rewards


class GRPOStepTimer:
    """Synchronized phase timer plus rollout prefill/decode CUDA events."""

    def __init__(self, warmup_steps: int) -> None:
        self.warmup_steps = warmup_steps
        self.current: dict[str, Any] = {}
        self.records: list[dict[str, Any]] = []
        self._step_wall_start = 0.0
        self._optimizer_start = 0.0
        self._in_rollout = False
        self._rollout_forward_events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []
        self._pending_forward: list[
            tuple[str, torch.cuda.Event]
        ] = []

    @staticmethod
    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def begin_step(self, step: int) -> None:
        self.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.current = {
            "global_step": step,
            "measured": step > self.warmup_steps,
            "phases_s": defaultdict(float),
        }
        self._step_wall_start = time.perf_counter()

    @contextmanager
    def phase(self, name: str):
        self.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.synchronize()
            self.current["phases_s"][name] += time.perf_counter() - start

    def begin_optimizer(self) -> None:
        self.synchronize()
        self._optimizer_start = time.perf_counter()

    def end_optimizer(self) -> None:
        self.synchronize()
        if self._optimizer_start:
            self.current["phases_s"]["optimizer_s"] += (
                time.perf_counter() - self._optimizer_start
            )
            self._optimizer_start = 0.0

    def begin_rollout(self) -> None:
        self._in_rollout = True
        self._rollout_forward_events.clear()
        self._pending_forward.clear()

    def end_rollout(self) -> None:
        self.synchronize()
        prefill_ms = 0.0
        decode_ms = 0.0
        prefill_calls = 0
        decode_calls = 0
        for kind, start, end in self._rollout_forward_events:
            elapsed = start.elapsed_time(end)
            if kind == "prefill":
                prefill_ms += elapsed
                prefill_calls += 1
            else:
                decode_ms += elapsed
                decode_calls += 1
        phases = self.current["phases_s"]
        phases["rollout_prefill_model_s"] += prefill_ms / 1000.0
        phases["rollout_decode_model_s"] += decode_ms / 1000.0
        self.current["rollout_prefill_calls"] = prefill_calls
        self.current["rollout_decode_calls"] = decode_calls
        self._in_rollout = False
        self._pending_forward.clear()

    def model_forward_pre_hook(
        self,
        _module: nn.Module,
        args: tuple,
        kwargs: dict,
    ) -> None:
        if not self._in_rollout:
            return
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args and isinstance(args[0], torch.Tensor):
            input_ids = args[0]
        sequence_length = (
            int(input_ids.shape[-1])
            if isinstance(input_ids, torch.Tensor)
            else 2
        )
        kind = "prefill" if sequence_length > 1 else "decode"
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._pending_forward.append((kind, event))

    def model_forward_hook(
        self,
        _module: nn.Module,
        _args: tuple,
        _kwargs: dict,
        _output: Any,
    ) -> None:
        if not self._in_rollout or not self._pending_forward:
            return
        kind, start = self._pending_forward.pop()
        end = torch.cuda.Event(enable_timing=True)
        end.record()
        self._rollout_forward_events.append((kind, start, end))

    def set_batch_metrics(self, inputs: dict[str, Any], num_generations: int) -> None:
        prompt_mask = inputs.get("prompt_mask")
        completion_mask = inputs.get("completion_mask")
        if not isinstance(prompt_mask, torch.Tensor):
            return
        sequences = int(prompt_mask.shape[0])
        self.current.update(
            {
                "completion_sequences": sequences,
                "unique_prompts": sequences // num_generations,
                "prompt_tokens": int(prompt_mask.sum().item()),
                "completion_tokens": int(completion_mask.sum().item()),
            }
        )

    def end_step(self, step: int) -> None:
        self.synchronize()
        end_to_end_s = time.perf_counter() - self._step_wall_start
        phases = dict(self.current["phases_s"])
        rollout_total = phases.get("rollout_total_s", 0.0)
        rollout_known = (
            phases.get("rollout_generate_s", 0.0)
            + phases.get("reward_s", 0.0)
        )
        training_total = phases.get("training_step_s", 0.0)
        train_forward = phases.get("train_forward_loss_s", 0.0)
        phases["rollout_score_and_postprocess_s"] = max(
            0.0, rollout_total - rollout_known
        )
        phases["train_forward_backward_s"] = max(
            0.0, training_total - rollout_total
        )
        phases["train_backward_s"] = max(
            0.0,
            training_total - rollout_total - train_forward,
        )
        self.current["global_step"] = step
        self.current["end_to_end_s"] = end_to_end_s
        self.current["phases_s"] = phases
        self.current["peak_memory_bytes"] = {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }
        completion_tokens = self.current.get("completion_tokens", 0)
        prompt_tokens = self.current.get("prompt_tokens", 0)
        sequences = self.current.get("completion_sequences", 0)
        self.current["throughput"] = {
            "completion_tokens_per_s": (
                completion_tokens / end_to_end_s if end_to_end_s else 0.0
            ),
            "total_tokens_per_s": (
                (prompt_tokens + completion_tokens) / end_to_end_s
                if end_to_end_s
                else 0.0
            ),
            "completion_sequences_per_s": (
                sequences / end_to_end_s if end_to_end_s else 0.0
            ),
        }
        self.records.append(self.current)
        self.current = {}


class TimingCallback(TrainerCallback):
    def __init__(self, timer: GRPOStepTimer) -> None:
        self.timer = timer

    def on_step_begin(self, args, state, control, **kwargs):
        self.timer.begin_step(state.global_step + 1)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.timer.begin_optimizer()

    def on_optimizer_step(self, args, state, control, **kwargs):
        self.timer.end_optimizer()

    def on_step_end(self, args, state, control, **kwargs):
        self.timer.end_step(state.global_step)


class TimedGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        *args,
        benchmark_timer: GRPOStepTimer,
        native_cache_model: nn.Module | None = None,
        **kwargs,
    ) -> None:
        self.benchmark_timer = benchmark_timer
        self.native_cache_model = native_cache_model
        self._native_cache_hook = None
        callbacks = list(kwargs.pop("callbacks", []) or [])
        callbacks.append(TimingCallback(benchmark_timer))
        super().__init__(*args, callbacks=callbacks, **kwargs)

    def _generate_and_score_completions(self, inputs):
        with self.benchmark_timer.phase("rollout_total_s"):
            result = super()._generate_and_score_completions(inputs)
        self.benchmark_timer.set_batch_metrics(
            result,
            num_generations=self.num_generations,
        )
        return result

    def _calculate_rewards(
        self,
        inputs,
        prompts,
        completions,
        completion_ids_list,
    ):
        with self.benchmark_timer.phase("reward_s"):
            return super()._calculate_rewards(
                inputs,
                prompts,
                completions,
                completion_ids_list,
            )

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        with self.benchmark_timer.phase("train_forward_loss_s"):
            return super().compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

    def training_step(self, model, inputs, num_items_in_batch):
        with self.benchmark_timer.phase("training_step_s"):
            return super().training_step(model, inputs, num_items_in_batch)

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        if (
            self.native_cache_model is not None
            and self._native_cache_hook is None
        ):
            self._native_cache_hook = register_native_nvfp4_optimizer_hook(
                optimizer,
                self.native_cache_model,
            )
        return optimizer


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size <= 0 or args.num_generations <= 1:
        raise ValueError("Batch size must be positive and generations > 1")
    if args.batch_size % args.num_generations:
        raise ValueError(
            "per-device batch size must be divisible by num_generations"
        )
    if args.mode == "native_full_nvfp4" and args.batch_size % 16:
        raise ValueError(
            "Native NVFP4 requires per-device batch size divisible by 16 "
            "so both decode and training rows satisfy the SM120 block shape"
        )
    if args.max_completion_length <= 0:
        raise ValueError("--max-completion-length must be positive")
    if args.warmup_steps < 0 or args.measured_steps <= 0:
        raise ValueError("Invalid warmup/measured step counts")


def wrap_generation(
    model: nn.Module,
    timer: GRPOStepTimer,
    *,
    gradient_checkpointing: bool,
) -> None:
    original_generate = model.generate

    def timed_generate(*args, **kwargs):
        kwargs.setdefault("use_cache", True)
        was_checkpointing = bool(
            getattr(model, "is_gradient_checkpointing", False)
        )
        if was_checkpointing:
            model.gradient_checkpointing_disable()
        timer.begin_rollout()
        try:
            with timer.phase("rollout_generate_s"):
                return original_generate(*args, **kwargs)
        finally:
            timer.end_rollout()
            if was_checkpointing and gradient_checkpointing:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

    model.generate = timed_generate


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    actual_uuid = str(torch.cuda.get_device_properties(device).uuid)
    if args.expected_gpu_uuid:
        expected = args.expected_gpu_uuid.removeprefix("GPU-")
        actual = actual_uuid.removeprefix("GPU-")
        if expected != actual:
            raise RuntimeError(
                f"GPU UUID mismatch: expected {expected}, got {actual}"
            )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    dataset = load_dataset(
        "json",
        data_files={"train": str(args.dataset_jsonl)},
        split="train",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    setup_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model.to(device)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - setup_start

    replaced_modules: list[str] = []
    decomposition_seconds = 0.0
    if args.mode == "native_full_nvfp4":
        require_native_nvfp4()
        decomposition_start = time.perf_counter()
        replaced_modules = replace_linear_with_native_full_nvfp4(
            model,
            rank=args.rank,
            stochastic_rounding=args.stochastic_gradient_quantization,
        )
        torch.cuda.synchronize()
        decomposition_seconds = time.perf_counter() - decomposition_start

    model.config.use_cache = not args.gradient_checkpointing
    timer = GRPOStepTimer(warmup_steps=args.warmup_steps)
    pre_hook = model.register_forward_pre_hook(
        timer.model_forward_pre_hook,
        with_kwargs=True,
    )
    post_hook = model.register_forward_hook(
        timer.model_forward_hook,
        with_kwargs=True,
    )
    wrap_generation(
        model,
        timer,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    total_steps = args.warmup_steps + args.measured_steps
    training_args = GRPOConfig(
        output_dir=str(args.run_dir),
        overwrite_output_dir=True,
        do_train=True,
        max_steps=total_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        generation_batch_size=args.batch_size,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        generation_kwargs={
            "min_new_tokens": args.max_completion_length,
        },
        learning_rate=args.learning_rate,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        use_vllm=False,
        beta=0.0,
        loss_type="grpo",
        temperature=1.0,
        shuffle_dataset=False,
        dataloader_num_workers=0,
        logging_strategy="no",
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = TimedGRPOTrainer(
        model=model,
        reward_funcs=custom_accuracy_reward,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        benchmark_timer=timer,
        native_cache_model=(
            model if args.mode == "native_full_nvfp4" else None
        ),
    )

    print(
        f"Starting complete GRPO benchmark: mode={args.mode}, "
        f"batch={args.batch_size}, unique_prompts="
        f"{args.batch_size // args.num_generations}, "
        f"generations={args.num_generations}, steps={total_steps}",
        flush=True,
    )
    train_result = trainer.train()
    timer.synchronize()
    pre_hook.remove()
    post_hook.remove()

    measured = [record for record in timer.records if record["measured"]]
    if len(measured) != args.measured_steps:
        raise RuntimeError(
            f"Expected {args.measured_steps} measured records, "
            f"got {len(measured)}"
        )

    phase_names = sorted(
        {
            name
            for record in measured
            for name in record["phases_s"]
        }
    )
    timing_summary = {
        name: summarize(
            [float(record["phases_s"].get(name, 0.0)) for record in measured]
        )
        for name in phase_names
    }
    timing_summary["end_to_end_s"] = summarize(
        [float(record["end_to_end_s"]) for record in measured]
    )
    throughput_summary = {
        name: summarize(
            [float(record["throughput"][name]) for record in measured]
        )
        for name in measured[0]["throughput"]
    }

    result = {
        "schema_version": 1,
        "benchmark": "complete_grpo_optimizer_step",
        "mode": args.mode,
        "repeat": args.repeat,
        "model_path": str(args.model_path.resolve()),
        "dataset_jsonl": str(args.dataset_jsonl.resolve()),
        "batch_semantics": {
            "per_device_completion_sequences": args.batch_size,
            "num_generations_per_prompt": args.num_generations,
            "unique_prompts_per_step": (
                args.batch_size // args.num_generations
            ),
            "gradient_accumulation_steps": 1,
        },
        "max_completion_length": args.max_completion_length,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "gradient_checkpointing": args.gradient_checkpointing,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "optimizer": "adamw_torch_fused",
        "grpo": {
            "trainer": "trl.GRPOTrainer",
            "loss_type": "grpo",
            "beta": 0.0,
            "steps_per_generation": 1,
            "use_vllm": False,
            "fixed_completion_workload": True,
            "reward": "repository_equivalent_custom_accuracy_reward",
        },
        "native_full_nvfp4": {
            "enabled": args.mode == "native_full_nvfp4",
            "backend": (
                "transformer_engine.general_gemm"
                if args.mode == "native_full_nvfp4"
                else None
            ),
            "weight_svd_rank": (
                args.rank if args.mode == "native_full_nvfp4" else None
            ),
            "activation": (
                "mean_residual_native_nvfp4"
                if args.mode == "native_full_nvfp4"
                else None
            ),
            "gradient": (
                "mean_residual_native_nvfp4"
                if args.mode == "native_full_nvfp4"
                else None
            ),
            "residual_weight": (
                "packed_native_nvfp4"
                if args.mode == "native_full_nvfp4"
                else None
            ),
            "right_singular_vectors_v": (
                "packed_native_nvfp4"
                if args.mode == "native_full_nvfp4"
                else None
            ),
            "left_singular_vectors_u": (
                "bf16" if args.mode == "native_full_nvfp4" else None
            ),
            "singular_values": (
                "bf16" if args.mode == "native_full_nvfp4" else None
            ),
            # BitLinear's merged rollout weight is a fake-QDQ shortcut.
            # Repacking that BF16 sum would quantize the U/S contribution
            # again and therefore change the native Full-NVFP4 method.
            "rollout_weight_merge": False,
            "replaced_linear_count": len(replaced_modules),
            "stochastic_gradient_quantization": (
                args.stochastic_gradient_quantization
            ),
        },
        "setup_seconds": {
            "model_load": model_load_seconds,
            "weight_svd_and_replacement": decomposition_seconds,
        },
        "timing_s": timing_summary,
        "throughput": throughput_summary,
        "peak_memory_bytes": {
            "allocated": max(
                record["peak_memory_bytes"]["allocated"]
                for record in measured
            ),
            "reserved": max(
                record["peak_memory_bytes"]["reserved"]
                for record in measured
            ),
        },
        "records": timer.records,
        "trainer_metrics": train_result.metrics,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "transformer_engine": transformer_engine.__version__,
            "trl": trl.__version__,
            "accelerate": accelerate.__version__,
            "datasets": datasets.__version__,
            "gpu_name": torch.cuda.get_device_name(device),
            "gpu_uuid": actual_uuid,
            "cuda_visible_devices": os.environ.get(
                "CUDA_VISIBLE_DEVICES", ""
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(
        f"Wrote {args.output}; measured median step="
        f"{timing_summary['end_to_end_s']['median']:.3f} s, "
        f"completion throughput="
        f"{throughput_summary['completion_tokens_per_s']['median']:.1f} tok/s",
        flush=True,
    )


if __name__ == "__main__":
    main()
