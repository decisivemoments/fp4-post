#!/usr/bin/env python3
"""Summarize paired complete-GRPO BF16/native-NVFP4 benchmark results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    for path in sorted(args.results_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        result["_path"] = str(path)
        results.append(result)
    if not results:
        raise RuntimeError(f"No result JSON files in {args.results_dir}")

    rows = []
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for result in results:
        model = Path(result["model_path"]).name
        batch = result["batch_semantics"]["per_device_completion_sequences"]
        generations = result["batch_semantics"]["num_generations_per_prompt"]
        completion = result["max_completion_length"]
        repeat = result["repeat"]
        mode = result["mode"]
        step_s = result["timing_s"]["end_to_end_s"]["median"]
        completion_tps = result["throughput"][
            "completion_tokens_per_s"
        ]["median"]
        peak_gib = (
            result["peak_memory_bytes"]["allocated"] / (1024**3)
        )
        timing = result["timing_s"]

        def phase_median(name: str) -> float:
            return timing.get(name, {}).get("median", 0.0)

        row = {
            "model": model,
            "batch": batch,
            "generations": generations,
            "max_completion": completion,
            "repeat": repeat,
            "mode": mode,
            "step_s": step_s,
            "completion_tokens_per_s": completion_tps,
            "peak_allocated_gib": peak_gib,
            "rollout_total_s": phase_median("rollout_total_s"),
            "rollout_prefill_model_s": phase_median(
                "rollout_prefill_model_s"
            ),
            "rollout_decode_model_s": phase_median(
                "rollout_decode_model_s"
            ),
            "reward_s": phase_median("reward_s"),
            "score_postprocess_s": phase_median(
                "rollout_score_and_postprocess_s"
            ),
            "train_forward_backward_s": phase_median(
                "train_forward_backward_s"
            ),
            "train_forward_loss_s": phase_median(
                "train_forward_loss_s"
            ),
            "train_backward_s": phase_median("train_backward_s"),
            "optimizer_s": phase_median("optimizer_s"),
            "result_path": result["_path"],
        }
        rows.append(row)
        grouped[(model, batch, generations, completion, repeat)][mode] = row

    paired = []
    for key, modes in sorted(grouped.items()):
        if {"bf16", "native_full_nvfp4"} - modes.keys():
            continue
        bf16 = modes["bf16"]
        native = modes["native_full_nvfp4"]
        paired.append(
            {
                "model": key[0],
                "batch": key[1],
                "generations": key[2],
                "max_completion": key[3],
                "repeat": key[4],
                "bf16_step_s": bf16["step_s"],
                "nvfp4_step_s": native["step_s"],
                "step_speedup_bf16_over_nvfp4": (
                    bf16["step_s"] / native["step_s"]
                ),
                "bf16_completion_tokens_per_s": bf16[
                    "completion_tokens_per_s"
                ],
                "nvfp4_completion_tokens_per_s": native[
                    "completion_tokens_per_s"
                ],
                "throughput_speedup_nvfp4_over_bf16": (
                    native["completion_tokens_per_s"]
                    / bf16["completion_tokens_per_s"]
                ),
                "bf16_peak_allocated_gib": bf16["peak_allocated_gib"],
                "nvfp4_peak_allocated_gib": native["peak_allocated_gib"],
            }
        )

    aggregate_groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in paired:
        aggregate_groups[
            (
                row["model"],
                row["batch"],
                row["generations"],
                row["max_completion"],
            )
        ].append(row)

    aggregates = []
    for key, repeat_rows in sorted(aggregate_groups.items()):
        source_rows = {
            mode: [
                row
                for row in rows
                if (
                    row["model"],
                    row["batch"],
                    row["generations"],
                    row["max_completion"],
                )
                == key
                and row["mode"] == mode
                and row["repeat"]
                in {item["repeat"] for item in repeat_rows}
            ]
            for mode in ("bf16", "native_full_nvfp4")
        }

        def mode_median(mode: str, field: str) -> float:
            return median(row[field] for row in source_rows[mode])

        bf16_step = mode_median("bf16", "step_s")
        nvfp4_step = mode_median("native_full_nvfp4", "step_s")
        bf16_tps = mode_median("bf16", "completion_tokens_per_s")
        nvfp4_tps = mode_median(
            "native_full_nvfp4", "completion_tokens_per_s"
        )
        aggregate = {
            "model": key[0],
            "batch": key[1],
            "generations": key[2],
            "max_completion": key[3],
            "repeat_count": len(repeat_rows),
            "bf16_step_s": bf16_step,
            "nvfp4_step_s": nvfp4_step,
            "step_speedup_bf16_over_nvfp4": bf16_step / nvfp4_step,
            "bf16_completion_tokens_per_s": bf16_tps,
            "nvfp4_completion_tokens_per_s": nvfp4_tps,
            "throughput_speedup_nvfp4_over_bf16": nvfp4_tps / bf16_tps,
            "bf16_peak_allocated_gib": mode_median(
                "bf16", "peak_allocated_gib"
            ),
            "nvfp4_peak_allocated_gib": mode_median(
                "native_full_nvfp4", "peak_allocated_gib"
            ),
        }
        for field in (
            "rollout_total_s",
            "rollout_prefill_model_s",
            "rollout_decode_model_s",
            "reward_s",
            "score_postprocess_s",
            "train_forward_backward_s",
            "train_forward_loss_s",
            "train_backward_s",
            "optimizer_s",
        ):
            aggregate[f"bf16_{field}"] = mode_median("bf16", field)
            aggregate[f"nvfp4_{field}"] = mode_median(
                "native_full_nvfp4", field
            )
        aggregates.append(aggregate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "grpo_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    paired_path = args.output_dir / "grpo_paired_summary.json"
    with paired_path.open("w", encoding="utf-8") as handle:
        json.dump(paired, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    aggregate_path = args.output_dir / "grpo_aggregate_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregates, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    markdown = [
        "# Complete GRPO step benchmark",
        "",
        "每项数据是各独立 repeat 的 step median 再取中位数；"
        "`speedup > 1` 表示 Native Full-NVFP4 更快。",
        "",
        "| Model | Batch | G | Completion | Repeats | BF16 step (s) | NVFP4 step (s) | Step speedup | BF16 tok/s | NVFP4 tok/s | Throughput speedup | Peak GiB BF16/NVFP4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        markdown.append(
            f"| {row['model']} | {row['batch']} | "
            f"{row['generations']} | {row['max_completion']} | "
            f"{row['repeat_count']} | "
            f"{row['bf16_step_s']:.3f} | {row['nvfp4_step_s']:.3f} | "
            f"{row['step_speedup_bf16_over_nvfp4']:.3f}x | "
            f"{row['bf16_completion_tokens_per_s']:.1f} | "
            f"{row['nvfp4_completion_tokens_per_s']:.1f} | "
            f"{row['throughput_speedup_nvfp4_over_bf16']:.3f}x | "
            f"{row['bf16_peak_allocated_gib']:.2f}/"
            f"{row['nvfp4_peak_allocated_gib']:.2f} |"
        )
    markdown.extend(
        [
            "",
            "## Phase medians",
            "",
            "| Model | Batch | Mode | Rollout | Prefill | Decode | Train fwd+bwd | Train fwd | Train bwd | Optimizer |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregates:
        for mode, label in (
            ("bf16", "BF16"),
            ("nvfp4", "Native Full-NVFP4"),
        ):
            markdown.append(
                f"| {row['model']} | {row['batch']} | {label} | "
                f"{row[f'{mode}_rollout_total_s']:.3f} | "
                f"{row[f'{mode}_rollout_prefill_model_s']:.3f} | "
                f"{row[f'{mode}_rollout_decode_model_s']:.3f} | "
                f"{row[f'{mode}_train_forward_backward_s']:.3f} | "
                f"{row[f'{mode}_train_forward_loss_s']:.3f} | "
                f"{row[f'{mode}_train_backward_s']:.3f} | "
                f"{row[f'{mode}_optimizer_s']:.3f} |"
            )
    markdown_path = args.output_dir / "grpo_summary.md"
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(
        f"Wrote {csv_path}, {paired_path}, {aggregate_path}, "
        f"and {markdown_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
