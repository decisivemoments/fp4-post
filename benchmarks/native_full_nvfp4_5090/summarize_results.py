#!/usr/bin/env python3
"""Summarize repeated BF16/native full-NVFP4 step-time runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


TIMING_KEYS = [
    "wall_ms",
    "gpu_total_ms",
    "forward_ms",
    "backward_ms",
    "optimizer_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--threshold-percent", type=float, default=5.0)
    return parser.parse_args()


def load_results(paths: list[Path]) -> list[dict[str, Any]]:
    results = []
    for path in paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        result["_path"] = str(path)
        results.append(result)
    return results


def comparison_key(result: dict[str, Any]) -> tuple[Any, ...]:
    return (
        Path(result["model_path"]).name,
        result["batch_size"],
        result["seq_length"],
        result["gradient_checkpointing"],
        result["attention_implementation"],
    )


def median_run_metric(
    runs: list[dict[str, Any]],
    timing_key: str,
    statistic: str = "median",
) -> float:
    values = [
        float(run["timing_ms"][timing_key][statistic]) for run in runs
    ]
    return statistics.median(values)


def decide(changes: list[float], threshold: float) -> str:
    if all(change <= -threshold for change in changes):
        return "stable NVFP4 speedup"
    if all(abs(change) < threshold for change in changes):
        return "within ±5%"
    aggregate = statistics.median(changes)
    if aggregate >= threshold:
        return "NVFP4 slower"
    if aggregate <= -threshold:
        return "NVFP4 faster, not stable in every repeat"
    return "mixed / within threshold"


def main() -> None:
    args = parse_args()
    grouped: dict[
        tuple[Any, ...],
        dict[str, dict[int, dict[str, Any]]],
    ] = defaultdict(lambda: defaultdict(dict))
    for result in load_results(args.results):
        mode = result["mode"]
        repeat = int(result.get("repeat", 1))
        grouped[comparison_key(result)][mode][repeat] = result

    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for key, modes in sorted(grouped.items()):
        if "bf16" not in modes or "native_full_nvfp4" not in modes:
            continue
        common_repeats = sorted(
            set(modes["bf16"]) & set(modes["native_full_nvfp4"])
        )
        if not common_repeats:
            continue

        bf16_runs = [modes["bf16"][repeat] for repeat in common_repeats]
        nvfp4_runs = [
            modes["native_full_nvfp4"][repeat]
            for repeat in common_repeats
        ]
        repeat_rows = []
        changes = []
        speedups = []
        for repeat, bf16, nvfp4 in zip(
            common_repeats,
            bf16_runs,
            nvfp4_runs,
        ):
            bf16_ms = float(bf16["timing_ms"]["wall_ms"]["median"])
            nvfp4_ms = float(nvfp4["timing_ms"]["wall_ms"]["median"])
            speedup = bf16_ms / nvfp4_ms
            change = (nvfp4_ms - bf16_ms) / bf16_ms * 100
            changes.append(change)
            speedups.append(speedup)
            repeat_rows.append(
                {
                    "repeat": repeat,
                    "bf16_median_ms": bf16_ms,
                    "native_full_nvfp4_median_ms": nvfp4_ms,
                    "speedup_bf16_over_nvfp4": speedup,
                    "nvfp4_latency_change_percent": change,
                }
            )

        aggregate_bf16_ms = median_run_metric(bf16_runs, "wall_ms")
        aggregate_nvfp4_ms = median_run_metric(
            nvfp4_runs,
            "wall_ms",
        )
        bf16_p10_ms = median_run_metric(bf16_runs, "wall_ms", "p10")
        bf16_p90_ms = median_run_metric(bf16_runs, "wall_ms", "p90")
        nvfp4_p10_ms = median_run_metric(
            nvfp4_runs,
            "wall_ms",
            "p10",
        )
        nvfp4_p90_ms = median_run_metric(
            nvfp4_runs,
            "wall_ms",
            "p90",
        )
        aggregate_speedup = aggregate_bf16_ms / aggregate_nvfp4_ms
        aggregate_change = (
            (aggregate_nvfp4_ms - aggregate_bf16_ms)
            / aggregate_bf16_ms
            * 100
        )
        phase_metrics = {}
        for timing_key in TIMING_KEYS:
            bf16_value = median_run_metric(bf16_runs, timing_key)
            nvfp4_value = median_run_metric(nvfp4_runs, timing_key)
            phase_metrics[timing_key] = {
                "bf16_median_of_run_medians": bf16_value,
                "native_full_nvfp4_median_of_run_medians": nvfp4_value,
                "speedup_bf16_over_nvfp4": bf16_value / nvfp4_value,
                "nvfp4_latency_change_percent": (
                    (nvfp4_value - bf16_value) / bf16_value * 100
                ),
            }

        bf16_peak = statistics.median(
            [
                float(run["peak_memory_bytes"]["allocated"])
                for run in bf16_runs
            ]
        )
        nvfp4_peak = statistics.median(
            [
                float(run["peak_memory_bytes"]["allocated"])
                for run in nvfp4_runs
            ]
        )
        decision = decide(changes, args.threshold_percent)
        row = {
            "model": key[0],
            "batch_size": key[1],
            "seq_length": key[2],
            "gradient_checkpointing": key[3],
            "repeats": len(common_repeats),
            "bf16_median_ms": aggregate_bf16_ms,
            "bf16_p10_ms": bf16_p10_ms,
            "bf16_p90_ms": bf16_p90_ms,
            "native_full_nvfp4_median_ms": aggregate_nvfp4_ms,
            "native_full_nvfp4_p10_ms": nvfp4_p10_ms,
            "native_full_nvfp4_p90_ms": nvfp4_p90_ms,
            "speedup_bf16_over_nvfp4": aggregate_speedup,
            "nvfp4_latency_change_percent": aggregate_change,
            "bf16_padded_tokens_per_second": (
                key[1] * key[2] / (aggregate_bf16_ms / 1000)
            ),
            "native_full_nvfp4_padded_tokens_per_second": (
                key[1] * key[2] / (aggregate_nvfp4_ms / 1000)
            ),
            "bf16_peak_allocated_gib": bf16_peak / 2**30,
            "native_full_nvfp4_peak_allocated_gib": nvfp4_peak / 2**30,
            "decision": decision,
        }
        rows.append(row)
        details.append(
            {
                **row,
                "repeat_results": repeat_rows,
                "per_repeat_speedups": speedups,
                "per_repeat_latency_changes_percent": changes,
                "phase_timing_ms": phase_metrics,
                "native_kernel_claims": {
                    "all_runs_marked_native": all(
                        run["native_full_nvfp4"]["native_fp4_gemm"]
                        for run in nvfp4_runs
                    ),
                    "backend": nvfp4_runs[0][
                        "native_full_nvfp4"
                    ]["backend"],
                    "weight_svd_rank": nvfp4_runs[0][
                        "native_full_nvfp4"
                    ]["weight_svd_rank"],
                    "activation_mode": nvfp4_runs[0][
                        "native_full_nvfp4"
                    ]["activation_mode"],
                    "gradient_mode": nvfp4_runs[0][
                        "native_full_nvfp4"
                    ]["gradient_mode"],
                    "gradient_stochastic_rounding": nvfp4_runs[0][
                        "native_full_nvfp4"
                    ]["gradient_stochastic_rounding"],
                },
            }
        )

    if not rows:
        raise RuntimeError("No repeated BF16/native full-NVFP4 pairs found")

    header = (
        "| Model | BF16 p10/median/p90 | Native NVFP4 p10/median/p90 | "
        "BF16/NVFP4 | NVFP4 latency | Peak GiB BF16/NVFP4 | Decision |\n"
        "|---|---:|---:|---:|---:|---:|---|\n"
    )
    body = "".join(
        "| {model} | {bf16_p10:.3f}/{bf16:.3f}/{bf16_p90:.3f} ms | "
        "{nvfp4_p10:.3f}/{nvfp4:.3f}/{nvfp4_p90:.3f} ms | "
        "{speedup:.3f}x | "
        "{change:+.2f}% | {bf16_mem:.2f}/{nvfp4_mem:.2f} | "
        "{decision} |\n".format(
            model=row["model"],
            bf16_p10=row["bf16_p10_ms"],
            bf16=row["bf16_median_ms"],
            bf16_p90=row["bf16_p90_ms"],
            nvfp4_p10=row["native_full_nvfp4_p10_ms"],
            nvfp4=row["native_full_nvfp4_median_ms"],
            nvfp4_p90=row["native_full_nvfp4_p90_ms"],
            speedup=row["speedup_bf16_over_nvfp4"],
            change=row["nvfp4_latency_change_percent"],
            bf16_mem=row["bf16_peak_allocated_gib"],
            nvfp4_mem=row["native_full_nvfp4_peak_allocated_gib"],
            decision=row["decision"],
        )
        for row in rows
    )
    notes = (
        "\nValues are medians of the available independent-run statistics. "
        "`BF16/NVFP4 > 1` means NVFP4 is faster. Batch size, sequence "
        "length, and warmup/measurement counts are recorded in each source "
        "JSON.\n"
    )
    markdown = header + body + notes
    print(markdown)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "threshold_percent": args.threshold_percent,
                    "results": details,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
