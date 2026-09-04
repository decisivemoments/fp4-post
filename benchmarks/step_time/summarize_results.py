#!/usr/bin/env python3
"""Summarize paired BF16/NVFP4 benchmark JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
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


def main() -> None:
    args = parse_args()
    grouped: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for result in load_results(args.results):
        grouped[comparison_key(result)][result["mode"]] = result

    rows: list[dict[str, Any]] = []
    for key, modes in sorted(grouped.items()):
        if "bf16" not in modes or "nvfp4" not in modes:
            continue
        bf16 = modes["bf16"]
        nvfp4 = modes["nvfp4"]
        bf16_ms = bf16["timing_ms"]["wall_ms"]["median"]
        nvfp4_ms = nvfp4["timing_ms"]["wall_ms"]["median"]
        speedup = bf16_ms / nvfp4_ms
        change_percent = (nvfp4_ms - bf16_ms) / bf16_ms * 100
        if change_percent <= -args.threshold_percent:
            decision = "NVFP4 faster"
        elif change_percent >= args.threshold_percent:
            decision = "NVFP4 slower"
        else:
            decision = "within threshold"

        rows.append(
            {
                "model": key[0],
                "batch_size": key[1],
                "seq_length": key[2],
                "gradient_checkpointing": key[3],
                "bf16_median_ms": bf16_ms,
                "nvfp4_median_ms": nvfp4_ms,
                "speedup_bf16_over_nvfp4": speedup,
                "nvfp4_latency_change_percent": change_percent,
                "bf16_tokens_per_second": bf16[
                    "padded_tokens_per_second_at_median"
                ],
                "nvfp4_tokens_per_second": nvfp4[
                    "padded_tokens_per_second_at_median"
                ],
                "bf16_peak_allocated_gib": (
                    bf16["peak_memory_bytes"]["allocated"] / 2**30
                ),
                "nvfp4_peak_allocated_gib": (
                    nvfp4["peak_memory_bytes"]["allocated"] / 2**30
                ),
                "nvfp4_qdq_compile_status": nvfp4["nvfp4"][
                    "compile_qdq_status"
                ],
                "decision": decision,
            }
        )

    if not rows:
        raise RuntimeError("No paired BF16/NVFP4 results found")

    header = (
        "| Model | BS | Seq | GC | BF16 median (ms) | NVFP4 median (ms) | "
        "BF16/NVFP4 speedup | NVFP4 latency vs BF16 | Decision |\n"
        "|---|---:|---:|:---:|---:|---:|---:|---:|---|\n"
    )
    body = "".join(
        "| {model} | {batch_size} | {seq_length} | {gc} | "
        "{bf16:.3f} | {nvfp4:.3f} | {speedup:.3f}x | {change:+.2f}% | "
        "{decision} |\n".format(
            model=row["model"],
            batch_size=row["batch_size"],
            seq_length=row["seq_length"],
            gc="on" if row["gradient_checkpointing"] else "off",
            bf16=row["bf16_median_ms"],
            nvfp4=row["nvfp4_median_ms"],
            speedup=row["speedup_bf16_over_nvfp4"],
            change=row["nvfp4_latency_change_percent"],
            decision=row["decision"],
        )
        for row in rows
    )
    note = (
        "\n`speedup = BF16 median / NVFP4 median`; values above 1 mean NVFP4 "
        "is faster. A positive latency change means NVFP4 is slower. The "
        "current direct_fp4 path is fake-QDQ followed by BF16 `torch.matmul`, "
        "not a native FP4 Tensor Core GEMM.\n"
    )
    markdown = header + body + note
    print(markdown)

    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown, encoding="utf-8")
    if args.csv_output:
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
