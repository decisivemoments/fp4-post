#!/usr/bin/env python3
"""Create a robust NVTX cost estimate for native full-NVFP4."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


CATEGORIES = {
    "pack_scale_mean": [
        "native_nvfp4.activation_mean_and_pack",
        "native_nvfp4.gradient_mean_and_pack",
        "native_nvfp4.weight_pack",
    ],
    "native_fp4_gemm": [
        "native_nvfp4.fprop_native_gemm",
        "native_nvfp4.dgrad_native_gemm",
        "native_nvfp4.wgrad_native_gemm",
    ],
    "bf16_mean_correction": [
        "native_nvfp4.fprop_mean_correction_bf16",
        "native_nvfp4.dgrad_mean_correction_bf16",
        "native_nvfp4.wgrad_mean_correction_bf16",
    ],
    "bf16_rank64_path": [
        "native_nvfp4.lowrank_fprop_bf16",
        "native_nvfp4.lowrank_backward_bf16",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvtx-csv", type=Path, required=True)
    parser.add_argument("--formal-summary", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.nvtx_csv.open(newline="", encoding="utf-8") as handle:
        rows = {
            row["Range"].lstrip(":"): row for row in csv.DictReader(handle)
        }
    summary = json.loads(args.formal_summary.read_text(encoding="utf-8"))
    formal = next(
        result
        for result in summary["results"]
        if result["model"] == args.model
    )
    formal_step_ms = float(formal["native_full_nvfp4_median_ms"])

    ranges: dict[str, dict[str, Any]] = {}
    for names in CATEGORIES.values():
        for name in names:
            row = rows[name]
            instances = int(row["Range Instances"])
            median_ns = float(row["Proj Med (ns)"])
            ranges[name] = {
                "instances": instances,
                "projected_median_us": median_ns / 1000,
                "robust_total_ms": instances * median_ns / 1e6,
                "raw_total_projected_ms": (
                    float(row["Total Proj Time (ns)"]) / 1e6
                ),
            }

    categories = {}
    categorized_ms = 0.0
    for category, names in CATEGORIES.items():
        robust_ms = sum(ranges[name]["robust_total_ms"] for name in names)
        categorized_ms += robust_ms
        categories[category] = {
            "ranges": names,
            "robust_total_ms": robust_ms,
            "percent_of_formal_step": robust_ms / formal_step_ms * 100,
        }

    payload = {
        "schema_version": 1,
        "model": args.model,
        "formal_native_step_median_ms": formal_step_ms,
        "method": (
            "NVTX projected median per range instance multiplied by the "
            "instance count. This suppresses profiler-induced long-tail "
            "outliers. Percentages are attribution estimates because "
            "asynchronous/nested NVTX projections can overlap."
        ),
        "categories": categories,
        "categorized_total_ms": categorized_ms,
        "categorized_percent_of_formal_step": (
            categorized_ms / formal_step_ms * 100
        ),
        "unattributed_or_overlapped_ms": formal_step_ms - categorized_ms,
        "ranges": ranges,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["categories"], indent=2))


if __name__ == "__main__":
    main()
