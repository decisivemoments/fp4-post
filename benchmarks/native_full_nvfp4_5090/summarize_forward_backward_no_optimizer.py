#!/usr/bin/env python3
"""Summarize no-optimizer forward/backward capacity and timing results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def median_metric(runs: list[dict[str, Any]], key: str) -> float:
    return statistics.median(
        float(run["timing_ms"][key]["median"]) for run in runs
    )


def main() -> None:
    args = parse_args()
    grouped: dict[
        tuple[str, str, int, str], list[dict[str, Any]]
    ] = defaultdict(
        list
    )
    for path in sorted(args.results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("benchmark_scope") != (
            "forward_loss_backward_no_optimizer"
        ):
            raise RuntimeError(f"Wrong benchmark scope in {path}")
        if result.get("optimizer_constructed") is not False:
            raise RuntimeError(f"Optimizer was constructed in {path}")
        model = Path(result["model_path"]).name
        execution = (
            "cuda_graph"
            if result.get("cuda_graph", {}).get("enabled") is True
            else "eager"
        )
        grouped[
            (execution, model, int(result["batch_size"]), result["mode"])
        ].append(
            result
        )
    if not grouped:
        raise RuntimeError(f"No result JSON files in {args.results_dir}")

    aggregates: list[dict[str, Any]] = []
    for (execution, model, batch, mode), runs in sorted(grouped.items()):
        forward_ms = median_metric(runs, "forward_ms")
        backward_ms = median_metric(runs, "backward_ms")
        total_ms = median_metric(runs, "forward_backward_ms")
        peak_gib = statistics.median(
            float(run["peak_memory_bytes"]["allocated"]) / 2**30
            for run in runs
        )
        peak_reserved_gib = statistics.median(
            float(run["peak_memory_bytes"]["reserved"]) / 2**30
            for run in runs
        )
        tokens = batch * int(runs[0]["seq_length"])
        aggregates.append(
            {
                "execution": execution,
                "model": model,
                "batch_size": batch,
                "mode": mode,
                "repeats": len(runs),
                "forward_ms": forward_ms,
                "backward_ms": backward_ms,
                "forward_backward_ms": total_ms,
                "padded_tokens_per_second": tokens / (total_ms / 1000),
                "peak_allocated_gib": peak_gib,
                "peak_reserved_gib": peak_reserved_gib,
            }
        )

    by_key = {
        (
            row["execution"],
            row["model"],
            row["batch_size"],
            row["mode"],
        ): row
        for row in aggregates
    }
    paired: list[dict[str, Any]] = []
    for execution, model, batch in sorted(
        {
            (row["execution"], row["model"], row["batch_size"])
            for row in aggregates
        }
    ):
        bf16 = by_key.get((execution, model, batch, "bf16"))
        native = by_key.get(
            (execution, model, batch, "native_full_nvfp4")
        )
        if bf16 is None or native is None:
            continue
        paired.append(
            {
                "execution": execution,
                "model": model,
                "batch_size": batch,
                "bf16_forward_ms": bf16["forward_ms"],
                "nvfp4_forward_ms": native["forward_ms"],
                "forward_speedup_bf16_over_nvfp4": (
                    bf16["forward_ms"] / native["forward_ms"]
                ),
                "bf16_backward_ms": bf16["backward_ms"],
                "nvfp4_backward_ms": native["backward_ms"],
                "backward_speedup_bf16_over_nvfp4": (
                    bf16["backward_ms"] / native["backward_ms"]
                ),
                "bf16_forward_backward_ms": bf16[
                    "forward_backward_ms"
                ],
                "nvfp4_forward_backward_ms": native[
                    "forward_backward_ms"
                ],
                "speedup_bf16_over_nvfp4": (
                    bf16["forward_backward_ms"]
                    / native["forward_backward_ms"]
                ),
                "bf16_padded_tokens_per_second": bf16[
                    "padded_tokens_per_second"
                ],
                "nvfp4_padded_tokens_per_second": native[
                    "padded_tokens_per_second"
                ],
                "bf16_peak_allocated_gib": bf16["peak_allocated_gib"],
                "nvfp4_peak_allocated_gib": native[
                    "peak_allocated_gib"
                ],
                "bf16_peak_reserved_gib": bf16["peak_reserved_gib"],
                "nvfp4_peak_reserved_gib": native[
                    "peak_reserved_gib"
                ],
            }
        )

    maxima: list[dict[str, Any]] = []
    for execution, model in sorted(
        {(row["execution"], row["model"]) for row in aggregates}
    ):
        modes = {}
        for mode in ("bf16", "native_full_nvfp4"):
            candidates = [
                row
                for row in aggregates
                if row["execution"] == execution
                and row["model"] == model
                and row["mode"] == mode
            ]
            if candidates:
                modes[mode] = max(
                    candidates,
                    key=lambda row: row["batch_size"],
                )
        if set(modes) == {"bf16", "native_full_nvfp4"}:
            maxima.append(
                {
                    "execution": execution,
                    "model": model,
                    "bf16": modes["bf16"],
                    "native_full_nvfp4": modes[
                        "native_full_nvfp4"
                    ],
                    "max_throughput_speedup_nvfp4_over_bf16": (
                        modes["native_full_nvfp4"][
                            "padded_tokens_per_second"
                        ]
                        / modes["bf16"]["padded_tokens_per_second"]
                    ),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.output_dir / "forward_backward_aggregate.json"
    paired_path = args.output_dir / "forward_backward_paired.json"
    maxima_path = args.output_dir / "forward_backward_maxima.json"
    aggregate_path.write_text(
        json.dumps(aggregates, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paired_path.write_text(
        json.dumps(paired, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    maxima_path.write_text(
        json.dumps(maxima, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# Forward/backward only, no optimizer",
        "",
        "`BF16/NVFP4 > 1` 表示 Native Full-NVFP4 更快。",
        "",
        "## Maximum common batch",
        "",
        "| Execution | Model | Batch | BF16 fwd/bwd/total (ms) | NVFP4 fwd/bwd/total (ms) | Total speedup | BF16/NVFP4 tok/s | Peak alloc/reserved GiB BF16/NVFP4 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in paired:
        markdown.append(
            f"| {row['execution']} | {row['model']} | "
            f"{row['batch_size']} | "
            f"{row['bf16_forward_ms']:.3f}/"
            f"{row['bf16_backward_ms']:.3f}/"
            f"{row['bf16_forward_backward_ms']:.3f} | "
            f"{row['nvfp4_forward_ms']:.3f}/"
            f"{row['nvfp4_backward_ms']:.3f}/"
            f"{row['nvfp4_forward_backward_ms']:.3f} | "
            f"{row['speedup_bf16_over_nvfp4']:.3f}x | "
            f"{row['bf16_padded_tokens_per_second']:.1f}/"
            f"{row['nvfp4_padded_tokens_per_second']:.1f} | "
            f"{row['bf16_peak_allocated_gib']:.2f}/"
            f"{row['bf16_peak_reserved_gib']:.2f} vs "
            f"{row['nvfp4_peak_allocated_gib']:.2f}/"
            f"{row['nvfp4_peak_reserved_gib']:.2f} |"
        )
    markdown.extend(
        [
            "",
            "## Maximum batch per mode",
            "",
            "| Execution | Model | BF16 batch / tok/s | NVFP4 batch / tok/s | NVFP4/BF16 max throughput |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in maxima:
        bf16 = row["bf16"]
        native = row["native_full_nvfp4"]
        markdown.append(
            f"| {row['execution']} | {row['model']} | "
            f"{bf16['batch_size']} / "
            f"{bf16['padded_tokens_per_second']:.1f} | "
            f"{native['batch_size']} / "
            f"{native['padded_tokens_per_second']:.1f} | "
            f"{row['max_throughput_speedup_nvfp4_over_bf16']:.3f}x |"
        )
    markdown_path = args.output_dir / "forward_backward_summary.md"
    markdown_path.write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {aggregate_path}, {paired_path}, {maxima_path}, "
        f"and {markdown_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
