#!/usr/bin/env python3
"""Create a paper-ready comparison from saved rollout-quality JSONL logs.

Example:
  python tools/plot_rollout_comparison.py \
    --run 'Direct FP4=outputs/grpo/model/direct_fp4/rollout_quality' \
    --run 'Mean self-distillation=outputs/grpo/model/full/rollout_quality' \
    --run 'BF16=outputs/grpo/model/bf16/rollout_quality' \
    --output AuthorKit27/Figures/rollout_comparison
"""

import argparse
import json
from pathlib import Path


COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]  # Okabe-Ito palette


def load_run(spec: str):
    if "=" not in spec:
        raise ValueError("Each --run must use LABEL=ROLLOUT_QUALITY_DIRECTORY")
    label, raw_path = spec.split("=", 1)
    path = Path(raw_path) / "step_quality.jsonl"
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda record: record["step"])
    return label, records


def moving_mean(values, window):
    result = []
    for index in range(len(values)):
        left = max(0, index - window + 1)
        chunk = [value for value in values[left:index + 1] if value is not None]
        result.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return result


def plot_series(ax, records, label, color, getter, window, band=False):
    import math

    steps = [record["step"] for record in records]
    values = [getter(record) for record in records]
    ax.plot(steps, values, color=color, alpha=0.20, linewidth=0.8)
    smooth = moving_mean(values, window)
    ax.plot(steps, smooth, color=color, linewidth=2.1, label=label)
    if band:
        counts = [record.get("num_samples", 0) for record in records]
        se = [math.sqrt(value * (1.0 - value) / count) if count else 0.0
              for value, count in zip(smooth, counts)]
        lo = [max(0.0, value - 1.96 * error) for value, error in zip(smooth, se)]
        hi = [min(1.0, value + 1.96 * error) for value, error in zip(smooth, se)]
        ax.fill_between(steps, lo, hi, color=color, alpha=0.12, linewidth=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True,
                        help="LABEL=rollout_quality directory; repeat for each method")
    parser.add_argument("--output", required=True, help="Output path without extension")
    parser.add_argument("--window", type=int, default=20, help="Trailing smoothing window in logged steps")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    runs = [load_run(spec) for spec in args.run]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.25), constrained_layout=True)

    panels = [
        ("Bad rollout rate", "Fraction of completions", lambda r: r.get("bad_ratio", 0.0), True),
        ("Mean logged task reward", "Mean logged task reward", lambda r: r.get("reward_summary", {}).get("reward_mean"), False),
        ("Generation diversity", "Unique-token ratio", lambda r: r.get("metric_summary", {}).get("unique_token_ratio_mean"), False),
    ]
    for axis, (title, ylabel, getter, band) in zip(axes, panels):
        for index, (label, records) in enumerate(runs):
            plot_series(axis, records, label, COLORS[index % len(COLORS)], getter, args.window, band)
        axis.set_title(title)
        axis.set_xlabel("GRPO step")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, loc="best")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
