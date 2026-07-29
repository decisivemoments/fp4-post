#!/usr/bin/env python3
"""Make the three offline figures used by the short GRPO rollout study.

The script deliberately reads only completed JSONL records.  It never imports
training code, TensorBoard, PyTorch, or model weights, so it is safe to run
after manually interrupting a 10-step training job.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


COLORS = {"before": "#D55E00", "after": "#0072B2"}  # colour-blind safe
LABELS = {"before": "Direct FP4", "after": "Mean self-distillation"}


def read_jsonl(path: Path):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_quality_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.name == "rollout_quality" else path / "rollout_quality"


def load_step_records(raw_path: str, max_steps: int):
    path = resolve_quality_dir(raw_path) / "step_quality.jsonl"
    records = [r for r in read_jsonl(path) if int(r.get("step", -1)) < max_steps]
    by_step = defaultdict(list)
    for record in records:
        by_step[int(record["step"])].append(record)

    # A step can be logged more than once in framework-specific reward calls.
    # Aggregate such duplicates before plotting so every x-coordinate means one
    # optimizer step rather than one reward-function invocation.
    result = []
    for step in sorted(by_step):
        rows = by_step[step]
        result.append({
            "step": step,
            "bad_ratio": sum(float(r.get("bad_ratio", 0.0)) for r in rows) / len(rows),
            "reward_mean": _mean_or_none(
                [r.get("reward_summary", {}).get("reward_mean") for r in rows]
            ),
        })
    return result


def _mean_or_none(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def load_bad_type_distribution(raw_path: str, max_steps: int):
    path = resolve_quality_dir(raw_path) / "sample_quality.jsonl"
    samples = [r for r in read_jsonl(path) if int(r.get("step", -1)) < max_steps]
    bad_samples = [r for r in samples if r.get("is_bad", False)]
    counts = Counter()
    for sample in bad_samples:
        counts.update(sample.get("bad_types", []))
    return counts, len(bad_samples), len(samples)


def style_axis(axis):
    axis.grid(axis="y", alpha=0.25, linewidth=0.6)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel("GRPO step")


def plot_time_series(before, after, key, ylabel, title, output, max_steps):
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    for name, records in [("before", before), ("after", after)]:
        steps = [r["step"] for r in records]
        values = [r[key] if r[key] is not None else float("nan") for r in records]
        axis.plot(steps, values, marker="o", markersize=4, linewidth=1.8,
                  color=COLORS[name], label=LABELS[name])
    axis.set_xlim(-0.25, max_steps - 0.75)
    axis.set_xticks(range(max_steps))
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    style_axis(axis)
    axis.legend(frameon=False, loc="best")
    if key == "bad_ratio":
        axis.set_ylim(0.0, 1.0)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")


def plot_bad_type_distribution(counts, bad_count, total_count, output):
    import matplotlib.pyplot as plt

    if not counts:
        raise ValueError("No bad examples were found in the selected direct-FP4 log.")
    labels, values = zip(*counts.most_common())
    shares = [100.0 * value / bad_count for value in values]
    fig_height = max(2.8, 0.38 * len(labels) + 1.2)
    fig, axis = plt.subplots(figsize=(5.0, fig_height), constrained_layout=True)
    bars = axis.barh(labels[::-1], shares[::-1], color="#D55E00")
    for bar, share, count in zip(bars, shares[::-1], values[::-1]):
        axis.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                  f"{share:.1f}% ({count})", va="center", fontsize=8)
    axis.set_xlabel("Share of bad completions triggering rule (%)")
    axis.set_title(f"Direct-FP4 failure types (n={bad_count} bad / {total_count} total)")
    axis.set_xlim(0, max(shares) * 1.20)
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True,
                        help="Direct-FP4 output directory or its rollout_quality subdirectory")
    parser.add_argument("--after", required=True,
                        help="Mean self-distillation output directory or its rollout_quality subdirectory")
    parser.add_argument("--model-label", required=True,
                        help="Used in filenames, e.g. qwen2_5_math_1_5b")
    parser.add_argument("--output-dir", default="AuthorKit27/Figures/rollout_short_runs")
    parser.add_argument("--max-steps", type=int, default=10)
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

    before = load_step_records(args.before, args.max_steps)
    after = load_step_records(args.after, args.max_steps)
    if not before or not after:
        raise ValueError("Both runs need at least one step_quality record in the selected range.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.model_label
    plot_time_series(before, after, "bad_ratio", "Bad completion ratio",
                     "Direct FP4 vs. Mean self-distillation",
                     Path(f"{prefix}_bad_ratio"), args.max_steps)
    plot_time_series(before, after, "reward_mean", "Mean reward",
                     "Reward signal with Mean self-distillation",
                     Path(f"{prefix}_reward"), args.max_steps)
    counts, bad_count, total_count = load_bad_type_distribution(args.before, args.max_steps)
    plot_bad_type_distribution(counts, bad_count, total_count,
                               Path(f"{prefix}_bad_type_distribution"))


if __name__ == "__main__":
    main()
