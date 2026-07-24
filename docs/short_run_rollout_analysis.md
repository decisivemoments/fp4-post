# 10-Step GRPO Rollout Study

## Purpose

For each model, compare direct FP4 GRPO (before QAT) with QAT plus moving-mean
FP4 GRPO (after QAT) over the first ten optimizer steps. This is a diagnostic
study, not a downstream benchmark run.

The training phase only records rollout text-quality and reward data. Plotting
is an offline operation, so manually stopping a run after ten steps does not
discard the records that have already been flushed to JSONL.

## What Is Collected

Enable `ANALYZE_ROLLOUT=true`. Each GRPO output directory then contains:

```text
rollout.jsonl
rollout_quality/sample_quality.jsonl
rollout_quality/step_quality.jsonl
rollout_quality/sample_quality.csv
rollout_quality/step_quality.csv
metis_diagnostics.jsonl
```

`sample_quality.jsonl` contains one record per completion, including the raw
text, reward, rule labels, and raw quality metrics. `step_quality.jsonl`
contains the step-level bad ratio, reward mean/std, bad-type rates, and group
health statistics.

For this study, do not enable logits capture:

```bash
COLLECT_ROLLOUT_LOGITS=false
PLOT_ROLLOUT_ON_TRAIN_END=false
```

The rollout/train logits mismatch code remains experimental because its capture
and reward-callback timing must be validated separately. It is not required for
the three figures below.

## Run Protocol

Use the same model, data subset, decoding configuration, batch size, number of
generations, seed, and `GRPO_MAX_STEPS=10` for each pair. The only intended
difference is the method.

```bash
GRPO_MAX_STEPS=10 ANALYZE_ROLLOUT=true \
  bash scripts/grpo/run_experiment.sh grpo direct_fp4 <model_key>

GRPO_MAX_STEPS=10 ANALYZE_ROLLOUT=true \
  bash scripts/grpo/run_experiment.sh grpo full <model_key>
```

If you stop a run manually, wait until the current logging line finishes when
possible. The JSONL writers use line buffering, so completed records are
available for offline analysis.

## Offline Figures

After both runs, create exactly three figures for one model:

```bash
python3 tools/analyze_rollout_short_run.py \
  --before outputs/grpo/<model_key>/direct_fp4 \
  --after outputs/grpo/<model_key>/full \
  --model-label <model_key> \
  --max-steps 10
```

Outputs are written to `AuthorKit27/Figures/rollout_short_runs/`:

```text
<model>_bad_ratio.pdf/png
<model>_reward.pdf/png
<model>_bad_type_distribution.pdf/png
```

The third plot is multi-label: a completion can trigger more than one bad type.
Its denominator is the number of bad completions, so bar percentages need not
sum to 100%. This is intentional; it preserves the fact that, for example, a
token loop can also have low lexical diversity.

## Quality Rules Used

The current GRPO entrypoint uses the common thresholds in
`rollout_quality.QualityConfig`, but disables the generic repeated-$n$-gram
rule (`enable_repetition=false`). It still records the corresponding numerical
metrics and still flags token loops, low diversity, short outputs, garbling,
formatting/symbol/punctuation/ellipsis floods, and excessive Chinese--English
switching. Keep this configuration identical across the before/after pair.

## Interpretation Boundary

These short runs can support a diagnostic claim about early rollout stability:
direct FP4 has a higher bad ratio and weaker reward signal than the QAT-adapted
path, if the plotted data show that pattern. They do not establish final
downstream accuracy or convergence. Use the longer matched runs and EvalScope
results for those claims.
