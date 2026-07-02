# FP4 GRPO Experiment Setup

## Research Question

The first-stage test asks whether FP4 damages GRPO mainly by polluting rollout quality, and whether QAT self-distillation plus moving-mean residual FP4 repairs that damage.

## Data And Models

Default paths are defined in `configs/grpo/experiment_env.sh`.

- Training data: `DATA_ROOT/deepmath-103K`
- Eval data: local EvalScope datasets under `DATA_ROOT`, passed through `--dataset-args`
- Model root: `MODEL_ROOT`

Supported model keys in `scripts/grpo/run_experiment.sh`:

- `qwen2_0_5b`
- `qwen2_5_0_5b`
- `qwen2_5_math_1_5b`
- `qwen2_5_math_1_5b_base`
- `qwen2_5_math_7b`
- `qwen2_5_math_7b_base`
- `qwen3_1_7b`
- `qwen3_8b`
- `qwen3_8b_base`

Each model key has a model-specific environment file under `configs/grpo/model_env/`. `run_experiment.sh` loads it automatically before filling shared defaults.
The `_base` suffix is reserved for non-instruct/base checkpoints, so their outputs stay separate from the existing instruct-model experiments.

| Model key | Env file | GRPO per-device batch | Grad accum | Generations | Max completion | DeepSpeed config |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `qwen2_0_5b` | `configs/grpo/model_env/qwen2_0_5b.sh` | 8 | 4 | 4 | 1024 | `ds_config_zero2_no_offload.json` |
| `qwen2_5_0_5b` | `configs/grpo/model_env/qwen2_5_0_5b.sh` | 8 | 4 | 4 | 1024 | `ds_config_zero2_no_offload.json` |
| `qwen2_5_math_1_5b` | `configs/grpo/model_env/qwen2_5_math_1_5b.sh` | 4 | 8 | 4 | 1024 | `ds_config_zero2_no_offload.json` |
| `qwen2_5_math_1_5b_base` | `configs/grpo/model_env/qwen2_5_math_1_5b_base.sh` | 4 | 8 | 4 | 1024 | `ds_config_zero2_no_offload.json` |
| `qwen3_1_7b` | `configs/grpo/model_env/qwen3_1_7b.sh` | 4 | 8 | 4 | 1024 | `ds_config_zero2_no_offload.json` |
| `qwen2_5_math_7b` | `configs/grpo/model_env/qwen2_5_math_7b.sh` | 1 | 16 | 4 | 1024 | `ds_config_zero2.json` |
| `qwen2_5_math_7b_base` | `configs/grpo/model_env/qwen2_5_math_7b_base.sh` | 1 | 16 | 4 | 1024 | `ds_config_zero2.json` |
| `qwen3_8b` | `configs/grpo/model_env/qwen3_8b.sh` | 1 | 16 | 4 | 1024 | `ds_config_zero2.json` |
| `qwen3_8b_base` | `configs/grpo/model_env/qwen3_8b_base.sh` | 1 | 16 | 4 | 1024 | `ds_config_zero2.json` |

You can still override any value from the command line:

```bash
GRPO_BATCH_SIZE=2 GRPO_NUM_GENERATIONS=2 bash scripts/grpo/run_experiment.sh grpo bf16 qwen2_5_math_7b
```

## First-Stage Runs

Small 0.5B sanity run:

```bash
bash scripts/grpo/run_phase1.sh qwen2_0_5b
```

The default first-stage budget is 1000 GRPO steps and 1000 QAT steps. Override it when needed:

```bash
GRPO_MAX_STEPS=3000 QAT_MAX_STEPS=3000 bash scripts/grpo/run_phase1.sh qwen2_0_5b
```

Scale to the 1.5B math model:

```bash
GRPO_MAX_STEPS=3000 QAT_MAX_STEPS=3000 bash scripts/grpo/run_phase1.sh qwen2_5_math_1_5b
```

## Individual Runs

```bash
bash scripts/grpo/run_experiment.sh grpo bf16 qwen2_0_5b
bash scripts/grpo/run_experiment.sh grpo direct_fp4 qwen2_0_5b
bash scripts/grpo/run_experiment.sh qat full qwen2_0_5b
python tools/convert_bitlinear_to_standard.py \
  outputs/qat/qwen2_0_5b/full/final \
  outputs/qat/qwen2_0_5b/full/final-merged \
  --verify
bash scripts/grpo/run_experiment.sh grpo qat_fp4 qwen2_0_5b
bash scripts/grpo/run_experiment.sh grpo moving_mean qwen2_0_5b
bash scripts/grpo/run_experiment.sh grpo full qwen2_0_5b
```

By default GRPO uses `configs/grpo/ds_config_zero2_no_offload.json`, which keeps ZeRO-2 optimizer states on GPU. If a larger model does not fit, switch back to CPU optimizer offload:

```bash
DEEPSPEED_CONFIG=configs/grpo/ds_config_zero2.json bash scripts/grpo/run_experiment.sh grpo bf16 qwen2_0_5b
```

Method definitions:

- `bf16`: no Metis replacement.
- `direct_fp4`: Metis FP4 with the fixed W-SVD weight path, while activation and grad both use direct FP4. Valid for GRPO only.
- `qat_fp4`: starts from the QAT checkpoint and uses the same direct activation/grad FP4 path.
- `moving_mean`: starts from the base model and uses the fixed W-SVD weight path, while activation and grad both use moving-mean residual FP4.
- `full`: for QAT, distills with W-SVD plus moving-mean activation/grad; for GRPO, starts from that QAT checkpoint and keeps moving-mean activation/grad.

Metis configuration rule:

- W uses one fixed SVD quantization path, controlled by `METIS_WEIGHT_SVD` and `METIS_WEIGHT_SVD_RANK`.
- Activation and grad always use the same runtime strategy in these scripts.
- For GRPO `direct_fp4` and `qat_fp4`, activation and grad both disable residual SVD/mean handling.
- For QAT distillation (`stage=qat full`), W uses SVD while activation and grad both enable `metis_mode=mean` with `METIS_ACTIVATION_GRAD_RANK`.
- For `moving_mean` and `full`, activation and grad both enable `metis_mode=mean` with `METIS_ACTIVATION_GRAD_RANK`.

## QAT Checkpoint Merge

QAT saves Metis `BitLinear` weights split as `ulinear`, `s`, `vlinear`, and `warmup_linear`. Before a QAT checkpoint is used as a normal `model_name_or_path`, merge it back to standard Linear weights:

```bash
python tools/convert_bitlinear_to_standard.py \
  outputs/qat/<model_key>/full/final \
  outputs/qat/<model_key>/full/final-merged \
  --verify
```

To test an intermediate QAT checkpoint, choose the step explicitly:

```bash
python tools/convert_bitlinear_to_standard.py \
  outputs/qat/qwen2_5_math_1_5b_base/full/checkpoint-1000 \
  outputs/qat/qwen2_5_math_1_5b_base/full/checkpoint-1000-merged \
  --verify
```

Then run GRPO from that merged checkpoint:

```bash
QAT_CHECKPOINT=checkpoint-1000 bash scripts/grpo/run_experiment.sh grpo full qwen2_5_math_1_5b_base
QAT_CHECKPOINT=1000 bash scripts/grpo/run_experiment.sh grpo qat_fp4 qwen2_5_math_1_5b_base
```

`QAT_CHECKPOINT=final` is the default. Numeric values such as `QAT_CHECKPOINT=1000` are expanded to `checkpoint-1000`. `run_experiment.sh` looks for:

```text
outputs/qat/<model_key>/full/<QAT_CHECKPOINT>-merged/
```

You can bypass that convention with:

```bash
QAT_SOURCE=/path/to/merged/model bash scripts/grpo/run_experiment.sh grpo full <model_key>
```

## Evaluation

Run evalscope on any trained model directory:

```bash
bash scripts/grpo/run_evalscope.sh outputs/grpo/qwen2_0_5b/bf16
```

If the second argument is omitted, the output name is inferred from the path as `<algorithm>-<model_key>-<method>`. For example, the command above writes to:

```text
outputs/evalscope/grpo-qwen2_0_5b-bf16/
```

The default eval datasets are:

```bash
aime24 aime25 amc math_500 minerva_math arc gpqa_diamond mmlu_pro
```

Change them with:

```bash
EVALSCOPE_DATASETS="aime24 math_500" bash scripts/grpo/run_evalscope.sh <model_path> <name>
```

When the first argument is a model key, the script uses the base model path and model-specific eval batch size:

```bash
bash scripts/grpo/run_evalscope.sh qwen2_5_math_1_5b base-1.5b
```

For checkpoint paths, set `MODEL_KEY` if the path does not contain the model key:

```bash
MODEL_KEY=qwen2_5_math_7b bash scripts/grpo/run_evalscope.sh /path/to/checkpoint exp-7b
```

The script passes local datasets in this form:

```bash
evalscope eval \
  --datasets aime24 aime25 amc math_500 minerva_math arc gpqa_diamond mmlu_pro \
  --dataset-hub Local \
  --dataset-args '{"aime24":{"dataset_id":"DATA_ROOT/evalscope_aime24"},...}' \
  --eval-batch-size "$EVAL_BATCH_SIZE"
```

Pass local evalscope-version-specific flags with `EVALSCOPE_EXTRA_ARGS`, for example `EVALSCOPE_EXTRA_ARGS="--limit 10"`.

## Output Artifacts

GRPO 检测与分析代码的详细说明见 `docs/grpo_diagnostics.md`。

Training outputs:

```text
outputs/qat/<model_key>/<method>/
outputs/grpo/<model_key>/<method>/
```

TensorBoard event files are written directly into each experiment's fixed `runs` directory:

```text
outputs/qat/<model_key>/<method>/runs/
outputs/grpo/<model_key>/<method>/runs/
```

Open GRPO experiments with:

```bash
tensorboard --logdir outputs/grpo
```

Open QAT experiments with:

```bash
tensorboard --logdir outputs/qat
```

Rollout quality logs are written under each GRPO output directory:

```text
rollout.jsonl
rollout_quality/sample_quality.jsonl
rollout_quality/step_quality.jsonl
rollout_analysis/sample_analysis.jsonl
rollout_analysis/step_analysis.jsonl
metis_diagnostics.jsonl
```

Eval outputs:

```text
outputs/evalscope/<eval_name>/
```

## Current Claim Boundary

This setup covers the first executable comparison: BF16, direct FP4, QAT only, moving-mean only, and QAT plus moving mean.

The rollout-vs-train attribution experiment is not fully implemented yet, because the current `GRPOTrainer` path uses the same policy model for rollout generation and train forward. That experiment needs a separate rollout model or a controlled generate/forward override before it can honestly run `FP4 rollout + BF16 train` and `BF16 rollout + FP4 train`.
