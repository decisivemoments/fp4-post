#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

MODEL_KEY="${1:-qwen2_0_5b}"
MODEL_ENV_FILE="${MODEL_ENV_FILE:-configs/grpo/model_env/${MODEL_KEY}.sh}"
if [[ -f "${MODEL_ENV_FILE}" ]]; then
    source "${MODEL_ENV_FILE}"
fi
source configs/grpo/experiment_env.sh

QAT_CHECKPOINT="${QAT_CHECKPOINT:-final}"
if [[ "${QAT_CHECKPOINT}" == "final" || "${QAT_CHECKPOINT}" == checkpoint-* ]]; then
    QAT_CHECKPOINT_NAME="${QAT_CHECKPOINT}"
else
    QAT_CHECKPOINT_NAME="checkpoint-${QAT_CHECKPOINT}"
fi

# Keep the first pass short unless the caller overrides it.
export GRPO_MAX_STEPS="${GRPO_MAX_STEPS:-1000}"
export QAT_MAX_STEPS="${QAT_MAX_STEPS:-1000}"

bash scripts/grpo/run_experiment.sh grpo bf16 "${MODEL_KEY}"
bash scripts/grpo/run_experiment.sh grpo direct_fp4 "${MODEL_KEY}"
bash scripts/grpo/run_experiment.sh qat full "${MODEL_KEY}"
python tools/convert_bitlinear_to_standard.py \
    "${QAT_OUTPUT_ROOT}/${MODEL_KEY}/full/${QAT_CHECKPOINT_NAME}" \
    "${QAT_OUTPUT_ROOT}/${MODEL_KEY}/full/${QAT_CHECKPOINT_NAME}-merged" \
    --verify
bash scripts/grpo/run_experiment.sh grpo qat_fp4 "${MODEL_KEY}"
bash scripts/grpo/run_experiment.sh grpo moving_mean "${MODEL_KEY}"
bash scripts/grpo/run_experiment.sh grpo full "${MODEL_KEY}"
