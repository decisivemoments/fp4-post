#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

USER_SET_EVAL_BATCH_SIZE=0
if [[ -n "${EVAL_BATCH_SIZE+x}" ]]; then
    USER_SET_EVAL_BATCH_SIZE=1
fi

source configs/grpo/experiment_env.sh

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/grpo/run_evalscope.sh <model_key_or_model_path> [eval_name]

Examples:
  bash scripts/grpo/run_evalscope.sh qwen2_5_math_1_5b base-1.5b
  bash scripts/grpo/run_evalscope.sh qwen2_5_math_1_5b_base base-math-1.5b
  bash scripts/grpo/run_evalscope.sh outputs/grpo/qwen2_0_5b/bf16
  MODEL_KEY=qwen2_5_math_7b bash scripts/grpo/run_evalscope.sh /path/to/checkpoint exp-7b

Defaults:
  EVALSCOPE_DATASETS="aime24 aime25 amc math_500 minerva_math arc gpqa_diamond mmlu_pro"
  EVAL_BATCH_SIZE is loaded from configs/grpo/model_env/<model_key>.sh when model_key is known.
  EVALSCOPE_EXTRA_ARGS="" can be used to pass cluster/version-specific evalscope flags, e.g. LIMIT.
  EVALSCOPE_PRINT_NAME_ONLY=true prints the inferred output name and exits.
USAGE
}

if [[ $# -lt 1 ]]; then
    usage
    exit 2
fi

model_path_for_key() {
    case "$1" in
        qwen2_0_5b) echo "${MODEL_ROOT}/qwen2-0.5B-instruct" ;;
        qwen2_5_0_5b) echo "${MODEL_ROOT}/qwen-2.5-0.5b-instruct" ;;
        qwen2_5_math_1_5b) echo "${MODEL_ROOT}/qwen2.5-math-1.5b-instruct" ;;
        qwen2_5_math_1_5b_base) echo "${MODEL_ROOT}/qwen2.5-math-1.5b" ;;
        qwen2_5_math_7b) echo "${MODEL_ROOT}/qwen2.5-math-7b-instruct" ;;
        qwen2_5_math_7b_base) echo "${MODEL_ROOT}/qwen2.5-math-7b" ;;
        qwen3_1_7b) echo "${MODEL_ROOT}/qwen3-1.7B" ;;
        qwen3_8b) echo "${MODEL_ROOT}/qwen3-8b" ;;
        qwen3_8b_base) echo "${MODEL_ROOT}/qwen3-8b-base" ;;
        *) return 1 ;;
    esac
}

infer_model_key_from_path() {
    local path="$1"
    local key
    for key in qwen2_0_5b qwen2_5_0_5b qwen2_5_math_1_5b_base qwen2_5_math_1_5b qwen2_5_math_7b_base qwen2_5_math_7b qwen3_1_7b qwen3_8b_base qwen3_8b; do
        if [[ "${path}" == *"/${key}/"* ]]; then
            echo "${key}"
            return 0
        fi
    done
    return 1
}

sanitize_eval_name_part() {
    local value="$1"
    value="${value%/}"
    value="${value//\//-}"
    value="${value// /_}"
    echo "${value}"
}

infer_eval_name() {
    local path="$1"
    local model_key="$2"
    local trimmed_path="${path%/}"
    local IFS='/'
    local -a parts
    local i
    local algorithm=""
    local method=""

    read -r -a parts <<< "${trimmed_path}"

    if [[ -n "${model_key}" ]]; then
        for i in "${!parts[@]}"; do
            if [[ "${parts[$i]}" == "${model_key}" ]]; then
                if (( i > 0 )); then
                    algorithm="${parts[$((i - 1))]}"
                fi
                if (( i + 1 < ${#parts[@]} )); then
                    method="${parts[$((i + 1))]}"
                fi
                break
            fi
        done
    fi

    if [[ -z "${algorithm}" || "${algorithm}" == "outputs" ]]; then
        if [[ -n "${model_key}" && "${trimmed_path}" == "$(model_path_for_key "${model_key}" 2>/dev/null || true)" ]]; then
            algorithm="base"
        else
            algorithm="custom"
        fi
    fi

    if [[ -z "${method}" ]]; then
        if [[ "${algorithm}" == "base" ]]; then
            method="pretrained"
        else
            method="$(basename "${trimmed_path}")"
        fi
    fi

    if [[ -z "${model_key}" ]]; then
        model_key="$(basename "$(dirname "${trimmed_path}")")"
    fi

    printf "%s-%s-%s\n" \
        "$(sanitize_eval_name_part "${algorithm}")" \
        "$(sanitize_eval_name_part "${model_key}")" \
        "$(sanitize_eval_name_part "${method}")"
}

MODEL_ARG="$1"
if MODEL_PATH_FROM_KEY="$(model_path_for_key "${MODEL_ARG}")"; then
    MODEL_KEY="${MODEL_ARG}"
    MODEL_PATH="${MODEL_PATH_FROM_KEY}"
else
    MODEL_PATH="${MODEL_ARG}"
    MODEL_KEY="${MODEL_KEY:-}"
    if [[ -z "${MODEL_KEY}" ]]; then
        MODEL_KEY="$(infer_model_key_from_path "${MODEL_PATH}" || true)"
    fi
fi

if [[ -n "${MODEL_KEY}" ]]; then
    MODEL_ENV_FILE="${MODEL_ENV_FILE:-configs/grpo/model_env/${MODEL_KEY}.sh}"
    if [[ -f "${MODEL_ENV_FILE}" ]]; then
        if [[ "${USER_SET_EVAL_BATCH_SIZE}" -eq 0 ]]; then
            unset EVAL_BATCH_SIZE
        fi
        source "${MODEL_ENV_FILE}"
        echo "Loaded model env: ${MODEL_ENV_FILE}"
    else
        echo "Warning: model env file not found: ${MODEL_ENV_FILE}; using shared eval defaults." >&2
    fi
fi

EVAL_NAME="${2:-$(infer_eval_name "${MODEL_PATH}" "${MODEL_KEY}")}"
OUT_DIR="${EVALSCOPE_OUTPUT_ROOT}/${EVAL_NAME}"

if [[ "${EVALSCOPE_PRINT_NAME_ONLY:-false}" == "true" ]]; then
    echo "${EVAL_NAME}"
    echo "${OUT_DIR}"
    exit 0
fi

mkdir -p "${OUT_DIR}"

if ! command -v evalscope >/dev/null 2>&1; then
    echo "evalscope is not on PATH. Activate the environment that contains evalscope, then rerun this script." >&2
    exit 127
fi

read -r -a DATASETS <<< "${EVALSCOPE_DATASETS}"

DATASET_ARGS_JSON="{\"aime24\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_aime24\"},\"aime25\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_aime25\"},\"amc\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_amc_22-24\"},\"math_500\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_math500\"},\"minerva_math\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_minerva\"},\"arc\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_arc\"},\"gpqa_diamond\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_gpqa\"},\"mmlu_pro\":{\"dataset_id\":\"${DATA_ROOT}/evalscope_mmlu_pro\"}}"

echo "Running evalscope: model=${MODEL_PATH} output=${OUT_DIR}"
echo "Datasets: ${EVALSCOPE_DATASETS}"
echo "Eval batch size: ${EVAL_BATCH_SIZE}"

evalscope eval \
    --model "${MODEL_PATH}" \
    --datasets "${DATASETS[@]}" \
    --dataset-hub "${EVALSCOPE_DATASET_HUB}" \
    --dataset-args "${DATASET_ARGS_JSON}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --work-dir "${OUT_DIR}" \
    ${EVALSCOPE_EXTRA_ARGS:-}
