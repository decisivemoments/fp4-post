#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE_ROOT}/deepmath-103k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/benchmarks/step_time}"
MODELS="${MODELS:-qwen2_5_0_5b qwen2_5_math_1_5b}"
MODES="${MODES:-bf16 nvfp4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURED_STEPS="${MEASURED_STEPS:-20}"
NUM_CACHE_SAMPLES="${NUM_CACHE_SAMPLES:-64}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
COMPILE_QDQ="${COMPILE_QDQ:-false}"
CACHE_QUANTIZED_WEIGHT="${CACHE_QUANTIZED_WEIGHT:-false}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"

model_path_for_key() {
    case "$1" in
        qwen2_5_0_5b)
            echo "${WORKSPACE_ROOT}/qwen2.5-0.5b-instruct"
            ;;
        qwen2_5_math_1_5b)
            echo "${WORKSPACE_ROOT}/qwen2.5-math-1.5b"
            ;;
        *)
            echo "Unknown model key: $1" >&2
            return 2
            ;;
    esac
}

bool_flag() {
    local value="$1"
    local flag="$2"
    if [[ "${value}" == "true" ]]; then
        echo "${flag}"
    elif [[ "${value}" != "false" ]]; then
        echo "Expected true or false, got: ${value}" >&2
        return 2
    fi
}

mkdir -p "${OUTPUT_ROOT}/cache" "${OUTPUT_ROOT}/results"

gradient_checkpointing_flag="$(bool_flag "${GRADIENT_CHECKPOINTING}" "--gradient-checkpointing")"
compile_qdq_flag="$(bool_flag "${COMPILE_QDQ}" "--compile-qdq")"
cache_weight_flag="$(bool_flag "${CACHE_QUANTIZED_WEIGHT}" "--cache-quantized-weight")"

result_files=()
for model_key in ${MODELS}; do
    model_path="$(model_path_for_key "${model_key}")"
    cache_path="${OUTPUT_ROOT}/cache/${model_key}-deepmath-train-seq${SEQ_LENGTH}-n${NUM_CACHE_SAMPLES}.pt"

    python benchmarks/step_time/prepare_deepmath.py \
        --dataset-dir "${DATASET_DIR}" \
        --model-path "${model_path}" \
        --output "${cache_path}" \
        --seq-length "${SEQ_LENGTH}" \
        --num-samples "${NUM_CACHE_SAMPLES}"

    for mode in ${MODES}; do
        suffix="bs${BATCH_SIZE}-seq${SEQ_LENGTH}-gc${GRADIENT_CHECKPOINTING}"
        if [[ "${mode}" == "nvfp4" ]]; then
            suffix="${suffix}-compile${COMPILE_QDQ}-cache${CACHE_QUANTIZED_WEIGHT}"
        fi
        result_path="${OUTPUT_ROOT}/results/${model_key}-${mode}-${suffix}.json"

        command=(
            python benchmarks/step_time/benchmark_train_step.py
            --model-path "${model_path}"
            --data-cache "${cache_path}"
            --mode "${mode}"
            --output "${result_path}"
            --batch-size "${BATCH_SIZE}"
            --seq-length "${SEQ_LENGTH}"
            --warmup-steps "${WARMUP_STEPS}"
            --steps "${MEASURED_STEPS}"
            --attn-implementation "${ATTN_IMPLEMENTATION}"
        )
        if [[ -n "${gradient_checkpointing_flag}" ]]; then
            command+=("${gradient_checkpointing_flag}")
        fi
        if [[ "${mode}" == "nvfp4" && -n "${compile_qdq_flag}" ]]; then
            command+=("${compile_qdq_flag}")
        fi
        if [[ "${mode}" == "nvfp4" && -n "${cache_weight_flag}" ]]; then
            command+=("${cache_weight_flag}")
        fi

        echo "Running ${model_key} ${mode}"
        "${command[@]}"
        result_files+=("${result_path}")
    done
done

python benchmarks/step_time/summarize_results.py \
    "${result_files[@]}" \
    --markdown-output "${OUTPUT_ROOT}/summary.md" \
    --csv-output "${OUTPUT_ROOT}/summary.csv"
