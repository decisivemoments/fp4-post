#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE_ROOT}/deepmath-103k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/native_full_nvfp4_5090}"
MODELS="${MODELS:-qwen2_5_0_5b qwen2_5_math_1_5b}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
WARMUP_STEPS="${WARMUP_STEPS:-10}"
MEASURED_STEPS="${MEASURED_STEPS:-30}"
NUM_CACHE_SAMPLES="${NUM_CACHE_SAMPLES:-64}"
REPEATS="${REPEATS:-3}"
RANK="${RANK:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
STOCHASTIC_GRADIENT_QUANTIZATION="${STOCHASTIC_GRADIENT_QUANTIZATION:-false}"
EXPECTED_GPU_UUID="${EXPECTED_GPU_UUID:-}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

actual_gpu_uuid="$(
    python -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)'
)"
actual_gpu_name="$(
    python -c 'import torch; print(torch.cuda.get_device_name(0))'
)"
echo "Resolved CUDA device 0: ${actual_gpu_name}, UUID=${actual_gpu_uuid}"
if [[ -n "${EXPECTED_GPU_UUID}" ]]; then
    normalized_expected="${EXPECTED_GPU_UUID#GPU-}"
    normalized_actual="${actual_gpu_uuid#GPU-}"
    if [[ "${normalized_actual}" != "${normalized_expected}" ]]; then
        echo "GPU UUID mismatch: expected ${EXPECTED_GPU_UUID}" >&2
        exit 2
    fi
fi

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

append_bool_flag() {
    local value="$1"
    local flag="$2"
    if [[ "${value}" == "true" ]]; then
        printf '%s\n' "${flag}"
    elif [[ "${value}" != "false" ]]; then
        echo "Expected true or false, got: ${value}" >&2
        return 2
    fi
}

mkdir -p "${OUTPUT_ROOT}/cache" "${OUTPUT_ROOT}/results"

gradient_checkpointing_flag="$(
    append_bool_flag "${GRADIENT_CHECKPOINTING}" "--gradient-checkpointing"
)"
stochastic_gradient_flag="$(
    append_bool_flag \
        "${STOCHASTIC_GRADIENT_QUANTIZATION}" \
        "--stochastic-gradient-quantization"
)"

result_files=()
for model_key in ${MODELS}; do
    model_path="$(model_path_for_key "${model_key}")"
    cache_name="${model_key}-deepmath-train-seq${SEQ_LENGTH}-n${NUM_CACHE_SAMPLES}.pt"
    cache_path="${OUTPUT_ROOT}/cache/${cache_name}"

    python benchmarks/step_time/prepare_deepmath.py \
        --dataset-dir "${DATASET_DIR}" \
        --model-path "${model_path}" \
        --output "${cache_path}" \
        --seq-length "${SEQ_LENGTH}" \
        --num-samples "${NUM_CACHE_SAMPLES}"

    for repeat in $(seq 1 "${REPEATS}"); do
        if (( repeat % 2 == 1 )); then
            modes=(bf16 native_full_nvfp4)
        else
            modes=(native_full_nvfp4 bf16)
        fi

        for mode in "${modes[@]}"; do
            suffix="repeat${repeat}-bs${BATCH_SIZE}-seq${SEQ_LENGTH}-gc${GRADIENT_CHECKPOINTING}"
            result_path="${OUTPUT_ROOT}/results/${model_key}-${mode}-${suffix}.json"
            command=(
                python
                benchmarks/native_full_nvfp4_5090/benchmark_train_step.py
                --model-path "${model_path}"
                --data-cache "${cache_path}"
                --mode "${mode}"
                --output "${result_path}"
                --batch-size "${BATCH_SIZE}"
                --seq-length "${SEQ_LENGTH}"
                --warmup-steps "${WARMUP_STEPS}"
                --steps "${MEASURED_STEPS}"
                --rank "${RANK}"
                --repeat "${repeat}"
                --attn-implementation "${ATTN_IMPLEMENTATION}"
            )
            if [[ -n "${gradient_checkpointing_flag}" ]]; then
                command+=("${gradient_checkpointing_flag}")
            fi
            if [[ "${mode}" == "native_full_nvfp4" && -n "${stochastic_gradient_flag}" ]]; then
                command+=("${stochastic_gradient_flag}")
            fi

            echo "Running ${model_key} ${mode} repeat ${repeat}/${REPEATS}"
            "${command[@]}"
            result_files+=("${result_path}")
        done
    done
done

python benchmarks/native_full_nvfp4_5090/summarize_results.py \
    "${result_files[@]}" \
    --markdown-output "${OUTPUT_ROOT}/summary.md" \
    --csv-output "${OUTPUT_ROOT}/summary.csv" \
    --json-output "${OUTPUT_ROOT}/summary.json"
