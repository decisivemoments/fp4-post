#!/usr/bin/env bash
set -euo pipefail

# Native Full-NVFP4 always keeps residual W and V as separate packed FP4
# operands. BitLinear's fake-QDQ rollout merge is intentionally unsupported.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
DATASET_DIR="${DATASET_DIR:-${WORKSPACE_ROOT}/deepmath-103k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/grpo_native_full_nvfp4_5090}"
MODELS="${MODELS:-qwen2_5_0_5b qwen2_5_math_1_5b}"
BATCH_SIZES="${BATCH_SIZES:-32 64 128}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-64}"
MIN_PROMPT_TOKENS="${MIN_PROMPT_TOKENS:-64}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-128}"
NUM_DATA_SAMPLES="${NUM_DATA_SAMPLES:-${DATASET_SIZE:-512}}"
WARMUP_STEPS="${WARMUP_STEPS:-1}"
MEASURED_STEPS="${MEASURED_STEPS:-${MEASURE_STEPS:-3}}"
REPEATS="${REPEATS:-1}"
RANK="${RANK:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
GPU_UUID="${GPU_UUID:-}"
if [[ -n "${GPU_UUID}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
fi
EXPECTED_GPU_UUID="${EXPECTED_GPU_UUID:-${GPU_UUID}}"
CONTINUE_ON_OOM="${CONTINUE_ON_OOM:-true}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_ROOT}/data" "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/runs"

actual_gpu_uuid="$(python -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)')"
actual_gpu_name="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Resolved CUDA device 0: ${actual_gpu_name}, UUID=${actual_gpu_uuid}"
echo "Native method: packed FP4 residual W + packed FP4 V; BF16 U/s; no rollout merge"
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

gradient_checkpointing_flag=()
if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
    gradient_checkpointing_flag=(--gradient-checkpointing)
elif [[ "${GRADIENT_CHECKPOINTING}" != "false" ]]; then
    echo "GRADIENT_CHECKPOINTING must be true or false" >&2
    exit 2
fi

failure_manifest="${OUTPUT_ROOT}/failures.jsonl"
for model_key in ${MODELS}; do
    model_path="$(model_path_for_key "${model_key}")"
    data_path="${OUTPUT_ROOT}/data/${model_key}-deepmath.jsonl"
    python benchmarks/native_full_nvfp4_5090/prepare_grpo_data.py \
        --dataset-dir "${DATASET_DIR}" \
        --model-path "${model_path}" \
        --output "${data_path}" \
        --num-samples "${NUM_DATA_SAMPLES}" \
        --min-prompt-tokens "${MIN_PROMPT_TOKENS}" \
        --max-prompt-tokens "${MAX_PROMPT_TOKENS}"

    for batch_size in ${BATCH_SIZES}; do
        if (( batch_size % NUM_GENERATIONS != 0 )); then
            echo "Skipping batch=${batch_size}: not divisible by generations=${NUM_GENERATIONS}"
            continue
        fi
        if (( batch_size % 16 != 0 )); then
            echo "Skipping batch=${batch_size}: native paired run requires a multiple of 16"
            continue
        fi

        for repeat in $(seq 1 "${REPEATS}"); do
            if (( repeat % 2 == 1 )); then
                modes=(bf16 native_full_nvfp4)
            else
                modes=(native_full_nvfp4 bf16)
            fi
            for mode in "${modes[@]}"; do
                tag="${model_key}-${mode}-bs${batch_size}-g${NUM_GENERATIONS}-c${MAX_COMPLETION_LENGTH}-r${repeat}"
                output="${OUTPUT_ROOT}/results/${tag}.json"
                run_dir="${OUTPUT_ROOT}/runs/${tag}"
                command=(
                    python
                    benchmarks/native_full_nvfp4_5090/benchmark_grpo_step.py
                    --model-path "${model_path}"
                    --dataset-jsonl "${data_path}"
                    --mode "${mode}"
                    --output "${output}"
                    --run-dir "${run_dir}"
                    --batch-size "${batch_size}"
                    --num-generations "${NUM_GENERATIONS}"
                    --max-completion-length "${MAX_COMPLETION_LENGTH}"
                    --warmup-steps "${WARMUP_STEPS}"
                    --measured-steps "${MEASURED_STEPS}"
                    --rank "${RANK}"
                    --seed 2026
                    --repeat "${repeat}"
                    --attn-implementation "${ATTN_IMPLEMENTATION}"
                )
                if [[ -n "${EXPECTED_GPU_UUID}" ]]; then
                    command+=(--expected-gpu-uuid "${EXPECTED_GPU_UUID}")
                fi
                command+=("${gradient_checkpointing_flag[@]}")

                echo "Running ${tag}"
                set +e
                "${command[@]}"
                status=$?
                set -e
                if (( status != 0 )); then
                    printf '{"tag":"%s","exit_code":%d}\n' \
                        "${tag}" "${status}" >> "${failure_manifest}"
                    if [[ "${CONTINUE_ON_OOM}" != "true" ]]; then
                        exit "${status}"
                    fi
                    echo "Run failed (usually capacity/OOM); continuing: ${tag}" >&2
                fi
            done
        done
    done
done

python benchmarks/native_full_nvfp4_5090/summarize_grpo_results.py \
    --results-dir "${OUTPUT_ROOT}/results" \
    --output-dir "${OUTPUT_ROOT}"
