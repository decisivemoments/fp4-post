#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/outputs/native_full_nvfp4_5090/cache}"
MODELS="${MODELS:-qwen2_5_0_5b qwen2_5_math_1_5b}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
WARMUP_STEPS="${WARMUP_STEPS:-5}"
MEASURED_STEPS="${MEASURED_STEPS:-20}"
REPEATS="${REPEATS:-3}"
RANK="${RANK:-64}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
EXPECTED_GPU_UUID="${EXPECTED_GPU_UUID:-}"
CUDA_GRAPH="${CUDA_GRAPH:-false}"
CUDA_GRAPH_CAPTURE_WARMUP_STEPS="${CUDA_GRAPH_CAPTURE_WARMUP_STEPS:-3}"

if [[ "${CUDA_GRAPH}" == "true" ]]; then
    EXECUTION_TAG="cuda_graph"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/forward_backward_cuda_graph_5090/formal}"
    # RTX 5090 boundaries measured with separate processes. CUDA Graph's
    # private pools reduce the native path's maximum batch.
    QWEN05_PAIRED_BATCH="${QWEN05_PAIRED_BATCH:-19}"
    QWEN05_BF16_MAX_BATCH="${QWEN05_BF16_MAX_BATCH:-28}"
    QWEN15_PAIRED_BATCH="${QWEN15_PAIRED_BATCH:-13}"
    QWEN15_BF16_MAX_BATCH="${QWEN15_BF16_MAX_BATCH:-25}"
elif [[ "${CUDA_GRAPH}" == "false" ]]; then
    EXECUTION_TAG="eager"
    OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/forward_backward_no_optimizer_5090/formal}"
    QWEN05_PAIRED_BATCH="${QWEN05_PAIRED_BATCH:-22}"
    QWEN05_BF16_MAX_BATCH="${QWEN05_BF16_MAX_BATCH:-28}"
    QWEN15_PAIRED_BATCH="${QWEN15_PAIRED_BATCH:-17}"
    QWEN15_BF16_MAX_BATCH="${QWEN15_BF16_MAX_BATCH:-25}"
else
    echo "CUDA_GRAPH must be true or false" >&2
    exit 2
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${OUTPUT_ROOT}/results"

actual_gpu_uuid="$(python -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)')"
actual_gpu_name="$(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
echo "Resolved CUDA device 0: ${actual_gpu_name}, UUID=${actual_gpu_uuid}"
echo "Scope: forward + loss + backward; optimizer is not constructed"
echo "Native method: packed FP4 residual W + packed FP4 V; BF16 U/s"
echo "Execution: ${EXECUTION_TAG}"
if [[ -n "${EXPECTED_GPU_UUID}" ]]; then
    normalized_expected="${EXPECTED_GPU_UUID#GPU-}"
    normalized_actual="${actual_gpu_uuid#GPU-}"
    if [[ "${normalized_actual}" != "${normalized_expected}" ]]; then
        echo "GPU UUID mismatch: expected ${EXPECTED_GPU_UUID}" >&2
        exit 2
    fi
fi

model_config() {
    case "$1" in
        qwen2_5_0_5b)
            MODEL_PATH="${WORKSPACE_ROOT}/qwen2.5-0.5b-instruct"
            DATA_CACHE="${CACHE_ROOT}/qwen2_5_0_5b-deepmath-train-seq512-n64.pt"
            PAIRED_BATCH="${QWEN05_PAIRED_BATCH}"
            BF16_MAX_BATCH="${QWEN05_BF16_MAX_BATCH}"
            ;;
        qwen2_5_math_1_5b)
            MODEL_PATH="${WORKSPACE_ROOT}/qwen2.5-math-1.5b"
            DATA_CACHE="${CACHE_ROOT}/qwen2_5_math_1_5b-deepmath-train-seq512-n64.pt"
            PAIRED_BATCH="${QWEN15_PAIRED_BATCH}"
            BF16_MAX_BATCH="${QWEN15_BF16_MAX_BATCH}"
            ;;
        *)
            echo "Unknown model key: $1" >&2
            return 2
            ;;
    esac
    if [[ ! -f "${DATA_CACHE}" ]]; then
        echo "Missing prepared token cache: ${DATA_CACHE}" >&2
        return 2
    fi
}

run_one() {
    local model_key="$1"
    local mode="$2"
    local batch="$3"
    local repeat="$4"
    local tag="${model_key}-${mode}-${EXECUTION_TAG}-bs${batch}-seq${SEQ_LENGTH}-r${repeat}"
    local output="${OUTPUT_ROOT}/results/${tag}.json"
    local command=(
        python
        benchmarks/native_full_nvfp4_5090/benchmark_train_step.py
        --model-path "${MODEL_PATH}"
        --data-cache "${DATA_CACHE}"
        --mode "${mode}"
        --output "${output}"
        --batch-size "${batch}"
        --seq-length "${SEQ_LENGTH}"
        --warmup-steps "${WARMUP_STEPS}"
        --steps "${MEASURED_STEPS}"
        --rank "${RANK}"
        --repeat "${repeat}"
        --attn-implementation "${ATTN_IMPLEMENTATION}"
        --no-optimizer
    )
    if [[ "${CUDA_GRAPH}" == "true" ]]; then
        command+=(
            --cuda-graph
            --cuda-graph-capture-warmup-steps
            "${CUDA_GRAPH_CAPTURE_WARMUP_STEPS}"
        )
    fi
    if [[ "${GRADIENT_CHECKPOINTING}" == "true" ]]; then
        command+=(--gradient-checkpointing)
    elif [[ "${GRADIENT_CHECKPOINTING}" != "false" ]]; then
        echo "GRADIENT_CHECKPOINTING must be true or false" >&2
        return 2
    fi
    echo "Running ${tag}"
    "${command[@]}"
}

for model_key in ${MODELS}; do
    model_config "${model_key}"
    for repeat in $(seq 1 "${REPEATS}"); do
        if (( repeat % 2 == 1 )); then
            paired_modes=(bf16 native_full_nvfp4)
        else
            paired_modes=(native_full_nvfp4 bf16)
        fi
        for mode in "${paired_modes[@]}"; do
            run_one "${model_key}" "${mode}" "${PAIRED_BATCH}" "${repeat}"
        done
        if (( BF16_MAX_BATCH != PAIRED_BATCH )); then
            run_one \
                "${model_key}" \
                "bf16" \
                "${BF16_MAX_BATCH}" \
                "${repeat}"
        fi
    done
done

python benchmarks/native_full_nvfp4_5090/summarize_forward_backward_no_optimizer.py \
    --results-dir "${OUTPUT_ROOT}/results" \
    --output-dir "${OUTPUT_ROOT}"
