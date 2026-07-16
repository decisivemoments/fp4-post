#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/grpo/run_experiment.sh qat  <method> <model_key>
  bash scripts/grpo/run_experiment.sh grpo <method> <model_key>

Stages:
  qat      Run bf16-teacher to FP4-student QAT self-distillation.
  grpo     Run GRPO training.

Methods:
  bf16              BF16 GRPO baseline. Valid for stage=grpo.
  direct_fp4        Fixed W-SVD path; activation and grad both use direct FP4. Valid for stage=grpo.
  qat_fp4           QAT checkpoint; activation and grad both use direct FP4.
  moving_mean       Fixed W-SVD path; activation and grad both use moving mean.
  full              QAT distillation, or GRPO from QAT checkpoint; activation and grad both use moving mean.

Model keys:
  qwen2_0_5b        MODEL_ROOT/qwen2-0.5B-instruct
  qwen2_5_0_5b      MODEL_ROOT/qwen-2.5-0.5b-instruct
  qwen2_5_math_1_5b MODEL_ROOT/qwen2.5-math-1.5b-instruct
  qwen2_5_math_1_5b_base MODEL_ROOT/qwen2.5-math-1.5b
  qwen2_5_math_7b   MODEL_ROOT/qwen2.5-math-7b-instruct
  qwen2_5_math_7b_base MODEL_ROOT/qwen2.5-math-7b
  qwen3_1_7b        MODEL_ROOT/qwen3-1.7B
  qwen3_8b          MODEL_ROOT/qwen3-8b
  qwen3_8b_base     MODEL_ROOT/qwen3-8b-base

Useful overrides:
  DATA_ROOT=... MODEL_ROOT=...
  GRPO_OUTPUT_ROOT=outputs/grpo QAT_OUTPUT_ROOT=outputs/qat
  GRPO_MAX_STEPS=2000 QAT_MAX_STEPS=1000 CUDA_VISIBLE_DEVICES=0,1,2,3
  QAT_RESUME_FROM_CHECKPOINT=latest  # or checkpoint-500 / /path/to/checkpoint-500
  QAT_CHECKPOINT=final              # or checkpoint-1000 / 1000
  QAT_SOURCE=/path/to/merged/model  # overrides QAT_CHECKPOINT for grpo qat_fp4/full
  MODEL_ENV_FILE=configs/grpo/model_env/qwen2_0_5b.sh
  DEEPSPEED_CONFIG=configs/grpo/ds_config_zero2.json
  GRPO_GENERATION_USE_CACHE=true
  GRPO_GRADIENT_CHECKPOINTING=true
  METIS_WEIGHT_SVD=true METIS_WEIGHT_SVD_RANK=64 METIS_ACTIVATION_GRAD_RANK=64
  METIS_CACHE_QUANTIZED_WEIGHT=true  # uses extra GPU memory
  METIS_COMPILE_QDQ=true             # compile fused NVFP4 QDQ
  METIS_MERGE_ROLLOUT_WEIGHTS=true   # rollout-only merged W-SVD GEMM
USAGE
}

if [[ $# -lt 3 ]]; then
    usage
    exit 2
fi

STAGE="$1"
METHOD="$2"
MODEL_KEY="$3"

MODEL_ENV_FILE="${MODEL_ENV_FILE:-configs/grpo/model_env/${MODEL_KEY}.sh}"
if [[ -f "${MODEL_ENV_FILE}" ]]; then
    source "${MODEL_ENV_FILE}"
    LOADED_MODEL_ENV="${MODEL_ENV_FILE}"
else
    echo "Warning: model env file not found: ${MODEL_ENV_FILE}; using shared defaults." >&2
    LOADED_MODEL_ENV="shared defaults only"
fi

source configs/grpo/experiment_env.sh

mkdir -p "${GRPO_OUTPUT_ROOT}" "${QAT_OUTPUT_ROOT}"
echo "Loaded model env: ${LOADED_MODEL_ENV}"
echo "Runtime config: deepspeed=${DEEPSPEED_CONFIG}, grpo_bs=${GRPO_BATCH_SIZE}, grpo_accum=${GRPO_GRAD_ACCUM}, generations=${GRPO_NUM_GENERATIONS}, max_completion=${GRPO_MAX_COMPLETION_LENGTH}"
echo "Metis optimization: cache_quantized_weight=${METIS_CACHE_QUANTIZED_WEIGHT}, compile_qdq=${METIS_COMPILE_QDQ}"

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
        *)
            echo "Unknown model key: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
}

BASE_MODEL_PATH="$(model_path_for_key "${MODEL_KEY}")"
QAT_CHECKPOINT="${QAT_CHECKPOINT:-final}"
if [[ "${QAT_CHECKPOINT}" == "final" || "${QAT_CHECKPOINT}" == checkpoint-* ]]; then
    QAT_CHECKPOINT_NAME="${QAT_CHECKPOINT}"
else
    QAT_CHECKPOINT_NAME="checkpoint-${QAT_CHECKPOINT}"
fi
DEFAULT_QAT_RAW_SOURCE="${QAT_OUTPUT_ROOT}/${MODEL_KEY}/full/${QAT_CHECKPOINT_NAME}"
DEFAULT_QAT_MERGED_SOURCE="${QAT_MERGED_SOURCE:-${DEFAULT_QAT_RAW_SOURCE}-merged}"
DEFAULT_QAT_SOURCE="${QAT_SOURCE:-${DEFAULT_QAT_MERGED_SOURCE}}"

METIS_ARGS=()
GRPO_MODEL_PATH="${BASE_MODEL_PATH}"
RUN_NAME="${MODEL_KEY}-${METHOD}"
METIS_WEIGHT_ARGS=(
    --metis_enable_forward_svd "${METIS_WEIGHT_SVD}"
    --metis_forward_svd_rank "${METIS_WEIGHT_SVD_RANK}"
    --metis_cache_quantized_weight "${METIS_CACHE_QUANTIZED_WEIGHT}"
    --metis_compile_qdq "${METIS_COMPILE_QDQ}"
)

case "${METHOD}" in
    bf16)
        if [[ "${STAGE}" != "grpo" ]]; then
            echo "Method bf16 is only valid for stage=grpo." >&2
            exit 2
        fi
        METIS_ARGS=(--use_metis false)
        ;;
    direct_fp4)
        METIS_ARGS=(
            --use_metis true
            --metis_mode mean
            --metis_enable_forward_svd false
            --metis_forward_svd_rank 0
            --metis_cache_quantized_weight "${METIS_CACHE_QUANTIZED_WEIGHT}"
            --metis_compile_qdq "${METIS_COMPILE_QDQ}"
            --metis_enable_activation_svd false
            --metis_enable_backward_svd false
            --metis_activation_lowrank_svd 0
            --metis_backward_lowrank_svd 0
        )
        ;;
    qat_fp4)
        if [[ "${STAGE}" != "grpo" ]]; then
            echo "Method qat_fp4 is only valid for stage=grpo. Use stage=qat full to create the checkpoint." >&2
            exit 2
        fi
        GRPO_MODEL_PATH="${DEFAULT_QAT_SOURCE}"
        METIS_ARGS=(
            --use_metis true
            --metis_mode mean
            "${METIS_WEIGHT_ARGS[@]}"
            --metis_enable_activation_svd false
            --metis_enable_backward_svd false
            --metis_activation_lowrank_svd 0
            --metis_backward_lowrank_svd 0
        )
        ;;
    moving_mean)
        METIS_ARGS=(
            --use_metis true
            --metis_mode mean
            "${METIS_WEIGHT_ARGS[@]}"
            --metis_enable_activation_svd true
            --metis_enable_backward_svd true
            --metis_activation_lowrank_svd "${METIS_ACTIVATION_GRAD_RANK}"
            --metis_backward_lowrank_svd "${METIS_ACTIVATION_GRAD_RANK}"
        )
        ;;
    full)
        if [[ "${STAGE}" == "grpo" ]]; then
            GRPO_MODEL_PATH="${DEFAULT_QAT_SOURCE}"
        fi
        METIS_ARGS=(
            --use_metis true
            --metis_mode mean
            "${METIS_WEIGHT_ARGS[@]}"
            --metis_enable_activation_svd true
            --metis_enable_backward_svd true
            --metis_activation_lowrank_svd "${METIS_ACTIVATION_GRAD_RANK}"
            --metis_backward_lowrank_svd "${METIS_ACTIVATION_GRAD_RANK}"
        )
        ;;
    *)
        echo "Unknown method: ${METHOD}" >&2
        usage >&2
        exit 2
        ;;
esac

COMMON_DATA_ARGS=(
    --dataset_name "${TRAIN_DATASET}"
    --dataset_train_split "${DATASET_TRAIN_SPLIT}"
)

if [[ -n "${DATASET_TEST_SPLIT:-}" ]]; then
    COMMON_DATA_ARGS+=(--dataset_test_split "${DATASET_TEST_SPLIT}")
fi

if [[ "${STAGE}" == "qat" ]]; then
    if [[ "${METHOD}" != "full" ]]; then
        echo "For QAT, run: bash scripts/grpo/run_experiment.sh qat full ${MODEL_KEY}" >&2
        exit 2
    fi

    OUT_DIR="${QAT_OUTPUT_ROOT}/${MODEL_KEY}/${METHOD}"
    TB_DIR="${OUT_DIR}/runs"
    EXTRA_ARGS=()
    if [[ -n "${QAT_MAX_STEPS:-}" ]]; then
        EXTRA_ARGS+=(--max_steps "${QAT_MAX_STEPS}")
    fi
    if [[ "${QAT_RESUME_FROM_CHECKPOINT}" != "false" && -n "${QAT_RESUME_FROM_CHECKPOINT}" ]]; then
        EXTRA_ARGS+=(--resume_from_checkpoint "${QAT_RESUME_FROM_CHECKPOINT}")
    fi

    echo "Running QAT: method=${METHOD} model=${BASE_MODEL_PATH} output=${OUT_DIR}"
    accelerate launch --config_file "${ACCELERATE_DS_CONFIG}" src/grpo/qat_distill.py \
        "${COMMON_DATA_ARGS[@]}" \
        --model_name_or_path "${BASE_MODEL_PATH}" \
        --num_train_epochs "${QAT_EPOCHS}" \
        --per_device_train_batch_size "${QAT_BATCH_SIZE}" \
        --gradient_accumulation_steps "${QAT_GRAD_ACCUM}" \
        --learning_rate "${QAT_LEARNING_RATE}" \
        --max_prompt_length "${QAT_MAX_PROMPT_LENGTH}" \
        --max_new_tokens "${QAT_MAX_NEW_TOKENS}" \
        --save_steps "${QAT_SAVE_STEPS}" \
        --logging_steps "${QAT_LOGGING_STEPS}" \
        --output_dir "${OUT_DIR}" \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --tensorboard_log_dir "${TB_DIR}" \
        --kl_top_k "${QAT_KL_TOP_K}" \
        "${METIS_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
elif [[ "${STAGE}" == "grpo" ]]; then
    OUT_DIR="${GRPO_OUTPUT_ROOT}/${MODEL_KEY}/${METHOD}"
    TB_DIR="${OUT_DIR}/runs"
    EXTRA_ARGS=()
    if [[ -n "${GRPO_MAX_STEPS:-}" ]]; then
        EXTRA_ARGS+=(--max_steps "${GRPO_MAX_STEPS}")
    fi

    if [[ "${METHOD}" == "qat_fp4" || "${METHOD}" == "full" ]]; then
        echo "QAT checkpoint: raw=${DEFAULT_QAT_RAW_SOURCE} merged=${GRPO_MODEL_PATH}"
        if [[ ! -e "${GRPO_MODEL_PATH}" ]]; then
            echo "Merged QAT model not found: ${GRPO_MODEL_PATH}" >&2
            if [[ -z "${QAT_SOURCE:-}" ]]; then
                echo "Create it with:" >&2
                echo "  python tools/convert_bitlinear_to_standard.py ${DEFAULT_QAT_RAW_SOURCE} ${GRPO_MODEL_PATH} --verify" >&2
                echo "Or set QAT_SOURCE=/path/to/merged/model." >&2
            fi
            exit 2
        fi
    fi

    echo "Running GRPO: method=${METHOD} model=${GRPO_MODEL_PATH} output=${OUT_DIR}"
    accelerate launch --config_file "${ACCELERATE_CONFIG}" src/grpo/grpo.py \
        "${COMMON_DATA_ARGS[@]}" \
        --model_name_or_path "${GRPO_MODEL_PATH}" \
        --report_to "${REPORT_TO}" \
        --num_train_epochs "${GRPO_EPOCHS}" \
        --per_device_train_batch_size "${GRPO_BATCH_SIZE}" \
        --gradient_accumulation_steps "${GRPO_GRAD_ACCUM}" \
        --learning_rate "${GRPO_LEARNING_RATE}" \
        --max_prompt_length "${GRPO_MAX_PROMPT_LENGTH}" \
        --max_completion_length "${GRPO_MAX_COMPLETION_LENGTH}" \
        --num_generations "${GRPO_NUM_GENERATIONS}" \
        --output_dir "${OUT_DIR}" \
        --logging_dir "${TB_DIR}" \
        --reward_funcs "${REWARD_FUNCS}" \
        --eval_strategy "${GRPO_EVAL_STRATEGY}" \
        --save_strategy steps \
        --save_steps "${GRPO_SAVE_STEPS}" \
        --deepspeed "${DEEPSPEED_CONFIG}" \
        --bf16 true \
        --gradient_checkpointing "${GRPO_GRADIENT_CHECKPOINTING}" \
        --generation_use_cache "${GRPO_GENERATION_USE_CACHE}" \
        --metis_merge_rollout_weights "${METIS_MERGE_ROLLOUT_WEIGHTS}" \
        --analyze_rollout "${ANALYZE_ROLLOUT}" \
        --resume_from_checkpoint "${GRPO_RESUME}" \
        --use_custom_analysis "${USE_CUSTOM_ANALYSIS}" \
        --logging_steps "${GRPO_LOGGING_STEPS}" \
        "${METIS_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
else
    echo "Unknown stage: ${STAGE}" >&2
    usage >&2
    exit 2
fi
