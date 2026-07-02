# --model_name_or_path outputs/grpo/Qwen2_5-0.5B-after-QAT-fp4 \
# /inspire/ssd/project/pretrain-test/p-shangli/jyzhang/model/qwen2-0.5B-instruct
# /inspire/ssd/project/pretrain-test/p-shangli/jyzhang/model/qwen-2.5-0.5b-instruct
export HF_ENDPOINT=https://hf-mirror.com
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p outputs/grpo

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
accelerate launch --config_file configs/grpo/multi_gpu.yaml src/grpo/grpo.py \
    --dataset_name /inspire/ssd/project/pretrain-test/p-shangli/jyzhang/data/deepmath-103K \
    --model_name_or_path /inspire/ssd/project/pretrain-test/p-shangli/jyzhang/model/qwen-2.5-0.5b-instruct \
    --report_to tensorboard \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --output_dir outputs/grpo/Qwen2_5-0.5B-grpo-longcontext-test-keep-runs \
    --logging_dir outputs/grpo/Qwen2_5-0.5B-grpo-longcontext-test-keep-runs/runs \
    --reward_funcs accuracy_reward \
    --deepspeed configs/grpo/ds_config_zero2_no_offload.json \
    --bf16 true \
    --use_metis false \
    --analyze_rollout false \
    --resume_from_checkpoint true \
    --use_custom_analysis true \
    --max_completion_length 8192
    # --gradient_checkpointing false
