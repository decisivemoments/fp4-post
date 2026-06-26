export HF_ENDPOINT=https://hf-mirror.com
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p outputs/dpo

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file configs/dpo/multi_gpu.yaml src/dpo/dpo.py \
    --dataset_name /home/jyzhang/download/ultrafeedback_binarized \
    --model_name_or_path /home/jyzhang/download/qwen-2.5-0.5b-instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir outputs/dpo/Qwen2_5-0.5B-DPO \
    --no_remove_unused_columns
