export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --config_file multi_gpu.yaml  grpo.py \
    --dataset_name /inspire/ssd/project/pretrain-test/p-shangli/jyzhang/data/deepmath-103K \
    --model_name_or_path /inspire/ssd/project/pretrain-test/p-shangli/jyzhang/model/qwen-2.5-0.5b-instruct \
    --report_to tensorboard \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --output_dir Qwen2_5-0.5B-grpo \
    --reward_funcs accuracy_reward \
    --deepspeed ds_config_zero2.json \
    --bf16 true \
    --use_metis true
    # --gradient_checkpointing false