export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0,1,2,3 \
accelerate launch --config_file fsdp.yaml dpo.py \
    --dataset_name /home/jyzhang/download/ultrafeedback_binarized \
    --model_name_or_path /home/jyzhang/download/qwen-2.5-0.5b-instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing False \
    --eval_strategy steps \
    --eval_steps 50 \
    --dtype bfloat16 \
    --output_dir Qwen2_5-0.5B-DPO-bf16 \
    --no_remove_unused_columns
