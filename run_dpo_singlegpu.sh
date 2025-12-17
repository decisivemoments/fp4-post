export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=1 \
python  dpo.py \
    --dataset_name /inspire/hdd/project/yunweiyuhuifu/p-shangli/zxt/data/ultrafeedback_binarized \
    --model_name_or_path /inspire/hdd/project/yunweiyuhuifu/p-shangli/zxt/qwen2.5-0.5B-instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --dtype bfloat16 \
    --output_dir Qwen2_5-0.5B-DPO-bf16 \
    --no_remove_unused_columns