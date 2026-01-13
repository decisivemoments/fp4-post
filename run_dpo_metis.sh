export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
accelerate launch --config_file multi_gpu.yaml run_dpo_metis.py \
    --dataset_name /inspire/hdd/project/yunweiyuhuifu/p-shangli/zxt/data/ultrafeedback_binarized \
    --model_name_or_path /inspire/hdd/project/yunweiyuhuifu/p-shangli/zxt/qwen2.5-0.5B-instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2_5-0.5B-DPO-metis-fp16-linear32 \
    --no_remove_unused_columns  |& tee -a training_output.log