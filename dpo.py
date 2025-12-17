# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

"""
# Full training
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-7 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns
```

# LoRA:
```bash
python trl/scripts/dpo.py \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 1000 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 50 \
    --output_dir Qwen2-0.5B-DPO \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```
"""
import json
from pathlib import Path

import argparse
import os

import torch
from accelerate import logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import (
    DatasetMixtureConfig,
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def main(script_args, training_args, model_args, dataset_args):
    ################
    # Model
    ###################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    print(f"Loading model with dtype: {dtype}")
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    print(f"Model dtype after loading: {model.dtype}")
    print(f"First parameter dtype: {next(model.parameters()).dtype}")
    print(f"FSDP mixed_precision: {training_args.fsdp_config.get('mixed_precision', 'Not set')}")
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load the dataset
    if dataset_args.datasets and script_args.dataset_name:
        logger.warning(
            "Both `datasets` and `dataset_name` are provided. The `datasets` argument will be used to load the "
            "dataset and `dataset_name` will be ignored."
        )
        dataset = get_dataset(dataset_args)
    elif dataset_args.datasets and not script_args.dataset_name:
        dataset = get_dataset(dataset_args)
    elif not dataset_args.datasets and script_args.dataset_name:
        dataset = load_dataset(
            script_args.dataset_name, name=script_args.dataset_config, streaming=script_args.dataset_streaming
        )
    else:
        raise ValueError("Either `datasets` or `dataset_name` must be provided.")

    # Initialize the DPO trainer
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=peft_config,
    )
    # --- 2. 检查 DPOTrainer 初始化后的内部模型精度 ---
    print("\n--- Trainer Initialization Information ---")
    # 获取 trainer 内部模型（可能经过 FSDP 包装）
    internal_model = trainer.model
    try:
        # 打印第一个参数作为整体参考
        first_internal_param_name, first_internal_param = next(internal_model.named_parameters())
        print(f"After DPOTrainer Init - Internal model param dtype: {first_internal_param.dtype} on device: {first_internal_param.device}")
        print(f"After DPOTrainer Init - First internal parameter name: {first_internal_param_name}")

        # 定义你想检查的特定参数名称列表
        target_param_names = [
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]

        # 遍历模型参数，查找并打印目标参数的类型
        print("\nSpecific Linear Layer Parameter dtypes:")
        found_params = set()
        for name, param in internal_model.named_parameters():
            if name in target_param_names:
                print(f"  {name}: {param.dtype}")
                found_params.add(name)
                # 可选：如果找到了所有目标参数，可以提前退出循环
                # if len(found_params) == len(target_param_names):
                #     break
        
        # 检查是否有目标参数没有找到
        missing_params = set(target_param_names) - found_params
        if missing_params:
            print(f"\nWarning: Could not find parameters: {missing_params}")


        print(f"Original model dtype (before trainer init): {model.dtype}") # 这里打印原始模型加载时的 dtype
    except StopIteration:
        print("After DPOTrainer Init - Could not find parameters in the trainer's internal model.")

    # 检查 Accelerator 的混合精度设置
    print(f"After DPOTrainer Init - Accelerator Mixed Precision: {trainer.accelerator.mixed_precision}")
    # print(f"After DPOTrainer Init - Accelerator State: {trainer.accelerator.state}") # 可选，信息可能较多
    print("--- Trainer Initialization Information ---\n")

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (ScriptArguments, DPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("dpo", help="Run the DPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    def print_and_save_args(args, name, output_dir=None):
        print(f"\n{'='*80}")
        print(f"📋 {name}")
        print('='*80)
        
        args_dict = {}
        if hasattr(args, '__dict__'):
            for key, value in vars(args).items():
                # 处理不可序列化的对象
                try:
                    json.dumps(value)
                    args_dict[key] = value
                except (TypeError, ValueError):
                    args_dict[key] = str(value)
                print(f"  {key:40s}: {value}")
        else:
            print(args)
            args_dict = str(args)
        
        # 保存到文件
        if output_dir:
            output_path = Path(output_dir) / f"{name.lower().replace(' ', '_')}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(args_dict, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved to: {output_path}")
        
        print('='*80)
    
    # 获取输出目录
    output_dir = getattr(training_args, 'output_dir', 'Qwen2_5-0.5B-DPO')
    
    print_and_save_args(script_args, "Script Arguments", output_dir)
    print_and_save_args(training_args, "Training Arguments", output_dir)
    print_and_save_args(model_args, "Model Arguments", output_dir)
    print_and_save_args(dataset_args, "Dataset Arguments", output_dir)
    main(script_args, training_args, model_args, dataset_args)