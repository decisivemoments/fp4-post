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

from typing import Any, Dict, List
import torch

class PaddedDataCollatorForDPO:
    """自定义 Data Collator for DPO,确保序列长度是 block_size 的倍数"""
    
    def __init__(self, tokenizer, block_size=16):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def _pad_to_multiple(self, tensor: torch.Tensor, target_length: int, pad_value: int) -> torch.Tensor:
        """将 tensor pad 到指定长度"""
        if len(tensor) >= target_length:
            return tensor[:target_length]
        
        padding_length = target_length - len(tensor)
        padding = torch.full((padding_length,), pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding])
    
    def _get_padded_length(self, length: int) -> int:
        """计算 pad 后的长度(向上取整到 block_size 的倍数)"""
        return ((length + self.block_size - 1) // self.block_size) * self.block_size
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 定义需要处理的键组
        key_groups = {
            'prompt': ('prompt_input_ids', 'prompt_attention_mask'),
            'chosen': ('chosen_input_ids', 'chosen_attention_mask'),
            'rejected': ('rejected_input_ids', 'rejected_attention_mask'),
        }
        
        batch = {}
        
        # 处理每组键
        for group_name, (input_ids_key, attention_mask_key) in key_groups.items():
            # 检查第一个 feature 是否包含这些键
            if input_ids_key not in features[0]:
                continue
            
            # 找到这组中最长的序列
            max_length = max(len(feature[input_ids_key]) for feature in features)
            
            # 向上取整到 block_size 的倍数
            padded_length = self._get_padded_length(max_length)
            
            # 收集并 pad 所有样本
            input_ids_list = []
            attention_mask_list = []
            
            for feature in features:
                # 处理 input_ids
                input_ids = feature[input_ids_key]
                if not isinstance(input_ids, torch.Tensor):
                    input_ids = torch.tensor(input_ids, dtype=torch.long)
                
                padded_input_ids = self._pad_to_multiple(input_ids, padded_length, self.pad_token_id)
                input_ids_list.append(padded_input_ids)
                
                # 处理或生成 attention_mask
                if attention_mask_key in feature:
                    attention_mask = feature[attention_mask_key]
                    if not isinstance(attention_mask, torch.Tensor):
                        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                else:
                    # 如果没有 attention_mask,根据 input_ids 生成
                    attention_mask = torch.ones(len(input_ids), dtype=torch.long)
                
                padded_attention_mask = self._pad_to_multiple(attention_mask, padded_length, 0)
                attention_mask_list.append(padded_attention_mask)
            
            # 堆叠成 batch
            batch[input_ids_key] = torch.stack(input_ids_list)
            batch[attention_mask_key] = torch.stack(attention_mask_list)
        
        # 处理其他可能的键(如 labels, pixel_values 等)
        for key in features[0].keys():
            if key not in batch:
                # 对于非序列数据,直接收集
                if isinstance(features[0][key], (int, float, str, bool)):
                    batch[key] = [feature[key] for feature in features]
                elif isinstance(features[0][key], torch.Tensor):
                    batch[key] = torch.stack([feature[key] for feature in features])
                else:
                    batch[key] = [feature[key] for feature in features]
        
        print("\n=== Data Collator Debug ===")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}")
        print(f"Padded to multiple of {self.block_size}")
        print("===========================\n")
        
        return batch


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
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # 创建自定义 data collator
    data_collator = PaddedDataCollatorForDPO(tokenizer, block_size=16)
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
        # data_collator=data_collator,
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
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
    def print_and_save_args(args, name, output_dir=None):

        
        args_dict = {}
        if hasattr(args, '__dict__'):
            for key, value in vars(args).items():
                # 处理不可序列化的对象
                try:
                    json.dumps(value)
                    args_dict[key] = value
                except (TypeError, ValueError):
                    args_dict[key] = str(value)
        else:
            args_dict = str(args)
        
        # 保存到文件
        if output_dir:
            output_path = Path(output_dir) / f"{name.lower().replace(' ', '_')}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(args_dict, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved to: {output_path}")
        
    
    # 获取输出目录
    output_dir = getattr(training_args, 'output_dir', 'Qwen2_5-0.5B-DPO')
    
    print_and_save_args(script_args, "Script Arguments", output_dir)
    print_and_save_args(training_args, "Training Arguments", output_dir)
    print_and_save_args(model_args, "Model Arguments", output_dir)
    print_and_save_args(dataset_args, "Dataset Arguments", output_dir)
    main(script_args, training_args, model_args, dataset_args)