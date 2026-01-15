# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

import argparse
from collections import defaultdict
import importlib
import os
import sys
from dataclasses import dataclass, field

import torch
from accelerate import logging
from datasets import load_dataset

from trl import (
    DatasetMixtureConfig,
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import accuracy_reward, get_soft_overlong_punishment, reasoning_accuracy_reward, think_format_reward
import json
from pathlib import Path

logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


reward_funcs_registry = {
    "accuracy_reward": accuracy_reward,
    "reasoning_accuracy_reward": reasoning_accuracy_reward,
    "think_format_reward": think_format_reward,
    "get_soft_overlong_punishment": get_soft_overlong_punishment(max_completion_len=1280, soft_punish_cache=256),
}


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str`, *optional*):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
        reward_funcs (`list[str]`, *optional*):
            Reward functions to use. Supported values are:
                - `"accuracy_reward"`
                - `"reasoning_accuracy_reward"`
                - `"think_format_reward"`
                - `"get_soft_overlong_punishment"` (used value are `max_completion_len=1280`, `soft_punish_cache=256`)
                - any dotted import path " (e.g., `'my_lib.rewards.custom_reward'`).
    """

    reward_model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: list[str] | None = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported values are: `accuracy_reward`,  `reasoning_accuracy_reward`, `think_format_reward`, "
            "`get_soft_overlong_punishment` (used values are `max_completion_len=1280`, `soft_punish_cache=256`), or "
            "any dotted import path (e.g., `'my_lib.rewards.custom_reward'`)."
        },
    )


class ActivationCapture:
    def __init__(self, save_dir="activation_analysis", rank=0):
        self.save_dir = save_dir
        self.rank = rank
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储每10步的完整SVD
        self.svd_snapshots = defaultdict(list)
        
        # 存储每步的相似度和差距
        self.svd_deltas = defaultdict(list)
        
        # 上一步的SVD结果
        self.prev_svd = {}
        
        self.hooks = []
        self.current_step = 0
        self.now_monitor_layer = 0
        self.monitor_layer_len = 0
        
    def capture_hook(self, name):
        def hook(module, input, output):
            # 获取input tensor
            if isinstance(input, tuple):
                input_tensor = input[0]
            else:
                input_tensor = input
                
            if not isinstance(input_tensor, torch.Tensor):
                return
            
            # 只处理requires_grad=True的情况
            if not input_tensor.requires_grad:
                return
            
            
            # (bs, seq, hs) -> (bs*seq, hs)
            bs, seq, hs = input_tensor.shape
            act_flat = input_tensor.detach().reshape(-1, hs).cpu().float()
            
            # SVD分解，取前64个
            U, S, Vt = torch.linalg.svd(act_flat, full_matrices=False)
            U_64 = U[:, :64]  # (bs*seq, 64)
            S_64 = S[:64]      # (64,)
            V_64 = Vt[:64, :].T  # (hs, 64)
            
            # 任务1: 每步保存完整SVD
            self.svd_snapshots[name].append({
                'step': self.current_step,
                'U': U_64.clone(),
                'S': S_64.clone(),
                'V': V_64.clone()
            })
            
            # 任务2: 每步计算与上一步的差异
            if name in self.prev_svd:
                prev = self.prev_svd[name]
                
                min_len = min(U_64.shape[0], prev['U'].shape[0])
                U_64_trunc = U_64[:min_len, :]
                prev_U_trunc = prev['U'][:min_len, :]
                # 左奇异向量余弦相似度 (64,)
                U_cos = torch.nn.functional.cosine_similarity(
                    U_64_trunc.T, prev_U_trunc.T, dim=1
                )
                
                # 右奇异向量余弦相似度 (64,)
                V_cos = torch.nn.functional.cosine_similarity(
                    V_64.T, prev['V'].T, dim=1
                )
                
                # 奇异值相对差距 (64,)
                S_rel = (S_64 - prev['S']) / (prev['S'] + 1e-8)
                
                # 奇异值绝对差距 (64,)
                S_abs = S_64 - prev['S']
                
                self.svd_deltas[name].append({
                    'step': self.current_step,
                    'U_cos': U_cos.clone(),
                    'V_cos': V_cos.clone(),
                    'S_rel': S_rel.clone(),
                    'S_abs': S_abs.clone()
                })
            
            # 更新上一步的SVD
            self.prev_svd[name] = {
                'U': U_64.clone(),
                'S': S_64.clone(),
                'V': V_64.clone()
            }
            
            # 每10步保存一次
            if self.current_step % 10 == 0:
                self.save_results()
            
            self.now_monitor_layer += 1
            if self.now_monitor_layer == self.monitor_layer_len:
                self.increment_step()
                self.now_monitor_layer = 0
                
        return hook
    
    def save_results(self):
        """保存结果到文件"""
        # 保存SVD快照
        for layer_name, snapshots in self.svd_snapshots.items():
            if snapshots:
                safe_name = layer_name.replace('.', '_')
                save_path = os.path.join(self.save_dir, f"{safe_name}_snapshots_rank{self.rank}.pt")
                
                # 追加模式：先读取已有数据，再合并保存
                if os.path.exists(save_path):
                    existing_data = torch.load(save_path)
                    existing_data.extend(snapshots)
                    torch.save(existing_data, save_path)
                else:
                    torch.save(snapshots, save_path)
        
        # 保存SVD差异
        for layer_name, deltas in self.svd_deltas.items():
            if deltas:
                safe_name = layer_name.replace('.', '_')
                save_path = os.path.join(self.save_dir, f"{safe_name}_deltas_rank{self.rank}.pt")
                
                # 追加模式：先读取已有数据，再合并保存
                if os.path.exists(save_path):
                    existing_data = torch.load(save_path)
                    existing_data.extend(deltas)
                    torch.save(existing_data, save_path)
                else:
                    torch.save(deltas, save_path)
        
        print(f"💾 Rank {self.rank} saved results at step {self.current_step}")
        
        # 清理已保存的数据
        self.svd_snapshots.clear()
        self.svd_deltas.clear()

    
    def increment_step(self):
        """训练步骤计数"""
        self.current_step += 1
    
    def register_hooks(self, model, target_layers):
        self.monitor_layer_len = len(target_layers)
        for name, module in model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(self.capture_hook(name))
                self.hooks.append(hook)
                print(f"📌 Registered hook: {name}")
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        # 最后保存一次
        self.save_results()

# 指定要监控的层
target_layers = [
    'model.layers.0.self_attn.q_proj',
    'model.layers.11.self_attn.q_proj',
    'model.layers.23.self_attn.q_proj',
]



def main(script_args, training_args, model_args, dataset_args):
    # Get the reward models and functions
    reward_funcs = []
    if script_args.reward_model_name_or_path:
        reward_funcs.append(script_args.reward_model_name_or_path)

    if script_args.reward_funcs:
        for func_name in script_args.reward_funcs:
            if func_name in reward_funcs_registry:
                reward_funcs.append(reward_funcs_registry[func_name])
            elif "." in func_name:
                module_path, func_name = func_name.rsplit(".", 1)
                sys.path.insert(0, os.getcwd())
                module = importlib.import_module(module_path)
                reward_func = getattr(module, func_name)
                reward_funcs.append(reward_func)
            else:
                raise ValueError(
                    f"Could not load reward function '{func_name}'. Expected one of "
                    f"{list(reward_funcs_registry.keys())} or a valid import path."
                )
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)

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

    training_args.model_init_kwargs = model_kwargs

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

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # 获取当前进程的rank
    rank = trainer.accelerator.process_index

    # 初始化时传入rank
    capture = ActivationCapture(save_dir="activation_analysis", rank=rank)
    capture.register_hooks(trainer.model, target_layers)

    # Train the model
    trainer.train()

    # Log training complete
    trainer.accelerator.print("✅ Training completed.")

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)
    trainer.accelerator.print(f"💾 Model saved to {training_args.output_dir}.")

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.accelerator.print(f"🤗 Model pushed to the Hub in https://huggingface.co/{trainer.hub_model_id}.")


def make_parser(subparsers: argparse._SubParsersAction | None = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig, DatasetMixtureConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)
    return parser

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

if __name__ == "__main__":
    parser = make_parser()
    # When using the trl cli, this script may be run with additional arguments, corresponding accelerate arguments.
    # To ensure that their parsing does not interfere with the script arguments, parse the arguments with
    # `return_remaining_strings=True`, then ignore the remaining strings.
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    output_dir = getattr(training_args, 'output_dir', 'Qwen2_5-0.5B-DPO')
    
    print_and_save_args(script_args, "Script Arguments", output_dir)
    print_and_save_args(training_args, "Training Arguments", output_dir)
    print_and_save_args(model_args, "Model Arguments", output_dir)
    print_and_save_args(dataset_args, "Dataset Arguments", output_dir)

    main(script_args, training_args, model_args, dataset_args)