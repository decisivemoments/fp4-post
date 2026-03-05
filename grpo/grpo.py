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

from transformers import AutoModelForCausalLM
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
import matplotlib.pyplot as plt
sys.path.append("..") 
from Metis.Metis import BitLinear

class MetisArgs:
    """Metis 配置参数"""
    def __init__(self):
        self.device = "cuda"
        self.enable_forward_svd = True
        self.enable_lowbit = True
        self.enable_te = False
        
        self.q_forward_input = "nvfp4e2m1bnosr"  
        self.q_forward_weight = "nvfp4e2m1bnosr"
        self.q_backward_input = "nvfp4e2m1b"
        self.q_backward_weight = "nvfp4e2m1b"
        self.q_backward_outputgrad = "nvfp4e2m1b"
        
        self.enable_backward_svd = True
        self.backward_lowrank_svd = 64
        self.backward_lowrank_niter = 2
        
        self.enable_activation_svd = True
        self.activation_lowrank_svd = 64
        self.activation_lowrank_niter = 0
        
        self.activation_broadcast_dim = 0
        self.backward_broadcast_dim = -1
        
        self.enable_nv_recipe = False
        
        self.gradacc_broadcast = False
        self.gradacc_broadcast_steps = 1
        
        self.forward_svd_warmup_steps = 50 #we will split manually after replace nn.linear with bitlinear
        self.forward_svd_rank = 64
        self.tp_simulation = False
        self.tp_parts = 4

        self.need_rollout = True
        self.momentum_beta = 0.9

def replace_linear_with_metis(model, dtype, metis_args, target_modules=None, compute_dtype=None):
    """
    递归替换模型中的 nn.Linear 为 BitLinear
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    if compute_dtype is None:
        compute_dtype = torch.float32
    
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear_with_metis(
                model=module, 
                dtype=dtype, 
                metis_args=metis_args, 
                target_modules=target_modules, 
                compute_dtype=compute_dtype
            )
        
        if name in target_modules and isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            
            new_layer = BitLinear(
                in_features=in_features,
                out_features=out_features,
                args=metis_args,
                bias=has_bias,
                dtype=dtype,
                compute_dtype=compute_dtype
            )
            
            # 🔥 检查参数是否为空（Zero-3 的特征）
            if module.weight.numel() == 0:
                # 只在需要时导入 deepspeed
                import deepspeed
                with deepspeed.zero.GatheredParameters([module.weight], modifier_rank=0):
                    with torch.no_grad():
                        new_layer.warmup_linear.weight.copy_(module.weight)
                        if has_bias:
                            with deepspeed.zero.GatheredParameters([module.bias], modifier_rank=0):
                                new_layer.warmup_linear.bias.copy_(module.bias)
            else:
                with torch.no_grad():
                    new_layer.warmup_linear.weight.copy_(module.weight)
                    if has_bias:
                        new_layer.warmup_linear.bias.copy_(module.bias)
            
            new_layer.split()
            setattr(model, name, new_layer)
    
    return model



def replace_model_with_metis(model, metis_args = None):
    """替换指定层为 Metis 实现"""
    
    if metis_args is None:
        metis_args = MetisArgs()
    dtype = torch.float32
    compute_dtype = torch.float32
    # 替换目标层
    model = replace_linear_with_metis(
        model = model,
        dtype = dtype,
        metis_args = metis_args,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        compute_dtype=compute_dtype
    )
    
    return model

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

    use_custom_analysis: bool = field(
        default=False, # 默认值
        metadata={"help": "Whether to enable custom activation analysis using ActivationCapture."} # 帮助信息
    )

    use_metis: bool = field(
        default=False,
        metadata={"help": "Whether to use Metis to fp4 train."}
    )

    print_args: bool = field(
        default=True,
        metadata={"help": "Whether to print all arguments at the start of training."}
    )


class ActivationCapture:
    def __init__(self, save_dir="activation_analysis", rank=0):
        self.save_dir = save_dir
        self.rank = rank
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储每1步的完整SVD
        self.svd_snapshots = defaultdict(list)
        # 存储原始X矩阵
        self.original_Xs = defaultdict(list)
        
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
            
            # 任务1: 存储原始X矩阵，只存储batch中的第一个
            first_batch_X = input_tensor[0].detach().cpu().float()  # 取第一个batch的序列 (seq, hs)
            self.original_Xs[name].append({
                'step': self.current_step,
                'X': first_batch_X.clone()
            })
            
            # SVD分解，取前64个
            U, S, Vt = torch.linalg.svd(act_flat, full_matrices=False)
            U_64 = U[:, :64]  # (bs*seq, 64)
            S_64 = S[:64]      # (64,)
            V_64 = Vt[:64, :].T  # (hs, 64)
            
            # 任务2: 每步保存完整SVD
            self.svd_snapshots[name].append({
                'step': self.current_step,
                'U': U_64.clone(),
                'S': S_64.clone(),
                'V': V_64.clone()
            })
            
            # 任务3: 每步都进行矩阵分析和绘图
            self.analyze_and_plot_matrices(name, act_flat, U_64, S_64, V_64, self.current_step)
            
            # 每10步保存一次
            if self.current_step % 10 == 0:
                self.save_results()
            
            self.now_monitor_layer += 1
            if self.now_monitor_layer == self.monitor_layer_len:
                self.increment_step()
                self.now_monitor_layer = 0
                
        return hook
    
    def analyze_and_plot_matrices(self, name, original_X, U_k, S_k, V_k, step_idx):
        """
        分析并绘制三个矩阵的数据绝对值分布直方图
        """
        # 设置参数
        k = 16  # 可以修改这个值
        beta = 0.9  # 动量系数
        
        # 获取或初始化动量V
        momentum_key = f"{name}_momentum_v"
        if not hasattr(self, momentum_key):
            setattr(self, momentum_key, V_k[:, :k].clone())
            #说明是第0步，那么也不用执行后面的了
            return
        
        momentum_v = getattr(self, momentum_key)
        
        # 重构前k个奇异向量对应的矩阵 X_k = U_k @ diag(S_k) @ V_k^T
        S_k_diag = torch.diag(S_k[:k])
        U_k_reduced = U_k[:, :k]
        V_k_reduced = V_k[:, :k]
        reduced_X = U_k_reduced @ S_k_diag @ V_k_reduced.T

        # 计算投影矩阵 X_proj = X @ V_momentum @ V_momentum^T
        projected_X = original_X @ momentum_v @ momentum_v.T
        
        # 更新动量V
        current_v_k = V_k[:, :k]
        # 符号对齐
        dot_products = (momentum_v * current_v_k).sum(dim=0)
        sign_correction = torch.sign(dot_products)
        current_v_aligned = current_v_k * sign_correction
        momentum_v = momentum_v * beta + current_v_aligned * (1 - beta)
        momentum_v = torch.nn.functional.normalize(momentum_v, dim=0)
        setattr(self, momentum_key, momentum_v)
        
        # 绘制三个矩阵的绝对值分布直方图
        self.plot_matrix_abs_distribution(original_X, reduced_X, projected_X, name, step_idx, k)
    
    def plot_matrix_abs_distribution(self, original_X, reduced_X, projected_X, name, step_idx, k):
        """
        绘制三个矩阵的数据绝对值分布直方图
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 获取绝对值
        orig_abs = original_X.abs().flatten().numpy()
        reduced_abs = reduced_X.abs().flatten().numpy()
        proj_abs = projected_X.abs().flatten().numpy()
        
        # 过滤掉零值以避免log scale问题
        orig_abs = orig_abs[orig_abs > 0]
        reduced_abs = reduced_abs[reduced_abs > 0]
        proj_abs = proj_abs[proj_abs > 0]
        
        # 绘制直方图 - 第一个矩阵用填充，第二、三个只画线
        counts_orig, bins_orig, _ = ax.hist(orig_abs, bins=100, alpha=0.6, label='Original Matrix |X|', log=True, density=True)
        ax.hist(reduced_abs, bins=bins_orig, alpha=1.0, label=f'Reduced Matrix |X_k| (k={k})', log=True, density=True, 
                histtype='step', linewidth=2)
        ax.hist(proj_abs, bins=bins_orig, alpha=1.0, label=f'Projected Matrix |X*V*V^T|', log=True, density=True, 
                histtype='step', linewidth=2)
        
        ax.set_xlabel('Absolute Value (Log Scale)')
        ax.set_ylabel('Density (Log Scale)')
        ax.set_title(f'Distribution of Absolute Values - Layer: {name}, Step: {step_idx}, Rank: {self.rank}')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        os.makedirs(os.path.join(self.save_dir, "pics"), exist_ok=True)
        safe_name = name.replace('.', '_')
        filename = os.path.join(self.save_dir, "pics", f"matrix_distribution_{safe_name}_step_{step_idx}_rank{self.rank}.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"已保存图片: {filename}")
        plt.close()
    
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
        
        # 保存原始X矩阵
        for layer_name, X_data in self.original_Xs.items():
            if X_data:
                safe_name = layer_name.replace('.', '_')
                save_path = os.path.join(self.save_dir, f"{safe_name}_original_Xs_rank{self.rank}.pt")
                
                # 追加模式：先读取已有数据，再合并保存
                if os.path.exists(save_path):
                    existing_data = torch.load(save_path)
                    existing_data.extend(X_data)
                    torch.save(existing_data, save_path)
                else:
                    torch.save(X_data, save_path)
        
        print(f"💾 Rank {self.rank} saved results at step {self.current_step}")
        
        # 清理已保存的数据
        self.svd_snapshots.clear()
        self.original_Xs.clear()

    
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

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs)
    if script_args.use_metis:
        model = replace_model_with_metis(model)
    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    if script_args.use_custom_analysis:
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
    
    if script_args.print_args:
        print_and_save_args(script_args, "Script Arguments", output_dir)
        print_and_save_args(training_args, "Training Arguments", output_dir)
        print_and_save_args(model_args, "Model Arguments", output_dir)
        print_and_save_args(dataset_args, "Dataset Arguments", output_dir)

    main(script_args, training_args, model_args, dataset_args)