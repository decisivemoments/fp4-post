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
from transformers import AutoModelForCausalLM, GenerationConfig
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
from metis_monitor import MetisDiagnosticCallback, RolloutRewardWrapper, DiagnosticsAnalyzer
from rollout_quality import QualityConfig, QualityAnalyzer

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

        self.metis_mode = "mean"  # 可选 "svd" 或 "mean"

def replace_linear_with_metis(model, dtype, metis_args, target_modules=None, compute_dtype=None):
    """
    递归替换模型中的 nn.Linear 为 BitLinear
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    if compute_dtype is None:
        compute_dtype = torch.float32
    
    for name, module in list(model.named_modules()):  # ← list化防止替换时迭代器失效
        if name.split('.')[-1] not in target_modules:
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        
        # 获取父模块和子模块名
        if '.' in name:
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

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
        
        new_layer.layer_name = name  # ← 赋全名，供 mean_cache key 使用
        new_layer.split()
        setattr(parent, child_name, new_layer)
    
    return model


def replace_model_with_metis(model, metis_args = None):
    """替换指定层为 Metis 实现"""
    
    if metis_args is None:
        metis_args = MetisArgs()
    dtype = torch.bfloat16
    compute_dtype = torch.bfloat16
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



from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def custom_accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[float | None]:
    r"""
    Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If gold is not parseable → try simple answer matching (True/False/Yes/No).
        - If simple matching also fails → return `None` to skip the example.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        solution: (`list[str]`):
            List of the raw-text solutions to the questions/problems/prompts.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].
    """

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution, strict=True):
        try:
            # ── 防御：content / sol 为空或非字符串时直接跳过 ──────────────────
            if not isinstance(content, str) or not isinstance(sol, str):
                rewards.append(0.0)
                continue

            content_stripped = content.strip()
            sol_stripped = sol.strip()

            # ── 首先尝试解析为数学表达式 ──────────────────────────────────────
            gold_parsed = parse(sol_stripped)

            if len(gold_parsed) != 0:
                # 数学表达式场景
                answer_parsed = parse(
                    content_stripped,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(units=True),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                reward = float(verify(gold_parsed, answer_parsed))

            else:
                # ── 非数学表达式场景：在句子末尾找独立词 ──────────────────────
                words = content_stripped.split()

                # 核心修复：空内容时无法提取答案，直接给 0 分（而非崩溃）
                if not words:
                    reward = 0.0
                else:
                    last_word = words[-1].rstrip('.,!?;:').lower()

                    if sol_stripped in ("True", "False", "Yes", "No"):
                        reward = 1.0 if last_word == sol_stripped.lower() else 0.0
                    else:
                        # 既不是数学表达式，也不是简单答案
                        reward = 0.0

        except Exception as e:
            # ── 兜底：任何未预期异常都记录并跳过，不中断训练 ─────────────────
            print(f"[custom_accuracy_reward] Unexpected error — skipping sample. "
                  f"sol={sol!r}, content_preview={content[:80]!r}, error={e}")
            reward = 0.0

        rewards.append(reward)

    return rewards

reward_funcs_registry = {
    "accuracy_reward": custom_accuracy_reward,
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
            pass
        return hook
    
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
    rollout_wrappers = []
    if len(reward_funcs) > 0 and callable(reward_funcs[0]):
        rollout_log_path = os.path.join(training_args.output_dir, "rollout.jsonl")
        quality_cfg = QualityConfig(
            min_char_len=8,
            repeat_2gram_ratio_thresh=0.30,
            mixed_lang_switch_thresh=4,
            enable_repetition=False,
            # 如果任务是纯中文推理，关闭 mixed_lang 检测
            # enable_mixed_lang=False,
        )

        wrapper = RolloutRewardWrapper(
            reward_funcs[0],
            log_path=rollout_log_path,
            quality_cfg=quality_cfg,
        )
        reward_funcs[0] = wrapper
        rollout_wrappers.append(wrapper)
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
        print("use metis, we will replace layer with metis")
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

    diagnostic_cb = MetisDiagnosticCallback(
        model=trainer.model,
        output_dir=training_args.output_dir,
        log_every_steps=4,           # 每 4 步记录一次轻量统计
        rank_check_every_steps=10,    # 每 10 步计算一次有效秩
        saturation_alert_threshold=0.05,
        rollout_wrappers=rollout_wrappers,
    )
    trainer.add_callback(diagnostic_cb)
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

    if trainer.accelerator.is_main_process:
        # analyzer = DiagnosticsAnalyzer(training_args.output_dir)
        # analyzer.print_summary()
        # analyzer.plot_all()
        analyzer = QualityAnalyzer(output_dir / "rollout_quality")
        analyzer.plot_all()

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
    training_args.set_save(strategy="steps", steps=50)
    
    # 自动从模型目录读取 generation_config 并同步到 training_args
    # gen_config_path = os.path.join(model_args.model_name_or_path, "generation_config.json")
    # if os.path.exists(gen_config_path):
    #     gen_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)
        
    #     # 只在用户没有显式覆盖时才同步（检查是否还是 dataclass 默认值）
    #     if training_args.temperature == 1.0 and hasattr(gen_config, "temperature"):
    #         training_args.temperature = gen_config.temperature
    #     if training_args.top_p == 1.0 and hasattr(gen_config, "top_p"):
    #         training_args.top_p = gen_config.top_p
    #     if training_args.top_k == 0 and hasattr(gen_config, "top_k"):
    #         training_args.top_k = gen_config.top_k
    #     if hasattr(gen_config, "repetition_penalty"):
    #         training_args.repetition_penalty = gen_config.repetition_penalty
        
    #     print(f"✅ Synced generation config from model: "
    #         f"temp={training_args.temperature}, "
    #         f"top_p={training_args.top_p}, "
    #         f"top_k={training_args.top_k}")
    if script_args.print_args:
        print_and_save_args(script_args, "Script Arguments", output_dir)
        print_and_save_args(training_args, "Training Arguments", output_dir)
        print_and_save_args(model_args, "Model Arguments", output_dir)
        print_and_save_args(dataset_args, "Dataset Arguments", output_dir)

    main(script_args, training_args, model_args, dataset_args)