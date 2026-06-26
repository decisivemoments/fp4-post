#!/usr/bin/env python3
import sys
import torch
from transformers import AutoModelForCausalLM
import json
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

# 导入原始 DPO 脚本
from dpo import main, make_parser

import argparse
from transformers.models.qwen2 import modeling_qwen2
from Metis.Metis import BitLinear 
import os

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


def replace_linear_with_metis(model, dtype, metis_args, target_modules=None, compute_dtype = None):
    """
    递归替换模型中的 nn.Linear 为 BitLinear
    
    Args:
        model: 要修改的模型
        metis_args: Metis 配置参数
        target_modules: 要替换的模块名称列表,如 ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
    if compute_dtype is None:
        compute_dtype = torch.float32
    
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            # 递归处理子模块
            replace_linear_with_metis(model = module,dtype = dtype, metis_args = metis_args, target_modules = target_modules, compute_dtype=compute_dtype)
        
        # 检查是否是目标层
        if name in target_modules and isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = module.out_features
            has_bias = module.bias is not None
            
            # 创建 BitLinear 替换
            new_layer = BitLinear(
                in_features=in_features,
                out_features=out_features,
                args=metis_args,
                bias=has_bias,
                dtype = dtype,
                compute_dtype=compute_dtype
            )
            
            # 复制原始权重
            with torch.no_grad():
                new_layer.warmup_linear.weight.copy_(module.weight)
                if has_bias:
                    new_layer.warmup_linear.bias.copy_(module.bias)
            
            new_layer.split()
            # 替换模块
            setattr(model, name, new_layer)
    
    return model


def load_model_with_metis(model_name_or_path, metis_args, model_kwargs):
    """加载模型并替换指定层为 Metis 实现"""
    # 先加载原始模型
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    
    # 替换目标层
    model = replace_linear_with_metis(
        model, 
        metis_args,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    return model

# Monkey patch AutoModelForCausalLM.from_pretrained
_original_from_pretrained = AutoModelForCausalLM.from_pretrained

def from_pretrained_with_metis(cls, *args, **kwargs):
    model = _original_from_pretrained(*args, **kwargs)
    
    dtype = torch.float32
    # dtype = kwargs.get("dtype")
    # mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "bf16")
    # if mixed_precision == "bf16":
    #     compute_dtype = torch.bfloat16
    # elif mixed_precision == "fp16":
    #     compute_dtype = torch.float16
    # else:
    #     compute_dtype = torch.float32
    compute_dtype = torch.float32
    # 应用 Metis 替换
    metis_args = MetisArgs()
    model = replace_linear_with_metis(model = model, dtype=dtype , metis_args = metis_args, compute_dtype=compute_dtype)
    
    return model

# 替换方法
AutoModelForCausalLM.from_pretrained = classmethod(from_pretrained_with_metis)

if __name__ == "__main__":
    parser = make_parser()
    os.environ["ACCELERATE_MIXED_PRECISION"] = "fp16"
    script_args, training_args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    training_args.fp16 = True
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
    output_dir = getattr(training_args, 'output_dir', 'outputs/dpo/Qwen2_5-0.5B-DPO')
    print_and_save_args(script_args, "Script Arguments", output_dir)
    print_and_save_args(training_args, "Training Arguments", output_dir)
    print_and_save_args(model_args, "Model Arguments", output_dir)
    print_and_save_args(dataset_args, "Dataset Arguments", output_dir)
    
    main(script_args, training_args, model_args, dataset_args)
