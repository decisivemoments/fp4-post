# convert_bitlinear_to_standard.py
"""
将使用 BitLinear 训练的模型参数转换回标准的 Linear 格式
支持两种模式：
1. SVD分解模式 (ulinear + vlinear + s) -> 标准 weight
2. 低秩+残差模式 (ulinear + vlinear + s + warmup_linear) -> 标准 weight
"""
import torch
import argparse
from safetensors.torch import save_file, load_file
from collections import OrderedDict
import re

def convert_model_dtype(state_dict, target_dtype=torch.float32):
    """
    将模型参数统一转换为目标数据类型
    
    Args:
        state_dict: 模型状态字典
        target_dtype: 目标数据类型，默认为 torch.float32
    
    Returns:
        转换后的状态字典
    """
    converted_state_dict = OrderedDict()
    
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # 只转换浮点数张量，保持整数类型的参数不变
            if torch.is_floating_point(value):
                converted_state_dict[key] = value.to(target_dtype)
            else:
                converted_state_dict[key] = value
        else:
            converted_state_dict[key] = value
    
    return converted_state_dict

def reconstruct_weight_from_svd(ulinear_weight, s, vlinear_weight, warmup_weight=None, warmup_bias=None):
    """
    从 SVD 分解重建完整权重
    
    Args:
        ulinear_weight: [out_features, rank]
        s: [rank]
        vlinear_weight: [rank, in_features]
        warmup_weight: [out_features, in_features] (可选，残差部分)
        warmup_bias: [out_features] (可选)
    
    Returns:
        weight: [out_features, in_features]
        bias: [out_features] or None
    """
    # 重建低秩部分: U @ diag(S) @ V
    lowrank_weight = ulinear_weight @ torch.diag(s) @ vlinear_weight
    
    # 如果有残差，加上
    if warmup_weight is not None:
        weight = lowrank_weight + warmup_weight
    else:
        weight = lowrank_weight
    
    return weight, warmup_bias

def convert_bitlinear_checkpoint(input_path, output_path, verbose=True):
    """
    转换 BitLinear checkpoint 到标准格式
    """
    print(f"\n🔄 Converting BitLinear checkpoint...")
    print(f"📥 Input:  {input_path}")
    print(f"📤 Output: {output_path}")
    
    # 加载原始参数
    if input_path.endswith('.safetensors'):
        state_dict = load_file(input_path)
    else:
        checkpoint = torch.load(input_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    print(f"\n📊 Original parameters: {len(state_dict)}")
    
    # 新的 state_dict
    new_state_dict = OrderedDict()
    
    # 找出所有 BitLinear 模块的前缀
    bitlinear_prefixes = set()
    for key in state_dict.keys():
        # 匹配 xxx.ulinear.weight 或 xxx.vlinear.weight
        match = re.match(r'(.+)\.(ulinear|vlinear|warmup_linear)\.', key)
        if match:
            bitlinear_prefixes.add(match.group(1))
    
    print(f"\n🔍 Found {len(bitlinear_prefixes)} BitLinear modules")
    
    processed_keys = set()
    
    # 处理每个 BitLinear 模块
    for prefix in sorted(bitlinear_prefixes):
        if verbose:
            print(f"\n  Processing: {prefix}")
        
        # 收集该模块的所有参数
        ulinear_weight = state_dict.get(f"{prefix}.ulinear.weight")
        vlinear_weight = state_dict.get(f"{prefix}.vlinear.weight")
        s = state_dict.get(f"{prefix}.s")
        warmup_weight = state_dict.get(f"{prefix}.warmup_linear.weight")
        warmup_bias = state_dict.get(f"{prefix}.warmup_linear.bias")
        
        # 标记已处理的 key
        for suffix in ['ulinear.weight', 'vlinear.weight', 's', 
                       'warmup_linear.weight', 'warmup_linear.bias', 'ulinear.bias']:
            key = f"{prefix}.{suffix}"
            if key in state_dict:
                processed_keys.add(key)
        
        # 重建权重
        if ulinear_weight is not None and vlinear_weight is not None and s is not None:
            # SVD 分解模式
            weight, bias = reconstruct_weight_from_svd(
                ulinear_weight, s, vlinear_weight, 
                warmup_weight, warmup_bias
            )
            
            new_state_dict[f"{prefix}.weight"] = weight
            if bias is not None:
                new_state_dict[f"{prefix}.bias"] = bias
            
            if verbose:
                print(f"    ✓ Reconstructed from SVD")
                print(f"      ulinear: {tuple(ulinear_weight.shape)}")
                print(f"      s: {tuple(s.shape)}")
                print(f"      vlinear: {tuple(vlinear_weight.shape)}")
                if warmup_weight is not None:
                    print(f"      warmup: {tuple(warmup_weight.shape)} (residual)")
                print(f"      -> weight: {tuple(weight.shape)}")
        
        elif warmup_weight is not None:
            # 只有 warmup_linear（未 split 的情况）
            new_state_dict[f"{prefix}.weight"] = warmup_weight
            if warmup_bias is not None:
                new_state_dict[f"{prefix}.bias"] = warmup_bias
            
            if verbose:
                print(f"    ✓ Copied from warmup_linear")
                print(f"      -> weight: {tuple(warmup_weight.shape)}")
    
    # 复制其他未处理的参数（embedding, norm等）
    for key, value in state_dict.items():
        if key not in processed_keys:
            new_state_dict[key] = value
    
    print(f"\n📊 Converted parameters: {len(new_state_dict)}")
    
    new_state_dict = convert_model_dtype(new_state_dict, target_dtype = torch.bfloat16)
    
    # 保存
    if output_path.endswith('.safetensors'):
        save_file(new_state_dict, output_path)
    else:
        torch.save({'model_state_dict': new_state_dict}, output_path)
    
    print(f"\n✅ Conversion completed!")
    
    return new_state_dict

def verify_conversion(original_path, converted_path):
    """验证转换是否正确"""
    print("\n🔍 Verifying conversion...")
    
    # 加载两个文件
    if original_path.endswith('.safetensors'):
        original = load_file(original_path)
    else:
        original = torch.load(original_path, map_location='cpu')
        original = original.get('model_state_dict', original)
    
    if converted_path.endswith('.safetensors'):
        converted = load_file(converted_path)
    else:
        converted = torch.load(converted_path, map_location='cpu')
        converted = converted.get('model_state_dict', converted)
    
    # 检查是否所有非 BitLinear 参数都保持不变
    common_keys = set(original.keys()) & set(converted.keys())
    print(f"\n📊 Common parameters: {len(common_keys)}")
    
    all_match = True
    for key in common_keys:
        if not torch.allclose(original[key], converted[key], rtol=1e-5, atol=1e-6):
            print(f"  ⚠️  Mismatch: {key}")
            all_match = False
    
    if all_match:
        print("✅ All common parameters match!")
    else:
        print("⚠️  Some parameters don't match (this is expected for BitLinear modules)")
    
    # 显示参数数量变化
    print(f"\n📊 Parameter count:")
    print(f"  Original:  {len(original)}")
    print(f"  Converted: {len(converted)}")

if __name__ == "__main__":
    # python convert_bitlinear_to_standard.py 
    parser = argparse.ArgumentParser(
        description="Convert BitLinear checkpoint to standard Linear format"
    )
    parser.add_argument("input", type=str, help="Input checkpoint path (.pt or .safetensors)")
    parser.add_argument("output", type=str, help="Output checkpoint path (.pt or .safetensors)")
    parser.add_argument("--verify", action="store_true", help="Verify conversion after completion")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # 转换
    new_state_dict = convert_bitlinear_checkpoint(
        args.input, 
        args.output, 
        verbose=not args.quiet
    )
    
    # 验证
    if args.verify:
        verify_conversion(args.input, args.output)
