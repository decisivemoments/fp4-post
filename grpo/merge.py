import torch
from safetensors.torch import load_file, save_file
import os

def merge_with_diag(input_path, output_path):
    print(f"正在加载：{input_path}")
    tensors = load_file(input_path, device="cpu")
    
    # 找出所有包含 ulinear.weight 的层前缀
    layer_prefixes = set()
    
    for key in tensors.keys():
        if "ulinear.weight" in key:
            # 提取基础前缀（去掉 .ulinear.weight）
            base = key.replace(".ulinear.weight", "")
            layer_prefixes.add(base)
    
    print(f"找到 {len(layer_prefixes)} 个需要合并的层")
    
    new_tensors = {}
    keys_to_remove = set()
    
    for base in sorted(layer_prefixes):
        u_key = f"{base}.ulinear.weight"
        s_key = f"{base}.s"
        v_key = f"{base}.vlinear.weight"
        w_key = f"{base}.warmup_linear.weight"
        target_key = f"{base}.weight"
        
        # bias 相关的键
        w_bias_key = f"{base}.warmup_linear.bias"
        target_bias_key = f"{base}.bias"
        
        # 检查所有必要的权重组件是否存在
        if all(key in tensors for key in [u_key, s_key, v_key, w_key]):
            u = tensors[u_key]
            s = tensors[s_key]
            v = tensors[v_key]
            w = tensors[w_key]
            
            dtype = w.dtype
            
            # 计算 USV + Warmup
            diag_s = torch.diag(s)
            merged = torch.matmul(torch.matmul(u, diag_s), v) + w
            
            # 保持原数据类型
            new_tensors[target_key] = merged.to(dtype)
            
            # 标记旧 weight 相关的 key 以便删除
            keys_to_remove.update([u_key, s_key, v_key, w_key])
            
            # 处理 bias
            if w_bias_key in tensors:
                new_tensors[target_bias_key] = tensors[w_bias_key]
                keys_to_remove.add(w_bias_key)
                print(f"已迁移 bias：{target_bias_key} (Shape: {tensors[w_bias_key].shape})")
            
            print(f"已合并：{target_key} (Shape: {merged.shape})")
        else:
            missing = [k for k in [u_key, s_key, v_key, w_key] if k not in tensors]
            print(f"跳过：{base} (缺少组件: {missing})")

    # 构建最终字典：保留未变动的 + 新增的合并权重
    final_dict = {k: v for k, v in tensors.items() if k not in keys_to_remove}
    final_dict.update(new_tensors)
    
    print(f"\n统计信息:")
    print(f"原始键数量: {len(tensors)}")
    print(f"移除键数量: {len(keys_to_remove)}")
    print(f"新增键数量: {len(new_tensors)}")
    print(f"最终键数量: {len(final_dict)}")
    
    print(f"\n保存至：{output_path}")
    save_file(final_dict, output_path)
    print("完成。")

if __name__ == "__main__":
    input_file = "/inspire/ssd/project/pretrain-test/p-shangli/jyzhang/posttrain_pro/trl_run/grpo/Qwen2_5-0.5B-instruct-fp4/checkpoint-300/model.safetensors"
    output_file = input_file.replace(".safetensors", "_merged.safetensors")
    
    if os.path.exists(input_file):
        merge_with_diag(input_file, output_file)
    else:
        print("文件不存在")