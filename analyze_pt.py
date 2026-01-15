import torch
import os
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_pt_file(file_path):
    """
    读取并分析 .pt 文件的结构
    
    Args:
        file_path (str): .pt 文件路径
    """
    print(f"正在分析文件: {file_path}")
    print(f"文件大小: {os.path.getsize(file_path) / (1024**2):.2f} MB")
    
    
    # 加载 .pt 文件
    state_dict = torch.load(file_path, map_location='cpu')
    
    if isinstance(state_dict, dict):
        analyze_state_dict(state_dict)
    elif isinstance(state_dict, list):
        # compute_mean_U_cos_sim(state_dict)
        # compute_mean_V_cos_sim(state_dict)
        # compute_V_cos_sim_with_base(state_dict)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.99)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.9)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.8)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.7)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.6)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.5)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.4)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.3)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.2)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0.1)
        # compute_V_cos_sim_with_momentum_aligned(state_dict, beta=0)
        # compute_V_cos_sim_with_last_V(state_dict)
        k = 16  # 可以修改这个值
        beta = 0.9  # 动量系数
        analyze_and_plot_matrices(state_dict, k, beta)
        # analyze_list(state_dict)
    else:
        print(f"未知的数据类型: {type(state_dict)}")
        print(f"数据内容: {state_dict}")


def analyze_and_plot_matrices(data_list, k, beta):
    """
    分析并绘制三个矩阵的数据绝对值分布直方图
    """
    print(f"\n开始分析矩阵，k={k}, beta={beta}")
    
    # 初始化动量V
    momentum_v = data_list[0]["V"][:, :k].clone()
    
    for step_idx, item in enumerate(data_list):
        if step_idx == 0:
            continue
            
        # 获取当前步的U, S, V
        U = item["U"]
        S = item["S"]
        V = item["V"]
        
        # 重构原始矩阵 X = U @ diag(S) @ V^T
        S_diag = torch.diag(S)
        original_X = U @ S_diag @ V.T
        
        # 重构前k个奇异向量对应的矩阵 X_k = U_k @ diag(S_k) @ V_k^T
        U_k = U[:, :k]
        S_k = S[:k]
        V_k = V[:, :k]
        S_k_diag = torch.diag(S_k)
        reduced_X = U_k @ S_k_diag @ V_k.T
        
        # 更新动量V
        current_v_k = V[:, :k]
        # 符号对齐
        dot_products = (momentum_v * current_v_k).sum(dim=0)
        sign_correction = torch.sign(dot_products)
        current_v_aligned = current_v_k * sign_correction
        momentum_v = momentum_v * beta + current_v_aligned * (1 - beta)
        momentum_v = torch.nn.functional.normalize(momentum_v, dim=0)
        
        # 计算投影矩阵 X_proj = X @ V_momentum @ V_momentum^T
        projected_X = original_X @ momentum_v @ momentum_v.T
        
        # 绘制三个矩阵的绝对值分布直方图
        plot_matrix_abs_distribution(original_X, reduced_X, projected_X, step_idx, k)

def plot_matrix_abs_distribution(original_X, reduced_X, projected_X, step_idx, k):
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
    ax.set_title(f'Distribution of Absolute Values - Step {step_idx}')
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    os.makedirs("./pic", exist_ok=True)
    filename = f"./pic/matrix_distribution_step_{step_idx}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"已保存图片: {filename}")
    plt.close()

def analyze_state_dict(state_dict):
    """
    分析 state_dict 结构
    """
    print(f"\n=== State Dict 分析 ===")
    print(f"总键数量: {len(state_dict)}")
    
    tensor_info = defaultdict(list)
    non_tensor_info = []
    
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            dtype = str(value.dtype)
            shape = list(value.shape)
            size_mb = value.numel() * value.element_size() / (1024**2)
            tensor_info[dtype].append({
                'key': key,
                'shape': shape,
                'size_mb': size_mb,
                'num_params': value.numel()
            })
        else:
            non_tensor_info.append((key, type(value), value))
    
    # 打印张量信息
    total_params = 0
    total_size_mb = 0
    
    print(f"\n--- 张量分析 ---")
    for dtype, tensors in tensor_info.items():
        dtype_total_params = sum(t['num_params'] for t in tensors)
        dtype_total_size = sum(t['size_mb'] for t in tensors)
        total_params += dtype_total_params
        total_size_mb += dtype_total_size
        
        print(f"\n数据类型: {dtype}")
        print(f"  张量数量: {len(tensors)}")
        print(f"  总参数量: {dtype_total_params:,}")
        print(f"  总大小: {dtype_total_size:.2f} MB")
        
        # 显示前几个张量的详细信息
        for i, t in enumerate(tensors[:5]):  # 只显示前5个
            print(f"    {t['key']}: shape={t['shape']}, size={t['size_mb']:.3f}MB")
        if len(tensors) > 5:
            print(f"    ... 还有 {len(tensors)-5} 个张量")
    
    print(f"\n--- 非张量信息 ---")
    for key, data_type, value in non_tensor_info:
        print(f"  {key}: {data_type} = {value}")
    
    print(f"\n=== 总结 ===")
    print(f"总参数量: {total_params:,}")
    print(f"总大小: {total_size_mb:.2f} MB")

def analyze_list(data_list):
    """
    分析列表结构
    """
    print(f"\n=== 列表分析 ===")
    print(f"列表长度: {len(data_list)}")
    
    for i, item in enumerate(data_list):
        if torch.is_tensor(item):
            print(f"  [{i}]: Tensor - shape={list(item.shape)}, dtype={item.dtype}")
        else:
            print(f"  [{i}]: {type(item)} - {item}")


def compute_mean_U_cos_sim(data_list):
    k = 16
    u_cos = torch.zeros(k)
    for i, item in enumerate(data_list):
        if i == 0 or i == 1:
            continue
        u_cos += item["U_cos"].abs()[:k]
    u_cos /= len(data_list) - 2
    print(f"\nMean U_cos similarity (excluding first two items): {u_cos}")
def compute_mean_V_cos_sim(data_list):
    k = 16
    v_cos = torch.zeros(k)
    for i, item in enumerate(data_list):
        if i == 0 or i == 1:
            continue
        v_cos += item["V_cos"].abs()[:k]
    v_cos /= len(data_list) - 2
    print(f"\nMean V_cos similarity (excluding first two items): {v_cos}")

def compute_V_cos_sim_with_base(data_list):
    k = 64
    base_v = torch.zeros(896, k)
    for i, item in enumerate(data_list):
        base_v += item["V"][:, :k]
    base_v /= len(data_list)
    mean_v_cos = torch.zeros(k)
    for i, item in enumerate(data_list):
        v = item["V"]
        cos_sim = torch.nn.functional.cosine_similarity(v[:, :k], base_v, dim=0)
        mean_v_cos += cos_sim.abs()[:k]
    mean_v_cos /= len(data_list)
    print(f"\nMean V cosine similarity with base V: {mean_v_cos}")

def compute_V_cos_sim_with_momentum_aligned(data_list, beta=0.9):
    k = 64
    # 初始化 base_v
    base_v = data_list[0]["V"][:, :k].clone() 
    mean_v_cos = torch.zeros(k)
    
    for i, item in enumerate(data_list):
        if i == 0: continue # 跳过初始化步
        
        current_v = item["V"][:, :k]
        
        # --- 关键修正步骤：符号对齐 ---
        # 计算当前向量与 Base 向量的点积
        # shape: (k,)
        dot_products = (base_v * current_v).sum(dim=0) 
        
        # 如果点积为负，说明方向反了，把 current_v 翻转过来
        sign_correction = torch.sign(dot_products) # 1 or -1
        current_v_aligned = current_v * sign_correction
        # ---------------------------

        # 计算相似度 (用对齐后的)
        cos_sim = torch.nn.functional.cosine_similarity(current_v_aligned, base_v, dim=0)
        mean_v_cos += cos_sim.abs()
        
        # 动量更新 (用对齐后的)
        base_v = base_v * beta + current_v_aligned * (1 - beta)
        
        # 再次正交化 (Renormalization/Orthogonalization)
        # 简单的加权平均会破坏单位模长和正交性，建议至少做一次归一化
        base_v = torch.nn.functional.normalize(base_v, dim=0)
        # 如果追求极致，这里应该做 Gram-Schmidt，但对于 Top-K 归一化通常够了

    mean_v_cos /= (len(data_list) - 1)
    print(f"Beta {beta} with Alignment: {mean_v_cos}")

def compute_V_cos_sim_with_last_V(data_list):
    k = 64
    mean_v_cos = torch.zeros(k)
    last_v = data_list[0]["V"][:, :k].clone()
    for i, item in enumerate(data_list):
        if i == 0:
            continue
        current_v = item["V"][:, :k]
        cos_sim = torch.nn.functional.cosine_similarity(current_v, last_v, dim=0)
        mean_v_cos += cos_sim.abs()[:k]
        last_v = current_v
    mean_v_cos /= (len(data_list) - 1)
    print(f"\nMean V cosine similarity with last V: {mean_v_cos}")


# 使用示例
if __name__ == "__main__":
    # 替换为你的 .pt 文件路径
    # pt_file_path = "/home/jyzhang/pro/trl_run/grpo/activation_analysis/model_layers_0_self_attn_q_proj_deltas_rank0.pt"
    # pt_file_path = "/home/jyzhang/pro/trl_run/grpo/activation_analysis/model_layers_11_self_attn_q_proj_deltas_rank0.pt"
    # pt_file_path = "/home/jyzhang/pro/trl_run/grpo/activation_analysis/model_layers_23_self_attn_q_proj_deltas_rank0.pt"
    # pt_file_path = "/home/jyzhang/pro/trl_run/grpo/activation_analysis/model_layers_0_self_attn_q_proj_snapshots_rank7.pt"
    # pt_file_path = "/home/jyzhang/pro/trl_run/grpo/activation_analysis/model_layers_11_self_attn_q_proj_snapshots_rank7.pt"
    pt_file_path = "/home/jyzhang/pro/trl_run/grpo/activation_analysis/model_layers_23_self_attn_q_proj_snapshots_rank7.pt"
    
    if os.path.exists(pt_file_path):
        analyze_pt_file(pt_file_path)
    else:
        print(f"文件不存在: {pt_file_path}")