# read_safetensors.py
"""
读取 safetensors 文件并显示层级结构
"""
import argparse
from safetensors import safe_open
from collections import defaultdict
import re

def parse_layer_structure(keys):
    """解析参数名，构建层级结构"""
    structure = defaultdict(lambda: defaultdict(list))
    
    for key in keys:
        # 提取层号和模块名
        # 例如: model.layers.0.self_attn.q_proj.weight
        match = re.match(r'model\.layers\.(\d+)\.(\w+)\.(.+)', key)
        if match:
            layer_idx = int(match.group(1))
            module_type = match.group(2)  # self_attn, mlp
            param_path = match.group(3)   # q_proj.weight
            structure[layer_idx][module_type].append(param_path)
        else:
            # 其他参数（embed, norm等）
            structure['other']['other'].append(key)
    
    return structure

def print_tree_structure(structure):
    """以树形结构打印"""
    print("\n" + "="*80)
    print("Model Structure Tree")
    print("="*80)
    
    # 打印非层参数
    if 'other' in structure:
        print("\n📦 Other Parameters:")
        for key in sorted(structure['other']['other']):
            print(f"  └─ {key}")
    
    # 打印各层
    layer_indices = sorted([k for k in structure.keys() if isinstance(k, int)])
    for layer_idx in layer_indices:
        print(f"\n📦 Layer {layer_idx}:")
        for module_type in sorted(structure[layer_idx].keys()):
            print(f"  ├─ {module_type}:")
            params = sorted(structure[layer_idx][module_type])
            for i, param in enumerate(params):
                prefix = "  │  └─" if i == len(params)-1 else "  │  ├─"
                print(f"{prefix} {param}")

def analyze_safetensors(file_path, show_shapes=True, filter_pattern=None):
    """分析 safetensors 文件"""
    print(f"\n🔍 Analyzing: {file_path}\n")
    
    with safe_open(file_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        
        # 过滤
        if filter_pattern:
            keys = [k for k in keys if re.search(filter_pattern, k)]
            print(f"📌 Filtered by pattern: {filter_pattern}")
        
        print(f"📊 Total parameters: {len(keys)}")
        
        # 显示所有参数名和形状
        if show_shapes:
            print("\n" + "="*80)
            print("All Parameters with Shapes")
            print("="*80)
            for key in sorted(keys):
                tensor = f.get_tensor(key)
                print(f"{key:80s} {str(tuple(tensor.shape)):20s} {tensor.dtype}")
        
        # 显示层级结构
        structure = parse_layer_structure(keys)
        print_tree_structure(structure)
        
        # 统计信息
        print("\n" + "="*80)
        print("Statistics")
        print("="*80)
        
        # 按模块类型分组
        module_counts = defaultdict(int)
        for key in keys:
            if 'self_attn' in key:
                if 'q_proj' in key:
                    module_counts['q_proj'] += 1
                elif 'k_proj' in key:
                    module_counts['k_proj'] += 1
                elif 'v_proj' in key:
                    module_counts['v_proj'] += 1
                elif 'o_proj' in key:
                    module_counts['o_proj'] += 1
            elif 'mlp' in key:
                if 'gate_proj' in key:
                    module_counts['gate_proj'] += 1
                elif 'up_proj' in key:
                    module_counts['up_proj'] += 1
                elif 'down_proj' in key:
                    module_counts['down_proj'] += 1
        
        print("\n📈 Module counts:")
        for module, count in sorted(module_counts.items()):
            print(f"  {module:20s}: {count}")
        
        return keys

if __name__ == "__main__":
    # python read_safetensors.py model.safetensors --no-shapes
    parser = argparse.ArgumentParser(description="Read and analyze safetensors file structure")
    parser.add_argument("file_path", type=str, help="Path to the safetensors file")
    parser.add_argument("--no-shapes", action="store_true", help="Don't show tensor shapes")
    parser.add_argument("--filter", type=str, default=None, 
                       help="Regex pattern to filter parameter names (e.g., 'q_proj|k_proj')")
    
    args = parser.parse_args()
    
    analyze_safetensors(args.file_path, show_shapes=not args.no_shapes, filter_pattern=args.filter)
