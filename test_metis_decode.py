# accelerate launch --num_processes 4 test_momentum_v_decode.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import json
from tqdm import tqdm
from grpo.grpo import replace_model_with_metis
import torch
from torch.utils.data import DataLoader, Dataset

# ==================== 配置参数 ====================
model_name = "/home/jyzhang/download/qwen-2.5-0.5b-instruct"
dataset_path = "/home/jyzhang/download/deepmath-103K"
use_metis = True
batch_size = 1  # 每次处理的prompt数量
num_responses_per_prompt = 8  # 每个prompt生成的回答数量
max_new_tokens = 256
output_file = "qa_results.json"

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

        self.need_rollout = False
        self.momentum_beta = 0.9

# ==================== 加载模型和tokenizer ====================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda"
)

if use_metis:
    model = replace_model_with_metis(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ==================== 自定义Dataset ====================
class PromptDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        prompt = sample.get('prompt')
        
        # 如果prompt是列表格式，提取content
        if isinstance(prompt, list) and len(prompt) > 0:
            if isinstance(prompt[0], dict) and 'content' in prompt[0]:
                prompt = prompt[0]['content']
                prompt = prompt + "\n\nPlease provide your final answer in \\boxed{} format."
        
        return {
            'prompt': prompt,
            'sample_id': idx
        }

# ==================== 加载数据集 ====================
dataset = load_dataset(dataset_path)
if isinstance(dataset, dict):
    dataset = dataset['train']

prompt_dataset = PromptDataset(dataset)

# 创建DataLoader
dataloader = DataLoader(
    prompt_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

# ==================== 生成函数 ====================
def generate_responses(prompts, num_responses=8):
    """为一批prompts生成多个回答"""
    expanded_prompts = []
    for prompt in prompts:
        expanded_prompts.extend([prompt] * num_responses)
    
    messages_batch = []
    for prompt in expanded_prompts:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        messages_batch.append(messages)
    
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in messages_batch
    ]
    
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        padding_side='left',
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    model.eval()
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1,
            top_p=1,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=False
        )
    
    
    
    # 解码responses
    response_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
    del model_inputs
    torch.cuda.empty_cache()

    # 重新组织
    all_responses = []
    for i in range(len(prompts)):
        all_responses.append(responses[i * num_responses:(i + 1) * num_responses])
    
    return all_responses

# ==================== 主处理循环 ====================
processed_count = 0
save_interval = 10  # 每10步保存一次

for batch in tqdm(dataloader, desc="Processing batches"):
    prompts = batch['prompt']
    sample_ids = batch['sample_id']
    
    # 生成回答
    batch_responses = generate_responses(prompts, num_responses=num_responses_per_prompt)
    
    # 立即保存当前batch的结果
    batch_results = []
    for prompt, responses, sample_id in zip(prompts, batch_responses, sample_ids):
        batch_results.append({
            "prompt": prompt,
            "responses": responses,
            "sample_id": sample_id.item() if torch.is_tensor(sample_id) else sample_id
        })
    

    output_file = f"metis_result/results_step{processed_count}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    
    processed_count += len(prompts)

    if processed_count % 100 == 0:
        print(f"Processed {processed_count} prompts")

print(f"\n{'='*50}")
print(f"Total prompts processed: {len(processed_count)}")
print(f"Total responses generated: {len(processed_count) * num_responses_per_prompt}")
print(f"{'='*50}")

