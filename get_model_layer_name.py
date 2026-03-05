from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "/home/jyzhang/download/qwen-2.5-0.5b-instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Linear):
        print(name)