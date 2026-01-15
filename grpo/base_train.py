# train_grpo.py
from datasets import load_dataset
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = load_dataset("/home/jyzhang/download/deepmath-103K", split="train")

trainer = GRPOTrainer(
    model="/home/jyzhang/download/qwen-2.5-0.5b-instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()