# qat_distill.py
# QAT Distillation: bf16 teacher → fp4 student (Metis)
# Loss = KL(teacher || student) on teacher-generated responses

import os
import sys
import json
import copy
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    get_cosine_schedule_with_warmup,
)
from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_dataset,
)
from torch.utils.tensorboard import SummaryWriter

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from Metis.Metis import BitLinear
from grpo import (   # 复用 grpo.py 里已有的工具函数
    MetisArgs,
    replace_model_with_metis,
    print_and_save_args,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# 参数定义
# ─────────────────────────────────────────────

@dataclass
class QATScriptArguments(ScriptArguments):
    # ── 蒸馏超参 ──────────────────────────────
    num_responses_per_prompt: int = field(
        default=1,
        metadata={"help": "每个 prompt 由 teacher 生成多少条 response 用于蒸馏。"}
    )
    kl_temperature: float = field(
        default=1.0,
        metadata={"help": "KL 蒸馏温度，越大分布越平滑。"}
    )
    max_new_tokens: int = field(
        default=1024,
        metadata={"help": "Teacher 生成 response 的最大 token 数。"}
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Prompt 截断长度。"}
    )

    # ── 训练控制 ──────────────────────────────
    num_train_epochs: int = field(
        default=1,
        metadata={"help": "训练轮数。"}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "最大 optimizer step 数；>0 时优先于 num_train_epochs 提前停止。"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "每卡 batch size（prompt 数）。"}
    )
    gradient_accumulation_steps: int = field(
        default=8,
        metadata={"help": "梯度累积步数。"}
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "学习率。"}
    )
    warmup_ratio: float = field(
        default=0.05,
        metadata={"help": "Warmup 比例。"}
    )
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "梯度裁剪阈值。"}
    )

    # ── 保存 & 日志 ───────────────────────────
    output_dir: str = field(
        default="qat_distill_output",
        metadata={"help": "输出目录。"}
    )
    tensorboard_log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "TensorBoard 日志目录；不传时使用 output_dir/runs。"}
    )
    save_steps: int = field(
        default=100,
        metadata={"help": "每隔多少步保存一次 checkpoint。"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "每隔多少步打印一次 loss。"}
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "QAT checkpoint to resume from. Use a checkpoint path, checkpoint-N, "
                "`latest`, `true`, or leave empty to start fresh."
            )
        },
    )

    # ── Metis ─────────────────────────────────
    use_metis: bool = field(
        default=True,
        metadata={"help": "是否对 student 启用 Metis fp4 替换。"}
    )
    metis_mode: str = field(
        default="mean",
        metadata={"help": "Metis activation residual mode. Supported by BitLinear: `mean` or `svd`."}
    )
    metis_enable_forward_svd: bool = field(
        default=True,
        metadata={"help": "Whether Metis decomposes weights into low-rank plus low-bit residual."}
    )
    metis_enable_activation_svd: bool = field(
        default=True,
        metadata={"help": "Whether Metis applies activation residual quantization. Disable for vanilla direct FP4."}
    )
    metis_enable_backward_svd: bool = field(
        default=True,
        metadata={"help": "Whether Metis applies low-rank/residual quantization to backward output gradients."}
    )
    metis_forward_svd_rank: int = field(
        default=64,
        metadata={"help": "Metis forward weight low-rank rank. Use 0 with forward SVD disabled for direct FP4."}
    )
    metis_activation_lowrank_svd: int = field(
        default=64,
        metadata={"help": "Metis activation low-rank rank used when activation residual quantization is enabled."}
    )
    metis_backward_lowrank_svd: int = field(
        default=64,
        metadata={"help": "Metis backward low-rank rank used when backward residual quantization is enabled."}
    )
    metis_cache_quantized_weight: bool = field(
        default=False,
        metadata={"help": "Cache dequantized weights until the parameter version changes."}
    )
    metis_compile_qdq: bool = field(
        default=False,
        metadata={"help": "Compile the fused NVFP4 quantize-dequantize function with torch.compile."}
    )
    print_args: bool = field(
        default=True,
        metadata={"help": "是否在训练开始时打印所有参数。"}
    )
    
    kl_top_k: int = field(
        default=100,
        metadata={"help": "KL loss 只计算 top-k logits，-1 表示全量计算"}
    )


# ─────────────────────────────────────────────
# 数据 collator
# ─────────────────────────────────────────────

class QATCollator:
    """
    把一个 batch 的 prompt dict 整理成 tokenized 输入。
    支持 conversational（list of messages）和纯文本两种格式。
    """

    def __init__(self, tokenizer, max_prompt_length: int):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length

    def __call__(self, examples: list[dict]) -> dict:
        # 取出原始 prompt 字段（兼容 "prompt" / "messages" 两种 key）
        prompts = []
        for ex in examples:
            if "prompt" in ex:
                prompts.append(ex["prompt"])
            elif "messages" in ex:
                prompts.append(ex["messages"])
            else:
                raise KeyError(f"样本中找不到 'prompt' 或 'messages' 字段，keys={list(ex.keys())}")

        # 判断是否为对话格式
        is_conv = isinstance(prompts[0], list)

        if is_conv:
            encoded = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                tokenize=True,
                padding=True,
                padding_side="left",
                truncation=True,
                max_length=self.max_prompt_length,
                return_tensors="pt",
                return_dict=True,
            )
        else:
            encoded = self.tokenizer(
                prompts,
                padding=True,
                padding_side="left",
                truncation=True,
                max_length=self.max_prompt_length,
                return_tensors="pt",
            )

        return {
            "input_ids":      encoded["input_ids"],       # (B, S_p)
            "attention_mask": encoded["attention_mask"],  # (B, S_p)
        }


QAT_RESUME_STATE_NAME = "qat_resume_state.json"
ACCELERATOR_STATE_DIRNAME = "accelerator_state"


def _checkpoint_step(path: str | os.PathLike) -> int:
    match = re.search(r"checkpoint-(\d+)$", str(path).rstrip("/"))
    return int(match.group(1)) if match else -1


def resolve_qat_resume_checkpoint(resume_from_checkpoint: Optional[str], output_dir: str) -> Optional[str]:
    if resume_from_checkpoint is None:
        return None

    value = str(resume_from_checkpoint).strip()
    if value == "" or value.lower() in {"false", "none", "null"}:
        return None

    if value.lower() in {"true", "latest"}:
        candidates = [
            p for p in Path(output_dir).glob("checkpoint-*")
            if p.is_dir() and _checkpoint_step(p) >= 0
        ]
        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found under {output_dir}")
        return str(max(candidates, key=_checkpoint_step))

    path = Path(value)
    if not path.is_absolute() and not path.exists():
        path = Path(output_dir) / value
    if not path.exists():
        raise FileNotFoundError(f"QAT resume checkpoint not found: {path}")
    return str(path)


def load_qat_model_weights(model: torch.nn.Module, checkpoint_dir: str, accelerator: Accelerator):
    checkpoint = Path(checkpoint_dir)
    safetensor_index = checkpoint / "model.safetensors.index.json"
    bin_index = checkpoint / "pytorch_model.bin.index.json"
    single_safetensor = checkpoint / "model.safetensors"
    single_bin = checkpoint / "pytorch_model.bin"

    def load_weight_file(path: Path):
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            return load_file(str(path), device="cpu")
        return torch.load(str(path), map_location="cpu")

    if safetensor_index.exists() or bin_index.exists():
        index_path = safetensor_index if safetensor_index.exists() else bin_index
        with index_path.open() as f:
            weight_map = json.load(f)["weight_map"]
        for shard_name in sorted(set(weight_map.values())):
            state_dict = load_weight_file(checkpoint / shard_name)
            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.unexpected_keys and accelerator.is_main_process:
                accelerator.print(
                    f"⚠️ Unexpected keys while loading {shard_name}: "
                    f"{len(incompatible.unexpected_keys)}"
                )
            del state_dict
        return

    if single_safetensor.exists():
        state_dict = load_weight_file(single_safetensor)
    elif single_bin.exists():
        state_dict = load_weight_file(single_bin)
    else:
        raise FileNotFoundError(
            f"No model.safetensors/pytorch_model.bin checkpoint weights found in {checkpoint_dir}"
        )

    incompatible = model.load_state_dict(state_dict, strict=False)
    if accelerator.is_main_process and (incompatible.missing_keys or incompatible.unexpected_keys):
        accelerator.print(
            "⚠️ Loaded model-only QAT checkpoint with "
            f"{len(incompatible.missing_keys)} missing and "
            f"{len(incompatible.unexpected_keys)} unexpected keys."
        )


def save_qat_checkpoint(
    accelerator: Accelerator,
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    save_path: str,
    global_step: int,
    epoch: int,
    next_batch_in_epoch: int,
):
    accelerator.print(f"💾 Saving checkpoint to {save_path}")
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        save_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_path)

    accelerator.wait_for_everyone()
    accelerator.save_state(os.path.join(save_path, ACCELERATOR_STATE_DIRNAME))
    if accelerator.is_main_process:
        with open(os.path.join(save_path, QAT_RESUME_STATE_NAME), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "global_step": global_step,
                    "epoch": epoch,
                    "next_batch_in_epoch": next_batch_in_epoch,
                },
                f,
                indent=2,
            )
    accelerator.wait_for_everyone()


# ─────────────────────────────────────────────
# KL Loss
# ─────────────────────────────────────────────

def compute_kl_loss(
    student_logits: torch.Tensor,   # (B, S_r, V)
    teacher_logits: torch.Tensor,   # (B, S_r, V)
    response_mask:  torch.Tensor,   # (B, S_r)  1=有效 token，0=pad
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL(teacher || student)，只在 response 有效 token 上计算均值。

    KL(P||Q) = sum P * (log P - log Q)
    这里 P = teacher（软标签），Q = student（被训练方）。
    """
    T = temperature

    # 缩放后的 log softmax
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)  # (B, S_r, V)
    teacher_probs     = F.softmax(teacher_logits / T, dim=-1)      # (B, S_r, V)

    # 逐 token KL：sum over vocab
    kl_per_token = (teacher_probs * (teacher_probs.log() - student_log_probs)).sum(dim=-1)  # (B, S_r)

    # 只在有效 token 上平均
    masked_kl = (kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)

    # 温度补偿（标准蒸馏做法）
    return masked_kl * (T ** 2)

def compute_kl_loss_topk(
    student_logits: torch.Tensor,         # (B, S_r, V)
    teacher_topk_logits: torch.Tensor,    # (B, S_r, K) - 已经是 logits
    teacher_topk_indices: torch.Tensor,   # (B, S_r, K)
    response_mask: torch.Tensor,          # (B, S_r)
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    使用预计算的 top-k logits 计算 KL 散度
    """
    T = temperature
    B, S_r, K = teacher_topk_logits.shape
    
    # 1. Teacher top-k 的概率分布
    teacher_topk_probs = F.softmax(teacher_topk_logits / T, dim=-1)  # (B, S_r, K)
    
    # 2. 从 student 的全量 logits 中提取对应的 top-k 位置
    student_topk_logits = torch.gather(
        student_logits / T,
        dim=-1,
        index=teacher_topk_indices
    )  # (B, S_r, K)
    
    # 3. Student 在 top-k 上的 log softmax（近似）
    student_topk_log_probs = F.log_softmax(student_topk_logits, dim=-1)
    teacher_topk_log_probs = teacher_topk_probs.log()
    
    # 4. KL(teacher || student)
    kl_per_token = (teacher_topk_probs * (teacher_topk_log_probs - student_topk_log_probs)).sum(dim=-1)
    
    masked_kl = (kl_per_token * response_mask).sum() / response_mask.sum().clamp(min=1)
    return masked_kl * (T ** 2)




# ─────────────────────────────────────────────
# Teacher 生成
# ─────────────────────────────────────────────

@torch.no_grad()
def teacher_generate(
    teacher_model,
    input_ids:      torch.Tensor,   # (B, S_p)
    attention_mask: torch.Tensor,   # (B, S_p)
    num_responses:  int,
    generation_config: GenerationConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    用 teacher 对每个 prompt 生成 num_responses 条 response。

    返回：
        prompt_ids_rep   (B*N, S_p)   重复后的 prompt
        response_ids     (B*N, S_r)   生成的 response token
        response_mask    (B*N, S_r)   有效 token mask（EOS 之后置 0）
    """
    B, S_p = input_ids.shape
    N = num_responses
    
    # 把每个 prompt 重复 N 次
    input_ids_rep      = input_ids.repeat_interleave(N, dim=0)       # (B*N, S_p)
    attention_mask_rep = attention_mask.repeat_interleave(N, dim=0)  # (B*N, S_p)

    output = teacher_model.generate(
        input_ids=input_ids_rep,
        attention_mask=attention_mask_rep,
        generation_config=generation_config,
        return_dict_in_generate=False,   # 只要 sequences Tensor
    )  # (B*N, S_p + S_r)

    response_ids_full = output[:, S_p:]   # (B*N, S_r)

    # 构造 response mask：EOS 之后（含 EOS）全部有效，EOS 之后置 0
    eos_token_id = generation_config.eos_token_id
    if isinstance(eos_token_id, list):
        is_eos = torch.zeros_like(response_ids_full, dtype=torch.bool)
        for eid in eos_token_id:
            is_eos |= (response_ids_full == eid)
    else:
        is_eos = (response_ids_full == eos_token_id)

    S_r = response_ids_full.size(1)
    eos_idx = torch.full((B * N,), S_r, dtype=torch.long, device=response_ids_full.device)
    has_eos = is_eos.any(dim=1)
    eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]

    seq_idx = torch.arange(S_r, device=response_ids_full.device).unsqueeze(0)  # (1, S_r)
    response_mask = (seq_idx <= eos_idx.unsqueeze(1)).long()                    # (B*N, S_r)

    return input_ids_rep, response_ids_full, response_mask


# ─────────────────────────────────────────────
# 主训练逻辑
# ─────────────────────────────────────────────

def main(args: QATScriptArguments, model_args: ModelConfig, dataset_args: DatasetMixtureConfig):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
    )

    os.makedirs(args.output_dir, exist_ok=True)
    resume_checkpoint = resolve_qat_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    resume_has_accelerator_state = False
    resume_state = {}
    if resume_checkpoint is not None:
        resume_state_path = os.path.join(resume_checkpoint, QAT_RESUME_STATE_NAME)
        accelerator_state_path = os.path.join(resume_checkpoint, ACCELERATOR_STATE_DIRNAME)
        resume_has_accelerator_state = os.path.isdir(accelerator_state_path)
        if os.path.exists(resume_state_path):
            with open(resume_state_path, encoding="utf-8") as f:
                resume_state = json.load(f)
        accelerator.print(
            f"🔁 Resuming QAT from {resume_checkpoint} "
            f"({'full accelerator state' if resume_has_accelerator_state else 'model weights only'})"
        )

    if args.print_args and accelerator.is_main_process:
        print_and_save_args(args, "QAT Script Arguments", args.output_dir)
        print_and_save_args(model_args, "Model Arguments", args.output_dir)

    # ── Tokenizer ─────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 加载 bf16 teacher（frozen） ────────────
    accelerator.print("📦 Loading teacher model (bf16, frozen)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        device_map="cpu", 
    )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)

    # ── 加载 fp4 student（Metis） ──────────────
    accelerator.print("📦 Loading student model (fp4 via Metis)...")
    student_model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )
    if args.use_metis:
        accelerator.print("🔧 Replacing student layers with Metis BitLinear (fp4)...")
        metis_args = MetisArgs(
            metis_mode=args.metis_mode,
            enable_forward_svd=args.metis_enable_forward_svd,
            enable_activation_svd=args.metis_enable_activation_svd,
            enable_backward_svd=args.metis_enable_backward_svd,
            forward_svd_rank=args.metis_forward_svd_rank,
            activation_lowrank_svd=args.metis_activation_lowrank_svd,
            backward_lowrank_svd=args.metis_backward_lowrank_svd,
            cache_quantized_weight=args.metis_cache_quantized_weight,
            compile_qdq=args.metis_compile_qdq,
        )
        student_model = replace_model_with_metis(student_model, metis_args)

    if resume_checkpoint is not None and not resume_has_accelerator_state:
        accelerator.print(
            "⚠️ This checkpoint has no accelerator_state; loading model weights only. "
            "Optimizer, scheduler, dataloader position, and RNG state will not be exactly restored."
        )
        load_qat_model_weights(student_model, resume_checkpoint, accelerator)

    student_model.gradient_checkpointing_enable()
    
    # ── Generation config ─────────────────────
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # ── 数据集 ────────────────────────────────
    accelerator.print("📂 Loading dataset...")
    if dataset_args.datasets:
        dataset = get_dataset(dataset_args)
    elif args.dataset_name:
        dataset = load_dataset(args.dataset_name, name=args.dataset_config)
    else:
        raise ValueError("必须提供 datasets 或 dataset_name。")

    train_dataset = dataset[args.dataset_train_split]

    collator = QATCollator(tokenizer, max_prompt_length=args.max_prompt_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # ── Optimizer & Scheduler ─────────────────
    optimizer = torch.optim.AdamW(
        student_model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    
    if accelerator.is_main_process:
        writer = SummaryWriter(log_dir=args.tensorboard_log_dir or f"{args.output_dir}/runs")
    else:
        writer = None

    total_steps = (
        len(train_dataloader)
        * args.num_train_epochs
        // args.gradient_accumulation_steps
    )
    if args.max_steps and args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ── Accelerate 准备 ───────────────────────

    student_model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, scheduler
    )

    global_step = 0
    start_epoch = 0
    skip_batches_in_epoch = 0
    if resume_checkpoint is not None and resume_has_accelerator_state:
        accelerator.load_state(os.path.join(resume_checkpoint, ACCELERATOR_STATE_DIRNAME))
        global_step = int(resume_state.get("global_step", _checkpoint_step(resume_checkpoint)))
        start_epoch = int(resume_state.get("epoch", 0))
        skip_batches_in_epoch = int(resume_state.get("next_batch_in_epoch", 0))
        accelerator.print(
            f"🔁 Restored full QAT state: global_step={global_step}, "
            f"epoch={start_epoch}, skip_batches={skip_batches_in_epoch}"
        )
    elif resume_checkpoint is not None:
        global_step = max(0, _checkpoint_step(resume_checkpoint))
        for _ in range(global_step):
            scheduler.step()
        accelerator.print(
            f"🔁 Continuing from model weights at global_step={global_step}; "
            "optimizer history is fresh."
        )

    # ── 训练循环 ──────────────────────────────
    accelerator.print(f"🚀 Starting QAT distillation — total steps: {total_steps}")

    running_loss = 0.0

    for epoch in range(start_epoch, args.num_train_epochs):
        student_model.train()

        pbar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{args.num_train_epochs}",
            disable=not accelerator.is_main_process  # 只在主进程显示
        )
        # 在 main() 开头添加参数
        top_k = args.kl_top_k if hasattr(args, 'kl_top_k') else -1  # 从配置读取，默认 -1（全量）

        for step, batch in enumerate(pbar):
            if epoch == start_epoch and step < skip_batches_in_epoch:
                continue

            if args.max_steps and args.max_steps > 0 and global_step >= args.max_steps:
                break

            input_ids      = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            # ── Step 1: Teacher 生成 response ─────
            with torch.no_grad():
                teacher_model.to(accelerator.device)
                prompt_ids_rep, response_ids, response_mask = teacher_generate(
                    teacher_model=teacher_model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_responses=args.num_responses_per_prompt,
                    generation_config=generation_config,
                )

                BN, S_p = prompt_ids_rep.shape
                S_r     = response_ids.shape[1]

                full_ids = torch.cat([prompt_ids_rep, response_ids], dim=1)
                full_mask = torch.cat([
                    torch.ones(BN, S_p, dtype=torch.long, device=full_ids.device),
                    response_mask,
                ], dim=1)

                # Teacher full-forward
                teacher_out = teacher_model(input_ids=full_ids, attention_mask=full_mask)
                teacher_logits_full = teacher_out.logits[:, S_p - 1 : S_p + S_r - 1, :]  # (B*N, S_r, V)
                
                # ── 根据 top_k 决定保存全量还是 top-k ──
                if top_k > 0:
                    # 只保存 top-k 的 logits 和 indices
                    teacher_topk_logits, teacher_topk_indices = teacher_logits_full.topk(
                        top_k, dim=-1
                    )  # (B*N, S_r, K), (B*N, S_r, K)
                    teacher_topk_logits = teacher_topk_logits.clone()
                    teacher_topk_indices = teacher_topk_indices.clone()
                    teacher_logits_full = None  # 不保存全量
                else:
                    # 保存全量 logits
                    teacher_logits_full = teacher_logits_full.detach().clone()
                    teacher_topk_logits = None
                    teacher_topk_indices = None
                
                teacher_model.to("cpu")
                del teacher_out
                torch.cuda.empty_cache()

            # ── Step 2: Student full-forward ──────
            with accelerator.accumulate(student_model):
                student_out    = student_model(input_ids=full_ids, attention_mask=full_mask)
                student_logits = student_out.logits[:, S_p - 1 : S_p + S_r - 1, :]
                
                # ── 根据 top_k 选择 loss 函数 ──
                if top_k > 0:
                    loss = compute_kl_loss_topk(
                        student_logits=student_logits,
                        teacher_topk_logits=teacher_topk_logits,
                        teacher_topk_indices=teacher_topk_indices,
                        response_mask=response_mask,
                        temperature=args.kl_temperature,
                    )
                else:
                    loss = compute_kl_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits_full,
                        response_mask=response_mask.float(),
                        temperature=args.kl_temperature,
                    )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(student_model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ── 日志 & 保存 ───────────────────────
            running_loss += loss.detach().item()
            if writer is not None:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/learning_rate', scheduler.get_last_lr()[0], global_step)

            if accelerator.sync_gradients:
                global_step += 1
                lr_now   = scheduler.get_last_lr()[0]
                pbar.set_postfix({  # ← 加这行，实时显示 loss
                    "loss": f"{running_loss:.4f}",
                    "lr": f"{lr_now:.2e}",
                    "step": f"{global_step}/{total_steps}"
                })

                if global_step % args.logging_steps == 0 and accelerator.is_main_process:
                    avg_loss = running_loss / args.logging_steps
                    accelerator.print(
                        f"[epoch {epoch+1} | step {global_step}/{total_steps}] "
                        f"loss={avg_loss:.4f}  lr={lr_now:.2e}"
                    )
                    running_loss = 0.0

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    save_qat_checkpoint(
                        accelerator=accelerator,
                        model=student_model,
                        tokenizer=tokenizer,
                        save_path=save_path,
                        global_step=global_step,
                        epoch=epoch,
                        next_batch_in_epoch=step + 1,
                    )

        skip_batches_in_epoch = 0

        if args.max_steps and args.max_steps > 0 and global_step >= args.max_steps:
            break

    # ── 最终保存 ──────────────────────────────
    accelerator.print("✅ Training complete.")
    final_path = os.path.join(args.output_dir, "final")
    unwrapped = accelerator.unwrap_model(student_model)
    unwrapped.save_pretrained(
        final_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(final_path)
    accelerator.print(f"💾 Final model saved to {final_path}")
    if writer is not None:
        writer.close()


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = TrlParser((QATScriptArguments, ModelConfig, DatasetMixtureConfig))
    args, model_args, dataset_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(args, model_args, dataset_args)
