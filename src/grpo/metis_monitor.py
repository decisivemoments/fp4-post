# metis_monitor.py
# Metis FP4 训练过程诊断监控模块

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import TrainerCallback
from rollout_quality import (
    QualityConfig, analyze_sample, aggregate_step, QualityLogger, analyze_batch
)
from rollout_analysis import (
    AnalysisConfig, AnalysisLogger, analyze_sample as analyze_sample_logits,
    aggregate_step as aggregate_analysis_step,
)

# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def compute_saturation_rate(tensor: torch.Tensor, fp4_max: float = 6.0) -> float:
    """
    计算 FP4 量化饱和率：有多少比例的值会被 clip 到最大值。
    saturation_rate > 0.05 时需要警惕。
    """
    try:
        abs_max = tensor.abs().max().item()
        if abs_max < 1e-8:
            return 0.0
        scaled = tensor.float() / abs_max * fp4_max
        saturated = (scaled.abs() >= fp4_max * 0.95).float().mean().item()
        return saturated
    except Exception:
        return -1.0


def compute_effective_rank(tensor: torch.Tensor, max_rank: int = 64) -> float:
    """
    计算激活矩阵的有效秩（Roy & Vetterli 2007）。
    erank = exp(H(p))，其中 p_i = sigma_i / sum(sigma)。
    有效秩骤降（< 5）是表示坍塌的预警信号。
    """
    try:
        X = tensor.reshape(-1, tensor.shape[-1]).float()
        if X.shape[0] < 4 or X.shape[1] < 4:
            return -1.0
        q = min(max_rank, min(X.shape) - 1)
        if q < 1:
            return -1.0
        _, S, _ = torch.svd_lowrank(X, q=q, niter=2)
        S = S[S > 1e-6]
        if len(S) == 0:
            return 0.0
        p = S / S.sum()
        entropy = -(p * torch.log(p + 1e-10)).sum()
        return torch.exp(entropy).item()
    except Exception:
        return -1.0


def detect_repetition(text: str, threshold: float = 0.3) -> bool:
    """
    检测文本是否存在大量重复（如 ,strlen,strlen,...）。
    unique_token_ratio < threshold 时判定为重复退化。
    """
    if len(text) < 50:
        return False
    # 按逗号、空格、换行分割
    tokens = re.split(r'[,\s\n]+', text.strip())
    tokens = [t for t in tokens if t]
    if len(tokens) < 10:
        return False
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < threshold


def safe_json_value(v):
    """把不能直接 JSON 序列化的值转成可序列化的形式。"""
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, torch.Tensor):
        return v.item() if v.numel() == 1 else v.tolist()
    return str(v)

class RolloutLogitsCapture:

    def __init__(self):
        self._buffer: Optional[Dict] = None
        self._original_generate = None

    def attach(self, model):
        self._original_generate = model.generate
        capture = self

        def patched_generate(input_ids, **kwargs):
            # 保存原始的 return_dict_in_generate 设置
            original_return_dict = kwargs.get("return_dict_in_generate", False)

            # 强制开启，用于捕获 logits
            kwargs["output_logits"] = True
            kwargs["return_dict_in_generate"] = True
            kwargs.pop("output_scores", None)

            output = capture._original_generate(input_ids, **kwargs)

            # 捕获我们需要的数据
            S_p = input_ids.shape[1]
            rollout_logits = torch.stack(output.logits, dim=1)  # (B, S_r, V)

            capture._buffer = {
                "prompt_ids":     input_ids.cpu(),
                "response_ids":   output.sequences[:, S_p:].cpu(),
                "rollout_logits": rollout_logits.cpu(),
            }

            # ↓ 关键：对外的返回值恢复成调用方期望的格式
            if original_return_dict:
                return output                   # 调用方本来就要 dict，原样返回
            else:
                return output.sequences         # 调用方期望 Tensor，只返回 sequences


        model.generate = patched_generate

    def pop(self) -> Optional[Dict]:
        """取出并清空 buffer。"""
        buf = self._buffer
        self._buffer = None
        return buf

    def detach(self, model):
        if self._original_generate is not None:
            model.generate = self._original_generate
            self._original_generate = None


# ─────────────────────────────────────────────
# Rollout 日志包装器
# ─────────────────────────────────────────────

class RolloutRewardWrapper:
    """
    包装 reward function，每次调用时：
    1. 记录原始 rollout 文本到 rollout.jsonl（保留原有行为）
    2. 调用 rollout_quality 模块做质量检测
    3. 按 step 聚合后写入 QualityLogger
    """

    def __init__(
        self,
        reward_fn,
        log_path: str,
        quality_output_dir: str = None,
        quality_cfg: QualityConfig = None,
        analysis_output_dir: str = None,    # 新增
        analysis_cfg: AnalysisConfig = None, # 新增
        max_text_len: int = 300,
        group_size: Optional[int] = None,
    ):
        self.reward_fn = reward_fn
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_text_len = max_text_len
        self.group_size = int(group_size) if group_size and int(group_size) > 0 else None
        self._step = 0
        self._log_file = open(self.log_path, "a", buffering=1, encoding="utf-8")
        self.__name__ = getattr(reward_fn, '__name__', repr(reward_fn))

        # 质量检测
        self._quality_cfg = quality_cfg or QualityConfig()
        quality_dir = quality_output_dir or str(self.log_path.parent / "rollout_quality")
        self._quality_logger = QualityLogger(quality_dir)
        
        # 分析模块
        self._analysis_cfg = analysis_cfg or AnalysisConfig()
        analysis_dir = analysis_output_dir or str(self.log_path.parent / "rollout_analysis")
        self._analysis_logger = AnalysisLogger(analysis_dir)

        # 暂存 rollout logits（由外部在生成后调用 set_rollout_logits 传入）
        self._pending_rollout_logits: Optional[List[torch.Tensor]] = None
        self._pending_prompt_ids:     Optional[List[torch.Tensor]] = None
        self._pending_response_ids:   Optional[List[torch.Tensor]] = None
        self._model_ref = None   # 由外部调用 set_model 传入

    def set_model(self, model):
        """传入当前 policy model，用于 train full-forward。"""
        self._model_ref = model

    def set_rollout_data(
        self,
        prompt_ids:     List[torch.Tensor],   # 每条样本的 prompt token ids，list of (S_p,)
        response_ids:   List[torch.Tensor],   # 每条样本的 response token ids，list of (S_r,)
        rollout_logits: Optional[List[torch.Tensor]] = None,  # 每条样本的 rollout logits，list of (S_r, V)
    ):
        """
        在 rollout 生成完成后、reward 计算前调用。
        rollout_logits 如果训练框架没有保存可以传 None，
        此时只做 train full-forward 侧的分析（task2），跳过 mismatch（task1）。
        """
        self._pending_prompt_ids     = prompt_ids
        self._pending_response_ids   = response_ids
        self._pending_rollout_logits = rollout_logits

    def __call__(self, completions, **kwargs):
        rewards = self.reward_fn(completions, **kwargs)
        reward_list = rewards if rewards is not None else [None] * len(completions)

        texts = []
        for comp in completions:
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            texts.append(text)

        # 质量检测（批量）
        reward_floats = [
            float(r) if r is not None else None
            for r in reward_list
        ]
        group_ids = None
        # TRL emits candidates prompt-by-prompt for standard GRPO generation.
        # Record this assumption explicitly and decline to synthesize groups if
        # the batch shape is inconsistent, rather than logging misleading data.
        if self.group_size is not None and len(texts) % self.group_size == 0:
            group_ids = [f"{self._step}:{i // self.group_size}" for i in range(len(texts))]

        sample_results = analyze_batch(
            texts=texts,
            step=self._step,
            rewards=reward_floats,
            group_ids=group_ids,
            cfg=self._quality_cfg,
        )

        # ── 合并写入：rollout 原文 + 质量指标，一条记录 ──────────────
        for i, (result, reward) in enumerate(zip(sample_results, reward_list)):
            text = texts[i]
            entry = {
                # ── 基础信息 ──────────────────────────────────────
                "step":        self._step,
                "sample_id":   result["sample_id"],
                "reward":      safe_json_value(reward),

                # ── 质量判定 ──────────────────────────────────────
                "is_bad":      result["is_bad"],
                "bad_types":   result["bad_types"],

                # ── 关键指标（一眼能看出问题在哪）────────────────
                "char_len":              result["metrics"]["char_len"],
                "repeat_2gram_ratio":    result["metrics"]["repeat_2gram_ratio"],
                "max_token_run":         result["metrics"]["max_token_run"],
                "unique_token_ratio":    result["metrics"]["unique_token_ratio"],
                "non_printable_ratio":   result["metrics"]["non_printable_ratio"],
                "newline_ratio":         result["metrics"]["newline_ratio"],
                "mixed_lang_switches":   result["metrics"]["mixed_lang_switches"],
                "special_char_ratio":    result["metrics"]["special_char_ratio"],

                # ── 原始文本（放最后，方便阅读）──────────────────
                "text": text,
            }
            self._log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # step 聚合单独写质量日志
        self._quality_logger.log_samples(sample_results)
        step_agg = aggregate_step(sample_results, texts=texts)
        self._quality_logger.log_step(step_agg)

        # ── 分析模块（task1 + task2）────────────────────────────────
        cfg = self._analysis_cfg
        should_analyze = (
            self._model_ref is not None
            and self._pending_response_ids is not None
            and self._step % cfg.log_every_n_steps == 0
        )

        if should_analyze:
            device = next(self._model_ref.parameters()).device
            n_analyze = min(len(texts), cfg.max_samples_per_step)

            analyses  = []
            bad_flags = []
            rwd_list  = []

            for i in range(n_analyze):
                prompt_ids   = self._pending_prompt_ids[i]
                response_ids = self._pending_response_ids[i]
                ro_logits    = (
                    self._pending_rollout_logits[i]
                    if self._pending_rollout_logits is not None
                    else None
                )

                analysis = analyze_sample_logits(
                    model          = self._model_ref,
                    tokenizer      = None,          # 当前不需要 tokenizer
                    prompt_ids     = prompt_ids,
                    response_ids   = response_ids,
                    rollout_logits = ro_logits,
                    cfg            = cfg,
                    device         = device,
                )
                analyses.append(analysis)

                sr = sample_results[i]
                bad_flags.append(sr["is_bad"])
                rwd_list.append(reward_floats[i])

                self._analysis_logger.log_sample(
                    step      = self._step,
                    sample_id = sr["sample_id"],
                    is_bad    = sr["is_bad"],
                    bad_types = sr["bad_types"],
                    reward    = reward_floats[i],
                    analysis  = analysis,
                )

            step_agg = aggregate_analysis_step(
                step            = self._step,
                sample_analyses = analyses,
                is_bad_flags    = bad_flags,
                rewards         = rwd_list,
            )
            self._analysis_logger.log_step(step_agg)

            # 清空 pending
            self._pending_prompt_ids     = None
            self._pending_response_ids   = None
            self._pending_rollout_logits = None


        return rewards

    def set_step(self, step: int):
        self._step = step

    def close(self):
        self._log_file.close()
        self._quality_logger.close()
        self._analysis_logger.close()



# ─────────────────────────────────────────────
# 主监控 Callback
# ─────────────────────────────────────────────

class MetisDiagnosticCallback(TrainerCallback):
    """
    轻量级 Metis FP4 训练诊断 Callback。

    每 log_every_steps 步记录：
    - 关键层激活统计（mean / std / abs_max / saturation_rate）
    - 关键参数梯度统计（norm / has_nan / has_inf）
    - reward 分布统计（从 logs 中提取）

    每 rank_check_every_steps 步额外记录：
    - 激活有效秩（effective_rank，开销较大）

    所有记录写入 output_dir/metis_diagnostics.jsonl。
    告警（NaN / saturation > 阈值）实时打印到 stdout。
    """

    # 默认监控的层名后缀（适配 Qwen2.5-0.5B，可在构造时覆盖）
    DEFAULT_TARGET_SUFFIXES = [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.mlp.down_proj",
        "layers.11.self_attn.q_proj",
        "layers.11.self_attn.v_proj",
        "layers.11.mlp.down_proj",
        "layers.23.self_attn.q_proj",
        "layers.23.self_attn.v_proj",
        "layers.23.mlp.down_proj",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        output_dir: str,
        log_every_steps: int = 10,
        rank_check_every_steps: int = 50,
        saturation_alert_threshold: float = 0.05,
        target_suffixes: list = None,
        rollout_wrappers: list = None,   # 传入 RolloutRewardWrapper 列表，用于同步 step
        rollout_capture: RolloutLogitsCapture = None
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "metis_diagnostics.jsonl"
        self.alert_path = self.output_dir / "metis_alerts.txt"

        self.log_every = log_every_steps
        self.rank_check_every = rank_check_every_steps
        self.saturation_threshold = saturation_alert_threshold
        self.target_suffixes = target_suffixes or self.DEFAULT_TARGET_SUFFIXES
        self.rollout_wrappers = rollout_wrappers or []

        # 激活缓存：hook 写入，on_step_end 读取后清空
        self._activation_buffer = {}          # name -> stats dict（轻量）
        self._activation_tensor_buffer = {}   # name -> tensor（仅 rank_check 步保留）

        self._alert_lines = []
        self._do_rank_check = False

        self._register_hooks()
        self._capture = rollout_capture

    # ── hook 注册 ──────────────────────────────

    def _register_hooks(self):
        self._hooks = []
        registered = []
        for name, module in self.model.named_modules():
            if any(name.endswith(s) for s in self.target_suffixes):
                h = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(h)
                registered.append(name)
        print(f"[MetisDiagnostic] Registered hooks on {len(registered)} layers:")
        for n in registered:
            print(f"  • {n}")

    def _make_hook(self, name: str):
        def hook(module, input, output):
            try:
                x = input[0].detach()
                stats = {
                    "input_mean":        x.mean().item(),
                    "input_std":         x.std().item(),
                    "input_abs_max":     x.abs().max().item(),
                    "saturation_rate":   compute_saturation_rate(x),
                    "has_nan":           torch.isnan(x).any().item(),
                    "has_inf":           torch.isinf(x).any().item(),
                    "shape":             list(x.shape),
                }
                self._activation_buffer[name] = stats
                if self._do_rank_check:
                    # 只在 rank_check 步保存原始张量（内存开销较大，其他步不保存）
                    self._activation_tensor_buffer[name] = x.cpu()
            except Exception as e:
                self._activation_buffer[name] = {"hook_error": str(e)}
        return hook

    # ── Trainer 事件 ──────────────────────────

    def on_step_end(self, args, state, control, logs=None, **kwargs):
        step = state.global_step

        # 同步 rollout wrapper 的 step 计数
        for w in self.rollout_wrappers:
            w.set_step(step)

        if step % self.log_every != 0:
            self._activation_buffer.clear()
            self._activation_tensor_buffer.clear()
            self._do_rank_check = False
            return

        # 是否本步做 effective rank 计算
        self._do_rank_check = (step % self.rank_check_every == 0)

        record = {
            "step": step,
            "activations": {},
            "grad_stats": {},
            "reward_logs": {},
            "alerts": [],
        }

        # ── 激活统计 ──
        for name, stats in self._activation_buffer.items():
            record["activations"][name] = stats

            # 告警检查
            if stats.get("has_nan"):
                msg = f"[step={step}] 🔥 NaN in activation: {name}"
                self._alert(msg, record)
            if stats.get("has_inf"):
                msg = f"[step={step}] 🔥 Inf in activation: {name}"
                self._alert(msg, record)
            if stats.get("saturation_rate", 0) > self.saturation_threshold:
                msg = (f"[step={step}] ⚠️  High saturation in {name}: "
                       f"{stats['saturation_rate']:.3f} > {self.saturation_threshold}")
                self._alert(msg, record)

        # ── 有效秩（仅 rank_check 步）──
        if self._do_rank_check:
            for name, tensor in self._activation_tensor_buffer.items():
                erank = compute_effective_rank(tensor)
                record["activations"][name]["effective_rank"] = erank
                if 0 < erank < 5:
                    msg = f"[step={step}] ⚠️  Low effective rank in {name}: {erank:.2f}"
                    self._alert(msg, record)

        # ── 梯度统计 ──
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            # 只记录 BitLinear 相关参数
            if not any(k in name for k in ("vlinear", "ulinear", ".s", "warmup_linear")):
                continue
            g = param.grad.detach()
            gstats = {
                "grad_norm":    g.norm().item(),
                "grad_abs_max": g.abs().max().item(),
                "has_nan":      torch.isnan(g).any().item(),
                "has_inf":      torch.isinf(g).any().item(),
            }
            record["grad_stats"][name] = gstats

            if gstats["has_nan"]:
                self._alert(f"[step={step}] 🔥 NaN in grad: {name}", record)
            if gstats["has_inf"]:
                self._alert(f"[step={step}] 🔥 Inf in grad: {name}", record)

        # ── reward logs（从 trainer logs 中提取）──
        if logs:
            for k, v in logs.items():
                record["reward_logs"][k] = safe_json_value(v)

        # ── 写入 jsonl ──
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ── 清空缓存 ──
        self._activation_buffer.clear()
        self._activation_tensor_buffer.clear()
        self._do_rank_check = False
        
        # Logits capture is optional.  Short rollout-quality runs deliberately
        # disable it to avoid retaining one vocabulary-sized tensor per token.
        buf = self._capture.pop() if self._capture is not None else None
        if buf is not None:
            B = buf["prompt_ids"].shape[0]
            for wrapper in self.rollout_wrappers:
                wrapper.set_rollout_data(
                    prompt_ids     = [buf["prompt_ids"][i]     for i in range(B)],
                    response_ids   = [buf["response_ids"][i]   for i in range(B)],
                    rollout_logits = [buf["rollout_logits"][i] for i in range(B)],
                )

    def on_train_end(self, args, state, control, **kwargs):
        # 移除所有 hook
        for h in self._hooks:
            h.remove()
        # 关闭 rollout wrapper 文件句柄
        for w in self.rollout_wrappers:
            w.close()
        # 保存告警汇总
        if self._alert_lines:
            with open(self.alert_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self._alert_lines))
            print(f"[MetisDiagnostic] ⚠️  {len(self._alert_lines)} alerts saved to {self.alert_path}")
        print(f"[MetisDiagnostic] ✅ Diagnostics saved to {self.log_path}")

    # ── 内部工具 ──────────────────────────────

    def _alert(self, msg: str, record: dict):
        print(msg)
        self._alert_lines.append(msg)
        record["alerts"].append(msg)


# ─────────────────────────────────────────────
# 离线分析工具（训练结束后调用）
# ─────────────────────────────────────────────

class DiagnosticsAnalyzer:
    """
    读取 metis_diagnostics.jsonl 和 rollout.jsonl，
    生成可视化图表和统计摘要。

    用法（训练结束后）：
        analyzer = DiagnosticsAnalyzer("output_dir")
        analyzer.plot_all()
        analyzer.print_summary()
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.diag_path = self.output_dir / "metis_diagnostics.jsonl"
        self.rollout_path = self.output_dir / "rollout.jsonl"
        self._diag_records = None
        self._rollout_records = None

    def _load_diag(self):
        if self._diag_records is not None:
            return self._diag_records
        records = []
        if not self.diag_path.exists():
            print(f"[Analyzer] {self.diag_path} not found.")
            return records
        with open(self.diag_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self._diag_records = records
        return records

    def _load_rollout(self):
        if self._rollout_records is not None:
            return self._rollout_records
        records = []
        if not self.rollout_path.exists():
            print(f"[Analyzer] {self.rollout_path} not found.")
            return records
        with open(self.rollout_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self._rollout_records = records
        return records

    def plot_all(self, save_dir: str = None):
        """生成所有诊断图表并保存。"""
        import matplotlib.pyplot as plt

        save_dir = Path(save_dir) if save_dir else self.output_dir / "diagnostic_plots"
        save_dir.mkdir(parents=True, exist_ok=True)

        self._plot_rollout_quality(save_dir, plt)
        self._plot_saturation(save_dir, plt)
        self._plot_effective_rank(save_dir, plt)
        self._plot_grad_norms(save_dir, plt)
        self._plot_activation_stats(save_dir, plt)

        print(f"[Analyzer] 📊 All plots saved to {save_dir}")

    def _plot_rollout_quality(self, save_dir, plt):
        records = self._load_rollout()
        if not records:
            return

        # 按 step 聚合
        step_data = defaultdict(lambda: {
            "rewards": [], "lens": [], "rep_count": 0, "none_count": 0, "total": 0
        })
        for r in records:
            s = r.get("step", 0)
            d = step_data[s]
            d["total"] += 1
            reward = r.get("reward")
            if reward is None:
                d["none_count"] += 1
            else:
                d["rewards"].append(float(reward))
            d["lens"].append(r.get("completion_len", 0))
            if r.get("is_repetitive"):
                d["rep_count"] += 1

        steps = sorted(step_data.keys())
        reward_means = [
            np.mean(step_data[s]["rewards"]) if step_data[s]["rewards"] else np.nan
            for s in steps
        ]
        none_rates = [
            step_data[s]["none_count"] / max(step_data[s]["total"], 1) for s in steps
        ]
        rep_rates = [
            step_data[s]["rep_count"] / max(step_data[s]["total"], 1) for s in steps
        ]
        len_means = [
            np.mean(step_data[s]["lens"]) if step_data[s]["lens"] else np.nan
            for s in steps
        ]

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Rollout Quality over Training Steps", fontsize=14)

        axes[0, 0].plot(steps, reward_means, color="steelblue")
        axes[0, 0].set_title("Reward Mean")
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(steps, none_rates, color="tomato")
        axes[0, 1].set_title("None Rate (reward=None 的比例)")
        axes[0, 1].set_xlabel("Step")
        axes[0, 1].set_ylabel("Rate")
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(steps, rep_rates, color="orange")
        axes[1, 0].set_title("Repetition Rate (重复退化比例)")
        axes[1, 0].set_xlabel("Step")
        axes[1, 0].set_ylabel("Rate")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(steps, len_means, color="green")
        axes[1, 1].set_title("Completion Length Mean")
        axes[1, 1].set_xlabel("Step")
        axes[1, 1].set_ylabel("Chars")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "rollout_quality.png", dpi=150)
        plt.close(fig)

    def _plot_saturation(self, save_dir, plt):
        records = self._load_diag()
        if not records:
            return

        # 收集所有层名
        all_layers = set()
        for r in records:
            all_layers.update(r.get("activations", {}).keys())
        all_layers = sorted(all_layers)

        if not all_layers:
            return

        steps = [r["step"] for r in records]
        fig, ax = plt.subplots(figsize=(14, 5))
        for layer in all_layers:
            vals = [
                r["activations"].get(layer, {}).get("saturation_rate", np.nan)
                for r in records
            ]
            ax.plot(steps, vals, label=layer.split(".")[-3] + "." + layer.split(".")[-1], alpha=0.8)

        ax.axhline(y=0.05, color="red", linestyle="--", label="alert threshold (0.05)")
        ax.set_title("FP4 Saturation Rate per Layer")
        ax.set_xlabel("Step")
        ax.set_ylabel("Saturation Rate")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "saturation_rate.png", dpi=150)
        plt.close(fig)

    def _plot_effective_rank(self, save_dir, plt):
        records = self._load_diag()
        if not records:
            return

        # 只有做了 rank_check 的步才有 effective_rank
        rank_records = [
            r for r in records
            if any(
                "effective_rank" in r.get("activations", {}).get(l, {})
                for l in r.get("activations", {})
            )
        ]
        if not rank_records:
            return

        all_layers = set()
        for r in rank_records:
            for l, stats in r.get("activations", {}).items():
                if "effective_rank" in stats:
                    all_layers.add(l)
        all_layers = sorted(all_layers)

        steps = [r["step"] for r in rank_records]
        fig, ax = plt.subplots(figsize=(14, 5))
        for layer in all_layers:
            vals = [
                r["activations"].get(layer, {}).get("effective_rank", np.nan)
                for r in rank_records
            ]
            ax.plot(steps, vals, label=layer.split(".")[-3] + "." + layer.split(".")[-1], alpha=0.8)

        ax.axhline(y=5, color="red", linestyle="--", label="collapse threshold (5)")
        ax.set_title("Effective Rank per Layer")
        ax.set_xlabel("Step")
        ax.set_ylabel("Effective Rank")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "effective_rank.png", dpi=150)
        plt.close(fig)

    def _plot_grad_norms(self, save_dir, plt):
        records = self._load_diag()
        if not records:
            return

        all_params = set()
        for r in records:
            all_params.update(r.get("grad_stats", {}).keys())
        all_params = sorted(all_params)

        if not all_params:
            return

        steps = [r["step"] for r in records]
        fig, ax = plt.subplots(figsize=(14, 5))
        for param in all_params:
            vals = [
                r["grad_stats"].get(param, {}).get("grad_norm", np.nan)
                for r in records
            ]
            short_name = ".".join(param.split(".")[-3:])
            ax.plot(steps, vals, label=short_name, alpha=0.7)

        ax.set_title("Gradient Norm per BitLinear Parameter")
        ax.set_xlabel("Step")
        ax.set_ylabel("Grad Norm")
        ax.set_yscale("log")
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "grad_norms.png", dpi=150)
        plt.close(fig)

    def _plot_activation_stats(self, save_dir, plt):
        records = self._load_diag()
        if not records:
            return

        all_layers = set()
        for r in records:
            all_layers.update(r.get("activations", {}).keys())
        all_layers = sorted(all_layers)

        if not all_layers:
            return

        steps = [r["step"] for r in records]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Activation Statistics per Layer")

        for layer in all_layers:
            short = layer.split(".")[-3] + "." + layer.split(".")[-1]
            abs_maxs = [
                r["activations"].get(layer, {}).get("input_abs_max", np.nan)
                for r in records
            ]
            stds = [
                r["activations"].get(layer, {}).get("input_std", np.nan)
                for r in records
            ]
            axes[0].plot(steps, abs_maxs, label=short, alpha=0.8)
            axes[1].plot(steps, stds, label=short, alpha=0.8)

        axes[0].set_title("Input Abs Max")
        axes[0].set_xlabel("Step")
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_title("Input Std")
        axes[1].set_xlabel("Step")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "activation_stats.png", dpi=150)
        plt.close(fig)

    def print_summary(self):
        """打印关键指标的文字摘要。"""
        diag = self._load_diag()
        rollout = self._load_rollout()

        print("\n" + "=" * 60)
        print("  Metis Diagnostics Summary")
        print("=" * 60)

        if diag:
            print(f"\n📋 Diagnostic records: {len(diag)} steps logged")
            # 找出所有告警
            all_alerts = []
            for r in diag:
                all_alerts.extend(r.get("alerts", []))
            print(f"⚠️  Total alerts: {len(all_alerts)}")
            if all_alerts:
                print("  First 5 alerts:")
                for a in all_alerts[:5]:
                    print(f"    {a}")

            # 最后一步的 saturation rate
            last = diag[-1]
            print(f"\n📊 Last step ({last['step']}) saturation rates:")
            for layer, stats in last.get("activations", {}).items():
                sr = stats.get("saturation_rate", "N/A")
                short = layer.split(".")[-3] + "." + layer.split(".")[-1]
                print(f"    {short}: {sr:.4f}" if isinstance(sr, float) else f"    {short}: {sr}")

        if rollout:
            print(f"\n📝 Rollout records: {len(rollout)} completions logged")
            none_count = sum(1 for r in rollout if r.get("reward") is None)
            rep_count = sum(1 for r in rollout if r.get("is_repetitive"))
            print(f"  None reward rate:   {none_count / len(rollout):.3f}")
            print(f"  Repetition rate:    {rep_count / len(rollout):.3f}")
            lens = [r.get("completion_len", 0) for r in rollout]
            print(f"  Completion len:     mean={np.mean(lens):.0f}, std={np.std(lens):.0f}")

        print("=" * 60 + "\n")
