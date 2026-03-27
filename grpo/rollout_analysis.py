# rollout_analysis.py
# 观测两件事：
#   1. rollout 生成时的 token 概率 vs 训练时 full forward 的 token 概率（mismatch）
#   2. 每个 token 位置的 ppl / entropy / top-k overlap（退化定位）

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

@dataclass
class AnalysisConfig:
    topk: int = 5                        # top-k overlap 的 k
    max_new_tokens: int = 1024            # 最多分析前多少个 response token
    log_every_n_steps: int = 1           # 每隔多少 step 做一次分析（省显存）
    max_samples_per_step: int = 128        # 每 step 最多分析几条（太多会慢）
    compute_mismatch: bool = True        # 是否做 task1（mismatch）
    compute_token_stats: bool = True     # 是否做 task2（ppl/entropy/topk）


# ─────────────────────────────────────────────────────────────
# 核心计算
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def compute_token_stats_from_logits(
    logits: torch.Tensor,           # (T, V)，已经是 response 部分的 logits
    response_ids: torch.Tensor,     # (T,)，response token ids
    topk: int = 5,
) -> Dict[str, List]:
    """
    给定 response 部分的 logits 和对应的 token ids，
    计算每个位置的 nll / ppl / entropy / top-k 统计。

    logits[t] 是预测第 t 个 response token 的分布，
    response_ids[t] 是实际生成的第 t 个 token。

    返回 dict，每个 key 对应一个长度为 T 的列表。
    """
    T = logits.shape[0]
    assert response_ids.shape[0] == T, \
        f"logits T={T} vs response_ids T={response_ids.shape[0]}"

    log_probs = F.log_softmax(logits.float(), dim=-1)   # (T, V)
    probs     = log_probs.exp()                          # (T, V)

    # 每个位置实际 token 的 log prob
    token_logp = log_probs[
        torch.arange(T), response_ids
    ].tolist()                                           # list[float], len=T

    # entropy: -sum(p * log p)
    entropy = (-(probs * log_probs).sum(dim=-1)).tolist()   # list[float]

    # top-k
    topk_vals, topk_ids = probs.topk(topk, dim=-1)      # (T, k)
    top1_prob   = topk_vals[:, 0].tolist()
    topk_mass   = topk_vals.sum(dim=-1).tolist()
    topk_ids_list = topk_ids.tolist()                    # list[list[int]]

    # 是否实际 token 在 top-k 里
    actual_in_topk = [
        int(response_ids[t].item() in topk_ids_list[t])
        for t in range(T)
    ]

    return {
        "token_logp":      token_logp,        # log p(y_t)
        "token_nll":       [-v for v in token_logp],
        "token_ppl":       [math.exp(-v) for v in token_logp],
        "entropy":         entropy,
        "top1_prob":       top1_prob,
        "topk_mass":       topk_mass,
        "topk_ids":        topk_ids_list,
        "actual_in_topk":  actual_in_topk,    # 0/1，用于 topk_overlap 统计
    }


def aggregate_token_stats(token_stats: Dict[str, List]) -> Dict[str, float]:
    """
    把 per-token 统计聚合成句子级标量。
    """
    nll   = token_stats["token_nll"]
    ppl   = token_stats["token_ppl"]
    ent   = token_stats["entropy"]
    top1  = token_stats["top1_prob"]
    topkm = token_stats["topk_mass"]
    in_tk = token_stats["actual_in_topk"]
    T     = len(nll)

    def mean(lst): return sum(lst) / len(lst) if lst else 0.0
    def maxv(lst): return max(lst) if lst else 0.0

    return {
        "seq_len":           T,
        "mean_token_ppl":    mean(ppl),
        "max_token_ppl":     maxv(ppl),
        "mean_nll":          mean(nll),
        "ppl":               math.exp(mean(nll)),   # sentence-level PPL
        "mean_entropy":      mean(ent),
        "max_entropy":       maxv(ent),
        "min_top1_prob":     min(top1) if top1 else 0.0,
        "mean_top1_prob":    mean(top1),
        "mean_topk_mass":    mean(topkm),
        "topk_overlap":      mean(in_tk),           # 实际 token 落在 top-k 里的比例
    }


def compute_mismatch(
    rollout_logp: List[float],
    train_logp:   List[float],
) -> Dict[str, float]:
    assert len(rollout_logp) == len(train_logp)
    T = len(rollout_logp)
    if T == 0:
        return {}

    # 转成概率
    rollout_probs = [math.exp(lp) for lp in rollout_logp]
    train_probs   = [math.exp(lp) for lp in train_logp]

    # 概率差值（有符号：rollout - train）
    prob_gaps     = [r - t for r, t in zip(rollout_probs, train_probs)]
    abs_prob_gaps = [abs(g) for g in prob_gaps]

    # 概率比率（rollout / train），log 空间直接相减再 exp）
    ratios        = [math.exp(r - t) for r, t in zip(rollout_logp, train_logp)]

    def mean(lst): return sum(lst) / len(lst)
    def maxv(lst): return max(lst)
    def p95(lst):
        s = sorted(lst)
        return s[int(0.95 * (len(s) - 1))]

    return {
        # 概率差值
        "mean_prob_gap":      mean(prob_gaps),       # 正：rollout 更自信，负：train 更自信
        "mean_abs_prob_gap":  mean(abs_prob_gaps),   # 平均绝对偏差
        "max_abs_prob_gap":   maxv(abs_prob_gaps),   # 最大偏差（定位退化 token）
        "p95_abs_prob_gap":   p95(abs_prob_gaps),    # 去掉极端值的上界

        # 概率比率
        "mean_prob_ratio":    mean(ratios),           # 均值接近 1 说明整体对齐
        "max_prob_ratio":     maxv(ratios),           # 最大比率（rollout 严重高估的 token）
        "p95_prob_ratio":     p95(ratios),

        # PPL 保留，方便和外部指标对齐
        "ppl_rollout":        math.exp(-mean(rollout_logp)),
        "ppl_train":          math.exp(-mean(train_logp)),

        # 逐 token，方便位置分析
        "per_token_prob_gap": prob_gaps,
        "per_token_ratio":    ratios,
    }



# ─────────────────────────────────────────────────────────────
# 主分析函数：对单条样本做完整分析
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def analyze_sample(
    model,
    tokenizer,
    prompt_ids:    torch.Tensor,    # (S_p,)
    response_ids:  torch.Tensor,    # (S_r,)
    rollout_logits: Optional[torch.Tensor],  # (S_r, V)，rollout 时保存的 logits
    cfg: AnalysisConfig,
    device: torch.device,
) -> Dict:
    """
    对单条样本做 task1 + task2 分析。

    rollout_logits：
        如果训练框架保存了 rollout 时每步的 logits，直接传入。
        如果没有（只有 token ids），则跳过 mismatch，只做 task2 的 train forward。

    返回 dict 包含：
        - rollout_stats:  task2 基于 rollout logits 的 per-token 统计
        - train_stats:    task2 基于 train full-forward logits 的 per-token 统计
        - mismatch:       task1 mismatch 指标（如果 rollout_logits 存在）
        - rollout_agg:    rollout_stats 的句子级聚合
        - train_agg:      train_stats 的句子级聚合
    """
    S_r = min(response_ids.shape[0], cfg.max_new_tokens)
    response_ids = response_ids[:S_r].to(device)

    result = {}

    # ── Task2 rollout 侧（如果有 rollout logits）────────────────
    if cfg.compute_token_stats and rollout_logits is not None:
        ro_logits = rollout_logits[:S_r].to(device)   # (S_r, V)
        rollout_stats = compute_token_stats_from_logits(ro_logits, response_ids, cfg.topk)
        result["rollout_stats"] = rollout_stats
        result["rollout_agg"]   = aggregate_token_stats(rollout_stats)

    # ── Train full-forward ──────────────────────────────────────
    # 拼接 prompt + response，做一次完整 forward
    full_ids = torch.cat([prompt_ids.to(device), response_ids], dim=0).unsqueeze(0)  # (1, S_p+S_r)

    outputs = model(input_ids=full_ids, use_cache=False)
    full_logits = outputs.logits[0]   # (S_p+S_r, V)

    # response 部分的 logits：预测第 t 个 response token 的 logits
    # 是 full_logits[S_p-1 : S_p+S_r-1]（causal LM 的 offset）
    S_p = prompt_ids.shape[0]
    train_logits = full_logits[S_p - 1: S_p - 1 + S_r]   # (S_r, V)

    if cfg.compute_token_stats:
        train_stats = compute_token_stats_from_logits(train_logits, response_ids, cfg.topk)
        result["train_stats"] = train_stats
        result["train_agg"]   = aggregate_token_stats(train_stats)

    # ── Task1 mismatch ──────────────────────────────────────────
    if cfg.compute_mismatch and rollout_logits is not None:
        rollout_logp = result["rollout_stats"]["token_logp"]
        train_logp   = result["train_stats"]["token_logp"]
        result["mismatch"] = compute_mismatch(rollout_logp, train_logp)

    return result


# ─────────────────────────────────────────────────────────────
# Step 级聚合
# ─────────────────────────────────────────────────────────────

def aggregate_step(
    step: int,
    sample_analyses: List[Dict],
    is_bad_flags:    List[bool],
    rewards:         Optional[List[float]] = None,
) -> Dict:
    """
    对同一 step 的多条样本分析结果做聚合。
    分别对 good / bad 样本统计。
    """
    n = len(sample_analyses)
    if n == 0:
        return {"step": step}

    rewards = rewards or [None] * n

    def _mean(lst):
        lst = [v for v in lst if v is not None]
        return sum(lst) / len(lst) if lst else None

    def _p95(lst):
        lst = sorted(v for v in lst if v is not None)
        if not lst:
            return None
        idx = int(0.95 * (len(lst) - 1))
        return lst[idx]

    # ── mismatch 聚合 ───────────────────────────────────────────
    mismatch_agg = {}
    mismatch_keys = [
        "mean_prob_gap",
        "mean_abs_prob_gap",
        "max_abs_prob_gap",
        "p95_abs_prob_gap",
        "mean_prob_ratio",
        "max_prob_ratio",
        "p95_prob_ratio",
        "ppl_rollout",
        "ppl_train",
    ]
    for key in mismatch_keys:
        vals = [
            s["mismatch"][key]
            for s in sample_analyses
            if "mismatch" in s and key in s["mismatch"]
        ]
        mismatch_agg[f"{key}_mean"] = _mean(vals)
        mismatch_agg[f"{key}_p95"]  = _p95(vals)

    # ── token stats 聚合（分 good / bad）───────────────────────
    def _agg_token_stats_group(indices, source_key):
        """对指定样本子集，聚合 train 或 rollout 的句子级指标。"""
        agg_keys = [
            "mean_token_ppl", "max_token_ppl", "ppl",
            "mean_entropy", "max_entropy",
            "min_top1_prob", "mean_topk_mass", "topk_overlap",
        ]
        out = {}
        for key in agg_keys:
            vals = [
                sample_analyses[i][source_key][key]
                for i in indices
                if source_key in sample_analyses[i]
                and key in sample_analyses[i][source_key]
            ]
            out[f"{key}_mean"] = _mean(vals)
        return out

    good_idx = [i for i, bad in enumerate(is_bad_flags) if not bad]
    bad_idx  = [i for i, bad in enumerate(is_bad_flags) if bad]

    token_stats_agg = {
        "train": {
            "all":  _agg_token_stats_group(list(range(n)), "train_agg"),
            "good": _agg_token_stats_group(good_idx, "train_agg"),
            "bad":  _agg_token_stats_group(bad_idx,  "train_agg"),
        },
    }
    if any("rollout_agg" in s for s in sample_analyses):
        token_stats_agg["rollout"] = {
            "all":  _agg_token_stats_group(list(range(n)), "rollout_agg"),
            "good": _agg_token_stats_group(good_idx, "rollout_agg"),
            "bad":  _agg_token_stats_group(bad_idx,  "rollout_agg"),
        }

    # ── 位置级统计（按 token position 聚合均值）────────────────
    pos_stats = _aggregate_by_position(sample_analyses, is_bad_flags)

    return {
        "step":           step,
        "num_samples":    n,
        "num_bad":        len(bad_idx),
        "num_good":       len(good_idx),
        "mismatch":       mismatch_agg,
        "token_stats":    token_stats_agg,
        "pos_stats":      pos_stats,
    }


def _aggregate_by_position(
    sample_analyses: List[Dict],
    is_bad_flags:    List[bool],
    max_pos:         int = 128,
) -> Dict:
    """
    按 token 位置聚合，分别对 good / bad 样本统计：
    - mean_token_ppl_by_pos
    - mean_entropy_by_pos
    - topk_overlap_by_pos
    """
    # 收集每个位置的数值，分 good / bad
    pos_data = {
        "good": {"ppl": {}, "entropy": {}, "topk_overlap": {}},
        "bad":  {"ppl": {}, "entropy": {}, "topk_overlap": {}},
    }

    for i, (analysis, is_bad) in enumerate(zip(sample_analyses, is_bad_flags)):
        group = "bad" if is_bad else "good"

        # 优先用 rollout_stats，没有则用 train_stats
        stats_key = "rollout_stats" if "rollout_stats" in analysis else "train_stats"
        if stats_key not in analysis:
            continue

        stats = analysis[stats_key]
        T = min(len(stats["token_ppl"]), max_pos)

        for t in range(T):
            for metric, src_key in [
                ("ppl",        "token_ppl"),
                ("entropy",    "entropy"),
                ("topk_overlap", "actual_in_topk"),
            ]:
                if t not in pos_data[group][metric]:
                    pos_data[group][metric][t] = []
                pos_data[group][metric][t].append(stats[src_key][t])

    # 转成均值列表
    def _to_mean_list(pos_dict, max_pos):
        result = []
        for t in range(max_pos):
            vals = pos_dict.get(t, [])
            result.append(sum(vals) / len(vals) if vals else None)
        # 截断末尾的 None
        while result and result[-1] is None:
            result.pop()
        return result

    out = {}
    for group in ["good", "bad"]:
        out[group] = {
            "mean_token_ppl_by_pos":    _to_mean_list(pos_data[group]["ppl"],         max_pos),
            "mean_entropy_by_pos":      _to_mean_list(pos_data[group]["entropy"],      max_pos),
            "topk_overlap_by_pos":      _to_mean_list(pos_data[group]["topk_overlap"], max_pos),
        }
    return out


# ─────────────────────────────────────────────────────────────
# 日志写入
# ─────────────────────────────────────────────────────────────

class AnalysisLogger:
    """
    写两个文件：
    - sample_analysis.jsonl：每条样本的聚合指标（不写 per-token 原始数据，太大）
    - step_analysis.jsonl：每 step 的聚合统计
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._sample_f = open(
            self.output_dir / "sample_analysis.jsonl", "a", buffering=1, encoding="utf-8"
        )
        self._step_f = open(
            self.output_dir / "step_analysis.jsonl", "a", buffering=1, encoding="utf-8"
        )

    def log_sample(
        self,
        step:       int,
        sample_id:  str,
        is_bad:     bool,
        bad_types:  List[str],
        reward:     Optional[float],
        analysis:   Dict,
    ):
        record = {
            "step":      step,
            "sample_id": sample_id,
            "is_bad":    is_bad,
            "bad_types": bad_types,
            "reward":    reward,
        }
        # 只写聚合指标，不写 per-token 原始列表（太大）
        if "rollout_agg" in analysis:
            record["rollout_agg"] = analysis["rollout_agg"]
        if "train_agg" in analysis:
            record["train_agg"] = analysis["train_agg"]
        if "mismatch" in analysis:
            # mismatch 里的 per_token_logp_gap 太大，只写标量
            m = {k: v for k, v in analysis["mismatch"].items()
                if k not in ("per_token_prob_gap", "per_token_ratio")}
            record["mismatch"] = m
        self._sample_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._sample_f.flush()

    def log_step(self, step_result: Dict):
        self._step_f.write(json.dumps(step_result, ensure_ascii=False) + "\n")
        self._step_f.flush()

    def close(self):
        self._sample_f.close()
        self._step_f.close()


# ─────────────────────────────────────────────────────────────
# 离线可视化
# ─────────────────────────────────────────────────────────────

class AnalysisPlotter:
    """
    读取 step_analysis.jsonl，画图。

    用法：
        plotter = AnalysisPlotter("output/analysis")
        plotter.plot_all()
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self._records = None

    def _load(self):
        if self._records is not None:
            return self._records
        path = self.output_dir / "step_analysis.jsonl"
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self._records = records
        return records

    def plot_all(self, save_dir: Optional[str] = None):
        import matplotlib.pyplot as plt

        save_path = Path(save_dir) if save_dir else self.output_dir / "plots"
        save_path.mkdir(parents=True, exist_ok=True)
        records = self._load()
        if not records:
            print("[AnalysisPlotter] No data.")
            return

        steps = [r["step"] for r in records]

        # ── 图1：mismatch 随 step 变化 ──────────────────────────
        mismatch_metrics = [
            ("mean_logp_gap_mean",     "mean logp gap"),
            ("mean_abs_logp_gap_mean", "mean |logp gap|"),
            ("ppl_rollout_mean",       "PPL rollout"),
            ("ppl_train_mean",         "PPL train"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Rollout-Train Mismatch over Steps")
        for ax, (key, label) in zip(axes.flatten(), mismatch_metrics):
            vals = [r.get("mismatch", {}).get(key) for r in records]
            ax.plot(steps, [v if v is not None else float("nan") for v in vals],
                    color="steelblue", linewidth=1.5)
            ax.set_title(label)
            ax.set_xlabel("Step")
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path / "mismatch_over_steps.png", dpi=150)
        plt.close(fig)

        # ── 图2：good vs bad 的 ppl / entropy 随 step ──────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle("PPL & Entropy: Good vs Bad Samples over Steps")

        for ax, (metric, label) in zip(axes, [
            ("ppl_mean",         "Sentence PPL"),
            ("mean_entropy_mean","Mean Entropy"),
        ]):
            for group, color in [("good", "steelblue"), ("bad", "tomato")]:
                vals = [
                    r.get("token_stats", {}).get("train", {}).get(group, {}).get(metric)
                    for r in records
                ]
                ax.plot(steps, [v if v is not None else float("nan") for v in vals],
                        label=f"{group} (train)", color=color, linewidth=1.5)
            ax.set_title(label)
            ax.set_xlabel("Step")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / "ppl_entropy_good_vs_bad.png", dpi=150)
        plt.close(fig)

        # ── 图3：最后一个 step 的 by-position 曲线 ─────────────
        last = records[-1]
        pos_stats = last.get("pos_stats", {})
        if pos_stats:
            fig, axes = plt.subplots(1, 3, figsize=(18, 4))
            fig.suptitle(f"Token-level Stats by Position (step={last['step']})")

            for ax, (metric, label) in zip(axes, [
                ("mean_token_ppl_by_pos", "Token PPL"),
                ("mean_entropy_by_pos",   "Entropy"),
                ("topk_overlap_by_pos",   "Top-k Overlap"),
            ]):
                for group, color in [("good", "steelblue"), ("bad", "tomato")]:
                    vals = pos_stats.get(group, {}).get(metric, [])
                    if vals:
                        ax.plot(range(len(vals)), vals,
                                label=group, color=color, linewidth=1.5)
                ax.set_title(label)
                ax.set_xlabel("Token Position")
                ax.legend()
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path / f"by_position_step{last['step']}.png", dpi=150)
            plt.close(fig)

        print(f"[AnalysisPlotter] 📊 Plots saved to {save_path}")
