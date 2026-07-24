# rollout_quality.py
# Rollout 文本质量检测模块 —— 规则型，无需模型

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────

@dataclass
class QualityConfig:
    """所有阈值均可覆盖，支持不同任务单独实例化。"""

    # 格式异常类
    min_char_len: int = 8
    non_printable_ratio_thresh: float = 0.05
    newline_ratio_thresh: float = 0.30
    special_char_ratio_thresh: float = 0.15
    punct_ratio_thresh: float = 0.50          # 标点占比超过此值视为异常
    ellipsis_ratio_thresh: float = 0.15       # 连续省略号占比

    # 重复异常类
    repeat_2gram_ratio_thresh: float = 0.30
    repeat_3gram_ratio_thresh: float = 0.20
    max_token_run_thresh: int = 6
    unique_token_ratio_thresh: float = 0.12   # 仅在 char_len >= min_len_for_diversity 时生效
    min_len_for_diversity: int = 60           # 太短的文本不做 diversity 判断

    # 语言混杂类
    mixed_lang_switch_thresh: int = 4
    min_len_for_lang_check: int = 20          # 太短的文本不做语言判断

    # 可关闭的规则（设为 False 即跳过）
    enable_empty: bool = True
    enable_too_short: bool = True
    enable_garbled: bool = True
    enable_format_abnormal: bool = True
    enable_symbol_flood: bool = True
    enable_repetition: bool = True
    enable_token_loop: bool = True
    enable_low_diversity: bool = True
    enable_mixed_lang: bool = True
    enable_punct_flood: bool = True          
    enable_ellipsis_flood: bool = True       


# ─────────────────────────────────────────────────────────────
# 分词工具（轻量，不依赖 tokenizer）
# ─────────────────────────────────────────────────────────────

_ZH_RANGE = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uff00-\uffef]')
_EN_WORD = re.compile(r'[a-zA-Z]+')
_PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
_SPECIAL_CHARS = re.compile(r'[*#~|`@]')
_ELLIPSIS = re.compile(r'\.{2,}|。{2,}')


def _simple_tokenize(text: str) -> List[str]:
    """
    轻量分词：
    - 中文按字切
    - 英文按空格/标点切词
    - 数字单独保留
    返回 token 列表。
    """
    tokens = []
    i = 0
    while i < len(text):
        ch = text[i]
        if _ZH_RANGE.match(ch):
            tokens.append(ch)
            i += 1
        elif ch.isalpha():
            # 英文单词
            j = i
            while j < len(text) and text[j].isalpha():
                j += 1
            tokens.append(text[i:j].lower())
            i = j
        elif ch.isdigit():
            j = i
            while j < len(text) and text[j].isdigit():
                j += 1
            tokens.append(text[i:j])
            i = j
        elif ch in (' ', '\t', '\r'):
            i += 1
        elif ch == '\n':
            tokens.append('\n')
            i += 1
        else:
            tokens.append(ch)
            i += 1
    return tokens


def _is_zh_char(ch: str) -> bool:
    return bool(_ZH_RANGE.match(ch))


def _is_en_char(ch: str) -> bool:
    return ch.isascii() and ch.isalpha()


# ─────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────

def _compute_metrics(text: str) -> Dict[str, Any]:
    """
    计算单条文本的所有原始指标，不做判定。
    返回 dict，所有值均为 Python 原生类型（可直接 JSON 序列化）。
    """
    metrics: Dict[str, Any] = {}

    # ── 基础 ──────────────────────────────────────────────────
    stripped = text.strip()
    char_len = len(stripped)
    metrics["char_len"] = char_len
    metrics["is_empty"] = char_len == 0

    if char_len == 0:
        # 空文本，其余指标全部填 0
        for k in [
            "non_printable_ratio", "newline_ratio", "punct_ratio",
            "ellipsis_ratio", "special_char_ratio",
            "repeat_2gram_ratio", "repeat_3gram_ratio",
            "max_token_run", "max_substring_repeat", "unique_token_ratio",
            "has_zh", "has_en", "mixed_lang_switches", "mixed_lang_ratio",
        ]:
            metrics[k] = 0
        return metrics

    total_chars = len(text)  # 用原始长度（含空白）算比例更准确

    # ── 格式类 ────────────────────────────────────────────────

    # 不可打印字符（排除常见空白）
    non_printable = sum(
        1 for ch in text
        if unicodedata.category(ch) in ('Cc', 'Cf', 'Cs', 'Co', 'Cn')
        and ch not in ('\n', '\r', '\t', ' ')
    )
    metrics["non_printable_ratio"] = round(non_printable / total_chars, 4)

    newline_count = text.count('\n')
    metrics["newline_ratio"] = round(newline_count / total_chars, 4)

    punct_count = len(_PUNCT.findall(text))
    metrics["punct_ratio"] = round(punct_count / total_chars, 4)

    ellipsis_chars = sum(len(m) for m in _ELLIPSIS.findall(text))
    metrics["ellipsis_ratio"] = round(ellipsis_chars / total_chars, 4)

    special_count = len(_SPECIAL_CHARS.findall(text))
    metrics["special_char_ratio"] = round(special_count / total_chars, 4)

    # ── 重复类 ────────────────────────────────────────────────
    tokens = _simple_tokenize(stripped)
    token_count = len(tokens)

    if token_count >= 2:
        bigrams = [(tokens[i], tokens[i + 1]) for i in range(token_count - 1)]
        bigram_counts = Counter(bigrams)
        repeat_bigrams = sum(v for v in bigram_counts.values() if v > 1)
        metrics["repeat_2gram_ratio"] = round(repeat_bigrams / max(len(bigrams), 1), 4)
    else:
        metrics["repeat_2gram_ratio"] = 0.0

    if token_count >= 3:
        trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(token_count - 2)]
        trigram_counts = Counter(trigrams)
        repeat_trigrams = sum(v for v in trigram_counts.values() if v > 1)
        metrics["repeat_3gram_ratio"] = round(repeat_trigrams / max(len(trigrams), 1), 4)
    else:
        metrics["repeat_3gram_ratio"] = 0.0

    # 连续相同 token 最大长度
    max_run = 1
    cur_run = 1
    for i in range(1, token_count):
        if tokens[i] == tokens[i - 1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1
    metrics["max_token_run"] = max_run if token_count > 0 else 0

    # 最长重复子串（用字符级，限制搜索长度避免超时）
    metrics["max_substring_repeat"] = _max_substring_repeat(stripped, max_search_len=500)

    # unique token ratio
    if token_count > 0:
        metrics["unique_token_ratio"] = round(len(set(tokens)) / token_count, 4)
    else:
        metrics["unique_token_ratio"] = 0.0

    # ── 语言混杂类 ────────────────────────────────────────────
    zh_chars = sum(1 for ch in stripped if _is_zh_char(ch))
    en_chars = sum(1 for ch in stripped if _is_en_char(ch))
    lang_chars = zh_chars + en_chars

    metrics["has_zh"] = zh_chars > 0
    metrics["has_en"] = en_chars > 0

    # 中英切换次数：统计 zh/en 字符段的交替次数
    switches, lang_ratio = _compute_lang_switches(stripped, zh_chars, en_chars, char_len)
    metrics["mixed_lang_switches"] = switches
    metrics["mixed_lang_ratio"] = round(lang_ratio, 4)

    return metrics


def _max_substring_repeat(text: str, max_search_len: int = 500) -> int:
    """
    找最长的重复子串长度。
    只搜索前 max_search_len 个字符，避免 O(n^2) 超时。
    用滑动窗口：对每个长度 L，检查是否有子串出现超过一次。
    """
    s = text[:max_search_len]
    n = len(s)
    if n < 4:
        return 0

    best = 0
    # 从长到短搜索，找到第一个就停
    for length in range(n // 2, 1, -1):
        seen = set()
        found = False
        for i in range(n - length + 1):
            sub = s[i:i + length]
            if sub in seen:
                best = length
                found = True
                break
            seen.add(sub)
        if found:
            break
    return best


def _compute_lang_switches(
    text: str,
    zh_count: int,
    en_count: int,
    char_len: int,
) -> Tuple[int, float]:
    """
    返回 (切换次数, 次要语言字符占比)。
    切换次数：连续的 zh 段和 en 段交替出现的次数。
    """
    if char_len == 0 or (zh_count == 0 and en_count == 0):
        return 0, 0.0

    # 构建语言序列：'z'=中文, 'e'=英文, 'o'=其他
    lang_seq = []
    for ch in text:
        if _is_zh_char(ch):
            lang_seq.append('z')
        elif _is_en_char(ch):
            lang_seq.append('e')
        # 其他字符跳过（不计入切换）

    if not lang_seq:
        return 0, 0.0

    # 压缩连续相同语言
    compressed = [lang_seq[0]]
    for c in lang_seq[1:]:
        if c != compressed[-1]:
            compressed.append(c)

    switches = len(compressed) - 1

    # 次要语言占比
    total_lang = zh_count + en_count
    minor = min(zh_count, en_count)
    lang_ratio = minor / total_lang if total_lang > 0 else 0.0

    return switches, lang_ratio


# ─────────────────────────────────────────────────────────────
# 判定逻辑
# ─────────────────────────────────────────────────────────────

def _apply_rules(
    metrics: Dict[str, Any],
    cfg: QualityConfig,
) -> Tuple[bool, List[str]]:
    """
    根据 metrics 和 cfg 判定 bad types。
    返回 (is_bad, bad_types_list)。
    """
    bad_types = []

    char_len = metrics["char_len"]

    # empty
    if cfg.enable_empty and metrics["is_empty"]:
        bad_types.append("empty")
        return True, bad_types  # 空文本直接返回，后续指标无意义

    # too_short
    if cfg.enable_too_short and char_len < cfg.min_char_len:
        bad_types.append("too_short")

    # garbled
    if cfg.enable_garbled and metrics["non_printable_ratio"] > cfg.non_printable_ratio_thresh:
        bad_types.append("garbled")

    # format_abnormal（换行率过高）
    if cfg.enable_format_abnormal and metrics["newline_ratio"] > cfg.newline_ratio_thresh:
        bad_types.append("format_abnormal")

    # symbol_flood（特殊符号过多）
    if cfg.enable_symbol_flood and metrics["special_char_ratio"] > cfg.special_char_ratio_thresh:
        bad_types.append("symbol_flood")

    # repetition（2gram 或 3gram 重复率过高）
    if cfg.enable_repetition and (
        metrics["repeat_2gram_ratio"] > cfg.repeat_2gram_ratio_thresh
        or metrics["repeat_3gram_ratio"] > cfg.repeat_3gram_ratio_thresh
    ):
        bad_types.append("repetition")

    # token_loop（连续相同 token 过长）
    if cfg.enable_token_loop and metrics["max_token_run"] > cfg.max_token_run_thresh:
        bad_types.append("token_loop")

    # low_diversity（unique token ratio 过低，仅在文本够长时判断）
    if (
        cfg.enable_low_diversity
        and char_len >= cfg.min_len_for_diversity
        and metrics["unique_token_ratio"] < cfg.unique_token_ratio_thresh
    ):
        bad_types.append("low_diversity")

    # mixed_lang（中英混杂切换过多，仅在文本够长时判断）
    if (
        cfg.enable_mixed_lang
        and char_len >= cfg.min_len_for_lang_check
        and metrics["has_zh"]
        and metrics["has_en"]
        and metrics["mixed_lang_switches"] > cfg.mixed_lang_switch_thresh
    ):
        bad_types.append("mixed_lang")
    
    # punct_flood（标点占比过高）
    if cfg.enable_punct_flood and metrics["punct_ratio"] > cfg.punct_ratio_thresh:
        bad_types.append("punct_flood")

    # ellipsis_flood（连续省略号过多）
    if cfg.enable_ellipsis_flood and metrics["ellipsis_ratio"] > cfg.ellipsis_ratio_thresh:
        bad_types.append("ellipsis_flood")

    return len(bad_types) > 0, bad_types


# ─────────────────────────────────────────────────────────────
# 单样本入口
# ─────────────────────────────────────────────────────────────

def analyze_sample(
    text: str,
    step: int = 0,
    sample_id: str = "",
    reward: Optional[float] = None,
    group_id: Optional[str] = None,
    cfg: Optional[QualityConfig] = None,
) -> Dict[str, Any]:
    """
    分析单条 rollout 文本，返回完整的 sample-level 结果。

    参数：
        text       : rollout 文本
        step       : 训练步数
        sample_id  : 样本唯一 ID（可选）
        reward     : 该样本的 reward 值（可选）
        group_id   : GRPO group ID（可选）
        cfg        : QualityConfig，不传则使用默认值

    返回示例：
        {
            "step": 120,
            "sample_id": "xxx",
            "reward": 0.0,
            "group_id": "g1",
            "is_bad": True,
            "bad_types": ["repetition", "mixed_lang"],
            "metrics": { ... }
        }
    """
    if cfg is None:
        cfg = QualityConfig()

    metrics = _compute_metrics(text)
    is_bad, bad_types = _apply_rules(metrics, cfg)

    result: Dict[str, Any] = {
        "step": step,
        "sample_id": sample_id,
        "is_bad": is_bad,
        "bad_types": bad_types,
        "metrics": metrics,
    }
    if reward is not None:
        result["reward"] = reward
    if group_id is not None:
        result["group_id"] = group_id

    return result


def analyze_batch(
    texts: List[str],
    step: int = 0,
    sample_ids: Optional[List[str]] = None,
    rewards: Optional[List[Optional[float]]] = None,
    group_ids: Optional[List[Optional[str]]] = None,
    cfg: Optional[QualityConfig] = None,
) -> List[Dict[str, Any]]:
    """
    批量分析一组 rollout 文本，返回 sample-level 结果列表。
    """
    if cfg is None:
        cfg = QualityConfig()

    n = len(texts)
    sample_ids = sample_ids or [f"{step}_{i}" for i in range(n)]
    rewards = rewards or [None] * n
    group_ids = group_ids or [None] * n

    return [
        analyze_sample(
            text=texts[i],
            step=step,
            sample_id=sample_ids[i],
            reward=rewards[i],
            group_id=group_ids[i],
            cfg=cfg,
        )
        for i in range(n)
    ]


# ─────────────────────────────────────────────────────────────
# Step 聚合
# ─────────────────────────────────────────────────────────────

ALL_BAD_TYPES = [
    "empty", "too_short", "garbled", "format_abnormal",
    "symbol_flood", "punct_flood", "ellipsis_flood",
    "repetition", "token_loop", "low_diversity", "mixed_lang",
]

METRIC_KEYS = [
    "char_len", "non_printable_ratio", "newline_ratio", "punct_ratio",
    "ellipsis_ratio", "special_char_ratio",
    "repeat_2gram_ratio", "repeat_3gram_ratio",
    "max_token_run", "max_substring_repeat", "unique_token_ratio",
    "mixed_lang_switches", "mixed_lang_ratio",
]


def _percentile(values: List[float], p: float) -> float:
    """简单百分位数，不依赖 numpy。"""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = idx - lo
    return round(sorted_v[lo] * (1 - frac) + sorted_v[hi] * frac, 6)


def aggregate_step(
    sample_results: List[Dict[str, Any]],
    texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    对同一 step 的 sample-level 结果列表做聚合，
    返回 step-level 统计结果。
    """
    if not sample_results:
        return {}

    step = sample_results[0]["step"]
    n = len(sample_results)

    # ── bad ratio ──────────────────────────────────────────────
    bad_count = sum(1 for r in sample_results if r["is_bad"])
    bad_ratio = round(bad_count / n, 4)

    # ── bad type ratio ─────────────────────────────────────────
    bad_type_counts: Dict[str, int] = defaultdict(int)
    for r in sample_results:
        for bt in r["bad_types"]:
            bad_type_counts[bt] += 1
    bad_type_ratio = {
        bt: round(bad_type_counts[bt] / n, 4)
        for bt in ALL_BAD_TYPES
    }

    # ── metric summary ─────────────────────────────────────────
    metric_summary: Dict[str, float] = {}
    for key in METRIC_KEYS:
        vals = [
            r["metrics"][key]
            for r in sample_results
            if key in r["metrics"] and isinstance(r["metrics"][key], (int, float))
        ]
        if vals:
            mean_v = sum(vals) / len(vals)
            std_v = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
            metric_summary[f"{key}_mean"] = round(mean_v, 4)
            metric_summary[f"{key}_std"] = round(std_v, 4)
            metric_summary[f"{key}_p50"] = _percentile(vals, 50)
            metric_summary[f"{key}_p95"] = _percentile(vals, 95)

    # ── reward 相关统计 ────────────────────────────────────────
    reward_summary: Dict[str, Any] = {}
    samples_with_reward = [r for r in sample_results if r.get("reward") is not None]

    if samples_with_reward:
        all_rewards = [r["reward"] for r in samples_with_reward]
        reward_summary["reward_mean"] = round(sum(all_rewards) / len(all_rewards), 4)
        reward_summary["reward_std"] = round(
            (sum((v - reward_summary["reward_mean"]) ** 2 for v in all_rewards) / len(all_rewards)) ** 0.5,
            4
        )

        bad_rewards = [r["reward"] for r in samples_with_reward if r["is_bad"]]
        good_rewards = [r["reward"] for r in samples_with_reward if not r["is_bad"]]

        reward_summary["bad_samples_reward_mean"] = (
            round(sum(bad_rewards) / len(bad_rewards), 4) if bad_rewards else None
        )
        reward_summary["good_samples_reward_mean"] = (
            round(sum(good_rewards) / len(good_rewards), 4) if good_rewards else None
        )

        # 各 bad type 的 reward 均值
        reward_by_bad_type: Dict[str, Optional[float]] = {}
        for bt in ALL_BAD_TYPES:
            bt_rewards = [
                r["reward"] for r in samples_with_reward
                if bt in r["bad_types"] and r.get("reward") is not None
            ]
            reward_by_bad_type[bt] = (
                round(sum(bt_rewards) / len(bt_rewards), 4) if bt_rewards else None
            )
        reward_summary["reward_mean_by_bad_type"] = reward_by_bad_type

    # ── group 相关统计 ─────────────────────────────────────────
    # GRPO uses within-prompt groups.  Group-level health is more informative
    # than a global bad ratio: a step can have acceptable global quality while
    # still containing groups with no usable candidate for a relative update.
    group_summary: Dict[str, Any] = {}
    samples_with_group = [r for r in sample_results if r.get("group_id") is not None]

    if samples_with_group:
        group_bad: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "bad": 0, "rewards": [], "texts": []}
        )
        for idx, r in enumerate(sample_results):
            if r.get("group_id") is None:
                continue
            gid = r["group_id"]
            group_bad[gid]["total"] += 1
            if r["is_bad"]:
                group_bad[gid]["bad"] += 1
            if r.get("reward") is not None:
                group_bad[gid]["rewards"].append(float(r["reward"]))
            if texts is not None and idx < len(texts):
                normalized = " ".join(texts[idx].split()).strip().lower()
                group_bad[gid]["texts"].append(normalized)

        group_bad_ratios = [
            v["bad"] / v["total"]
            for v in group_bad.values()
            if v["total"] > 0
        ]
        group_summary["num_groups"] = len(group_bad)
        group_summary["group_bad_ratio_mean"] = round(
            sum(group_bad_ratios) / len(group_bad_ratios), 4
        ) if group_bad_ratios else 0.0
        group_summary["group_bad_ratio_p95"] = _percentile(group_bad_ratios, 95)
        group_summary["all_bad_group_ratio"] = round(
            sum(v["bad"] == v["total"] for v in group_bad.values()) / len(group_bad), 4
        )

        reward_stds = []
        zero_variance_groups = 0
        duplicate_rates = []
        for v in group_bad.values():
            rewards = v["rewards"]
            if len(rewards) >= 2:
                mean_reward = sum(rewards) / len(rewards)
                reward_std = (sum((x - mean_reward) ** 2 for x in rewards) / len(rewards)) ** 0.5
                reward_stds.append(reward_std)
                if reward_std == 0.0:
                    zero_variance_groups += 1
            group_texts = [text for text in v["texts"] if text]
            if group_texts:
                duplicate_rates.append(1.0 - len(set(group_texts)) / len(group_texts))

        if reward_stds:
            group_summary["group_reward_std_mean"] = round(sum(reward_stds) / len(reward_stds), 4)
            group_summary["zero_reward_variance_group_ratio"] = round(
                zero_variance_groups / len(reward_stds), 4
            )
        if duplicate_rates:
            group_summary["exact_duplicate_rate_mean"] = round(
                sum(duplicate_rates) / len(duplicate_rates), 4
            )
        # 每个 group 的详情
        group_summary["group_detail"] = {
            gid: {
                "total": v["total"],
                "bad_count": v["bad"],
                "bad_ratio": round(v["bad"] / v["total"], 4),
                "reward_std": round(
                    (sum((x - sum(v["rewards"]) / len(v["rewards"])) ** 2 for x in v["rewards"]) / len(v["rewards"])) ** 0.5,
                    4,
                ) if len(v["rewards"]) >= 2 else None,
                "exact_duplicate_rate": round(
                    1.0 - len(set(text for text in v["texts"] if text)) / len([text for text in v["texts"] if text]),
                    4,
                ) if any(v["texts"]) else None,
            }
            for gid, v in group_bad.items()
        }

    return {
        "step": step,
        "num_samples": n,
        "bad_ratio": bad_ratio,
        "bad_type_ratio": bad_type_ratio,
        "metric_summary": metric_summary,
        "reward_summary": reward_summary,
        "group_summary": group_summary,
    }


# ─────────────────────────────────────────────────────────────
# 导出工具
# ─────────────────────────────────────────────────────────────

class QualityLogger:
    """
    负责把 sample-level 和 step-level 结果持久化到 jsonl / csv。

    用法：
        logger = QualityLogger(output_dir="output/quality")
        logger.log_samples(sample_results)          # 每 step 调用一次
        logger.log_step(aggregate_step(sample_results))
        logger.close()
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._sample_jsonl = open(
            self.output_dir / "sample_quality.jsonl", "a", buffering=1, encoding="utf-8"
        )
        self._step_jsonl = open(
            self.output_dir / "step_quality.jsonl", "a", buffering=1, encoding="utf-8"
        )

        # CSV：sample-level（延迟初始化 header）
        self._sample_csv_path = self.output_dir / "sample_quality.csv"
        self._sample_csv_initialized = self._sample_csv_path.exists()
        self._sample_csv_file = open(self._sample_csv_path, "a", newline="", encoding="utf-8")
        self._sample_csv_writer = None

        # CSV：step-level（延迟初始化 header）
        self._step_csv_path = self.output_dir / "step_quality.csv"
        self._step_csv_initialized = self._step_csv_path.exists()
        self._step_csv_file = open(self._step_csv_path, "a", newline="", encoding="utf-8")
        self._step_csv_writer = None

    def log_samples(self, sample_results: List[Dict[str, Any]]):
        """写入 sample-level 结果（jsonl + csv）。"""
        for r in sample_results:
            self._sample_jsonl.write(json.dumps(r, ensure_ascii=False) + "\n")

        # CSV：展平 metrics
        flat_rows = [_flatten_sample(r) for r in sample_results]
        if flat_rows:
            if self._sample_csv_writer is None:
                fieldnames = list(flat_rows[0].keys())
                self._sample_csv_writer = csv.DictWriter(
                    self._sample_csv_file, fieldnames=fieldnames
                )
                if not self._sample_csv_initialized:
                    self._sample_csv_writer.writeheader()
                    self._sample_csv_initialized = True
            for row in flat_rows:
                self._sample_csv_writer.writerow(row)

    def log_step(self, step_result: Dict[str, Any]):
        """写入 step-level 聚合结果（jsonl + csv）。"""
        self._step_jsonl.write(json.dumps(step_result, ensure_ascii=False) + "\n")

        flat = _flatten_step(step_result)
        if self._step_csv_writer is None:
            fieldnames = list(flat.keys())
            self._step_csv_writer = csv.DictWriter(
                self._step_csv_file, fieldnames=fieldnames
            )
            if not self._step_csv_initialized:
                self._step_csv_writer.writeheader()
                self._step_csv_initialized = True
        self._step_csv_writer.writerow(flat)

    def close(self):
        for f in [
            self._sample_jsonl, self._step_jsonl,
            self._sample_csv_file, self._step_csv_file,
        ]:
            try:
                f.close()
            except Exception:
                pass


def _flatten_sample(r: Dict[str, Any]) -> Dict[str, Any]:
    """把 sample-level 结果展平为一行 CSV 格式。"""
    row: Dict[str, Any] = {
        "step": r.get("step", ""),
        "sample_id": r.get("sample_id", ""),
        "is_bad": int(r.get("is_bad", False)),
        "bad_types": "|".join(r.get("bad_types", [])),
        "reward": r.get("reward", ""),
        "group_id": r.get("group_id", ""),
    }
    for k, v in r.get("metrics", {}).items():
        row[f"m_{k}"] = int(v) if isinstance(v, bool) else v
    return row


def _flatten_step(r: Dict[str, Any]) -> Dict[str, Any]:
    """把 step-level 结果展平为一行 CSV 格式。"""
    row: Dict[str, Any] = {
        "step": r.get("step", ""),
        "num_samples": r.get("num_samples", ""),
        "bad_ratio": r.get("bad_ratio", ""),
    }
    for bt, ratio in r.get("bad_type_ratio", {}).items():
        row[f"bad_{bt}"] = ratio
    for k, v in r.get("metric_summary", {}).items():
        row[f"ms_{k}"] = v
    rs = r.get("reward_summary", {})
    for k in ["reward_mean", "reward_std", "bad_samples_reward_mean", "good_samples_reward_mean"]:
        row[f"rs_{k}"] = rs.get(k, "")
    gs = r.get("group_summary", {})
    for k in [
        "num_groups", "group_bad_ratio_mean", "group_bad_ratio_p95",
        "all_bad_group_ratio", "group_reward_std_mean",
        "zero_reward_variance_group_ratio", "exact_duplicate_rate_mean",
    ]:
        row[f"gs_{k}"] = gs.get(k, "")
    return row


# ─────────────────────────────────────────────────────────────
# 离线分析（训练结束后调用）
# ─────────────────────────────────────────────────────────────

class QualityAnalyzer:
    """
    读取已保存的 step_quality.jsonl，生成可视化图表。

    用法：
        analyzer = QualityAnalyzer("output/quality")
        analyzer.plot_all()
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self._step_records: Optional[List[Dict]] = None

    def _load(self) -> List[Dict]:
        if self._step_records is not None:
            return self._step_records
        path = self.output_dir / "step_quality.jsonl"
        records = []
        if not path.exists():
            print(f"[QualityAnalyzer] {path} not found.")
            return records
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        self._step_records = records
        return records

    def plot_all(self, save_dir: Optional[str] = None):
        import matplotlib.pyplot as plt

        save_dir_path = Path(save_dir) if save_dir else self.output_dir / "quality_plots"
        save_dir_path.mkdir(parents=True, exist_ok=True)

        records = self._load()
        if not records:
            print("[QualityAnalyzer] No data to plot.")
            return

        steps = [r["step"] for r in records]

        # ── 图1：bad_ratio + 各 bad type ratio ─────────────────
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        fig.suptitle("Rollout Quality: Bad Sample Ratio over Steps")

        bad_ratios = [r.get("bad_ratio", 0) for r in records]
        axes[0].plot(steps, bad_ratios, color="tomato", linewidth=2, label="bad_ratio")
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel("Bad Ratio")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        colors = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999", "#66c2a5"
        ]
        for i, bt in enumerate(ALL_BAD_TYPES):
            vals = [r.get("bad_type_ratio", {}).get(bt, 0) for r in records]
            axes[1].plot(steps, vals, label=bt, color=colors[i % len(colors)], alpha=0.8)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Bad Type Ratio")
        axes[1].set_xlabel("Step")
        axes[1].legend(fontsize=8, ncol=3)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir_path / "bad_ratio_over_steps.png", dpi=150)
        plt.close(fig)

        # ── 图2：reward 对比（good vs bad）──────────────────────
        bad_r = [r.get("reward_summary", {}).get("bad_samples_reward_mean") for r in records]
        good_r = [r.get("reward_summary", {}).get("good_samples_reward_mean") for r in records]

        if any(v is not None for v in bad_r + good_r):
            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(steps, [v if v is not None else float("nan") for v in good_r],
                    label="good samples reward mean", color="steelblue")
            ax.plot(steps, [v if v is not None else float("nan") for v in bad_r],
                    label="bad samples reward mean", color="tomato")
            ax.set_title("Reward Mean: Good vs Bad Samples")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward Mean")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir_path / "reward_good_vs_bad.png", dpi=150)
            plt.close(fig)

        # ── 图3：关键 metric 时序 ────────────────────────────────
        key_metrics = [
            "repeat_2gram_ratio", "unique_token_ratio",
            "mixed_lang_switches", "max_token_run",
        ]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle("Key Metrics over Steps")
        axes_flat = axes.flatten()

        for idx, key in enumerate(key_metrics):
            mean_key = f"{key}_mean"
            p95_key = f"{key}_p95"
            means = [r.get("metric_summary", {}).get(mean_key, float("nan")) for r in records]
            p95s = [r.get("metric_summary", {}).get(p95_key, float("nan")) for r in records]
            ax = axes_flat[idx]
            ax.plot(steps, means, label="mean", color="steelblue")
            ax.plot(steps, p95s, label="p95", color="orange", linestyle="--")
            ax.set_title(key)
            ax.set_xlabel("Step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir_path / "metric_trends.png", dpi=150)
        plt.close(fig)

        print(f"[QualityAnalyzer] 📊 Plots saved to {save_dir_path}")
