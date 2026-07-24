# GRPO 检测与分析代码说明

这份文档只说明当前代码里已经存在的检测、分析、日志类在做什么，以及默认是否开启。

> **Current short-run workflow (2026-07-17).** For the 10-step direct-FP4
> versus QAT+moving-mean study, use
> [short_run_rollout_analysis.md](short_run_rollout_analysis.md) as the
> operational guide. The current default is `ANALYZE_ROLLOUT=false`; enabling
> it records text quality and reward only. Logit capture and automatic plotting
> are both opt-in (`COLLECT_ROLLOUT_LOGITS=false` and
> `PLOT_ROLLOUT_ON_TRAIN_END=false` by default).

## 总开关

检测逻辑主要由 `src/grpo/grpo.py` 里的两个参数控制：

| 参数 | `grpo.py` 代码默认值 | `scripts/grpo/run_experiment.sh` 默认值 | 作用 |
| --- | --- | --- | --- |
| `--analyze_rollout` | `false` | `true`，来自 `configs/grpo/experiment_env.sh` 的 `ANALYZE_ROLLOUT=true` | 开启 rollout 文本质量、rollout logits 捕获、Metis 训练诊断 callback |
| `--use_custom_analysis` | `false` | `false`，来自 `USE_CUSTOM_ANALYSIS=false` | 开启额外 activation mean hook，写 pickle 文件 |

旧脚本当前状态：

| 脚本 | `analyze_rollout` | `use_custom_analysis` |
| --- | --- | --- |
| `scripts/grpo/run_grpo.sh` | `true` | `true` |
| `scripts/grpo/run_grpo_long_context.sh` | `false` | `true` |
| `scripts/grpo/run_experiment.sh` | 默认 `true`，可用 `ANALYZE_ROLLOUT=false` 覆盖 | 默认 `false`，可用 `USE_CUSTOM_ANALYSIS=true` 覆盖 |
| `scripts/grpo/run_qat.sh` / `src/grpo/qat_distill.py` | 不使用这些检测类 | 不使用这些检测类 |

## 运行时检测类

### `RolloutRewardWrapper`

位置：`src/grpo/metis_monitor.py`

开启条件：

- `--analyze_rollout true`
- 至少有一个 callable reward function，例如当前默认的 `accuracy_reward`

它包装第一个 reward function。每次 GRPO 计算 completions 的 reward 时，它会：

- 先调用原 reward function，得到每条 completion 的 reward。
- 从 completion 中取文本。
- 调用 `rollout_quality.analyze_batch()` 做规则检测。
- 写一份合并日志到 `rollout.jsonl`，包含 step、sample_id、reward、is_bad、bad_types、若干文本质量指标和原始文本。
- 通过 `QualityLogger` 写更完整的 sample / step 质量文件。
- 如果已有 rollout token ids / logits，还会调用 `rollout_analysis.analyze_sample()` 做 token-level 分析。

输出文件：

```text
<output_dir>/rollout.jsonl
<output_dir>/rollout_quality/sample_quality.jsonl
<output_dir>/rollout_quality/sample_quality.csv
<output_dir>/rollout_quality/step_quality.jsonl
<output_dir>/rollout_quality/step_quality.csv
<output_dir>/rollout_analysis/sample_analysis.jsonl
<output_dir>/rollout_analysis/step_analysis.jsonl
```

当前用途：

- 计算 bad ratio。
- 记录 reward mean / std。
- 区分 good samples 和 bad samples 的 reward。
- 为 rollout 退化案例留原文。

注意：

- 它只包装第一个 reward function。如果同时传多个 reward function，目前只有第一个 reward function 会被这个 wrapper 包住。
- `rollout_analysis` 依赖 `RolloutLogitsCapture` 和 callback 传入的 pending token/logit 数据。当前代码里这部分存在 step 时序错位风险，见本文“当前注意点”。

### `RolloutLogitsCapture`

位置：`src/grpo/metis_monitor.py`

开启条件：

- `--analyze_rollout true`

它 monkey-patch `model.generate`：

- 强制传入 `output_logits=True` 和 `return_dict_in_generate=True`。
- 从 generate 输出里取 `output.logits`。
- 保存当前 batch 的 `prompt_ids`、`response_ids`、`rollout_logits` 到内部 buffer。
- 对外仍按调用方原本期望返回 tensor 或 generate output。

输出：

- 不直接写文件。
- 它的 buffer 会被 `MetisDiagnosticCallback.on_step_end()` 取出，再交给 `RolloutRewardWrapper.set_rollout_data()`。

当前用途：

- 为 `rollout_analysis.py` 提供 rollout 时每个 generated token 的 logits。
- 支持计算 rollout logits 和 train full-forward logits 的 mismatch。

注意：

- 保存 logits 的内存开销很大，尤其 vocab 大、生成长时。
- 当前 callback 是在 `on_step_end` 才把 capture buffer 交给 wrapper，而 reward wrapper 是在 reward 计算时运行。也就是说 token/logit 数据和 wrapper 分析之间可能存在 step 对齐问题，需要后续单独修正后才能把 mismatch 指标当成稳定结果。

### `MetisDiagnosticCallback`

位置：`src/grpo/metis_monitor.py`

开启条件：

- `--analyze_rollout true`

创建位置：`src/grpo/grpo.py`

当前构造参数：

```python
log_every_steps=4
rank_check_every_steps=10
saturation_alert_threshold=0.05
```

它做三件事：

1. 注册 forward hook，监控若干层的输入 activation。
2. 在 step 结束时记录 activation / gradient / reward logs。
3. 训练结束时关闭 rollout wrapper 文件句柄，保存告警。

默认监控层后缀：

```text
layers.0.self_attn.q_proj
layers.0.self_attn.v_proj
layers.0.mlp.down_proj
layers.11.self_attn.q_proj
layers.11.self_attn.v_proj
layers.11.mlp.down_proj
layers.23.self_attn.q_proj
layers.23.self_attn.v_proj
layers.23.mlp.down_proj
```

每个 logging step 记录的 activation 指标：

- `input_mean`
- `input_std`
- `input_abs_max`
- `saturation_rate`
- `has_nan`
- `has_inf`
- `shape`

gradient 指标：

- 只记录参数名中包含 `vlinear`、`ulinear`、`.s`、`warmup_linear` 的参数。
- 记录 `grad_norm`、`grad_abs_max`、`has_nan`、`has_inf`。

告警：

- activation 有 NaN。
- activation 有 Inf。
- activation saturation rate 大于 `0.05`。
- gradient 有 NaN / Inf。
- effective rank 大于 0 且小于 5。

输出文件：

```text
<output_dir>/metis_diagnostics.jsonl
<output_dir>/metis_alerts.txt
```

注意：

- 这个 callback 不要求 `--use_metis true`。如果 BF16 baseline 也开了 `--analyze_rollout true`，它也会注册 hook 和写 diagnostics；只是 gradient 参数名过滤主要面向 Metis/BitLinear，BF16 下可能没有多少 grad 记录。
- `effective_rank` 当前代码有时序问题：`_do_rank_check` 在 `on_step_end` 才被置为 true，但 forward hook 已经在该 step 之前跑完。因此当前很可能不会实际写出 effective rank。这个指标暂时不要作为有效实验结果。

### `ActivationCapture`

位置：`src/grpo/grpo.py`

开启条件：

- `--use_custom_analysis true`

它注册 forward hook 到固定的 `target_layers`。这些层包括第 0、11、23 层的 attention q/k/v/o 和 MLP up/gate/down。

它记录两类 forward：

- prefill：`seq_len > 1` 且 `x.requires_grad == false`，只更新 `current_prefill_len`。
- 训练阶段：`x.requires_grad == true` 且 module 在 training mode，计算 activation mean。

每条记录包含：

- `mean_full`：当前训练输入 flatten 后的 hidden-dim mean。
- `mean_prefill`：按上一次记录的 prefill 长度截取后计算的 hidden-dim mean。
- `shape`
- `prefill_len`

输出文件：

```text
<output_dir>/activation_analysis/layer_<layer_name>_rank0.pkl
```

注意：

- 只有 rank 0 写文件。
- 当前 `grpo.py` 里训练结束后没有调用 `capture.remove_hooks()`，所以文件句柄主要依赖进程退出时关闭。能落盘，但不是很干净。
- 这个开关默认很重，`run_experiment.sh` 默认关闭。

## 文本质量检测

实现位置：`src/grpo/rollout_quality.py`

入口：

- `analyze_sample()`
- `analyze_batch()`
- `aggregate_step()`
- `QualityLogger`
- `QualityAnalyzer`

### `QualityConfig`

默认规则阈值：

| 指标 | 默认阈值 | bad type |
| --- | ---: | --- |
| 空文本 | 直接 bad | `empty` |
| 字符数 `< 8` | 8 | `too_short` |
| 不可打印字符比例 `> 0.05` | 0.05 | `garbled` |
| 换行比例 `> 0.30` | 0.30 | `format_abnormal` |
| 特殊符号比例 `> 0.15` | 0.15 | `symbol_flood` |
| 标点比例 `> 0.50` | 0.50 | `punct_flood` |
| 省略号比例 `> 0.15` | 0.15 | `ellipsis_flood` |
| 2-gram 重复比例 `> 0.30` | 0.30 | `repetition` |
| 3-gram 重复比例 `> 0.20` | 0.20 | `repetition` |
| 连续相同 token 数 `> 6` | 6 | `token_loop` |
| 长文本 unique token ratio `< 0.12` | 0.12 | `low_diversity` |
| 中英切换次数 `> 4` | 4 | `mixed_lang` |

`grpo.py` 开启 rollout 分析时覆盖了几个默认项：

```python
QualityConfig(
    min_char_len=8,
    repeat_2gram_ratio_thresh=0.30,
    mixed_lang_switch_thresh=4,
    enable_repetition=False,
)
```

也就是说：当前 GRPO 入口默认关闭了 `repetition` 这个 bad type，但仍然保留 `token_loop`、`low_diversity`、`punct_flood` 等其他重复/退化相关规则。

### `QualityLogger`

写 sample-level 和 step-level 结果：

- sample-level：每条 rollout 文本的 bad_types 和原始指标。
- step-level：bad ratio、各 bad type ratio、metric mean/std/p50/p95、reward mean/std、good/bad reward mean。

### `QualityAnalyzer`

离线读取：

```text
<output_dir>/rollout_quality/step_quality.jsonl
```

生成图：

```text
<output_dir>/rollout_quality/quality_plots/bad_ratio_over_steps.png
<output_dir>/rollout_quality/quality_plots/reward_good_vs_bad.png
<output_dir>/rollout_quality/quality_plots/metric_trends.png
```

当前 `grpo.py` 在训练结束后会自动运行：

```python
QualityAnalyzer(os.path.join(training_args.output_dir, "rollout_quality")).plot_all()
```

如果 `--analyze_rollout false`，对应数据不存在，训练结束会打印 no data。

## Token/logit 分析

实现位置：`src/grpo/rollout_analysis.py`

入口：

- `AnalysisConfig`
- `analyze_sample()`
- `aggregate_step()`
- `AnalysisLogger`
- `AnalysisPlotter`

默认配置：

| 配置 | 默认值 | 作用 |
| --- | ---: | --- |
| `topk` | 5 | top-k overlap 的 k |
| `max_new_tokens` | 1024 | 每条 response 最多分析前 1024 个 token |
| `log_every_n_steps` | 1 | 每 step 都尝试分析 |
| `max_samples_per_step` | 128 | 每 step 最多分析 128 条 |
| `compute_mismatch` | `true` | 计算 rollout logits vs train full-forward logits |
| `compute_token_stats` | `true` | 计算 PPL、entropy、top-k overlap |

它计算两类东西：

1. token-level stats：
   - `token_logp`
   - `token_nll`
   - `token_ppl`
   - `entropy`
   - `top1_prob`
   - `topk_mass`
   - `actual_in_topk`

2. rollout-train mismatch：
   - `mean_prob_gap`
   - `mean_abs_prob_gap`
   - `max_abs_prob_gap`
   - `p95_abs_prob_gap`
   - `mean_prob_ratio`
   - `max_prob_ratio`
   - `p95_prob_ratio`
   - `ppl_rollout`
   - `ppl_train`

输出文件：

```text
<output_dir>/rollout_analysis/sample_analysis.jsonl
<output_dir>/rollout_analysis/step_analysis.jsonl
```

注意：

- 这部分当前由 `RolloutRewardWrapper` 间接调用。
- 它会对 `prompt + response` 再做一次 model forward，开销不小。
- 由于 rollout logits 捕获和 reward wrapper 的时序目前可能不完全对齐，`mismatch` 暂时应该当作实验性日志，不要直接作为论文主指标。

## 离线诊断绘图

### `DiagnosticsAnalyzer`

位置：`src/grpo/metis_monitor.py`

它读取：

```text
<output_dir>/metis_diagnostics.jsonl
<output_dir>/rollout.jsonl
```

可以画：

- rollout quality
- saturation rate
- effective rank
- grad norm
- activation stats

当前 `grpo.py` 里没有自动调用它，相关代码被注释掉了：

```python
# analyzer = DiagnosticsAnalyzer(training_args.output_dir)
# analyzer.print_summary()
# analyzer.plot_all()
```

所以默认只自动跑 `QualityAnalyzer`，不会自动跑 `DiagnosticsAnalyzer`。

### `AnalysisPlotter`

位置：`src/grpo/rollout_analysis.py`

它读取：

```text
<output_dir>/rollout_analysis/step_analysis.jsonl
```

可以画 mismatch、good/bad PPL/entropy、按 token 位置的曲线。

当前没有被 `grpo.py` 自动调用。

## 其他分析脚本

这些不是训练时自动检测，属于训练后手动分析工具：

| 文件 | 作用 | 默认是否自动运行 |
| --- | --- | --- |
| `src/grpo/activation_mean_ana.py` | 读取旧格式 activation `.pt` 文件，画 activation mean 曲线 | 否 |
| `src/grpo/mean_cross_step.py` | 读取 activation pickle 目录，分析跨 step mean、预测误差等 | 否 |
| `src/grpo/quality.py` | 硬编码一个 `QualityAnalyzer` 路径做离线绘图 | 否 |

## 当前注意点

1. `run_experiment.sh` 默认会打开 `ANALYZE_ROLLOUT=true`，因此第一阶段实验会默认写 rollout 文本质量日志、Metis diagnostics、rollout analysis 文件。
2. `run_experiment.sh` 默认关闭 `USE_CUSTOM_ANALYSIS=false`，因此不会默认写 activation mean pickle。
3. `run_grpo.sh` 目前同时打开 `--analyze_rollout true` 和 `--use_custom_analysis true`，输出会比较重。
4. QAT 脚本当前没有接入这些检测类，只写 QAT loss 和 checkpoint。
5. `effective_rank` 当前实现很可能不会实际记录，需要修 callback 时序后再使用。
6. `rollout_analysis` 的 mismatch 指标当前可能存在 step 对齐问题，需要修 capture -> wrapper 的传递时序后再作为正式指标。
