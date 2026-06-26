# 后训练低精度项目中期答辩整理

## 1. 项目定位

本项目研究大语言模型后训练阶段的低精度训练，重点关注 DPO 与 GRPO 两类偏好/强化学习后训练算法，目前主线集中在 GRPO 的 FP4 训练可行性上。

后训练不同于普通监督微调。以 GRPO 为例，训练流程需要当前 policy model 先 rollout 生成答案，再根据答案质量计算 reward，并用组内相对优势更新模型。因此，低精度不仅影响反向传播，也直接影响 rollout 质量。一旦低精度 rollout 本身出现退化，奖励信号会变弱甚至失真，RL 会在错误或低质量样本上继续优化。

当前阶段的核心故事线可以概括为：

1. FP4 后训练的首要瓶颈不是简单的数值溢出，而是 rollout 质量和 rollout/train 分布一致性。
2. 针对 rollout 质量问题，项目实现了 rollout 质量分析器，并探索 QAT self-distillation 作为 FP4 进入 RL 前的预适配方法。
3. 针对 FP4 量化精度不足问题，项目在 Metis BitLinear 中实现了 mean-bias FP4 方法，用移动均值降低激活动态范围。
4. 目前实验显示，直接 FP4 GRPO 不稳定；经过 QAT self-distillation 后再进行 FP4 GRPO，训练信号和 rollout 质量明显改善。

## 2. 项目代码结构

与答辩相关的主要文件如下：

- `grpo/grpo.py`：GRPO 主训练脚本，基于 TRL `GRPOTrainer`，接入自定义数学准确率 reward、Metis FP4 层替换、rollout 质量分析和诊断回调。
- `grpo/qat_distill.py`：QAT self-distillation 实现，使用 bf16 teacher 生成回答并提供 logits 软标签，训练 Metis FP4 student。
- `grpo/rollout_quality.py`：rollout 文本质量分析器，基于规则统计 bad sample ratio、重复、过短、乱码、低多样性、中英混杂等指标，并输出 jsonl/csv/图片。
- `grpo/metis_monitor.py`：训练诊断模块，记录 rollout、reward、激活统计、梯度统计、rollout/train logits mismatch。
- `Metis/Metis/bitlinear.py`：低精度线性层核心实现，包含 FP4 量化、SVD/mean 两种 Metis 模式，以及 mean cache 逻辑。
- `main-1.pdf`：mean-bias FP4 方法和早期实验总结。

## 3. GRPO 后训练任务设置

当前 GRPO 实验使用 Qwen2.5-0.5B-Instruct 作为基座模型，数据集为 `deepmath-103K`，reward 使用自定义 `accuracy_reward`。该 reward 首先尝试解析标准答案和模型答案中的数学表达式，并用 `math_verify` 判断是否等价；对于 True/False/Yes/No 等简单答案，则使用末尾词匹配。

训练配置的共同点：

- `num_generations = 8`：每个 prompt 生成 8 个 completion，用于 GRPO 组内比较。
- `max_completion_length = 256`
- `temperature = 1.0`
- `top_p = 1.0`
- `learning_rate = 1e-6`
- `beta = 0.0`
- `epsilon = 0.2`

本次准备用于答辩对比的三组实验：

| 实验 | 目录 | 说明 |
|---|---|---|
| 直接 FP4 GRPO | `grpo/Qwen2_5-0.5B-grpo/runs/Mar10_06-28-55_trl-trian-1--c344722add84-az6slw7r53` | 从原始 Qwen2.5-0.5B-Instruct 直接替换 Metis FP4 后训练 |
| QAT self-distill 后 FP4 GRPO | `grpo/Qwen2_5-0.5B-grpo-metis-after-qat/runs/Mar12_03-27-23_trl-trian-1--c344722add84-az6slw7r53` | 先做 QAT self-distillation，再进入 Metis FP4 GRPO |
| bf16 GRPO baseline | `grpo/Qwen2_5-0.5B-grpo-bs-8/runs/Mar15_09-52-41_trl-trian-1--c344722add84-mphxxja6sx` | 不启用 Metis，作为高精度训练参考 |

## 4. 问题一：低精度 rollout 质量退化

### 4.1 问题定义

GRPO 的训练信号来自模型自己的 rollout。如果 FP4 量化导致生成文本在训练早期就出现乱码、重复、低多样性或语义断裂，那么 reward 函数即使本身正确，也只能在一组低质量回答中选择“相对没那么差”的样本。长期来看，这会导致：

- reward 方差不足，组内比较信号变弱；
- policy update 方向受噪声影响；
- clipped ratio 长期偏高或震荡；
- 训练无法稳定收敛，甚至强化退化模式。

### 4.2 Rollout 质量分析器

项目实现了轻量规则型 rollout 质量分析器，不依赖额外模型，适合训练过程中在线记录。它对每条 completion 计算：

- 长度类：空文本、过短文本、字符长度；
- 格式类：不可打印字符比例、换行比例、特殊符号比例、标点比例、省略号比例；
- 重复类：2-gram/3-gram 重复率、连续相同 token 最大长度、最长重复子串；
- 多样性：unique token ratio；
- 语言混杂：中英字符段切换次数和次要语言占比；
- reward 关联：分别统计 good/bad sample 的 reward 均值。

训练过程中，`RolloutRewardWrapper` 包装 reward function，在 reward 计算时同步记录 rollout 原文、质量判定和聚合指标。输出包括：

- `rollout_quality/sample_quality.jsonl`
- `rollout_quality/step_quality.jsonl`
- `rollout_quality/quality_plots/bad_ratio_over_steps.png`
- `rollout_quality/quality_plots/metric_trends.png`
- `rollout_quality/quality_plots/reward_good_vs_bad.png`

答辩中建议使用以下两组图片对比 rollout 质量：

- `grpo/Qwen2_5-0.5B-grpo/rollout_quality/quality_plots`
- `grpo/Qwen2_5-0.5B-grpo-metis-after-qat/rollout_quality/quality_plots`

### 4.3 当前观察

从已保存的 `step_quality.jsonl` 聚合看：

| 指标 | 直接 FP4 GRPO | QAT 后 FP4 GRPO |
|---|---:|---:|
| step 范围 | 0-659 | 0-763 |
| 平均 bad ratio | 0.5781 | 0.1959 |
| bad ratio 中位数 | 0.9375 | 0.1875 |
| 最后 10 条记录 bad ratio 均值 | 0.8320 | 0.2070 |
| 平均 reward | 0.0601 | 0.2424 |
| 最后 10 条记录 reward 均值 | 0.0477 | 0.2312 |

这说明直接 FP4 GRPO 中，rollout 质量很快恶化，bad ratio 长期处于高位；QAT self-distillation 后，bad ratio 明显降低，reward 也更接近可训练状态。

## 5. 方法一：QAT self-distillation 预适配

### 5.1 方法动机

直接把 bf16 模型替换为 FP4 路径后进入 GRPO，相当于要求 RL 同时完成两件事：

1. 学会数学任务上的偏好优化；
2. 适应 FP4 量化带来的表示和 logits 偏移。

这对 GRPO 来说过重，因为 RL 依赖 rollout 质量。如果进入 RL 前模型已经因量化退化，后续训练很难恢复。因此项目引入 QAT self-distillation：先让 FP4 student 模仿 bf16 teacher 的输出分布，再进行 GRPO。

### 5.2 算法流程

`grpo/qat_distill.py` 中的流程如下：

1. 加载同一个基座模型两份：
   - teacher：bf16，冻结参数；
   - student：bf16 参数存储，但线性层替换为 Metis BitLinear，前向走 FP4 量化路径。
2. 对每个 prompt，teacher 使用采样生成 response。
3. 将 prompt 和 teacher response 拼接，teacher 做 full-forward，得到 response token 位置上的 logits。
4. student 在同一 prompt-response 上做 full-forward，得到 student logits。
5. 只在 response 有效 token 上计算 KL 蒸馏损失：

   `KL(P_teacher || P_student)`

6. 使用温度系数进行标准蒸馏缩放：

   `loss = masked_KL * T^2`

7. 为降低显存开销，代码支持只保存 teacher logits 的 top-k：
   - `kl_top_k = 100` 时，仅取 teacher top-100 logits 和 indices；
   - student 侧用 `torch.gather` 取相同 vocabulary 位置；
   - 在 top-k 子分布上近似计算 KL。

### 5.3 当前 QAT 配置

当前 QAT 配置保存在：

`grpo/Qwen2_5-0.5B-instruct-fp4/qat_script_arguments.json`

关键配置：

- teacher/student 来源：Qwen2.5-0.5B-Instruct；
- 数据集：`deepmath-103K`；
- `num_train_epochs = 1`
- `per_device_train_batch_size = 32`
- `gradient_accumulation_steps = 8`
- `learning_rate = 1e-6`
- `max_prompt_length = 512`
- `max_new_tokens = 1024`
- `use_metis = true`
- `kl_temperature = 1.0`
- `kl_top_k = 100`

### 5.4 方法贡献总结

QAT self-distillation 的价值在于把“量化适应”从 RL 阶段前移。它不是直接优化 reward，而是先恢复 FP4 路径下的基础生成分布，使后续 GRPO 能在更可靠的 rollout 上学习。当前实验中，QAT 后 FP4 GRPO 的 reward、clipped ratio、bad ratio 均显著优于直接 FP4 GRPO。

## 6. 问题二：FP4 激活量化精度不足

### 6.1 问题定义

FP4 动态范围和有效表示级数都很有限。若直接对激活 `x` 做 FP4 量化，激活均值偏移会占用大量量化范围，导致真正有区分度的 residual 信息被压缩。对于后训练，尤其是 GRPO rollout，这种前向精度损失会直接反映在生成 token 分布上。

早期诊断显示：

- FP4 训练中未观察到 NaN/Inf，说明主要问题不是数值溢出；
- 激活 abs_max 相对稳定；
- 但 rollout 质量早期退化，reward 峰值偏低，clipped ratio 不稳定；
- 因此核心问题更可能是 FP4 前向精度不足和 rollout/train 分布偏移。

### 6.2 Mean-bias FP4 方法

`main-1.pdf` 和 `Metis/Metis/bitlinear.py` 中实现的 mean-bias 思路是：不直接量化 `x`，而是先减去均值，只量化残差：

```text
x_hat = QuantFP4(x - mu) + mu
```

训练阶段，输入形状为 `(B, S, H)` 时，均值沿 batch 和 sequence 维计算：

```text
mu = mean(x, dim=(B, S))
```

也就是说，每个 hidden channel 有一个均值，量化器主要处理以 0 为中心的残差 `x - mu`。这样可以降低残差动态范围，让 FP4 的有限表示能力集中在更重要的变化部分。

推理阶段分为 prefill 和 decode：

- prefill：对完整 prompt 的激活计算一次均值，并写入 `mean_cache`；
- decode：每步只有新 token，使用移动均值更新缓存；
- 每次量化都使用当前缓存均值做 bias removal。

代码对应位置在 `LinearLowbitFunction.svd_quant()` 内部的 `_mean_quant_single()`：

- 首次或 prefill 时：`me = input_flat.mean(dim=0, keepdim=True)`；
- decode 且 cache 已存在时：

  ```text
  me = (cached_mean * cached_count + this_mean * new_count_delta) / total_count
  ```

- 然后执行 residual FP4 quant/dequant：

  ```text
  input_res = input_flat - me
  input_res = quant(input_res)
  input_res = rquant(input_res)
  output = me + input_res
  ```

### 6.3 Metis BitLinear 中的实现

`Metis/Metis/bitlinear.py` 中 `BitLinear` 支持两类 Metis 模式：

- `metis_mode = "svd"`：对激活或梯度做低秩 SVD 分解，低秩部分和残差分别处理；
- `metis_mode = "mean"`：使用 mean-bias residual quantization。

当前 `grpo/grpo.py` 中 `MetisArgs` 默认设置：

- forward input/weight 使用 `nvfp4e2m1bnosr`；
- backward input/weight/outputgrad 使用 `nvfp4e2m1b`；
- `enable_activation_svd = True`，但 `metis_mode = "mean"`，实际进入 mean 分支；
- `enable_backward_svd = True`；
- `forward_svd_rank = 64`；
- target modules 为 Qwen 的 `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`。

注意：当前仍存在一个关键 mismatch。训练时可以看到完整 response，均值按完整序列计算；推理 decode 时只能基于 prefill 和已生成 token 做移动均值，不能访问未来 token。因此 mean-bias 方法虽然改善了 residual 量化，但 rollout/train 的均值估计方式仍不完全一致。

## 7. 实验结果整理

### 7.1 直接 FP4 GRPO

实验目录：

`grpo/Qwen2_5-0.5B-grpo`

特征：

- 从原始 Qwen2.5-0.5B-Instruct 直接替换 Metis FP4；
- rollout 质量退化严重；
- `step_quality` 平均 bad ratio 为 0.5781，中位数高达 0.9375；
- trainer state 中 reward 最后为 0.1000，最后 5 次日志均值 0.0903；
- clipped ratio 最后 5 次日志均值 0.6337，说明更新仍处于高裁剪状态；
- entropy 从早期较高值下降到约 0.048，存在分布变窄/退化风险。

### 7.2 QAT self-distill 后 FP4 GRPO

实验目录：

`grpo/Qwen2_5-0.5B-grpo-metis-after-qat`

特征：

- model 来源为 `Qwen2_5-0.5B-after-QAT-fp4`；
- 平均 bad ratio 降至 0.1959；
- reward 平均值提升至 0.2424；
- trainer state 中 reward 最后为 0.2844，最后 5 次日志均值 0.2706；
- clipped ratio 最后 5 次日志均值约 0.0014，基本回到稳定区间；
- completion 平均长度从早期约 253 降至最后约 40，说明模型更快生成终止答案，训练信号更集中。

### 7.3 bf16 GRPO baseline

实验目录：

`grpo/Qwen2_5-0.5B-grpo-bs-8`

特征：

- 不启用 Metis；
- 使用 bf16 作为高精度参考；
- 该实验运行步数更长，batch size 为 8；
- rollout/train mismatch 早期很小，例如 step 4 的 `mean_abs_prob_gap_mean` 约 0.0037，说明 bf16 rollout 与训练 forward 的概率分布高度一致；
- 可作为判断 FP4 mismatch 和 rollout 退化程度的对照。

### 7.4 可用于 PPT 的结果表

| 对比项 | 直接 FP4 | QAT 后 FP4 | bf16 baseline |
|---|---:|---:|---:|
| 是否启用 Metis | 是 | 是 | 否 |
| 是否经过 QAT self-distill | 否 | 是 | 否 |
| 主要模型来源 | 原始 instruct | QAT 后 fp4 | 原始 instruct |
| 平均 bad ratio | 0.5781 | 0.1959 | 0.6545* |
| rollout reward 平均 | 0.0601 | 0.2424 | 0.2925* |
| trainer 最后 reward | 0.1000 | 0.2844 | 当前目录未找到 trainer_state |
| 最后 clipped ratio | 0.6406 | 0.0000 | 当前目录未找到 trainer_state |

注：表中 rollout 质量统计来自各实验目录下的 `rollout_quality/step_quality.jsonl`；直接 FP4 和 QAT 后 FP4 的 trainer 指标来自 checkpoint 中的 `trainer_state.json`。bf16 baseline 的 `step_quality` 运行步数和 batch size 与前两组不同，且后期 bad ratio 受训练后分布变化影响较大，因此答辩中更适合使用它作为高精度 rollout/train mismatch 参考，而不是直接与前两组逐项比较。

## 8. 当前阶段结论

1. 直接 FP4 GRPO 的主要风险是 rollout 质量早期退化，而不是 NaN/Inf 这类显式数值错误。
2. rollout 质量分析器已经可以把主观观察转成可量化指标，能够支撑“直接 FP4 生成质量差、QAT 后改善”的论证。
3. mean-bias FP4 通过量化 residual 而不是原始 activation，缓解了 FP4 动态范围不足问题，是当前 Metis FP4 路径的核心改进。
4. QAT self-distillation 能显著改善 FP4 进入 GRPO 前的状态，使 reward 更高、bad ratio 更低、clipped ratio 更稳定。
5. 仍需进一步验证：QAT 训练步数、数据规模、下游任务损失、mean cache 与训练均值 mismatch 对最终效果的影响。

## 9. 下一步工作

### 9.1 QAT self-distillation 深入探索

需要回答的问题：

- QAT self-distillation 训练多少步最合适？
- 只做少量 warmup 是否足够恢复 rollout 质量？
- QAT 会不会损害 bf16 原模型能力或下游任务性能？
- top-k KL 中 `k=100` 是否足够？是否需要比较 `k=50/100/200/full logits`？
- 是否需要加入 CE hard-label loss，与 KL loss 混合？
- QAT 后的模型在进入 GRPO 前，纯 decode 质量和数学准确率能恢复到什么程度？

建议实验：

- 固定数据集，比较 QAT steps：100/500/1000/1 epoch；
- 固定 QAT checkpoint，做 GSM8K/MathQA/DeepMath 子集评测；
- 比较 QAT 前后 rollout quality；
- 比较 QAT 前后 FP4 logits 与 bf16 logits 的 KL、top-k overlap、token-level PPL。

### 9.2 Mean-bias FP4 方法验证

需要回答的问题：

- mean-bias FP4 在多个数据集上是否都能稳定训练？
- prefill mean、decode moving mean、training full-sequence mean 的差距有多大？
- mismatch 是否随生成长度累积？
- 哪些层对 mean 估计误差最敏感？
- mean cache 是否需要分层、分阶段或按 position 修正？

建议实验：

- 在 DeepMath、GSM8K、通用 instruction 数据上验证 mean-fp4 GRPO；
- 记录每层 prefill mean、decode mean、training mean 的 cosine similarity 和 L2 error；
- 画 token position 维度上的 mismatch 曲线，定位退化从 early token 还是 long generation 中后段开始；
- 比较 `metis_mode="mean"` 与 `metis_mode="svd"` 以及直接 FP4 的差别。

### 9.3 GRPO 抗噪机制

后续可以在 RL 阶段加入：

- rollout 异常样本过滤或降权；
- 对 bad sample group 降低学习权重；
- 调整 clip range、learning rate、KL penalty；
- 将 rollout quality 指标加入训练 dashboard，用于早停或自动告警。

## 10. PPT 建议结构

1. 背景：后训练为什么需要 rollout，低精度为什么更难。
2. 项目目标：面向 DPO/GRPO 的 FP4 后训练，本阶段聚焦 GRPO。
3. 问题发现：直接 FP4 GRPO rollout 退化，reward 信号弱。
4. 工具建设：rollout 质量分析器，将文本退化量化。
5. 方法一：QAT self-distillation，先适应 FP4 再进入 RL。
6. 方法二：mean-bias FP4，量化 residual 降低动态范围。
7. 实验对比：直接 FP4、QAT 后 FP4、bf16 baseline。
8. 结果展示：bad ratio 图、reward 图、clipped ratio 图、rollout/train mismatch 图。
9. 当前结论：QAT 后 FP4 明显改善，但 mean mismatch 仍需解决。
10. 下一步：QAT 步数/下游损失、多数据集 mean-fp4、抗噪 GRPO。

## 11. 答辩时可强调的一句话

本项目目前的核心发现是：FP4 后训练失败并不是简单的训练不收敛，而是低精度前向首先破坏了 rollout 质量，使 GRPO 的奖励信号和更新方向都被污染；因此我们从两个层面解决问题，一方面用 mean-bias FP4 改善量化前向精度，另一方面用 QAT self-distillation 在 RL 前恢复 FP4 policy 的生成分布。
