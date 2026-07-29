# `fp4_post_aaai_draft.tex` 表述与一致性检查

审读对象：`AuthorKit27/fp4_post_aaai_draft.tex`（当前共 346 行）

检查范围：需要修改的表述、前后不一致、术语或对象的多重定义，以及机器生成感较强的写法。行号均对应本次检查时的源文件。本文没有直接改动原稿，也没有核验引用文献是否正确。

## 总体判断

稿件的核心问题不是语法错误，而是“方法名称、实验条件和结论归因”没有完全对齐。最需要先处理的是以下几点：

1. 文中把第一阶段交替称为 QAT、QAD、self-distillation 和 QAT self-distillation，但 Related Work 又明确区分了 QAT 与 QAD。
2. Direct FP4、NV-Hadamard FP4、moving-mean FP4 和完整两阶段方案没有形成统一的条件表；正文、主结果表和 reward 表覆盖的条件不同。
3. Direct FP4 与完整方案同时改变了蒸馏、权重分解和 activation/gradient quantization。现有对比只能支持“完整配置优于 Direct FP4”，不能单独证明提升来自 QAT。
4. 标题和贡献项把 moving mean 笼统地写成 causal，但训练阶段使用 full-sequence mean，只有 autoregressive decoding 使用 causal cache。
5. 蒸馏公式写的是完整词表 KL 的求和，实际描述和默认设置却是 top-\(k\) 子分布，并且正文公式没有体现有效 token 归一化。
6. “preliminary run”“available diagnostic run”“final matched comparison”和“establish”级别的结论同时出现，证据口径从暂定结果跳到了确定性结论。
7. 最明显的机器味来自结果的多次复述、抽象评价词过密，以及用 `Taken together`、`consistent picture`、`establish` 等套话代替具体数字。

稿件没有明显的聊天残留、emoji、粗体滥用或过量破折号。章节标题使用 Title Case、技术复合词使用连字符、方法段采用被动语态，都可能是正常的会议论文写法，不应机械地按照一般“去 AI 味”规则删除。

## 一、必须优先解决的前后不一致

### 1. 实验条件在不同位置对不上

涉及位置：第 30、202、208、250、270、281--294、312--322 行。

当前各处出现的条件如下：

| 条件 | 第 30 行注释中的计划 | 第 202 行“四个条件” | 主结果表 | Reward 表 |
|---|---:|---:|---:|---:|
| Initial/pretrained checkpoint | 未列 | 有 | 有 | 无 |
| BF16 GRPO | 有 | 有 | 有 | 有 |
| Direct FP4 GRPO | 有 | 无 | 无 | 有 |
| NV-Hadamard FP4 GRPO | 有 | 有 | 有，但全为 `--` | 无 |
| QAT/QAD + moving-mean FP4 GRPO | 有 | 有 | 有 | 有 |

问题：

- 第 202 行声称是 “final matched comparison”，但排除了后文反复讨论的 Direct FP4。
- 主结果表保留了没有结果的 NV-Hadamard 行，却没有 Direct FP4 行。
- Reward 表有 Direct FP4，却没有 NV-Hadamard。
- “pretrained checkpoint”不是 post-training condition。对 `Qwen2.5-0.5B-Instruct` 而言，称其为 `Pretrained` 也容易让人误以为是 base pretrained model。

建议先建立一张唯一的 condition matrix，并让摘要、Experiments、所有表格和图例照抄同一组名称。若当前只有部分结果，应把第 202 行改为：

> We report the conditions for which completed runs are available. The full matched comparison will additionally include ...

如果最终确实要做完整对比，建议统一为：

- Initial checkpoint（evaluation reference，不算 training condition）
- BF16 GRPO
- Direct FP4 GRPO
- NV-Hadamard FP4 GRPO
- Moving-mean FP4 GRPO without distillation
- QAD + moving-mean FP4 GRPO

其中后两项是否都保留，取决于作者是否要把 QAD 的作用与 moving mean 的作用分开。

### 2. 目前不能把完整配置的收益单独归因于 QAT

涉及位置：第 29、41、46、250、259、301、331、335、341 行。

Direct FP4 与 `QAT + moving mean` 的差别至少包括：

- 是否经过 teacher--student distillation；
- 是否使用 rank-64 weight decomposition；
- activation/gradient 是否使用 mean-residual quantization；
- decoding 是否使用 moving-mean cache。

因此，第 250 行的

> QAT pre-adaptation substantially reduces this rate

以及第 335 行的

> QAT self-distillation addresses this failure

都把组合效应写成了 QAT 的单独因果效应。没有 `moving mean only` 和 `QAT only` 等消融时，更稳妥的写法是：

> The full QAD + moving-mean configuration has a lower bad-rollout rate than direct FP4 over the logged steps.

若要保留 “QAT stabilizes” 这一结论，需要补齐至少以下消融：

1. Direct FP4；
2. moving mean without distillation；
3. distillation followed by direct FP4；
4. distillation followed by moving-mean FP4。

### 3. QAT 与 QAD 的关系没有说清楚

涉及位置：标题及第 29、46、63、73、168--184、208、217、238 行。

第 63 行把两者定义为：

- QAT：通过 fine-tuning 适应量化 forward path；
- QAD：用高精度 teacher 对齐 quantized student。

但本文第一阶段使用 teacher logits 和 KL loss，按该定义更接近 QAD。后文却持续称它为 `QAT self-distillation` 或 `QAT pre-adaptation`。

这不一定是方法错误，但必须选择一种稳定口径：

- 方案 A：全篇称为 `quantization-aware distillation (QAD)`；
- 方案 B：明确说明 “we use QAT as the umbrella term; the Stage-I objective is QAD”，此后用 `QAD stage` 指具体 loss，用 `two-stage QAT-to-GRPO pipeline` 指整体框架。

不建议同时无解释地使用 `QAT self-distillation`、`quantization-aware self-distillation` 和 `QAD`。

### 4. “causal moving mean”只适用于 decode，不适用于整个训练路径

涉及位置：标题及第 41、47、73、115--128、159--166、208、226、339 行。

第 159--166 行明确说明：

- training forward 使用 full-sequence mean；
- decoding 使用 prefix information 更新 moving-mean cache；
- 两者存在 train--decode mismatch。

因此，第 47 行

> a causal moving-mean residual FP4 implementation for both training and autoregressive decode

会让读者理解成训练阶段也使用 causal mean。建议改为：

> mean-residual FP4 for training, together with a causal running-mean variant for autoregressive decoding

标题也可改为 `... with Mean-Residual Quantization and Causal Decode-Time Averaging`，或者在方法开头明确 “causal” 仅修饰 decode-time cache。

### 5. Figure 1 caption 与权重量化公式矛盾

涉及位置：第 78、104--109 行。

第 78 行说：

> only the residual matrices use FP4 quantization

但式（3）同时量化 \(V_r^\top\) 和 \(R_W\)：

\[
U_r\Sigma_r Q_{\mathrm{FP4}}(V_r^\top)+Q_{\mathrm{FP4}}(R_W).
\]

因此 “only the residual matrices” 不成立。可改为：

> For activations and output gradients, FP4 is applied only to the centered residuals. In the weight path, both \(V_r^\top\) and \(R_W\) are quantized, while \(U_r\) and \(\Sigma_r\) remain in higher precision.

### 6. “fixed decomposition”与“trainable master parameters”边界不清

涉及位置：第 78、93、97、104 行。

文中同时出现：

- `fixed weight decomposition`；
- `trainable master parameters remain in higher precision`；
- `We factor each selected weight matrix before post-training`。

读者无法判断固定的是：

- 只固定 rank 和计算图；
- 固定 \(U_r,\Sigma_r,V_r,R_W\)；
- 只在初始化时做一次 SVD，之后所有 factor 继续训练；
- 还是每次更新 master weight 后重新分解。

对当前仓库实现的交叉检查显示，这些 factor/residual 作为参数参与模型计算。因此，`fixed weight decomposition` 很可能应改成 `a fixed decomposition topology initialized by SVD`，并明确哪些参数更新、是否重新分解。

### 7. 蒸馏目标的完整 KL、top-\(k\) KL 与归一化没有对齐

涉及位置：第 170--176、224 行。

式（10）写的是完整词表分布上的

\[
T^2\sum_{t\in y}\mathrm{KL}(p_T\|p_S),
\]

但第 176 行和 protocol 表说明默认只保留 teacher top-\(k\) logits，\(k=100\)。当前实现是在 teacher 的 top-\(k\) support 上分别重新归一化 teacher 和 student，再计算近似 KL；这不是完整词表 KL。

另外，式（10）使用求和，而实现对有效 response tokens 做均值。两者会导致 loss 随回答长度产生不同的尺度。

建议：

1. 先给出 full-vocabulary objective；
2. 再定义 \(I_t^{(k)}\) 为 teacher top-\(k\) token set；
3. 明确 \(\tilde p_T^{(k)}\) 和 \(\tilde p_S^{(k)}\) 都在该集合上重新归一化；
4. 把公式写成除以有效 response-token 数量的 masked mean；
5. 称其为 `top-k-renormalized KL approximation`，不要把它直接写成完整 KL。

### 8. “reward”“relative reward”“advantage”和训练日志混用

涉及位置：第 29、37、39、71、78、176、224、259、264、329 行。

当前至少有五种相关对象：

- task reward \(r(x,y_i)\)；
- mathematical-equivalence/accuracy reward；
- group-normalized advantage \(\widehat A_i\)；
- TensorBoard `train/reward`；
- 泛称的 `reward signal` 或 `group-relative reward`。

GRPO 中通常是 task reward 经过组内处理后形成 relative advantage，而不是 reward 本身必然 “group-relative”。建议全文采用：

- `task reward`：单条 completion 的判分；
- `group-relative advantage`：组内标准化后的训练量；
- `mean logged task reward`：图和表中的 `train/reward`；
- `learning signal`：只有在泛指 reward/advantage 对更新的共同作用时使用。

第 224 行的 `group-relative accuracy reward` 尤其容易混淆，建议改为 `exact-match/equivalence task reward; group-normalized advantages`。

### 9. teacher、reference policy、rollout policy 没有建立关系

涉及位置：第 41、82、93、170--187 行。

文中先后出现：

- BF16 teacher；
- BF16 counterpart；
- BF16 reference policy；
- current policy；
- rollout policy；
- FP4 student/adapted policy。

第 93 行突然引入 `BF16 reference policy`，后面没有说明它是否是 GRPO 的 KL reference、是否与 teacher 共用 checkpoint、是否冻结。建议在 Method 开头集中定义：

> \(\pi_T\) is the frozen BF16 distillation teacher; \(\pi_{\mathrm{ref}}\) is the frozen GRPO reference policy; \(\pi_{\mathrm{old}}\) is the rollout policy used to compute probability ratios; and \(\pi_\theta\) is the trainable FP4 policy.

如果实际实现中没有某个对象，就不要为了套用标准术语而保留它。

### 10. 摘要中的绝对结论与 7B Direct FP4 结果不符

涉及位置：第 29、321 行。

摘要说坏 rollout 会

> prevent stable policy optimization

但 Reward 表中 7B Direct FP4 从 0.1641 上升到 0.2949。即使它弱于其他条件，也不能概括为所有规模都 “prevent” optimization。建议改为：

> can weaken or destabilize policy optimization, particularly in the 0.5B and 1.5B runs

### 11. “simulation”与“降低实际成本”的论证没有分开

涉及位置：第 29、35、93 行。

摘要称实现为 `NVFP4 simulation`，Introduction 又把 FP4 的价值写成降低 memory/compute cost，但文中没有报告真实硬件吞吐、显存或 wall-clock 指标。模拟 QDQ 可以研究数值行为，却不自动证明系统成本下降。

建议区分：

> FP4 hardware can reduce training cost; this paper evaluates numerical behavior with an NVFP4 simulation and does not claim measured hardware speedups.

如果已有真实 FP4 kernel 和系统测量，则应补充硬件、kernel、吞吐和峰值显存数据。

### 12. “preliminary / available”与“final / establish”证据等级冲突

涉及位置：第 29、202、250、270、329--343 行。

稿件一方面说：

- `Preliminary controlled runs`；
- `available ... diagnostic run`；
- NV-Hadamard 结果尚缺；

另一方面又说：

- `final matched comparison`；
- `demonstrating`；
- `These findings establish`；
- `essential for reproducible ... research`。

在结果不完整、没有 seed-level variance、Reward 表只报首尾点时，应统一为探索性语气：

- `the logged runs indicate`；
- `is consistent with`；
- `the current evidence supports`；
- `we do not yet isolate`；
- `should be verified with matched seeds`。

### 13. Protocol 说“明确设置预算”，表中却没有预算

涉及位置：第 213--228 行。

caption 声称：

> QAT and GRPO step budgets are set explicitly for each reported run.

但表内没有 QAT steps、GRPO steps、epochs 或 seen prompts。建议增加 `Training budget` 行，并给出每个规模的实际值；如果不同 run 不同，则不要称为 matched comparison。

### 14. 结论要求报告的信息在正文中没有报告

涉及位置：第 204、208--227、343 行。

最后一句说完整 quantization recipe、decode configuration、hardware/software environment 和 seed variation 对复现 “essential”，但当前稿件没有给出硬件/软件版本与 seed-level variation，decode 配置也只给出部分参数。

有两种处理方式：

- 在 Experiments 或 appendix 补齐这些信息；
- 将第 343 行改成明确的 limitation/future reporting commitment，而不是像已经满足了复现要求。

### 15. 模型名称与方法标签不一致

涉及位置：第 200、281、284、289、294 行。

- 第 200 行是 `Qwen2.5-0.5B-Instruct`，表中写成 `Qwen2.5-0.5B`。
- `Pretrained` 对 Instruct checkpoint 不准确，建议统一为 `Initial checkpoint`。
- `QAT + moving mean` 这一表格标签省略了后续 GRPO，容易被理解成只做了 QAT。建议统一为 `QAD + moving-mean FP4 GRPO`，或使用作者最终选定的名称。

第 284 行还建议核对原始 EvalScope 报告：`MATH-500 = 6` 与 `ARC = 29.6` 的组合中，29.6 恰好等于同一模型 initial checkpoint 的 MATH-500 分数，存在复制或列错位的可能。即使数值正确，也应把 `6` 统一格式化为 `6.0` 或与全表相同的小数位数。

## 二、存在多重定义或边界不清的术语

| 术语 | 当前用法 | 风险 | 建议的唯一口径 |
|---|---|---|---|
| QAT | 泛指量化适应；也专指 teacher--student KL stage | 与第 63 行的 QAD 定义冲突 | QAT 作上位概念，Stage I 明确称 QAD；或全篇只用 QAD |
| Direct FP4 | NVFP4 simulation、direct-FP4 path、禁用 decomposition/residual 的 baseline | 与 NV-Hadamard 的边界只在部分位置出现 | 在 Experiments 用一张配置表定义 W/A/G、RHT、distillation 和 initialization |
| Proposed condition | 有时指 rank-64 weight path，有时指 moving mean，有时指完整两阶段 pipeline | 读者不知道比较单位 | 只用完整方法名；谈组件时分别写 `rank-64 weight path`、`mean-residual A/G path` |
| FP4 policy/path/operator | 有时指整个模型，有时只指 selected linear projections，有时只是 QDQ simulation | 容易夸大 FP4 覆盖范围 | 明确 `selected linear projections use simulated FP4 operands; master parameters and specified branches remain BF16` |
| Moving mean | moving mean、moving-mean residual、mean bias、mean-residual、causal cache | 同一组件像多个方法 | 训练称 `full-sequence mean-residual quantization`，解码称 `causal running-mean cache` |
| Reward | accuracy reward、equivalence reward、group-relative reward、reward signal、`train/reward` | reward 与 advantage 混淆 | 使用上一节给出的四级命名 |
| Cache | 第 225 行仅写 `cache enabled` | 不知道是 KV cache、quantized-weight cache 还是 moving-mean cache | 写出准确 cache 类型；若多个都开，分别列出 |
| Bad rollout | rule-based aggregate、multi-label category、good/bad response class | 阈值和聚合逻辑没有定义 | 给出每个规则、阈值以及 aggregate label 的 OR/加权逻辑 |
| Mean over \(L\) | 第 115 行把 batch 与 sequence 合并；第 159 行又写 token-counted cache | 不清楚 mean 是 per sequence、per group、per batch 还是 per layer | 明确 reduction axes、padding mask 和 cache ownership |

### 关于 mean 的额外定义问题

式（4）用 \(L\) 合并 batch 与 sequence positions，但实际序列通常含 padding。第 157 行又提到 token masks，第 159 行说只统计 “available prompt positions”。建议把均值直接写成 masked mean，并说明：

- 是否每个 sequence 单独缓存；
- 同一 prompt 的 \(G\) 个 completion 是否共享 prefill mean、之后各自更新；
- 不同 batch element 是否会互相影响均值；
- \(n_t\) 是否只统计非 padding token；
- cache 是 per layer、per projection，还是全模型共享。

否则 “causal” 只说明没有使用本序列未来 token，仍不足以说明不同序列之间是否发生耦合。

## 三、需要修改的具体表述

下表按重要性列出代表性位置。建议不是要求逐字照抄，而是给出更明确的方向。

| 行号 | 原表述或关键词 | 问题 | 建议 |
|---:|---|---|---|
| 29 | `large-language-model post-training` | 不自然的复合连字符 | `large language model post-training` |
| 29 | `can corrupt early rollouts` | `corrupt` 偏情绪化，且没有给频率 | `increases the incidence of repetition, mixed-language output, and early termination in the logged initial rollouts` |
| 29 | `group-relative reward signals` | reward/advantage 混淆 | `within-group reward variation and the resulting advantage estimates` |
| 29 | `substantially lower ... and higher rewards` | 没有数字、方差或统计检验 | 给出具体差值；若单 seed，则写 `lower in the logged run` |
| 35 | `particularly effective`、`attractive route` | 宣传式评价词偏多 | 直接说明已观察到的效果和成本来源 |
| 35 | `FP4 arithmetic offers ... reducing ... cost` | 当前主要是 simulation，收益没有实测 | 分开写硬件动机与本文实际评估范围 |
| 37 | `a challenge that is absent from fixed-batch training` | 绝对化；真正缺失的是 on-policy feedback loop | `introduces an on-policy feedback loop that fixed-batch training does not have` |
| 37 | `diverse, interpretable rollouts` | `diverse` 是否必要没有定义 | `rollout groups that receive meaningful, nonconstant task rewards` |
| 39 | `nonsensical repetitions` | 主观、不可复现 | 使用诊断标签，如 `token loops`、`low lexical diversity` |
| 39 | `low-information groups` | 没有可操作定义 | 指明是 low reward variance、few valid answers 还是 repeated responses |
| 39 | `stable activation-magnitude summaries` | 未展示指标或阈值 | 给出图表/appendix 引用，或删去 |
| 41 | `responses that define the teacher's decoding distribution` | 一组采样 response 不“定义”完整分布 | `teacher logits evaluated along teacher-generated response prefixes` |
| 41 | `targets the source of the instability` | 单因果源未经消融证明 | `targets one hypothesized source of instability` |
| 43 | `This work makes the following contributions` | 单句本身正常，但与前后密集 signposting 叠加后模板感强 | 可保留；压缩相邻的 `Our solution` 和 `Motivation` 重复内容 |
| 55 | 多篇工作连续使用 `attributes / identifies / combine / applies` | 像逐条生成的文献目录 | 按“spectral / centering / rotation / RL”问题线索分组并加入比较句 |
| 59 | 一整段连续罗列 PTQ 方法 | 信息密、缺少综合，机器生成式 related-work catalog | 拆成两段，并明确这些方法为何不能直接处理 on-policy sampling |
| 63 | `recently introduced` | 时间性模糊且容易过期 | `introduced in 2026` 或只保留 citation |
| 63 | `This line of work establishes ... practical` | 从有限文献推出宽泛结论 | `These results motivate testing distillation before FP4 RL post-training` |
| 65 | `Taken together`、`This setting motivates` | 常见模板连接词，且重复 Introduction | 直接写尚未解决的 gap 和本文处理的变量 |
| 71 | `Consequently, a useful optimization step presupposes...` | 名词化、绕 | `GRPO needs reward variation within each sampled group. Repetition or malformed output can remove that variation.` |
| 73 | `provides an overview ... before the component details below` | 元话语，不提供内容 | `Figure 1 shows the two stages and the quantized branches.` |
| 78 | `ice-blue ... snowflake ... warm ... robot` | 图注像设计说明或宣传素材，不像技术图注 | 只说明 frozen/trainable modules、data flow 和 precision |
| 82 | `More concretely` | 可删的过渡语 | 直接从 `Let ...` 开始 |
| 82--90 | completion-level \(\rho_i(\theta)\) 的 `standard clipped form` | GRPO 常按 token 定义 ratio/aggregation；这里的简式没有说明是 sequence-level abstraction | 明确这是简化的 sequence-level notation，或给出与实现一致的 token-level objective |
| 97 | `let its truncated rank-r singular-value decomposition be W = ... + R_W` | truncated SVD 本身不是右侧含 residual 的整个等式 | `decompose \(W\) into a rank-\(r\) truncated-SVD approximation and a residual` |
| 104 | `quantization-friendly residual` | 带结论色彩，尚未给误差数据 | `the residual, whose dynamic range is evaluated separately` |
| 111 | `forward GeMM` | 常用写法是 `GEMM` | 全文统一为 `GEMM` |
| 128 | `generic Metis ... configuration interface` | 实现细节打断方法逻辑 | 移到 implementation details 或 appendix |
| 157 | `without cancelling any term` | 防御式措辞，不清楚在回应什么 | 直接说明 QDQ 后 residual sums 不保证为零，因此保留 cross terms |
| 166 | `inherent train--decode mismatch` | `inherent` 太强；这是当前均值估计设计带来的 | `a train--decode mismatch in the current estimator` |
| 200 | `challenging mathematical problems` | 宣传式数据集描述 | `mathematical problems curated for RL and supervised fine-tuning` |
| 204 | `false},  and` | 多余逗号和空格 | `false} and` |
| 204 | `AMC`、`corresponding EvalScope report` | benchmark 版本和评测配置过泛 | 给出 AMC 年份/split、EvalScope 版本、answer extraction 和全部 generation settings |
| 208 | `encoded by the training entry points` | 仓库说明式措辞，不像论文方法 | 直接列出 protocol；代码入口放脚注或 reproducibility appendix |
| 226 | `according to condition / corresponding condition` | 循环定义 | 对每个 condition 明确列 W/A/G quantizer |
| 234 | `per-completion and per-step ...` 后接混合列表 | 指标层级不清 | 分成 completion-level、token-level 和 step-level 三组 |
| 241 | `This decomposition is important` | 自我评价 | 直接说明它排除了什么替代解释；若不能排除则改成较弱表述 |
| 250 | `available ... run` + `substantially reduces` | 一边强调结果暂定，一边强结论 | 用实际均值/区间；单 run 用描述性语气 |
| 259 | `materially higher level` | 空泛评价 | 报起点、终点、平均值或 AUC，并说明比较步数 |
| 270 | `substantial`、`large margins`、`strong` | 宣传式结论，且选择性强调正向任务 | 用每个模型相对 initial/BF16 的具体变化，并同时写出下降任务 |
| 301 | `a different regime` | 含义不明确 | `The 0.5B model does not show the downstream gains observed at 1.5B and 7B.` |
| 301 | `Taken together ... preserve strong reasoning performance` | 对 0.5B 退化和 7B ARC 大幅下降着墨不足 | 改成 “benefits are model- and task-dependent” 并列出反例 |
| 329 | `These endpoints expose whether...` | 首尾两个 noisy log point 不能判定学习信号质量 | `These endpoints provide a descriptive summary; trajectory statistics are reported in Fig. ...` |
| 331 | `effective preparation step` | 缺少消融和多 seed，因果过强 | `the full configuration is associated with a more usable logged reward trajectory` |
| 335 | `form a consistent picture`、`demonstrating`、`central challenge` | 套话加过度概括 | 删除该总结段，或只保留尚未在前文说过的限制 |
| 339 | `targets ... before ... closes the feedback loop` | 比喻式、抽象 | 直接说 Stage I 在开始 on-policy updates 前降低 teacher/student distribution mismatch |
| 341 | `These findings establish ... central criterion` | 证据等级过高 | `The results suggest that rollout diagnostics should accompany loss and accuracy measurements in FP4 RL experiments.` |
| 343 | `remaining challenges clarify the scope` | 公式化 limitation 开头 | 直接从具体限制开始 |

## 四、机器味较重的段落与原因

### 1. Introduction 的叙事过于“整齐”

第 37--48 行依次使用：

- `The relevant question is therefore ...`
- `We study the following question ...`
- `Our starting observation ...`
- `Our solution ...`
- `This two-stage design targets ...`
- `This work makes the following contributions ...`

每句都在宣布下一步，而不是直接提供证据。这种连续 signposting 很像自动生成的论文模板。建议保留研究问题和贡献列表，但删除至少一半的过渡宣告，把观察、方法和原因直接连起来。

### 2. Related Work 像“论文名 + 一个动词”的目录

第 55、59、63 行大量采用 `X attributes...`、`Y identifies...`、`Z applies...`。单个句子没有问题，连续出现时会像模型按检索结果逐条摘要。更自然的写法应围绕作者自己的问题组织：

1. 哪些方法处理 weight outliers；
2. 哪些方法处理 activation mean/outliers；
3. 哪些方法针对真实 FP4 training；
4. 为什么这些工作仍没有隔离 on-policy rollout failure。

### 3. 结果被连续复述了五次

核心结论“Direct FP4 最弱，QAT + moving mean 改善 reward，1.5B/7B 较好，0.5B/ARC 有例外”出现在：

- 第 250--259 行；
- 第 270--301 行；
- 第 329--331 行；
- 第 333--335 行；
- 第 339--343 行。

这类同义复述是全文最明显的机器味。建议：

- Figure 段只读图；
- Table 段只比较数值；
- 删除 `Empirical Takeaway` 小节，或把它改成一段只讨论未解释现象；
- Conclusion 只回答研究问题、限定证据范围和指出下一项实验，不再重复所有模型数字。

### 4. 评价词替代了测量

高频词包括：

`attractive`、`particularly effective`、`substantially`、`materially`、`large margins`、`strong`、`central`、`important`、`establish`、`essential`。

这些词并非禁用词，但本稿中经常没有紧随数值、置信区间、seed variance 或消融结果。优先替换规则：

- 能给数字时给数字；
- 只有单 run 时写 `in the logged run`；
- 只有相关性时写 `is associated with`；
- 没有消融时写 `the full configuration`，不要写某一组件 “causes/addresses”。

### 5. Figure 1 图注的视觉修辞不符合其余文风

`ice-blue teacher robot`、`snowflake`、`warm FP4 student robot` 的写法有演示文稿或宣传图风格。论文图注应让黑白打印、色觉差异或不查看装饰元素的读者仍能理解方法。建议用 module names、frozen/trainable status、precision 和 arrows 描述。

## 五、代表性段落改写

以下版本主要展示去除机器味和校准结论的方法。它们仍需按最终实验、seed 数量和统计口径调整，不能直接视为定稿。

### 摘要建议稿

> Low-precision RL post-training differs from fixed-data training because the policy generates the samples used for its own updates. In our NVFP4 simulation, applying FP4 to selected linear operations before GRPO produced frequent token loops, mixed-language output, and early termination in the initial rollouts. These failures reduced reward variation within sampled groups. We therefore use two stages. A frozen BF16 teacher first provides token-distribution targets for an FP4 student on teacher-generated responses. The adapted student is then optimized with GRPO. During training, activations and output gradients are quantized around their full-sequence means; during autoregressive decoding, each activation mean is estimated from the available prefix. In the available 0.5B diagnostic run, the full distillation-plus-moving-mean configuration had a lower bad-rollout rate than direct FP4. Its final logged mean task reward was also higher than direct FP4 at 0.5B, 1.5B, and 7B. These comparisons are descriptive until the matched-seed study is complete.

这一版本刻意避免把收益单独归因于 QAT，也没有宣称 simulation 已经带来硬件加速。

### Main Results 首段建议稿

> Table~\ref{tab:main} shows that the effect of the full FP4 pipeline depends on model scale and task. For Qwen2.5-Math-1.5B, the proposed model improves over the initial checkpoint on AIME24, AMC, MATH-500, ARC, and GPQA-Diamond, but its AIME25 score decreases from 10.0 to 6.67. Relative to BF16 GRPO, it is 6.0 points lower on AMC and 2.4 points lower on MATH-500, while ARC is 1.78 points higher. For Qwen2.5-Math-7B, the proposed model matches or exceeds BF16 GRPO on AIME24, AIME25, AMC, and MATH-500, but is 18.45 points lower on ARC and 2.53 points lower on GPQA-Diamond. The 0.5B model does not retain the BF16 gains on MATH-500 or ARC.

这一写法不使用 `substantial`、`strong` 或 `large margins`，正负结果也保持对称。

### Conclusion 建议稿

> Direct FP4 ended with a lower logged GRPO reward than BF16 and the full distillation-plus-moving-mean configuration at all three model scales. The full configuration also reduced visible rollout failures in the available 0.5B diagnostic run. These comparisons do not yet isolate the contribution of distillation from weight decomposition and mean-residual quantization. Downstream accuracy is also uneven: the 0.5B model regresses on MATH-500 and ARC, and the 7B model remains below BF16 GRPO on ARC. The next evaluation should therefore use matched seeds and component-wise ablations, and should report the train--decode mean mismatch together with hardware and software details.

这一版把结论限制在现有证据上，并把“下一步需要什么证据”写清楚。

## 六、二次反 AI 审核

对上述建议再做一次 “What still makes this look AI-generated?” 检查后，仍需警惕以下残留：

- 不要把每段都写成整齐的“现象—原因—方案—意义”四步结构；
- 不要在摘要、Results takeaway 和 Conclusion 三处重复同一套形容词；
- 不要用精确小数制造确定性，同时省略 seeds、step budget 和方差；
- 不要为了句式变化，把同一个对象轮换称为 policy、student、model、path 和 configuration；
- 不要把 `supports`、`indicates`、`suggests` 机械地轮换。先判断证据强度，再选一个词；
- 学术论文不需要刻意加入个人化口吻。这里的“自然”应来自具体、克制和术语稳定，而不是加入 `I think` 或口语化短句。

## 建议修改顺序

1. 先确定唯一的实验 condition matrix 和方法名。
2. 区分 QAT 上位概念与 QAD 具体目标，统一所有图表标签。
3. 改写蒸馏公式、moving-mean reduction axes 和模型角色定义。
4. 将所有对 QAT 的单独因果归因改成对 `full configuration` 的描述，或补齐消融。
5. 核对主结果表的 0.5B 行、模型后缀和小数格式。
6. 用实际数字替换 `substantially/materially/strong/central`。
7. 删除或重写 `Empirical Takeaway`，减少 Results 与 Conclusion 的重复。
8. 最后再做句法层面的简化，包括删除元话语、缩短 Related Work 长句和简化 Figure 1 caption。

## 七、专项补充：过度防御性表达

### 判断

文章中存在中等程度的过度防御。它不是贯穿全文的主导语气，但在 Related Work、Method 的若干解释句，以及 Results/Conclusion 的结论包装中比较集中。

这里的“过度防御”不等于谨慎，也不等于所有否定句都应删除。本次采用以下判断标准：

- 文字在读者尚未提出质疑前，主动声明实现或 reward function “没有问题”；
- 用 `not merely`、`not simply`、`cannot ... solely`、`not only ... but` 等结构先树立一个较弱的替代解释，再把本文观点写成更完整的答案；
- 在没有直接消融或统计证据时，提前排除 overflow、task difficulty、reward calibration 或 initialization 等替代解释；
- Related Work 连续多次说明前人“没有处理本文问题”，使创新边界看起来像反复辩护；
- 用 `different regime`、`principal exception` 或“挑战反而明确了范围”等措辞缓冲负面结果。

必要的限制说明、数学条件和图表阅读说明不属于过度防御。问题主要在否定式论证的密度，以及部分否定结论强于现有证据。

### 1. 优先修改的防御性表述

| 行号 | 原表述 | 为什么显得过度防御 | 建议改写 |
|---:|---|---|---|
| 37 | `not only whether ... but whether ...` | 先把 fixed-batch loss 设为不充分的“较弱问题”，再抬高 rollout 视角；这是典型负面对照式立论 | `FP4 RL post-training depends on teacher-forced fidelity and on rollout quality, which determines within-group reward variation.` |
| 39 | `The problem is not explained by visible numerical overflow ... Instead ...` | 在正文尚未展示相应诊断数据时，提前排除 overflow，像在替实现正确性辩护 | 若有数据，直接报指标并引用图表；若无图表，改为 `The logged runs contained no NaN or Inf events, although rollout quality still degraded.`，不要写成已经排除机制 |
| 59、63、65 | `They do not address ...`、`rather than ...`、`does not directly address ...` | 三个相邻段落重复划定“前人没做、本文才做”的边界，创新性辩护过密 | 合并为一次具体比较：`Prior PTQ work evaluates fixed inference quantizers, whereas existing QAD work targets post-hoc accuracy recovery. We study distillation before on-policy FP4 optimization, where quantized samples determine subsequent updates.` |
| 71 | `even when the reward function itself is correctly implemented` | 主动声明 reward 实现正确，但正文没有出现相应质疑；像代码免责说明 | 删除该从句：`Repeated or malformed completions can reduce reward variation within a sampled group.` |
| 91 | `the relevant failure is not merely ...` | 用 `not merely X` 把量化误差描述成过于简单的替代观点，再强调本文视角 | `In FP4 GRPO, quantization error also changes sampled trajectories and therefore the rewards and advantages used for subsequent updates.` |
| 128 | `Although the implementation retains ... rather than computing ...` | 为通用配置接口作辩解，正文读者并不需要知道接口为何存在 | 移到 appendix；正文只写 `The activation path uses mean-residual quantization in place of activation SVD.` |
| 157 | `without cancelling any term` | 暗示可能有人认为作者错误地漏项或约项，语气像预先回应质疑 | `Because QDQ residuals and masked reductions are not guaranteed to be zero mean, the implementation retains all four terms.` |
| 241 | `This decomposition is important: it distinguishes ... from a mere change in task difficulty or reward calibration.` | failure labels 能说明可见退化类型，却不足以排除 task difficulty 或 reward calibration；句子在证据不足时抢先关闭替代解释 | `The labels show which visible failure modes contribute to the aggregate bad-rollout rate. They do not isolate every source of reward change.` |
| 331 | `not simply a consequence of a favorable starting reward` | 用首尾两个日志点排除 initialization explanation，证据不够；而且没有处理训练噪声、步数和 seed variation | 只陈述观察：`At 0.5B and 7B, the two FP4 runs start at similar logged rewards and end at different values. Matched seeds are needed to determine whether this separation is reproducible.` |
| 339 | `cannot be evaluated solely as ...` | 以禁止式判断开场，像在反驳一个文中并未明确提出的 reviewer position | `Evaluation of FP4 RL post-training should include rollout quality because the policy generates the trajectories used for its own updates.` |
| 341 | `must generate ... not merely execute ...` | `must X, not merely Y` 是结论中的负面排比，强调姿态多于新增信息 | `Rollout diagnostics complement fixed-batch numerical error and downstream accuracy when evaluating an FP4 policy.` |

### 2. Related Work 的“创新边界防御”重复

最集中的问题在第 59--65 行。三个连续段落分别说：

1. PTQ `do not address the coupled rollout-and-update process`；
2. QAD 的目标是 post-hoc recovery `rather than on-policy reward optimization`；
3. quantization-aware adaptation `does not directly address the transition` 到 FP4 on-policy policy。

这三句话实际上只表达一个 gap：现有工作没有直接研究“量化 rollout 参与后续 policy update”的闭环。连续说三次会产生两种副作用：

- 像在预先防守 novelty，而不是准确定位本文与最接近工作的差异；
- 使用 `do not address` 容易把前人的研究范围说得过窄，尤其当文献可能包含训练或 rollout 相关实验时。

建议只保留一次边界说明，并把“前人没有做”改成“评估目标不同”：

> PTQ methods primarily evaluate a fixed quantized model at inference time, while QAD recovers the accuracy of a quantized model after post-training. Our setting places the quantized model inside the on-policy loop: its sampled completions determine the rewards and updates used in the next training step.

这段已经足够说明差异。随后可直接进入本文方法，不需要再加 `Taken together`、`does not directly address` 和 `This setting motivates`。

### 3. 结果段中的防御性缓冲

除显式否定句外，部分写法通过包装负面结果来保护整体叙事。

#### 第 270 行：`only` 与 `principal exception`

> MATH-500 differs by only 2.4 points ...
>
> The principal exception is ARC ...

`only` 替读者判断 2.4 points 是否小，`principal exception` 则把 18.5-point ARC gap 包装成整体趋势之外的例外。如果没有预设 equivalence margin 或统计区间，建议只报差值：

> At 1.5B, QAD + moving mean is 2.4 points below BF16 GRPO on MATH-500. At 7B, it is 18.45 points below BF16 GRPO on ARC and 2.53 points below on GPQA-Diamond.

#### 第 301 行：`a different regime`

> The 0.5B instruction model exhibits a different regime ...

`different regime` 暗示已经发现一种机制或 scaling transition，但当前表格只能显示这个模型表现较差。它也会把反例从主结论中隔离出去。建议改为：

> The 0.5B model does not show the downstream gains observed for the two math-specialized models.

随后直接列出 MATH-500 和 ARC 的变化，不要马上用 `Taken together` 转回正面结论。

#### 第 343 行：把 limitation 写成正面贡献

> The remaining challenges clarify the scope of the approach.

这类句子把限制重新包装成“帮助澄清范围”，是常见的防御性 limitation 开头。建议直接承认：

> The current study has three limitations.

之后列出 train--decode mean mismatch、rule-based classifier 的覆盖范围和跨任务不稳定性即可。第 343 行的具体限制本身是必要内容，不应删除。

### 4. 建议保留的否定或限定表述

以下位置看起来也含否定词，但主要承担数学定义、图表解释或真实限制，不属于过度防御：

| 行号 | 表述 | 为什么应保留 |
|---:|---|---|
| 59 | PTQ `without replaying its original optimization procedure` | 是 PTQ 的定义性说明，不是在替本文辩护 |
| 122 | `apply FP4 only to the centered residual` | 明确量化范围；需要与第 78 行 figure caption 修正后保持一致 |
| 166 | `does not access future generated tokens` | 是 causal claim 的必要条件，但最好同时明确 cache 的 reduction axes |
| 166 | full-sequence 与 prefix-only forward `can ... use different means` | 是真实的 train--decode mismatch，应保留并量化 |
| 241 | multi-label proportions `need not sum to one` | 是读图所必需的说明 |
| 301 | `does not retain the BF16 gains` | 直接报告负面结果，比委婉包装更自然 |
| 343 | classifier `captures visible degeneration rather than every form of semantic failure` | 是合理的测量边界，前提是不要随后把它写成方法优势 |

### 5. 推荐的集中改写

下面将最明显的防御性段落改成直接陈述。修改重点是“说观察和范围”，而不是“先反驳一个想象中的质疑”。

#### Introduction 诊断段

> Direct FP4 produced token loops, mixed-language fragments, and short responses early in GRPO. The corresponding logs contained no NaN or Inf events, while the recorded activation-magnitude summaries remained within [reported range]. These observations point to decoding sensitivity as a candidate mechanism, but they do not isolate its cause. The resulting groups contained fewer usable responses and less reward variation.

如果没有可报告的 activation range 或附录图，应删除该指标，不要用 `stable summaries` 代替数据。

#### Related Work 收束段

> PTQ methods primarily evaluate fixed quantized models at inference time, while QAD recovers quantized-model accuracy after post-training. We study a different point in the pipeline: the FP4 model generates the completions used to compute rewards and subsequent policy updates. We therefore distill the quantized policy before GRPO and retain the same quantized path during rollout generation.

#### Results 中关于起始 reward 的段落

> Direct FP4 has the lowest final logged reward at all three scales. At 0.5B, direct FP4 and the full configuration start at 0.0200 and 0.0100 and end at 0.0398 and 0.2316, respectively. At 7B, they start at 0.1641 and 0.1660 and end at 0.2949 and 0.3379. These endpoint comparisons are descriptive; matched seeds and trajectory-level summaries are needed to assess whether the separation is reproducible.

#### Conclusion 开头

> FP4 RL post-training should be evaluated on the trajectories produced by the quantized policy as well as on fixed-batch numerical error. In the logged experiments, direct FP4 has the lowest final reward at all three model scales, while the full distillation-plus-moving-mean configuration yields higher final rewards. The comparison does not isolate the contribution of distillation from the accompanying quantization changes.

### 6. 反 AI 复核：删除防御句后仍可能存在的问题

仅把 `not merely`、`not simply` 和 `cannot solely` 改成肯定句还不够。若改写后仍出现以下模式，防御感会以另一种形式回来：

- 用 `clearly`、`indeed`、`importantly` 或 `notably` 强行告诉读者如何解读证据；
- 把 `not explained by X` 换成同样未经验证的 `points to Y`，却不标明这是候选解释；
- 每个 Related Work 段落最后都加一句 “in contrast, our work ...”；
- 用 `suggests` 替换所有强结论，但仍然省略 seed、variance 和消融；
- 承认一个 limitation 后立即补一句它“opens an important direction”或“clarifies the broader scope”。

最终版本应遵循一个简单原则：先报观察，再说明这些观察能支持什么，最后明确尚未区分的解释。能用表格或数字回答的问题，不用辩解句回答。
