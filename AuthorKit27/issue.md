# AAAI-27 投稿格式与 Reproducibility Checklist 核查报告

## 摘要

当前稿件不建议原样提交。正文的页数、官方模板、匿名性、PDF 纸张与基本字体嵌入均合规，但仍有 5 项明确的格式风险：`\raggedbottom` 覆盖模板排版、三张表的标题位置错误、两张表字号低于要求、表格内使用负间距，以及三张实验图包含 AAAI 明确要求移除或转曲的 `CID TrueType / Identity-H` 字体。Checklist 还存在两项会直接影响可信度的不一致：既有数据集引用并不完整，却回答了 `yes`；正文没有多次运行或不确定性统计，却声称提供了分布信息。

AAAI-27 主赛道正文截止时间为 2026-07-28 23:59 AoE，即北京时间 2026-07-29 19:59；补充材料与代码截止时间为 2026-07-31 23:59 AoE，即北京时间 2026-08-01 19:59。投稿前应优先完成本文“必须修改”部分。

官方依据：

- [AAAI-27 Main Technical Track Call](https://aaai.org/conference/aaai/aaai-27/main-technical-track-call/)
- [AAAI-27 Submission Instructions](https://aaai.org/conference/aaai/aaai-27/submission-instructions/)
- [AAAI-27 Supplementary Material](https://aaai.org/conference/aaai/aaai-27/supplementary-material/)
- [AAAI-27 Author Kit](https://aaai.org/authorkit27/)

## 先读结论

| 优先级 | 问题 | 决策 |
|---|---|---|
| P0 | 正文表格标题、字号、负间距不符合 Author Kit | 修改后再提交 |
| P0 | 三张图包含 `Identity-H` 字体 | 重新导出或转曲后替换 |
| P0 | Checklist 4.11 与单次运行结果矛盾 | 改为 `no`，或补充多随机种子统计 |
| P0 | 当前仓库没有 Supplement 所声称的代码包 | 若代码在别处，按 Code and Data ZIP 单独上传 |
| P1 | `\raggedbottom` 覆盖模板排版 | 删除 |
| P1 | Checklist 数据集引用和示例存在问题 | 修正文献与答案 |
| P1 | 计算环境、随机种子和运行次数不完整 | 在 Supplement 中补齐 |

## 一、正文必须修改的问题

### 1. 删除 `\raggedbottom`

- 位置：[fp4_post_aaai_draft.tex 第 26 行](AuthorKit27/fp4_post_aaai_draft.tex#L26)
- 观察：正文在 `\maketitle` 后调用了 `\raggedbottom`。
- 风险：`aaai2027.sty` 默认使用 `\flushbottom`；该命令改变页面的垂直排版和留白，违反“不得通过命令改变模板间距和外观”的要求。
- 修改：删除这一行，不要用其他垂直间距命令替代。

### 2. 三张表的 caption 必须移到表格下方

AAAI-27 Author Kit 明确要求 table caption 位于表格下方。当前三张表均将 `\caption` 放在 `tabular` 之前：

- [Table 1，第 253 行](AuthorKit27/fp4_post_aaai_draft.tex#L253)
- [Table 2，第 289 行](AuthorKit27/fp4_post_aaai_draft.tex#L289)
- [Table 3，第 307 行](AuthorKit27/fp4_post_aaai_draft.tex#L307)

修改时将每个 `\caption` 和相应的 `\label` 移到 `\end{tabular}` 后。图的 caption 已经位于图下方，无需调整。

### 3. Table 2 和 Table 3 字号过小

- 位置：[Table 2，第 288 行](AuthorKit27/fp4_post_aaai_draft.tex#L288)；[Table 3，第 306 行](AuthorKit27/fp4_post_aaai_draft.tex#L306)
- 观察：两张表使用 `\scriptsize`，约为 7pt。
- 官方要求：表格正文应为 10pt，必要时最低可降至 9pt。
- 修改：至少改为 `\footnotesize`；如仍放不下，应缩短列名、减少小数位或改为跨双栏表格，不能继续缩小字体或使用 `\resizebox`。

Table 1 使用 `\footnotesize`，处于允许下限。

### 4. 删除 Table 2 中的负间距

- 位置：[fp4_post_aaai_draft.tex 第 298 行](AuthorKit27/fp4_post_aaai_draft.tex#L298)
- 观察：表格后使用了 `\\[-2pt]`。
- 风险：这是在表格附近手动压缩垂直间距，与 Author Kit 禁止通过负间距挤压版面的要求冲突。
- 修改：删除 `[-2pt]`；如需表下注释，按自然行距排版。

### 5. 重新处理三张图的 `Identity-H` 字体

PDF 字体检查发现以下图形嵌入了 `DejaVuSans`，类型为 `CID TrueType / Identity-H`：

- `AuthorKit27/Figures/direct_fp4_bad_types_qwen_0_5b.pdf`
- `AuthorKit27/Figures/rollout_bad_ratio_qwen_0_5b.pdf`
- `AuthorKit27/Figures/rollout_reward_qwen_0_5b.pdf`

AAAI Author Kit 要求将 `CID`、`Identity-H` 等字体转为 outlines 或移除。建议使用 AAAI 兼容的 Times/Helvetica 字体重新生成图表，或在导出时将文字转曲。重新生成后应再次运行：

```bash
pdffonts AuthorKit27/Figures/<figure>.pdf
pdffonts AuthorKit27/fp4_post_aaai_draft.pdf
```

目标是没有 `Type 3`，也没有 `Identity-H` 字体。

## 二、Reproducibility Checklist 问题

### 1. 官方示例被误改

- 位置：[ReproducibilityChecklist.tex 第 68 行](AuthorKit27/ReproducibilityChecklist.tex#L68)
- 观察：示例中“修改前”和“修改后”都显示为 `yes`。
- 修改：将第一个示例答案恢复为官方原文 `Type your response here`，只保留第二个示例为 `yes`。

### 2. 第 3.5 项的 `yes` 缺少完整证据

- 位置：[Checklist 3.5](AuthorKit27/ReproducibilityChecklist.tex#L160)
- 当前答案：`yes`
- 问题：正文引用了 DeepMath-103K、MATH-500、ARC 和 GPQA-Diamond，但 AIME 2024、AIME 2025、AMC 以及 EvalScope 没有相应来源引用。
- 修改方案：
  1. 为 AIME、AMC 和 EvalScope 补充适当引用，再保留 `yes`；或
  2. 如果无法在截止前补齐，将答案改为 `no`。

不要在匿名投稿正文或 Supplement 中加入指向在线匿名仓库的链接；AAAI-27 的专项补充材料规则明确禁止此类 Web 指针。

### 3. 第 4.11 项不应回答 `yes`

- 位置：[Checklist 4.11](AuthorKit27/ReproducibilityChecklist.tex#L208)
- 当前答案：`yes`
- 观察：正文报告单次 checkpoint accuracy、训练 reward 端点和轨迹，但没有多随机种子结果、标准差、置信区间、误差条或统计分布；第 4.10 项也已回答没有报告运行次数。
- 判断：训练轨迹和移动平均不能替代结果的不确定性分析，因此当前 `yes` 与论文证据不一致。
- 修改：若不补充多次运行，改为 `no`。若保留 `yes`，应至少报告每个主要条件的运行次数、随机种子、均值和变异性。

### 4. 第 1.3 项的 `yes` 证据较弱

- 位置：[Checklist 1.3](AuthorKit27/ReproducibilityChecklist.tex#L105)
- 当前答案：`yes`
- 问题：正文有相关工作引用，但没有明确标识供非专业读者复现实验所需的 pedagogical/background references。
- 修改：补充并明确指出 GRPO、NVFP4、EvalScope 等背景或操作性参考资料；否则改为 `no`。

### 5. 理论贡献部分应保持一致

- 位置：[Checklist 2.1–2.8](AuthorKit27/ReproducibilityChecklist.tex#L112)
- 当前状态：2.1 回答没有理论贡献，但后续 proof intuition 又回答 `partial`。
- 修改：若论文定位为方法与实证贡献，应在各题允许的选项内统一使用 `no` 或 `NA`，避免给审稿人造成“存在未完整证明的理论主张”的印象。

## 三、可复现性材料的实质缺口

### 1. 当前仓库没有 Supplement 声称的代码

[Supplement 第 22 行](AuthorKit27/fp4_post_supplementary.tex#L22)称 source release 提供以下内容：

- `scripts/grpo/run_phase1.sh`
- `configs/grpo/experiment_env.sh`
- `scripts/grpo/run_evalscope.sh`
- 量化线性层、rollout diagnostics、配置与绘图代码

但当前仓库只有论文、图、模板和编译产物，没有这些代码或配置。AAAI-27 明确说明，“录用后公开”的承诺不能作为当前可复现性证据；审稿人会按提交时实际提供的材料判断。

决策：

- 如果代码位于其他目录，应整理为独立的 Supplementary Code and Data Package ZIP，并在补充材料截止前上传。
- 如果暂时无法提供，应避免在 Supplement 中写成已经提供，并保持 Checklist 4.3、4.4 为 `no`。
- Code ZIP 应匿名化，移除 Git 历史、用户名、绝对路径、日志中的作者或服务器信息。

### 2. 计算环境说明仍不完整

[Supplement 第 20 行](AuthorKit27/fp4_post_supplementary.tex#L20)已说明 8×H100 80GB、Accelerate 和 DeepSpeed ZeRO-2，主文还说明了 RTX 5090 benchmark，但仍缺：

- CPU 型号、主机 RAM；
- 操作系统；
- NVIDIA driver、CUDA、cuDNN；
- PyTorch、Transformers、Accelerate、DeepSpeed、EvalScope 版本；
- 随机种子和确定性设置；
- QAT/GRPO 总步数或总训练 budget；
- AdamW 的 betas、weight decay、epsilon；
- 关键生成参数和量化 block/scale 配置。

因此 Checklist 4.8 和 4.13 回答 `partial` 是合理的，但这些缺口会降低实际可复现性。

## 四、已经合规的项目

- 本地 `aaai2027.sty` 与 AAAI 官网 Author Kit 的 SHA-256 完全一致。
- 使用 `\documentclass[letterpaper]{article}` 和 `\usepackage[submission]{aaai2027}`。
- 主 PDF 为 US Letter、双栏、PDF 1.7、未加密。
- 主 PDF 共 8 页；正文在第 7 页结束，第 8 页只有参考文献，符合“最多 7 页正文、总计最多 9 页，超过第 7 页只能放参考文献”的要求。
- 作者显示为 `Anonymous submission`，affiliation 为空。
- 未发现作者姓名、单位、邮件、Acknowledgments 或外部匿名仓库链接。
- PDF 元数据没有 `Author`、`Title`、`Subject`、`Keywords` 等身份信息。
- 没有页码、书签、嵌入式 Web 链接、加密或附件。
- 所有字体均已嵌入，未发现 Type 3 字体；但上述 `Identity-H` 图形字体仍须处理。
- 编译日志没有 `Overfull` box，只有少量不影响合规性的 `Underfull` 警告。
- Figure caption 均位于图下方。
- 使用官方 `aaai2027.bst`，没有手动覆盖 bibliography style。
- Checklist 作为独立 PDF 上传的方式符合 AAAI-27 要求。
- Supplement 作为独立 PDF 的形式正确，且没有外部 Web 指针。

## 五、建议的提交前顺序

1. 修正文三张表：caption 下移、字号恢复到至少 9pt、删除负间距。
2. 删除 `\raggedbottom`。
3. 重新生成三张实验图，清除 `Identity-H` 字体。
4. 修正 Checklist 示例、3.5、4.11、1.3 和理论贡献部分。
5. 补充 AIME、AMC、EvalScope 引用。
6. 在 Supplement 中补齐软件版本、随机种子、运行次数和训练 budget。
7. 准备匿名 Code and Data ZIP；不要把当前整个 `AuthorKit27` 文件夹或原始 `From QAT to GRPO.zip` 当作代码补充材料上传。
8. 重新编译后检查页数、表格位置、字体、PDF 元数据和编译日志。

## 六、本次核查没有证明的内容

本报告是格式与 Checklist 一致性审计，不是对论文创新性、实验数据真实性或方法正确性的完整同行评审。当前环境没有安装 `pdflatex`，因此没有进行全新 clean build；检查基于现有最新 PDF、对应 `.tex`、编译日志、字体清单和 AAAI-27 官方 Author Kit。修改后仍需在可用的 TeX 环境中重新编译并复查最终 PDF。
