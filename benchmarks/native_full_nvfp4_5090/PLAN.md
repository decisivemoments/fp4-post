# Qwen Full-NVFP4 原生 Tensor Core 单步性能实验计划

状态：**已批准并执行完成（2026-07-26）**

最终结果见 `RESULTS.md`。正式数据位于 ignored 的
`outputs/native_full_nvfp4_5090/formal_clean/`；12 个结果文件均通过物理
GPU 3 UUID 审计。

## 后续批准的完整 GRPO 扩展

用户确认早期 batch 1/4 train-only step 不能回答实际 GRPO 吞吐后，批准
了以下扩展：

- step 改为每一步均执行 rollout/generate、reward、GRPO policy
  forward/loss、backward 和 optimizer；
- completion batch sweep 为 32/64/128，`G=4`，gradient accumulation
  为 1；
- prompt 长度固定筛选到 64–128 tokens，completion 强制为 64 tokens，
  使 BF16/NVFP4 的每步工作量相同；
- 两个模型分别使用能同时容纳 BF16/NVFP4 的最大 batch 做 3 个独立
  repeat；每个 repeat 为 2 warmup + 10 measured steps；
- 另外记录 rollout prefill/decode、reward、train forward/backward、
  optimizer、吞吐和峰值显存；
- 原生 full 路径始终分别使用 packed FP4 residual weight 与 packed FP4
  \(V\)。`metis_merge_rollout_weights` 是 fake-QDQ shortcut，不进入
  原生实验。

该扩展的最终结果见 [GRPO_RESULTS.md](GRPO_RESULTS.md)，原始数据位于
`outputs/grpo_native_full_nvfp4_5090/`。

## 无 optimizer 的最大 batch 扩展

2026-07-27 追加纯 policy-train 实验：

- 不构造 optimizer，不分配 optimizer state；
- 计时仅包含 forward、causal-LM loss 和 backward；
- gradient clear 放在计时窗口之外；
- sequence length 512，gradient checkpointing 开启；
- 搜索 BF16/native 各自最大稳定 batch；
- 在最大共同 batch 比较相同工作量；
- 在各自最大 batch 比较峰值 tokens/s；
- 3 个独立进程，每个进程 5 warmup + 20 measured steps。

结果见
[FORWARD_BACKWARD_RESULTS.md](FORWARD_BACKWARD_RESULTS.md)。

## 1. 目标

在单张 RTX 5090 上，对以下两个本地模型进行严格的单步训练性能比较：

- Qwen2.5-0.5B-Instruct
- Qwen2.5-Math-1.5B

主比较对象只有两项：

1. 标准 BF16；
2. 当前 `full` 方法的原生 NVFP4 实现：
   - 权重矩阵执行 rank-64 W-SVD；
   - activation 使用 mean-residual；
   - backward gradient 使用 mean-residual；
   - 占主要计算量的低比特 GEMM 必须使用 RTX 5090 原生 FP4
     Tensor Core。

需要回答：

- 稳态下一个完整 optimizer step 是否比 BF16 快；
- 快或慢多少；
- 加速/减速来自 forward、backward 还是 optimizer；
- 数据转换、缩放、mean correction 和权重打包是否抵消了 FP4 GEMM
  的收益；
- 是否确实使用了原生 FP4 Tensor Core，而不是 BF16 fake-QDQ。

## 2. 对上一轮实验的处理

上一轮测量的是最低开销的 `direct_fp4` fake-QDQ 路径。该实现把张量
量化后重新展开为 BF16，再调用 BF16 `torch.matmul`，所以它不能回答
本计划的问题。

上一轮脚本和结果只保留为：

- BF16 时间量级参考；
- 数据预处理和计时框架的起点；
- fake-QDQ correctness reference。

它不会被作为本次 `full + native NVFP4` 的最终结果。

## 3. `full` 方法的计算定义

### 3.1 权重路径

对当前代码所替换的 Qwen 投影层：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

保持当前 full 方法的分解：

```text
W = W_residual + U diag(s) V
```

其中：

- W-SVD rank 固定为 64；
- `W_residual` 和 `V` 的主要 GEMM 使用原生 NVFP4；
- `U`、`s` 和必要的 correction 使用 BF16；
- optimizer 仍更新高精度 master parameters；
- 每次参数更新后使 packed FP4 weight cache 失效；
- gradient-checkpoint recomputation 内允许复用同一版本的 packed weight。

### 3.2 Activation mean-residual

对进入低比特线性层的 activation：

```text
x = mean(x) + residual(x)
```

不能先把结果恢复成 BF16 再执行大矩阵乘法。目标实现为：

```text
native_fp4_gemm(fp4(residual(x)), fp4(weight))
+ BF16 mean-correction
```

mean correction 只允许使用低秩、向量或较小的 BF16 运算。大尺寸主
GEMM 必须保留 FP4 packed operands。

Q/K/V 共享同一次 activation residual 计算与 FP4 packing；
Gate/Up 同样共享，避免重复转换。

### 3.3 Backward gradient mean-residual

对 backward 的 `grad_output`：

```text
g = mean(g) + residual(g)
```

主 dgrad/Wgrad 计算使用原生 FP4 Tensor Core。展开 Wgrad 时：

```text
(mean(g) + residual(g))^T (mean(x) + residual(x))
```

其中 residual-residual 是原生 FP4 GEMM，其余 mean 相关项使用 BF16
低秩/outer-product correction。必须保留这些 correction，不能通过
简单丢弃均值来换取速度。

## 4. 原生 FP4 的硬性判定标准

仅仅启用 Transformer Engine recipe 或出现 `Float4Tensor` 不足以证明
硬件 FP4 已生效。必须同时满足：

1. 大 GEMM 的输入在调用点仍是 packed NVFP4 + scale metadata；
2. 大 GEMM 前不存在把完整 operands 转回 BF16 的操作；
3. Transformer Engine 明确选择 NVFP4 recipe/backend；
4. Nsight Systems 或 PyTorch profiler 能看到 Transformer Engine
   NVFP4 GEMM kernel；
5. 至少抽取一个代表性 kernel，用 Nsight Compute、kernel metadata
   或指令级证据确认它走 Blackwell FP4 Tensor Core，而非 BF16 Tensor
   Core；
6. profiler 证据和软件版本写入最终报告。

如果无法证明以上条件，该结果不得标记为“原生 NVFP4”。

## 5. 实施阶段

### 阶段 A：Transformer Engine 能力与 API 探针

审批后先进行只读检查和最小探针：

- 固定使用现有 `sm-container`；
- 记录 PyTorch、CUDA、Transformer Engine、驱动和 GPU capability；
- 检查已安装 TE 2.15 中 NVFP4 recipe、FP4 tensor、quantizer 和 GEMM
  API；
- 确认 RTX 5090 / SM120 对相应 kernel 的运行时支持；
- 确认当前 Qwen 矩阵形状、对齐和 block-scaling 约束；
- 查清 TE 原生 NVFP4 scaling contract 与当前 simulator 的
  1x16 block/scaling 是否一致。

阶段门：

- 若 TE 公共或底层扩展 API 能表达自定义 packed FP4 GEMM，进入阶段 B；
- 若只能使用完整 `te.Linear`，评估是否能插入 W-SVD 和 mean
  corrections；
- 若现有 TE 无法暴露所需 native backward GEMM，先报告具体限制，
  不得静默退回 fake-QDQ；
- 编写自定义 CUDA FP4 kernel 不在本计划默认授权范围内，若确实需要，
  先单独请求批准。

### 阶段 B：矩阵级原生 NVFP4 PoC

针对两个模型的真实投影形状建立 microbenchmark：

- BF16 GEMM；
- TE 原生 NVFP4 GEMM；
- NVFP4 quantize/pack + GEMM；
- cached packed weight + activation pack + GEMM；
- forward、dgrad、Wgrad 分别测试。

同时输出：

- kernel trace；
- GEMM-only latency；
- 包含 pack/scale 的实际 latency；
- 数值误差；
- 峰值显存。

阶段门：

- 原生 FP4 kernel 证据成立；
- 至少一个代表性 Qwen 大矩阵的 native GEMM 本体快于 BF16；
- 若连 GEMM 本体都没有加速，停止模型级改造并先汇报。

### 阶段 C：实现独立 Native Full-NVFP4 backend

在不破坏现有 fake-QDQ 路径的前提下增加显式 backend，例如：

```text
--metis-gemm-backend fake_qdq
--metis-gemm-backend native_nvfp4
```

计划包含：

- 新的 native FP4 linear/autograd 路径；
- rank-64 W-SVD 参数布局；
- activation mean-residual 和 correction；
- gradient mean-residual 和 correction；
- Q/K/V、Gate/Up activation pack 共享；
- packed weight cache 与 optimizer-version invalidation；
- gradient-checkpoint recomputation cache；
- BF16 fallback 只用于不支持的非主要算子、U/S 路径和 correction；
- 日志中明确打印每类 GEMM 的实际 backend/dtype。

禁止：

- 用 BF16 dequantized full tensor 执行主 GEMM；
- 把 fake-QDQ 结果命名为 native NVFP4；
- 因 API 不方便而省略 mean 或 W-SVD；
- BF16/NVFP4 使用不同数据、初始权重或 optimizer 配置。

### 阶段 D：正确性验证

按以下顺序验证：

1. 小矩阵代数测试：
   - SVD reconstruction；
   - activation mean correction；
   - dgrad/Wgrad mean correction；
   - cache invalidation；
   - shared packing。
2. 使用同一批 packed values/scales，对比：
   - native FP4 输出；
   - 将同一 FP4 数据显式 dequantize 后得到的 reference。
3. 单层 forward/backward：
   - 输出误差；
   - input gradient；
   - weight/U/S/V gradients。
4. 每个 Qwen 模型执行 2–3 个 optimizer-step smoke test：
   - loss 有限；
   - gradients 有限；
   - 参数实际更新；
   - 无 cache stale-data。

所有 tolerance 根据 BF16 accumulation 和 TE NVFP4 rounding 预先写进测试，
不能在看到结果后临时放宽。

### 阶段 E：模型级正式计时

主实验固定：

- 单卡物理 GPU 3：RTX 5090；
- batch size：1；
- sequence length：512；
- gradient checkpointing：开启；
- attention implementation：两组相同；
- optimizer：fused AdamW；
- learning rate：`1e-5`；
- 数据：同一批本地 DeepMath token cache；
- W-SVD rank：64；
- activation mode：mean；
- gradient mode：mean；
- 预热：10 steps；
- 正式计时：每次 30 steps；
- 独立进程重复：3 次；
- BF16/NVFP4 运行顺序交替，降低温度和频率漂移影响。

每个 timed step 包含：

```text
forward + loss + backward + optimizer.step + zero_grad
```

排除：

- 模型加载；
- 数据 tokenization；
- 初始 W-SVD；
- 首次 kernel JIT/编译；
- optimizer state 首次分配；
- host-to-device 数据复制。

除主实验外，再增加一个诊断 workload：

- 在 32 GiB 内为每个模型选择可稳定运行的较大 microbatch；
- 用于判断 batch=1 是否没有充分占满 Tensor Core；
- 该结果单独展示，不替代 batch=1 主结论。

### 阶段 F：结果审计与报告

报告至少包含：

- total step median、p10、p90；
- forward/backward/optimizer 分阶段时间；
- padded tokens/s；
- 峰值 allocated/reserved 显存；
- NVFP4 相对 BF16 的 speedup；
- pack、scale、mean correction、SVD low-rank path 的时间占比；
- 三次独立运行的一致性；
- GPU 温度、P-state 和是否存在其他进程；
- 原生 FP4 kernel 的 profiler 证据；
- 数值正确性结果；
- 明确的适用范围和未覆盖项。

判定规则：

- 三次独立运行的 median 都快至少 5%，才报告“有稳定加速”；
- 变化在 ±5% 内报告“基本持平”；
- 否则报告“变慢”；
- 即使结果更快，若原生 FP4 kernel 证据不成立，也不能归因于 FP4
  Tensor Core。

## 6. 最终实验矩阵

| 模型 | 主模式 | W-SVD | Activation | Gradient | 主 GEMM |
|---|---|---:|---|---|---|
| Qwen2.5-0.5B-Instruct | BF16 | 无 | BF16 | BF16 | BF16 Tensor Core |
| Qwen2.5-0.5B-Instruct | Full NVFP4 | rank 64 | mean-residual | mean-residual | 原生 NVFP4 Tensor Core |
| Qwen2.5-Math-1.5B | BF16 | 无 | BF16 | BF16 | BF16 Tensor Core |
| Qwen2.5-Math-1.5B | Full NVFP4 | rank 64 | mean-residual | mean-residual | 原生 NVFP4 Tensor Core |

诊断项可以包含 native direct-FP4 或 fake full reference，但不会混入主结果表。

## 7. 预计交付物

审批并执行后预计新增：

```text
benchmarks/native_full_nvfp4_5090/
  PLAN.md
  README.md
  benchmark_gemm.py
  benchmark_train_step.py
  summarize_results.py
  RESULTS.md

src/Metis/Metis/
  native_nvfp4.py

tests/
  test_native_nvfp4.py

scripts/benchmark/
  run_native_full_nvfp4_5090.sh
```

生成的 cache、profiler traces、JSON 和 CSV 继续写入 ignored `outputs/`，
仓库只保留可复现代码和整理后的结果报告。

## 8. 风险与停止条件

主要风险：

- TE 2.15 对 consumer Blackwell 的 NVFP4 training API 与数据中心 Blackwell
  支持范围不同；
- public TE API 可能只支持整层 `te.Linear`，不直接暴露本方法需要的
  mean-decomposed operands；
- native scaling layout 可能与当前 fake simulator 不同；
- pack/scale 和 BF16 correction 可能抵消 FP4 GEMM 收益；
- batch=1、seq=512 的矩阵可能不能充分发挥 FP4 Tensor Core；
- W-SVD 和 additional residual/low-rank path 增加额外 GEMM；
- stochastic rounding 会提高逐步数值差异，需要固定随机种子和统计验证。

停止并汇报的条件：

- 不能证明 native FP4 kernel 被调用；
- 必须把主 operands 完整 dequantize 为 BF16 才能计算；
- native FP4 GEMM 本体在代表性矩阵上不快于 BF16；
- correctness gate 失败；
- 完成任务需要转向新写自定义 CUDA kernel。

## 9. 审批边界

用户已批准本计划。执行前的审批边界为：

- 不修改当前训练实现；
- 不新增 native backend；
- 不运行新的 GPU benchmark；
- 不安装或升级容器依赖；
- 不启动 profiler。

用户批准后，从阶段 A 开始执行，并按阶段门继续。
