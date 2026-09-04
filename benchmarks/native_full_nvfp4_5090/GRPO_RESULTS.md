# RTX 5090 完整 GRPO：BF16 vs Native Full-NVFP4

日期：2026-07-26

## 结论

本实验里的一个 step 是一个完整的 GRPO optimizer step，包含：

```text
rollout/generate
-> accuracy reward 与 advantage/postprocess
-> policy forward + GRPO loss
-> backward
-> optimizer.step
```

它不只是 train-only forward/backward。计时从 Trainer 的 step begin 到
step end，不包含下一批数据的 dataloader fetch、checkpoint 保存和日志。
`steps_per_generation=1`，所以每个 optimizer step 都重新 rollout。

在两个模型各自能同时容纳 BF16 和 NVFP4 的最大 batch 上，Native
Full-NVFP4 的**完整 GRPO step 都没有加速**：

| 模型 | Completion batch | Unique prompts / G | BF16 step | Native NVFP4 step | BF16/NVFP4 | BF16 completion tok/s | NVFP4 completion tok/s | Peak GiB BF16/NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 128 | 32 / 4 | 2.030 s | 6.174 s | 0.329x | 4,035.8 | 1,326.8 | 17.86 / 23.15 |
| Qwen2.5-Math-1.5B | 64 | 16 / 4 | 2.324 s | 6.883 s | 0.338x | 1,762.7 | 595.1 | 16.75 / 23.49 |

表中是 3 个独立进程的 step median 再取中位数；每个进程包含 2 个
warmup step 和 10 个 measured step。`BF16/NVFP4 > 1` 才表示 NVFP4
更快。当前结果分别是：

- 0.5B：Native Full-NVFP4 完整 step 比 BF16 慢 **3.04x**；
- 1.5B：Native Full-NVFP4 完整 step 比 BF16 慢 **2.96x**。

## 增大 batch 是否有效

有效，但只改善了相对差距，没有让完整 GRPO step 发生反转。固定
prompt 64–128 tokens、completion 64 tokens、`G=4` 后的 batch sweep
如下。每项是 1 warmup + 3 measured steps 的容量/趋势实验：

| 模型 | Completion batch | BF16 step | Native NVFP4 step | BF16/NVFP4 | BF16 tok/s | NVFP4 tok/s | Peak GiB BF16/NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.5B | 32 | 1.330 s | 5.646 s | 0.236x | 1,540.2 | 362.8 | 6.58 / 8.36 |
| 0.5B | 64 | 1.591 s | 5.711 s | 0.279x | 2,574.9 | 717.3 | 10.34 / 13.30 |
| 0.5B | 128 | 2.086 s | 6.245 s | 0.334x | 3,926.6 | 1,311.8 | 17.86 / 23.15 |
| 1.5B | 32 | 1.775 s | 6.649 s | 0.267x | 1,153.6 | 308.0 | 12.72 / 17.02 |
| 1.5B | 64 | 2.424 s | 7.101 s | 0.341x | 1,689.7 | 576.9 | 16.74 / 23.46 |

因此原先关于“大 batch 才能发挥 FP4”的判断，在 **policy train
子阶段**得到验证：

| 模型 / batch | 阶段 | BF16 | Native NVFP4 | BF16/NVFP4 |
|---|---|---:|---:|---:|
| 0.5B / 128 | policy forward + backward | 0.668 s | 0.801 s | 0.834x |
| 1.5B / 64 | policy forward + backward | 0.832 s | 0.801 s | **1.039x** |

1.5B、batch 64 的 native policy train 子阶段稳定地快约 **3.9%**。
但这不是完整 GRPO 加速；它只占 native 完整 step 的约 11.6%。

## 为什么完整 step 仍然慢

最大 paired batch 的分阶段中位数如下：

| 模型 / batch | 模式 | Rollout total | Prefill | Decode | Policy fwd+bwd | Optimizer |
|---|---|---:|---:|---:|---:|---:|
| 0.5B / 128 | BF16 | 1.350 s | 0.087 s | 1.054 s | 0.668 s | 0.006 s |
| 0.5B / 128 | Native NVFP4 | 5.351 s | 0.109 s | 5.023 s | 0.801 s | 0.008 s |
| 1.5B / 64 | BF16 | 1.465 s | 0.126 s | 1.207 s | 0.832 s | 0.016 s |
| 1.5B / 64 | Native NVFP4 | 6.059 s | 0.125 s | 5.784 s | 0.801 s | 0.018 s |

瓶颈是 autoregressive decode：

- 0.5B 的 native decode 比 BF16 慢约 4.77x；
- 1.5B 的 native decode 比 BF16 慢约 4.79x。

每个 decode token 的矩阵行数仍然较小，而 full 方法每层需要 activation
mean/pack、residual-weight FP4 GEMM、V FP4 GEMM、BF16
\(U/\Sigma\) 路径和 mean correction。BF16 baseline 只有原始的一路
linear GEMM。大 batch 能让较大的 policy training 矩阵充分利用 Tensor
Core，却无法消除 64 次串行 decode 中的额外量化、kernel launch 和低秩
路径开销。

## 显存与最大 batch

Qwen2.5-Math-1.5B 的 BF16 batch 128 可以运行：

```text
step median = 3.632 s
completion throughput = 2,255.3 tok/s
peak allocated = 24.81 GiB
```

同一模型的 Native Full-NVFP4 batch 128 OOM，因此公平的 paired
比较只能停在 batch 64。按各模式当前最大可运行配置比较，1.5B native
吞吐为 BF16 的约 26.4%（595.1 vs 2,255.3 completion tok/s）。

Native 路径使用 FP4 Tensor Core 并不意味着整个训练状态只占 4 bit：
可训练 BF16 residual/U/s/V、grad、optimizer state 和 packed cache 都
需要保留；full SVD 路径还增加了低秩参数，所以本实现的峰值显存高于
BF16 baseline。

## 原生方法的正确语义

对每个被替换的 projection：

```text
W = R + U diag(s) V
```

- \(R\) 和 \(V\) 分别量化并缓存为 Transformer Engine packed NVFP4；
- activation 使用 `mean + FP4(residual)`；
- backward gradient 使用 `mean + FP4(residual)`；
- \(R\) 与 \(V\) 的 fprop/dgrad/wgrad 主 GEMM 分别调用
  `transformer_engine.pytorch.cpp_extensions.general_gemm`；
- \(U\)、\(s\) 和 mean correction 保持 BF16。

原生路径没有调用 `metis_merge_rollout_weights`。`bitlinear.py` 中的
merge 先把 fake-quantized \(R_q\) 和 \(V_q\) 作为 BF16 tensor 合成
weight，再执行 BF16 `torch.matmul`，是 fake-QDQ 的 rollout shortcut。
如果把这个合成结果重新 pack 成 NVFP4，会把原本应保持 BF16 的
\(U\Sigma\) 贡献再次量化，得到的已经不是当前 full 方法。

真实 GRPO training row 数会出现能被 16 整除、但不能被 TE 2.15 SM120
Wgrad kernel 接受的形状（例如 1872）。实现只对 residual rows 补零到
32 对齐，mean 仍按真实 rows 计算，并在 fprop/dgrad 输出处裁回原 shape。

## 实验配置

- GPU：NVIDIA GeForce RTX 5090，UUID
  `GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959`；
- PyTorch：`2.12.0a0+0291f960b6.nv26.04.48445190`；
- CUDA：13.2；
- Transformer Engine：`2.15.0+42b8400`；
- Transformers / TRL：4.57.6 / 0.29.1；
- 模型：本地 `qwen2.5-0.5b-instruct` 和
  `qwen2.5-math-1.5b`；
- 数据：本地 DeepMath-103K train parquet，SHA-256
  `e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7`；
- 每步 completion 固定 64 tokens，prompt 筛选为 64–128 tokens；
- `G=4`，gradient accumulation 1，gradient checkpointing 开启；
- rank 64，fused AdamW，learning rate `1e-6`，seed 2026；
- TRL `loss_type=grpo`、`beta=0`、`use_vllm=false`；
- BF16/NVFP4 的运行顺序按 repeat 交替；
- installed TE 的 stochastic FP4 conversion kernel 不支持当前 SM120a
  build，因此 gradient NVFP4 使用 deterministic rounding；主 GEMM
  仍是原生 SM120 block-scaled E2M1 Tensor Core kernel。

## 限制

1. 64-token completion 下，大多数 accuracy reward 为 0；0.5B 两个模式
   和 1.5B native 的 measured reward 都为 0，1.5B BF16 只有极少量
   non-zero reward。这是完整 GRPO 控制流和算子执行的性能实验，不能
   用来判断训练收敛或最终精度。
2. 强制固定 completion 长度是为了让每步 token 数严格相同。更长的真实
   rollout、不同 `G`、vLLM 或异步 generation 可能改变阶段占比。
3. 结果来自一张指定 RTX 5090；没有把其他 GPU 或多卡通信混入结果。

## Artifact

```text
outputs/grpo_native_full_nvfp4_5090/formal_fixed_c64/
  results/                         # batch sweep JSON
  failures.jsonl                   # 1.5B native batch 128 OOM

outputs/grpo_native_full_nvfp4_5090/final_repeats_fixed_c64/
  results/                         # 12 个正式 JSON
  grpo_summary.csv
  grpo_paired_summary.json
  grpo_aggregate_summary.json
  grpo_summary.md
```

准备好的两份 tokenizer-specific DeepMath JSONL 在各输出目录的 `data/`
下。运行与复现方法见 [README.md](README.md)。
