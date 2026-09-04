# RTX 5090 纯 Forward/Backward：BF16 vs Native Full-NVFP4

日期：2026-07-27

## 实验定义

本实验只计时：

```text
model forward + causal-LM loss + loss.backward()
```

没有构造 optimizer，不执行 `optimizer.step()`，optimizer state 为 0
bytes。每一步的 gradient clear 使用 `set_to_none=True`，并放在 CUDA
计时窗口之后。峰值显存因此只来自模型参数、packed weight cache、输入、
forward activations、loss workspace 和 parameter gradients。

两种模式使用相同的：

- 本地 DeepMath token cache；
- sequence length 512；
- gradient checkpointing；
- SDPA；
- BF16 model/LM-head；
- 3 个独立进程，每个进程 5 warmup + 20 measured steps。

Native Full-NVFP4 对 projection weight 做 rank-64 W-SVD，residual \(R\)
和 \(V\) 分别使用 packed native FP4 GEMM，\(U/\Sigma\) 保持 BF16；
activation 和 backward gradient 均使用 mean-residual。

## 最大共同 batch：公平比较

| 模型 | Batch | BF16 forward | NVFP4 forward | BF16 backward | NVFP4 backward | BF16 total | NVFP4 total | BF16/NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 22 | 101.771 ms | 131.842 ms | 236.531 ms | 324.745 ms | 338.502 ms | 456.626 ms | 0.741x |
| Qwen2.5-Math-1.5B | 17 | 182.129 ms | 170.020 ms | 479.925 ms | 473.946 ms | 662.114 ms | 643.925 ms | **1.028x** |

结论：

- 0.5B：native forward+backward 仍比 BF16 慢 **34.9%**；
- 1.5B：native forward+backward 比 BF16 快 **2.82%**；
  - forward 快 7.12%；
  - backward 快 1.26%。

1.5B 的三次独立进程均得到相同方向：

```text
BF16 total medians:   661.392 / 662.114 / 662.286 ms
NVFP4 total medians:  643.773 / 643.925 / 644.332 ms
```

因此 1.5B 的小幅加速不是单次运行噪声。

## 最大稳定 batch 与峰值吞吐

容量搜索使用独立进程，从能运行的点逐步增加 batch，并用相邻失败点确认
边界：

| 模型 | 模式 | 最大稳定 batch | 相邻失败 batch | 正式峰值显存 |
|---|---|---:|---:|---:|
| 0.5B | BF16 | 28 | 29 OOM | 30.06 GiB |
| 0.5B | Native Full-NVFP4 | 22 | 23 OOM | 26.46 GiB |
| 1.5B | BF16 | 25 | 26 allocation failure | 29.47 GiB |
| 1.5B | Native Full-NVFP4 | 17 | 18 OOM | 26.03 GiB |

按每种模式自己的最大稳定 batch 比较：

| 模型 | BF16 batch / tokens/s | NVFP4 batch / tokens/s | NVFP4/BF16 峰值吞吐 |
|---|---:|---:|---:|
| 0.5B | 28 / 33,118.5 | 22 / 24,667.9 | 0.745x |
| 1.5B | 25 / 12,898.3 | 17 / 13,517.1 | **1.048x** |

即使 Native 1.5B 能容纳的 batch 更小，它的峰值吞吐仍比 BF16 高
**4.80%**。0.5B 则仍明显落后。

## 显存解释

最大共同 batch 下：

| 模型 / batch | BF16 peak | NVFP4 peak | BF16 baseline allocation | NVFP4 baseline allocation |
|---|---:|---:|---:|---:|
| 0.5B / 22 | 23.83 GiB | 26.46 GiB | 0.98 GiB | 1.47 GiB |
| 1.5B / 17 | 20.98 GiB | 26.03 GiB | 2.94 GiB | 4.52 GiB |

这里 baseline allocation 是进入 forward 前、已经完成模型加载、W-SVD
和 native packed-weight warmup 后的 allocated memory。Native full
路径同时保留 BF16 \(R/U/\Sigma/V\) 参数和 packed FP4 cache，所以参数
基线与峰值都高于 BF16；FP4 Tensor Core 计算不等于把全部训练状态压成
4 bit。

容量边界主要由 causal-LM 大词表 logits/cross-entropy backward workspace
触发。例如 0.5B batch 29 和 1.5B native batch 18 都在 loss backward
阶段 OOM。这部分对 BF16/native 共用，但 native 较高的参数和 packed
cache 基线减少了剩余空间。

## 如何理解结果

这组实验支持“大 batch 才能发挥 FP4”的判断，但模型规模很关键：

- 0.5B 的 projection 矩阵仍不够大，额外的 mean、pack、两路
  residual/V GEMM、BF16 low-rank 和 correction 成本无法摊平；
- 1.5B 的 MLP projection 足够大，原生 FP4 Tensor Core 收益开始超过
  Full-NVFP4 的附加开销，得到约 2.8% 的同-batch step 加速和约 4.8%
  的最大吞吐提升。

这只回答纯 policy forward/backward。它不包含 rollout、reward、
optimizer 或通信，因此不能替代完整 GRPO 结果；完整 GRPO 仍受
autoregressive decode 限制。

## Artifact

```text
outputs/forward_backward_no_optimizer_5090/capacity/
  # 容量探测成功结果；失败边界保留在终端记录

outputs/forward_backward_no_optimizer_5090/formal/
  results/                             # 18 个正式 JSON
  forward_backward_aggregate.json
  forward_backward_paired.json
  forward_backward_maxima.json
  forward_backward_summary.md
```

每个正式 JSON 都记录：

```text
benchmark_scope = forward_loss_backward_no_optimizer
optimizer = null
optimizer_constructed = false
optimizer_state_bytes = 0
```

运行方法见 [README.md](README.md)。

CUDA Graph 捕获后的 matched-batch 与容量结果见
[CUDA_GRAPH_RESULTS.md](CUDA_GRAPH_RESULTS.md)。
