# RTX 5090 上 BF16 与 Native Full-NVFP4 Train-only 对比

> **范围说明：**本页是固定输入、只包含 policy
> forward/backward/optimizer 的早期 train-only benchmark，**不是完整
> GRPO step**，也不含 rollout、reward 或 advantage 构造。完整 GRPO
> batch sweep 与最终结论见 [GRPO_RESULTS.md](GRPO_RESULTS.md)。

日期：2026-07-26

## 结论

在本页 train-only workload（batch size 1、sequence length 512、
gradient checkpointing）下，当前 full 方法使用 RTX 5090 原生 FP4
Tensor Core 后，**两个模型的 policy 训练 step 都没有加速，反而约慢
4.2–4.35 倍**。

| 模型 | BF16 p10/median/p90 | Native full-NVFP4 p10/median/p90 | BF16/NVFP4 | NVFP4 延迟变化 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 86.001/88.469/95.357 ms | 381.182/385.141/395.519 ms | 0.230x | +335.34% |
| Qwen2.5-Math-1.5B | 110.654/111.124/118.286 ms | 464.688/467.857/504.298 ms | 0.238x | +321.02% |

这里 `BF16/NVFP4 > 1` 才表示 NVFP4 更快。表中统计量是三个独立进程
对应统计量的中位数；每个进程包含 10 个 warmup step 和 30 个 measured
step。

换算为 padded tokens/s：

| 模型 | BF16 | Native full-NVFP4 |
|---|---:|---:|
| Qwen2.5-0.5B-Instruct | 5,787.35 | 1,329.38 |
| Qwen2.5-Math-1.5B | 4,607.47 | 1,094.35 |

## 三次独立运行

| 模型 | Repeat | BF16 median | Native full-NVFP4 median | BF16/NVFP4 |
|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 1 | 88.469 ms | 385.141 ms | 0.230x |
| Qwen2.5-0.5B-Instruct | 2 | 93.495 ms | 382.290 ms | 0.245x |
| Qwen2.5-0.5B-Instruct | 3 | 86.375 ms | 390.691 ms | 0.221x |
| Qwen2.5-Math-1.5B | 1 | 110.685 ms | 466.578 ms | 0.237x |
| Qwen2.5-Math-1.5B | 2 | 111.124 ms | 474.002 ms | 0.234x |
| Qwen2.5-Math-1.5B | 3 | 114.897 ms | 467.857 ms | 0.246x |

所有 repeat 都远超 5% 的判定阈值，因此结论是稳定变慢。

## 分阶段时间

下表仍是三个独立 run median 的中位数。

| 模型 | 模式 | Forward | Backward | Optimizer | Total |
|---|---|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | BF16 | 25.181 ms | 56.099 ms | 5.151 ms | 88.469 ms |
| Qwen2.5-0.5B-Instruct | Native full-NVFP4 | 108.991 ms | 268.554 ms | 7.276 ms | 385.141 ms |
| Qwen2.5-Math-1.5B | BF16 | 30.094 ms | 66.411 ms | 14.651 ms | 111.124 ms |
| Qwen2.5-Math-1.5B | Native full-NVFP4 | 130.722 ms | 322.397 ms | 17.548 ms | 467.857 ms |

NVFP4 相对 BF16 的阶段延迟变化：

- 0.5B：forward `+332.84%`，backward `+378.71%`，optimizer
  `+41.25%`；
- 1.5B：forward `+334.38%`，backward `+385.46%`，optimizer
  `+19.77%`。

## 为什么原生 FP4 GEMM 没有转化为 step 加速

矩阵级 microbenchmark 的结果很清楚：

- 0.5B 的 attention、MLP 和 rank-64 GEMM 全部慢于 BF16，速度比仅
  `0.346–0.713x`；
- 1.5B 的 attention 和 rank-64 GEMM 仍只有 `0.350–0.592x`；
- 1.5B 的大 MLP GEMM 本体确实更快：
  - up projection：fprop/dgrad/wgrad 为 `1.431/1.326/1.616x`；
  - down projection：fprop/dgrad/wgrad 为 `1.327/1.413/1.614x`。

也就是说，5090 的 FP4 Tensor Core 对足够大的 MLP 矩阵有效，但在
`M=512` 时，attention 和 rank-64 路径主要受约 `0.05 ms` 的 kernel
启动/调度下限影响。

对 Qwen2.5-Math-1.5B 的一个完整 train-only native step 做 NVTX +
Nsight Systems 审计后，用“每类 range 的 projected median × instance
count”抑制 profiler 长尾，得到以下近似归因：

| 类别 | 稳健估计 | 占正式 step |
|---|---:|---:|
| activation/gradient mean + pack/scale + weight pack | 93.233 ms | 19.93% |
| native FP4 fprop/dgrad/wgrad GEMM | 82.294 ms | 17.59% |
| BF16 mean corrections | 88.366 ms | 18.89% |
| rank-64 BF16 fprop/backward | 25.445 ms | 5.44% |

这些比例是 profiler 归因估计，不是可严格相加的互斥计时，因为异步和
嵌套 NVTX projection 会重叠。原始 range 数量与 median 均保留在
`qwen15_profile_breakdown.json` 和 Nsight CSV 中。

主要问题是：占 step 较小的一部分 native GEMM 即使局部加速，也不足以
覆盖 mean、quantize、scale、swizzle、dequantize correction、额外
rank-64 路径和更多参数的 optimizer 开销。

## 原生 FP4 Tensor Core 证据

实现调用
`transformer_engine.pytorch.cpp_extensions.general_gemm`，主 operands
在调用点仍是带 scale metadata 的 packed `NVFP4Tensor`，没有先恢复为
BF16 再执行主 GEMM。

Nsight Systems 在完整 1.5B train-only step 中记录到 1,568 个原生
block-scaled GEMM 实例。核心 kernel 名为：

```text
cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_
ue4m3xe2m1_ue4m3xe2m1_f32_bf16_bf16_128x128x256...
```

其中 `sm120`、`bstensorop`、`block_scaled` 和两个
`ue4m3xe2m1` operand 明确对应 Blackwell SM120 的 block-scaled
NVFP4 Tensor Core 路径。两类普通/stream-K kernel 合计约占该 profile
GPU kernel execution time 的 13.1%。

Nsight Compute 因机器未开放 performance counter 权限而返回
`ERR_NVGPUCTRPERM`；因此硬件判定采用 kernel metadata，而不是 NCU
counter。

## 正确性

`tests/test_native_nvfp4.py` 的 7 项测试全部通过：

- native mean backward 与同一 packed value 的 dequantized reference；
- full rank-64 forward 与 dequantized reference；
- rank-64 SVD reconstruction；
- full autograd 对 residual/V/U/s/bias 路径的反向接线；
- Q/K/V、Gate/Up activation packing 共享；
- fused AdamW step 后 packed weight cache 失效。
- 非 32 对齐的真实 GRPO row 数会安全 pad 到 SM120 Wgrad 所需边界，
  并在输出处裁回原 shape。

dgrad/wgrad 相对 L2 误差分别约为 `0.439%/0.437%`，低于预设 `0.5%`
门槛。逐元素比较不适合这里，因为 SM120 block-scaled GEMM 与普通 BF16
matmul 的 scale application/reduction order 不同。

模型级 smoke 结果：

- 0.5B：`794/794` 个 gradient tensor 有限；
- 1.5B：`926/926` 个 gradient tensor 有限；
- 两个模型的 loss 均有限，且 optimizer step 后探针参数发生变化。

## 显存和初始化

| 模型 | BF16 peak allocated | Native peak allocated | BF16 参数量 | Native 参数量 |
|---|---:|---:|---:|---:|
| 0.5B | 4.40 GiB | 5.18 GiB | 494,032,768 | 529,236,352 |
| 1.5B | 12.58 GiB | 14.78 GiB | 1,543,714,304 | 1,617,585,920 |

rank-64 SVD + module replacement 在空闲物理 GPU 3 上约需 4.91 s（0.5B）
和 13.50 s（1.5B），按计划排除在 step-time 外。

## Batch=4 诊断

为了判断 batch=1 是否没有充分利用 Tensor Core，额外运行了一轮
batch=4、sequence length 512 的诊断（5 warmup + 10 measured）。该结果
不替代主实验：

| 模型 | BF16 median | Native median | BF16/NVFP4 | NVFP4 延迟变化 | Peak GiB BF16/NVFP4 |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 95.433 ms | 389.162 ms | 0.245x | +307.79% | 6.98/7.99 |
| Qwen2.5-Math-1.5B | 175.968 ms | 479.833 ms | 0.367x | +172.68% | 13.01/15.82 |

更大的 M 让 1.5B 的相对差距从约 4.21x 缩小到约 2.73x，但 NVFP4 仍
明显变慢；0.5B 基本没有改善。

## 实验配置和数据

- GPU：宿主物理 GPU 3，RTX 5090，UUID
  `GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959`；
- 正式运行前后：P1，SM clock 约 2.895–2.902 GHz；
- PyTorch：`2.12.0a0+0291f960b6.nv26.04.48445190`；
- CUDA：13.2；
- Transformer Engine：`2.15.0+42b8400`；
- Transformers：4.52.3；
- batch size 1，sequence length 512，SDPA；
- gradient checkpointing 开启；
- fused AdamW，learning rate `1e-5`；
- rank 64，activation mean-residual，gradient mean-residual；
- DeepMath train parquet SHA-256：
  `e0c5b2fc11978d735a7710273920676977b533e185284044c3eafa63a24479d7`；
- seed 2026，同一组 64 个 sample index；两个模型分别用自己的 tokenizer。

正式 run 中存在少数近似周期性的系统长尾，BF16 和 NVFP4 都会出现。
因此主结论使用 30 步 median，并同时保留 p10/p90 和每步原始记录。

## 限制

1. 容器内 CUDA ordinal 采用 `FASTEST_FIRST`，`CUDA_VISIBLE_DEVICES=3`
   并不对应宿主物理 GPU 3。正式结果只使用 UUID 锁定后、隔离在
   `formal_clean/` 的 12 个 JSON；它们全部通过 UUID 审计。
2. 已安装 TE 的 stochastic FP4 conversion kernel 没有以本机所需的
   architecture-specific SM120a 形式编译。启用 gradient stochastic
   rounding 会逐线程报告：
   `FP4 cvt PTX instructions are architecture-specific`。正式计时因此
   使用确定性 gradient NVFP4 rounding。主 GEMM 仍是原生 FP4。
   这比软件模拟随机舍入更有利于性能，所以当前“仍慢约 4.2x”的结论
   可视为现有容器下偏乐观的 NVFP4 性能下界，但不能宣称已测到
   stochastic-rounding 版本的精确时间。
3. 本页主结论只回答 batch 1、sequence 512。后续已经用完整 GRPO
   workload 搜索到 batch 128；结果单独记录在
   [GRPO_RESULTS.md](GRPO_RESULTS.md)，不应与本页数值混合。

## Artifact

正式原始数据（ignored，不提交 Git）：

```text
outputs/native_full_nvfp4_5090/formal_clean/
  cache/
  results/                         # 12 个正式 JSON
  summary.csv
  summary.json
  summary.md
  profile/
    qwen15_native_one_step.nsys-rep
    qwen15_native_one_step.sqlite
    qwen15_native_one_step_cuda_gpu_kern_sum.csv
    qwen15_native_one_step_nvtx_gpu_proj_sum.csv
    qwen15_profile_breakdown.json

outputs/native_full_nvfp4_5090/diagnostic_bs4/
  results/
  summary.{csv,json,md}
```

矩阵级结果：

```text
outputs/native_full_nvfp4_5090/gemm_microbench.json
```
