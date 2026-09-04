# 单层推理计算量与 RTX 5090 算力利用率

本实验固定 `sequence_length=512`，扫描 batch `1, 2, 4, 8, 16, 32`。这里的
Qwen 分别是 `Qwen2.5-0.5B-Instruct`、`Qwen2.5-1.5B-Instruct`、
`Qwen2.5-7B-Instruct`；运行脚本会从实际加载的 config 读取 shape，而非依赖
下表的硬编码值。

| 模型 | hidden H | MLP I | Q/O | K/V |
| --- | ---: | ---: | ---: | ---: |
| 0.5B | 896 | 4,864 | 896 | 128 |
| 1.5B | 1,536 | 8,960 | 1,536 | 256 |
| 7B | 3,584 | 18,944 | 3,584 | 512 |

## 计算口径

矩阵乘 `A[M,K] @ B[K,N]` 的逻辑计算量为 `2MKN FLOPs`，其中
`M = batch × 512`。单个 projection 的 `(K,N)` 是：Q/O `(H,H)`、K/V
`(H,H_kv)`、Gate/Up `(H,I)`、Down `(I,H)`。一个 decoder block 的线性
GEMM 计算量为七个 projection 的和：

`2 × M × (2H² + 2HH_kv + 3HI)` FLOPs。

这不把 RMSNorm、RoPE、SiLU、残差加法和 attention 的 `QK^T/PV` 计入
“线性层”数字；后者在完整 block 时间里实际存在。因此 block 的
`logical_peak_utilization` 是 **projection-GEMM 等效利用率**，不是整个 GPU
的硬件计数器利用率。需要硬件级 Tensor Core 指标时，应针对脚本的稳定区间再
跑 Nsight Compute。

## 每个 decoder block 的逻辑 GEMM 计算量

| batch | 0.5B | 1.5B | 7B |
| ---: | ---: | ---: | ---: |
| 1 | 15.27 GFLOP | 47.92 GFLOP | 238.64 GFLOP |
| 2 | 30.53 GFLOP | 95.83 GFLOP | 477.28 GFLOP |
| 4 | 61.07 GFLOP | 191.66 GFLOP | 954.56 GFLOP |
| 8 | 122.14 GFLOP | 383.33 GFLOP | 1.91 TFLOP |
| 16 | 244.28 GFLOP | 766.65 GFLOP | 3.82 TFLOP |
| 32 | 488.55 GFLOP | 1.53 TFLOP | 7.64 TFLOP |

## 峰值与实测利用率

默认采用 RTX 5090 dense Tensor Core 峰值：BF16 `419 TFLOP/s`、NVFP4
`1,676 TFLOP/s`。二者都是 dense 值；不要把 NVIDIA 宣传的 3,352 AI TOPS
（2:4 sparse FP4）用作本 benchmark 的分母。脚本允许通过
`--bf16-peak-tflops` 与 `--nvfp4-peak-tflops` 覆盖，以匹配最终确认的时钟或
厂商规格。

对于某次测得的中位延迟 `t_ms`，脚本写入：

`logical_TFLOP/s = logical_FLOPs / (t_ms × 10^9)`

`logical_peak_utilization = logical_TFLOP/s / peak_TFLOP/s`

BF16 的分母为 419，NVFP4 为 1,676。因而这个值能直接比较相同 workload 的
端到端有效吞吐；NVFP4 结果还包含本方法必须的 activation quantization、均值
BF16 correction 和低秩 BF16 path，不能把它误解为纯 FP4 GEMM kernel 的占用率。

本方法每个原 projection 的执行工作还可拆为：FP4 residual `(M,K,N)`、FP4
V `(M,K,r)`、BF16 low-rank `(M,r,N)` 和 BF16 mean correction `(1,K,N)`，
其中 `r=64`。因此它的实际指令 FLOPs 大于上面的逻辑 dense projection FLOPs；
逻辑数用于与原始 BF16 模型的端到端等效吞吐公平比较。
