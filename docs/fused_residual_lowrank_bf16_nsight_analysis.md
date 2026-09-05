# Fused residual FP4 + BF16 low-rank：Nsight 分析记录

> 记录日期：2026-09-04。本文保留命令、原始报告位置、测量边界、硬件证据和结论，
> 使后续优化可以从同一基线继续，而不是依赖口头记忆。

## 1. 被分析的算子和边界

目标实验算子位于：

- `csrc/fused_residual_lowrank_bf16/fused_residual_lowrank_bf16.cu`
- `csrc/fused_residual_lowrank_bf16/fused_epilogue.hpp`

它计算下面的后量化 forward 子路径：

```text
Y[M,N] = alpha * FP4(X[M,K]) @ FP4(R[N,K])^T + Z[M,64] @ U[N,64]^T
```

其中 residual 主 GEMM 是 CUTLASS SM120 block-scaled FP4 mainloop；rank-64
correction 是 8 个 warp 的 BF16 `16x16x16` WMMA tile，FP32 accumulate，结果先写进
与 CUTLASS D-layout 同构的 shared scratch，再由正常 epilogue 只写一次 `Y`。

这个实验**不含** activation pack、V FP4 GEMM、V mean correction、`singular_values`
scale、residual mean correction 和 module bias；它只回答“materialize Z 后，能否融合
residual FP4 GEMM 与 BF16 U correction，避免 `[M,N]` intermediate”的问题。

对照基线完全相同地排除上述部分：

```text
TE residual FP4 GEMM -> C[M,N]
torch.addmm_(Z, U.T, C) -> C[M,N]
```

## 2. 环境与工具

| 项目 | 实际值 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, SM120 (`major=12, minor=0`) |
| SM 数 | 170 |
| 每 SM registers | 65,536 个 32-bit registers |
| 每 SM shared memory | 102,400 B |
| 每 SM 最大 thread 数 | 1,536（48 warp） |
| Nsight Systems | 2026.2.1 |
| Nsight Compute | 2026.1.1 |

Nsight Compute 可以启动，但 counter collection 被宿主机限制：

```text
ERR_NVGPUCTRPERM: The user does not have permission to access NVIDIA GPU
Performance Counters on the target device.
```

因此本次不能直接报告 `dram__bytes`、`sm__throughput`、`tensor__throughput` 或
warp-stall counter。下面的带宽/算力判断严格区分了“Nsight Systems 已证实的资源事实”
与“由数据流得出的下界”；不能把后者误称为硬件 counter 实测值。

## 3. 可复现命令与原始工件

在容器 `sm-container` 内执行，GPU UUID 固定为：

```bash
cd /workspace/fp4_post
export CUDA_VISIBLE_DEVICES=GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99
export PYTHONPATH=src

nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none \
  --cuda-memory-usage=true \
  --output outputs/inference_nvfp4_5090/profiles/fused_residual_lowrank_bf16/m131072_n4864_k896_r64 \
  python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_bf16.py \
  --rows 131072 --input-features 896 --output-features 4864 --warmup 2 --iterations 3

nsys stats --force-export=true --force-overwrite=true \
  --report cuda_gpu_kern_sum,cuda_api_sum \
  outputs/inference_nvfp4_5090/profiles/fused_residual_lowrank_bf16/m131072_n4864_k896_r64.nsys-rep
```

保存的原始工件：

| workload | `.nsys-rep` | SQLite | text summary |
| --- | --- | --- | --- |
| `M=8192,N=4864,K=896,r=64` | `outputs/inference_nvfp4_5090/profiles/fused_residual_lowrank_bf16/m8192_n4864_k896_r64.nsys-rep` | 同名 `.sqlite` | 可由 `nsys stats` 重建 |
| `M=131072,N=4864,K=896,r=64` | `outputs/inference_nvfp4_5090/profiles/fused_residual_lowrank_bf16/m131072_n4864_k896_r64.nsys-rep` | 同名 `.sqlite` | `m131072_n4864_k896_r64_nsys_stats.txt` |

> Systems trace 本身有启动、module load 和 profiler 扰动；本文使用 CUDA GPU kernel
> duration，而不使用 host API 总时间比较 steady-state kernel 性能。正式 latency 仍由
> `torch.cuda.Event` benchmark 的中位数给出。

## 4. Nsight Systems 实测结果

### 4.1 小 prefill：`M=8192,N=4864,K=896,r=64`

Systems 的 kernel median：

| 路径 | kernel | median |
| --- | --- | ---: |
| fused | CUTLASS FP4 residual + BF16 tiled epilogue | 210.3 us |
| baseline | TE SM120 FP4 residual GEMM | 83.3 us |
| baseline | CUTLASS/cuBLASLt BF16 `addmm` | 75.4 us |

分离基线的两个主 kernel 合计约 158.7 us，小于 fused 的 210.3 us。这解释了小
prefill 下 fused 通常慢于基线：减少一次大 `C` 的读写，不足以抵消把两条不同形状
mainloop 放在一个低 occupancy CTA 内的损失。

### 4.2 大 prefill：`M=131072,N=4864,K=896,r=64`

Systems 的 5 次采样 median 和 launch 配置：

| 路径 | median | grid | block | registers/thread | dynamic shared |
| --- | ---: | ---: | ---: | ---: | ---: |
| fused CUTLASS FP4 + BF16 tile | 2.953 ms | `38 x 1024` | 384 | 168 | 99,328 B |
| TE FP4 residual GEMM | 1.245 ms | `1024 x 38` | 384 | 168 | 88,064 B |
| BF16 `addmm` | 1.861 ms | `32768 x 19` | 128 | 56 | 9,216 B |

同一 trace 中，分离两核合计约 3.106 ms，fused 为约 2.953 ms，即避免 intermediate
输出 traffic 产生约 4.9% kernel-time 收益。独立 CUDA-event benchmark 受时钟和
系统负载影响，会在约 1.00–1.07x 之间波动；两种测量均表明收益很小，而非数量级提升。

## 5. 已证实的资源瓶颈：occupancy

fused CTA 的 register demand 是：

```text
384 threads * 168 registers/thread = 64,512 registers / CTA
```

这几乎耗尽一个 SM 的 65,536 registers；同时 `99,328 / 102,400 = 97.0%` 的 SM
shared memory 也只允许一个 CTA。因此：

```text
resident CTAs / SM = 1
resident warps / SM = 384 / 32 = 12
max theoretical warps / SM = 1536 / 32 = 48
warp occupancy upper bound = 12 / 48 = 25%
```

这是以 Nsight launch metadata 和 CUDA device properties 直接计算的硬上界，而非
猜测。FP4 residual-only kernel 也因 `168 regs/thread` 和 `88,064 B shared` 被限制为
1 CTA/SM；而 standalone BF16 `addmm` 的资源上限约为：

```text
min(floor(65536 / (128*56)), floor(102400 / 9216), floor(1536 / 128))
= min(9, 11, 12) = 9 blocks / SM = 36 warps / SM = 75% thread occupancy
```

因此当前 fused kernel 不能最大程度利用 tensor core：不是 BF16 tile 退化了，而是它
必须在原本就只有 25% warp occupancy 的 FP4 CTA 中完成。SASS 已验证 low-rank tile
包含 `HMMA.16816.F32.BF16`，所以 BF16 运算确实进入 tensor core、FP32 accumulate；
问题是可同时发射/隐藏 latency 的 warp 数不足。

## 6. 带宽与计算量：能和不能从当前数据得出的结论

对大 prefill：

```text
FP4 residual FLOP = 2*M*N*K = 1.142461 PFLOP
BF16 low-rank FLOP = 2*M*N*64 = 81.604 GFLOP
total arithmetic     = 1.224066 PFLOP
fused median         = 2.953 ms
```

若把不同格式的 FLOP 机械相加，得到约 414.5 TFLOP/s；这**不能**同 NVFP4 或 BF16
单独理论峰值直接比，因为两段的 operand format、MMA throughput 和执行阶段不同。
它只能说明低秩 BF16 计算量约为 residual FP4 的 7.14%，却带来明显的 CTA resource
压力和 epilogue 工作。

最终输出 `Y` 为 1.1875 GiB BF16。只按一次 final store 计算，带宽下界为：

```text
1.1875 GiB / 2.953 ms = 431.8 GB/s
```

这只是下界，未包含 X/R/scale/Z/U reads、shared/L2 traffic 和 write allocate。它小于
高端显卡常见的峰值 HBM/GDDR 带宽量级，因而不能证明“已经跑满显存带宽”。在没有
Nsight Compute counter 的当前权限下，严谨结论是：**未获得证据表明 DRAM bandwidth
已饱和；反而有直接证据表明 register/shared 限制的 25% occupancy 是主要约束。**

## 7. 当前数据组织是否合理

已经正确、应保留的选择：

1. 保留 `Z[M,64]` materialization。取消它会让每个 N tile 重算 `X @ V.T`，代价远高于
16 MiB 的 Z。
2. `U[N,64]` 按 row-major 存储，并作为 column-major B 被 WMMA 解释；不发生显式
transpose/materialization。
3. residual FP4 accumulator 和 BF16 rank-64 correction 合并后，只做一次最终 D store。
4. 不在当前 FP4 CTA 内 staging `Z[64,64]`、`U[32,64]`。实测额外约 12 KiB shared
使 fused kernel 从 3.144 ms 退化到 7.483 ms，因为它挤压 FP4 TMA pipeline 并增加 CTA
barrier。
5. 优先依赖 L2、而非 CTA shared，复用 Z/U。大 workload 中 `U` 仅约 0.59 MiB，
`Z` 仅 16 MiB，均显著小于现代高端 GPU 的 L2 容量；grid 的 N 维为 38，若 scheduler
连续派发同一 M tile 的 N tiles，Z 会自然有 L2 reuse。下一次实验可将 Z/U 设为 CUDA
stream access-policy window 的 `cudaAccessPropertyPersisting`，同时记录 L2 bytes；这
不会增加 CTA shared 或降低 occupancy。该假设尚未获得 hardware counter 验证。

当前不合理、也是主要机会的部分：

1. BF16 correction 是在 FP4 mainloop 结束后的 epilogue producer 中执行；它不能像
standalone `addmm` 一样以 75% occupancy 独立调度。
2. fused CTA 的 99,328 B shared 和 168 registers/thread 几乎不给第二个 CTA 留资源；
任何继续增加 CTA-local cache 的方案都会更差。
3. 8 个 BF16 WMMA warp 生成 `[64,32]` tile 后，需要写 FP32 `lowrank_accum`，所有
participant 再 scatter 到 D-swizzled BF16 scratch，并经 LDSM 读回 epilogue fragment。
该转换不是 HBM 往返，但会消耗 shared bandwidth、barrier 和寄存器。

## 8. 优化方向，按可信度排序

### A. 不再增加 shared staging（高置信度否定项）

已被实际 benchmark 否定。Z/U 在该 CTA 内缓存看似减少 global load，但对 25%
occupancy 的 FP4 mainloop，额外 shared 只会让 pipeline 更差。

### B. 重新设计 epilogue tile / warp ownership（中等置信度）

当前 low-rank natural tile 是 `[64,32]`，只让 8 个 warp 做 `16x16` WMMA；CTA 有 12
warp，但输出 scratch、R2S consumer 和 FP4 producer 的协议固定。应先尝试能减少
`FP32 row-major scratch -> swizzled BF16 scratch -> LDSM` 转换次数的 tile 设计，而不是
盲目扩大 Z/U cache。验收门槛是仍不增大 `dynamic shared` 或 `registers/thread`，否则
occupancy 仍是 1 CTA/SM。

### C. 真正 dual-mainloop / fragment-level fusion（高潜力、高工作量）

理想数据流是让 BF16 low-rank accumulator 在 residual FP4 mainloop 的最后阶段直接
和 D fragment 对齐，消除 `lowrank_accum` 与 scratch conversion。当前 CUTLASS SM120
builder 只支持 F8F6F4 MMA，不提供 BF16 `CollectiveMma`；实现它需要手写 SM120
BF16 TMA/LDSM/MMA collective 或升级至有该支持的 CUTLASS。此路线才可能提升 tensor
core 调度，但工程量大，并且不能假定会得到超过数个百分点的端到端收益。

### D. Z/U 分别量化为 NVFP4（算法允许时，优先级最高）

这不改变 `singular_values` 不可折叠进 U 的约束：Z 与 U 分别 quantize 后再做 FP4
GEMM。它能让第二段使用项目已经验证的 SM120 block-scaled FP4 collective，而不是
在 BF16 legacy tile 上与 25% occupancy 竞争。需要把 quantization 成本、误差和完整
linear end-to-end latency 一起测量；不得只报告 low-rank GEMM。

## 9. 取得完整硬件 counter 的方法

宿主机管理员需要允许 non-admin GPU profiling，例如加载 NVIDIA 驱动时关闭
`NVreg_RestrictProfilingToAdminUsers`，或由管理员以允许 profiling 的安全策略启动
容器。权限具备后，推荐：

```bash
ncu --target-processes all --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*cutlass::device_kernel.*' \
  --export fused_full \
  python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_bf16.py \
  --rows 131072 --input-features 896 --output-features 4864 --warmup 3 --iterations 10
```

优先查看：

- `gpu__time_duration`；
- `dram__bytes_read/write.sum`、`lts__t_bytes.sum`，判断 DRAM/L2 带宽；
- `sm__throughput`、`sm__pipe_tensor_cycles_active`，判断 tensor core active；
- `smsp__warps_active.avg.pct_of_peak_sustained_active`，验证 25% occupancy；
- `smsp__warp_issue_stalled_*`，区分 memory、barrier、scoreboard 和 dispatch stall；
- register count 与 shared-memory occupancy report。
