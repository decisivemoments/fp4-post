# SM120 全线性层优化：residual FP4 + low-rank BF16 U 融合设计草案

> 状态：讨论稿。本文只定义设计、验证门槛与风险；不改变 forward 实现。

## 1. 目标与边界

本设计服务于完整推理 linear，而不是独立低秩 GEMM。目标范围是
`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` 的总 forward latency；
设 `r=64`，单个 projection 当前的有效计算为：

```text
Z[M,r] = ((X_residual[M,K] @ V_fp4[r,K]^T) + mean_correction_V[1,r]) ⊙ s[1,r]
Y[M,N] =  fp4_gemm(X_residual[M,K], R_fp4[N,K])
        + mean_correction_R[1,N]
        + Z[M,r] @ U_bf16[N,r]^T
        + module_bias[1,N]
```

其中 `X_residual`、`R`、`V` 的 FP4 表示及其量化语义必须保持现状；第一版的
`U` 明确保持 BF16，不量化，也不把 `s` 折叠进 `U`。

目标不是把整个 `X(R + USV)^T` 重结合为一个权重 GEMM。那样会改变现有算法的
量化边界和误差行为。目标是让 `Y` 的 tile 只从寄存器写入一次，而不是先写
`X@Rᵀ`，再由 `addmm_` 读回并写入 `Z@Uᵀ` 的累加结果。

### 1.1 全层优化的度量口径

每个 projection 的目标时间应包含：activation pack、residual FP4 GEMM、V FP4
GEMM、`s` scale、low-rank correction、mean correction 和 module bias；不能只报告
`Z@Uᵀ`。Qwen 0.5B、`M=131072` 的当前 benchmark 表明不同层的瓶颈不同：

| projection | BF16 linear | 当前 NVFP4 linear（centered pack） | 优先级 |
| --- | ---: | ---: | --- |
| q | 0.986 ms | 1.389 ms | 中：必须先消除 NVFP4 固定成本 |
| k | 0.199 ms | 0.636 ms | 低：小 N，launch/pack 主导 |
| v | 0.206 ms | 0.636 ms | 低：小 N，launch/pack 主导 |
| o | 0.979 ms | 1.078 ms | 中 |
| gate | 5.085 ms | 3.767 ms | 高 |
| up | 5.112 ms | 3.787 ms | 最高 |
| down | 5.318 ms | 3.504 ms | 高 |

因此本 BF16-U fusion 首先为 up/gate/down 做；同一 kernel API 后续覆盖 q/o，
但不应为了 k/v 的小 N 牺牲大 projection 的 tile 或 shared-memory 配置。

## 2. 一个必须保留的中间张量：Z

“把 activation 与 residual/USV 融合”容易被理解为在每个输出 CTA 内直接算：

```text
(X @ V^T) @ U^T
```

这在 RTX 5090 上不是正确路线。每个 `[tile_M,tile_N]` 输出 CTA 都需要同一
`Z[tile_M,64]`；若不 materialize `Z`，它会为每一个 N tile 重算 `X@Vᵀ`。以
Qwen 0.5B up-proj 的 `N=4864`、`tile_N=128` 为例，约有 38 个 N tile，即同一
V projection 会被重复约 38 次。

SM120 没有 B200 那样可用于此目的的 TMA multicast；CTA 间不能以低成本共享这
个 `[tile_M,64]` 结果。因此第一版应保留一次独立的 V FP4 GEMM，物化很小的
`Z[M,64]`，然后融合 **residual GEMM 与 BF16 U GEMM**。这是本设计唯一的
materialized intermediate。

对 Qwen 0.5B up-proj、`M=131072,N=4864,r=64`：

| 张量或计算 | 大小/工作量 |
| --- | ---: |
| 最终 `Y` BF16 | 1.1875 GiB |
| 保留的 `Z` BF16 | 16 MiB |
| `U` BF16 | 0.594 MiB |
| `Z@Uᵀ` | 81.60 GFLOP |
| 可消除的 `Y` write + reread | 2.375 GiB |

这里的 2.375 GiB 是本 fusion 的核心机会；相比之下，16 MiB 的 `Z` 是合理
代价。

## 3. 当前路径与目标路径

```text
当前：
  Xq,Rq ── FP4 GEMM + biasR ──> C[M,N] 写入 HBM
  Xq,Vq ── FP4 GEMM + biasV ──> Vout ── scale(s) ──> Z[M,64]
  Z,Ubf16,C ── BF16 addmm(beta=1) ──> C[M,N] 读 + 最终写

目标：
  Xq,Vq ── FP4 GEMM + biasV ──> Vout ── scale(s) ──> Z[M,64]
  Xq,Rq,Z,Ubf16,biasR ── fused CTA ──> Y[M,N] 只写一次
```

因此，第一版不会减少 activation `Xq` 的读取次数：V GEMM 与 fused output GEMM
都需要它。减少的是 residual-output `C[M,N]` 的中间 HBM round trip。

## 4. 推荐的 SM120 kernel 结构

### 4.1 CTA ownership

一个 CTA 负责一个 `[128,128]` 的 `Y` tile，采用当前已验证的 SM120 FP4
collective：

```text
CTA(m_tile, n_tile)
  accumulator_R = blockscaled_fp4_mainloop(Xq[m_tile,:], Rq[n_tile,:])
  accumulator_U = bf16_rank64_mainloop(Z[m_tile,:], Ubf16[n_tile,:])
  accumulator   = alpha_R * accumulator_R + accumulator_U
  Y tile        = bf16_epilogue(accumulator + mean_correction_R + module_bias)
```

`accumulator_R` 与 `accumulator_U` 都保留在 CTA 的 register fragments 中；只能
有一个 epilogue 和一次 D-store。不能先调用当前 `GemmUniversalAdapter` 得到
`C`，然后在另一个 kernel 做 U GEMM，因为那仍会产生中间写回。

### 4.2 两个 mainloop 的职责

| 部分 | 数据格式 | 推荐实现 | 原因 |
| --- | --- | --- | --- |
| residual | X/R: NVFP4 E2M1 + E4M3 scale | 复用现有 SM120 `OpClassBlockScaledTensorOp` collective | 已证明与 TE NVFP4 同精度、同量级 |
| low-rank U | Z/U: BF16，K=64 | 已验证的 8-warp `16×16×16` BF16 WMMA，4 个 K=16 steps | 当前 CUTLASS SM120 builder 无 BF16 collective；避免 U 再量化 |
| epilogue | FP32 accumulator → BF16 | SM120 pointer-array EVT | device alpha、mean correction、module bias 一次完成 |

SM120 对 BF16 的硬件 primitive 仍是 legacy MMA；真正 Blackwell-specific 的部分是 residual 的
block-scaled FP4 mainloop、TMA pipeline、persistent scheduler 与 epilogue。
CuTe BF16 atom 已完成验证，但其 `16×8` warp tile 与当前 epilogue fragment 协议
不匹配，详见第 7 节的实验结果；当前生产候选保留 WMMA。

### 4.3 必须复制/扩展的 CUTLASS 层级

`GemmUniversalAdapter` 封装了“一条 mainloop + 一条 epilogue”，不能直接调用
两次后共享 accumulator。实现时应以当前项目的
`csrc/lowrank_nvfp4_sm120/lowrank_nvfp4_sm120.cu` 为 mainloop reference，创建
项目专用 kernel layer：

1. 保留 FP4 `CollectiveMainloop::load/init/mma` 的 TMA/pipeline 部分；
2. 在 FP4 mainloop 完成、最终 epilogue 之前，加载 `Z` 与 `U` tile；
3. 执行 BF16 `r=64` tiled MMA，累加至相同的 FP32 output fragment；
4. 调用一个支持 `alpha_R` device pointer、bias vector 的 SM120 EVT epilogue。

CUTLASS 3.x 的设计正是由 collective mainloop 与 collective epilogue 组合成
kernel；SM120 也提供完整 EVT epilogue family。此处需要增加的是第二条
mainloop，而不是在 Python 中串联两个 GEMM。

## 5. 为什么有机会超过“两个 GEMM 调库”

单独比较 `Z@Uᵀ` 与 `addmm_` 没有意义：`addmm_` 已经接近 BF16 standalone 的
最佳实现，而单独 kernel 必须读写整个 `C`。融合后比较对象应是完整序列：

```text
TE residual FP4 GEMM -> C 写
+ torch.addmm_(Z,U^T,C) -> C 读/写
```

而目标 kernel 的 HBM 行为是：

```text
fused FP4 residual + BF16 U -> Y 只写
```

理论上消除了 2.375 GiB 的 C traffic。`Z@Uᵀ` 的 81.60 GFLOP 很小；若
BF16 mainloop 与 FP4 residual 的 accumulator/epilogue 调度没有造成 register
spill 或占用率崩溃，它应主要填补 FP4 pipeline 的空隙，而非重新引入一个完整的
memory-bound C update。这个结论是性能假设，不是保证，必须用第 7 节的基线
验证。

### 5.1 4× 目标的正确口径

`BF16 × BF16 -> FP4 × FP4` 的约 4× 是我们希望尽量逼近的 **完整 linear 的
计算部分**目标，而不是只优化一个低秩 GEMM。它是硬件计算吞吐上限，不会自动包含
activation pack、mean correction、launch、输出 store 等固定成本。刚用本仓库
baseline 在 RTX 5090 上测得 up-proj 的组成：

| 当前部分 | Qwen 0.5B up-proj，M=131072 | 时间 |
| --- | ---: | ---: |
| residual FP4 GEMM | `131072×896×4864` | 1.279 ms |
| V FP4 GEMM | `131072×896×64` | 0.095 ms |
| scale + BF16 `addmm_` | `131072×64×4864` | 1.901 ms |
| 合计 | — | 3.349 ms |

若把整条路径机械地除以 4，目标是 0.837 ms；它已经低于当前 residual FP4 GEMM
自身的 1.279 ms。这不是降低目标，而是说明全层接近 4× 至少还需要三类优化：

1. 本文的 single-store fusion，消除 residual output 的 write/read；
2. 进一步降低/隐藏 activation pack 与 V branch 的固定成本；
3. 将 U/Z low-rank mainloop 从 BF16 改为独立量化后的 NVFP4。

第一版 BF16-U fusion 的阶段性目标是：让 residual + U 组合接近 residual FP4
GEMM 的时间，加上 V branch 后争取约 1.4–1.8 ms 总时长，即约 1.9–2.4× up/gate
端到端提升。它是为最终全层接近 4× 铺平 output 数据流，而不是最终目标本身。

后续若要逼近 low-rank 乘法的 4×，需要在本 BF16 fusion 验证完数据流后，将 U
与 Z 分别量化为 NVFP4，并把第二条 BF16 mainloop 替换为 SM120 block-scaled FP4
mainloop。那是下一阶段，而不是把 BF16 U kernel 人为和 FP4 理论峰值比较。

SM120 的单 CTA shared-memory 上限是 99 KiB，且每 SM 128 KiB、48 warps；因此
初版必须测量 `sizeof(SharedStorage)`、寄存器数与 occupancy。不能盲目为 U 分配
大 staging buffer。`Z[128,64]` 与 `U[128,64]` 各为 16 KiB BF16，双缓冲会很快
侵占 SM120 的共享内存预算。NVIDIA 的 tuning guide 给出了这些 SM120 resource
limits；CUTLASS 官方也明确将 SM120 block-scaled mainloop 与 SM120 EVT 作为
GeForce Blackwell 路线。 [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/)
[CUTLASS SM120 support](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md)

## 6. 关键实现风险与处理

| 风险 | 触发原因 | 处理/判定 |
| --- | --- | --- |
| 重算 V projection | 试图不物化 Z | 设计禁止；Z 必须先由单独 FP4 GEMM 写一次 |
| 共享内存超 99 KiB | FP4 TMA stages + U 双缓冲太大 | 先单缓冲 U，记录 shared/register/occupancy，再考虑 pipeline |
| register spill | 同时持有 FP4 与 BF16 accumulator fragments | 首版固定 `[128,128]`，只在 profiler 显示 spill 后降 tile 或改变 warp mapping |
| BF16 branch 抢占 FP4 scheduler | 两类 MMA 在同一 CTA | 分阶段串行，而非假设两类 tensor core 可无代价并发；测总 kernel 时长 |
| mean/bias 漏项 | 当前 residual/V 均有 mean correction | fused epilogue 必须收 `mean_correction_R + module_bias`；Z 必须来自已含 `mean_correction_V` 的 V path |
| alpha 语义错误 | TE NVFP4 有 global amax correction | 复用当前 custom operator 的 device alpha formula 与 swizzled scale layout |
| output 精度漂移 | accumulation/reorder 与 TE 不同 | 对同一 packed input，以 BF16 U baseline 作 comparison，定义 max/mean error gate |

## 7. 逐阶段实施与验收

### Phase A：建立没有误导性的 baseline

固定同一组 `Xq,Rq,Vq,Z,U,C,bias`，并计时 CUDA events：

1. **baseline-0**：当前 `TE residual FP4 GEMM + torch.addmm_`。
2. **baseline-1**：当前 `TE residual FP4 GEMM + project BF16 low-rank kernel`。
   仅用于确认 standalone 不值得继续优化。
3. **target**：新的 fused FP4 residual + BF16 U kernel。

所有 timing 都不包括 X/Z/U quantization；它们在三者中保持完全相同。第一轮只测
`M=131072,N=4864,K=896,r=64` 的 Qwen 0.5B up-proj prefill；kernel 正确后立刻
复测 gate、down、q/o、k/v，并回到完整 linear benchmark 纳入 activation pack。

### Phase B：功能 kernel

- 先无 bias，验证 `FP4 residual + BF16 ZU`；
- 加 `mean_correction_R`；
- 合并 module bias 至同一 N-vector；
- 确保输出没有中间 `[M,N]` allocation 或 store；
- Nsight Systems 中只应看到 V GEMM、fused kernel，以及必要的小 alpha/scale kernel。

#### 2026-09-04 原型状态：SM120 EVT/R2S 坐标风险

`csrc/fused_residual_lowrank_bf16/` 已实现 FP4 residual mainloop、device-side
alpha、以及一个暂时以标量 rank-64 dot 验证数据流的 EVT leaf。它不产生独立
`[M,N]` residual output，因此数据流目标没有退化成 `GEMM -> C -> addmm`。

不过该 kernel **尚不可用于性能或集成结论**。TE residual-only 路径已经与 custom
mainloop 对齐；加入 low-rank 项后，128x128 功能测试仍不通过。调试确认
`ConsumerStoreArgs::tCcD` 是 TMA/R2S 向量 predication 坐标，不能作为每个
accumulator scalar lane 的 `(m,n)`。目前改用 virtual row/column counting tensors
并经 `thread_mma.partition_C`、`thread_r2s.retile_S` 映射；mismatch 已从约 98%
降至约 91%。进一步按 `rows(_, epi_m, epi_n)` 选择 subtile 后降至约 85%，确认
R2S 坐标链路有效。lane probe 显示同一物理 output row 内仍存在非线性的
vector/lane permutation，连续 N 元素并不对应连续 R2S scalar indices；因此不能以
scalar coordinate lookup 继续硬解该 permutation。

后续实现应在 shared memory 中先构造当前 `[64,32]` low-rank subtile、通过标准
R2S auxiliary-load 路径喂给 epilogue。这将复用 CUTLASS 已验证的 lane
permutation；不允许退回 materialize `[M,N]`。

shared-tile 原型已经以 `DefaultCopy` 编译并完成了不死锁的运行验证。其数值仍未
通过：这不是 low-rank 数学或同步问题，而是普通 row-major scratch layout 与
`args.tiled_copy` 所描述的 D shared-memory tiler 不同。下一版必须从
CollectiveEpilogue 的 D shared layout atom 构造同构的 `[64,32]` scratch layout，
再把 `DefaultCopy` 替换为 LDSM S2R；在此之前不得计时。

### 2026-09-04 shared-tile 功能验证结果

已改为 provisional/final epilogue：provisional builder 导出相同配置下的
`SmemLayoutAtomD` 与 `EpilogueTile`，final callback 用该 atom 重建 scratch 的
swizzled `[64,32]` layout 并以 LDSM S2R 读取。以下 gate 已通过：

- 128×128 residual-FP4 + ZU 与 `TE GEMM + addmm_` 对齐；
- residual-only 为零 low-rank 的隔离测试对齐；
- 256×256 多 CTA tile offset 测试对齐。

功能版使用每元素 64 次标量 BF16 FMA 来形成 shared ZU tile。128×128×128,
rank-64 的中位时间为 0.398 ms，而 TE residual GEMM + addmm 为 0.072 ms（约
0.18×）；它证实单 D-store 数据流正确，但绝不代表可用性能。下一步唯一应优化的
部分是将 shared `[64,32]` ZU tile 的标量生成替换成协作式 BF16 tensor-core MMA，
不能再把时间花在 residual D2D traffic 上。

在更接近 Qwen-0.5B up-proj 的 warmed-up 小 prefill 点
`M=8192,N=4864,K=896,r=64`，TE residual GEMM + addmm 为 0.213 ms，而当前
scalar shared-tile fusion 为 1.734 ms（0.123×）。未 warm-up 的首个 TE 调用约
98 ms，不能用于比较。这个点进一步证明优化目标应完全聚焦 BF16 MMA tile 生成；
单 D-store 本身不能补偿 64 次标量 dot 的计算代价。

### 2026-09-04 BF16 WMMA tile 结果

已将 scalar tile generator 替换为 8 warp 的 BF16 WMMA：每 warp 计算一个
`16×16` 输出块、4 次 K=16 MMA 覆盖 rank 64，并把 FP32 accumulator scatter 到
与 D 相同的 swizzled scratch。原有 128×128、residual-only 与 256×256 多 CTA
四项正确性测试全部通过。

| workload | TE residual FP4 GEMM + addmm | fused WMMA single-D-store | speedup |
| --- | ---: | ---: | ---: |
| M=8192,N=4864,K=896,r=64 | 0.207 ms | 0.229 ms | 0.905× |
| M=131072,N=4864,K=896,r=64 | 3.160 ms | 3.089 ms | **1.023×** |

这证明“避免 residual `[M,N]` 中间落盘”在目标大 prefill 上确实有正收益，但只有
约 2.3%。

### 2026-09-04 Z/U shared staging 复测与结论

随后实现并测试了把一个 epilogue `[64,32]` subtile 的 `Z[64,64]`（8 KiB）和
`U[32,64]`（4 KiB）搬进 shared memory、让 8 个 WMMA warp 复用的版本。这个直觉
在独立 BF16 GEMM 中可能成立，但在本 fused kernel 中失败：callback 的 shared
storage 会被 `StageCountAutoCarveout` 从 FP4 TMA mainloop 的 pipeline 预算中扣除，
并且每个 subtile 额外需要一次 CTA 同步。

| workload | TE residual FP4 GEMM + addmm | direct-global WMMA | Z/U shared-staged WMMA |
| --- | ---: | ---: | ---: |
| M=8192,N=4864,K=896,r=64 | 0.205 ms | 0.229 ms | 0.494 ms |
| M=131072,N=4864,K=896,r=64 | 3.161 ms | 3.144 ms | 7.483 ms |

因此当前代码保留 direct-global WMMA 版本：它只使用原本就需要的 low-rank scratch
和 FP32 tile accumulator，不为 Z/U 分配 shared buffer。这里的瓶颈不是未缓存的
Z/U load，而是把低秩 BF16 MMA 串入 FP4 mainloop 后的 CTA 并行度、epilogue
synchronization 与资源竞争。下一步不能再增加 CTA-local staging；需要转向真正的
dual-mainloop/fragment-level integration，或将 Z/U 分别量化为 NVFP4 后重新评估。

### 2026-09-04 CuTe/CUTLASS BF16 tiled-MMA 子模块验证

为避免把 WMMA API 当作最终设计，原型进一步用 CuTe
`MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>` 构造了 BF16 tiled-MMA。CuTe 路径
可以编译，并且最初的 `Layout<4,2>` / `[64,16]` 分两次方案通过全部 4 项数值测试；
但性能为 0.279 ms（M=8192）和 3.978 ms（M=131072），明显弱于 direct WMMA。

从已编译 cubin 可见，这个 SM120 FP4 fused kernel 的 CTA 为 384 threads（12 warp）。
CuTe BF16 atom 的硬件粒度是每 warp `16×8×16`，而当前 R2S epilogue 的 natural
low-rank scratch subtile 是 `[64,32]`。因此 `Layout<4,2>` 只使用 8 warp 且必须
串行做两次。尝试 `Layout<4,3>` 处理 `[64,24]`、再以 4 warp 补 `[64,8]` 虽然能
编译，但尾 tile 违反当前 callback 的 producer/consumer 协议，4 项测试全部失败，
并且即使忽略数值错误也只有 3.719 ms。

结论：本代 SM120 **当前 EVT callback 内**不能以这个 CuTe BF16 atom 替换已验证的
`16×16×16` WMMA 实现。最终代码恢复为 WMMA，保留 4 项正确性测试通过的版本。这里
并不是放弃 CUTLASS：FP4 residual mainloop、TMA pipeline、D-layout scratch 和
epilogue 全部仍由 CUTLASS 管理；只是 BF16 low-rank 的细粒度 tensor-core atom 与
现有 callback 的 warp/fragment 形状不匹配。若继续投入 CUTLASS BF16，正确方向是
把 BF16 `CollectiveMma` 作为独立第二条 mainloop 与 residual fragment 合并，而不是
在 EVT leaf 中拼装 SM80 atom。

此外，检查本仓库锁定的 CUTLASS SM120 builder
`include/cutlass/gemm/collective/builders/sm120_mma_builder.inl` 后确认：其
`CollectiveBuilder<arch::Sm120, OpClassTensorOp, ...>` 目前有静态限制，只支持
F8F6F4 MMA，并不支持 BF16 operands。因此无法在 RTX 5090/SM120 上直接实例化
第二条 BF16 `CollectiveMma` mainloop；已有
`csrc/lowrank_bf16_fused/lowrank_bf16_cutlass.cu` 的 SM80 device GEMM 也会被
GeForce Blackwell 拒绝，不能作为生产候选。该判断已做运行时复验：扩展可以用
`sm_120a` 编译，但在 RTX 5090 上首次调用返回 `CUTLASS run failed: Error Internal`。

这给出了后续技术边界：若保持 BF16 U，生产实现只能使用当前已验证的 legacy
BF16 MMA（或 cuBLAS/TE `addmm_`）；若需要完全 CUTLASS-managed dual-mainloop，
则须等待/引入带 SM120 BF16 collective 支持的 CUTLASS，或自行实现 SM120 BF16
collective 的 TMA、LDSM、MMA 与 epilogue 协议。就当前 5090 的性能数据，后一条
工作量远大于 single-store 带来的约 0–4% 收益，优先级应低于 Z/U 单独 NVFP4 化。

保留的 BF16 WMMA tile 已通过 cubin SASS 复验：反汇编可见多条
`HMMA.16816.F32.BF16`，因此 BF16 operands 确实走 Tensor Core，且 accumulation
为 FP32；没有退化为 scalar FMA。性能差距应归因于 fused epilogue 的 tile/sync/
resource 调度，而不是 BF16 MMA 指令选择错误。

### Phase C：性能 gate

| gate | 通过条件 |
| --- | --- |
| 正确性 | 与 baseline-0 同 packed input 的 BF16 输出满足预先固定 tolerance |
| HBM 行为 | profiler 无 residual `C` 的中间 D2D/write-read pair |
| kernel 性能 | target ≤ baseline-0 的 residual GEMM + addmm 总时长 |
| 集成性能 | native linear 的端到端 forward 降时，不只单 kernel 降时 |
| 回退 | 任一 M/N 不满足时继续使用现有 TE + addmm 路径 |

## 8. B200 的后续版本（不属于第一版）

数学分解与 Z materialization 原则不变，但 B200/SM100 不能复制 SM120 device
代码。B200 版本应改用 `tcgen05`、TMEM accumulator、TMA multicast 与 CTA group
autotuning；其 228 KiB shared memory 和 64 warp/SM 允许更激进的 dual-mainloop
pipeline。SM120/5090 则应维持 1x1x1 cluster 与本设计的单 CTA ownership。两者
共享 dispatch API、测试和数值规范，不共享 kernel body。 [CUTLASS Blackwell
functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

## 9. 讨论时需要确认的三点

1. 是否同意：第一版保留一次 `Z[M,64]` materialization，只融合 residual 和
   BF16 U GEMM？
2. 是否同意：第一轮只锁定 up-proj prefill 的 `M=131072,N=4864`，不先为 decode
   或 down-proj 调 tile？
3. 是否同意：第一版的成功定义是融合后总时长低于 `TE residual + BF16 addmm`，
   而不是要求 standalone BF16 U kernel 单独胜过 `addmm_`？
