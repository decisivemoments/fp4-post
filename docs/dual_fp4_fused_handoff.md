# SM120 dual-FP4 single-D-store kernel：Sol 交接文档

## 0. 交接结论

目标算子尚**未完成**。当前仓库已有两个已验证的基线：

1. residual FP4 GEMM + BF16 rank-64 `addmm_`；
2. residual FP4 GEMM + 单独的 NVFP4 rank-64 GEMM（`beta=1` 累加）。

需要实现的是第三种、严格 single-D-store 的 kernel：

```text
Y[M,N] = alpha_R * FP4(X[M,K]) @ FP4(R[N,K])^T
       + alpha_L * FP4(Z[M,64]) @ FP4(U[N,64])^T
```

其中 `Y` 只允许一次 global-memory store；不得 materialize residual
`C[M,N]` 后再由第二个 GEMM 读改写。

目标 GPU 为 RTX 5090 / SM120，主 workload：

```text
M=131072, N=4864, K=896, rank=64
```

## 1. 已提交基线与源码地图

当前基线提交：

```text
511d197 Add SM120 low-rank fusion experiments
```

关键文件：

| 文件 | 作用 | 状态 |
|---|---|---|
| `csrc/fused_residual_lowrank_bf16/fused_residual_lowrank_bf16.cu` | residual FP4 mainloop + BF16 callback fusion host/kernel instantiation | 可编译、正确 |
| `csrc/fused_residual_lowrank_bf16/fused_epilogue.hpp` | BF16 rank-64 epilogue callback；说明 D-swizzled scratch 路线 | 可编译、正确 |
| `csrc/lowrank_nvfp4_sm120/lowrank_nvfp4_sm120.cu` | 独立 rank-64 SM120 NVFP4 GEMM，`C += A@B.T` | 可编译、正确 |
| `src/Metis/Metis/lowrank_nvfp4_sm120.py` | TE packed tensor 到独立 custom NVFP4 kernel 的 Python wrapper | 可用 |
| `src/Metis/Metis/native_nvfp4.py` | 模型级 `lowrank_compute="nvfp4"` 两-GEMM 路径 | 可用，inference-only |
| `tests/test_fused_residual_lowrank_bf16.py` | 原 BF16 single-store 功能测试 | 通过 |
| `tests/test_lowrank_nvfp4_sm120.py` | custom rank-64 NVFP4 与 TE beta=1 的对齐测试 | 通过 |
| `benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_bf16.py` | BF16 fused 对照 benchmark | 可用 |
| `benchmarks/inference_nvfp4_5090/benchmark_lowrank_nvfp4_sm120.py` | custom/TE rank-64 NVFP4 benchmark | 可用 |

当前工作区可能包含未完成的：

```text
csrc/fused_residual_lowrank_nvfp4/dual_fp4_kernel.hpp
```

它只是从 CUTLASS kernel layer 复制出的 WIP，已加入第二组 params/load-init，**不可编译、不可作为实现基础信任**。可保留作源码定位参考，也可删掉重写。

## 2. 数学与独立量化语义

逻辑权重近似为：

```text
W ≈ R + U diag(s) V
Z = (X @ V^T) diag(s)
Y = X @ R^T + Z @ U^T
```

`Z[M,64]` 必须 materialize：若在每个 N tile 重算 `X @ V.T`，会让 V
projection 重复约 `N/128` 次。对目标形状，`Z` 约 16 MiB；最终 `Y` 约
1.1875 GiB。

四个 FP4 operands 必须分别量化：

```text
Xq/Rq: residual GEMM 的 data、E4M3 block scales、amax
Zq/Uq: low-rank GEMM 的 data、E4M3 block scales、amax
```

不要把 `singular_values` 折进 `U`；当前算法的语义是在 runtime 先生成
`Z = Vout * singular_values`，然后独立量化 `Z` 与 `U`。

TE/CUTLASS 的 global alpha 为：

```text
alpha(A,B) = amax(A) * amax(B) / (6 * 6 * 448 * 448)
```

因此 raw FP4 accumulators 不可直接相加。需要满足：

```text
Acc = Acc_R + (alpha_L / alpha_R) * Acc_L
Y   = alpha_R * Acc
```

`alpha_R == 0` 的测试边界必须处理：可选择非零的 base alpha（`alpha_R`，否则
`alpha_L`，否则 1），再对另一项计算 ratio，不能除零。

## 3. 当前 BF16 fused kernel 为什么正确但不是目标

当前 `LowrankBf16Dot` 的数据流：

```text
BF16 WMMA [64,32] result
→ FP32 row-major shared scratch
→ BF16 D-swizzled shared scratch
→ LDSM into epilogue register fragment
→ plus(residual fragment, lowrank fragment)
→ one D/Y store
```

它消除了大 `C[M,N]` 的 HBM round trip，但 BF16 WMMA 输出 fragment 与 residual
FP4 mainloop fragment ownership 不同，所以需要 scratch/layout conversion/barriers。

`D-swizzled` 是 CUTLASS epilogue 对最终 D tile 使用的 shared-memory address layout；
必须使用 `SmemLayoutAtomD` 对应 layout，普通 row-major scratch 会导致 LDSM
读取的 lane/element 映射错误。

## 4. 方案二：所需 kernel 生命周期

最终实现位置不是 EVT callback，而是 SM120 `GemmUniversal` kernel layer 中：

```text
CollectiveMainloop::mma(...)             // residual
<插入 low-rank FP4 load/MMA/accumulate>
CollectiveEpilogue::store(...)           // one global Y store
```

目标的每个 `[128,128]` work tile 生命周期：

```text
1. Producer: TMA load X/R residual stages
2. Consumer warpgroups: residual FP4 MMA → Acc_R (FP32 register fragments)
3. Consumer releases residual pipeline stages
4. Producer: reuse those same shared-memory stages; TMA load Z/U rank-64 tile
5. Consumer warpgroups: low-rank FP4 MMA → Acc_L with SAME TiledMma / fragment ownership
6. Per register slot: Acc_R += alpha_ratio * Acc_L
7. Existing CUTLASS epilogue store: BF16 Y, exactly once
```

关键约束：不能同时配置两套完整 SM120 TMA pipeline。residual-only fused kernel
已接近 shared-memory 上限；两套 pipeline 会超过 SM120 每 SM 可用 shared memory。
必须同 CTA、同 work tile、**顺序复用** pipeline storage。

## 5. CUTLASS 参考与建议的改造方法

当前 residual kernel经 `KernelScheduleAuto` 选择 SM120 cooperative warp-specialized
路径。主要参考：

```text
.deps/cutlass/include/cutlass/gemm/kernel/
  sm120_gemm_tma_warpspecialized_cooperative_asymmetric_dma.hpp

.deps/cutlass/include/cutlass/gemm/collective/
  sm120_blockscaled_mma_tma.hpp
```

`sm120_blockscaled_mma_tma.hpp` 的 `mma(...)` 明确展示：

```text
clear(accum)
→ shared-to-register FP4 operands/scales
→ cute::gemm(tiled_mma, ..., accum)
```

对方案二，建议建立项目内 custom kernel class，而不是改 vendored `.deps/cutlass`：

1. 从实际被 residual kernel 实例化的 `GemmUniversal` specialization 复制 kernel layer；
2. 扩展 `Arguments/Params`：增加 `lowrank_problem_shape`、第二个 mainloop params、
   `alpha_ratio` device pointer；
3. 改写 persistent scheduler，使 producer/consumer 针对**同一个 work tile**先完成
   residual、再完成 low-rank，而不是 residual producer 预取全部 tile 后再处理 low-rank；
4. rank-64 phase 复用 residual `TensorStorage` 与 pipeline barrier storage；在 phase 3/4
   之间需要所有相关 producer/consumer 完成 release/arrive，不能覆盖尚在使用的 stage；
5. low-rank phase 可使用临时 `Acc_L` 后与 `Acc_R` 做 per-element FP32 add；首个功能
   版本允许临时 accumulator 增加 register pressure。后续再把 collective MMA 改为可
   累加到现有 `Acc_R` 的变体（移除其内部 `clear(accum)`）；
6. 只在完成 two-mainloop accumulation 后调用原 epilogue/store。

注意：标准 `GemmUniversal` 只支持一个 mainloop；仅给 epilogue callback 传入 packed
Z/U 不足以实现 fragment-level fusion。callback 只能看到 epilogue 消费的局部 vector，
无法控制完整 `[128,128]` MMA fragment 或 residual TMA shared pipeline。

## 6. 为什么不能用两个小 GEMM 依赖 L1/L2 代替

如下路径不是严格 fusion：

```text
C_tile = residual GEMM
C_tile += low-rank GEMM(beta=1)
```

即使单 tile（128x128 BF16 = 32 KiB）可能 L2 hit：

- 首个 GEMM 仍须进行 global store；
- 第二个 GEMM 仍有 global load/read-modify-write 语义；
- L1 不跨 kernel/SM 保持；
- L2 hit 由调度/驱逐决定，不能保证；
- 全 `C` 远大于 L2，cuBLASLt/TE 不提供“同 tile 紧邻、同 SM”配对调度。

这可以作为 cache-locality 对照，不可替代 single-D-store kernel。

## 7. 已有测量与性能目标

目标 shape：`M=131072,N=4864,K=896,r=64`。

| 路径 | 时间 | 备注 |
|---|---:|---|
| TE residual FP4 + BF16 addmm | 约 3.16 ms | 后量化子路径 |
| 当前 BF16 single-store fusion | 约 3.09–3.14 ms | 约 1.02x，环境波动 |
| 独立 custom rank-64 NVFP4 | 1.698 ms | `C += Zq@Uq.T` |
| TE rank-64 NVFP4 | 1.683 ms | 同精度参考 |

Nsight Systems 曾观察到 BF16 fused CTA：384 threads、168 regs/thread、约 99,328 B
dynamic shared，最多 1 CTA/SM，理论 25% warp occupancy。不要为 Z/U 增加另一套
CTA-local cache；此前 staging Z/U 使大 shape 从约 3.14 ms 退化到约 7.48 ms。

验收不要只比较 isolated rank-64 GEMM。至少报告：

```text
TE residual FP4 + TE NVFP4(beta=1)
BF16 single-store fusion
new dual-FP4 single-store fusion
```

并分别报告小 prefill（M=8192）与大 prefill（M=131072）。

## 8. 功能验收

新测试至少覆盖：

1. `128x128x128, r=64`：与两次 TE NVFP4（residual + beta=1 lowrank）对齐；
2. residual 为零、low-rank 非零：验证 alpha fallback；
3. low-rank 为零：验证 residual 完全保留；
4. `256x256`：验证多 CTA tile offsets；
5. 使用两个独立 quantizer/packed object：验证没有混用 X/R 与 Z/U scale/amax。

建议公差以现有 `test_lowrank_nvfp4_sm120.py` 的同精度标准起步：

```python
torch.testing.assert_close(actual, expected, rtol=2e-2, atol=1.25e-1)
```

功能参考：

```python
expected = residual_te_nvfp4(...)
general_gemm(packed_u, packed_z, out_dtype=torch.bfloat16,
             layout="TN", out=expected, beta=1.0, accumulate=True)
```

## 9. 构建、测试、benchmark 与 NCU 命令

标准开发环境是**宿主机**的 `torch` conda environment；它已包含 PyTorch 和
Transformer Engine。不要假设必须使用 Docker 容器。

```bash
cd /home/dy/zxt/fp4_post
source ../conda-init.sh       # activates conda environment: torch
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

目标卡应通过 `CUDA_VISIBLE_DEVICES` 显式选择。现有 Nsight Systems 基线使用：

```text
GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99
```

例如：

```bash
export CUDA_VISIBLE_DEVICES=GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99
```

现有验证：

```bash
pytest -q tests/test_lowrank_nvfp4_sm120.py

python benchmarks/inference_nvfp4_5090/benchmark_lowrank_nvfp4_sm120.py \
  --rows 131072 --columns 4864 --warmup 5 --iterations 20

python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_bf16.py \
  --rows 131072 --input-features 896 --output-features 4864 \
  --rank 64 --warmup 5 --iterations 20
```

宿主机 NCU 已可用。新 dual-FP4 kernel 通过功能测试后，优先运行：

```bash
ncu --target-processes all --set full \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*DualFp4.*|.*device_kernel.*' \
  --launch-count 1 \
  --export outputs/inference_nvfp4_5090/profiles/dual_fp4_fused/ncu_dual_fp4 \
  --force-overwrite \
  python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_nvfp4.py \
    --rows 131072 --input-features 896 --output-features 4864 \
    --rank 64 --warmup 3 --iterations 10
```

若新 benchmark 的文件名或 kernel demangled 名称不同，应调整上面的 filter，确保只
profile fused kernel。优先记录：

```text
gpu__time_duration
dram__bytes_read.sum / dram__bytes_write.sum
lts__t_bytes.sum
sm__pipe_tensor_cycles_active
smsp__warps_active.avg.pct_of_peak_sustained_active
smsp__warp_issue_stalled_*
```

同时保留 CUDA-event median，避免仅使用 profiler 受扰动的单次时间作为性能结论。

## 10. 不要做的事情

- 不要把已经存在的 `lowrank_compute="nvfp4"` 两-GEMM 路径称作 fused；
- 不要在 Python 中串两个 GEMM 后称作 single store；
- 不要把 `Z` 取消 materialization；
- 不要将 `singular_values` 折进 U；
- 不要同时分配两个 full shared-memory pipeline；
- 不要在未通过 zero-residual / multi-tile tests 前测试性能；
- 不要只报告 low-rank GEMM 单独的速度。
