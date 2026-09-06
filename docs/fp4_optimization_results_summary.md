# FP4 block 优化：关键结果与原始数据索引

## 结论摘要

截至 2026-09-06，Qwen 2.5 0.5B 的一个 decoder layer（layer 0）在 RTX 5090 上，
目标 large-prefill shape `batch=256, sequence=512, M=131072` 的受控结果是：

| 路径 | Median | 相对 BF16 | 原始数据 |
|---|---:|---:|---|
| 完全 BF16 | 32.819 ms | 1.000x | `transformer_qwen0.5b_control_bf16_b256.json` |
| legacy native NVFP4：无 dual、无 RMSNorm→quant fusion | 28.024 ms | 1.171x | `transformer_qwen0.5b_control_legacy_nvfp4_b256.json` |
| 优化 native NVFP4：dual-FP4 + RMSNorm→quant fusion | **20.005 ms** | **1.640x** | `transformer_qwen0.5b_control_dual_rmsnorm_b256.json` |
| full-weight direct NVFP4，speed-only、忽略准确性/bias | 22.929 ms | 1.433x | `transformer_qwen0.5b_direct_nvfp4_b256.json` |

因此：

```text
optimized / legacy NVFP4 = 28.024 / 20.005 = 1.401x
optimized / BF16         = 32.819 / 20.005 = 1.640x
direct full-weight FP4 / BF16 = 32.854 / 22.929 = 1.433x
```

前三项是同一套模型、GPU、shape、activation layout、rank、warmup/iteration 下的受控
比较；direct-FP4 使用单独的同配置 BF16 run，所以只与其同文件中的 BF16 比较。

## 受控 block benchmark 的定义

所有 native NVFP4 条件使用 Qwen 0.5B、rank=64、rowwise activation、
`activation_pack_backend=centered_cuda`、`lowrank_compute=nvfp4`、warmup=5、
20 次 CUDA-event sample 的 median。完整 block 包含 RMSNorm、attention、RoPE、SiLU、
residual 等，而不仅是 projection GEMM。

| 条件 | dual-FP4 `DualFp4GemmUniversal` | RMSNorm→quant | 说明 |
|---|---|---|---|
| 完全 BF16 | 不适用 | 不适用 | 原始 Qwen decoder layer |
| legacy NVFP4 | 显式关闭 | 关闭 | residual FP4 GEMM 后，TE low-rank NVFP4 `beta=1` 二次 GEMM 累加 |
| 优化 NVFP4 | 开启 | 开启 | dual-FP4 final-store kernel；QKV、Gate/Up 直接消费 RMSNorm packed activation |
| direct NVFP4 | 不适用 | 不适用 | 每个完整权重/activation 直接 TE FP4 GEMM；忽略 bias，非准确性路径 |

受控原始 JSON 位于（这些 output 文件被 gitignore，不是本提交的副本）：

- [BF16 raw JSON](../outputs/inference_nvfp4_5090/transformer_qwen0.5b_control_bf16_b256.json)
- [legacy NVFP4 raw JSON](../outputs/inference_nvfp4_5090/transformer_qwen0.5b_control_legacy_nvfp4_b256.json)
- [dual + RMS fused raw JSON](../outputs/inference_nvfp4_5090/transformer_qwen0.5b_control_dual_rmsnorm_b256.json)
- [direct full-weight NVFP4 raw JSON](../outputs/inference_nvfp4_5090/transformer_qwen0.5b_direct_nvfp4_b256.json)

对应命令入口是 [benchmark_qwen_inference.py](../benchmarks/inference_nvfp4_5090/benchmark_qwen_inference.py)。
legacy 条件使用 `--disable-dual-fp4-fusion`；优化条件使用 `--fuse-rmsnorm-quant`。

## Dual-FP4 kernel：功能、isolated 性能和 NCU

dual-FP4 实现把 residual FP4 与 rank-64 Z/U FP4 的两个 MMA phase 放在同一 CTA，顺序
复用 TMA/shared pipeline，最终 epilogue 加 mean-correction bias 并只 store 一次 BF16 D。

目标 isolated post-quantization shape：`M=131072,N=4864,K=896,r=64`。

| 项目 | 数值 | 解释 |
|---|---:|---|
| TE residual FP4 + TE low-rank NVFP4 `beta=1` | 3.057 ms | 两个 GEMM 的参考 |
| dual-FP4 single-store | 1.563 ms | isolated kernel/event median，约 1.96x |
| NCU kernel duration | 1.634 ms | profiler 扰动下的单 kernel 时间，不用于替代 event median |
| L2 global write | 1,275,068,416 B | 恰等于 `131072×4864×2`，即一份 BF16 output |
| registers/thread | 168 | NCU launch resource |
| dynamic shared/CTA | 88,064 B | NCU launch resource |
| achieved occupancy | 20.85% | 384-thread CTA，资源限制为 1 CTA/SM |

原始/可追溯来源：

- [dual-FP4 mean-correction 验证与 NCU 解读](dual_fp4_mean_correction_validation.md)
- [结构化结果 JSON](dual_fp4_mean_correction_results.json)
- [NCU report](../outputs/inference_nvfp4_5090/profiles/dual_fp4_mean/fused_mean.ncu-rep)
- [NCU raw CSV](../outputs/inference_nvfp4_5090/profiles/dual_fp4_mean/ncu_raw.csv)
- [完整 native forward 的 Nsight Systems report](../outputs/inference_nvfp4_5090/profiles/dual_fp4_mean/native_fused.nsys-rep)

Nsight Systems trace 的关键证据是：最终 `DualFp4GemmUniversal` 后没有额外的 MxN
mean-correction add kernel；NCU 的 L2 write counter 则验证其最终 D store 仅写一份 output。

## RMSNorm + centered quant 融合

新 operator 不 materialize RMSNorm BF16 output。它：

```text
Phase A: X -> inv_rms[M] + per-column partial sum/min/max
Phase B/C: partial stats -> mean[H] + centered global amax
Phase D: X + inv_rms + gamma + mean -> TE rowwise NVFP4 packed data/scales
```

`inv_rms[M]` 保存为 FP32；目标 M 下仅 512 KiB，避免在 pack phase 再做整行平方和
reduction。QKV 共用一次 fused pack，Gate/Up 共用一次 fused pack。

功能证据：在 `(M,H)=(512,896)、(512,1536)、(512,3584)`，新 operator 相比
`Qwen2RMSNorm(x)` 后接原 centered pack 的 `mean`、`rowwise_data`、
`rowwise_scale_inv`、`amax_rowwise` 均 bitwise 相等。Qwen 0.5B 完整 layer 的旧 native
graph 与 fused RMSNorm graph 在同一输入和转换权重下也有 `torch.equal=True`。

来源：

- [设计与实现方案](rmsnorm_centered_nvfp4_fusion_plan.md)
- [RMSNorm block 结果/复现命令](rmsnorm_nvfp4_block_results.md)
- [直接 storage test](../tests/test_rmsnorm_nvfp4.py)
- [CUDA operator](../csrc/rmsnorm_nvfp4/rmsnorm_nvfp4.cu)

## direct full-weight NVFP4 speed-only 路径

`direct_nvfp4` 用 TE 直接量化完整 BF16 weights 与 runtime activation，然后调用 full
NVFP4 GEMM。它没有 SVD/residual/low-rank、没有 mean correction，也故意不加 module bias。
因此绝不能用于精度比较；其角色仅是当前 transformer graph 中的“直接 FP4”速度参考。

代码：[direct_nvfp4.py](../src/Metis/Metis/direct_nvfp4.py)。原始 measurement 是上文的
[direct raw JSON](../outputs/inference_nvfp4_5090/transformer_qwen0.5b_direct_nvfp4_b256.json)。

## 历史 JSON 的边界

[transformer_qwen0.5b.json](../outputs/inference_nvfp4_5090/transformer_qwen0.5b.json) 中的
batch=256 NVFP4 为 23.947 ms；该文件没有 git SHA 或 kernel-dispatch metadata。其文件
时间晚于 dual-FP4 mean-correction NCU 记录，且接近本次 dual-only 23.843 ms，不能用它
推断“未启用 dual-FP4”的性能。关于 dual 与 RMSNorm 融合的结论，应以本文受控三组结果
为准。

## 代码提交对应关系

| Commit | 内容 |
|---|---|
| `c9d04a0` | dual-FP4 residual low-rank fusion、mean epilogue、相关 benchmark/tests/docs |
| `f446b9f` | RMSNorm + centered NVFP4 operator、Qwen block wrapper、benchmark flag |
| `0c23674` | `--disable-dual-fp4-fusion` 受控 legacy baseline |
| `e81498f` | `direct_nvfp4` full-weight speed-only benchmark mode |
