# Transformer block：RMSNorm 与 centered NVFP4 activation packing 融合方案

## 结论

目标不是先生成 RMSNorm 的 BF16 output，再调用 centered quantizer；那仍会 materialize
完整 `Y[M,H]`。应实现一个不 materialize `Y` 的 inference pipeline，直接生成 native
linear 可消费的：

```text
mean[1,H] + TE-compatible packed_centered_activation[M,H]
```

Qwen decoder layer 的优先目标是两个共享输入组：

```text
input_layernorm     -> Q / K / V
post_attention_norm -> Gate / Up
```

Q/K/V 或 Gate/Up 共用同一份 packed activation。`O` 和 `Down` 的输入不是 RMSNorm
output，首版继续走现有 centered pack。本文件是实现方案，不含性能承诺。

## 已读基线：当前 mean + quant pipeline

相关代码：

- [centered_nvfp4.cu](../csrc/centered_nvfp4/centered_nvfp4.cu)
- [binding.cpp](../csrc/centered_nvfp4/binding.cpp)
- [centered_nvfp4.py](../src/Metis/Metis/centered_nvfp4.py)
- [native_nvfp4.py](../src/Metis/Metis/native_nvfp4.py)

当前 `fused_mean_and_quantize_centered_rowwise(A)` 的精确语义是：

```text
mu[h]       = BF16Round(sum_m float(A[m,h]) / M)
C[m,h]      = BF16Round(A[m,h] - mu[h])
amax        = max(abs(C[m,h]))
packed_C    = TE rowwise NVFP4 quantize(C; global amax, per-[row,16] scale)
```

它不是单 kernel，而是四个 launch：

```text
1. column_partial_stats: 读 A，写 partial_sum/min/max[ceil(M/256),H]
2. finalize_columns:     partial -> BF16 mu、每列 centered extrema、block_amax
3. reduce_block_amax:    block_amax -> global amax[1]
4. centered_pack:        再读 A 和 mu/amax，写 FP4 data 与 E4M3 scales
```

`min/max` 用于从每列 extrema 精确推出 `max(abs(A-mu))`，避免第三次完整扫描。
pack kernel 特意匹配 TE 的 BF16 subtraction、E4M3 rounding 和 E2M1x4 packing 顺序；
新实现必须保留其 packed storage 的 bitwise 兼容性。

## 融合后的数学接口

令 `X[M,H]` 是 norm 输入，`g[H]` 是 RMSNorm weight，`eps` 是模型配置。先定义与所用
Qwen/Transformers RMSNorm reference 完全一致的 BF16 输出 `Y`：

```text
r[m]   = rsqrt(sum_h float(X[m,h])^2 / H + eps)
Y[m,h] = RMSNormReferenceBF16(X[m,h], r[m], g[h])

mu[h]  = BF16Round(sum_m float(Y[m,h]) / M)
C[m,h] = BF16Round(Y[m,h] - mu[h])
```

输出仍为：

```text
PackedMeanActivation(mean=mu, quantized_residual=NVFP4(C))
```

因此 native linear 的 mean-bias 数学不变：

```text
Y @ W^T = dequant(NVFP4(C)) @ dequant(NVFP4(W))^T + mu @ dequant(W)^T
```

这里必须先锁定 reference 的 rounding：不同 Qwen/RMSNorm 实现可能在 gamma 的 dtype、
gamma 前后何时转 BF16、以及 FP32 accumulation 上不同。验收目标是“reference RMSNorm
后接当前 centered quantizer”的 packed storage bitwise 对齐，而非仅 float 接近。

## 为什么不能做成普通 single-launch kernel

存在两级全局依赖：每个 `r[m]` 依赖一整行 H，随后 `mu[h]` 和 global `amax` 又依赖全部
M 行的 Y。普通 CTA 无法在一次 launch 内等待全 grid 完成 column reduction 后安全 pack。

cooperative grid synchronization 理论上可做，但会限制 resident grid，对 large prefill
不是首版选择。正确目标是减少 HBM 往返和消除 Y，不是强行减少 launch 数。

## 推荐 CUDA pipeline

新 API 建议独立于 generic centered API：

```python
fused_rmsnorm_mean_and_quantize_centered_rowwise(
    x, rms_weight, eps, quantizer
) -> tuple[mean_bf16_1H, packed_centered_nvfp4]
```

继续让 TE 的 `NVFP4Quantizer.make_empty` 分配 storage；不得引入新 packed layout。

### A. `rmsnorm_column_partial_stats_kernel`

一个 CTA 处理完整 H 和一个 row group（候选 `R=64/128/256`）。对 group 内每一行：

1. cooperative/coalesced 读取 `X[m,:]`，FP32 reduce sum-of-squares 得 `inv_rms[m]`；
2. 计算 reference-compatible BF16 `Y[m,h]`；
3. 每线程拥有若干 h，在寄存器跨 row group 累加 `sum[h]`、`min[h]`、`max[h]`；
4. 写出 `partial_sum/min/max[group,h]`，以及 FP32 `inv_rms[M]`。

Qwen 0.5B/1.5B/7B 的 norm hidden H 为 896/1536/3584。256 或 512 threads 时每线程
大约持有 2--14 个 h 的三类 stats，适合先以寄存器实现。row-group、thread count 与
gamma 是否 cache 到 shared 必须用 benchmark/NCU 决定。

### B/C. 复用当前 finalize 和 global-amax reduction

首先复用 `finalize_columns_kernel` / `reduce_block_amax_kernel` 的逻辑：

```text
partial stats -> BF16 mean[1,H]
              -> centered column extrema -> global amax[1]
```

不可将 centered amax 偷换成 `max(abs(Y))`；实际 quantized operand 是 `Y-mean`。

### D. `rmsnorm_centered_pack_kernel`

每 CTA 处理一行或小 row group：

1. 读取完整 `X[m,:]`，并读取 Phase A 写出的 FP32 `inv_rms[m]`；
2. 用相同的 `inv_rms[m]` 计算 reference-compatible BF16 `Y`；
3. 计算 BF16 `Y-mean`、每 `[row,16]` block amax；
4. 按当前 `centered_pack_kernel` 的 E4M3 scale 与 `fp4e2m1x4` 顺序写 TE storage。

`inv_rms[M]` 应显式 materialize 为 FP32 向量。对目标 `M=131072`，它为 512 KiB；
Phase A 的一次写和 Phase D 的一次读合计约 1 MiB，远小于一遍 BF16 `X[M,H]`。同时它
避免 Phase D 再次执行整行平方、reduce 和 CTA 同步。整个 pipeline 仍只读 raw X 两遍：
一次统计、一次 pack。

## 预期节省与边界

分离路径至少有：

```text
RMSNorm: X read + Y[M,H] global write
centered stats: Y[M,H] global read
centered pack: Y[M,H] global read + packed output write
```

推荐 fused 路径为：

```text
stats: X read + partial-stat writes
pack:  X read + packed output write
```

它消除 Y 的完整写出和两次完整读取，代价是第二次读 X、partial stats，以及一份很小的
`inv_rms[M]` 写读。partial-stat traffic、M/H 和 baseline RMSNorm kernel 的效率会影响
收益；必须测量，不能仅从字节数推导端到端加速比。

首版不包含：gamma 静态折进权重、cooperative-grid 单 launch、RoPE/QKV GEMM 同 CTA，
或改变 TE layout。尤其 gamma 折权重会影响 SVD residual/low-rank decomposition 与
量化误差，必须作为独立实验。

## Block-level 图改写是必要条件

只替换 `QwenRMSNorm.forward()`、但仍返回普通 BF16 Y，无法消除 materialization。
需要在 decoder layer 显式把 packed activation 传给 consumers：

```text
AttentionNormQKVGroup:
  packed = fused_rmsnorm_mean_and_quantize(x, input_layernorm.weight, eps)
  q, k, v = native_q/k/v.forward_from_packed_activation(packed)

MlpNormGateUpGroup:
  packed = fused_rmsnorm_mean_and_quantize(residual_hidden,
                                            post_attention_layernorm.weight, eps)
  gate, up = native_gate/up.forward_from_packed_activation(packed)
```

需要给 `NativeFullNVFP4Linear` 提取 inference-only
`forward_from_packed_activation(PackedMeanActivation)`，使其跳过
`activation_group.get(input_tensor)`。Qwen 的 Q/K/V 位于 `self_attn` 内部，因此应包裹
attention/MLP 这两个小调用图；不要让 RMSNorm 返回“伪 tensor”，那会破坏普通 PyTorch
图语义且难以保证只 pack 一次。

首版范围：inference、rowwise-only activation、静态 Qwen decoder layer。训练、backward
columnwise pack、任意模型 FX rewrite 不在范围内。

## 验收

### 正确性

以实际 Qwen reference 锁定：

```python
y_ref = qwen_rmsnorm(x)
mean_ref, packed_ref = fused_mean_and_quantize_centered_rowwise(y_ref, q)
mean, packed = fused_rmsnorm_mean_and_quantize_centered_rowwise(x, gamma, eps, q)
```

必须覆盖 M=512/131072、H=896/1536/3584、非均匀 gamma、极端 input、实际 eps，并要求：

1. `mean` bitwise 对齐；
2. `rowwise_data`、`rowwise_scale_inv`、`amax_rowwise` bitwise 对齐；
3. 同一 native GEMM 的输出 bitwise 对齐；
4. Q/K/V、Gate/Up 各只发生一次 pack；
5. small-M 或不支持形状可安全 fallback 到当前路径。

### 性能与 profile

分别报告 RMSNorm、现有 centered pack、两者顺序组合、新 pipeline、QKV group、Gate/Up
group、完整 decoder layer。Nsight Systems 应证明 fused group 没有 `Y[M,H]` 的单独
materialization，并证明消费者共享一次 pack。

NCU 重点看 Phase A/D 的：

```text
dram/lts read-write bytes
launch__registers_per_thread
launch__dynamic_smem_per_block
smsp__warps_active.avg.*
smsp__warp_issue_stalled_{long_scoreboard,barrier,lg_throttle,...}
```

核心验收是完整 block latency 改善以及不再写/读 BF16 Y；不是孤立 kernel 的 occupancy。

## 建议实施顺序

1. 从实际 Qwen/Transformers runtime 写 reference test，锁定 RMSNorm BF16 rounding；
2. 实现新 fused pack API，先不改模型图，并与现有 packed storage bitwise 对齐；
3. 用 NCU 调 row-group、CTA threads、gamma cache；
4. 提取 native linear 的 `forward_from_packed_activation`；
5. 接入 AttentionNormQKVGroup，再接入 MlpNormGateUpGroup；
6. 以 Nsight Systems 和 full decoder layer benchmark 验收。
