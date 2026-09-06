# Dual-FP4 fused kernel：把 activation mean correction 融入 epilogue

## 结论

现有 dual-FP4 kernel 还不具备严格的 single-D-store 语义。它只计算了中心化
activation 的 residual 项：

```text
Y_partial = X_centered @ R^T + Z @ U^T
```

而 `native_nvfp4` 的 activation packing 定义是：

```text
X = X_centered + mean                 # mean shape: [1, K]
b_residual = mean @ R^T               # shape: [N]
Y = X_centered @ R^T + Z @ U^T + b_residual
```

当前 [`native_nvfp4.py`](../src/Metis/Metis/native_nvfp4.py) 为保证数值正确，调用
dual-FP4 kernel 后执行 `output.add_(residual_mean_correction)`。这会对完整的
`M x N` BF16 output 发起 global-memory read-modify-write，因而不再是目标中的
single final D store。

本任务的目标是：保留一次小向量计算 `b_residual`，但将它作为 bias 在 dual-FP4
kernel 的 epilogue 中加到 accumulator，随后只进行一次最终 BF16 D store。

## 不要做什么

不要把 `mean @ R^T` 重算到每个 `[128,128]` CTA 内。它是 `1 x K @ K x N`，在
每个 M tile 重复计算会重复读取 R 并将工作量放大约 `M / 128` 倍。

正确的划分是：

```text
一次小 BF16 GEMM:  mean[1,K] @ dequant(R)[K,N] -> b_residual[N]
一次 dual-FP4 CTA kernel:
  accumulator = alpha_R * Acc_R + alpha_L * Acc_L
  D[m,n] = bf16(accumulator[m,n] + b_residual[n])   # 唯一的 MxN store
```

读取 `b_residual[N]` 不会破坏 single-store：它只有约 9.5 KiB（目标 N=4864、BF16），
可由 CTA epilogue 读取/缓存；必须避免的是对 `D[M,N]` 的第二次读取和写回。

## 当前实现位置

| 层 | 文件 | 当前状态 | 所需变更 |
|---|---|---|---|
| Python fused API | `src/Metis/Metis/fused_residual_lowrank_nvfp4.py` | 只传四个 packed operands 和 output | 增加 `residual_mean_correction` 参数并传给 extension |
| PyBind 检查 | `csrc/fused_residual_lowrank_nvfp4/binding.cpp` | extension 参数没有 bias | 增加 BF16 CUDA contiguous vector `[N]`，校验 device/shape/不与 output alias |
| C++ launcher | `csrc/fused_residual_lowrank_nvfp4/fused_residual_lowrank_nvfp4.cu` | `args.epilogue` 是 alpha/beta=1/0、C=null | 选择带 **per-column bias** 的 SM120 epilogue fusion，并把 bias pointer/stride 填入 arguments |
| custom kernel | `csrc/fused_residual_lowrank_nvfp4/dual_fp4_kernel.hpp` | 将两组 FP4 accumulators 合并后调用 `collective_epilogue.store()` | 保持 MMA/pipeline 不变；让已有 epilogue store 消费 bias fusion arguments |
| native dispatch | `src/Metis/Metis/native_nvfp4.py` | kernel 后 `output.add_(residual_mean_correction)` | 传入 bias；删除这个 MxN `add_` |

## API 建议

保持新 API 显式，避免 silent omission：

```python
fused_residual_lowrank_nvfp4(
    packed_x, packed_residual, packed_z, packed_u,
    residual_mean_correction,  # BF16 contiguous CUDA [N]
    output,                    # BF16 contiguous CUDA [M,N]
)
```

`residual_mean_correction` 由 native 调用侧已有的计算生成：

```python
residual_dequantized = packed_residual_weight.dequantized()
residual_mean_correction = F.linear(
    packed_activation.mean, residual_dequantized
).squeeze(0).contiguous()
packed_residual_weight.release_dequantized()
```

不要把该向量误接为 `C`：`C` 会触发 beta*C 的 MxN load；这里需要的是 epilogue
bias leaf，且 `beta=0`、`C=nullptr` 必须保持。

## Epilogue 语义与 CUTLASS 方向

dual-FP4 的 raw MMA accumulator 要先按已有 `alphas` 逻辑缩放。最终 epilogue 的
数学顺序应为：

```text
acc = base_alpha * (residual_ratio * Acc_R + lowrank_ratio * Acc_L)
D[m,n] = convert_bf16(acc + float(b_residual[n]))
```

也就是说 bias 加在 FP32 accumulator 域，再转换为 BF16；不能先把 partial output
转成 BF16、另起 kernel 加 bias。

现有 `CollectiveBuilder` 默认的 linear-combination epilogue 没有 bias leaf。应改为
SM120 TMA warp-specialized、**per-column** bias 的 fusion operation（C 的逻辑 layout
是 row-major `[M,N]`，`b_residual[n]` 沿 M 广播）。CUTLASS 可参考：

```text
.deps/cutlass/include/cutlass/epilogue/fusion/operations.hpp
  fusion::LinCombPerColBias

.deps/cutlass/include/cutlass/epilogue/fusion/sm120_callbacks_tma_warpspecialized.hpp
  SM120 per-column bias callbacks and their Arguments::bias_ptr / dBias
```

具体 builder/collective 类型要依当前 CUTLASS checkout 能编译的 SM120 specialization
确定；不要机械使用 `PerRowBias`，那会沿 M 索引，和 `[N]` 的 output-channel bias
不符。完成后，`args.epilogue` 应表达：`alpha * accumulator + 0 * C + bias`，且
`bias_ptr` 指向 `residual_mean_correction`、其 stride 随 N 变化。

如果现有 `CollectiveBuilder` 不能直接实例化该 fusion operation，允许替换为对应的
project-local collective/epilogue typedef；不需要也不应改动 vendored `.deps/cutlass`。

## Native dispatch 的精确改动

在 `lowrank_compute == "nvfp4"` 的 compatible fused branch：

```python
fused_residual_lowrank_nvfp4(
    packed_activation.quantized_residual,
    packed_residual_weight.quantized,
    packed_scaled_v,
    packed_u_weight.quantized,
    residual_mean_correction.squeeze(0).contiguous(),
    output,
)
output = output[: packed_activation.rows]
# 删除：output.add_(residual_mean_correction)
```

现有 Z padding 逻辑必须保留：fused kernel 用 `packed_activation.packed_rows` 的 M，
而 `_native_mean_fprop` 返回的是逻辑 M 行。因此 `scaled_v_output` 在量化为 Z 前需补
零到 `packed_rows`，最终 output 再裁回 `rows`。

不满足 SM120/rank-64/对齐条件的分支仍走原 TE `beta=1` 路径，不能引用这个新 bias API。

## 功能与 single-store 验收

1. 更新 direct kernel test：构造 BF16 `bias[N]`，参考值为两次 TE NVFP4 GEMM 的
   BF16 output 加 `bias`；覆盖非零、全零、不同 scale/amax 与多 CTA tile。
2. 更新 full native validation：`lowrank_compute="nvfp4"` 与 BF16 low-rank reference
   比较，特别覆盖 `packed_rows != rows` 的 M（例如逻辑 M 不是 128 的倍数）。
3. 在 native fused 分支移除 `output.add_` 后，用 Nsight Systems/NVTX 确认不再出现
   对最终 `[M,N]` output 的单独 BF16 add kernel。
4. 用 NCU 记录 fused kernel 的 global write；不应存在第二个 MxN output read/write。
   `b_residual[N]` 的小 vector read 是预期行为。
5. 目标 shape `M=131072,N=4864,K=896,r=64` 做 CUDA-event median；报告完整 native
   forward，而非只报 dual-FP4 kernel 时间。

## 完成定义

只有在以下两点都成立时，dual-FP4 才可称为 strict single-D-store：

```text
1. 最终 D/Y [M,N] 仅由 dual-FP4 kernel 写一次；没有 kernel 后 output.add_。
2. D 的每个元素包含 residual mean correction、residual FP4 项和 low-rank FP4 项。
```
