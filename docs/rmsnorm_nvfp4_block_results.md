# RMSNorm + centered NVFP4：Qwen 0.5B 单层 block 结果

## 范围

本结果对应新的 RMSNorm-to-packed-activation 路径：`input_layernorm` 的输出直接供
Q/K/V 使用，`post_attention_layernorm` 的输出直接供 Gate/Up 使用。两处均不 materialize
normalized BF16 activation。O/Down 仍走现有 native centered pack。

测试 GPU 为 RTX 5090（UUID `GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99`），模型为本地
`/home/dy/zxt/qwen2.5-0.5b-instruct`，shape 为 batch=256、sequence=512，即
`M=131072,H=896`。每个结果 warmup=5、20 个 CUDA-event sample、报告 median。

## 正确性

新 pack 与以下 reference 的四项 storage 均 bitwise 相等：

```python
y_ref = Qwen2RMSNorm(x)
mean_ref, packed_ref = fused_mean_and_quantize_centered_rowwise(y_ref, q)
```

已在 `(M,H)=(512,896)、(512,1536)、(512,3584)` 检查 `mean`、`rowwise_data`、
`rowwise_scale_inv`、`amax_rowwise`。此外，完整 Qwen 0.5B layer 的“旧 native graph”和
“fused RMSNorm graph”在同一输入/同一转换权重下输出 `torch.equal=True`。

## 性能

| 路径 | median | 对本次 fused NVFP4 |
|---|---:|---:|
| 本次 BF16 block | 32.802 ms | 1.640x slower |
| 当前 native NVFP4，无 RMSNorm fusion | 23.843 ms | 1.192x slower |
| 新 RMSNorm-fused native NVFP4 | **20.002 ms** | 1.000x |
| 历史初始 native FP4 记录 | 23.947 ms | 1.197x slower |

因此新融合相对同次运行、无 RMSNorm fusion 的当前 native NVFP4：

```text
23.843 / 20.002 = 1.192x speedup
latency reduction = 16.1%
```

相对用户指定的历史初始结果
[`transformer_qwen0.5b.json`](../outputs/inference_nvfp4_5090/transformer_qwen0.5b.json) 的
batch=256 NVFP4 median 23.947 ms：

```text
23.947 / 20.002 = 1.197x speedup
latency reduction = 16.5%
```

本次 BF16 32.802 ms 与历史 BF16 33.204 ms 接近，说明历史比较没有明显的环境量级偏差。
结果 JSON 分别为：

- `outputs/inference_nvfp4_5090/transformer_qwen0.5b_rmsnorm_fused_b256.json`
- `outputs/inference_nvfp4_5090/transformer_qwen0.5b_dual_fp4_no_rmsnorm_fusion_b256.json`

## 复现

```bash
cd /home/dy/zxt/fp4_post
source ../conda-init.sh
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99

python benchmarks/inference_nvfp4_5090/benchmark_qwen_inference.py \
  --scope transformer --model qwen0.5b \
  --model-path /home/dy/zxt/qwen2.5-0.5b-instruct \
  --modes bf16 nvfp4 --batch-sizes 256 --seq-length 512 \
  --lowrank-compute nvfp4 --activation-layout rowwise \
  --activation-pack-backend centered_cuda --fuse-rmsnorm-quant \
  --warmup 5 --iterations 20 \
  --output outputs/inference_nvfp4_5090/transformer_qwen0.5b_rmsnorm_fused_b256.json
```

通过 sweep 脚本运行时设 `FUSE_RMSNORM_QUANT=1`；该 flag 只会传给 transformer scope，
不会影响 linear 或 Nsight 单-projection profile。
