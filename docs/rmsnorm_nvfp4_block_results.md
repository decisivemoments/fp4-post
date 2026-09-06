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

## 受控三组性能

| 路径 | dual-FP4 | RMSNorm→quant | median | 相对优化版 |
|---|---|---|---:|---:|
| 完全 BF16 block | — | — | 32.819 ms | 1.640x slower |
| legacy native NVFP4 | off | off | 28.024 ms | 1.401x slower |
| 优化 native NVFP4 | on | on | **20.005 ms** | 1.000x |

三个命令使用同一 RTX 5090、同一模型、batch=256、seq=512、rowwise centered activation、
rank=64、warmup=5、20 CUDA-event samples。legacy 组显式传
`--disable-dual-fp4-fusion`，因而强制回退到 residual FP4 GEMM 后由 TE low-rank NVFP4
`beta=1` 累加的两-GEMM 路径。

优化版相对 legacy native NVFP4：

```text
28.024 / 20.005 = 1.401x speedup
latency reduction = 28.6%
```

优化版相对完全 BF16：

```text
32.819 / 20.005 = 1.640x speedup
```

历史 `transformer_qwen0.5b.json` 的 NVFP4 23.947 ms 没有 commit/kernel-dispatch
provenance，且晚于 dual-FP4 结果生成时间，故不再用它判断 dual-FP4 的收益。
本次 BF16 32.819 ms 与历史 BF16 33.204 ms 接近，说明测试环境量级稳定。
结果 JSON 分别为：

- `outputs/inference_nvfp4_5090/transformer_qwen0.5b_rmsnorm_fused_b256.json`
- `outputs/inference_nvfp4_5090/transformer_qwen0.5b_dual_fp4_no_rmsnorm_fusion_b256.json`
- `outputs/inference_nvfp4_5090/transformer_qwen0.5b_control_bf16_b256.json`
- `outputs/inference_nvfp4_5090/transformer_qwen0.5b_control_legacy_nvfp4_b256.json`
- `outputs/inference_nvfp4_5090/transformer_qwen0.5b_control_dual_rmsnorm_b256.json`

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
