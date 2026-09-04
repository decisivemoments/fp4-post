# RTX 5090 Forward/Backward CUDA Graph 结果

日期：2026-07-27

## 结论

CUDA Graph 可以同时用于 BF16 和当前的 Native Full-NVFP4
forward/loss/backward：

- Qwen2.5-0.5B-Instruct 在最大共同 batch 19 时，NVFP4 仍比 BF16
  慢 17.4%；
- Qwen2.5-Math-1.5B 在最大共同 batch 13 时，NVFP4 的 step time
  比 BF16 低 2.96%，等价于 **1.031x** step throughput；
- 按各自最大稳定 batch 比较，1.5B 的 Native Full-NVFP4 峰值吞吐
  比 BF16 高 **5.55%**，0.5B 则低 14.8%。

CUDA Graph 对包含大量 FP4 packing、mean correction 和两路 GEMM 的
Native 路径帮助明显大于对 BF16 的帮助。不过，graph private pool
也降低了 Native 路径能容纳的最大 batch。

## 捕获范围

计时范围仍然只有：

```text
model forward + causal-LM loss + loss.backward()
```

没有构造 optimizer，不存在 optimizer state，也不执行
`optimizer.step()`。输入更新和 `model.zero_grad(set_to_none=True)` 均在
计时窗口外。

实现使用 `torch.cuda.make_graphed_callables` 分别捕获 autograd forward
和 backward graph。每个 graph 固定 batch/sequence shape，并复用固定
地址的 `input_ids`、`attention_mask` 和 `labels`。

Transformers 的动态 SDPA mask 优化会在捕获期间读取
`padding_mask.all()`，造成非法 host sync。benchmark 因此在 graph 外
构造固定地址的 4D boolean mask：

```text
causal lower-triangle AND key padding mask
```

每步只在计时窗口外更新 padding 部分。该 4D mask 与原 2D mask 生成的
causal/padding 语义一致。

## CUDA Graph 下 BF16 与 NVFP4

数值为 3 个独立进程的 step median 的中位数；每个进程使用
5 warmup + 20 measured steps。

| 模型 | Batch | BF16 forward | NVFP4 forward | BF16 backward | NVFP4 backward | BF16 total | NVFP4 total | BF16/NVFP4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 19 | 85.641 ms | 94.603 ms | 205.852 ms | 247.654 ms | 291.530 ms | 342.286 ms | 0.852x |
| Qwen2.5-Math-1.5B | 13 | 138.054 ms | 126.219 ms | 364.468 ms | 361.501 ms | 502.584 ms | 487.696 ms | **1.031x** |

1.5B 的收益分解为：

- forward：BF16/NVFP4 = 1.094x；
- backward：BF16/NVFP4 = 1.008x；
- forward+backward：BF16/NVFP4 = 1.031x。

## CUDA Graph 相对 eager 的收益

为避免 batch size 混淆，另运行了完全相同 batch、数据、seed 和
warmup/measured 配置的 eager 对照。

| 模型 | 模式 | Batch | Eager total | CUDA Graph total | Eager/Graph | Step time 降低 |
|---|---|---:|---:|---:|---:|---:|
| 0.5B | BF16 | 19 | 295.766 ms | 291.530 ms | 1.015x | 1.43% |
| 0.5B | Native Full-NVFP4 | 19 | 459.353 ms | 342.286 ms | **1.342x** | **25.49%** |
| 1.5B | BF16 | 13 | 505.480 ms | 502.584 ms | 1.006x | 0.57% |
| 1.5B | Native Full-NVFP4 | 13 | 526.655 ms | 487.696 ms | **1.080x** | **7.40%** |

BF16 的大 GEMM 已能较好占满 GPU，CPU launch overhead 占比很低。
Native Full-NVFP4 每个 projection 包含 mean、packing、residual/V FP4
GEMM、BF16 low-rank 和 correction 等更多操作，因此 graph replay
消除 host launch overhead 的收益更明显。

## 最大稳定 batch 与峰值吞吐

容量搜索使用独立进程；成功边界和相邻失败点如下：

| 模型 | 模式 | Eager 最大 batch | CUDA Graph 最大 batch | Graph 相邻失败 |
|---|---|---:|---:|---:|
| 0.5B | BF16 | 28 | 28 | 29 OOM |
| 0.5B | Native Full-NVFP4 | 22 | 19 | 20 OOM |
| 1.5B | BF16 | 25 | 25 | 26 OOM |
| 1.5B | Native Full-NVFP4 | 17 | 13 | 14 OOM |

按 graph 模式下各自最大 batch：

| 模型 | BF16 batch / tokens/s | NVFP4 batch / tokens/s | NVFP4/BF16 峰值吞吐 |
|---|---:|---:|---:|
| 0.5B | 28 / 33,353.9 | 19 / 28,420.7 | 0.852x |
| 1.5B | 25 / 12,930.4 | 13 / 13,647.9 | **1.055x** |

Graph replay 时 `torch.cuda.memory_allocated()` 不包含 graph private pool
中的全部内部存储，因此 graph 显存应同时查看 `memory_reserved()` 和
`nvidia-smi`，不能只看 allocated：

| 模型 / paired batch | BF16 allocated/reserved | NVFP4 allocated/reserved |
|---|---:|---:|
| 0.5B / 19 | 1.97 / 21.12 GiB | 4.41 / 26.54 GiB |
| 1.5B / 13 | 5.88 / 17.39 GiB | 10.28 / 26.15 GiB |

Native graph 私有池保留 FP4 packing/GEMM workspace，因此尽管 replay
中的 allocated 数值较低，实际容量上限反而比 eager 小。

## 正确性检查

- BF16 graph 冒烟测试：290/290 个 parameter gradient tensors 有限；
- Native graph 冒烟测试：794/794 个 parameter gradient tensors 有限；
- 12 组 matched graph/eager 运行共有 240 个 measured loss，逐项最大
  绝对差为 0；
- `tests/test_native_nvfp4.py`：7 passed。

## 复现

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -e EXPECTED_GPU_UUID=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -e OUTPUT_ROOT=/workspace/fp4_post/outputs/forward_backward_cuda_graph_5090/formal \
  -e CUDA_GRAPH=true \
  -e WARMUP_STEPS=5 \
  -e MEASURED_STEPS=20 \
  -e REPEATS=3 \
  -w /workspace/fp4_post \
  sm-container \
  bash scripts/benchmark/run_forward_backward_no_optimizer_5090.sh
```

runner 在 `CUDA_GRAPH=true` 时默认使用本次确认的 graph 容量边界。

## Artifact

```text
outputs/forward_backward_cuda_graph_5090/
  smoke/                 # BF16/native graph 有限梯度检查
  capacity/              # 成功的容量探测 JSON
  formal/
    results/             # 18 个 CUDA Graph 正式 JSON
    forward_backward_aggregate.json
    forward_backward_paired.json
    forward_backward_maxima.json
    forward_backward_summary.md
  eager_matched/
    results/             # 12 个同 batch eager 对照 JSON
    forward_backward_aggregate.json
    forward_backward_paired.json
    forward_backward_maxima.json
    forward_backward_summary.md
```
