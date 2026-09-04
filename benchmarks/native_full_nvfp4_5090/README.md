# Native Full-NVFP4 RTX 5090 Benchmark

该目录包含三类 benchmark：

1. 完整 GRPO optimizer step：rollout/generate、reward、GRPO policy
   forward/loss、backward 和 optimizer；
2. 不构造 optimizer 的纯 policy forward/loss/backward 最大 batch
   benchmark；
3. 早期的固定序列 train-only optimizer step，用于 kernel 与实现诊断。

完整 GRPO 的结果和分析见 [GRPO_RESULTS.md](GRPO_RESULTS.md)；早期
train-only optimizer-step 结果见 [RESULTS.md](RESULTS.md)；无 optimizer
结果见 [FORWARD_BACKWARD_RESULTS.md](FORWARD_BACKWARD_RESULTS.md)；
CUDA Graph 实现与结果见 [CUDA_GRAPH_RESULTS.md](CUDA_GRAPH_RESULTS.md)；
批准后的实验定义见 [PLAN.md](PLAN.md)。

## Native Full-NVFP4 的精确定义

本目录的原生 full 路径为：

- 对 Qwen 的 Q/K/V/O、Gate/Up/Down projection 做 rank-64 W-SVD；
- residual weight \(R=W-U\Sigma V\) 和 \(V\) 分别保持为 Transformer
  Engine packed NVFP4，并分别调用原生 FP4 GEMM；
- activation 和 backward gradient 使用 mean-residual；
- mean correction、U 和 singular value 路径使用 BF16；
- Q/K/V 以及 Gate/Up 共享 activation packing；
- optimizer step 后使 packed weight cache 失效。

`BitLinear.metis_merge_rollout_weights()` 只适用于 fake-QDQ：它把已经
QDQ 成 BF16 的 \(R_q\) 和 \(V_q\) 合成 BF16 weight 后执行 BF16
matmul。原生路径禁止使用该 shortcut；把合成后的 weight 再 pack 成
NVFP4 会额外量化 \(U\Sigma\) 路径，改变待测方法。

## 完整 GRPO benchmark

这里的 `batch` 是每个 optimizer step 的 completion sequence 数，而不是
unique prompt 数；`G=4` 时，batch 128 等于 32 个 prompt、每个生成 4
条 completion。gradient accumulation 固定为 1。

示例：在 RTX 5090 上运行 batch sweep：

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -e EXPECTED_GPU_UUID=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -e OUTPUT_ROOT=/workspace/fp4_post/outputs/grpo_native_full_nvfp4_5090/formal \
  -e BATCH_SIZES="32 64 128" \
  -e NUM_GENERATIONS=4 \
  -e MAX_COMPLETION_LENGTH=64 \
  -e MIN_PROMPT_TOKENS=64 \
  -e MAX_PROMPT_TOKENS=128 \
  -e WARMUP_STEPS=1 \
  -e MEASURED_STEPS=3 \
  -w /workspace/fp4_post \
  sm-container \
  bash scripts/benchmark/run_grpo_native_full_nvfp4_5090.sh
```

runner 会准备本地 DeepMath JSONL、交替执行 BF16/NVFP4、记录每步原始
数据，并生成 paired/aggregate JSON、CSV 和 Markdown 汇总。完整依赖版本
见 `requirements/benchmark-grpo-5090.txt`；CUDA、PyTorch 与 Transformer
Engine 由现有 SM120a 容器提供。

## 无 optimizer 的最大 batch forward/backward

该 runner 完全不构造 optimizer，只测 `forward + loss + backward`。默认
使用已完成容量搜索的边界：

```text
0.5B: paired/native batch 22, BF16 max batch 28
1.5B: paired/native batch 17, BF16 max batch 25
```

复现正式实验：

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -e EXPECTED_GPU_UUID=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -e OUTPUT_ROOT=/workspace/fp4_post/outputs/forward_backward_no_optimizer_5090/formal \
  -e WARMUP_STEPS=5 \
  -e MEASURED_STEPS=20 \
  -e REPEATS=3 \
  -w /workspace/fp4_post \
  sm-container \
  bash scripts/benchmark/run_forward_backward_no_optimizer_5090.sh
```

### CUDA Graph

同一 runner 支持固定 shape 的 forward/loss/backward CUDA Graph。
`CUDA_GRAPH=true` 时默认使用已经确认的 graph 容量边界：

```text
0.5B: paired/native batch 19, BF16 max batch 28
1.5B: paired/native batch 13, BF16 max batch 25
```

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

Graph 模式复用固定地址输入，并在计时窗口外更新静态 4D
causal+padding mask 和输入内容。结果与显存限制见
[CUDA_GRAPH_RESULTS.md](CUDA_GRAPH_RESULTS.md)。

## Train-only benchmark

脚本在现有 `sm-container` 内运行，仓库挂载为 `/workspace/fp4_post`。
不要用 CUDA ordinal 选择本机物理 GPU 3；当前容器的
`FASTEST_FIRST` 顺序与 `nvidia-smi` index 不一致。使用 UUID：

```bash
GPU_UUID=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959
docker exec \
  -e CUDA_VISIBLE_DEVICES="${GPU_UUID}" \
  -e EXPECTED_GPU_UUID="${GPU_UUID}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e OUTPUT_ROOT=/workspace/fp4_post/outputs/native_full_nvfp4_5090/formal_clean \
  -w /workspace/fp4_post \
  sm-container \
  bash scripts/benchmark/run_native_full_nvfp4_5090.sh
```

runner 会打印实际 CUDA device UUID；设置 `EXPECTED_GPU_UUID` 后，UUID
不一致会立即退出。

默认正式配置：

```text
batch=1, sequence=512, gradient checkpointing
10 warmup steps, 30 measured steps, 3 independent repeats
fused AdamW, lr=1e-5, SDPA, rank=64
```

BF16/NVFP4 顺序在三个 repeat 中交替。

## 单元测试

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -w /workspace/fp4_post \
  sm-container \
  bash -lc 'PYTHONPATH=src pytest -q tests/test_native_nvfp4.py'
```

## GEMM microbenchmark

```bash
docker exec \
  -e CUDA_VISIBLE_DEVICES=GPU-8b2ddb91-bae8-7aa0-8d16-91e4fa8ed959 \
  -w /workspace/fp4_post \
  sm-container \
  python benchmarks/native_full_nvfp4_5090/benchmark_gemm.py \
    --warmup 10 \
    --iterations 50 \
    --output outputs/native_full_nvfp4_5090/gemm_microbench.json
```

## 汇总

```bash
docker exec -w /workspace/fp4_post sm-container bash -lc \
  'python benchmarks/native_full_nvfp4_5090/summarize_results.py \
    outputs/native_full_nvfp4_5090/formal_clean/results/*.json \
    --markdown-output outputs/native_full_nvfp4_5090/formal_clean/summary.md \
    --csv-output outputs/native_full_nvfp4_5090/formal_clean/summary.csv \
    --json-output outputs/native_full_nvfp4_5090/formal_clean/summary.json'
```

Profiler 的稳健分组估计：

```bash
docker exec -w /workspace/fp4_post sm-container \
  python benchmarks/native_full_nvfp4_5090/summarize_profile.py \
    --nvtx-csv outputs/native_full_nvfp4_5090/formal_clean/profile/qwen15_native_one_step_nvtx_gpu_proj_sum.csv \
    --formal-summary outputs/native_full_nvfp4_5090/formal_clean/summary.json \
    --model qwen2.5-math-1.5b \
    --output outputs/native_full_nvfp4_5090/formal_clean/profile/qwen15_profile_breakdown.json
```

## Rounding 限制

当前 TE build 的 SM120 stochastic FP4 conversion kernel 不可执行；正式
结果使用确定性 gradient rounding。不要直接加
`--stochastic-gradient-quantization`，除非先确认 TE 已以 SM120a
architecture-specific target 重新编译。主 FP4 GEMM 与 rounding 选择无关，
仍使用 SM120 block-scaled E2M1 Tensor Core kernel。
