# RTX 5090 BF16 vs current NVFP4 step-time results

Date: 2026-07-26

## Conclusion

The current repository's lowest-overhead `direct_fp4` path does not speed up a
training step for either tested Qwen model. Even with compiled QDQ and
quantized-weight caching enabled, it is 2.44x slower than BF16 for
Qwen2.5-0.5B-Instruct and 2.54x slower for Qwen2.5-Math-1.5B.

This is expected from the implementation: `direct_fp4` applies fake
quantize-dequantize operations and then executes BF16 `torch.matmul`. It does
not dispatch a native FP4 Tensor Core GEMM. The benchmark therefore measures
the current code path, not the RTX 5090's theoretical native-FP4 throughput.

## Controlled setup

- GPU: one NVIDIA GeForce RTX 5090, compute capability 12.0, 32,607 MiB
- GPU selection: physical GPU 3 through `CUDA_VISIBLE_DEVICES=3`
- Container: `sm-controller-5090:pt26.04-te2.15-sm120a`
- PyTorch: `2.12.0a0+0291f960b6.nv26.04.48445190`
- CUDA: 13.2
- Transformers: 4.52.3
- Transformer Engine: 2.15.0
- Data: local DeepMath-103K train parquet, read directly with PyArrow
- Preprocessing: seed 2026, 64 deterministic samples per tokenizer, model chat
  template, padded/truncated to 512 tokens
- Workload: batch size 1, sequence length 512, gradient checkpointing enabled
- Optimizer: fused AdamW, learning rate `1e-5`
- Timing: 5 warmup steps followed by 20 measured steps
- Timed region: forward/loss + backward + optimizer update + gradient clearing
- NVFP4 variant: `direct_fp4` with SVD/mean-residual features disabled
- NVFP4 options: compiled QDQ requested and validated; quantized-weight cache on

Model loading, tokenization, host-to-device input copies, initial compilation,
warmup, and first optimizer-state allocation were excluded from the timed
region. BF16 and NVFP4 used the same cached batches and each run started from
the same local checkpoint.

## Results

| Model | Mode | Median step | p10-p90 | Forward median | Backward median | Optimizer median | Tokens/s | Peak allocated |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | BF16 | 93.431 ms | 88.646-94.508 ms | 26.852 ms | 60.980 ms | 5.567 ms | 5,480 | 4.425 GiB |
| Qwen2.5-0.5B-Instruct | NVFP4 | 228.048 ms | 227.272-230.033 ms | 61.013 ms | 161.647 ms | 5.243 ms | 2,245 | 5.104 GiB |
| Qwen2.5-Math-1.5B | BF16 | 111.118 ms | 110.551-118.289 ms | 30.192 ms | 66.070 ms | 14.642 ms | 4,608 | 12.585 GiB |
| Qwen2.5-Math-1.5B | NVFP4 | 281.730 ms | 279.202-307.709 ms | 73.473 ms | 191.645 ms | 15.019 ms | 1,817 | 15.032 GiB |

| Model | BF16/NVFP4 speed ratio | NVFP4 latency overhead | Decision |
|---|---:|---:|---|
| Qwen2.5-0.5B-Instruct | 0.410x | +144.08% | NVFP4 slower |
| Qwen2.5-Math-1.5B | 0.394x | +153.54% | NVFP4 slower |

The optimizer phase is essentially unchanged. Almost all extra latency comes
from forward and backward QDQ work. Peak allocated memory also increases,
rather than decreases, because the current path materializes dequantized BF16
tensors and retains quantized-weight cache tensors.

## Reproduction

Inside `sm-container`, from `/workspace/fp4_post`:

```bash
CUDA_VISIBLE_DEVICES=3 \
WARMUP_STEPS=5 \
MEASURED_STEPS=20 \
NUM_CACHE_SAMPLES=64 \
BATCH_SIZE=1 \
SEQ_LENGTH=512 \
GRADIENT_CHECKPOINTING=true \
COMPILE_QDQ=true \
CACHE_QUANTIZED_WEIGHT=true \
bash scripts/benchmark/run_step_time_5090.sh
```

Raw local JSON, CSV, and Markdown outputs are under
`outputs/benchmarks/step_time/` and are intentionally ignored by git.

## Boundary of the result

This is an isolated causal-LM optimizer-step benchmark. It does not include
GRPO rollout generation, reward evaluation, distributed communication, or
checkpoint I/O. It also does not time the `moving_mean`/`full` variants, which
add low-rank and residual work on top of the same simulated NVFP4 arithmetic.
Those operations answer different end-to-end questions. A future
implementation that passes packed FP4 tensors and scales directly to
Transformer Engine native FP4 GEMMs must be benchmarked again; the current
measurements cannot predict its speedup.
