# Qwen BF16 vs current NVFP4 step-time benchmark

This benchmark answers a narrow question: on one RTX 5090, is one complete
training step faster after replacing Qwen projection layers with the current
lowest-overhead `direct_fp4` path?

The measured 5090 results are recorded in [RESULTS_5090.md](RESULTS_5090.md).

A timed step includes:

1. causal-LM forward and loss,
2. backward,
3. fused AdamW update, and
4. gradient clearing.

Model loading, DeepMath tokenization, host-to-device input copies, warmup, and
the first optimizer-state allocation are outside the timed region. The scripts
use the same deterministic token cache for the BF16 and NVFP4 runs.

## Important scope

The current `direct_fp4` implementation simulates quantization by quantizing
and dequantizing tensors, then calls `torch.matmul` with BF16 tensors. It does
not issue native FP4 Tensor Core GEMMs. The result therefore measures the
current repository implementation, not the hardware's theoretical
native-NVFP4 throughput.

## Run inside the provided container

From `/workspace/fp4_post`:

```bash
CUDA_VISIBLE_DEVICES=3 \
  bash scripts/benchmark/run_step_time_5090.sh
```

Useful smoke-test overrides:

```bash
CUDA_VISIBLE_DEVICES=3 WARMUP_STEPS=1 MEASURED_STEPS=1 \
  MODELS=qwen2_5_0_5b \
  bash scripts/benchmark/run_step_time_5090.sh
```

The runner reads:

- `/workspace/qwen2.5-0.5b-instruct`
- `/workspace/qwen2.5-math-1.5b`
- `/workspace/deepmath-103k`

Override `WORKSPACE_ROOT`, `BATCH_SIZE`, `SEQ_LENGTH`, `WARMUP_STEPS`,
`MEASURED_STEPS`, `GRADIENT_CHECKPOINTING`, `COMPILE_QDQ`, or
`CACHE_QUANTIZED_WEIGHT` as needed.

Generated caches and results are written under
`outputs/benchmarks/step_time/`, which is ignored by git. The paired summary
reports median latency, tokens/s, peak allocated memory, and the ratio
`BF16 median / NVFP4 median`. A ratio above 1 means NVFP4 is faster.
