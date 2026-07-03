# Metis Quantization Runtime Optimization

## Question

Can GRPO reuse a simulated FP4 weight between optimizer updates, and can the
activation mean path avoid redundant quantization work without changing its
output?

## Computation Contract

For a parameter tensor `W`, the cached value is
`W_q = rquant(quant(W, get_scalar(W)))`. The cache key contains the parameter
version, device, dtype, quantizer class, and NV-recipe setting. Repeated forwards
reuse `W_q` only while this key is unchanged. An in-place optimizer update
increments `Parameter._version`, so the next forward recomputes `W_q`.

The cache stores the dequantized tensor used by the existing matrix multiply.
It does not change the manually implemented straight-through gradient: backward
still returns the weight gradient for the original parameter input.

For mean activation quantization, the output remains
`mean(x) + rquant(quant(x - mean(x), scalar), scalar)`. The old code computed a
second scalar from this output even though backward never read it. That second
block reduction has been removed.

## Expected Cost

The weight cache removes weight `get_scalar`, `quant`, and `rquant` calls from
all forwards after the first forward at a given parameter version. This is most
valuable during autoregressive rollout, where the same weights are invoked once
per generated token.

The cost is one persistent dequantized weight-sized tensor for every
`LinearLowbit` module. The current W-SVD rank-64 path uses `LinearLowbit` for the
V factor and residual weight, so the residual dominates the extra memory. Keep
the cache disabled when this extra BF16 memory does not fit.

## torch.compile

`METIS_COMPILE_QDQ=true` compiles only the stateless fused NVFP4 QDQ function.
It does not compile the rolling-mean dictionary updates, custom autograd
function, or the whole model. The fused function computes block scales, FP8
scale simulation, FP4 rounding, and direct dequantization without materializing
a separate quantized tensor between `quant` and `rquant`.

The compiled path uses dynamic shapes so prefill and decode can share the same
callable, but Inductor may still specialize graphs. Measure steady-state latency
after compilation warmup and inspect recompilation logs before enabling it for
long runs.

## Verification

Run on a server environment that has PyTorch and pytest:

```bash
PYTHONPATH=src pytest -q tests/test_metis_quant_cache.py
```

The tests require exact equality between the combined quantization helper and
the old three-call sequence. They also require one cache hit before a parameter
update, one cache miss after the update, and a valid weight gradient.

For an end-to-end timing comparison, keep all GRPO settings fixed and compare:

```bash
GRPO_MAX_STEPS=2 METIS_CACHE_QUANTIZED_WEIGHT=false \
  bash scripts/grpo/run_experiment.sh grpo full qwen2_5_math_1_5b_base

GRPO_MAX_STEPS=2 METIS_CACHE_QUANTIZED_WEIGHT=true \
  METIS_COMPILE_QDQ=true \
  bash scripts/grpo/run_experiment.sh grpo full qwen2_5_math_1_5b_base
```

Record peak GPU memory, rollout duration, and total step time. Two steps are
needed because the first step builds the cache and the second measures reuse
after one optimizer update.

## Claim Boundary

Static analysis supports automatic invalidation for normal PyTorch and
DeepSpeed ZeRO-2 in-place parameter updates. Numerical equivalence and measured
speedup still require the server-side tests above because the local environment
does not contain PyTorch or CUDA.
