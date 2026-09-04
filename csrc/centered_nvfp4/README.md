# Centered NVFP4 activation pack

This extension is the project-local implementation of the inference-only
operation

```text
mean = x.mean(dim=0)
packed = NVFP4Quantize(BF16Round(x - mean))
```

It targets exactly the Tensor Engine source revision used by the benchmark
container: `v2.15` (`42b840051647eef89761a16dfdff87e82bb253ab`). It is not a
replacement Transformer Engine distribution. Only the deterministic BF16,
rowwise-only NVFP4 path is vendored, and its raw output is written into an
`NVFP4Quantizer.make_empty()` allocation so it can be consumed by TE
`general_gemm` unchanged.

Supported first-version contract:

- CUDA Blackwell SM120a;
- contiguous BF16 `x[M, K]`, `M` and `K` divisible by 32;
- BF16 `mean[1, K]` produced by `torch.mean`;
- rowwise-only E2M1 / E4M3 NVFP4, deterministic rounding;
- no RHT, 2D quantization, stochastic rounding, backward, or columnwise
  storage.

The kernel must reproduce the current TE reference bitwise:

```python
reference = quantizer.quantize((x - x.mean(0, keepdim=True)).contiguous())
```

The test suite compares FP4 bytes, E4M3 scale bytes, global amax,
dequantization and native FP4 GEMM output before this backend can be enabled.

Build it into the source package directory used by the benchmarks:

```bash
PYTHONPATH=src python csrc/centered_nvfp4/setup.py build_ext --inplace
PYTHONPATH=src pytest -q tests/test_centered_nvfp4.py
```
