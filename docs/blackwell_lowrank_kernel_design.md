# Blackwell low-rank residual epilogue design

## Scope and measured bottleneck

The inference operation is

```text
Y[M,N] = GEMM_NVFP4(X[M,K], R[N,K]) + GEMM(Z[M,64], U[N,64])
Z = GEMM_NVFP4(X, V)·diag(s)
```

`U` and `Z` must remain independently quantized when the second product uses
NVFP4; `s` must not be folded into `U`.

For Qwen 0.5B up projection at `M=131072, N=4864`, an isolated BF16
`Z @ U.T` has only 82.2 GFLOP, but reads and writes the BF16 residual matrix:

| traffic | size |
| --- | ---: |
| read residual `Y` | 1.19 GiB |
| write updated `Y` | 1.19 GiB |
| avoidable intermediate write from residual GEMM | 1.19 GiB |
| avoidable intermediate reread by low-rank GEMM | 1.19 GiB |

The standalone `addmm_` already has the irreducible read+write pair.  A
standalone custom rank-64 kernel therefore has little room to beat cuBLAS:
the real win is a *dual-mainloop* kernel that writes `Y` once after both GEMMs.

## Current isolated-kernel result (RTX 5090)

The experimental BF16 kernel is correct, but intentionally remains outside
the inference path because it does not pass the standalone acceptance gate.
It now stages the shared `[64,128]` U tile once per CTA rather than loading it
once per M-warp:

| M | N | handwritten BF16 | `torch.addmm_` | result |
| ---: | ---: | ---: | ---: | --- |
| 8,192 | 4,864 | 0.176 ms | 0.088 ms | 2.00x slower |
| 131,072 | 4,864 | 2.797 ms | 1.778 ms | 1.57x slower |

The same kernel before U-tile staging took 3.31 ms at the large shape, so the
change is a real 18% improvement, but insufficient.  This is evidence against
spending further time on a standalone legacy-BF16 WMMA implementation.

The official CUTLASS SM120 NVFP4-BF16 kernel passes its reference check for
rank 64.  The project-local adaptation now consumes TE's actual packed data,
GEMM-swizzled scales, and global-amax alpha correction.  Its only valid
baseline is TE/cuBLASLt NVFP4 GEMM with the same packed operands and `beta=1`:

| M | N | K | project SM120 NVFP4 | TE NVFP4 |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 4,864 | 64 | 0.086 ms | 0.082 ms |
| 131,072 | 4,864 | 64 | **1.682 ms** | 1.684 ms |

At the full prefill shape, custom is equal to (and this run is 0.12% faster
than) TE; this difference is within ordinary run-to-run noise. It numerically
matches TE (`max_abs_error=0.125`, mean absolute error below `1e-6` for a mean
output magnitude around `6.39`). The smaller problem remains about 5% slower.
Reproduce
the same-precision result with
`benchmarks/inference_nvfp4_5090/benchmark_lowrank_nvfp4_sm120.py`; build the
extension first with `python csrc/lowrank_nvfp4_sm120/setup.py build_ext --inplace`.

The project wrapper computes TE's global-amax correction on the CUDA stream
and passes it through CUTLASS's pointer-array epilogue, applying
`amax(A)*amax(B)/(6*6*448*448)` without a host round trip.  This is necessary
because `scaled_v` is newly quantized each forward and its amax is only ready
on device. The same pointer-array epilogue is the correct place to add the
dual-mainloop production kernel's final scale and bias.

## B200 versus RTX 5090

| property | B200 / SM100 | RTX 5090 / SM120 |
| --- | --- | --- |
| current Tensor Core programming path | `tcgen05` + tensor memory (TMEM) | block-scaled `mma.sync.aligned` for FP4/FP8; legacy `mma.sync` for BF16 |
| CTA resources | 64 warps/SM, 228 KiB shared memory/SM | 48 warps/SM, 128 KiB shared memory/SM |
| specialized data movement | TMA, including multicast; 1- or 2-CTA groups | no TMA multicast, no dynamic datatypes; cluster must be 1x1x1 |
| recommended low-rank representation | FP4 inputs, TMEM accumulators, single epilogue | FP4 inputs, SM120 block-scaled MMA, single epilogue |

The two architectures are not binary compatible at the kernel level:
SM100 `tcgen05`/TMEM kernels cannot run on SM120.  In particular, using the
old WMMA C++ API on the 5090 does generate BF16 tensor-core instructions, but
does not provide the modern Blackwell persistent/TMA scheduling pipeline.

## 5090 kernel design

### Phase 1: preserve the best standalone baseline

Use `torch.addmm_`/cuBLASLt for the BF16 standalone operation.  It is the
acceptance baseline, not a temporary implementation to replace blindly.  The
handwritten WMMA experiment is retained only to validate correctness and show
why a rank-64, read-modify-write kernel is difficult to outperform.

For the actual NVFP4 algorithm, retain separate quantization of `Z` and `U`
and use the SM120 NVFP4 path.  Transformer Engine 2.15 dispatches this path
through cuBLASLt; the public CUTLASS SM120 FP4 example is the useful device
code reference for a project-local fused implementation.  It is the only
public SM120 CUTLASS path with the Blackwell warp-specialized FP4 MMA and
optimized epilogue.

### Phase 2: one CTA computes both products for one output tile

Build an SM120 CUTLASS/CuTe kernel with a `[128,128]` output tile and two
sequential FP4 mainloops:

1. load `X` and packed `R`, accumulate `X @ R.T` in registers;
2. load packed `Z` and packed `U`, add `Z @ U.T` into the same accumulator;
3. convert once to BF16 and write `Y` once.

Both mainloops use their own packed-scale layouts.  The epilogue owns the only
global `Y` store, so no atomics and no inter-kernel dependency are needed.  A
separate first GEMM still materializes `Z[M,64]`; it is only 16 MiB at the
largest benchmark and must not be recomputed for every `N` tile.

The first implementation should use a 1x1x1 cluster, cooperative schedule,
and CUTLASS's SM120 FP4 layout helpers.  It must not port B200's `tcgen05` or
TMEM code to the 5090.  Once functionally correct, compare cooperative and
ping-pong schedules, retaining the faster configuration per `(M,N,K)` family.

## Reading the SM120 reference implementation

The runnable reference is CUTLASS example
`79a_blackwell_geforce_nvfp4_bf16_gemm`.  These are the parts worth changing
and measuring, in this order:

| code-level choice | reference value | purpose |
| --- | --- | --- |
| architecture tag | `cutlass::arch::Sm120` | emits the GeForce Blackwell instruction path |
| operator class | `OpClassBlockScaledTensorOp` | selects E2M1 FP4 inputs with E4M3 block scales |
| CTA tile | `[128,128,128]` | enough work per persistent CTA to cover TMA and epilogue cost |
| cluster | `[1,1,1]` | RTX 5090 has no TMA multicast, so larger clusters do not create reuse |
| mainloop schedule | `KernelScheduleAuto` | currently resolves to the SM120 cooperative, warp-specialized schedule |
| epilogue | BF16 C/D, `alpha=beta=1` | performs `D = A@B + C` in the only output store |

Although the CTA K tile is 128, the generated mainloop supports a logical
`K=64`; the measured rank-64 run above proves this.  The disassembly identifies
the relevant instruction as `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`: a
block-scaled 16x8x64 FP4 operation.  The `SF` operands are scale-factor tensor
addresses, not ordinary scalar multipliers.  This is why a project kernel must
use CUTLASS's `LayoutSFA/LayoutSFB` helpers rather than treating TE's scale
buffer as a flat `[M,K/16]` tensor.

For the project algorithm, construct two independent such operand groups:

```text
mainloop 0: packed X, packed R, scale-X, scale-R
mainloop 1: packed Z, packed U, scale-Z, scale-U
shared epilogue: BF16 D = accumulator0 + accumulator1 + bias
```

This preserves the required independent quantization of `Z` and `U`.  It also
means that the two FP4 mainloops cannot be replaced by a single GEMM with a
modified weight matrix.

### Phase 3: acceptance gates

1. The isolated BF16 candidate must be within 5% of `addmm_`; otherwise it is
   not integrated.
2. The dual-FP4 kernel must match the existing two-TE-GEMM NVFP4 output within
   the normal NVFP4 tolerance and preserve the independently quantized `U` and
   `Z` invariant.
3. Measure end-to-end linear latency, not just kernel time.  The success
   criterion is a lower total forward latency than the current TE residual
   GEMM + TE low-rank GEMM sequence.

## B200 follow-up

The mathematical layout remains the same, but the B200 version should be a
separate SM100 implementation: `tcgen05` MMA, TMEM accumulator allocation,
TMA pipelines, and a 1-CTA then 2-CTA-group autotuned variant.  Share only
host-side dispatch and tests with SM120; do not share device code.
