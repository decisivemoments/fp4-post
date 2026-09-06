# SM120 dual-FP4 single-D-store：实现与验收

> 历史记录：本文的五参数 API 和测量对应尚未融合 activation mean correction 的版本。
> 当前六参数 API、完整 native forward 与 strict single-store 验收见
> [mean-correction 验证文档](dual_fp4_mean_correction_validation.md)。

2026-09-06，按 [交接文档](dual_fp4_fused_handoff.md) 完成独立 kernel、PyTorch
binding、packed-object wrapper、功能测试、两档 prefill benchmark 和 NCU 验证。
完整样本、计数器、环境版本与源码 SHA256 见
[dual_fp4_fused_results.json](dual_fp4_fused_results.json)。

## 实现

入口为 `Metis.Metis.fused_residual_lowrank_nvfp4.fused_residual_lowrank_nvfp4`：

```python
fused_residual_lowrank_nvfp4(packed_x, packed_residual, packed_z, packed_u, output)
```

四个参数均为独立量化后的 TE rowwise NVFP4 packed objects；wrapper 对每个对象分别
swizzle scales。调用方先 materialize `Z = (X @ V.T) * singular_values`，再量化
Z；U 独立量化，singular values 不折入 U。输出是 contiguous BF16 `[M,N]`。
支持正的 M、N（128 的倍数）、K（64 的倍数），rank 固定 64，inference-only，
不包含 bias 或 mean correction。保留已有模型级两-GEMM dispatch；本次实现的范围为
交接文档中的 post-quantization kernel。

源码位于 `csrc/fused_residual_lowrank_nvfp4/`：

- `dual_fp4_kernel.hpp`：项目内 CUTLASS kernel-layer adaptation。
- `fused_residual_lowrank_nvfp4.cu`：SM120 collective instantiation、device alpha、launch。
- `binding.cpp`：shape、dtype、device、contiguity、output alias 校验。
- `setup.py`：编译 SM120a 扩展。

源码核查发现 dense SM120 blockscaled Builder 实际选中的是
`sm90_gemm_tma_warpspecialized_cooperative.hpp` 中的 cooperative specialization，
因此以它为基础，而不是交接文档建议定位的 sparse/asymmetric-DMA specialization。
未修改 vendored CUTLASS。保留 NVIDIA BSD-3-Clause 许可证。

每个 `[128,128]` output tile 的顺序为：

1. Producer 使用 residual 的 TMA descriptors，将 X/R 和 scales 放入共享 ring。
2. 256 consumer threads 使用 SM120 FP4 MMA 得到 `Acc_R`，release 所读 stages。
3. Producer 的 `load_tail` 等待所有已用 stages release，再用第二组 descriptors
   将该 tile 的 Z/U 装入**同一个** ring；K=64 使用一个 K=128 TMA tile，越界部分
   由原 CUTLASS/TMA 路径处理。
4. Consumer 使用完全相同的 collective、TiledMma 和 fragment ownership 得到
   `Acc_L`，逐寄存器合并后调用原 epilogue。
5. Producer/consumer pipeline state 均累计推进 `ceil(K/128)+1`；不重置 barrier
   phase。Scheduler 仅在 producer 已发出该 tile 的两段 load 后获取下一 tile。
   禁止 split-K/Stream-K，避免重复累加 low-rank 项。

整个 CTA 只分配一份 mainloop TensorStorage 和 barrier storage。`load_tail` 只等待
empty barriers，不重建 pipeline；后续 tile 的预取也受同一 ring 的 acquire/release
保护。临时 `Acc_L` 为寄存器 fragment，无 shared scratch/layout conversion。

设备上的 1-thread setup kernel 计算：

```text
alpha_R = amax(X) * amax(R) / (6² * 448²)
alpha_L = amax(Z) * amax(U) / (6² * 448²)
base = alpha_R != 0 ? alpha_R : (alpha_L != 0 ? alpha_L : 1)
Acc = (alpha_R / base) * Acc_R + (alpha_L / base) * Acc_L
Y = BF16(base * Acc)
```

alpha 不经 `.item()`，使用 output device 的当前 CUDA stream。Epilogue 的 beta=0、
C pointer=nullptr，只在两段 MMA 完成后写最终 Y。Setup kernel 写 3 个 alpha 浮点数，
不会写 Y。TE 参考先把 residual 舍入至 BF16，再 beta=1 累加；本 kernel 只在最终
Y 舍入，因此不要求与两-GEMM bitwise 相同。

## 功能与内存检查

`tests/test_fused_residual_lowrank_nvfp4.py` 的 14 个 case 全部通过，保持
`rtol=2e-2, atol=1.25e-1`：

| 交接要求 / 风险 | 已验证的覆盖 |
|---|---|
| 基本同精度 TE 对齐 | 128×128×128，rank=64 |
| residual zero，alpha fallback | X=0 或 R=0，low-rank 非零 |
| low-rank zero，保留 residual | Z=0 或 U=0 |
| 两项 zero | 输出全零，无 NaN/Inf |
| 多 CTA、M/N offset | 256×256、256×384 |
| pipeline 多阶段及回绕 | K=64/128/896，8192×256×896 |
| 独立 scales/amax | 四个独立 quantizer，不同增益，TE/custom 分开的 packed objects |
| stream 与重复调用 | 非默认 stream；每个 case 连续调用三次；output 初始填 NaN |
| 真实 workload 全输出 | 8192×4864×896 和 131072×4864×896，逐 2048 行检查全部元素 |

Compute Sanitizer：memcheck 跑 12 个非 full-target cases，0 errors；racecheck 和
synccheck 分别跑 4 个小 tile cases（含 K=896），0 hazards / 0 errors。
只有 TE 的 `torch.jit.script` deprecation warning。

## 性能

GPU：RTX 5090，UUID `GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99`。
CUDA 13.2，driver 595.71.05，PyTorch 2.12.1+cu132，TE 2.18.0。
CUTLASS commit `59e3a3338d516ca6ce0e073af8da65289678a35c`。

同一进程中的三条路径，预先 materialize/quantize operands 和 swizzle scales；
每条路径 warmup=5、20 次 CUDA-event samples，取 median。包含 device alpha setup
和调用开销，不含 projection、quantization 或 bias。BF16 fusion 的 low-rank 部分
精度不同，仅作为交接指定的性能基线。

| M（N=4864,K=896,r=64） | TE FP4 + TE NVFP4 beta=1 | BF16 single-store | dual-FP4 single-store | 相对 TE 加速 |
|---:|---:|---:|---:|---:|
| 8192 | 0.162912 ms | 0.230368 ms | **0.122384 ms** | **1.33×** |
| 131072 | 3.056608 ms | 2.963248 ms | **1.562976 ms** | **1.96×** |

这是本机本次运行的测量值，不是跨设备或完整模型 forward 的性能保证。

## NCU 与 single-store 证据

只 profile `DualFp4GemmUniversal`，跳过前 3 次匹配 launch，采集 1 次 full report。
以下为 M=131072 的 profiler 数据；性能表仍使用未受 profiler 扰动的 event median。

| 指标 | 实测 |
|---|---:|
| GPU duration | 1.593408 ms |
| DRAM read（`dram__bytes_op_read.sum`） | 73,575,168 bytes |
| DRAM write（`dram__bytes_op_write.sum`） | 1,241,652,736 bytes |
| L2 total（`lts__t_sectors.sum × 32`） | 7,047,925,344 bytes |
| L2 global write（`lts__t_sectors_srcunit_tex_op_write.sum × 32`） | **1,275,068,416 bytes** |
| 期望 Y 大小（131072×4864×2） | **1,275,068,416 bytes** |
| Tensor pipe active / elapsed | 51.49% |
| SM achieved occupancy | 20.85% |
| SMSP active warps / peak（2.498280 / 12） | 20.82% |
| Threads / CTA | 384 |
| Registers / thread | 168 |
| Dynamic shared / CTA | 88,064 bytes |
| Register spill stores / loads（ptxas） | 0 / 0 |
| Register / shared occupancy limit | 各 1 CTA/SM |

L2 写入量与一份最终 Y 完全相等；结合 C=null、beta=0、只有合并后的 epilogue
store 的源码路径，验证了 single-D-store。DRAM 写入量小于 Y，因为 kernel 结束时
仍可有 dirty output 留在 L2，不能仅以 DRAM counter 等于 Y 作为验收条件。

当前 NCU 的 metric 名称与交接示例稍有差异，记录使用工具实际提供的名称。
主要 issue-stall ratios（per issue active）为 wait=2.405606、sleeping=2.252895、
math_pipe_throttle=2.007668、long_scoreboard=0.777730、barrier=0.634680。
这些是计数器 ratio，不是百分比；全部 stall 项保存在 JSON 中。

## 复现

宿主机环境（GPU 在沙箱外可见）：

```bash
cd /home/dy/zxt/fp4_post
source ../conda-init.sh
export CUDA_HOME=/home/dy/zxt/cuda-13.2
export CUDA_VISIBLE_DEVICES=GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MAX_JOBS=2 python csrc/fused_residual_lowrank_nvfp4/setup.py build_ext \
  --build-temp /tmp/dual_fp4_build/temp --build-lib /tmp/dual_fp4_build/lib --inplace
```

测试需要 pytest。本次在 `/tmp/dual_fp4_test_deps` 安装，未修改 conda environment：

```bash
python -m pip install --target /tmp/dual_fp4_test_deps pytest
export PYTHONPATH="/tmp/dual_fp4_test_deps:$PYTHONPATH"
python -m pytest -q -o cache_dir=/tmp/dual_fp4_pytest_cache \
  tests/test_fused_residual_lowrank_nvfp4.py
compute-sanitizer --tool memcheck --error-exitcode 86 python -m pytest -q \
  -o cache_dir=/tmp/dual_fp4_pytest_cache tests/test_fused_residual_lowrank_nvfp4.py \
  -k 'not target_prefill'
compute-sanitizer --tool racecheck --error-exitcode 86 python -m pytest -q \
  -o cache_dir=/tmp/dual_fp4_pytest_cache tests/test_fused_residual_lowrank_nvfp4.py \
  -k 'tiles and not 8192'
compute-sanitizer --tool synccheck --error-exitcode 86 python -m pytest -q \
  -o cache_dir=/tmp/dual_fp4_pytest_cache tests/test_fused_residual_lowrank_nvfp4.py \
  -k 'tiles and not 8192'
```

先通过上述功能测试，再执行：

```bash
for m in 8192 131072; do
  python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_nvfp4.py \
    --rows "$m" --input-features 896 --output-features 4864 --rank 64 \
    --warmup 5 --iterations 20 \
    --output "outputs/inference_nvfp4_5090/dual_fp4_fused/m${m}.json"
done
ncu --target-processes all --set full --kernel-name-base demangled \
  --kernel-name 'regex:.*DualFp4GemmUniversal.*' --launch-count 1 --launch-skip 3 \
  --export outputs/inference_nvfp4_5090/profiles/dual_fp4_fused/ncu_dual_fp4 \
  --force-overwrite \
  python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_nvfp4.py \
    --rows 131072 --only dual --warmup 3 --iterations 10
ncu --import outputs/inference_nvfp4_5090/profiles/dual_fp4_fused/ncu_dual_fp4.ncu-rep \
  --page raw --csv --print-units base \
  > outputs/inference_nvfp4_5090/profiles/dual_fp4_fused/ncu_raw.csv
```

原始测试/构建日志与 timing JSON 在 `outputs/inference_nvfp4_5090/dual_fp4_fused/`；
25 MiB NCU report 和 raw CSV 在 `outputs/inference_nvfp4_5090/profiles/dual_fp4_fused/`。
这些 outputs 被项目 gitignore 排除，关键结果与源码校验和另存于本文链接的 JSON。
