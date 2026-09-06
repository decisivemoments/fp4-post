# Dual-FP4 activation mean correction：实现与验收

2026-09-06，完成 [交接要求](dual_fp4_mean_correction_handoff.md)。当前 kernel 的
最终 BF16 输出包含 residual FP4、low-rank FP4 和 activation mean correction。
完整测量样本、NCU counters、Nsight Systems kernel 时间线和源码 SHA256 保存在
[结果 JSON](dual_fp4_mean_correction_results.json)。

## 实现与 API

```python
fused_residual_lowrank_nvfp4(
    packed_x, packed_residual, packed_z, packed_u,
    residual_mean_correction, output,
)
```

`residual_mean_correction` 是必须显式提供的 contiguous CUDA BF16 `[N]`，与 output
同 device，不能与 output alias。PyBind 检查 dtype、shape、device、contiguity、alias。
原五参数 API 不再接受，以免调用方静默遗漏 correction。

C++ `CollectiveBuilder` 选择 `fusion::LinCombPerColBias<ElementOutput, float>`。
当前 CUTLASS 的 SM120 callbacks 继承对应 SM90 callbacks，能直接实例化，无需修改
vendored CUTLASS。`bias_ptr` 指向 correction，`dBias=(0,1,0)`，沿 M 广播、随 N
索引。Epilogue 保持 beta=0、C=nullptr：

```text
D[m,n] = BF16(base_alpha * (residual_ratio * Acc_R + lowrank_ratio * Acc_L)
              + float(residual_mean_correction[n]))
```

两段 FP4 MMA、fragment ownership、pipeline storage 和 phase 顺序均未修改。
Correction 在 FP32 域加入，只有最终 epilogue 转为 BF16 并写 D。

Native fused 分支仍只计算一次 `F.linear(mean, dequant(R))` 小向量，随后释放 R 的
临时 BF16 dequantized storage。把小向量传给新 API，删除了 kernel 后的
`output.add_(residual_mean_correction)`。Z 仍先乘 singular values、按需要补零至
`packed_rows`，再单独量化；最终 output 裁至逻辑 rows。

可选 module bias 也先与 correction 合并成一个小 BF16 向量，fused 分支不再执行
输出尺寸的 `output + bias`。这会改变部分 BF16 舍入顺序，已在带 bias 的 native
测试中覆盖。不满足 SM120/rank64/对齐条件时仍走原 TE 分支，其 bias 路径保持原语义。

## 正确性

共 **36 项通过**：20 项 direct kernel/API tests 与 16 项 native tests。

- Direct kernel 保持 `rtol=0.02, atol=0.125`，参考为两次 TE NVFP4 GEMM 的 BF16
  输出再加 BF16 correction。覆盖非零/零 correction、zero residual、zero low-rank、
  两项全零、独立 amax/scales、多 CTA、非默认 stream，以及两个完整 prefill shape。
- `[N]` 使用随列变化的 correction，检查广播方向和跨 N tile 的 offset；两项 FP4
  全零时要求输出与广播后的 correction bitwise 相同。
- 新 API 拒绝错误 dtype、rank/shape、CPU device、非 contiguous stride 和 output alias。
- Native 新增 8 项：逻辑 M=128、97、225、48，分别有/无 module bias。M=97/225
  验证 `packed_rows != rows` 且走 fused；M=48 验证走 TE fallback。输入均值显式
  偏移 +2，遗漏 mean correction 会产生显著错误。
- Native 与独立 TE 同精度组合参考逐元素对齐（`rtol=0.02, atol=0.02`），同时与
  BF16 low-rank forward 比较。相对 BF16 的 L2 为 6.779%–6.880%；独立 TE FP4
  参考也为 6.779%–6.879%，说明主要偏差来自 Z/U 的 FP4 量化。测试限制融合路径
  相对 BF16 的 L2 不得超过独立 TE 参考误差加 0.5 个百分点；本次最大差异仅约
  0.0042 个百分点。最初直接设 4% 的 BF16 门限会连 TE fallback 也拒绝，因此用
  同精度参考分离量化误差，未放宽 direct kernel 的同精度公差。
- 原 native backward、optimizer cache、inference-only 等回归测试均通过。

Compute Sanitizer 对 12 个非 full-target direct cases 分别运行 memcheck、racecheck、
synccheck，全部为 0 errors / 0 hazards。日志有 TE deprecation 和 cuBLAS context
初始化 warning，没有功能错误。

## 完整 native forward 性能

目标 `M=131072,N=4864,K=896,r=64`，RTX 5090 UUID
`GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99`，CUDA 13.2、PyTorch 2.12.1+cu132、
TE 2.18.0。每条路径 warmup=5、20 次 CUDA-event samples 的 median：

| 完整 forward 路径 | median |
|---|---:|
| 旧 dual-FP4 kernel + 输出 mean add | 4.058336 ms |
| 新 dual-FP4 mean epilogue | **2.368704 ms** |
| BF16 low-rank native reference | 4.076736 ms |

相对旧路径 **1.71×**，延迟降低约 **41.6%**。测量每次清除 activation cache，包含
activation mean、center/quantize、R/V mean correction、V projection、singular-value
乘法、Z quantization 和最终 GEMM；weight cache 预热，采用 TE rowwise-only activation
packing，module bias=False。未把 isolated dual-FP4 kernel 时间冒充完整 forward。

旧路径使用上一任务保存的原始无 bias 扩展二进制，再调用 `out.add_(correction)`，
其余 native 操作与新路径一致。原始扩展副本及 checksum 已保留，未用当前新 kernel
加零 bias 代替旧实现。计时结果为这次本机运行，不是完整模型吞吐保证。

## Single-store 证据

Nsight Systems 在 `dual_fp4_mean.full_native_forward` NVTX 范围采集一次完整 forward：
21 个 GPU kernels，最后一个是 `DualFp4GemmUniversal`，其后 **0 kernels**。前面的
BF16 elementwise 操作对应 activation centering 和 V/singular-value scaling；最终
`[M,N]` 没有单独 correction add。完整 kernel 名称、stream 和时间戳在结果 JSON，
原始 `.nsys-rep` / SQLite 位于 `outputs/inference_nvfp4_5090/profiles/dual_fp4_mean/`。

NCU 仅采集新 bias-enabled dual-FP4 kernel，M=131072：

| 指标 | 值 |
|---|---:|
| `lts__t_sectors_srcunit_tex_op_write.sum × 32` | **1,275,068,416 bytes** |
| 期望输出 `M×N×2` | **1,275,068,416 bytes** |
| `dram__bytes_op_read.sum` | 73,589,760 bytes |
| `dram__bytes_op_write.sum` | 1,241,758,976 bytes |
| GPU duration（profiler） | 1.634016 ms |
| Registers/thread | 168 |
| Dynamic shared/CTA | 88,064 bytes |

L2 global writes 恰好等于一份最终 BF16 Y；DRAM write 较小是部分 dirty output 仍留在
L2。与 native trace 和 C=null/beta=0 的源码共同证明没有第二轮输出 read-modify-write。
约 9.5 KiB correction vector 的读是预期行为。

## 复现

```bash
cd /home/dy/zxt/fp4_post
source ../conda-init.sh
export CUDA_HOME=/home/dy/zxt/cuda-13.2
export CUDA_VISIBLE_DEVICES=GPU-f9d7a2e3-1ba4-f5f2-2a72-426201979f99
export PYTHONPATH="/tmp/dual_fp4_test_deps:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
MAX_JOBS=2 python csrc/fused_residual_lowrank_nvfp4/setup.py build_ext \
  --build-temp /tmp/dual_fp4_mean_build/temp --build-lib /tmp/dual_fp4_mean_build/lib --inplace
python -m pytest -q -s -o cache_dir=/tmp/dual_fp4_mean_pytest \
  tests/test_fused_residual_lowrank_nvfp4.py tests/test_native_nvfp4.py
```

`/tmp/dual_fp4_test_deps` 是本次 pytest 安装目录；如果环境已有 pytest，可省略。
完整 forward benchmark：

```bash
for mode in legacy fused bf16; do
  python benchmarks/inference_nvfp4_5090/benchmark_dual_fp4_mean_native.py \
    --mode "$mode" --rows 131072 --warmup 5 --iterations 20 \
    --legacy-extension outputs/inference_nvfp4_5090/dual_fp4_mean/_fused_residual_lowrank_nvfp4_cuda.cpython-312-x86_64-linux-gnu.so \
    --output "outputs/inference_nvfp4_5090/dual_fp4_mean/${mode}.json"
done
nsys profile --trace=cuda,nvtx --sample=none --capture-range=cudaProfilerApi \
  --capture-range-end=stop --force-overwrite=true \
  --output=outputs/inference_nvfp4_5090/profiles/dual_fp4_mean/native_fused \
  python benchmarks/inference_nvfp4_5090/benchmark_dual_fp4_mean_native.py \
    --profile --mode fused --rows 131072
ncu --target-processes all --set full --kernel-name-base demangled \
  --kernel-name 'regex:.*DualFp4GemmUniversal.*' --launch-skip 3 --launch-count 1 \
  --force-overwrite --export outputs/inference_nvfp4_5090/profiles/dual_fp4_mean/fused_mean \
  python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_nvfp4.py \
    --rows 131072 --only dual --warmup 3 --iterations 10
```

NCU 命令采集 isolated kernel 仅用于 traffic/resource 检查；forward 性能取上面的
native benchmark。原始 build/test/sanitizer 日志和 timing JSON 在
`outputs/inference_nvfp4_5090/dual_fp4_mean/`。outputs 被 gitignore 排除，关键数据
与源码 checksum 另存于本文链接的 JSON；旧验证文档已标记为历史五参数 API。
