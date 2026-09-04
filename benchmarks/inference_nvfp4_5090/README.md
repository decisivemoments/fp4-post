# Qwen 单层 BF16 / Native-NVFP4 推理基准

该目录提供两种固定 `sequence_length=512` 的推理 benchmark：

- `linear`：Q、K、V、O、Gate、Up、Down 七个 projection 分别计时；
- `transformer`：一个完整 Qwen decoder layer（RMSNorm、attention、MLP 和
  residual 均在内）计时。

NVFP4 模式使用 `NativeFullNVFP4Linear`，即当前 Metis 的 rank-64 SVD、FP4
residual/V GEMM 和 BF16 correction 路径。计时前完成 SVD 和权重 pack；每次
forward 的 activation pack 保留在计时区间。

`--activation-mode fresh` 是默认值：在两个预分配但地址不同的 activation
间交替，确保每次 native linear 都执行 activation pack，同时不将额外的
`clone`/H2D copy 计入时间。`cached` 则复用同一 activation，只适用于观察
FP4 GEMM 与后续路径的理论上限。

```bash
cd fp4_post
PYTHONPATH=src python benchmarks/inference_nvfp4_5090/benchmark_qwen_inference.py \
  --scope linear --model qwen0.5b --model-path /models/Qwen2.5-0.5B-Instruct \
  --batch-sizes 1 2 4 8 16 32 --seq-length 512 \
  --output outputs/inference_nvfp4_5090/linear_qwen0.5b.json
```

`--model-path` 应指向本地模型权重；省略时会使用 Hugging Face 的相应 Qwen id。
一次完成三种模型与两个层级的 sweep：

```bash
MODEL_PATH_0_5B=/models/Qwen2.5-0.5B-Instruct \
MODEL_PATH_1_5B=/models/Qwen2.5-1.5B-Instruct \
MODEL_PATH_7B=/models/Qwen2.5-7B-Instruct \
bash benchmarks/inference_nvfp4_5090/run_qwen_layer_sweep.sh
```

## 单层 NVTX / Nsight Systems profile

为避免一个 trace 混入七个 projection，先只采集一项。下面以 batch 128 的
`down_proj` 为例；`up_proj` 用同样命令替换 `--projections` 值即可。

```bash
nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output outputs/inference_nvfp4_5090/nsys_down_b128 \
  python benchmarks/inference_nvfp4_5090/benchmark_qwen_inference.py \
    --scope linear --model qwen0.5b --model-path /models/Qwen2.5-0.5B-Instruct \
    --modes nvfp4 --projections down_proj --batch-sizes 128 --seq-length 512 \
    --activation-mode fresh --profile-native-ranges --cuda-profiler-range \
    --warmup 5 --iterations 5 \
    --output outputs/inference_nvfp4_5090/profile_down_b128.json
```

`--profile-native-ranges` emits NVTX ranges for activation packing, native
FP4 GEMM, BF16 mean correction and the rank-64 BF16 path. The CUDA profiler
range starts only after warmup, so SVD and one-time weight packing are not in
the captured region.

Activation packing includes nested `activation_mean` and
`activation_center_and_pack` ranges, separating the global reduction from
residual materialization plus TE NVFP4 packing. `--activation-layout rowwise`
is an inference-only experiment that omits columnwise packing; never use it
for backward or training.

同一流程已写入 `run_qwen_layer_sweep.sh`，可只运行第一阶段而跳过完整
sweep：

```bash
RUN_SWEEP=0 RUN_PHASE1_PROFILE=1 \
NSYS_BIN=/usr/local/cuda/bin/nsys \
bash benchmarks/inference_nvfp4_5090/run_qwen_layer_sweep.sh
```

默认在 `outputs/inference_nvfp4_5090/phase1_profile/qwen0.5b/` 分别写入
`down_proj`、`up_proj` 的 `full` 与 experimental `rowwise` activation-layout
Nsight Systems report 与 JSON。可用
`PROFILE_BATCH_SIZE`、`PROFILE_PROJECTIONS`、`PROFILE_WARMUP`、
`PROFILE_ITERATIONS`、`PROFILE_ACTIVATION_LAYOUTS` 覆盖默认值。

## Low-rank U × scaled-V precision sweep

The low-rank benchmark keeps `U` and `scaled_v` separate as required by the
method: it packs the static U once, then quantizes each runtime
`scaled_v_output` before its NVFP4 GEMM. It reports BF16 latency, NVFP4
pack-plus-GEMM latency, packed-GEMM-only latency, and relative L2 error.

```bash
RUN_SWEEP=0 RUN_PHASE1_PROFILE=0 RUN_LOWRANK_BENCHMARK=1 \
bash benchmarks/inference_nvfp4_5090/run_qwen_layer_sweep.sh
```

By default this writes separate `up_proj` and `down_proj` JSON results. Use
`LOWRANK_PROJECTIONS` and `LOWRANK_BATCH_SIZES` to narrow the sweep.

每项 JSON 含中位延迟、逻辑 FLOPs、逻辑 TFLOP/s、相对于单卡 RTX 5090 dense
BF16/NVFP4 峰值的等效利用率，以及逐 batch/projection 的
`bf16_over_nvfp4` 加速比。计算定义及固定 batch 的 FLOPs 表见
[COMPUTE_ACCOUNTING.md](COMPUTE_ACCOUNTING.md)。
