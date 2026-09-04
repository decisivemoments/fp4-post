// Copyright (c) 2026.
//
// Inference-only centered NVFP4 E2M1/E4M3 implementation.  The layout is
// the compact rowwise layout allocated by TE 2.15 NVFP4Quantizer.make_empty:
// data[M, K/2], scales[round_up(M,128), round_up(K/16,4)], amax[1].
#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>

namespace {
constexpr int kBlockElements = 16;
constexpr int kColumnsPerStatsBlock = 128;
constexpr int kRowsPerPartial = 256;
constexpr float kFp4Max = 6.0f;
constexpr float kFp8E4M3Max = 448.0f;

__device__ inline __nv_bfloat16 centered_bf16(
    __nv_bfloat16 value,
    __nv_bfloat16 mean) {
  // This explicit cast is required: the TE reference first materializes a
  // BF16 residual, then computes amax and FP4 scales from that rounded value.
  return __float2bfloat16_rn(__bfloat162float(value) - __bfloat162float(mean));
}

__device__ inline void atomic_max_positive(float* address, float value) {
  atomicMax(reinterpret_cast<int*>(address), __float_as_int(value));
}

__global__ void centered_amax_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* mean,
    float* amax,
    int64_t elements,
    int64_t columns) {
  float local = 0.0f;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const __nv_bfloat16 residual = centered_bf16(input[index], mean[index % columns]);
    local = fmaxf(local, fabsf(__bfloat162float(residual)));
  }
  // The value is non-negative, so integer ordering is float ordering.
  if (local != 0.0f) atomic_max_positive(amax, local);
}

// One thread owns one column inside a [256 rows, 128 columns] tile.  At every
// loop iteration consecutive threads read consecutive columns, so the large
// activation read remains coalesced.  We retain min and max as well as sum:
// after mean[k] is known, max_i |x[i,k]-mean[k]| follows from the two extrema.
__global__ void column_partial_stats_kernel(
    const __nv_bfloat16* input,
    float* partial_sum,
    float* partial_min,
    float* partial_max,
    int64_t rows,
    int64_t columns) {
  const int64_t column = static_cast<int64_t>(blockIdx.x) * kColumnsPerStatsBlock + threadIdx.x;
  const int64_t tile = blockIdx.y;
  if (column >= columns) return;

  const int64_t row_begin = tile * kRowsPerPartial;
  const int64_t row_end = min(row_begin + kRowsPerPartial, rows);
  float sum = 0.0f;
  float min_value = INFINITY;
  float max_value = -INFINITY;
#pragma unroll 4
  for (int64_t row = row_begin; row < row_end; ++row) {
    const float value = __bfloat162float(input[row * columns + column]);
    sum += value;
    min_value = fminf(min_value, value);
    max_value = fmaxf(max_value, value);
  }
  const int64_t offset = tile * columns + column;
  partial_sum[offset] = sum;
  partial_min[offset] = min_value;
  partial_max[offset] = max_value;
}

__device__ inline float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

// Each block finalizes 128 columns.  This preserves the per-column partial
// sum order while distributing the large [ceil(M/256), K] read over K/128
// blocks instead of serializing it through one SM.
__global__ void finalize_columns_kernel(
    const float* partial_sum,
    const float* partial_min,
    const float* partial_max,
    __nv_bfloat16* mean,
    float* block_amax,
    int64_t rows,
    int64_t columns,
    int64_t partial_rows) {
  float thread_amax = 0.0f;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (column < columns) {
    float sum = 0.0f;
    float min_value = INFINITY;
    float max_value = -INFINITY;
#pragma unroll
    for (int64_t tile = 0; tile < partial_rows; ++tile) {
      const int64_t offset = tile * columns + column;
      sum += partial_sum[offset];
      min_value = fminf(min_value, partial_min[offset]);
      max_value = fmaxf(max_value, partial_max[offset]);
    }
    const __nv_bfloat16 column_mean = __float2bfloat16_rn(sum / static_cast<float>(rows));
    mean[column] = column_mean;
    // The pack path takes a BF16-rounded residual.  Apply precisely the same
    // rounding to the two extrema before deriving its amax.
    const __nv_bfloat16 min_residual = centered_bf16(
        __float2bfloat16_rn(min_value), column_mean);
    const __nv_bfloat16 max_residual = centered_bf16(
        __float2bfloat16_rn(max_value), column_mean);
    thread_amax = fmaxf(
        thread_amax,
        fmaxf(fabsf(__bfloat162float(min_residual)), fabsf(__bfloat162float(max_residual))));
  }
  thread_amax = warp_reduce_max(thread_amax);
  __shared__ float warp_amax[8];
  if ((threadIdx.x & 31) == 0) warp_amax[threadIdx.x / 32] = thread_amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    const float value = threadIdx.x < blockDim.x / 32 ? warp_amax[threadIdx.x] : 0.0f;
    const float result = warp_reduce_max(value);
    if (threadIdx.x == 0) block_amax[blockIdx.x] = result;
  }
}

__global__ void reduce_block_amax_kernel(
    const float* block_amax,
    float* amax,
    int64_t num_blocks) {
  float local_amax = 0.0f;
  for (int64_t index = threadIdx.x; index < num_blocks; index += blockDim.x) {
    local_amax = fmaxf(local_amax, block_amax[index]);
  }
  local_amax = warp_reduce_max(local_amax);
  __shared__ float warp_amax[8];
  if ((threadIdx.x & 31) == 0) warp_amax[threadIdx.x / 32] = local_amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    const float value = threadIdx.x < blockDim.x / 32 ? warp_amax[threadIdx.x] : 0.0f;
    const float result = warp_reduce_max(value);
    if (threadIdx.x == 0) amax[0] = result;
  }
}

__device__ inline uint8_t fp8_raw(float value) {
  const __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(value);
  return *reinterpret_cast<const uint8_t*>(&fp8);
}

__device__ inline float fp8_to_float(uint8_t raw) {
  const __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
  return static_cast<float>(fp8);
}

// This is deliberately identical to TE's generic rowwise NVFP4 path:
// `fp4e2m1x4(make_float4(x0 * scale, ...))`.  The scalar FP4 constructor
// follows a slightly different lowering on Blackwell at conversion boundaries.
__device__ inline uint16_t fp4x4_raw_te_rn(
    __nv_bfloat16 value0,
    __nv_bfloat16 value1,
    __nv_bfloat16 value2,
    __nv_bfloat16 value3,
    float scale) {
  const __nv_fp4x4_e2m1 output = __nv_fp4x4_e2m1(make_float4(
      __bfloat162float(value0) * scale,
      __bfloat162float(value1) * scale,
      __bfloat162float(value2) * scale,
      __bfloat162float(value3) * scale));
  return *reinterpret_cast<const uint16_t*>(&output);
}

__global__ void centered_pack_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* mean,
    uint8_t* data,
    uint8_t* scales,
    const float* global_amax,
    int64_t rows,
    int64_t columns,
    int64_t scale_stride) {
  const int64_t block_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t blocks_per_row = columns / kBlockElements;
  const int64_t total_blocks = rows * blocks_per_row;
  if (block_index >= total_blocks) return;

  const int64_t row = block_index / blocks_per_row;
  const int64_t block_column = block_index % blocks_per_row;
  const int64_t column = block_column * kBlockElements;
  __nv_bfloat16 values[kBlockElements];
  float block_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlockElements; ++i) {
    values[i] = centered_bf16(input[row * columns + column + i], mean[column + i]);
    block_amax = fmaxf(block_amax, fabsf(__bfloat162float(values[i])));
  }

  const float amax = *global_amax;
  const float global_encode = (amax == 0.0f) ? 1.0f : (kFp8E4M3Max * kFp4Max / amax);
  // Equivalent to TE v2.15 compute_decoding_scaling_factor for deterministic
  // rowwise NVFP4: S_dec_b = amax_block * S_enc / 6, rounded to E4M3.
  const uint8_t scale_raw = fp8_raw(block_amax * global_encode / kFp4Max);
  scales[row * scale_stride + block_column] = scale_raw;
  const float scale = fp8_to_float(scale_raw);
  // The TE tuned-1D dispatch (with NVTE_USE_FAST_MATH unset) intentionally
  // forms this as two round-to-nearest divisions, rather than S_enc / scale.
  // The two formulas differ by one ULP at some E2M1 decision boundaries.
  const float global_decode = __fdiv_rn(1.0f, global_encode);
  const float encode = scale == 0.0f ? 1.0f : __fdiv_rn(1.0f, global_decode * scale);

  const int64_t data_offset = row * (columns / 2) + block_column * (kBlockElements / 2);
#pragma unroll
  for (int group = 0; group < kBlockElements / 4; ++group) {
    const int offset = group * 4;
    const uint16_t packed = fp4x4_raw_te_rn(
        values[offset], values[offset + 1], values[offset + 2], values[offset + 3], encode);
    data[data_offset + group * 2] = static_cast<uint8_t>(packed & 0xff);
    data[data_offset + group * 2 + 1] = static_cast<uint8_t>(packed >> 8);
  }
}
}  // namespace

void quantize_centered_rowwise_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor rowwise_data,
    torch::Tensor rowwise_scale_inv,
    torch::Tensor amax) {
  const auto rows = input.size(0);
  const auto columns = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  amax.zero_();
  constexpr int threads = 256;
  const int blocks = std::min<int64_t>((input.numel() + threads - 1) / threads, 4096);
  centered_amax_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(mean.data_ptr()),
      amax.data_ptr<float>(), input.numel(), columns);
  const int64_t quant_blocks = rows * (columns / kBlockElements);
  const int pack_grid = (quant_blocks + threads - 1) / threads;
  centered_pack_kernel<<<pack_grid, threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(mean.data_ptr()),
      rowwise_data.data_ptr<uint8_t>(), rowwise_scale_inv.data_ptr<uint8_t>(),
      amax.data_ptr<float>(), rows, columns, rowwise_scale_inv.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_mean_and_quantize_centered_rowwise_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor rowwise_data,
    torch::Tensor rowwise_scale_inv,
    torch::Tensor amax,
    torch::Tensor partial_sum,
    torch::Tensor partial_min,
    torch::Tensor partial_max,
    torch::Tensor block_amax) {
  const auto rows = input.size(0);
  const auto columns = input.size(1);
  const auto partial_rows = partial_sum.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  const dim3 stats_block(kColumnsPerStatsBlock);
  const dim3 stats_grid((columns + kColumnsPerStatsBlock - 1) / kColumnsPerStatsBlock,
                        partial_rows);
  column_partial_stats_kernel<<<stats_grid, stats_block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      partial_sum.data_ptr<float>(), partial_min.data_ptr<float>(), partial_max.data_ptr<float>(),
      rows, columns);
  constexpr int finalize_threads = kColumnsPerStatsBlock;
  const int finalize_grid = (columns + finalize_threads - 1) / finalize_threads;
  finalize_columns_kernel<<<finalize_grid, finalize_threads, 0, stream>>>(
      partial_sum.data_ptr<float>(), partial_min.data_ptr<float>(), partial_max.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(mean.data_ptr()), block_amax.data_ptr<float>(),
      rows, columns, partial_rows);
  constexpr int amax_reduce_threads = 256;
  reduce_block_amax_kernel<<<1, amax_reduce_threads, 0, stream>>>(
      block_amax.data_ptr<float>(), amax.data_ptr<float>(), block_amax.numel());

  constexpr int pack_threads = 256;
  const int64_t quant_blocks = rows * (columns / kBlockElements);
  const int pack_grid = (quant_blocks + pack_threads - 1) / pack_threads;
  centered_pack_kernel<<<pack_grid, pack_threads, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(mean.data_ptr()),
      rowwise_data.data_ptr<uint8_t>(), rowwise_scale_inv.data_ptr<uint8_t>(),
      amax.data_ptr<float>(), rows, columns, rowwise_scale_inv.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
