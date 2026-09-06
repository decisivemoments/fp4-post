// Copyright (c) 2026. Project-local RMSNorm + centered NVFP4 inference pack.
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
constexpr int kThreads = 256;
constexpr int kRowsPerPartial = 256;
constexpr int kRowsPerPackBlock = 4;
constexpr int kMaxColumns = 4096;
constexpr float kFp4Max = 6.0f;
constexpr float kFp8E4M3Max = 448.0f;

__device__ inline float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ inline float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__device__ inline float block_reduce_sum(float value) {
  __shared__ float warp_values[8];
  value = warp_reduce_sum(value);
  if ((threadIdx.x & 31) == 0) warp_values[threadIdx.x / 32] = value;
  __syncthreads();
  value = threadIdx.x < blockDim.x / 32 ? warp_values[threadIdx.x] : 0.0f;
  value = warp_reduce_sum(value);
  if (threadIdx.x == 0) warp_values[0] = value;
  __syncthreads();
  return warp_values[0];
}

__device__ inline __nv_bfloat16 rmsnorm_output_bf16(
    __nv_bfloat16 x, __nv_bfloat16 gamma, float inv_rms) {
  // Match Qwen2RMSNorm: normalize in FP32, convert to input BF16, then apply
  // its BF16 weight.  The final explicit rounding mirrors BF16 elementwise
  // multiplication when model parameters have been converted to BF16.
  const __nv_bfloat16 normalized = __float2bfloat16_rn(
      __bfloat162float(x) * inv_rms);
  return __float2bfloat16_rn(
      __bfloat162float(normalized) * __bfloat162float(gamma));
}

__device__ inline __nv_bfloat16 centered_bf16(
    __nv_bfloat16 value, __nv_bfloat16 mean) {
  return __float2bfloat16_rn(
      __bfloat162float(value) - __bfloat162float(mean));
}

__device__ inline uint8_t fp8_raw(float value) {
  const __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(value);
  return *reinterpret_cast<const uint8_t*>(&fp8);
}

__device__ inline float fp8_to_float(uint8_t raw) {
  const __nv_fp8_e4m3 fp8 = *reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
  return static_cast<float>(fp8);
}

__device__ inline uint16_t fp4x4_raw_te_rn(
    __nv_bfloat16 value0, __nv_bfloat16 value1,
    __nv_bfloat16 value2, __nv_bfloat16 value3, float scale) {
  const __nv_fp4x4_e2m1 output = __nv_fp4x4_e2m1(make_float4(
      __bfloat162float(value0) * scale,
      __bfloat162float(value1) * scale,
      __bfloat162float(value2) * scale,
      __bfloat162float(value3) * scale));
  return *reinterpret_cast<const uint16_t*>(&output);
}

__global__ void rmsnorm_column_partial_stats_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* gamma,
    float eps, float* inv_rms, float* partial_sum, float* partial_min,
    float* partial_max, int64_t rows, int64_t columns) {
  extern __shared__ __nv_bfloat16 shared_x[];
  const int64_t group = blockIdx.x;
  const int64_t row_begin = group * kRowsPerPartial;
  const int64_t row_end = min(row_begin + kRowsPerPartial, rows);
  const int slots = (columns + blockDim.x - 1) / blockDim.x;
  // H is limited by the binding so these arrays are statically register
  // allocated.  One thread owns h = tid + slot * blockDim.x.
  float sums[kMaxColumns / kThreads];
  float mins[kMaxColumns / kThreads];
  float maxs[kMaxColumns / kThreads];
#pragma unroll
  for (int slot = 0; slot < kMaxColumns / kThreads; ++slot) {
    sums[slot] = 0.0f;
    mins[slot] = INFINITY;
    maxs[slot] = -INFINITY;
  }

  for (int64_t row = row_begin; row < row_end; ++row) {
    float local_sumsq = 0.0f;
    for (int h = threadIdx.x; h < columns; h += blockDim.x) {
      const __nv_bfloat16 x = input[row * columns + h];
      shared_x[h] = x;
      const float value = __bfloat162float(x);
      local_sumsq += value * value;
    }
    const float sumsq = block_reduce_sum(local_sumsq);
    const float row_inv_rms = rsqrtf(sumsq / static_cast<float>(columns) + eps);
    if (threadIdx.x == 0) inv_rms[row] = row_inv_rms;
    __syncthreads();
    for (int slot = 0; slot < slots; ++slot) {
      const int h = threadIdx.x + slot * blockDim.x;
      if (h < columns) {
        const __nv_bfloat16 y = rmsnorm_output_bf16(
            shared_x[h], gamma[h], row_inv_rms);
        const float value = __bfloat162float(y);
        sums[slot] += value;
        mins[slot] = fminf(mins[slot], value);
        maxs[slot] = fmaxf(maxs[slot], value);
      }
    }
    __syncthreads();
  }

  for (int slot = 0; slot < slots; ++slot) {
    const int h = threadIdx.x + slot * blockDim.x;
    if (h < columns) {
      const int64_t offset = group * columns + h;
      partial_sum[offset] = sums[slot];
      partial_min[offset] = mins[slot];
      partial_max[offset] = maxs[slot];
    }
  }
}

__global__ void finalize_columns_kernel(
    const float* partial_sum, const float* partial_min, const float* partial_max,
    __nv_bfloat16* mean, float* block_amax, int64_t rows, int64_t columns,
    int64_t partial_rows) {
  float thread_amax = 0.0f;
  const int64_t column = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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
    const __nv_bfloat16 column_mean = __float2bfloat16_rn(
        sum / static_cast<float>(rows));
    mean[column] = column_mean;
    const __nv_bfloat16 min_residual = centered_bf16(
        __float2bfloat16_rn(min_value), column_mean);
    const __nv_bfloat16 max_residual = centered_bf16(
        __float2bfloat16_rn(max_value), column_mean);
    thread_amax = fmaxf(fabsf(__bfloat162float(min_residual)),
                         fabsf(__bfloat162float(max_residual)));
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
    const float* block_amax, float* amax, int64_t num_blocks) {
  float local = 0.0f;
  for (int64_t index = threadIdx.x; index < num_blocks; index += blockDim.x) {
    local = fmaxf(local, block_amax[index]);
  }
  local = warp_reduce_max(local);
  __shared__ float warp_amax[8];
  if ((threadIdx.x & 31) == 0) warp_amax[threadIdx.x / 32] = local;
  __syncthreads();
  if (threadIdx.x < 32) {
    const float value = threadIdx.x < blockDim.x / 32 ? warp_amax[threadIdx.x] : 0.0f;
    const float result = warp_reduce_max(value);
    if (threadIdx.x == 0) amax[0] = result;
  }
}

__global__ void rmsnorm_centered_pack_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* gamma,
    const float* inv_rms, const __nv_bfloat16* mean, uint8_t* data,
    uint8_t* scales, const float* global_amax, int64_t rows, int64_t columns,
    int64_t scale_stride) {
  extern __shared__ __nv_bfloat16 shared_x[];
  const int64_t row_begin = static_cast<int64_t>(blockIdx.x) * kRowsPerPackBlock;
  const int64_t row_end = min(row_begin + kRowsPerPackBlock, rows);
  const int blocks_per_row = columns / kBlockElements;
  for (int64_t row = row_begin; row < row_end; ++row) {
    for (int h = threadIdx.x; h < columns; h += blockDim.x) {
      shared_x[h] = input[row * columns + h];
    }
    __syncthreads();
    for (int block_column = threadIdx.x; block_column < blocks_per_row;
         block_column += blockDim.x) {
      const int column = block_column * kBlockElements;
      __nv_bfloat16 values[kBlockElements];
      float block_amax = 0.0f;
#pragma unroll
      for (int i = 0; i < kBlockElements; ++i) {
        const int h = column + i;
        const __nv_bfloat16 y = rmsnorm_output_bf16(
            shared_x[h], gamma[h], inv_rms[row]);
        values[i] = centered_bf16(y, mean[h]);
        block_amax = fmaxf(block_amax, fabsf(__bfloat162float(values[i])));
      }
      const float amax = *global_amax;
      const float global_encode = amax == 0.0f ? 1.0f :
          (kFp8E4M3Max * kFp4Max / amax);
      const uint8_t scale_raw = fp8_raw(block_amax * global_encode / kFp4Max);
      scales[row * scale_stride + block_column] = scale_raw;
      const float scale = fp8_to_float(scale_raw);
      const float global_decode = __fdiv_rn(1.0f, global_encode);
      const float encode = scale == 0.0f ? 1.0f :
          __fdiv_rn(1.0f, global_decode * scale);
      const int64_t data_offset = row * (columns / 2) + block_column * 8;
#pragma unroll
      for (int group = 0; group < 4; ++group) {
        const int offset = group * 4;
        const uint16_t packed = fp4x4_raw_te_rn(
            values[offset], values[offset + 1], values[offset + 2],
            values[offset + 3], encode);
        data[data_offset + group * 2] = static_cast<uint8_t>(packed & 0xff);
        data[data_offset + group * 2 + 1] = static_cast<uint8_t>(packed >> 8);
      }
    }
    __syncthreads();
  }
}
}  // namespace

void fused_rmsnorm_mean_and_quantize_centered_rowwise_cuda(
    torch::Tensor input, torch::Tensor rms_weight, double eps, torch::Tensor mean,
    torch::Tensor rowwise_data, torch::Tensor rowwise_scale_inv, torch::Tensor amax,
    torch::Tensor inv_rms, torch::Tensor partial_sum, torch::Tensor partial_min,
    torch::Tensor partial_max, torch::Tensor block_amax) {
  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1);
  const int64_t partial_rows = partial_sum.size(0);
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());
  const size_t shared_bytes = columns * sizeof(__nv_bfloat16);
  rmsnorm_column_partial_stats_kernel<<<partial_rows, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(rms_weight.data_ptr()),
      static_cast<float>(eps), inv_rms.data_ptr<float>(), partial_sum.data_ptr<float>(),
      partial_min.data_ptr<float>(), partial_max.data_ptr<float>(), rows, columns);
  const int finalize_grid = (columns + 127) / 128;
  finalize_columns_kernel<<<finalize_grid, 128, 0, stream>>>(
      partial_sum.data_ptr<float>(), partial_min.data_ptr<float>(), partial_max.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(mean.data_ptr()), block_amax.data_ptr<float>(),
      rows, columns, partial_rows);
  reduce_block_amax_kernel<<<1, kThreads, 0, stream>>>(
      block_amax.data_ptr<float>(), amax.data_ptr<float>(), block_amax.numel());
  const int pack_grid = (rows + kRowsPerPackBlock - 1) / kRowsPerPackBlock;
  rmsnorm_centered_pack_kernel<<<pack_grid, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(rms_weight.data_ptr()), inv_rms.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(mean.data_ptr()), rowwise_data.data_ptr<uint8_t>(),
      rowwise_scale_inv.data_ptr<uint8_t>(), amax.data_ptr<float>(), rows, columns,
      rowwise_scale_inv.size(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
