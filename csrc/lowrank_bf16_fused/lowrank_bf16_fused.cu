// Experimental Tensor-Core kernel for C[M,N] += A[M,64] @ B[64,N].
//
// A block owns a [128,128] output tile. Each warp owns one 16-row slice and
// walks eight 16-column slices. This warp-private ownership avoids CTA-wide
// synchronizations in the rank-64 inner loop. This is intentionally an
// isolated benchmark kernel; it is not yet wired into NativeFullNVFP4Linear.
#include <cuda_bf16.h>
#include <mma.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

namespace {
using namespace nvcuda;
constexpr int kRank = 64;
constexpr int kWarpSize = 32;
constexpr int kWarps = 8;
constexpr int kTileM = 16;
constexpr int kTileN = 16;
constexpr int kBlockM = 128;
constexpr int kBlockN = 128;

__global__ __launch_bounds__(kWarps * kWarpSize)
void lowrank_add_rank64_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ c,
    int64_t rows,
    int64_t columns) {
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kBlockM + warp * kTileM;
  const int64_t n_block_base = static_cast<int64_t>(blockIdx.x) * kBlockN;
  // All eight M-warps consume exactly the same [64,128] U tile.  Keeping it
  // in shared memory eliminates eight redundant global operand streams per
  // CTA.  This is deliberately separate from C's epilogue staging: the
  // latter is only a 16x16 temporary per warp.
  __shared__ __align__(16) __nv_bfloat16 b_shared[kRank][kBlockN];
  __shared__ __align__(16) float accumulator_shared[kWarps][kTileM * kTileN];

  for (int index = threadIdx.x; index < kRank * kBlockN;
       index += blockDim.x) {
    const int k_row = index / kBlockN;
    const int n_col = index % kBlockN;
    b_shared[k_row][n_col] = b[static_cast<int64_t>(k_row) * columns +
                               n_block_base + n_col];
  }
  __syncthreads();

#pragma unroll
  for (int n_tile = 0; n_tile < kBlockN / kTileN; ++n_tile) {
    const int64_t n_base = n_block_base + n_tile * kTileN;
    wmma::fragment<wmma::matrix_a, kTileM, kTileN, kTileM,
                   __nv_bfloat16, wmma::row_major> a_fragment;
    wmma::fragment<wmma::matrix_b, kTileM, kTileN, kTileM,
                   __nv_bfloat16, wmma::row_major> b_fragment;
    wmma::fragment<wmma::accumulator, kTileM, kTileN, kTileM, float> accumulator;
    wmma::fill_fragment(accumulator, 0.0f);
#pragma unroll
    for (int k_tile = 0; k_tile < kRank / kTileM; ++k_tile) {
      wmma::load_matrix_sync(a_fragment, a + m_base * kRank + k_tile * kTileM, kRank);
      wmma::load_matrix_sync(b_fragment,
                             &b_shared[k_tile * kTileM][n_tile * kTileN],
                             kBlockN);
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    wmma::store_matrix_sync(&accumulator_shared[warp][0], accumulator, kTileN,
                            wmma::mem_row_major);
    __syncwarp();

    // Keep the residual add in the epilogue: C is read once and final C is
    // written once for this low-rank update.
#pragma unroll
    for (int index = lane; index < kTileM * kTileN; index += kWarpSize) {
      const int local_row = index / kTileN;
      const int local_col = index % kTileN;
      const int64_t offset = (m_base + local_row) * columns + n_base + local_col;
      c[offset] = __float2bfloat16_rn(
          __bfloat162float(c[offset]) + accumulator_shared[warp][index]);
    }
    __syncwarp();
  }
}
}  // namespace

void lowrank_add_bf16_cuda(torch::Tensor scaled_v, torch::Tensor u_transpose,
                           torch::Tensor residual_output) {
  const auto rows = scaled_v.size(0);
  const auto columns = u_transpose.size(1);
  const dim3 block(kWarps * kWarpSize);
  const dim3 grid(columns / kBlockN, rows / kBlockM);
  auto stream = at::cuda::getCurrentCUDAStream(scaled_v.get_device());
  lowrank_add_rank64_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(scaled_v.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(u_transpose.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(residual_output.data_ptr()), rows, columns);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
