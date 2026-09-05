// Project-local adaptation of CUTLASS example 79a's SM120 NVFP4 mainloop.
// SPDX-License-Identifier: BSD-3-Clause
//
// Inputs are TE-compatible rowwise-packed E2M1 data and GEMM-swizzled E4M3
// scales. B is physically [N, K], therefore its logical CUTLASS layout is
// column-major [K, N] and the kernel computes C[M,N] += A[M,K] * B[N,K]^T.
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace metis_sm120 {
using namespace cute;
using ElementFP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
// ``nv_float4_t`` is a type descriptor. CUTLASS arguments use its packed
// nibble data type and E4M3 scale-factor type, not the descriptor itself.
using ElementPacked = ElementFP4::DataType;
using ElementScale = ElementFP4::ScaleFactorType;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementOutput, cutlass::layout::RowMajor, 8,
    ElementOutput, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementFP4, cutlass::layout::RowMajor, 32,
    ElementFP4, cutlass::layout::ColumnMajor, 32,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void check(cutlass::Status status, const char* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, "SM120 NVFP4 ", operation,
              " failed: ", cutlassGetStatusString(status));
}

__global__ void nvfp4_global_alpha_kernel(
    const float* a_amax, const float* b_amax, float* alpha) {
  constexpr float kFactorInv = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);
  *alpha = (*a_amax) * (*b_amax) * kFactorInv;
}
}  // namespace metis_sm120

using namespace metis_sm120;

void lowrank_nvfp4_add_sm120_cuda(torch::Tensor a_data, torch::Tensor a_scales,
                                  torch::Tensor b_data, torch::Tensor b_scales,
                                  torch::Tensor a_amax, torch::Tensor b_amax,
                                  torch::Tensor output) {
  const int m = static_cast<int>(a_data.size(0));
  const int n = static_cast<int>(b_data.size(0));
  constexpr int k = 64;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using BlockConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  const auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  const auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  const auto scale_shape = cute::make_shape(m, n, k, 1);
  const auto layout_sfa = BlockConfig::tile_atom_to_shape_SFA(scale_shape);
  const auto layout_sfb = BlockConfig::tile_atom_to_shape_SFB(scale_shape);
  auto alpha = torch::empty({1}, torch::TensorOptions().device(output.device()).dtype(torch::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  nvfp4_global_alpha_kernel<<<1, 1, 0, stream>>>(
      a_amax.data_ptr<float>(), b_amax.data_ptr<float>(), alpha.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<ElementPacked*>(a_data.data_ptr()), stride_a,
       reinterpret_cast<ElementPacked*>(b_data.data_ptr()), stride_b,
       reinterpret_cast<ElementScale*>(a_scales.data_ptr()), layout_sfa,
       reinterpret_cast<ElementScale*>(b_scales.data_ptr()), layout_sfb},
      {{1.0f, 1.0f, alpha.data_ptr<float>(), nullptr}, reinterpret_cast<ElementOutput*>(output.data_ptr()), stride_c,
       reinterpret_cast<ElementOutput*>(output.data_ptr()), stride_d}};

  Gemm gemm;
  const auto workspace_size = Gemm::get_workspace_size(args);
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().device(output.device()).dtype(torch::kUInt8));
  check(gemm.can_implement(args), "can_implement");
  check(gemm.initialize(args, workspace.data_ptr(), stream), "initialize");
  check(gemm.run(stream), "run");
}
