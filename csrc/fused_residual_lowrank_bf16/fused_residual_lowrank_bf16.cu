// Functional single-store prototype. See fused_epilogue.hpp for scope.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
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

#include "fused_epilogue.hpp"

namespace metis_fused {
using namespace cute;
using ElementFP4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementPacked = ElementFP4::DataType;
using ElementScale = ElementFP4::ScaleFactorType;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm120;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

// Ask the same Builder configuration which D shared-memory atom/tile it will
// select, then instantiate the low-rank callback with that exact swizzle.
// The provisional fusion op does not change the D layout selection.
using ProvisionalFusion = cutlass::epilogue::fusion::ScaledAcc<
    ElementOutput, ElementAccumulator, float>;
using ProvisionalEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementOutput, cutlass::layout::RowMajor, 8,
    ElementOutput, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
    ProvisionalFusion>::CollectiveOp;
using LowrankScratchLayout = decltype(tile_to_shape(
    ProvisionalEpilogue::SmemLayoutAtomD{},
    make_shape(size<0>(shape(ProvisionalEpilogue::EpilogueTile{})),
               size<1>(shape(ProvisionalEpilogue::EpilogueTile{}))),
    Step<_1, _2>{}));

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementOutput, cutlass::layout::RowMajor, 8,
    ElementOutput, cutlass::layout::RowMajor, 8,
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
    metis::fused_epilogue::Callbacks<LowrankScratchLayout>>::CollectiveOp;

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

__global__ void alpha_kernel(const float* a, const float* b, float* out) {
  constexpr float kFactorInv = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);
  *out = (*a) * (*b) * kFactorInv;
}

void check(cutlass::Status status, const char* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, "fused residual kernel ", operation,
              " failed: ", cutlassGetStatusString(status));
}
}  // namespace metis_fused

void fused_residual_lowrank_bf16_cuda(
    torch::Tensor x_data, torch::Tensor x_scales,
    torch::Tensor residual_data, torch::Tensor residual_scales,
    torch::Tensor x_amax, torch::Tensor residual_amax,
    torch::Tensor z, torch::Tensor u, torch::Tensor output) {
  using namespace metis_fused;
  const int m = static_cast<int>(x_data.size(0));
  const int n = static_cast<int>(residual_data.size(0));
  const int k = static_cast<int>(x_data.size(1) * 2);
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using BlockConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  const auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  const auto stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  const auto scales_shape = cute::make_shape(m, n, k, 1);
  const auto layout_sfa = BlockConfig::tile_atom_to_shape_SFA(scales_shape);
  const auto layout_sfb = BlockConfig::tile_atom_to_shape_SFB(scales_shape);
  auto alpha = torch::empty({1}, torch::TensorOptions().device(output.device()).dtype(torch::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  alpha_kernel<<<1, 1, 0, stream>>>(x_amax.data_ptr<float>(), residual_amax.data_ptr<float>(), alpha.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1},
      {reinterpret_cast<ElementPacked*>(x_data.data_ptr()), stride_a,
       reinterpret_cast<ElementPacked*>(residual_data.data_ptr()), stride_b,
       reinterpret_cast<ElementScale*>(x_scales.data_ptr()), layout_sfa,
       reinterpret_cast<ElementScale*>(residual_scales.data_ptr()), layout_sfb},
      {{1.0f, alpha.data_ptr<float>(),
        reinterpret_cast<ElementOutput*>(z.data_ptr()), reinterpret_cast<ElementOutput*>(u.data_ptr()), 64, 64},
       reinterpret_cast<ElementOutput*>(output.data_ptr()), stride_c,
       reinterpret_cast<ElementOutput*>(output.data_ptr()), stride_d}};
  Gemm gemm;
  const auto workspace_size = Gemm::get_workspace_size(args);
  auto workspace = torch::empty({static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().device(output.device()).dtype(torch::kUInt8));
  check(gemm.can_implement(args), "can_implement");
  check(gemm.initialize(args, workspace.data_ptr(), stream), "initialize");
  check(gemm.run(stream), "run");
}
