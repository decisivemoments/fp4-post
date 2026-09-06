#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

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

#include "dual_fp4_kernel.hpp"

namespace metis_dual_fp4 {
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
    cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative,
    cutlass::epilogue::fusion::LinCombPerColBias<ElementOutput, float>>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementFP4, cutlass::layout::RowMajor, 32,
    ElementFP4, cutlass::layout::ColumnMajor, 32,
    ElementAccumulator, ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::DualFp4GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void check(cutlass::Status status, const char* operation) {
  TORCH_CHECK(status == cutlass::Status::kSuccess, "SM120 NVFP4 ", operation,
              " failed: ", cutlassGetStatusString(status));
}

__global__ void dual_fp4_alphas(
    const float* x, const float* r, const float* z, const float* u, float* alphas) {
  constexpr float factor = 1.0f / (6.0f * 6.0f * 448.0f * 448.0f);
  float ar = (*x) * (*r) * factor;
  float al = (*z) * (*u) * factor;
  float base = ar != 0.0f ? ar : (al != 0.0f ? al : 1.0f);
  alphas[0] = base;
  alphas[1] = ar / base;
  alphas[2] = al / base;
}
}  // namespace metis_dual_fp4

void fused_residual_lowrank_nvfp4_cuda(
    torch::Tensor x, torch::Tensor xs, torch::Tensor r, torch::Tensor rs,
    torch::Tensor xa, torch::Tensor ra,
    torch::Tensor z, torch::Tensor zs, torch::Tensor u, torch::Tensor us,
    torch::Tensor za, torch::Tensor ua, torch::Tensor correction, torch::Tensor output) {
  using namespace metis_dual_fp4;
  c10::cuda::CUDAGuard guard(output.device());
  const int m = x.size(0), n = r.size(0), k = x.size(1) * 2;
  using BlockConfig = CollectiveMainloop::Sm1xxBlkScaledConfig;
  using StrideA = GemmKernel::StrideA;
  using StrideB = GemmKernel::StrideB;
  using StrideC = GemmKernel::StrideC;
  using StrideD = GemmKernel::StrideD;
  auto mainloop_args = [&](torch::Tensor a, torch::Tensor sa,
                           torch::Tensor b, torch::Tensor sb, int inner) {
    auto shape = cute::make_shape(m, n, inner, 1);
    return CollectiveMainloop::Arguments{
        reinterpret_cast<ElementPacked*>(a.data_ptr()),
        cutlass::make_cute_packed_stride(StrideA{}, {m, inner, 1}),
        reinterpret_cast<ElementPacked*>(b.data_ptr()),
        cutlass::make_cute_packed_stride(StrideB{}, {n, inner, 1}),
        reinterpret_cast<ElementScale*>(sa.data_ptr()), BlockConfig::tile_atom_to_shape_SFA(shape),
        reinterpret_cast<ElementScale*>(sb.data_ptr()), BlockConfig::tile_atom_to_shape_SFB(shape)};
  };
  auto alphas = torch::empty({3}, output.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
  dual_fp4_alphas<<<1, 1, 0, stream>>>(xa.data_ptr<float>(), ra.data_ptr<float>(),
      za.data_ptr<float>(), ua.data_ptr<float>(), alphas.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  Gemm::Arguments args{};
  args.mode = cutlass::gemm::GemmUniversalMode::kGemm;
  args.problem_shape = {m, n, k, 1};
  args.mainloop = mainloop_args(x, xs, r, rs, k);
  args.lowrank_problem_shape = {m, n, 64, 1};
  args.lowrank_mainloop = mainloop_args(z, zs, u, us, 64);
  args.alphas = alphas.data_ptr<float>();
  // beta=0: no C load; store each final Y element once.
  args.epilogue = {{1.0f, 0.0f, alphas.data_ptr<float>(), nullptr},
      nullptr, cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1}),
      reinterpret_cast<ElementOutput*>(output.data_ptr()),
      cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1})};
  args.epilogue.thread.bias_ptr = reinterpret_cast<ElementOutput*>(correction.data_ptr());
  args.epilogue.thread.dBias = {cute::_0{}, cute::_1{}, 0};
  args.hw_info.device_id = output.get_device();
  args.hw_info.sm_count = at::cuda::getDeviceProperties(output.get_device())->multiProcessorCount;
  Gemm gemm;
  check(gemm.can_implement(args), "can_implement");
  auto workspace = torch::empty({static_cast<int64_t>(Gemm::get_workspace_size(args))},
                                output.options().dtype(torch::kUInt8));
  check(gemm.initialize(args, workspace.data_ptr(), stream), "initialize");
  check(gemm.run(stream), "run");
}
