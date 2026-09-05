// CUTLASS baseline for C[M,N] = A[M,64] * B[64,N] + C[M,N].
//
// The row-major [64,N] B storage is presented to CUTLASS as an [N,64]
// column-major operand, so no transpose materialization is needed.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>

namespace {
using Element = cutlass::bfloat16_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using Gemm = cutlass::gemm::device::Gemm<
    Element, LayoutA,
    Element, LayoutB,
    Element, LayoutC,
    float,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80>;
}  // namespace

void lowrank_add_bf16_cutlass_cuda(torch::Tensor scaled_v, torch::Tensor u_transpose,
                                   torch::Tensor residual_output) {
  const int m = static_cast<int>(scaled_v.size(0));
  const int n = static_cast<int>(u_transpose.size(1));
  constexpr int k = 64;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {reinterpret_cast<Element*>(scaled_v.data_ptr()), k},
      {reinterpret_cast<Element*>(u_transpose.data_ptr()), n},
      {reinterpret_cast<Element*>(residual_output.data_ptr()), n},
      {reinterpret_cast<Element*>(residual_output.data_ptr()), n},
      {1.0f, 1.0f});
  const size_t workspace_size = gemm.get_workspace_size(args);
  auto workspace = torch::empty(
      {static_cast<int64_t>(workspace_size)},
      torch::TensorOptions().device(scaled_v.device()).dtype(torch::kUInt8));
  auto check_status = [](cutlass::Status status, const char* operation) {
    TORCH_CHECK(status == cutlass::Status::kSuccess,
                "CUTLASS ", operation, " failed: ", cutlassGetStatusString(status));
  };
  check_status(gemm.can_implement(args), "can_implement");
  check_status(gemm.initialize(args, workspace.data_ptr(), at::cuda::getCurrentCUDAStream()),
               "initialize");
  check_status(gemm(at::cuda::getCurrentCUDAStream()), "run");
}
