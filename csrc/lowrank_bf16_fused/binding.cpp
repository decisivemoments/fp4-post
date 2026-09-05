// Experimental BF16 low-rank epilogue kernel.
#include <torch/extension.h>

void lowrank_add_bf16_cuda(torch::Tensor scaled_v, torch::Tensor u_transpose,
                           torch::Tensor residual_output);
void lowrank_add_bf16_cutlass_cuda(torch::Tensor scaled_v, torch::Tensor u_transpose,
                                   torch::Tensor residual_output);

void lowrank_add_bf16(torch::Tensor scaled_v, torch::Tensor u_transpose,
                       torch::Tensor residual_output) {
  TORCH_CHECK(scaled_v.is_cuda() && u_transpose.is_cuda() && residual_output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(scaled_v.scalar_type() == torch::kBFloat16 &&
                  u_transpose.scalar_type() == torch::kBFloat16 &&
                  residual_output.scalar_type() == torch::kBFloat16,
              "all tensors must be BF16");
  TORCH_CHECK(scaled_v.is_contiguous() && u_transpose.is_contiguous() &&
                  residual_output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(scaled_v.dim() == 2 && u_transpose.dim() == 2 && residual_output.dim() == 2,
              "all tensors must be rank-2");
  TORCH_CHECK(scaled_v.size(1) == 64 && u_transpose.size(0) == 64,
              "experimental kernel supports rank=64 only");
  TORCH_CHECK(residual_output.size(0) == scaled_v.size(0) &&
                  residual_output.size(1) == u_transpose.size(1),
              "matrix shapes must satisfy residual[M,N] += scaled_v[M,64] @ u_transpose[64,N]");
  TORCH_CHECK(scaled_v.size(0) % 16 == 0 && u_transpose.size(1) % 128 == 0,
              "experimental kernel requires M divisible by 16 and N divisible by 128");
  lowrank_add_bf16_cuda(scaled_v, u_transpose, residual_output);
}

void lowrank_add_bf16_cutlass(torch::Tensor scaled_v, torch::Tensor u_transpose,
                               torch::Tensor residual_output) {
  TORCH_CHECK(scaled_v.is_cuda() && u_transpose.is_cuda() && residual_output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(scaled_v.scalar_type() == torch::kBFloat16 &&
                  u_transpose.scalar_type() == torch::kBFloat16 &&
                  residual_output.scalar_type() == torch::kBFloat16,
              "all tensors must be BF16");
  TORCH_CHECK(scaled_v.is_contiguous() && u_transpose.is_contiguous() &&
                  residual_output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(scaled_v.dim() == 2 && u_transpose.dim() == 2 && residual_output.dim() == 2,
              "all tensors must be rank-2");
  TORCH_CHECK(scaled_v.size(1) == 64 && u_transpose.size(0) == 64,
              "experimental kernel supports rank=64 only");
  TORCH_CHECK(residual_output.size(0) == scaled_v.size(0) &&
                  residual_output.size(1) == u_transpose.size(1),
              "matrix shapes must satisfy residual[M,N] += scaled_v[M,64] @ u_transpose[64,N]");
  lowrank_add_bf16_cutlass_cuda(scaled_v, u_transpose, residual_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lowrank_add_bf16", &lowrank_add_bf16,
        "Experimental fused residual + BF16 rank-64 GEMM (CUDA)");
  m.def("lowrank_add_bf16_cutlass", &lowrank_add_bf16_cutlass,
        "Experimental CUTLASS residual + BF16 rank-64 GEMM (CUDA)");
}
