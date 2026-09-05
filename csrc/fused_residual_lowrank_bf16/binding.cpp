#include <torch/extension.h>

void fused_residual_lowrank_bf16_cuda(
    torch::Tensor x_data, torch::Tensor x_scales,
    torch::Tensor residual_data, torch::Tensor residual_scales,
    torch::Tensor x_amax, torch::Tensor residual_amax,
    torch::Tensor z, torch::Tensor u, torch::Tensor output);

void fused_residual_lowrank_bf16(
    torch::Tensor x_data, torch::Tensor x_scales,
    torch::Tensor residual_data, torch::Tensor residual_scales,
    torch::Tensor x_amax, torch::Tensor residual_amax,
    torch::Tensor z, torch::Tensor u, torch::Tensor output) {
  TORCH_CHECK(x_data.is_cuda() && x_scales.is_cuda() && residual_data.is_cuda() &&
                  residual_scales.is_cuda() && x_amax.is_cuda() && residual_amax.is_cuda() &&
                  z.is_cuda() && u.is_cuda() && output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(x_data.scalar_type() == torch::kUInt8 && residual_data.scalar_type() == torch::kUInt8,
              "packed FP4 data must be uint8");
  TORCH_CHECK(x_scales.scalar_type() == at::kFloat8_e4m3fn &&
                  residual_scales.scalar_type() == at::kFloat8_e4m3fn,
              "FP4 scales must be GEMM-swizzled float8_e4m3fn");
  TORCH_CHECK(z.scalar_type() == torch::kBFloat16 && u.scalar_type() == torch::kBFloat16 &&
                  output.scalar_type() == torch::kBFloat16,
              "Z, U, and output must be BF16");
  TORCH_CHECK(x_amax.scalar_type() == torch::kFloat32 && residual_amax.scalar_type() == torch::kFloat32 &&
                  x_amax.numel() == 1 && residual_amax.numel() == 1,
              "amax tensors must be scalar FP32");
  TORCH_CHECK(x_data.is_contiguous() && x_scales.is_contiguous() && residual_data.is_contiguous() &&
                  residual_scales.is_contiguous() && z.is_contiguous() && u.is_contiguous() && output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(x_data.dim() == 2 && residual_data.dim() == 2 && z.dim() == 2 && u.dim() == 2 && output.dim() == 2,
              "all matrices must be rank-2");
  TORCH_CHECK(x_data.size(0) == z.size(0) && residual_data.size(0) == u.size(0) &&
                  z.size(1) == 64 && u.size(1) == 64,
              "expected X[M,*], residual[N,*], Z[M,64], U[N,64]");
  TORCH_CHECK(output.size(0) == z.size(0) && output.size(1) == u.size(0),
              "output must be [M,N]");
  TORCH_CHECK(x_data.size(0) % 128 == 0 && residual_data.size(0) % 128 == 0,
              "functional prototype requires M and N divisible by 128");
  fused_residual_lowrank_bf16_cuda(
      x_data, x_scales, residual_data, residual_scales, x_amax, residual_amax, z, u, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_residual_lowrank_bf16", &fused_residual_lowrank_bf16,
        "Experimental single-store residual FP4 + low-rank BF16 fusion (CUDA)");
}
