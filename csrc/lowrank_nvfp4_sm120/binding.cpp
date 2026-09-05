// SM120 NVFP4 rank-64 GEMM entry point for PyTorch.
#include <torch/extension.h>

void lowrank_nvfp4_add_sm120_cuda(torch::Tensor a_data, torch::Tensor a_scales,
                                  torch::Tensor b_data, torch::Tensor b_scales,
                                  torch::Tensor a_amax, torch::Tensor b_amax,
                                  torch::Tensor output);

void lowrank_nvfp4_add_sm120(torch::Tensor a_data, torch::Tensor a_scales,
                             torch::Tensor b_data, torch::Tensor b_scales,
                             torch::Tensor a_amax, torch::Tensor b_amax,
                             torch::Tensor output) {
  TORCH_CHECK(a_data.is_cuda() && a_scales.is_cuda() && b_data.is_cuda() &&
                  b_scales.is_cuda() && a_amax.is_cuda() && b_amax.is_cuda() &&
                  output.is_cuda(),
              "all tensors must be CUDA tensors");
  TORCH_CHECK(a_data.scalar_type() == torch::kUInt8 &&
                  b_data.scalar_type() == torch::kUInt8,
              "NVFP4 packed data tensors must be uint8");
  // TE stores unswizzled scales as uint8 but exposes its GEMM-swizzled scale
  // allocation as torch.float8_e4m3fn. Both representations are one byte;
  // this kernel intentionally accepts only the latter because its CUTLASS
  // layout consumes GEMM-swizzled E4M3 factors.
  TORCH_CHECK(a_scales.scalar_type() == at::kFloat8_e4m3fn &&
                  b_scales.scalar_type() == at::kFloat8_e4m3fn,
              "scales must be GEMM-swizzled torch.float8_e4m3fn tensors");
  TORCH_CHECK(output.scalar_type() == torch::kBFloat16,
              "output must be BF16");
  TORCH_CHECK(a_amax.scalar_type() == torch::kFloat32 &&
                  b_amax.scalar_type() == torch::kFloat32 &&
                  a_amax.numel() == 1 && b_amax.numel() == 1,
              "amax tensors must be scalar FP32 CUDA tensors");
  TORCH_CHECK(a_data.is_contiguous() && a_scales.is_contiguous() &&
                  b_data.is_contiguous() && b_scales.is_contiguous() &&
                  output.is_contiguous(),
              "all tensors must be contiguous");
  TORCH_CHECK(a_data.dim() == 2 && a_scales.dim() == 2 && b_data.dim() == 2 &&
                  b_scales.dim() == 2 && output.dim() == 2,
              "all tensors must be rank-2");

  constexpr int64_t kRank = 64;
  const auto m = a_data.size(0);
  const auto n = b_data.size(0);
  TORCH_CHECK(a_data.size(1) * 2 == kRank && b_data.size(1) * 2 == kRank,
              "this kernel requires rank-64 packed FP4 operands");
  TORCH_CHECK(a_scales.size(0) == m && a_scales.size(1) == kRank / 16 &&
                  b_scales.size(0) == n && b_scales.size(1) == kRank / 16,
              "expected GEMM-swizzled NVFP4 scales shaped [rows, 4]");
  TORCH_CHECK(output.size(0) == m && output.size(1) == n,
              "output shape must be [A.rows, B.rows]");
  TORCH_CHECK(m % 128 == 0 && n % 128 == 0,
              "SM120 reference kernel requires M and N divisible by 128");
  lowrank_nvfp4_add_sm120_cuda(
      a_data, a_scales, b_data, b_scales, a_amax, b_amax, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lowrank_nvfp4_add_sm120", &lowrank_nvfp4_add_sm120,
        "SM120 block-scaled NVFP4 rank-64 C += A @ B.T (CUDA)");
}
