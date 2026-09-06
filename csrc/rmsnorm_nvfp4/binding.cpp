// Copyright (c) 2026. Project-local RMSNorm + centered NVFP4 binding.
#include <torch/extension.h>

void fused_rmsnorm_mean_and_quantize_centered_rowwise_cuda(
    torch::Tensor, torch::Tensor, double, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor);

void fused_rmsnorm_mean_and_quantize_centered_rowwise(
    torch::Tensor input, torch::Tensor rms_weight, double eps, torch::Tensor mean,
    torch::Tensor rowwise_data, torch::Tensor rowwise_scale_inv, torch::Tensor amax) {
  TORCH_CHECK(input.is_cuda() && input.scalar_type() == torch::kBFloat16 &&
                  input.dim() == 2 && input.is_contiguous(),
              "input must be contiguous CUDA BF16 [M,H]");
  TORCH_CHECK(rms_weight.is_cuda() && rms_weight.device() == input.device() &&
                  rms_weight.scalar_type() == torch::kBFloat16 &&
                  rms_weight.dim() == 1 && rms_weight.numel() == input.size(1) &&
                  rms_weight.is_contiguous(),
              "rms_weight must be contiguous CUDA BF16 [H] on input device");
  TORCH_CHECK(input.size(0) % 32 == 0 && input.size(1) % 32 == 0 &&
                  input.size(1) <= 4096,
              "RMSNorm NVFP4 v1 requires M/H divisible by 32 and H <= 4096");
  TORCH_CHECK(mean.device() == input.device() && mean.scalar_type() == torch::kBFloat16 &&
                  mean.is_contiguous() && mean.sizes() == torch::IntArrayRef({1, input.size(1)}),
              "mean must be contiguous BF16 [1,H] on input device");
  TORCH_CHECK(rowwise_data.device() == input.device() && rowwise_data.scalar_type() == torch::kUInt8 &&
                  rowwise_data.is_contiguous() && rowwise_data.sizes() == torch::IntArrayRef({input.size(0), input.size(1) / 2}),
              "rowwise_data has invalid TE NVFP4 shape");
  TORCH_CHECK(rowwise_scale_inv.device() == input.device() && rowwise_scale_inv.scalar_type() == torch::kUInt8 &&
                  rowwise_scale_inv.is_contiguous(), "rowwise_scale_inv must be contiguous uint8");
  TORCH_CHECK(amax.device() == input.device() && amax.scalar_type() == torch::kFloat32 &&
                  amax.numel() == 1, "amax must be float32[1] on input device");
  constexpr int64_t kRowsPerPartial = 256;
  const int64_t partial_rows = (input.size(0) + kRowsPerPartial - 1) / kRowsPerPartial;
  const auto fp32_options = input.options().dtype(torch::kFloat32);
  auto inv_rms = torch::empty({input.size(0)}, fp32_options);
  auto partial_sum = torch::empty({partial_rows, input.size(1)}, fp32_options);
  auto partial_min = torch::empty({partial_rows, input.size(1)}, fp32_options);
  auto partial_max = torch::empty({partial_rows, input.size(1)}, fp32_options);
  auto block_amax = torch::empty({(input.size(1) + 127) / 128}, fp32_options);
  fused_rmsnorm_mean_and_quantize_centered_rowwise_cuda(
      input, rms_weight, eps, mean, rowwise_data, rowwise_scale_inv, amax,
      inv_rms, partial_sum, partial_min, partial_max, block_amax);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_rmsnorm_mean_and_quantize_centered_rowwise",
        &fused_rmsnorm_mean_and_quantize_centered_rowwise,
        "RMSNorm + mean + centered rowwise NVFP4 pack (CUDA)");
}
