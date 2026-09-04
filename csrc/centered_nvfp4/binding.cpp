// Copyright (c) 2026. Project-local centered NVFP4 inference kernel.
#include <torch/extension.h>

void quantize_centered_rowwise_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor rowwise_data,
    torch::Tensor rowwise_scale_inv,
    torch::Tensor amax);

void fused_mean_and_quantize_centered_rowwise_cuda(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor rowwise_data,
    torch::Tensor rowwise_scale_inv,
    torch::Tensor amax,
    torch::Tensor partial_sum,
    torch::Tensor partial_min,
    torch::Tensor partial_max,
    torch::Tensor block_amax);

void quantize_centered_rowwise(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor rowwise_data,
    torch::Tensor rowwise_scale_inv,
    torch::Tensor amax) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(), "input must be contiguous [M, K]");
  TORCH_CHECK(mean.scalar_type() == torch::kBFloat16 && mean.is_contiguous(), "mean must be contiguous BF16");
  TORCH_CHECK(mean.dim() == 2 && mean.size(0) == 1 && mean.size(1) == input.size(1),
              "mean must have shape [1, K]");
  TORCH_CHECK(input.size(0) % 32 == 0 && input.size(1) % 32 == 0,
              "centered NVFP4 v1 requires M and K divisible by 32");
  TORCH_CHECK(rowwise_data.scalar_type() == torch::kUInt8, "rowwise_data must be uint8");
  TORCH_CHECK(rowwise_scale_inv.scalar_type() == torch::kUInt8, "rowwise_scale_inv must be uint8");
  TORCH_CHECK(amax.scalar_type() == torch::kFloat32 && amax.numel() == 1, "amax must be float32[1]");
  quantize_centered_rowwise_cuda(input, mean, rowwise_data, rowwise_scale_inv, amax);
}

void fused_mean_and_quantize_centered_rowwise(
    torch::Tensor input,
    torch::Tensor mean,
    torch::Tensor rowwise_data,
    torch::Tensor rowwise_scale_inv,
    torch::Tensor amax) {
  TORCH_CHECK(input.is_cuda(), "input must be CUDA");
  TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "input must be BF16");
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous(), "input must be contiguous [M, K]");
  TORCH_CHECK(mean.scalar_type() == torch::kBFloat16 && mean.is_contiguous(), "mean must be contiguous BF16");
  TORCH_CHECK(mean.dim() == 2 && mean.size(0) == 1 && mean.size(1) == input.size(1),
              "mean must have shape [1, K]");
  TORCH_CHECK(input.size(0) % 32 == 0 && input.size(1) % 32 == 0,
              "centered NVFP4 v1 requires M and K divisible by 32");
  TORCH_CHECK(rowwise_data.scalar_type() == torch::kUInt8, "rowwise_data must be uint8");
  TORCH_CHECK(rowwise_scale_inv.scalar_type() == torch::kUInt8, "rowwise_scale_inv must be uint8");
  TORCH_CHECK(amax.scalar_type() == torch::kFloat32 && amax.numel() == 1, "amax must be float32[1]");

  constexpr int64_t kRowsPerPartial = 256;
  const int64_t partial_rows = (input.size(0) + kRowsPerPartial - 1) / kRowsPerPartial;
  const auto stats_options = input.options().dtype(torch::kFloat32);
  const auto stats_shape = std::vector<int64_t>{partial_rows, input.size(1)};
  auto partial_sum = torch::empty(stats_shape, stats_options);
  auto partial_min = torch::empty(stats_shape, stats_options);
  auto partial_max = torch::empty(stats_shape, stats_options);
  constexpr int64_t kColumnsPerFinalizeBlock = 128;
  const int64_t num_finalize_blocks =
      (input.size(1) + kColumnsPerFinalizeBlock - 1) / kColumnsPerFinalizeBlock;
  auto block_amax = torch::empty({num_finalize_blocks}, stats_options);
  fused_mean_and_quantize_centered_rowwise_cuda(
      input, mean, rowwise_data, rowwise_scale_inv, amax,
      partial_sum, partial_min, partial_max, block_amax);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize_centered_rowwise", &quantize_centered_rowwise,
        "Centered rowwise NVFP4 quantization (CUDA)");
  m.def("fused_mean_and_quantize_centered_rowwise",
        &fused_mean_and_quantize_centered_rowwise,
        "Fused mean/amax centered rowwise NVFP4 quantization (CUDA)");
}
