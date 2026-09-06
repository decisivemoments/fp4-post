#include <torch/extension.h>
#include <limits>

void fused_residual_lowrank_nvfp4_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);

void fused_residual_lowrank_nvfp4(
    torch::Tensor x, torch::Tensor xs, torch::Tensor r, torch::Tensor rs,
    torch::Tensor xa, torch::Tensor ra,
    torch::Tensor z, torch::Tensor zs, torch::Tensor u, torch::Tensor us,
    torch::Tensor za, torch::Tensor ua, torch::Tensor correction, torch::Tensor output) {
  TORCH_CHECK(output.is_cuda() && output.scalar_type() == torch::kBFloat16 &&
              output.dim() == 2 && output.is_contiguous(),
              "output must be a contiguous rank-2 CUDA BF16 tensor");
  TORCH_CHECK(correction.device() == output.device() &&
              correction.scalar_type() == torch::kBFloat16 &&
              correction.dim() == 1 && correction.size(0) == output.size(1) &&
              correction.is_contiguous() && !correction.is_alias_of(output),
              "residual_mean_correction must be contiguous CUDA BF16 [N] on output device, without output alias");
  auto check_operand = [&](torch::Tensor data, torch::Tensor scales, torch::Tensor amax) {
    for (auto t : {data, scales, amax}) {
      TORCH_CHECK(t.device() == output.device() && t.is_contiguous(),
                  "all inputs must be contiguous and on the output CUDA device");
      TORCH_CHECK(!t.is_alias_of(output), "output must not alias an input");
    }
    TORCH_CHECK(data.scalar_type() == torch::kUInt8 && data.dim() == 2,
                "packed data must be rank-2 uint8");
    TORCH_CHECK(scales.scalar_type() == at::kFloat8_e4m3fn && scales.dim() == 2,
                "scales must be rank-2 GEMM-swizzled float8_e4m3fn");
    TORCH_CHECK(amax.scalar_type() == torch::kFloat32 && amax.numel() == 1,
                "amax must be scalar FP32");
    TORCH_CHECK(data.size(0) > 0 && data.size(0) % 128 == 0 &&
                data.size(0) <= std::numeric_limits<int>::max(),
                "operand rows must be positive, int32-sized multiples of 128");
    TORCH_CHECK(data.size(1) > 0 && data.size(1) % 32 == 0 &&
                data.size(1) <= std::numeric_limits<int>::max() / 2,
                "logical K must be a positive int32-sized multiple of 64");
    // NVFP4 scale layout pads K to groups of four 16-element blocks.
    TORCH_CHECK(scales.size(0) == data.size(0) && scales.size(1) == data.size(1) / 8,
                "invalid GEMM-swizzled scale shape");
  };
  check_operand(x, xs, xa); check_operand(r, rs, ra);
  check_operand(z, zs, za); check_operand(u, us, ua);
  TORCH_CHECK(x.size(1) == r.size(1), "X and R K dimensions must agree");
  TORCH_CHECK(z.size(1) == 32 && u.size(1) == 32, "low-rank operands require rank 64");
  TORCH_CHECK(x.size(0) == z.size(0) && r.size(0) == u.size(0), "low-rank tile rows must match residual");
  TORCH_CHECK(output.size(0) == x.size(0) && output.size(1) == r.size(0), "output must have shape [M,N]");
  fused_residual_lowrank_nvfp4_cuda(x, xs, r, rs, xa, ra, z, zs, u, us, za, ua, correction, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_residual_lowrank_nvfp4", &fused_residual_lowrank_nvfp4,
        "SM120 dual NVFP4 GEMM with one final BF16 D store");
}
