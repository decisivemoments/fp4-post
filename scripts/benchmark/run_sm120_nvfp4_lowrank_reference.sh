#!/usr/bin/env bash
# Build and run CUTLASS's official RTX 50 / SM120 NVFP4-BF16 GEMM reference.
#
# This is a reproducible architecture baseline for the rank-64 low-rank
# product.  It intentionally benchmarks a standalone GEMM with beta=1; the
# project target remains the future dual-mainloop kernel that avoids the
# intermediate residual-output write/read pair.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUTLASS_ROOT="${CUTLASS_ROOT:-${PROJECT_ROOT}/.deps/cutlass}"
BUILD_DIR="${CUTLASS_BUILD_DIR:-${PROJECT_ROOT}/.deps/cutlass-build}"
ROWS="${ROWS:-131072}"
COLUMNS="${COLUMNS:-4864}"
RANK="${RANK:-64}"
ITERATIONS="${ITERATIONS:-50}"

if [[ ! -f "${CUTLASS_ROOT}/CMakeLists.txt" ]]; then
  echo "CUTLASS checkout not found at ${CUTLASS_ROOT}." >&2
  echo "Clone NVIDIA/cutlass there or set CUTLASS_ROOT." >&2
  exit 2
fi

cmake -S "${CUTLASS_ROOT}" -B "${BUILD_DIR}" \
  -DCUTLASS_NVCC_ARCHS=120a \
  -DCUTLASS_ENABLE_TESTS=OFF \
  -DCUTLASS_ENABLE_EXAMPLES=ON
cmake --build "${BUILD_DIR}" \
  --target 79a_blackwell_geforce_nvfp4_bf16_gemm -j"${BUILD_JOBS:-2}"

exec "${BUILD_DIR}/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm" \
  --m="${ROWS}" --n="${COLUMNS}" --k="${RANK}" \
  --alpha=1 --beta=1 --iterations="${ITERATIONS}"
