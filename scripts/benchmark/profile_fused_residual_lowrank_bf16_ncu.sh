#!/usr/bin/env bash
# Profile the experimental single-store FP4 residual + BF16 rank-64 fusion.
#
# Run from the host:
#   bash scripts/benchmark/profile_fused_residual_lowrank_bf16_ncu.sh
#
# Optional environment overrides:
#   NCU_SET=full ROWS=8192 WARMUP=3 ITERATIONS=10 \
#     bash scripts/benchmark/profile_fused_residual_lowrank_bf16_ncu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONDA_INIT="${PROJECT_ROOT}/../conda-init.sh"

# Match the GPU used by the existing Nsight Systems records.  Override this
# when reserving a different RTX 5090.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROWS="${ROWS:-131072}"
INPUT_FEATURES="${INPUT_FEATURES:-896}"
OUTPUT_FEATURES="${OUTPUT_FEATURES:-4864}"
RANK="${RANK:-64}"
WARMUP="${WARMUP:-3}"
ITERATIONS="${ITERATIONS:-10}"
NCU_SET="${NCU_SET:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${PROJECT_ROOT}/outputs/inference_nvfp4_5090/profiles/fused_residual_lowrank_bf16"
REPORT="${REPORT_DIR}/ncu_m${ROWS}_n${OUTPUT_FEATURES}_k${INPUT_FEATURES}_r${RANK}_${STAMP}"

if [[ ! -f "${CONDA_INIT}" ]]; then
  echo "Missing conda activation script: ${CONDA_INIT}" >&2
  exit 1
fi

source "${CONDA_INIT}"
mkdir -p "${REPORT_DIR}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NCU report: ${REPORT}.ncu-rep"

cd "${PROJECT_ROOT}"
ncu \
  --target-processes all \
  --set "${NCU_SET}" \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*GemmUniversal.*' \
  --launch-count 1 \
  --export "${REPORT}" \
  --force-overwrite \
  python benchmarks/inference_nvfp4_5090/benchmark_fused_residual_lowrank_bf16.py \
    --rows "${ROWS}" \
    --input-features "${INPUT_FEATURES}" \
    --output-features "${OUTPUT_FEATURES}" \
    --rank "${RANK}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}"
