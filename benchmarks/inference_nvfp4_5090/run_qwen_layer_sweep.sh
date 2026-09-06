#!/usr/bin/env bash
# Example: MODEL_PATH_0_5B=/models/qwen05 bash benchmarks/inference_nvfp4_5090/run_qwen_layer_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# This script is intended to run directly on the host.  Keep the interpreter,
# Transformer Engine, and the CUDA toolkit consistent with the project setup.
source "${ROOT}/../conda-init.sh"

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs/inference_nvfp4_5090}"
SEQ_LENGTH="${SEQ_LENGTH:-512}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16 32 64 128 256}"
WARMUP="${WARMUP:-20}"
ITERATIONS="${ITERATIONS:-100}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3,4,5}"

# Phase 1: compare a known fast shape (down_proj) with a known slow shape
# (up_proj), both at a large enough M to make the contrast visible.  This is
# deliberately opt-in: an Nsight trace is much slower than the normal sweep.
RUN_SWEEP="${RUN_SWEEP:-1}"
RUN_PHASE1_PROFILE="${RUN_PHASE1_PROFILE:-1}"
RUN_LOWRANK_BENCHMARK="${RUN_LOWRANK_BENCHMARK:-0}"
RUN_LOWRANK_VALIDATION="${RUN_LOWRANK_VALIDATION:-0}"
PROFILE_BATCH_SIZE="${PROFILE_BATCH_SIZE:-256}"
PROFILE_WARMUP="${PROFILE_WARMUP:-5}"
PROFILE_ITERATIONS="${PROFILE_ITERATIONS:-5}"
PROFILE_PROJECTIONS="${PROFILE_PROJECTIONS:-down_proj up_proj}"
# "rowwise" omits backward-only columnwise activation packing.  Keep full
# first so a failed experimental TE layout does not hide the baseline trace.
PROFILE_ACTIVATION_LAYOUTS="${PROFILE_ACTIVATION_LAYOUTS:-rowwise}"
LOWRANK_COMPUTE="${LOWRANK_COMPUTE:-nvfp4}"
FUSE_RMSNORM_QUANT="${FUSE_RMSNORM_QUANT:-0}"
NSYS_BIN="${NSYS_BIN:-nsys}"
QUANT_BACKEND="${QUANT_BACKEND:-centered_cuda}"
# Nsight Systems GPU metrics are optional: unlike the CUDA/NVTX timeline they
# may require counter access and add sampling overhead.  Set e.g.
# NSYS_GPU_METRICS_DEVICE=0 to opt in for the visible CUDA device 0.
NSYS_GPU_METRICS_DEVICE="${NSYS_GPU_METRICS_DEVICE:-all}"

run_lowrank_benchmark() {
  local model="$1"
  local model_path="$2"
  local projection
  local path_args=()

  if [[ -n "${model_path}" ]]; then
    path_args=(--model-path "${model_path}")
  fi
  for projection in ${LOWRANK_PROJECTIONS:-up_proj down_proj}; do
    PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      python "${ROOT}/benchmarks/inference_nvfp4_5090/benchmark_lowrank_nvfp4.py" \
        --model "${model}" "${path_args[@]}" \
        --projection "${projection}" \
        --batch-sizes ${LOWRANK_BATCH_SIZES:-16 32 64 128 256} \
        --seq-length "${SEQ_LENGTH}" \
        --warmup "${WARMUP}" --iterations "${ITERATIONS}" \
        --output "${OUTPUT_ROOT}/lowrank_${model}_${projection}.json"
  done
}

run_lowrank_validation() {
  local model="$1"
  local model_path="$2"
  local projection
  local path_args=()

  if [[ -n "${model_path}" ]]; then
    path_args=(--model-path "${model_path}")
  fi
  for projection in ${LOWRANK_PROJECTIONS:-up_proj down_proj}; do
    PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      python "${ROOT}/benchmarks/inference_nvfp4_5090/validate_lowrank_nvfp4.py" \
        --model "${model}" "${path_args[@]}" \
        --projection "${projection}" \
        --batch-sizes ${LOWRANK_VALIDATION_BATCH_SIZES:-1 16 64 128} \
        --seq-length "${SEQ_LENGTH}" \
        --output "${OUTPUT_ROOT}/lowrank_validation_${model}_${projection}.json"
  done
}

run_phase1_profile() {
  local model="$1"
  local model_path="$2"
  local projection
  local activation_layout
  local profile_root="${OUTPUT_ROOT}/phase1_profile/${model}"
  local path_args=()
  local nsys_metric_args=()

  if [[ -n "${model_path}" ]]; then
    path_args=(--model-path "${model_path}")
  fi
  command -v "${NSYS_BIN}" >/dev/null || {
    echo "Nsight Systems executable not found: ${NSYS_BIN}" >&2
    return 127
  }
  if [[ -n "${NSYS_GPU_METRICS_DEVICE}" ]]; then
    nsys_metric_args=(--gpu-metrics-device="${NSYS_GPU_METRICS_DEVICE}")
  fi
  mkdir -p "${profile_root}"

  for activation_layout in ${PROFILE_ACTIVATION_LAYOUTS}; do
    for projection in ${PROFILE_PROJECTIONS}; do
      echo "Phase 1 profile: model=${model} projection=${projection} layout=${activation_layout} batch=${PROFILE_BATCH_SIZE}"
      PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${NSYS_BIN}" profile \
          --force-overwrite=true \
          --trace=cuda,nvtx,osrt \
          --capture-range=cudaProfilerApi \
          --capture-range-end=stop \
          "${nsys_metric_args[@]}" \
          --output "${profile_root}/${projection}_${activation_layout}_b${PROFILE_BATCH_SIZE}_${QUANT_BACKEND}_fused" \
        python "${ROOT}/benchmarks/inference_nvfp4_5090/benchmark_qwen_inference.py" \
          --scope linear --model "${model}" "${path_args[@]}" \
          --modes nvfp4 --projections "${projection}" \
          --batch-sizes "${PROFILE_BATCH_SIZE}" --seq-length "${SEQ_LENGTH}" \
          --activation-mode fresh --activation-layout "${activation_layout}" \
          --lowrank-compute "${LOWRANK_COMPUTE}" \
          --profile-native-ranges --cuda-profiler-range \
          --warmup "${PROFILE_WARMUP}" --iterations "${PROFILE_ITERATIONS}" \
          --output "${profile_root}/${projection}_${activation_layout}_b${PROFILE_BATCH_SIZE}_${QUANT_BACKEND}_fused.json" \
          --activation-pack-backend "${QUANT_BACKEND}"
    done
  done
}

# for model in qwen0.5b qwen1.5b qwen7b; do
for model in qwen0.5b; do
  case "${model}" in
    qwen0.5b) model_path="${MODEL_PATH_0_5B:-../../../qwen2.5-0.5b-instruct}" ;;
    qwen1.5b) model_path="${MODEL_PATH_1_5B:-}" ;;
    qwen7b) model_path="${MODEL_PATH_7B:-}" ;;
  esac
  path_args=()
  if [[ -n "${model_path}" ]]; then path_args=(--model-path "${model_path}"); fi
  if [[ "${RUN_SWEEP}" == "1" ]]; then
    for scope in linear transformer; do
      fuse_rmsnorm_args=()
      if [[ "${scope}" == "transformer" && "${FUSE_RMSNORM_QUANT}" == "1" ]]; then
        fuse_rmsnorm_args=(--fuse-rmsnorm-quant)
      fi
      PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" python "${ROOT}/benchmarks/inference_nvfp4_5090/benchmark_qwen_inference.py" \
        --scope "${scope}" --model "${model}" "${path_args[@]}" \
        --seq-length "${SEQ_LENGTH}" --batch-sizes ${BATCH_SIZES} \
        --lowrank-compute "${LOWRANK_COMPUTE}" \
        --warmup "${WARMUP}" --iterations "${ITERATIONS}" \
        --output "${OUTPUT_ROOT}/${scope}_${model}.json" \
        --activation-pack-backend "${QUANT_BACKEND}" \
        --activation-layout "${PROFILE_ACTIVATION_LAYOUTS}" \
        "${fuse_rmsnorm_args[@]}"
    done
  fi

  if [[ "${RUN_PHASE1_PROFILE}" == "1" ]]; then
    run_phase1_profile "${model}" "${model_path}"
  fi

  if [[ "${RUN_LOWRANK_BENCHMARK}" == "1" ]]; then
    run_lowrank_benchmark "${model}" "${model_path}"
  fi

  if [[ "${RUN_LOWRANK_VALIDATION}" == "1" ]]; then
    run_lowrank_validation "${model}" "${model_path}"
  fi
done
