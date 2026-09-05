// Experimental SM120 epilogue leaf for Y = alpha * FP4_GEMM + Z @ U.T.
//
// The rank-64 correction is generated as cooperative 16x16 BF16 tensor-core
// tiles in the epilogue.  It proves the single-D-store dataflow without
// materializing a full residual output matrix.
#pragma once

#include <cuda_bf16.h>
#include <mma.h>

#include "cute/tensor.hpp"
#include "cutlass/array.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"

namespace metis::fused_epilogue {

using namespace cute;
using namespace cutlass::epilogue::fusion;
using EmptyArguments = typename Sm90VisitorImpl<>::Arguments;

template <class ScratchLayout>
struct LowrankBf16Dot : Sm90VisitorImpl<> {
  using Element = cutlass::bfloat16_t;

  // One current epilogue subtile in the exact D shared-memory swizzle.
  struct SharedStorage {
    alignas(cutlass::detail::alignment_for_swizzle(ScratchLayout{}))
    Element lowrank[cute::cosize_v<ScratchLayout>];
    alignas(16) float lowrank_accum[64 * 32];
  };
  struct Arguments {
    Element const* z = nullptr;  // [M, 64], row major
    Element const* u = nullptr;  // [N, 64], row major
    int64_t ldz = 64;
    int64_t ldu = 64;
  };
  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(
      ProblemShape const&, Arguments const& args, void*) {
    return args;
  }
  template <class ProblemShape>
  static bool can_implement(ProblemShape const&, Arguments const& args) {
    return args.z != nullptr && args.u != nullptr;
  }
  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const&, Arguments const&) {
    return 0;
  }
  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const&, Arguments const&, void*, cudaStream_t,
      cutlass::CudaHostAdapter* = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool is_producer_load_needed() const { return false; }
  CUTLASS_DEVICE bool is_C_load_needed() const { return false; }
  CUTLASS_HOST_DEVICE LowrankBf16Dot() = default;
  CUTLASS_HOST_DEVICE LowrankBf16Dot(Params const& params, SharedStorage const& storage)
      : params_ptr(&params), smem_lowrank(const_cast<Element*>(storage.lowrank)),
        smem_lowrank_accum(const_cast<float*>(storage.lowrank_accum)) {}
  Params const* params_ptr = nullptr;
  Element* smem_lowrank = nullptr;
  float* smem_lowrank_accum = nullptr;

  template <class... Args>
  CUTLASS_DEVICE auto get_producer_load_callbacks(ProducerLoadArgs<Args...> const&) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class RTensor, class TiledS2R, class STensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE ConsumerStoreCallbacks(
        RTensor const& r_lowrank, TiledS2R const& tiled_s2r, STensor const& s_lowrank,
        Params const* params, Element* smem, float* smem_accum,
        int tile_m, int tile_n, int thread_idx, int participants)
        : r_lowrank(r_lowrank), tiled_s2r(tiled_s2r), s_lowrank(s_lowrank),
          params_ptr(params), smem_lowrank(smem), smem_lowrank_accum(smem_accum), tile_m(tile_m), tile_n(tile_n),
          thread_idx(thread_idx), participants(participants) {}

    RTensor r_lowrank;
    TiledS2R tiled_s2r;
    STensor s_lowrank;
    Params const* params_ptr;
    Element* smem_lowrank;
    float* smem_lowrank_accum;
    int tile_m, tile_n, thread_idx, participants;

    CUTLASS_DEVICE void begin_loop(int epi_m, int epi_n) {
      int64_t row_base = int64_t(tile_m) * 128 + int64_t(epi_m) * 64;
      int64_t col_base = int64_t(tile_n) * 128 + int64_t(epi_n) * 32;
      // Eight warps compute the [64,32] tile as 16x16 BF16 WMMA fragments.
      // U's physical [N,64] row-major storage is a [64,N] column-major B
      // operand, so no transpose materialization is introduced. These
      // operands deliberately come directly from global memory: buffering
      // them here costs 12 KiB of mainloop pipeline storage plus an extra CTA
      // synchronization, which is slower on the FP4 GEMM critical path.
      int warp = thread_idx / 32;
      if (warp < 8) {
        int warp_m = warp / 2;
        int warp_n = warp % 2;
        namespace wmma = nvcuda::wmma;
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (int k = 0; k < 64; k += 16) {
          wmma::load_matrix_sync(a,
              reinterpret_cast<const __nv_bfloat16*>(params_ptr->z) +
                  (row_base + warp_m * 16) * params_ptr->ldz + k,
              64);
          wmma::load_matrix_sync(b,
              reinterpret_cast<const __nv_bfloat16*>(params_ptr->u) +
                  (col_base + warp_n * 16) * params_ptr->ldu + k,
              64);
          wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(
            smem_lowrank_accum + warp_m * 16 * 32 + warp_n * 16,
            acc, 32, wmma::mem_row_major);
      }
      cutlass::arch::NamedBarrier::sync(
          participants, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
      Tensor scratch = make_tensor(make_smem_ptr(smem_lowrank), ScratchLayout{});
      for (int linear = thread_idx; linear < 64 * 32; linear += participants) {
        int local_m = linear / 32;
        int local_n = linear % 32;
        scratch(local_m, local_n) = Element(smem_lowrank_accum[linear]);
      }
      cutlass::arch::NamedBarrier::sync(
          participants, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
    }

    CUTLASS_DEVICE void previsit(int, int, int, bool) {
      using RLayoutS2R = decltype(cute::layout(TiledS2R{}.get_slice(0).retile_S(RTensor{})));
      Tensor tSR_rLowrank = make_tensor(r_lowrank.data(), RLayoutS2R{});
      copy(tiled_s2r, s_lowrank, tSR_rLowrank);
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE cutlass::Array<float, FragmentSize> visit(
        cutlass::Array<ElementAccumulator, FragmentSize> const&, int epi_v,
        int epi_m, int epi_n) {
      Tensor r_frg = recast<cutlass::Array<Element, FragmentSize>>(coalesce(r_lowrank));
      auto lowrank = r_frg(epi_v);
      cutlass::Array<float, FragmentSize> result;
#pragma unroll
      for (int i = 0; i < FragmentSize; ++i) result[i] = float(lowrank[i]);
      return result;
    }
  };

  template <bool ReferenceSrc, class... Args>
  CUTLASS_DEVICE auto get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [tile_m, tile_n, tile_k, tile_l] = args.tile_coord_mnkl;
    Tensor s_lowrank = make_tensor(make_smem_ptr(smem_lowrank), ScratchLayout{});
    // This layout is exactly the final D swizzle, so LDSM may use the same
    // vector ownership as the native epilogue path.
    auto tiled_s2r = conditional_return<ReferenceSrc>(
        make_tiled_copy_S(Copy_Atom<SM75_U32x2_LDSM_N, Element>{}, args.tiled_copy),
        make_tiled_copy_D(Copy_Atom<SM75_U32x2_LDSM_N, Element>{}, args.tiled_copy));
    auto tSR_sLowrank = tiled_s2r.get_slice(args.thread_idx).partition_S(s_lowrank);
    Tensor r_lowrank = make_tensor<Element>(take<0, 3>(shape(args.tCcD)));
    return ConsumerStoreCallbacks<decltype(r_lowrank), decltype(tiled_s2r), decltype(tSR_sLowrank)>(
        r_lowrank, tiled_s2r, tSR_sLowrank, params_ptr, smem_lowrank, smem_lowrank_accum,
        tile_m, tile_n, args.thread_idx, size(args.tiled_mma));
  }
};

using ScalarAlpha = Sm90ScalarBroadcastPtrArray<float, Stride<_0, _0, int64_t>>;
using ScaledAccumulator = Sm90EVT<
    Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
    ScalarAlpha, Sm90AccFetch>;
template <class ScratchLayout>
using Tree = Sm90EVT<
    Sm90Compute<cutlass::plus, cutlass::bfloat16_t, float, cutlass::FloatRoundStyle::round_to_nearest>,
    ScaledAccumulator,
    LowrankBf16Dot<ScratchLayout>>;

// Expose a flat host argument type while retaining CUTLASS's tree internally.
template <class ScratchLayout>
struct Callbacks : Tree<ScratchLayout> {
  using Impl = Tree<ScratchLayout>;
  using Impl::Impl;
  struct Arguments {
    float alpha = 1.0f;
    float const* alpha_ptr = nullptr;
    cutlass::bfloat16_t const* z = nullptr;
    cutlass::bfloat16_t const* u = nullptr;
    int64_t ldz = 64;
    int64_t ldu = 64;

    operator typename Impl::Arguments() const {
      typename ScalarAlpha::Arguments alpha_args{};
      alpha_args.scalars[0] = alpha;
      alpha_args.scalar_ptrs[0] = alpha_ptr;
      alpha_args.dScalar[0] = Stride<_0, _0, int64_t>{_0{}, _0{}, 0};
      return typename Impl::Arguments{
          typename ScaledAccumulator::Arguments{alpha_args, EmptyArguments{}, EmptyArguments{}},
          typename LowrankBf16Dot<ScratchLayout>::Arguments{z, u, ldz, ldu},
          EmptyArguments{}};
    }
  };
};

}  // namespace metis::fused_epilogue
