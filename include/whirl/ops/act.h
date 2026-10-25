// Activation-input and fused-path choice (ARCH-1 §6.5, arch025-F5): which input form (int8 GEMV, f32, fp8, f16) the
// GEMMs reading one activation take, whether their outputs are f16 (h16), whether a norm writes that form directly
// (rmsnormIn, the GDN two-input norm), and the decode-fusion predicates (fused rows, fused decode, wide verify).
// Pure functions of the problem (matrix shapes, rows), the numerics flags, the device profile's fields (copied
// into the flags by the caller) and the code object's kernel table. src/model/forward/dispatch.cpp builds the
// problem, asks here, and launches what comes back.
// SPDX-License-Identifier: Apache-2.0
//
// The precise rule (precise decode / verify rows take the f16 GEMM input) is preciseRows in matmul.h; chooseActIn
// asks it first. Tests: tests/ops/ops_tests.cpp (selection table: two device profiles x three modes).

#pragma once

#include "whirl/ops/matmul.h"

#include <cstdint>
#include <optional>
#include <span>

namespace whirl::ops {

// the input form of a GEMM on one activation (Model::ActIn before F5)
enum class ActIn : std::uint8_t { gemv, f32in, fp8, f16 };
const char* actInName(ActIn a);

// input form of the matmul of n rows of w; cls: the consumer's fp8 class bit (1 attn / GDN in, 2 FFN gate+up,
// 4 FFN down; MatmulFlags::fp8_mask)
ActIn chooseActIn(const MatShape& w, std::uint32_t n, std::uint8_t cls, const MatmulFlags& f, const kernels::MatmulKernels& k);

// what the f16-output (h16) choice reads besides the matrix
struct H16Tune {
    // device::Profile::h16_from_candidates (8060S): decided from the tuner's candidate list, not the tuned choices
    bool from_candidates = false;
    std::uint64_t tune_mask = ~0ull;  // tuner candidate mask (Model::tune_mask)
    std::uint32_t n_choices = 0;      // distinct prefill GEMM choices (qwen35::n_choices)
    bool h16_dq = false;              // KG-1: a fused quantized choice runs as dequant + gemm_ch (f16 out)
};
// whether the GEMM of n rows of w can write f16 outputs; tune: w's tuned choice per batch bucket (every bucket is
// read, so the answer is the same for a request's solo chunks and a segmented forward of any size)
bool chooseH16Out(const MatShape& w, std::span<const std::uint8_t> tune, std::uint32_t n, std::uint8_t cls, const MatmulFlags& f,
                  const H16Tune& h, const kernels::MatmulKernels& k);

// the input form all consumers share when it is fp8 or f16 (a norm / activation kernel can write it directly);
// act_fuse: WHIRL_ACT_FUSE (Model::act_fuse)
std::optional<ActIn> chooseActCommon(std::span<const MatShape> consumers, std::uint32_t n, std::uint8_t cls, bool act_fuse,
                                     const MatmulFlags& f, const kernels::MatmulKernels& k);

// prefill rmsnorm of n rows of n_embd for `consumers`: f != null -> that norm writes x16 (+ sx8 when fp8) directly;
// null -> the plain rmsnorm. g8t: the fragment-tiled fp8 set (Model::useTiledFp8)
struct RmsnormInPlan {
    Function f = nullptr;
    bool fp8 = false;
};
RmsnormInPlan chooseRmsnormIn(std::uint32_t n_embd, std::span<const MatShape> consumers, std::uint32_t n, std::uint8_t cls,
                              bool act_fuse, bool g8t, const MatmulFlags& f, const kernels::KernelTable& k);

// n > 16 GDN prefill: one rmsnorm_x8h16(t) writes qkv / gate on fp8 and contiguous [beta; alpha] on f16
struct GdnIn2Problem {
    MatShape qkv, gate, beta, alpha;
    std::uint32_t n = 0, n_embd = 0;
    bool fused = false;          // fusedRows(n)
    bool ba_buf = false;         // the [beta; alpha] output buffer exists
    bool ba_contiguous = false;  // rowsContiguous(beta, alpha)
};
struct GdnIn2Flags {
    bool gdn_in2_on = true;  // WHIRL_GDN_IN2
    bool gdn_ba_on = true;   // WHIRL_GDN_BA
    bool act_fuse = true;
    bool g8t = false;
};
bool chooseGdnIn2(const GdnIn2Problem& p, const GdnIn2Flags& g, const MatmulFlags& f, const kernels::KernelTable& k);

// ---- fused series
// decode fusions apply to small batches on the int8 GEMV path (Model::fused)
constexpr bool fusedRows(std::uint32_t n, std::uint32_t small_max, bool float_gemv, bool no_fuse) {
    return n <= small_max && !float_gemv && !no_fuse;
}
// the fused [beta; alpha] decode GEMV exists for these two matrices
bool gdnAbFusable(const MatShape& beta, const MatShape& alpha, const kernels::DeltaNetKernels& k);
// the DeltaNet shape the fused decode kernels are built for
constexpr bool gdnDecodeShape(std::uint32_t d_conv, std::uint32_t d_state, std::uint32_t head_v, std::uint32_t conv_ch) {
    return d_conv == 4 && d_state == 128 && head_v == 128 && conv_ch % 128 == 0;
}
// wide (17..32-row) batched verify = wideAllowed && fused decode && wideKernels (Model::wideCapable asks in that
// order, so the per-layer fused-decode walk only runs when the mode allows wide verifies)
// the mode: MoE only in balance / fast with WHIRL_MOE_WIDE=1 (precise MoE keeps 16-row verifies)
constexpr bool wideAllowed(bool wide_verify, bool moe, bool moe_wide, bool prec_dec) { return wide_verify && (!moe || (moe_wide && !prec_dec)); }
// the code object: the 17..32-token GEMV for q4_k
bool wideKernels(const kernels::MatmulKernels& k);

}  // namespace whirl::ops
