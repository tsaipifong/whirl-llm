// Gated-DeltaNet kernel choice (ARCH-1 §6.5, arch025-F6): which kernels a DeltaNet layer launches - the f16-output
// (h16) speed path, the fused decode step ([beta; alpha] GEMV + conv, or the precise f16 GEMM + gates), the prefill
// gates ([beta; alpha] in one GEMM or two), the conv (+ q/k l2norm) kernel, the recurrence (sequential or chunked:
// strix / f16-WMMA / f32 chunks) and the gated norm (straight into the out-projection's fp8 / f16 input or plain).
// Pure functions of the problem (shapes, rows), the numerics flags and the code object's kernel table.
// src/model/forward/gdn.cpp builds the problem, asks here, and launches what comes back (it alone knows the buffers).
// SPDX-License-Identifier: Apache-2.0
//
// The precise rule (precise decode / verify rows take the f16 GEMM, never the int8 fused step) is preciseAllows in
// matmul.h; chooseGdnDecode asks it. Tests: tests/ops/ops_tests.cpp (testGdnChoice: two device profiles x three modes).

#pragma once

#include "whirl/ops/act.h"
#include "whirl/ops/matmul.h"

#include <cstdint>
#include <span>

namespace whirl::ops {

// the layer shape (qwen35::Config: d_conv, d_state, headV(), convCh(), n_v_heads, n_k_heads)
struct GdnShape {
    std::uint32_t d_conv = 4, d_state = 128, head_v = 128, conv_ch = 0, n_v_heads = 0, n_k_heads = 0;
};

// ---- h16: the qkv / gate GEMMs write f16 for the conv and the gated norm (MXFP4 speed mode)
struct GdnH16Problem {
    MatShape qkv, gate, out;
    std::span<const std::uint8_t> qkv_tune, gate_tune;  // the tuned choice per batch bucket (Mat::tune)
    std::uint32_t n = 0;
    GdnShape s;
    bool fused = false;  // fusedRows(n)
};
struct GdnH16Flags {
    bool ffn_h16 = false;  // numerics item h16 (Model::ffn_h16)
    bool act_fuse = true;  // WHIRL_ACT_FUSE
};
bool chooseGdnH16(const GdnH16Problem& p, const GdnH16Flags& g, const MatmulFlags& f, const H16Tune& h, const kernels::KernelTable& k);

// ---- [beta; alpha] as one GEMM (beta's tune) into ba_buf, then gdn_gates_ba
struct GdnBaProblem {
    bool gdn_ba_on = true;       // WHIRL_GDN_BA
    bool ba_buf = false;         // the [beta; alpha] output buffer exists
    bool ba_contiguous = false;  // rowsContiguous(beta, alpha)
};
bool gdnBaGemm(const GdnBaProblem& p, const kernels::DeltaNetKernels& k);

// ---- fused decode step (n <= small_max rows on the int8 GEMV path)
// the fused decode applies: fused rows, a fused [beta; alpha] GEMV for these matrices, the decode shape
bool gdnFusable(bool fused, const MatShape& beta, const MatShape& alpha, const GdnShape& s, const kernels::DeltaNetKernels& k);
enum class GdnDecodeGates : std::uint8_t {
    precise_ba,     // precise rows: [beta; alpha] f16 GEMM + gdn_gates_ba, then gdn_conv_l2
    precise_split,  // precise rows: beta / alpha f16 GEMMs + gdn_gates, then gdn_conv_l2
    abconv,         // [beta; alpha] GEMV and the conv step in one launch (f)
    ab,             // [beta; alpha] GEMV (f = gdn_ab[beta type]), then gdn_conv_l2
};
const char* gdnDecodeGatesName(GdnDecodeGates g);
struct GdnDecodePlan {
    GdnDecodeGates gates = GdnDecodeGates::ab;
    Function f = nullptr;  // abconv / ab: the kernel
};
// dbg_flags bit 0 (WHIRL_DBG&1) turns the abconv launch off
GdnDecodePlan chooseGdnDecode(GgmlType beta_ty, std::uint32_t n, std::uint32_t small_max, bool prec_dec, std::uint32_t dbg_flags,
                              const GdnBaProblem& ba, const kernels::DeltaNetKernels& k);

// ---- prefill / unfused rows
enum class GdnGates : std::uint8_t {
    in2,    // [beta; alpha] f16 GEMM straight from x16b (rmsnormIn wrote it, chooseGdnIn2), then gdn_gates_ba
    ba,     // one [beta; alpha] GEMM + gdn_gates_ba
    split,  // beta / alpha GEMMs + gdn_gates
};
const char* gdnGatesName(GdnGates g);
GdnGates chooseGdnGates(bool gdn_in2, const GdnBaProblem& ba, const kernels::DeltaNetKernels& k);

enum class GdnConv : std::uint8_t {
    seg_h16,  // segmented prefill, h16: gdn_conv_l2n_h + gdn_conv_state_h per segment
    seg_par,  // segmented prefill: gdn_conv_par + gdn_conv_state per segment
    l2n_h,    // gdn_conv_l2n_h (f16 qkv; conv + q/k l2norm)
    l2n,      // gdn_conv_l2n (conv + q/k l2norm)
    par,      // gdn_conv_par
    seq,      // gdn_conv_seq (one row, or d_conv != 4)
};
const char* gdnConvName(GdnConv c);
struct GdnConvPlan {
    GdnConv conv = GdnConv::seq;
    bool l2norm = true;    // a separate l2norm launch follows (the conv kernel did not normalize q / k)
    bool state_h = false;  // the conv-state update reads f16 qkv (l2n_h / l2n / par: gdn_conv_state_h vs gdn_conv_state)
};
GdnConvPlan chooseGdnConv(bool segmented, bool gdn_h16, std::uint32_t n, const GdnShape& s, bool act_fuse, const kernels::DeltaNetKernels& k);

// the recurrence over n rows (one sequence or one segment)
enum class GdnScan : std::uint8_t { chunked, seq };
constexpr GdnScan chooseGdnScan(bool gdn_chunked, std::uint32_t n, const GdnShape& s) {
    return gdn_chunked && n >= 64 && s.d_state == 128 && s.head_v == 128 ? GdnScan::chunked : GdnScan::seq;
}

// chunked (n >= 64) recurrence: GDN-8S strix chunks (CL = 32, on the existing scratch when it fits), the f16-WMMA
// pair, or the f32 prep + scan pair
enum class GdnChunk : std::uint8_t { strix, wmma, f32 };
const char* gdnChunkName(GdnChunk c);
struct GdnChunkProblem {
    std::uint32_t n = 0;
    std::uint64_t n_v_heads = 0, n_k_heads = 0;
    std::uint64_t gc_m_bytes = 0, gc_w_bytes = 0, gc_u_bytes = 0;  // the scratch the strix layout must fit in
};
struct GdnChunkFlags {
    bool gdn_strix = false;
    std::uint32_t gdn_strix_min = 256;
    bool gdn_wmma = false;
};
struct GdnChunkPlan {
    GdnChunk kind = GdnChunk::f32;
    // strix: chunk count and the scratch offsets (T' | P in gc_m at 0 / tp; Qh | Kh | gate factors in gc_u at 0 / qk / 2 qk)
    std::uint64_t nch = 0, tp = 0, qk = 0;
};
GdnChunkPlan chooseGdnChunk(const GdnChunkProblem& p, const GdnChunkFlags& f, const kernels::DeltaNetKernels& k);

// the gated norm: f != null -> it writes the out-projection's input (x16, + sx8 when fp8) directly; null -> gdn_gated_norm
struct GdnNormPlan {
    Function f = nullptr;
    bool fp8 = false;
};
GdnNormPlan chooseGdnGatedNorm(const MatShape& out, std::uint32_t n, const GdnShape& s, bool gdn_h16, bool act_fuse, bool g8t,
                               const MatmulFlags& f, const kernels::KernelTable& k);

}  // namespace whirl::ops
