// MoE expert kernel choice (ARCH-1 K-4): decode experts (int8 moe_gu / moe_down, their MXFP4 whole-block twins,
// the precise f32-activation moe_gu_f / moe_down_f) and prefill grouped GEMMs (MXFP4 x fp8, the 8060S fused
// gate+up+SwiGLU, the gfx1201 f16 MXFP4 gemmr_moe*, the generic gemm_moe), as a pure function of the problem, the
// flags and the kernel table. src/model/qwen35/forward/ffn.cpp (Model::moeBlock) launches the plan.
// SPDX-License-Identifier: Apache-2.0
//
// Bit classes: precise decode experts take f32 activations, a token's result independent of its batch
// (moe_f32_row, the precise class: precise rows up to max_verify_rows always get it); balance / fast decode
// experts take int8 activations (moe_q8_row, every row its own one-token decode); the prefill grouped GEMMs are
// bit-identical within a numerics class (moe_f16_gemm: fused / gemmr / generic; moe_fp8_gemm).
// moeskip (fast, v0252-f5 from feat027): MoePlan::skip - the block's routing drops slots (id -1, kept weights renormalized)
// and every kernel of its path is a skip entry point; the per-row arithmetic class of the kept pairs is the plan's bits.

#pragma once

#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"

#include <cstdint>

namespace whirl::ops {

using gguf::GgmlType;
using hip::Dim3;
using hip::Function;

// moe_lossy_gemm (feat026-moep): the f16-activation grouped GEMMs with f16 expert outputs (balance item f16
// intermediates) or an atomic scatter combine (fast candidate); never precise; acceptance = task accuracy
enum class MoeBits : std::uint8_t { moe_f32_row, moe_q8_row, moe_f16_gemm, moe_fp8_gemm, moe_lossy_gemm };

struct MoeProblem {
    std::uint32_t n = 0;                  // token rows
    std::uint32_t E = 0, K = 0, F = 0, R = 0;  // n_embd, experts per token, expert FFN width, experts
    GgmlType gate_ty = GgmlType::f32, up_ty = GgmlType::f32, down_ty = GgmlType::f32;
    std::uint64_t gate_rb = 0, up_rb = 0;
    std::uint32_t gate_ncols = 0, down_ncols = 0;
    bool refs = false;                    // all three expert tensors carry MXFP4 load-time row exponents
    bool sx_buf = false, tiles64_buf = false;  // the fp8 scale buffer / the 64-token tile list exist
    bool cnt_buf = false;                 // feat026-moep: the fused-combine counters (n * ceil(E / kMoepMinBm) ints) exist
    bool rcnt_buf = false;                // v0251-a11port: the fused router's block counter (Scratch::moe_rcnt) exists
};
struct MoeFlags {
    bool prec_dec = false;
    std::uint32_t small_max = 16, max_verify_rows = 32, max_small_batch = 16;
    bool moe_mxw = true, moe_rbf = true, moe_fp8 = false;
    std::uint32_t moe_bn_force = 0;
    // feat026-moep (WHIRL_MOEP_GU / WHIRL_MOEP_DN variant names, WHIRL_MOEP_COMB; v0251-a7port: the gfx1151 default is
    // kernels::kMoepDefault, numerics::plan): index into kMoepGu / kMoepDn (-1 = off); combine 0 = moe_combine,
    // 1 = moe_combine4, 2 = fused into the down epilogue (moep only); not bit-identical (ignored when prec_dec, i.e.
    // precise): balance item f16 intermediates: 3 = f16 expert outputs + moe_combine4h, 4 = f16 expert outputs + fused
    // combine; fast candidate: 5 = atomic scatter into x (f32 order varies run to run)
    int moep_gu = -1, moep_dn = -1;
    std::uint32_t moep_comb = 0;
    // v0251-a7pol (B26-41): the combine classes the numerics MODE allows (moepCombFor; qwen35::Context::moeBlock sets it
    // from the plan's mode). Not prec_dec: WHIRL_Q8DEC=1 clears prec_dec in precise. Default: bitwise only.
    enum class MoepComb : std::uint8_t {
        bitwise,  // COMB 0-2 (precise)
        f16,      // + 3 / 4: f16 expert outputs (balance item f16 intermediates)
        atomic,   // + 5: atomic scatter, f32 order varies run to run (fast only)
    } moep_comb_ok = MoepComb::bitwise;
    // v0251-a11port / v0252-c5port (feat026-router, numerics::Flags::moe_rfuse: the MoE default, WHIRL_MOE_RFUSE=0 off): router
    // logits + top-k + shared gate in one launch (bit-identical to moe_logits_f32 + moe_topk) when the kernel exists
    bool moe_rfuse = false;
    // v0252-f5 (moeskip, numerics::Flags::moe_skip_thr / moe_skip_min: fast only): > 0 = drop the routed slots whose
    // renormalized weight is below it (the top moe_skip_min kept) in a block whose path has every skip entry point
    float moe_skip_thr = 0.f;
    std::uint32_t moe_skip_min = 1;
};
// v0251-a7pol: precise -> bitwise, balance -> f16, fast -> atomic (owner mode standard 2026-10-26)
inline MoeFlags::MoepComb moepCombFor(bool precise, bool fast) {
    return precise ? MoeFlags::MoepComb::bitwise : fast ? MoeFlags::MoepComb::atomic : MoeFlags::MoepComb::f16;
}
// v0251-a7pol (B26-43): the moep expert GEMMs only when the experts average at least this many tokens of the forward
// (n * K >= kMoepMinTokensPerExpert * R; Ornith-1.5-35B-A3B, 8 of 256 experts: n >= 1024 rows). Below it chooseMoe gives
// the shipped 0.2.5 plan (fused_gu + moe_combine; bitwise equal to moep, so only speed changes): the gate + up tile is
// 64 tokens of 16-token fragments that share each decoded weight fragment, and with <= 16 tokens per expert most
// experts fill one fragment (no sharing) of a 64 / 128-token tile. 8060S A7 verify 2026-11-04 (6 ABBA rounds, Ornith
// prefill, moep vs 0.2.5): 4 tokens / expert (pp128) -3.4 % balance / -2.0 % precise, 16 (pp512) -1.0 / -0.5 %,
// 32 (pp1024) +1.5 / +1.3 %, 64 (pp2048) +2.4 / +2.7 %. The same rule form as moeTile and llama.cpp's RDNA MoE tile
// choice (mmq.cu ncols_opt = tokens per expert).
inline constexpr std::uint32_t kMoepMinTokensPerExpert = 32;
inline constexpr bool moepWide(std::uint32_t n, std::uint32_t K, std::uint32_t R) {
    return static_cast<std::uint64_t>(n) * K >= static_cast<std::uint64_t>(kMoepMinTokensPerExpert) * R;
}
// v0252-precfix (PLAN V1-85): a prefill chunk joins a segmented forward (several requests' chunks in one forward) only
// with MORE rows than this, so that its MoE bit class is the same alone and inside any segmented forward (which has more
// than twice this many rows). chooseMoe gives the decode experts to n <= small_max (max_small_batch in a prefill forward)
// and, in precise, to every n <= max_verify_rows (92aa626: moe_f32_row; above it moe_f16_gemm): precise -> max_verify_rows,
// else max_small_batch (the server's join rule since v0.1.0, docs/guide/en/server.md). One rule for the server
// (engine_prefill.cpp), the model (Context::prefillSegs), golden and seqtest: change it together with chooseMoe's pl.small.
inline constexpr std::uint32_t moeSegJoinMin(bool prec_dec, std::uint32_t max_small_batch, std::uint32_t max_verify_rows) {
    return prec_dec && max_verify_rows > max_small_batch ? max_verify_rows : max_small_batch;
}
// v0251-a7port: the plan's moep flags (numerics::Flags moep_gu / moep_dn = variant index + 1, 0 = off; moep_comb) ->
// MoeFlags (index, -1 = off); qwen35::Context::moeBlock's assembly, pinned on the CPU (tests/ops/ops_moe_cases.inc)
inline void setMoep(MoeFlags& f, std::uint32_t plan_gu, std::uint32_t plan_dn, std::uint32_t plan_comb) {
    f.moep_gu = static_cast<int>(plan_gu) - 1;
    f.moep_dn = static_cast<int>(plan_dn) - 1;
    f.moep_comb = plan_comb;
}
enum class MoePrefillKind : std::uint8_t { fp8, fused_gu, gemmr, generic, moep };
struct MoePlan {
    bool small = false;      // decode experts (else the grouped prefill GEMMs)
    bool small_q8 = false;   // int8 decode experts (else precise f32 ones, when small)
    MoeBits bits = MoeBits::moe_f16_gemm;
    // decode: gate + up (one launch when the types match, else gate then up) and down
    bool gu_split = false;
    Function gu = nullptr, gu_up = nullptr, down = nullptr;
    Dim3 gu_grid{}, down_grid{};
    // router logits for small batches: moe_logits_f32 in pieces of max_small_batch rows
    Dim3 logits_grid{};
    // v0251-a11port: the fused router (moe_router_fused_i16 / _i24, grid logits_grid) instead of moe_logits_f32 + moe_topk;
    // null = the two-kernel path (flag off, n > max_small_batch, a prefill batch, a shape outside the kernel, no kernel)
    Function router = nullptr;
    // prefill
    MoePrefillKind kind = MoePrefillKind::generic;
    std::uint32_t bn = 64;           // token tile of the route / tile lists
    std::uint32_t max_tiles = 0;
    bool rbf = false;                // fp8: the row-block-fast 1-D grid variants
    Function fgu = nullptr, fdn = nullptr;      // fp8 / fused / gemmr: gate+up and down
    Function g_gate = nullptr, g_up = nullptr, g_down = nullptr;  // generic
    Dim3 gu_pgrid{}, dn_pgrid{};     // prefill gate+up (or gate, generic) / down grids
    std::uint32_t max_tiles64 = 0;   // fused: the 64-token tile list
    // moep: gate + up tile list (moe_tiles bn = gu_bn into the 64-token list buffer; down uses moe_route's list,
    // bn = the down variant's), fused-combine down twin
    std::uint32_t gu_bn = 0;
    bool comb_fused = false;         // moep: fdn is the gemm_moepdc_* twin; no moe_combine launch
    bool comb4 = false;              // moe_combine4 instead of moe_combine (any prefill kind)
    bool ydh = false;                // moep, balance / fast: expert outputs stored f16 (moe_combine4h or the fused f16 twin);
                                     // v0252-moedf16: also the fp8 kind with COMB 3 (fdn = gemm8r_moeh / _moe32h, W2 = null)
    bool comb_atomic = false;        // moep, fast candidate: fdn = gemm_moepda_* (x += w * y), shared expert first by
                                     // moe_combine4 with K = 0; no combine launch after
    Dim3 comb_grid{};
    // v0252-f5 (moeskip): this block drops slots. The routing writes id -1 / weight 0 for them (moe_topk_skip, or
    // moe_skip_apply after the fused router), gu / down are the skip entry points (decode), and the prefill launches take
    // moe_route_skip / moe_tiles_skip / moe_combine_skip / moe_combine4_skip (never comb_fused or ydh). false: moe_topk and
    // the plan above, no id is -1 (precise and balance always; fast where an entry point of the path is missing)
    bool skip = false;
};
// silu_mul_x8h: the fp8 path's activation kernel (MiscKernels; null: no fp8 path). v0251-a7pol: the moep flags apply only
// when moepWide(n, K, R) (else as moep_gu = moep_dn = -1, moep_comb = 0); COMB 3 / 4 need moep_comb_ok >= f16, COMB 5 atomic.
// v0252-f5: moe_skip_thr > 0 (and not prec_dec) -> MoePlan::skip where the path has every skip entry point (moeSkipOk)
MoePlan chooseMoe(const MoeProblem& p, const MoeFlags& f, const kernels::MoeKernels& k, Function silu_mul_x8h);
// v0252-c3port (feat026-moedec L1 + L3, port of feat026-moedec-p25 0ae5530 to gfx1201 only): the fused precise decode
// launch moe_gus*_f_mxfp4 (kernels/moe_fused.hip) - routed gate/up + SwiGLU and the shared expert's gate/up gemvh tiles
// + SwiGLU (f16 output for the shared down gemvh) in one launch, bitwise the chain moe_gu_f + silu_mul and f32_to_f16 +
// gemvh x2 + silu_mul + f32_to_f16 it replaces. Taken only where that chain would run with exactly those kernels
// (Flags::moe_fuse, WHIRL_MOE_FUSE, default off until the GPU A/B). Only the gfx1201 code object has the kernels: on the
// gfx1151 table the plan is never used (the 0.2.5.1 path).
struct MoeFuseProblem {
    std::uint32_t n = 0, C = 0, F = 0, K = 0;  // token rows, n_embd, expert FFN width, experts per token
    GgmlType gate_ty = GgmlType::f32, up_ty = GgmlType::f32;
    std::uint64_t gate_rb = 0, up_rb = 0;
    bool shared = false;  // the layer has a shared expert (sh_gate)
    GgmlType sg_ty = GgmlType::f32, su_ty = GgmlType::f32;
    std::uint64_t sg_rb = 0, su_rb = 0;
    std::uint32_t sg_nrows = 0, su_nrows = 0, sg_ncols = 0, su_ncols = 0;
    // the old chain's shared gate / up / down each take matmul's f16 path, gate / up on the gemvh GEMV with one block per
    // 16 rows (Context: chooseMatmulPath + chooseGemmF16, no grid cap) - the bits the fused tiles and the plain
    // gemmF16 of the down matrix reproduce; and the shared rows are n_ff (the ffn_g / x16 scratch size)
    bool shared_gemvh = false;
};
struct MoeFusePlan {
    bool use = false;
    Function f = nullptr;
    Dim3 grid{};
    std::uint32_t nsb = 0, nrb = 0;  // shared blocks (4 tile pairs each), routed blocks per pair (4 rows each)
    const char* why = "";            // why not taken (diagnostics / tests)
};
// pl: chooseMoe's plan of the same batch (the fused launch replaces pl.gu + silu_mul; pl.down still runs after it)
MoeFusePlan chooseMoeFuse(const MoeFuseProblem& p, bool enabled, const MoePlan& pl, const kernels::MoeKernels& k);
// token tile: 32 when experts average under ~48 tokens (short prompts), else 64
std::uint32_t moeTile(std::uint32_t n, std::uint32_t K, std::uint32_t R);

}  // namespace whirl::ops
