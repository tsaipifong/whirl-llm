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
    bool ydh = false;                // moep, balance / fast: expert outputs stored f16 (moe_combine4h or the fused f16 twin)
    bool comb_atomic = false;        // moep, fast candidate: fdn = gemm_moepda_* (x += w * y), shared expert first by
                                     // moe_combine4 with K = 0; no combine launch after
    Dim3 comb_grid{};
};
// silu_mul_x8h: the fp8 path's activation kernel (MiscKernels; null: no fp8 path). v0251-a7pol: the moep flags apply only
// when moepWide(n, K, R) (else as moep_gu = moep_dn = -1, moep_comb = 0); COMB 3 / 4 need moep_comb_ok >= f16, COMB 5 atomic
MoePlan chooseMoe(const MoeProblem& p, const MoeFlags& f, const kernels::MoeKernels& k, Function silu_mul_x8h);
// token tile: 32 when experts average under ~48 tokens (short prompts), else 64
std::uint32_t moeTile(std::uint32_t n, std::uint32_t K, std::uint32_t R);

}  // namespace whirl::ops
