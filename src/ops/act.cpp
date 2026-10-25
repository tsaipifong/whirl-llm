// Activation-input and fused-path choice (arch025-F5; moved from src/model/forward/dispatch.cpp Model::actIn /
// h16Out / actCommon / rmsnormIn / gdnIn2 / fusedDecode / wideCapable / gdnAbFusable). See whirl/ops/act.h.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/act.h"

namespace whirl::ops {

namespace {
constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }
Function x8(bool g8t, Function f, Function ft) { return g8t ? ft : f; }
}  // namespace

const char* actInName(ActIn a) {
    switch (a) {
        case ActIn::gemv: return "gemv";
        case ActIn::f32in: return "f32in";
        case ActIn::fp8: return "fp8";
        case ActIn::f16: return "f16";
    }
    return "?";
}

ActIn chooseActIn(const MatShape& w, std::uint32_t n, std::uint8_t cls, const MatmulFlags& f, const kernels::MatmulKernels& k) {
    if (preciseRows(f.prec_dec, n, f.gv_max)) return ActIn::f16;
    if (n <= f.gv_max && k.gemvq[ti(w.ty)] != nullptr && !f.float_gemv) return ActIn::gemv;
    if (n == 1 || floatGemvN(w, n, f.gv_max, k)) return ActIn::f32in;
    if (f.fp8_prefill && (f.fp8_mask & cls) != 0 && w.ty == GgmlType::mxfp4 && w.has_ref && k.gemm8[0] != nullptr) return ActIn::fp8;
    return ActIn::f16;
}

bool chooseH16Out(const MatShape& w, std::span<const std::uint8_t> tune, std::uint32_t n, std::uint8_t cls, const MatmulFlags& f,
                  const H16Tune& h, const kernels::MatmulKernels& k) {
    const std::size_t G = kernels::kGemmCfgs.size();
    switch (chooseActIn(w, n, cls, f, k)) {
        case ActIn::fp8:
            return k.gemm8h[0] != nullptr;
        case ActIn::f16:
            // Radeon 8060S: decided by the kernel set and the tuner's candidate list, never by
            // the tuned choices, so the output (f16 intermediates or not) does not depend on
            // autotune timings. The gfx1151 code object has no h16 fused / dequant GEMM, so
            // this is off for quantized weights while those are candidates.
            if (h.from_candidates) {
                for (std::uint32_t cj = 0; cj < h.n_choices && cj < 64; ++cj) {
                    if ((h.tune_mask & (1ull << cj)) == 0) continue;
                    if (cj >= 2 * G) {
                        const std::size_t si = cj - 2 * G;
                        if (k.gemms[si][ti(w.ty)] == nullptr || w.ncols % 256 != 0) continue;  // not a candidate (= tune.cpp)
                        if (k.gemmsh[si][ti(w.ty)] == nullptr) return false;
                        continue;
                    }
                    if (!((cj >= G || w.ty == GgmlType::f16) && k.gemmch[cj % G] != nullptr)) return false;
                }
                // (and the table must agree: a cached table may predate the candidate list)
            }
            // every bucket (not just n's), so the choice is the same for a request's
            // solo chunks and a segmented forward of any total size
            for (std::uint8_t choice : tune) {
                if (choice >= 2 * G) {
                    if (k.gemmsh[choice - 2 * G][ti(w.ty)] == nullptr) return false;
                    continue;
                }
                if (!((choice >= G || w.ty == GgmlType::f16 || h.h16_dq) && k.gemmch[choice % G] != nullptr)) return false;
            }
            return true;
        default:
            return false;
    }
}

std::optional<ActIn> chooseActCommon(std::span<const MatShape> consumers, std::uint32_t n, std::uint8_t cls, bool act_fuse,
                                     const MatmulFlags& f, const kernels::MatmulKernels& k) {
    if (!act_fuse || consumers.empty()) return std::nullopt;
    const ActIn mode = chooseActIn(consumers[0], n, cls, f, k);
    if (mode != ActIn::fp8 && mode != ActIn::f16) return std::nullopt;
    for (std::size_t i = 1; i < consumers.size(); ++i)
        if (chooseActIn(consumers[i], n, cls, f, k) != mode) return std::nullopt;
    return mode;
}

RmsnormInPlan chooseRmsnormIn(std::uint32_t n_embd, std::span<const MatShape> consumers, std::uint32_t n, std::uint8_t cls,
                              bool act_fuse, bool g8t, const MatmulFlags& f, const kernels::KernelTable& k) {
    RmsnormInPlan p;
    if (n_embd > 256 * 32) return p;
    const std::optional<ActIn> mode = chooseActCommon(consumers, n, cls, act_fuse, f, k);
    if (!mode) return p;
    p.fp8 = *mode == ActIn::fp8;
    p.f = p.fp8 ? x8(g8t, k.rmsnorm_x8, k.rmsnorm_x8t) : k.rmsnorm_x16;
    return p;
}

bool chooseGdnIn2(const GdnIn2Problem& p, const GdnIn2Flags& g, const MatmulFlags& f, const kernels::KernelTable& k) {
    if (!g.gdn_in2_on || p.fused || p.n <= 16 || p.n_embd > 8192) return false;
    if (x8(g.g8t, k.rmsnorm_x8h16, k.rmsnorm_x8h16t) == nullptr) return false;
    if (!g.gdn_ba_on || k.gdn_gates_ba == nullptr || !p.ba_buf || !p.ba_contiguous) return false;
    const MatShape pc[2] = {p.qkv, p.gate};
    const MatShape bc[2] = {p.beta, p.alpha};
    const auto pin = chooseActCommon(pc, p.n, 1, g.act_fuse, f, k);
    const auto bin = chooseActCommon(bc, p.n, 1, g.act_fuse, f, k);
    return pin && *pin == ActIn::fp8 && bin && *bin == ActIn::f16;
}

bool gdnAbFusable(const MatShape& beta, const MatShape& alpha, const kernels::DeltaNetKernels& k) {
    return k.gdn_ab[ti(beta.ty) % kernels::kNTypes] != nullptr && beta.ty == alpha.ty && beta.row_bytes == alpha.row_bytes &&
           beta.ncols % 256 == 0;
}

bool wideKernels(const kernels::MatmulKernels& k) { return k.gemvx[ti(GgmlType::q4_k)] != nullptr; }

}  // namespace whirl::ops
