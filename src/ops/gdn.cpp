// Gated-DeltaNet kernel choice (arch025-F6; moved from src/model/forward/gdn.cpp Model::gdnLayer / gdnChunked /
// gdnScan). See whirl/ops/gdn.h.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/gdn.h"

namespace whirl::ops {

namespace {
constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }
Function x8(bool g8t, Function f, Function ft) { return g8t ? ft : f; }
}  // namespace

bool chooseGdnH16(const GdnH16Problem& p, const GdnH16Flags& g, const MatmulFlags& f, const H16Tune& h, const kernels::KernelTable& k) {
    const GdnShape& s = p.s;
    if (!(g.ffn_h16 && g.act_fuse && !p.fused && s.d_conv == 4 && p.n > 1 && s.d_state == 128 && s.conv_ch % 128 == 0 && s.head_v == 128 &&
          s.n_v_heads <= 64 && k.gdn_conv_l2n_h != nullptr && k.gdn_conv_state_h != nullptr && chooseH16Out(p.qkv, p.qkv_tune, p.n, 1, f, h, k) &&
          chooseH16Out(p.gate, p.gate_tune, p.n, 1, f, h, k)))
        return false;
    // the conv / gated-norm f16 kernels need the qkv / gate and the out-projection inputs in one form
    const MatShape pc[2] = {p.qkv, p.gate};
    const ActIn pin = chooseActCommon(pc, p.n, 1, g.act_fuse, f, k).value_or(ActIn::gemv);
    const ActIn oin = chooseActCommon(std::span<const MatShape>(&p.out, 1), p.n, 1, g.act_fuse, f, k).value_or(ActIn::gemv);
    return (pin == ActIn::fp8 && oin == ActIn::fp8 && k.gated_norm_x8h != nullptr) ||
           (pin == ActIn::f16 && oin == ActIn::f16 && k.gated_norm_x16h != nullptr);
}

bool gdnBaGemm(const GdnBaProblem& p, const kernels::DeltaNetKernels& k) {
    return p.gdn_ba_on && k.gdn_gates_ba != nullptr && p.ba_buf && p.ba_contiguous;
}

bool gdnFusable(bool fused, const MatShape& beta, const MatShape& alpha, const GdnShape& s, const kernels::DeltaNetKernels& k) {
    return fused && gdnAbFusable(beta, alpha, k) && gdnDecodeShape(s.d_conv, s.d_state, s.head_v, s.conv_ch);
}

const char* gdnDecodeGatesName(GdnDecodeGates g) {
    switch (g) {
        case GdnDecodeGates::precise_ba: return "precise_ba";
        case GdnDecodeGates::precise_split: return "precise_split";
        case GdnDecodeGates::abconv: return "abconv";
        case GdnDecodeGates::ab: return "ab";
    }
    return "?";
}

GdnDecodePlan chooseGdnDecode(GgmlType beta_ty, std::uint32_t n, std::uint32_t small_max, bool prec_dec, std::uint32_t dbg_flags,
                              const GdnBaProblem& ba, const kernels::DeltaNetKernels& k) {
    GdnDecodePlan p;
    if (!preciseAllows(BitClass::q8_row, prec_dec, n, small_max)) {  // the fused step: n <= small_max
        // precise decode: [beta; alpha] on the f16 GEMM from h (f32, written by rmsnormQ8), the gates, then the conv step
        p.gates = gdnBaGemm(ba, k) ? GdnDecodeGates::precise_ba : GdnDecodeGates::precise_split;
        return p;
    }
    const Function abconv = (dbg_flags & 1) != 0 ? nullptr : k.gdn_abconv[ti(beta_ty)];
    if (abconv != nullptr) {
        p.gates = GdnDecodeGates::abconv;
        p.f = abconv;
        return p;
    }
    p.gates = GdnDecodeGates::ab;
    p.f = k.gdn_ab[ti(beta_ty)];
    return p;
}

const char* gdnGatesName(GdnGates g) {
    switch (g) {
        case GdnGates::in2: return "in2";
        case GdnGates::ba: return "ba";
        case GdnGates::split: return "split";
    }
    return "?";
}

GdnGates chooseGdnGates(bool gdn_in2, const GdnBaProblem& ba, const kernels::DeltaNetKernels& k) {
    if (gdn_in2) return GdnGates::in2;
    return gdnBaGemm(ba, k) ? GdnGates::ba : GdnGates::split;
}

const char* gdnConvName(GdnConv c) {
    switch (c) {
        case GdnConv::seg_h16: return "seg_h16";
        case GdnConv::seg_par: return "seg_par";
        case GdnConv::l2n_h: return "l2n_h";
        case GdnConv::l2n: return "l2n";
        case GdnConv::par: return "par";
        case GdnConv::seq: return "seq";
    }
    return "?";
}

GdnConvPlan chooseGdnConv(bool segmented, bool gdn_h16, std::uint32_t n, const GdnShape& s, bool act_fuse, const kernels::DeltaNetKernels& k) {
    GdnConvPlan p;
    if (segmented && gdn_h16) {
        // the solo f16 conv kernels per segment (segmented == solo bitwise)
        p.conv = GdnConv::seg_h16;
        p.l2norm = false;
        p.state_h = true;
    } else if (segmented) {
        p.conv = GdnConv::seg_par;
    } else if (s.d_conv == 4 && n > 1) {
        // conv + q/k l2norm in one pass (bitwise the same) when the shapes allow
        const bool conv_l2 = act_fuse && k.gdn_conv_l2n != nullptr && s.d_state == 128 && s.conv_ch % 128 == 0;
        p.conv = gdn_h16 ? GdnConv::l2n_h : conv_l2 ? GdnConv::l2n : GdnConv::par;
        p.l2norm = !conv_l2;
        p.state_h = gdn_h16;
    }
    return p;
}

const char* gdnChunkName(GdnChunk c) {
    switch (c) {
        case GdnChunk::strix: return "strix";
        case GdnChunk::wmma: return "wmma";
        case GdnChunk::f32: return "f32";
    }
    return "?";
}

GdnChunkPlan chooseGdnChunk(const GdnChunkProblem& p, const GdnChunkFlags& f, const kernels::DeltaNetKernels& k) {
    GdnChunkPlan pl;
    if (f.gdn_strix && p.n >= f.gdn_strix_min) {
        // GDN-8S: strix-llama.cpp chunks (CL = 32) on the existing scratch: T' | P in gc_m, Vh in gc_w,
        // Qh | Kh | gate factors in gc_u (each fits when n <= the batch the gc_* buffers were sized for)
        const std::uint64_t H = p.n_v_heads, Hk = p.n_k_heads, nch = (p.n + 31) / 32;
        const std::uint64_t tp = H * nch * 32 * 32 * 2, qk = Hk * p.n * 128 * 2, vh = H * p.n * 128 * 2, gb = H * nch * 32 * 8;
        if (2 * tp <= p.gc_m_bytes && vh <= p.gc_w_bytes && 2 * qk + gb <= p.gc_u_bytes) {
            pl.kind = GdnChunk::strix;
            pl.nch = nch;
            pl.tp = tp;
            pl.qk = qk;
            return pl;
        }
    }
    if (f.gdn_wmma && k.gdn_wprep != nullptr && k.gdn_wscan8 != nullptr) {
        pl.kind = GdnChunk::wmma;
        return pl;
    }
    pl.kind = GdnChunk::f32;
    return pl;
}

GdnNormPlan chooseGdnGatedNorm(const MatShape& out, std::uint32_t n, const GdnShape& s, bool gdn_h16, bool act_fuse, bool g8t,
                               const MatmulFlags& f, const kernels::KernelTable& k) {
    // gated norm straight into the out-proj GEMM input when possible
    std::optional<ActIn> out_in;
    if (s.head_v == 128 && s.n_v_heads <= 64) out_in = chooseActCommon(std::span<const MatShape>(&out, 1), n, 1, act_fuse, f, k);
    GdnNormPlan p;
    // (gdn_h16 implies out_in: chooseGdnH16 required the out-projection input to be fp8 or f16)
    if (gdn_h16)
        p.f = out_in == ActIn::fp8 ? x8(g8t, k.gated_norm_x8h, k.gated_norm_x8ht) : k.gated_norm_x16h;
    else if (out_in)
        p.f = *out_in == ActIn::fp8 ? x8(g8t, k.gated_norm_x8, k.gated_norm_x8t) : k.gated_norm_x16;
    p.fp8 = p.f != nullptr && out_in == ActIn::fp8;
    return p;
}

}  // namespace whirl::ops
