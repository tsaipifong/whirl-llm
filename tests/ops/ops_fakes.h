// Shared fakes of the kernel-choice tests (name-tagged fake kernel tables), moved verbatim from tests/ops/ops_tests.cpp
// (arch025-n8b1). Types of whirl/kernels_abi.h only: no op module, no device runtime. Each table function is inline so a
// program that uses some of them compiles the rest away without a warning.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/kernels_abi.h"

#include <cstddef>
#include <deque>
#include <string>

namespace {

namespace kn = whirl::kernels;
using whirl::gguf::GgmlType;
using Fn = whirl::hip::Function;

// name tags: a fake Function points at its name
std::deque<std::string> g_names;
inline Fn tag(const std::string& n) {
    g_names.push_back(n);
    return const_cast<char*>(g_names.back().c_str());
}
inline std::string nameOf(Fn f) { return f == nullptr ? std::string("null") : std::string(static_cast<const char*>(f)); }
constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }

// a code object with the int8 GEMV set for q4_k / q6_k / mxfp4 (one-token, multi-token, "g" twins, multi-row R=2),
// the f16 GEMM, gemv_f32_1, and optionally gemvw variant 1 for q4_k and gemvx for q4_k
inline kn::KernelTable fakeTable(bool gv_grp, bool gemvw, bool gemvx, bool fp8) {
    kn::KernelTable k;
    k.gv_grp = gv_grp;
    for (GgmlType t : {GgmlType::q4_k, GgmlType::q6_k, GgmlType::mxfp4}) {
        const std::string s = std::to_string(static_cast<int>(t));
        k.gemvq[ti(t)] = tag("gemvq_" + s);
        for (int nt = 2; nt <= 16; ++nt) {
            k.gemvq_nt[nt - 2][ti(t)] = tag("gemvq_nt" + std::to_string(nt) + "_" + s);
            k.gemvq_nt_g[nt - 2][ti(t)] = tag("ggemvq_nt" + std::to_string(nt) + "_" + s);
            k.gemvq_mr[0][nt - 2][ti(t)] = tag("gemvq_nt" + std::to_string(nt) + "r2_" + s);
            k.gemvq_mr_g[0][nt - 2][ti(t)] = tag("ggemvq_nt" + std::to_string(nt) + "r2_" + s);
            if (gemvw && t == GgmlType::q4_k) {
                k.gemvw[1][nt - 2][ti(t)] = tag("gemvw_nt" + std::to_string(nt) + "v1_" + s);
                k.gemvw_g[1][nt - 2][ti(t)] = tag("ggemvw_nt" + std::to_string(nt) + "v1_" + s);
            }
        }
    }
    k.gemv1[ti(GgmlType::f32)] = tag("gemv_f32_1");
    if (gemvx) k.gemvx[ti(GgmlType::q4_k)] = tag("gemvx_q4_k");
    k.gemvw_head_s = tag("gemvw_head_s");
    if (fp8) k.gemm8[0] = tag("gemm8_c0");
    return k;
}

inline kn::KernelTable actTable(bool gfx1201) {
    kn::KernelTable k = fakeTable(true, false, true, gfx1201);
    k.gemvq[ti(GgmlType::q8_0)] = tag("gemvq_q8_0");
    k.rmsnorm_x16 = tag("rmsnorm_x16");
    k.rmsnorm_x8 = tag("rmsnorm_x8");
    k.gdn_gates_ba = tag("gdn_gates_ba");
    k.gdn_ab[ti(GgmlType::q8_0)] = tag("gdn_ab_q8_0");
    k.gemmsh[0][ti(GgmlType::mxfp4)] = tag("gemmsh_c0_mxfp4");  // both code objects have the f16-out gemms twins
    if (gfx1201) {
        k.gemm8h[0] = tag("gemm8h_c0");
        k.rmsnorm_x8t = tag("rmsnorm_x8t");
        k.rmsnorm_x8h16 = tag("rmsnorm_x8h16");
        k.rmsnorm_x8h16t = tag("rmsnorm_x8h16t");
        for (std::size_t c = 0; c < kn::kGemmCfgs.size(); ++c) k.gemmch[c] = tag("gemm_ch" + std::to_string(c) + "_f16");
    }
    return k;
}

struct ActMode {
    const char* name;
    bool prec_dec, fp8_prefill, g8t;
};

inline kn::KernelTable gdnTable(bool gfx1201) {
    kn::KernelTable k = actTable(gfx1201);
    k.gdn_gates = tag("gdn_gates");
    k.gdn_conv_l2 = tag("gdn_conv_l2");
    k.gdn_step_norm = tag("gdn_step_norm");
    k.gdn_conv_l2n = tag("gdn_conv_l2n");
    k.gdn_conv_par = tag("gdn_conv_par");
    k.gdn_conv_state = tag("gdn_conv_state");
    k.gdn_conv_seq = tag("gdn_conv_seq");
    k.gdn_seq_128 = tag("gdn_seq_128");
    k.gdn_chunk_prep = tag("gdn_chunk_prep");
    k.gdn_chunk_scan = tag("gdn_chunk_scan");
    k.gdn_gated_norm = tag("gdn_gated_norm");
    k.gdn_abconv[ti(GgmlType::q8_0)] = tag("gdn_abconv_q8_0");
    k.gated_norm_x16 = tag("gated_norm_x16");
    k.gated_norm_x8 = tag("gated_norm_x8");
    k.gated_norm_x16h = tag("gated_norm_x16h");
    // the f16 conv kernels exist on both: on gfx1151 the h16 choice is turned off by the profile (h16_from_candidates)
    k.gdn_conv_l2n_h = tag("gdn_conv_l2n_h");
    k.gdn_conv_state_h = tag("gdn_conv_state_h");
    k.attn_prep = tag("attn_prep");
    k.attn_prep_m = tag("attn_prep_m");
    if (gfx1201) {
        k.gdn_wprep = tag("gdn_wprep");
        k.gdn_wscan8 = tag("gdn_wscan8");
        k.gated_norm_x8t = tag("gated_norm_x8t");
        k.gated_norm_x8h = tag("gated_norm_x8h");
        k.gated_norm_x8ht = tag("gated_norm_x8ht");
    } else {
        k.gdn_sx_prep = tag("gdn_sx_prep");
        k.gdn_sx_scan = tag("gdn_sx_scan");
    }
    return k;
}

}  // namespace
