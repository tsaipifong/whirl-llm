// EnvSnapshot (arch025 R-4 / N1): the model layer's reads of the process environment. envGet / envFlag are
// config.settings' (src/config/env_get.h, src/config/run_settings.cpp; arch025-N6, lane 1 leg 11).
// SPDX-License-Identifier: Apache-2.0

#include "config/env_snapshot.h"

#include "platform/env.h"
#include "config/env_get.h"  // envGet / envFlag (config.settings since arch025-N6, lane 1 leg 11)

#include <string>

namespace whirl::config {

namespace {
using S = EnvSnapshot;
constexpr std::array<S::Entry, S::n_names> k_names{{
    {"FP8", &S::fp8},
    {"FP8_MASK", &S::fp8_mask},
    {"G8T", &S::g8t},
    {"GDN_WMMA", &S::gdn_wmma},
    {"ACT_FUSE", &S::act_fuse},
    {"GEMMH", &S::gemmh},
    {"GEMMHQ", &S::gemmhq},
    {"ATTN_KX", &S::attn_kx},
    {"ATTN_KG", &S::attn_kg},
    {"ATTN_DQF", &S::attn_dqf},
    {"ATTN_SEG", &S::attn_seg},
    {"GDN_BA", &S::gdn_ba},
    {"GDN_IN2", &S::gdn_in2},
    {"Q4_RELAXED", &S::q4_relaxed},
    {"FFN_H16", &S::ffn_h16},
    {"GDN_STRIX", &S::gdn_strix},
    {"GDN_STRIX_MIN", &S::gdn_strix_min},
    {"MOE_FP8", &S::moe_fp8},
    {"KV", &S::kv},
    {"Q8DEC", &S::q8dec},
    {"RELAX", &S::relax},
    {"RELAX_K", &S::relax_k},
    {"RELAX_ALPHA", &S::relax_alpha},
    {"RELAX_EPS", &S::relax_eps},
    {"RELAX_DELTA", &S::relax_delta},
    {"SPEC_SAMPLE", &S::spec_sample},
    {"GEMV_BITWISE_FLOAT", &S::gemv_bitwise_float},
    {"FD_SPLITS", &S::fd_splits},
    {"GEMV_MAX", &S::gemv_max},
    {"GV_NMAX", &S::gv_nmax},
    {"GV_GROUP", &S::gv_group},
    {"NO_FUSE", &S::no_fuse},
    {"FLOAT_GEMV", &S::float_gemv},
    {"GDN_SEQ", &S::gdn_seq},
    {"NAIVE_ATTN", &S::naive_attn},
    {"DBG", &S::dbg},
    {"ATTN_WIDE", &S::attn_wide},
    {"ATTN_DQ4", &S::attn_dq4},
    {"ATTN_WS4", &S::attn_ws4},
    {"MOE_BN", &S::moe_bn},
    {"CODE_OBJECT", &S::code_object},
    {"MOE_MXW", &S::moe_mxw},
    {"MOE_WIDE", &S::moe_wide},
    {"MOE_RBF", &S::moe_rbf},
    {"GEMV_R", &S::gemv_r},
    {"GEMV_W", &S::gemv_w},
    {"GEMV_WH", &S::gemv_wh},
    {"GPU_SHARE", &S::gpu_share},
    {"GPU_WAIT", &S::gpu_wait},
    {"DEVICE", &S::device},
    {"DFLASH_PROF", &S::dflash_prof},
    {"HEAD_GEMVX", &S::head_gemvx},
    {"PREC_GEMVH", &S::prec_gemvh},
    {"PREC_GS", &S::prec_gs},
}};
// every field is in the table (a field added without an entry would never be read)
static_assert(sizeof(EnvSnapshot) == S::n_names * sizeof(S::Value));

const S::Entry* find(std::string_view name) {
    for (const S::Entry& e : k_names)
        if (name == e.name) return &e;
    return nullptr;
}
}  // namespace

const std::array<EnvSnapshot::Entry, EnvSnapshot::n_names>& EnvSnapshot::names() { return k_names; }

EnvSnapshot EnvSnapshot::capture() {
    EnvSnapshot s;
    for (const Entry& e : k_names) s.*e.field = envGet(e.name);
    return s;
}

EnvSnapshot EnvSnapshot::fromValues(const std::vector<std::pair<std::string, Value>>& values) {
    EnvSnapshot s;
    for (const auto& [n, v] : values) {
        const Entry* e = find(n);
        if (e == nullptr) throw std::invalid_argument("EnvSnapshot: no field for WHIRL_" + n);
        s.*e->field = v;
    }
    return s;
}

const EnvSnapshot::Value& EnvSnapshot::get(std::string_view name) const {
    const Entry* e = find(name);
    if (e == nullptr) throw std::logic_error("EnvSnapshot: WHIRL_" + std::string(name) + " is not captured (add it to names())");
    return this->*e->field;
}

}  // namespace whirl::config
