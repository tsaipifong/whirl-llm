// EnvSnapshot (arch025 R-4 / N1): the model layer's WHIRL_* environment, read once into an immutable value.
// SPDX-License-Identifier: Apache-2.0
//
// One field per variable, holding the raw value exactly as the process environment had it (plan C6: nothing is parsed
// here; std::nullopt = not set, "" = set to the empty string). Each reader keeps its own parse at its use site, so the
// semantics of every variable are the same as when the site called envGet. capture() is the only reader of the process
// environment for these names (and envGet / envFlag, defined in env_snapshot.cpp, the one for everyone else).
// Prior art: progress\arch025-N1.PRIOR_ART.md (llama.cpp common_params, hipfire ProcessConfig).
#pragma once

#include <array>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace whirl::config {

class EnvSnapshot {
public:
    using Value = std::optional<std::string>;

    // src/model/tune.cpp (loadOrTune; checkGemvBitwise: gemv_bitwise_float)
    Value fp8, fp8_mask, g8t, gdn_wmma, act_fuse, gemmh, gemmhq, attn_kx, attn_kg, attn_dqf, attn_seg, gdn_ba, gdn_in2, q4_relaxed,
        ffn_h16, gdn_strix, gdn_strix_min, moe_fp8, kv, q8dec, relax, relax_k, relax_alpha, relax_eps, relax_delta, spec_sample,
        gemv_bitwise_float;
    // numerics::noteSwitches (k_numerics_switches, src/common/numerics.cpp), read by loadOrTune through get(); the
    // second line: the runtime switches the front ends apply (qwen35::applyRuntimeEnv reads them with its own getter),
    // here only for noteSwitches' override record (feat026-o6-r1, review C5; gemv_r / gemv_w / gemv_wh below too)
    Value fd_splits, gemv_max, gv_nmax, gv_group, no_fuse, float_gemv, gdn_seq, naive_attn, dbg, attn_wide, attn_dq4, attn_ws4, moe_bn;
    Value head_chunk, dq4_dyn, gdn_v0, no_graph, wide_verify, draft_window, draft_window_min;
    // src/model/loader.cpp (kv, fd_splits above), src/model/config.cpp, src/model/device.cpp, src/model/dflash.cpp
    Value code_object, moe_mxw, moe_wide, moe_rbf, gemv_r, gemv_w, gemv_wh, gpu_share, gpu_wait, device, dflash_prof;

    struct Entry {
        const char* name;  // without the WHIRL_ prefix
        Value EnvSnapshot::* field;
    };
    static constexpr std::size_t n_names = 58;
    // every field, once; the reflection test checks the count against sizeof(EnvSnapshot)
    static const std::array<Entry, n_names>& names();

    // Reads WHIRL_<name> for every entry of names() (_dupenv_s, as envGet).
    static EnvSnapshot capture();
    // Tests: the given values, every other name not set. Throws std::invalid_argument on a name not in names().
    static EnvSnapshot fromValues(const std::vector<std::pair<std::string, Value>>& values);

    // The raw value by name (numerics::noteSwitches takes its names from a list). Throws std::logic_error on a name
    // that is not in names(): a reader that was not given a field is a programming error, never a silent "not set".
    const Value& get(std::string_view name) const;

    // envFlag semantics: not set -> def, else value != "0" (so "" and "auto" are on).
    static bool flag(const Value& v, bool def = false) { return v ? *v != "0" : def; }
};

// feat026-o6 (3), owner O6 (redone on N2b by feat026-o6-p25): research-only switches removed in 0.2.6, without the
// WHIRL_ prefix: HEAD_GEMVX (the output head on the wide gemvx GEMV), PREC_GEMVH (0: precise rows skip gemvh), PREC_GS
// (a forced precise gemms slot). Never in the user docs; their defaults are the product path, so removing them changes
// no default output. arch025-F7 had made them Model members (Model::head_gemvx / prec_gemvh / prec_gs through
// ForwardEnv); those, the snapshot fields and ForwardEnv are gone. A set one is ignored with a warning (not refused: a
// script that still sets one keeps working). The SSD fingerprint skips them too (src/cache/fingerprint.cpp).
const std::vector<const char*>& removedSwitches();
// One warning per removed switch that `get` reports set (default: the process environment, envGet).
std::vector<std::string> removedSwitchWarnings(const std::function<std::optional<std::string>(std::string_view)>& get);
std::vector<std::string> removedSwitchWarnings();

// The snapshot of the load that made `m` (LoadOptions::env, kept by Model::load). A Model that did not come from
// Model::load (host-only tests) has none: then the environment is read now, at the call, as the sites did before N1.
template <class M>
std::shared_ptr<const EnvSnapshot> snapshotOf(const M& m) {
    if (m.env_snap) return m.env_snap;
    return std::make_shared<const EnvSnapshot>(EnvSnapshot::capture());
}

}  // namespace whirl::config
