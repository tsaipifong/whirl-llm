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
    // numerics::noteSwitches (k_numerics_switches, src/common/numerics.cpp), read by loadOrTune through get()
    Value fd_splits, gemv_max, gv_nmax, gv_group, no_fuse, float_gemv, gdn_seq, naive_attn, dbg, attn_wide, attn_dq4, attn_ws4, moe_bn;
    // src/model/loader.cpp (kv, fd_splits above), src/model/config.cpp, src/model/device.cpp, src/model/dflash.cpp
    Value code_object, moe_mxw, moe_wide, moe_rbf, gemv_r, gemv_w, gemv_wh, gpu_share, gpu_wait, device, dflash_prof;
    // src/model/forward/dispatch.cpp (arch025-F7, through forwardEnv below): gemvLaunch, gemmFlags
    Value head_gemvx, prec_gemvh, prec_gs;
    // src/server/server_main.cpp (feat026-acc, through snapshotOf(model)): EngineOptions::spec_acc
    Value spec_acc;

    struct Entry {
        const char* name;  // without the WHIRL_ prefix
        Value EnvSnapshot::* field;
    };
    static constexpr std::size_t n_names = 55;
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

// arch025-F7: the forward's three research switches, parsed once per load (Model::load -> Model::head_gemvx /
// prec_gemvh / prec_gs) with the rules of the pre-F7 sites, verbatim:
//   HEAD_GEMVX: envFlag("HEAD_GEMVX", false), read at every wide gemvLaunch (forward.cpp:56, I3 (i)): not set -> off,
//               else value != "0" ("" and "auto" on)
//   PREC_GEMVH: static, first gemmFlags call (forward.cpp:245-248, I3 (ii)): v == nullptr || *v != '0' (the FIRST
//               character: "00" and "0x" are off, "" is on)
//   PREC_GS:    static, first gemmFlags call (forward.cpp:249-252): v != nullptr && *v != 0 ? atoi(v) : -1 ("" -> -1,
//               "abc" -> 0)
// Behaviour change (plan I3): a process that changes WHIRL_HEAD_GEMVX after the load no longer sees the new value at
// the next call; nothing in src/ or tests/ does (model tests: testForwardEnv grep). feat026-o6 removes the three
// switches for 0.2.6 (warn and ignore): on that merge these fields, ForwardEnv and the Model members go away.
struct ForwardEnv {
    bool head_gemvx = false;
    bool prec_gemvh = true;
    int prec_gs = -1;
};
inline ForwardEnv forwardEnv(const EnvSnapshot& env) {
    ForwardEnv f;
    f.head_gemvx = EnvSnapshot::flag(env.head_gemvx, false);
    f.prec_gemvh = !env.prec_gemvh || env.prec_gemvh->c_str()[0] != '0';
    f.prec_gs = env.prec_gs && env.prec_gs->c_str()[0] != 0 ? std::atoi(env.prec_gs->c_str()) : -1;
    return f;
}

// The snapshot of the load that made `m` (LoadOptions::env, kept by Model::load). A Model that did not come from
// Model::load (host-only tests) has none: then the environment is read now, at the call, as the sites did before N1.
template <class M>
std::shared_ptr<const EnvSnapshot> snapshotOf(const M& m) {
    if (m.env_snap) return m.env_snap;
    return std::make_shared<const EnvSnapshot>(EnvSnapshot::capture());
}

}  // namespace whirl::config
