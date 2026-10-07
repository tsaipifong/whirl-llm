// SPDX-License-Identifier: Apache-2.0

#include "cache/fingerprint.h"

#include "tier/kv_tier.h"
#include "whirl/common.h"

#include <algorithm>
#include <cctype>
#include <cwchar>
#include <format>

#ifdef _WIN32
#include <windows.h>
#endif

namespace whirl::cache {

std::uint64_t tierFingerprint(const FingerprintInputs& in) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    h = tier::hashStr(h, "whirl-kv-tier-1");
    h = tier::hashStr(h, std::format("exe {:x} model {} {} {} kv {} mtp {} gdn {} ck {} pg {} batch {}", in.exe_id,
                                     in.model_base, in.tensors, in.bytes, in.kv_name, in.use_mtp, in.n_gdn, in.ck_bytes,
                                     in.page_bytes, in.max_batch));
    h = tier::hashStr(h, "numerics " + in.numerics);
    std::vector<std::pair<std::string, std::string>> kv;
    for (const auto& p : in.env)
        if (fingerprintEnvKept(p.first)) kv.push_back(p);
    std::sort(kv.begin(), kv.end());
    for (const auto& [k, v] : kv) {
        h = tier::hashStr(h, k);
        h = tier::hashStr(h, "=");
        h = tier::hashStr(h, v);
        h = tier::hashStr(h, ";");
    }
    return h;
}

bool fingerprintEnvKept(std::string_view key) {
    std::string up(key);
    for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (up.rfind("WHIRL_", 0) != 0) return false;
    static const char* skip[] = {"WHIRL_KV_RAM_MB", "WHIRL_KV_SSD_DIR", "WHIRL_KV_SSD_GB", "WHIRL_KV_SSD_DELAY_MS",
                                 "WHIRL_KV_TIER_MIN", "WHIRL_R9700_LOCK_HELD", "WHIRL_GPU_WAIT", "WHIRL_GPU_SHARE",
                                 "WHIRL_PROFILE", "WHIRL_TRACE_ND", "WHIRL_GATHER_MS", "WHIRL_EXE", "WHIRL_LOOP_LOG",
                                 "WHIRL_TIER_VERIFY", "WHIRL_TIER_MIN_GAIN", "WHIRL_DECODE_MIN_TPS",
                                 "WHIRL_TIMER_PROBE"};
    for (const char* x : skip)
        if (up == x) return false;
    return true;
}

std::vector<std::pair<std::string, std::string>> fingerprintEnv() {
    std::vector<std::pair<std::string, std::string>> kv;
#ifdef _WIN32
    if (wchar_t* blk = GetEnvironmentStringsW()) {
        for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) {
            const std::string s = narrow(p);
            const std::size_t eq = s.find('=', 1);
            if (eq == std::string::npos) continue;
            std::string k = s.substr(0, eq);
            if (fingerprintEnvKept(k)) kv.emplace_back(k, s.substr(eq + 1));
        }
        FreeEnvironmentStringsW(blk);
    }
#endif
    return kv;
}

}  // namespace whirl::cache
