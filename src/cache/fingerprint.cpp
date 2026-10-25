// SPDX-License-Identifier: Apache-2.0

#include "cache/fingerprint.h"

#include "tier/kv_tier.h"
#include "whirl/common.h"

#include <algorithm>
#include <cctype>
#include <cwchar>
#include <filesystem>
#include <format>
#include <stdexcept>

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
    // legacy_no_gpu: the format before FIX-NUM P2-2 (tests compare with it). Otherwise the GPU identity
    // is required: no silent fallback to the old format when it is empty (FIX-TEST-A, FR-TEST P2-3).
    if (in.legacy_no_gpu) {
        if (!in.gpu_arch.empty() || !in.gpu_name.empty() || !in.kernel_set.empty())
            throw std::invalid_argument("tier fingerprint: legacy_no_gpu with GPU fields set");
    } else {
        if (in.gpu_arch.empty() || in.kernel_set.empty())
            throw std::invalid_argument(std::format("tier fingerprint: GPU identity missing (gpu_arch \"{}\", kernel_set \"{}\")",
                                                    in.gpu_arch, in.kernel_set));
        h = tier::hashStr(h, std::format("gpu {} name {} kernels {}", in.gpu_arch, in.gpu_name, in.kernel_set));
    }
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

FingerprintInputs serverFingerprintInputs(const ServerIdentity& s) {
    if (s.gpu_arch.empty()) throw std::invalid_argument("tier fingerprint: no GPU arch (hip::DeviceInfo::gcn_arch is empty)");
    if (s.kernel_arch.empty() || s.kernel_co_bytes == 0)
        throw std::invalid_argument(std::format("tier fingerprint: no kernel code object (arch \"{}\", {} bytes)", s.kernel_arch,
                                                s.kernel_co_bytes));
    FingerprintInputs in;
    in.exe_id = s.exe_id;
    in.model_base = narrow(std::filesystem::path(widen(s.model_path)).filename().wstring());
    in.tensors = s.tensors;
    in.bytes = s.bytes;
    in.kv_name = s.kv_name;
    in.use_mtp = s.use_mtp;
    in.n_gdn = s.n_gdn;
    in.ck_bytes = s.ck_bytes;
    in.page_bytes = s.page_bytes;
    in.max_batch = s.max_batch;
    in.numerics = s.numerics;
    in.gpu_arch = s.gpu_arch;
    in.gpu_name = s.gpu_name;
    in.kernel_set = std::format("{} co {}", s.kernel_arch, s.kernel_co_bytes);
    in.env = s.env;
    return in;
}

bool fingerprintEnvKept(std::string_view key) {
    std::string up(key);
    for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    if (up.rfind("WHIRL_", 0) != 0) return false;
    static const char* skip[] = {"WHIRL_KV_RAM_MB", "WHIRL_KV_SSD_DIR", "WHIRL_KV_SSD_GB", "WHIRL_KV_SSD_DELAY_MS",
                                 "WHIRL_KV_TIER_MIN", "WHIRL_R9700_LOCK_HELD", "WHIRL_GPU_WAIT", "WHIRL_GPU_SHARE",
                                 "WHIRL_PROFILE", "WHIRL_TRACE_ND", "WHIRL_GATHER_MS", "WHIRL_EXE", "WHIRL_LOOP_LOG",
                                 "WHIRL_TIER_VERIFY", "WHIRL_TIER_MIN_GAIN", "WHIRL_DECODE_MIN_TPS",
                                 "WHIRL_TIMER_PROBE", "WHIRL_PREFIX_CACHE_VERIFY", "WHIRL_LAUNCH_LOG",
                                 // arch025-S0a (I6): baseline dumps; they print, they change no kernel or KV format
                                 "WHIRL_ALLOC_LOG", "WHIRL_ALLOC_LOG_RAW", "WHIRL_DUMP_EXEC", "WHIRL_DUMP_KV",
                                 // launch-log raw word dump (arch024 left these out; they print, change nothing)
                                 "WHIRL_LAUNCH_LOG_RAW", "WHIRL_LAUNCH_LOG_RAW_MAX"};
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
