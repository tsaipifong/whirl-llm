// arch025-n8b1: cases moved verbatim from tests/cache/cache_tests.cpp; included by whirl-cache-tests (tests/cache/cache_tests.cpp,
// same cases, same order) and by the module's isolation program tests/iso/cache.fingerprint/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "cache/fingerprint.h"
#include "tier/kv_tier.h"
#include "whirl/common.h"
#include "whirl/numerics.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <map>
#include <numeric>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "fingerprint_env_names.inc"  // arch025-i6vis: k_fp_hashed
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>  // arch025 PL1: the reference copy of the base GetEnvironmentStringsW loop
#endif

using namespace whirl::cache;

// the pre-R-1 server_main.cpp tierFingerprint (env list instead of the process block)
static std::uint64_t oldFingerprint(const FingerprintInputs& in, std::vector<std::pair<std::string, std::string>> env) {
    using whirl::tier::hashStr;
    std::uint64_t h = 0xcbf29ce484222325ull;
    h = hashStr(h, "whirl-kv-tier-1");
    h = hashStr(h, std::format("exe {:x} model {} {} {} kv {} mtp {} gdn {} ck {} pg {} batch {}", in.exe_id,
                               in.model_base, in.tensors, in.bytes, in.kv_name, in.use_mtp, in.n_gdn, in.ck_bytes,
                               in.page_bytes, static_cast<std::uint32_t>(in.max_batch)));
    h = hashStr(h, "numerics " + in.numerics);
    std::sort(env.begin(), env.end());
    for (const auto& [k, v] : env) {
        h = hashStr(h, k);
        h = hashStr(h, "=");
        h = hashStr(h, v);
        h = hashStr(h, ";");
    }
    return h;
}

// FIX-TEST-A (FR-TEST P2-3): the server's fingerprint assembly (cache::serverFingerprintInputs, called
// by server_main.cpp). Mutation M5 (server_main stopped filling gpu_arch / gpu_name) passed every test.
static void testServerFingerprintInputs() {
    ServerIdentity s;
    s.exe_id = 0x77;
    s.model_path = "C:\\models\\x\\Qwen3.8-27B-Q4_K_M.gguf";
    s.tensors = 851;
    s.bytes = 16'000'000'000ull;
    s.kv_name = "q8h";
    s.use_mtp = true;
    s.n_gdn = 48;
    s.ck_bytes = 1 << 20;
    s.page_bytes = 1 << 18;
    s.max_batch = 4096;
    s.numerics = "balance";
    s.gpu_arch = "gfx1201";
    s.gpu_name = "AMD Radeon AI PRO R9700";
    s.kernel_arch = "gfx1201";
    s.kernel_co_bytes = 123456;
    s.env = {{"WHIRL_A", "1"}};
    const FingerprintInputs in = serverFingerprintInputs(s);
    CHECK(in.gpu_arch == "gfx1201");
    CHECK(in.gpu_name == "AMD Radeon AI PRO R9700");
    CHECK(in.kernel_set == "gfx1201 co 123456");
    CHECK(in.model_base == "Qwen3.8-27B-Q4_K_M.gguf");
    CHECK(in.kv_name == "q8h" && in.numerics == "balance" && in.max_batch == 4096 && in.use_mtp && in.n_gdn == 48);
    CHECK(in.tensors == 851 && in.bytes == 16'000'000'000ull && in.ck_bytes == (1u << 20) && in.page_bytes == (1u << 18));
    CHECK(in.exe_id == 0x77 && in.env.size() == 1);
    CHECK(!in.legacy_no_gpu);
    const std::uint64_t fp = tierFingerprint(in);
    // a different GPU (same exe, same model) -> a different fingerprint, field by field
    ServerIdentity o = s;
    o.gpu_arch = "gfx1151";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.gpu_name = "AMD Radeon(TM) 8060S Graphics";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.kernel_arch = "gfx1151";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.kernel_co_bytes += 1;
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.kv_name = "f16";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    // missing GPU identity -> error (no silent old format)
    auto throws = [](const ServerIdentity& x) {
        try {
            (void)serverFingerprintInputs(x);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    o = s;
    o.gpu_arch.clear();
    CHECK(throws(o));
    o = s;
    o.kernel_arch.clear();
    CHECK(throws(o));
    o = s;
    o.kernel_co_bytes = 0;
    CHECK(throws(o));
    o = s;
    o.gpu_name.clear();  // the name is informative only; the arch is the key
    CHECK(!throws(o));
}

static void testFingerprint() {
    FingerprintInputs in;
    in.exe_id = 0x1234abcd;
    in.model_base = "Qwen3.8-27B-Q4_K_M.gguf";
    in.tensors = 851;
    in.bytes = 16'000'000'000ull;
    in.kv_name = "f16";
    in.use_mtp = true;
    in.n_gdn = 48;
    in.ck_bytes = 1 << 20;
    in.page_bytes = 1 << 18;
    in.max_batch = 4096;
    in.numerics = "balance (items a,b)";
    in.legacy_no_gpu = true;  // the format checks below compare with the pre-FIX-NUM hash
    const std::uint64_t base = tierFingerprint(in);
    CHECK(base == tierFingerprint(in));
    CHECK(base == oldFingerprint(in, {}));
    // numerics mode is part of the key
    FingerprintInputs p = in;
    p.numerics = "precise (items)";
    CHECK(tierFingerprint(p) != base);
    CHECK(tierFingerprint(p) == oldFingerprint(p, {}));
    // KV format is part of the key (FR-CACHE: corrupting kv_name went unnoticed)
    for (const char* kv : {"q8h", "q8", "q8v", "q4", ""}) {
        p = in;
        p.kv_name = kv;
        CHECK(tierFingerprint(p) != base);
    }
    p = in;
    p.max_batch = 2048;
    CHECK(tierFingerprint(p) != base);
    p = in;
    p.exe_id += 1;
    CHECK(tierFingerprint(p) != base);
    // model identity / layout / mtp
    p = in;
    p.use_mtp = false;
    CHECK(tierFingerprint(p) != base);
    p = in;
    p.page_bytes += 1;
    CHECK(tierFingerprint(p) != base);
    // environment: WHIRL_* kept, order free, size / path / logging knobs ignored
    p = in;
    p.env = {{"WHIRL_KV_SSD_DIR", "D:\\x"}, {"PATH", "C:\\"}, {"WHIRL_PROFILE", "1"}, {"whirl_kv_ram_mb", "8"}};
    CHECK(tierFingerprint(p) == base);
    p.env = {{"WHIRL_B", "2"}, {"WHIRL_A", "1"}, {"OTHER", "x"}};
    const std::uint64_t e1 = tierFingerprint(p);
    CHECK(e1 != base);
    p.env = {{"WHIRL_A", "1"}, {"WHIRL_B", "2"}};
    CHECK(tierFingerprint(p) == e1);
    CHECK(e1 == oldFingerprint(in, {{"WHIRL_A", "1"}, {"WHIRL_B", "2"}}));
    p.env = {{"WHIRL_A", "1"}, {"WHIRL_B", "3"}};
    CHECK(tierFingerprint(p) != e1);
    // FIX-NUM P2-2: the GPU code object is part of the key (one exe, R9700 + 8060S kernels)
    {
        FingerprintInputs a = in, b = in;
        a.legacy_no_gpu = b.legacy_no_gpu = false;
        a.gpu_arch = "gfx1201";
        b.gpu_arch = "gfx1151";
        a.kernel_set = b.kernel_set = "co 1";
        CHECK(tierFingerprint(a) != tierFingerprint(b));
        CHECK(tierFingerprint(a) != base && tierFingerprint(b) != base);
        CHECK(tierFingerprint(a) == tierFingerprint(a));
        // device name and kernel set (FR-CACHE P1-1)
        FingerprintInputs e = a;
        e.gpu_name = "AMD Radeon(TM) 8060S Graphics";
        CHECK(tierFingerprint(e) != tierFingerprint(a));
        e = a;
        e.kernel_set = "gfx1151 co 123";
        CHECK(tierFingerprint(e) != tierFingerprint(a));
        // FIX-TEST-A (FR-TEST P2-3): an empty GPU identity is an error, not the old format
        auto throws = [](const FingerprintInputs& x) {
            try {
                (void)tierFingerprint(x);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        e = a;
        e.gpu_arch.clear();
        CHECK(throws(e));
        e = a;
        e.kernel_set.clear();
        CHECK(throws(e));
        e = a;
        e.gpu_arch.clear();
        e.gpu_name.clear();
        e.kernel_set.clear();
        CHECK(throws(e));  // all three empty: used to hash silently as the pre-FIX-NUM format
        e = a;
        e.legacy_no_gpu = true;  // legacy format with GPU fields set: contradictory
        CHECK(throws(e));
    }
    p.env = {{"Whirl_Relax", "1"}};  // prefix check is case-insensitive, the key is hashed as written
    CHECK(tierFingerprint(p) != base);
    CHECK(fingerprintEnvKept("WHIRL_NUMERICS"));
    CHECK(fingerprintEnvKept("whirl_x"));
    CHECK(!fingerprintEnvKept("WHIRL_KV_TIER_MIN"));
    CHECK(!fingerprintEnvKept("WHIRL_TIMER_PROBE"));
    // CACHEBAL: the cache-hit diagnostic compares, it does not change the stored KV format, so a verify run can
    // restore the entries a normal server wrote (the tier p1 gate does exactly that)
    CHECK(!fingerprintEnvKept("WHIRL_PREFIX_CACHE_VERIFY"));
    CHECK(!fingerprintEnvKept("WHIRL_LAUNCH_LOG"));  // ARCH-1 K-0 launch table (diagnostic)
    CHECK(!fingerprintEnvKept("WHIRL_LEAKCHECK_QUIET"));  // safe025 leak-check report switch (test builds only; rule I6)
    CHECK(!fingerprintEnvKept("WHIRL_LEAK_CHECK"));       // arch025-Q2 GPU-side leak count (rule I6)
    CHECK(!fingerprintEnvKept("whirl_leak_check"));       // the skip compare is case-insensitive like the prefix
#ifdef _WIN32
    // I6: each switch on and off gives the same fingerprint environment and the same fingerprint
    for (const char* sw : {"WHIRL_LEAKCHECK_QUIET", "WHIRL_LEAK_CHECK"}) {
        const auto off = fingerprintEnv();
        _putenv_s(sw, "1");
        const auto on = fingerprintEnv();
        _putenv_s(sw, "");
        CHECK(on == off);
        auto q = p;
        q.env = off;
        const auto fp_off = tierFingerprint(q);
        q.env = on;
        CHECK(tierFingerprint(q) == fp_off);
    }
    {
        // block proof of the on/off check itself: a variable that IS hashed changes the fingerprint environment
        const auto off = fingerprintEnv();
        _putenv_s("WHIRL_Q2_I6_PROBE", "1");
        const auto on = fingerprintEnv();
        _putenv_s("WHIRL_Q2_I6_PROBE", "");
        CHECK(on != off);
    }
#endif
    CHECK(!fingerprintEnvKept("WHIRLX"));
    // the process environment is already filtered
    for (const auto& [k, v] : fingerprintEnv()) CHECK(fingerprintEnvKept(k));
}

// arch025-S0a (I6, REVIEW-ARCH025 A1): the baseline dumps are diagnostics; turning them on must not change the SSD
// fingerprint of the process environment (else a t2.ssd / restart case run with a dump on would miss the entries
// a run without it wrote). Control: an ordinary WHIRL_* variable does change it, so the env path is live.
static void setEnv(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v != nullptr ? v : "");
#else
    if (v != nullptr) setenv(k, v, 1); else unsetenv(k);
#endif
}
static void testFingerprintDumpEnv() {
    FingerprintInputs in;
    in.exe_id = 0x1234abcd;
    in.model_base = "Swift-1.5-27B-MXFP4.gguf";
    in.tensors = 851;
    in.bytes = 16'000'000'000ull;
    in.kv_name = "q8h";
    in.use_mtp = true;
    in.n_gdn = 48;
    in.ck_bytes = 1 << 20;
    in.page_bytes = 1 << 18;
    in.max_batch = 4096;
    in.numerics = "balance (items a,b)";
    in.gpu_arch = "gfx1201";
    in.gpu_name = "AMD Radeon AI PRO R9700";
    in.kernel_set = "gfx1201 co 123";
    const char* dumps[] = {"WHIRL_ALLOC_LOG",    "WHIRL_ALLOC_LOG_RAW",  "WHIRL_DUMP_EXEC",
                           "WHIRL_DUMP_KV",      "WHIRL_LAUNCH_LOG_RAW", "WHIRL_LAUNCH_LOG_RAW_MAX"};
    for (const char* k : dumps) setEnv(k, nullptr);
    in.env = fingerprintEnv();
    const std::uint64_t off = tierFingerprint(in);
    for (const char* k : dumps) CHECK(!fingerprintEnvKept(k));
    setEnv("WHIRL_ALLOC_LOG", "C:\tmp\alloc");
    setEnv("WHIRL_ALLOC_LOG_RAW", "server");
    setEnv("WHIRL_DUMP_EXEC", "1");
    setEnv("WHIRL_DUMP_KV", "1");
    setEnv("WHIRL_LAUNCH_LOG_RAW", "1");
    setEnv("WHIRL_LAUNCH_LOG_RAW_MAX", "1000");
    in.env = fingerprintEnv();
    CHECK(tierFingerprint(in) == off);
    setEnv("WHIRL_S0A_CONTROL", "1");  // not a skipped name: part of the key
    in.env = fingerprintEnv();
    CHECK(tierFingerprint(in) != off);
    setEnv("WHIRL_S0A_CONTROL", nullptr);
    for (const char* k : dumps) setEnv(k, nullptr);
    in.env = fingerprintEnv();
    CHECK(tierFingerprint(in) == off);
}

// arch025-i6vis (rule I6, X4 K4 / G11): the bug-visstream-race test switch WHIRL_TEST_VIS_COPY_DELAY_MB
// (src/vision/vis_cli.cpp) slows one copy; it changes no kernel and no KV format, so the SSD fingerprint of the
// process environment must be the same with and without it (else a run with the switch misses every entry).
static FingerprintInputs i6Inputs() {
    FingerprintInputs in;
    in.exe_id = 0x1234abcd;
    in.model_base = "Swift-1.5-27B-MXFP4.gguf";
    in.tensors = 851;
    in.bytes = 16'000'000'000ull;
    in.kv_name = "q8h";
    in.use_mtp = true;
    in.n_gdn = 48;
    in.ck_bytes = 1 << 20;
    in.page_bytes = 1 << 18;
    in.max_batch = 4096;
    in.numerics = "balance (items a,b)";
    in.gpu_arch = "gfx1151";
    in.gpu_name = "AMD Radeon(TM) 8060S Graphics";
    in.kernel_set = "gfx1151 co 123";
    return in;
}

// sets k to v (nullptr: unset) and returns the previous value so the caller can put the shell's value back
static std::optional<std::string> swapEnv(const char* k, const char* v) {
    std::optional<std::string> old;
    if (const char* p = std::getenv(k)) old = p;
    setEnv(k, v);
    return old;
}

static void testFingerprintVisCopyDelayEnv() {
    const char* k = "WHIRL_TEST_VIS_COPY_DELAY_MB";
    CHECK(!fingerprintEnvKept(k));
    CHECK(!fingerprintEnvKept("whirl_test_vis_copy_delay_mb"));
    FingerprintInputs in = i6Inputs();
    const std::optional<std::string> old = swapEnv(k, nullptr);
    in.env = fingerprintEnv();
    const auto env_off = in.env;
    const std::uint64_t off = tierFingerprint(in);
    for (const char* v : {"1", "64", "4096"}) {
        setEnv(k, v);
        in.env = fingerprintEnv();
        CHECK(in.env == env_off);
        CHECK(tierFingerprint(in) == off);
    }
    setEnv(k, old ? old->c_str() : nullptr);
}

// arch025-i6vis: structural I6 check. Every WHIRL_* name the source reads must be classified: skipped by
// fingerprintEnvKept (test / debug / dump / size / path switch) or listed in k_fp_hashed (can change what is
// computed or stored). The scan knows the three ways the source names a variable:
//   (a) the full literal "WHIRL_X" (platform::env::get / getUtf8 / isSet, vis_cli envRaw, gpu_restrict, ...)
//   (b) a WHIRL_-prefixing reader called with the bare name: envGet / envFlag / envU32 / getEnv / dump::switchOn /
//       server env / dflash env ("X")
//   (c) a member-pointer name table entry {"X", &S::field} (config::EnvSnapshot, numerics::RuntimeValues / EnvValues)
using EnvNames = std::map<std::string, std::string>;  // name -> first file it was seen in

static void scanEnvNames(const std::string& text, const std::string& where, EnvNames& out) {
    static const std::regex full(R"re("(WHIRL_[A-Z0-9_]+)")re");
    static const std::regex call(R"re(\b(?:env|envGet|envFlag|envU32|getEnv|switchOn)\(\s*"([A-Z][A-Z0-9_]*)")re");
    static const std::regex table(R"re(\{\s*"([A-Z][A-Z0-9_]*)"\s*,\s*&)re");
    std::istringstream is(text);
    std::string line;
    while (std::getline(is, line)) {
        for (auto it = std::sregex_iterator(line.begin(), line.end(), full); it != std::sregex_iterator(); ++it)
            out.emplace((*it)[1].str(), where);
        for (const std::regex* re : {&call, &table})
            for (auto it = std::sregex_iterator(line.begin(), line.end(), *re); it != std::sregex_iterator(); ++it) {
                const std::string n = (*it)[1].str();
                out.emplace(n.rfind("WHIRL_", 0) == 0 ? n : "WHIRL_" + n, where);
            }
    }
}

static bool scanSourceEnvNames(EnvNames& out, std::size_t& files) {
#ifdef WHIRL_SOURCE_DIR
    namespace fs = std::filesystem;
    const fs::path root(WHIRL_SOURCE_DIR);
    files = 0;
    for (const char* sub : {"src", "include"}) {
        std::error_code ec;
        if (!fs::is_directory(root / sub, ec)) return false;
        for (const auto& e : fs::recursive_directory_iterator(root / sub)) {
            if (!e.is_regular_file()) continue;
            const std::string ext = e.path().extension().string();
            if (ext != ".cpp" && ext != ".h" && ext != ".hpp" && ext != ".hip" && ext != ".cuh" && ext != ".inc") continue;
            const std::string rel = e.path().lexically_relative(root).generic_string();
            if (rel == "src/cache/fingerprint.cpp") continue;  // the skip list itself is not a read
            std::ifstream f(e.path(), std::ios::binary);
            std::ostringstream ss;
            ss << f.rdbuf();
            scanEnvNames(ss.str(), rel, out);
            ++files;
        }
    }
    return true;
#else
    (void)out;
    (void)files;
    return false;
#endif
}

// names in `found` that are neither skipped nor in `hashed`; the core of the check, unit-tested below
static std::vector<std::string> unclassified(const EnvNames& found, const std::set<std::string>& hashed) {
    std::vector<std::string> bad;
    for (const auto& [n, where] : found)
        if (fingerprintEnvKept(n) && hashed.count(n) == 0) bad.push_back(n + " (" + where + ")");
    return bad;
}

static void testFingerprintEnvClassified() {
    const std::set<std::string> hashed(std::begin(k_fp_hashed), std::end(k_fp_hashed));
    CHECK(hashed.size() == std::size(k_fp_hashed));  // no duplicate in the list
    // scanner self-test on synthetic text: each of the three forms is found, the bare forms get the prefix, and a
    // new unclassified switch is reported (the check is live, not vacuous)
    {
        EnvNames s;
        scanEnvNames("x = platform::env::get(\"WHIRL_I6_NEW_A\");\n"
                     "if (envFlag(\"I6_NEW_B\", false)) {}\n"
                     "auto v = dump::switchOn(\"I6_NEW_C\");\n"
                     "{\"I6_NEW_D\", &S::field},\n"
                     "envU32(\"I6_NEW_E\", 0); env(\"I6_NEW_F\"); getEnv(\"I6_NEW_G\"); envGet(\"I6_NEW_H\");\n"
                     "envRaw(\"WHIRL_I6_NEW_I\"); envRawX(\"NOT_A\"); \"OTHER_NOT_B\";\n",
                     "synthetic", s);
        for (const char* n : {"WHIRL_I6_NEW_A", "WHIRL_I6_NEW_B", "WHIRL_I6_NEW_C", "WHIRL_I6_NEW_D", "WHIRL_I6_NEW_E",
                              "WHIRL_I6_NEW_F", "WHIRL_I6_NEW_G", "WHIRL_I6_NEW_H", "WHIRL_I6_NEW_I"})
            CHECK(s.count(n) == 1);
        CHECK(s.count("WHIRL_NOT_A") == 0 && s.count("WHIRL_OTHER_NOT_B") == 0);
        CHECK(s.size() == 9);
        CHECK(unclassified(s, hashed).size() == 9);
        EnvNames t;
        scanEnvNames("envGet(\"KV_SSD_DIR\"); envGet(\"KV\");\n", "synthetic", t);
        CHECK(unclassified(t, hashed).empty());  // one skipped, one hashed
    }
    EnvNames found;
    std::size_t files = 0;
    const bool scanned = scanSourceEnvNames(found, files);
    CHECK(scanned);  // no source tree: fail, never pass silently
    if (!scanned) {
        std::printf("FAIL i6: WHIRL_SOURCE_DIR not set or src/ include/ missing; the env scan did not run\n");
        return;
    }
    // each form occurs in the real tree (a scanner that stopped matching one form would shrink the check)
    for (const char* n : {"WHIRL_TEST_VIS_COPY_DELAY_MB", "WHIRL_VRAM_LIMIT_MB",  // (a)
                          "WHIRL_KV", "WHIRL_GEMVH_WAVES", "WHIRL_DUMP_EXEC", "WHIRL_DFLASH_PMIN",
                          "WHIRL_POOL_RESERVE_MB",                          // (b)
                          "WHIRL_FP8", "WHIRL_NO_GRAPH"})                   // (c)
        CHECK(found.count(n) == 1);
    CHECK(files > 100 && found.size() > 100);
    for (const std::string& n : unclassified(found, hashed)) {
        std::printf("FAIL i6: %s is read by the source but neither skipped by fingerprintEnvKept "
                    "(src/cache/fingerprint.cpp) nor listed in tests/cache/fingerprint_env_names.inc\n",
                    n.c_str());
        CHECK(false);
    }
    for (const std::string& n : hashed) {
        if (fingerprintEnvKept(n) == false) {
            std::printf("FAIL i6: %s is in fingerprint_env_names.inc (hashed) but fingerprintEnvKept skips it\n", n.c_str());
            CHECK(false);
        }
        if (found.count(n) == 0) {
            std::printf("FAIL i6: %s is in fingerprint_env_names.inc but the source no longer reads it\n", n.c_str());
            CHECK(false);
        }
    }
    // I6 for every skipped name, not only the new ones: on and off give the same fingerprint environment and the
    // same fingerprint. Control: a hashed name does change it.
    FingerprintInputs in = i6Inputs();
    in.env = fingerprintEnv();
    const auto env_off = in.env;
    const std::uint64_t off = tierFingerprint(in);
    std::size_t n_skipped = 0;
    for (const auto& [n, where] : found) {
        if (fingerprintEnvKept(n)) continue;
        ++n_skipped;
        const std::optional<std::string> old = swapEnv(n.c_str(), "1");
        in.env = fingerprintEnv();
        const bool same = in.env == env_off && tierFingerprint(in) == off;
        if (!same) std::printf("FAIL i6: setting skipped %s changed the SSD fingerprint\n", n.c_str());
        CHECK(same);
        setEnv(n.c_str(), old ? old->c_str() : nullptr);
    }
    CHECK(n_skipped >= 20);
    {
        const std::optional<std::string> old = swapEnv("WHIRL_RELAX", "1");
        in.env = fingerprintEnv();
        CHECK(tierFingerprint(in) != off);
        setEnv("WHIRL_RELAX", old ? old->c_str() : nullptr);
    }
    in.env = fingerprintEnv();
    CHECK(in.env == env_off);
}

// arch025-N2a: every WHIRL_* variable numerics::plan reads (Request::env, its A/B switches, Request::runtime) is kept by
// the tier fingerprint (not on its skip list), and changes Plan::bitsKey() in at least one context (value x model x
// mode x card x path) — so a bit-affecting variable can never be in the plan but invisible to bitsKey, or in bitsKey
// but skipped by the fingerprint. Exceptions with their owner: FD_SPLITS (attention splits: resolveFdSplits, the KV
// step N3; noteSwitches records it).
static void testFingerprintCoversBitsKey() {
    namespace nu = whirl::numerics;
    auto tgt = [](bool moe, bool s8060) {
        nu::Target t;
        t.gfx1151 = s8060;
        t.has_mxfp4 = true;
        t.moe = moe;
        t.moe_mxfp4 = moe;
        t.k_gemm8 = t.k_gemm8_moe = t.k_gemm8t = t.k_gdn_wmma = !s8060;
        t.k_kv_q8h = t.k_kv_q4 = t.mtp = t.k_spec_sample = t.k_gdn_v0 = true;
        t.fast_kv_q8h = s8060;
        t.strix_profile = t.k_gdn_sx = s8060;
        t.gdn_strix_shape = true;
        return t;
    };
    std::vector<std::string> names;
    for (const auto& e : nu::EnvValues::names()) names.push_back(e.name);
    for (std::size_t i = 0; nu::k_numerics_switches[i] != nullptr; ++i) names.push_back(nu::k_numerics_switches[i]);
    for (const auto& e : nu::RuntimeValues::names()) names.push_back(e.name);
    const std::vector<std::string> outside = {"FD_SPLITS"};
    const char* values[] = {"0", "1", "", "2", "5", "f16", "q4", "512", "0x3", "0.5", "64", "4096"};
    std::size_t n_checked = 0;
    for (const std::string& n : names) {
        CHECK(fingerprintEnvKept("WHIRL_" + n));
        if (std::find(outside.begin(), outside.end(), n) != outside.end()) continue;
        bool changes = false;
        for (const bool moe : {false, true})
            for (const bool s8060 : {false, true})
                for (const char* mode : {"precise", "balance", "fast"})
                    for (const char* v : values)
                        for (int path = 0; path < 3 && !changes; ++path) {
                            nu::Request r = nu::parseMode(mode, "t");
                            r.load_overrides = true;
                            if (path == 0) {
                                nu::Raw* f = r.env.find(n);
                                if (!f) continue;
                                *f = std::string(v);
                            } else {
                                r.front = path == 1 ? nu::Front::cli : nu::Front::server;
                                bool found = false;
                                for (const auto& e : nu::RuntimeValues::names())
                                    if (n == e.name) {
                                        r.runtime.*e.field = std::string(v);
                                        found = true;
                                    }
                                if (!found) continue;
                            }
                            nu::Request r0 = nu::parseMode(mode, "t");
                            r0.load_overrides = true;
                            r0.front = r.front;
                            const nu::Target t = tgt(moe, s8060);
                            try {
                                changes = nu::plan(r, t).bitsKey() != nu::plan(r0, t).bitsKey();
                            } catch (const std::exception&) {  // a value loadOrTune rejects (RELAX_K=f16): next
                            }
                        }
        if (!changes) std::printf("FAIL WHIRL_%s never changes numerics::Plan::bitsKey()\n", n.c_str());
        CHECK(changes);
        ++n_checked;
    }
    CHECK(n_checked + outside.size() == names.size());
    std::printf("  fingerprint covers bitsKey: %zu plan variables, %zu outside the plan\n", n_checked, outside.size());
}

// arch025 PL1 (high risk of the plan: a changed environment enumeration makes every SSD entry miss). Pins:
//  (1) fingerprintEnv() == a verbatim copy of the v0.2.4 / arch025-base loop (GetEnvironmentStringsW, narrow, the
//      '=' search from 1, fingerprintEnvKept), same pairs in the same order, on the live environment and with a fixed
//      set added (value with '=', non-ASCII value, lower-case name, a skipped name);
//  (2) the fixed set, filtered from fingerprintEnv(), gives exactly the literal pairs below;
//  (3) tierFingerprint of fixed inputs + that set == 0xe30c1ff63c026f60, computed independently by
//      scratch\a25_pl1\fp_expected.py (FNV-1a re-implementation; tierFingerprint itself is unchanged by PL1).
// arch025-S0a recorded no fingerprint value for arch025-base, so (1) is the base pin: the reference IS the base code.
#ifdef _WIN32
static std::vector<std::pair<std::string, std::string>> baseFingerprintEnv() {
    std::vector<std::pair<std::string, std::string>> kv;
    if (wchar_t* blk = GetEnvironmentStringsW()) {
        for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) {
            const std::string s = whirl::narrow(p);
            const std::size_t eq = s.find('=', 1);
            if (eq == std::string::npos) continue;
            std::string k = s.substr(0, eq);
            if (fingerprintEnvKept(k)) kv.emplace_back(k, s.substr(eq + 1));
        }
        FreeEnvironmentStringsW(blk);
    }
    return kv;
}
#endif

static void testFingerprintPlatformEnv() {
#ifdef _WIN32
    CHECK(fingerprintEnv() == baseFingerprintEnv());  // the environment the test runs in
    _putenv_s("WHIRL_PL1T_B", "2");
    _putenv_s("WHIRL_PL1T_A", "1");
    _putenv_s("whirl_pl1t_lower", "x");
    _putenv_s("WHIRL_PL1T_EQ", "a=b");
    _wputenv_s(L"WHIRL_PL1T_U", L"\u00e9\u4e2d");
    _putenv_s("WHIRL_KV_SSD_DIR", "D:\\pl1");  // skipped name
    const auto got = fingerprintEnv();
    CHECK(got == baseFingerprintEnv());
    std::vector<std::pair<std::string, std::string>> mine;
    for (const auto& p : got) {
        std::string up = p.first;
        for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (up.rfind("WHIRL_PL1T_", 0) == 0 || up == "WHIRL_KV_SSD_DIR") mine.push_back(p);
    }
    std::vector<std::pair<std::string, std::string>> want = {
        {"WHIRL_PL1T_A", "1"}, {"WHIRL_PL1T_B", "2"}, {"WHIRL_PL1T_EQ", "a=b"}, {"WHIRL_PL1T_U", "\xc3\xa9\xe4\xb8\xad"},
        {"whirl_pl1t_lower", "x"}};
    auto sorted = mine;
    std::sort(sorted.begin(), sorted.end());
    CHECK(sorted == want);
    FingerprintInputs in;
    in.exe_id = 0x1234abcd;
    in.model_base = "Swift-1.5-27B-MXFP4.gguf";
    in.tensors = 851;
    in.bytes = 16'000'000'000ull;
    in.kv_name = "q8h";
    in.use_mtp = true;
    in.n_gdn = 48;
    in.ck_bytes = 1 << 20;
    in.page_bytes = 1 << 18;
    in.max_batch = 4096;
    in.numerics = "balance (items a,b)";
    in.gpu_arch = "gfx1201";
    in.gpu_name = "AMD Radeon AI PRO R9700";
    in.kernel_set = "gfx1201 co 123";
    in.env = mine;
    const std::uint64_t fp = tierFingerprint(in);
    if (fp != 0xe30c1ff63c026f60ull) std::printf("  PL1 fingerprint 0x%016llx (expected 0xe30c1ff63c026f60)\n",
                                                 static_cast<unsigned long long>(fp));
    CHECK(fp == 0xe30c1ff63c026f60ull);
    for (const char* k : {"WHIRL_PL1T_B", "WHIRL_PL1T_A", "whirl_pl1t_lower", "WHIRL_PL1T_EQ", "WHIRL_PL1T_U", "WHIRL_KV_SSD_DIR"})
        _putenv_s(k, "");
    CHECK(fingerprintEnv() == baseFingerprintEnv());
#endif
}
