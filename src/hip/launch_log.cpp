// Launch log (ARCH-1 K-0), see launch_log.h.
// SPDX-License-Identifier: Apache-2.0

#include "launch_log.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace whirl::hip::launchlog {

namespace {

std::string folder() {
    const char* v = std::getenv("WHIRL_LAUNCH_LOG");
    return v != nullptr ? std::string(v) : std::string();
}

struct Key {
    std::string name;
    unsigned g[3], b[3], shared;
    bool operator<(const Key& o) const {
        return std::tie(name, g[0], g[1], g[2], b[0], b[1], b[2], shared) <
               std::tie(o.name, o.g[0], o.g[1], o.g[2], o.b[0], o.b[1], o.b[2], o.shared);
    }
};

struct Recorder {
    std::mutex mu;
    std::unordered_map<const void*, std::string> names;
    std::map<Key, std::uint64_t> counts;
    std::uint64_t n = 0;
    std::uint64_t seq = 1469598103934665603ull;  // FNV-1a 64 over the launch sequence (names, not addresses)
    void mix(const void* p, std::size_t len) {
        const auto* c = static_cast<const unsigned char*>(p);
        for (std::size_t i = 0; i < len; ++i) seq = (seq ^ c[i]) * 1099511628211ull;
    }
    ~Recorder() {
        if (!g_on) return;
        std::string cmd;
#ifdef _WIN32
        const wchar_t* w = GetCommandLineW();
        const int len = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
        if (len > 1) {
            cmd.resize(static_cast<std::size_t>(len - 1));
            WideCharToMultiByte(CP_UTF8, 0, w, -1, cmd.data(), len, nullptr, nullptr);
        }
        const unsigned long pid = GetCurrentProcessId();
#else
        const unsigned long pid = 0;
#endif
        const std::string path = folder() + "\\launch_" + std::to_string(pid) + ".txt";
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) return;
        std::fprintf(f, "# whirl launch log v1\n# cmd: %s\n# launches %llu seqhash %016llx kinds %zu\n", cmd.c_str(),
                     static_cast<unsigned long long>(n), static_cast<unsigned long long>(seq), counts.size());
        for (const auto& [k, c] : counts)
            std::fprintf(f, "%s %ux%ux%u %ux%ux%u %u %llu\n", k.name.c_str(), k.g[0], k.g[1], k.g[2], k.b[0], k.b[1], k.b[2], k.shared,
                         static_cast<unsigned long long>(c));
        std::fclose(f);
    }
};

Recorder& rec() {
    static Recorder r;
    return r;
}

}  // namespace

const bool g_on = [] {
    if (folder().empty()) return false;
    (void)rec();  // constructed now, so it is destroyed (and writes) after anything that launches
    return true;
}();

void onFunction(const void* f, const char* name) {
    if (!g_on || f == nullptr) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.names[f] = name;
}

void onLaunch(const void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared) {
    if (!g_on) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    const auto it = r.names.find(f);
    Key k{it != r.names.end() ? it->second : std::string("?"), {gx, gy, gz}, {bx, by, bz}, shared};
    r.counts[k] += 1;
    r.n += 1;
    r.mix(k.name.data(), k.name.size());
    const unsigned geo[7] = {gx, gy, gz, bx, by, bz, shared};
    r.mix(geo, sizeof geo);
}

}  // namespace whirl::hip::launchlog
