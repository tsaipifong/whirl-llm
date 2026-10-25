// Launch log (ARCH-1 K-0), see launch_log.h.
// SPDX-License-Identifier: Apache-2.0

#include "launch_log.h"

#include "platform/env.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace whirl::hip::launchlog {

namespace {

// bug-envpaths: UTF-8 of the wide value (was std::getenv: the ANSI code page), opened with _wfopen below
std::string folder() { return platform::env::getUtf8("WHIRL_LAUNCH_LOG").value_or(std::string()); }

struct Key {
    std::string name;
    unsigned g[3], b[3], shared;
    bool operator<(const Key& o) const {
        return std::tie(name, g[0], g[1], g[2], b[0], b[1], b[2], shared) <
               std::tie(o.name, o.g[0], o.g[1], o.g[2], o.b[0], o.b[1], o.b[2], o.shared);
    }
};

constexpr std::uint64_t kFnvBasis = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;
void fnv(std::uint64_t& h, const void* p, std::size_t len) {
    const auto* c = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < len; ++i) h = (h ^ c[i]) * kFnvPrime;
}

struct Alloc {
    std::uint64_t bytes, ordinal;
};

struct Recorder {
    std::mutex mu;
    bool closed = false;  // written at exit; later launches are ignored
    std::unordered_map<const void*, std::string> names;
    std::map<Key, std::uint64_t> counts;
    std::map<std::pair<Key, std::uint64_t>, std::uint64_t> arg_counts;
    std::map<std::uint64_t, Alloc> allocs;  // live device allocations by base address
    std::uint64_t n_alloc = 0;
    std::uint64_t n = 0;
    std::uint64_t seq = kFnvBasis;   // FNV-1a 64 over the launch sequence (names, not addresses)
    std::uint64_t aseq = kFnvBasis;  // the same with each launch's argument digest

    // one 8-byte argument word: inside an allocation (one past its end included) -> (ordinal, offset); below the
    // next allocation by at most its size (a base shifted back by a row offset, e.g. KvArgs k - r0 * row) ->
    // (ordinal, -distance); anything else as it is. Called with mu held.
    // raw: WHIRL_LAUNCH_LOG_RAW set -> each word is also written out ("kind:ordinal:offset" or "v:value", then
    // "=raw" with the unnormalised value) so two runs can be diffed launch by launch (debug aid, arch6e)
    // arch024-S0a (REVIEW-ARCH6-code C-1): a word below kLowestAddress is a value, whatever allocation it may seem to
    // hit: 0 is an unused pointer (single-matrix GvArgs w[1..2], the f16 pool's KvArgs ks / vs) and a small u64 is a
    // scalar (forward.cpp w.row_bytes); with one allocation at least as large as its base address they fell into the
    // "below" rule as (ordinal, base - v), which depends on where that allocation lies. Windows never maps the lowest
    // 64 KiB of an address space (the null-pointer region), so no device address and no allocation base is below it.
    static constexpr std::uint64_t kLowestAddress = 0x10000;
    void mixWord(std::uint64_t& h, std::uint64_t v, std::string* raw) const {
        char b[96];
        if (v < kLowestAddress) {
            const std::uint64_t rec[2] = {0, v};
            fnv(h, rec, sizeof rec);
            if (raw) { std::snprintf(b, sizeof b, " v:%llx", (unsigned long long)v); *raw += b; }
            return;
        }
        auto it = allocs.upper_bound(v);  // first base > v
        if (it != allocs.begin()) {
            const auto& [base, a] = *std::prev(it);
            if (v - base <= a.bytes) {
                const std::uint64_t rec[3] = {1, a.ordinal, v - base};
                fnv(h, rec, sizeof rec);
                if (raw) { std::snprintf(b, sizeof b, " in:%llu:%llu=%llx", (unsigned long long)a.ordinal, (unsigned long long)(v - base), (unsigned long long)v); *raw += b; }
                return;
            }
        }
        if (it != allocs.end() && it->first - v <= it->second.bytes) {
            const std::uint64_t rec[3] = {2, it->second.ordinal, it->first - v};
            fnv(h, rec, sizeof rec);
            if (raw) { std::snprintf(b, sizeof b, " below:%llu:%llu=%llx", (unsigned long long)it->second.ordinal, (unsigned long long)(it->first - v), (unsigned long long)v); *raw += b; }
            return;
        }
        const std::uint64_t rec[2] = {0, v};
        fnv(h, rec, sizeof rec);
        if (raw) { std::snprintf(b, sizeof b, " v:%llx", (unsigned long long)v); *raw += b; }
    }
    bool raw_on = std::getenv("WHIRL_LAUNCH_LOG_RAW") != nullptr;
    std::vector<std::string> raw_lines;
    // arch024-S0a (REVIEW-ARCH6-code C-2): the per-launch dump is held in memory until exit, ~7 KB a line with a
    // GdnSegs argument; at most raw_max lines are kept (WHIRL_LAUNCH_LOG_RAW_MAX, default kRawMaxDefault), the rest
    // only counted and named in the dump's last line, so a long process (the server) cannot grow without bound
    static constexpr std::size_t kRawMaxDefault = 200000;
    std::size_t raw_max = [] {
        const char* v = std::getenv("WHIRL_LAUNCH_LOG_RAW_MAX");
        if (v == nullptr) return kRawMaxDefault;
        char* end = nullptr;
        const unsigned long long n = std::strtoull(v, &end, 10);
        return end != v ? static_cast<std::size_t>(n) : kRawMaxDefault;
    }();
    std::uint64_t raw_dropped = 0;
    void keepRaw(std::string&& line) {
        if (raw_lines.size() < raw_max) raw_lines.push_back(std::move(line));
        else ++raw_dropped;
    }

    void write() {
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
        std::FILE* f = _wfopen(std::filesystem::path(std::u8string(path.begin(), path.end())).c_str(), L"wb");  // bug-envpaths
        if (f == nullptr) return;
        std::fprintf(f, "# whirl launch log v1\n# cmd: %s\n# launches %llu seqhash %016llx kinds %zu\n", cmd.c_str(),
                     static_cast<unsigned long long>(n), static_cast<unsigned long long>(seq), counts.size());
        std::fprintf(f, "# argseqhash %016llx argkinds %zu\n", static_cast<unsigned long long>(aseq), arg_counts.size());
        for (const auto& [k, c] : counts)
            std::fprintf(f, "%s %ux%ux%u %ux%ux%u %u %llu\n", k.name.c_str(), k.g[0], k.g[1], k.g[2], k.b[0], k.b[1], k.b[2], k.shared,
                         static_cast<unsigned long long>(c));
        for (const auto& [ka, c] : arg_counts) {
            const Key& k = ka.first;
            std::fprintf(f, "@ %s %ux%ux%u %ux%ux%u %u %016llx %llu\n", k.name.c_str(), k.g[0], k.g[1], k.g[2], k.b[0], k.b[1], k.b[2],
                         k.shared, static_cast<unsigned long long>(ka.second), static_cast<unsigned long long>(c));
        }
        std::fclose(f);
        if (!raw_on) return;
        const std::string rp = folder() + "\\launch_" + std::to_string(pid) + ".raw.txt";
        std::FILE* g = _wfopen(std::filesystem::path(std::u8string(rp.begin(), rp.end())).c_str(), L"wb");  // bug-envpaths
        if (g == nullptr) return;
        for (const std::string& l : raw_lines) std::fprintf(g, "%s\n", l.c_str());
        if (raw_dropped > 0)
            std::fprintf(g, "# raw lines dropped %llu (kept the first %zu; WHIRL_LAUNCH_LOG_RAW_MAX)\n",
                         static_cast<unsigned long long>(raw_dropped), raw_max);
        std::fclose(g);
    }
};

// never destroyed (REVIEW-ARCH4 P3-6): a thread still launching during static destruction finds it alive
Recorder& rec() {
    static Recorder* r = new Recorder;
    return *r;
}

// the arguments hip::launch staged for this thread's next launch
thread_local bool t_staged = false;
thread_local std::uint64_t t_digest = 0;
thread_local std::string t_raw;

void writeAtExit() {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    if (r.closed) return;
    r.closed = true;
    r.write();
}

}  // namespace

const bool g_on = [] {
    if (folder().empty()) return false;
    (void)rec();
    std::atexit(writeAtExit);
    std::at_quick_exit(writeAtExit);
    return true;
}();

void onFunction(const void* f, const char* name) {
    if (!g_on || f == nullptr) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.names[f] = name;
}

void onAlloc(std::uint64_t base, std::uint64_t bytes) {
    if (!g_on || base == 0) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.allocs[base] = Alloc{bytes, r.n_alloc++};
}

void onFree(std::uint64_t base) {
    if (!g_on || base == 0) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.allocs.erase(base);
}

void testAlloc(std::uint64_t base, std::uint64_t bytes) {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.allocs[base] = Alloc{bytes, r.n_alloc++};
}

void serverProcess() {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    const char* v = std::getenv("WHIRL_LAUNCH_LOG_RAW");
    r.raw_on = v != nullptr && std::strcmp(v, "server") == 0;
}

RawStats rawStats() {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    return RawStats{r.raw_on, r.raw_lines.size(), r.raw_dropped, r.raw_max};
}

void testRaw(bool on, std::size_t max_lines) {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.raw_on = on;
    r.raw_max = max_lines;
    r.raw_lines.clear();
    r.raw_dropped = 0;
}

void testRawPush(const char* line) {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.keepRaw(std::string(line));
}

void testClearAllocs() {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.allocs.clear();
    r.n_alloc = 0;
}

namespace {
// the argument digest (see launch_log.h); called with r.mu held; raw != nullptr also writes the words out
std::uint64_t digestLocked(const Recorder& r, void* const* params, const std::uint32_t* sizes, const PtrWords* ptr_words,
                           std::size_t n, std::string* raw) {
    std::uint64_t h = kFnvBasis;
    if (raw) raw->clear();
    for (std::size_t i = 0; i < n; ++i) {
        const auto* p = static_cast<const unsigned char*>(params[i]);
        const std::uint32_t sz = sizes[i];
        fnv(h, &sz, sizeof sz);
        if (raw) *raw += " |" + std::to_string(sz);
        std::uint32_t off = 0;
        const PtrWords may_ptr = ptr_words != nullptr ? ptr_words[i] : nullptr;
        for (; off + 8 <= sz; off += 8) {
            std::uint64_t v;
            std::memcpy(&v, p + off, 8);
            if (may_ptr == nullptr || may_ptr(off / 8)) {
                r.mixWord(h, v, raw);
            } else {  // not an address by the argument's type (arch6f): its value, whatever allocation it may hit
                const std::uint64_t rec[2] = {0, v};
                fnv(h, rec, sizeof rec);
                if (raw) {
                    char b[40];
                    std::snprintf(b, sizeof b, " v:%llx", (unsigned long long)v);
                    *raw += b;
                }
            }
        }
        fnv(h, p + off, sz - off);
        if (raw)
            for (std::uint32_t j = off; j < sz; ++j) {
                char b[8];
                std::snprintf(b, sizeof b, "%s%02x", j == off ? " t:" : "", p[j]);
                *raw += b;
            }
    }
    return h;
}
}  // namespace

std::uint64_t argDigest(void* const* params, const std::uint32_t* sizes, const PtrWords* ptr_words, std::size_t n) {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    return digestLocked(r, params, sizes, ptr_words, n, nullptr);
}

void stageArgs(void* const* params, const std::uint32_t* sizes, const PtrWords* ptr_words, std::size_t n) {
    if (!g_on) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    t_digest = digestLocked(r, params, sizes, ptr_words, n, r.raw_on ? &t_raw : nullptr);
    t_staged = true;
}

void onLaunch(const void* f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz, unsigned shared) {
    if (!g_on) return;
    const std::uint64_t digest = t_staged ? t_digest : 0;
    t_staged = false;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    if (r.closed) return;
    const auto it = r.names.find(f);
    Key k{it != r.names.end() ? it->second : std::string("?"), {gx, gy, gz}, {bx, by, bz}, shared};
    r.n += 1;
    fnv(r.seq, k.name.data(), k.name.size());
    const unsigned geo[7] = {gx, gy, gz, bx, by, bz, shared};
    fnv(r.seq, geo, sizeof geo);
    fnv(r.aseq, k.name.data(), k.name.size());
    fnv(r.aseq, geo, sizeof geo);
    fnv(r.aseq, &digest, sizeof digest);
    r.counts[k] += 1;
    if (r.raw_on) {
        char b[160];
        std::snprintf(b, sizeof b, "%llu %s %ux%ux%u %ux%ux%u %u %016llx", (unsigned long long)r.n, k.name.c_str(), gx, gy, gz, bx, by, bz, shared,
                      (unsigned long long)digest);
        r.keepRaw(std::string(b) + t_raw);
        t_raw.clear();
    }
    r.arg_counts[{std::move(k), digest}] += 1;
}

}  // namespace whirl::hip::launchlog
