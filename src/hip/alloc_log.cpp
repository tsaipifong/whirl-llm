// Allocation log (arch025-S0a), see alloc_log.h.
// SPDX-License-Identifier: Apache-2.0

#include "alloc_log.h"

#include "platform/env.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace whirl::hip::alloclog {

namespace {

// arch025 PL1b: through platform::env::get; read where it was (g_on's static initializer and the write at exit)
std::string folder() { return platform::env::get("WHIRL_ALLOC_LOG").value_or(std::string()); }

}  // namespace

// bug-launchlogroot: no folder, no file ("" + "\alloc_" would be the root of the current drive)
std::string exitPath(const std::string& dir, unsigned long pid) {
    if (dir.empty()) return std::string();
    return dir + "\\alloc_" + std::to_string(pid) + ".txt";
}

std::string exitPathNow(unsigned long pid) { return exitPath(folder(), pid); }

namespace {

constexpr std::uint64_t kFnvBasis = 1469598103934665603ull;
constexpr std::uint64_t kFnvPrime = 1099511628211ull;

enum Kind : std::uint64_t { kDev = 1, kHost = 2, kFree = 3, kHostFree = 4 };
const char* kindName(std::uint64_t k) {
    switch (k) {
        case kDev: return "dev";
        case kHost: return "host";
        case kFree: return "free";
        default: return "hostfree";
    }
}

struct Live {
    std::uint64_t bytes, ordinal;
};

struct Recorder {
    std::mutex mu;
    bool closed = false;
    bool list = true;
    std::size_t max_lines = kMaxLines;
    std::unordered_map<std::uint64_t, Live> dev, host;  // live allocations by base address
    std::uint64_t n_dev = 0, n_host = 0;                // ordinals handed out
    std::uint64_t n_free = 0, n_hostfree = 0, n_unknown = 0;
    std::uint64_t dev_bytes = 0, host_bytes = 0;        // allocated in total
    std::uint64_t live_dev = 0, live_host = 0, peak_dev = 0, peak_host = 0;
    std::uint64_t n = 0;
    std::uint64_t seq = kFnvBasis;
    std::vector<std::string> lines;
    std::uint64_t dropped = 0;

    void reset() {
        closed = false;
        dev.clear();
        host.clear();
        n_dev = n_host = n_free = n_hostfree = n_unknown = 0;
        dev_bytes = host_bytes = live_dev = live_host = peak_dev = peak_host = 0;
        n = 0;
        seq = kFnvBasis;
        lines.clear();
        dropped = 0;
    }

    // one event: hashed always, listed when the list is on (mu held)
    void event(std::uint64_t kind, std::uint64_t bytes, std::uint64_t ordinal, bool known) {
        const std::uint64_t rec[4] = {kind, bytes, ordinal, known ? 1u : 0u};
        const auto* c = reinterpret_cast<const unsigned char*>(rec);
        for (std::size_t i = 0; i < sizeof rec; ++i) seq = (seq ^ c[i]) * kFnvPrime;
        const std::uint64_t idx = n++;
        if (!list) return;
        if (lines.size() >= max_lines) {
            ++dropped;
            return;
        }
        char b[96];
        if (kind == kDev || kind == kHost)
            std::snprintf(b, sizeof b, "%llu %s %llu #%llu", (unsigned long long)idx, kindName(kind), (unsigned long long)bytes,
                          (unsigned long long)ordinal);
        else if (known)
            std::snprintf(b, sizeof b, "%llu %s #%llu %llu", (unsigned long long)idx, kindName(kind), (unsigned long long)ordinal,
                          (unsigned long long)bytes);
        else
            std::snprintf(b, sizeof b, "%llu %s #? 0", (unsigned long long)idx, kindName(kind));
        lines.emplace_back(b);
    }

    void alloc(bool is_dev, std::uint64_t base, std::uint64_t bytes) {
        if (closed) return;
        auto& m = is_dev ? dev : host;
        std::uint64_t& ord = is_dev ? n_dev : n_host;
        std::uint64_t& live = is_dev ? live_dev : live_host;
        std::uint64_t& peak = is_dev ? peak_dev : peak_host;
        (is_dev ? dev_bytes : host_bytes) += bytes;
        m[base] = Live{bytes, ord};
        live += bytes;
        peak = std::max(peak, live);
        event(is_dev ? kDev : kHost, bytes, ord, true);
        ++ord;
    }

    void release(bool is_dev, std::uint64_t base) {
        if (closed) return;
        auto& m = is_dev ? dev : host;
        ++(is_dev ? n_free : n_hostfree);
        const auto it = m.find(base);
        if (it == m.end()) {
            ++n_unknown;
            event(is_dev ? kFree : kHostFree, 0, 0, false);
            return;
        }
        (is_dev ? live_dev : live_host) -= it->second.bytes;
        event(is_dev ? kFree : kHostFree, it->second.bytes, it->second.ordinal, true);
        m.erase(it);
    }

    std::string text() const {
        std::string s;
        char b[256];
        std::snprintf(b, sizeof b, "# events %llu seqhash %016llx\n", (unsigned long long)n, (unsigned long long)seq);
        s += b;
        std::snprintf(b, sizeof b, "# dev %llu allocs %llu bytes, %llu frees (%zu live, %llu bytes; peak %llu bytes)\n",
                      (unsigned long long)n_dev, (unsigned long long)dev_bytes, (unsigned long long)n_free, dev.size(),
                      (unsigned long long)live_dev, (unsigned long long)peak_dev);
        s += b;
        std::snprintf(b, sizeof b, "# host %llu allocs %llu bytes, %llu frees (%zu live, %llu bytes; peak %llu bytes)\n",
                      (unsigned long long)n_host, (unsigned long long)host_bytes, (unsigned long long)n_hostfree, host.size(),
                      (unsigned long long)live_host, (unsigned long long)peak_host);
        s += b;
        std::snprintf(b, sizeof b, "# unknown frees %llu\n# list %s\n", (unsigned long long)n_unknown, list ? "on" : "off");
        s += b;
        for (const std::string& l : lines) {
            s += l;
            s += '\n';
        }
        if (dropped > 0) {
            std::snprintf(b, sizeof b, "# lines dropped %llu (kept the first %zu)\n", (unsigned long long)dropped, max_lines);
            s += b;
        }
        return s;
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
        const std::string path = exitPathNow(pid);
        if (path.empty()) return;  // the folder was cleared after the static initializer saw it: write nothing
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) return;
        const std::string body = text();
        std::fprintf(f, "# whirl alloc log v1\n# cmd: %s\n", cmd.c_str());
        std::fwrite(body.data(), 1, body.size(), f);
        std::fclose(f);
    }
};

// never destroyed: a thread still freeing during static destruction finds it alive
Recorder& rec() {
    static Recorder* r = new Recorder;
    return *r;
}

void writeAtExit() {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    if (r.closed) return;
    r.closed = true;
    r.write();
}

}  // namespace

bool g_on = [] {
    if (folder().empty()) return false;
    (void)rec();
    std::atexit(writeAtExit);
    std::at_quick_exit(writeAtExit);
    return true;
}();

void onDevAlloc(std::uint64_t base, std::uint64_t bytes) {
    if (!g_on) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.alloc(true, base, bytes);
}

void onDevFree(std::uint64_t base) {
    if (!g_on) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.release(true, base);
}

void onHostAlloc(std::uint64_t base, std::uint64_t bytes) {
    if (!g_on) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.alloc(false, base, bytes);
}

void onHostFree(std::uint64_t base) {
    if (!g_on) return;
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.release(false, base);
}

void serverProcess() {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    const std::optional<std::string> v = platform::env::get("WHIRL_ALLOC_LOG_RAW");
    r.list = v.has_value() && *v == "server";
}

void testEnable(bool on, bool list, std::size_t max_lines) {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    r.reset();
    r.list = list;
    r.max_lines = max_lines;
    g_on = on;
}

std::string testText() {
    Recorder& r = rec();
    std::lock_guard<std::mutex> lk(r.mu);
    return r.text();
}

}  // namespace whirl::hip::alloclog
