// GPU busy-ratio instrumentation (WHIRL_BUSY), see include/whirl/hip.h.
// SPDX-License-Identifier: Apache-2.0
//
// Off (WHIRL_BUSY unset or 0): every hook is an inline test of busy::g_mode
// and nothing else runs; no event, no allocation, no file.
//
// WHIRL_BUSY=3  M1 only: per-region counters of launches, host syncs and
//               H<->D / D2D bytes (thread-local: the calling thread only).
// WHIRL_BUSY=1  M1 + M2: also two timing events per sampled region (region
//               begin and just before the region's natural sync) = GPU span.
// WHIRL_BUSY_EVERY=N  span-sample every N-th region of each kind (default 8);
//                     counters are written for every region.
// WHIRL_BUSY_OUT=path JSON lines output (default whirl_busy.jsonl).

#include "whirl/hip.h"

#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace whirl::hip::busy {

namespace {

int readMode() {
    char* buf = nullptr;
    std::size_t len = 0;
    int m = 0;
    if (_dupenv_s(&buf, &len, "WHIRL_BUSY") == 0 && buf != nullptr) {
        m = std::atoi(buf);
        std::free(buf);
    }
    return (m == 1 || m == 3) ? m : 0;
}

std::string envStr(const char* k, const char* def) {
    char* buf = nullptr;
    std::size_t len = 0;
    std::string r = def;
    if (_dupenv_s(&buf, &len, k) == 0 && buf != nullptr) {
        r = buf;
        std::free(buf);
    }
    return r;
}

using Clock = std::chrono::steady_clock;
double nowUs() {
    return std::chrono::duration<double, std::micro>(Clock::now().time_since_epoch()).count();
}

struct Counters {
    std::uint64_t launches = 0, syncs = 0, h2d_n = 0, d2h_n = 0, d2d_n = 0, h2d_b = 0, d2h_b = 0, d2d_b = 0, memsets = 0,
                  graphs = 0;
    double sync_us = 0;  // host time blocked in synchronous calls
};

struct KindState {
    std::uint64_t index = 0;  // regions seen
};

struct Pending {
    hipEvent_t e0, e1;
    std::string head;  // JSON fields written so far (without the span)
};

struct Thread {
    Counters c;
    // current region
    bool active = false;
    int depth = 0;  // nested regions inside the active one (counted by the outer)
    const char* kind = "";
    Counters c0;
    double t0 = 0, t_mark = 0;
    bool sampled = false, marked = false;
    hipStream_t stream = nullptr;
    hipEvent_t e0 = nullptr, e1 = nullptr;
    hipStream_t last_stream = nullptr;
};

thread_local Thread t_;

std::mutex g_mu;
std::FILE* g_out = nullptr;
std::uint64_t g_lines = 0;
std::map<std::string, KindState> g_kinds;
std::vector<hipEvent_t> g_free_span;  // span event pairs
std::vector<Pending> g_pending;
int g_every = 8;

void writeLine(const std::string& s) {
    if (!g_out) {
        const std::string p = envStr("WHIRL_BUSY_OUT", "whirl_busy.jsonl");
        g_out = std::fopen(p.c_str(), "ab");
        if (!g_out) return;
        std::setvbuf(g_out, nullptr, _IOFBF, 1 << 16);
    }
    std::fwrite(s.data(), 1, s.size(), g_out);
    std::fputc('\n', g_out);
    // flushed per line (a few us per region): the server is usually killed, not exited
    std::fflush(g_out);
    ++g_lines;
}

hipEvent_t spanEvent() {
    if (!g_free_span.empty()) {
        hipEvent_t e = g_free_span.back();
        g_free_span.pop_back();
        return e;
    }
    hipEvent_t e = nullptr;
    (void)hipEventCreate(&e);
    return e;
}

void drainPending(bool block) {
    for (std::size_t i = 0; i < g_pending.size();) {
        Pending& p = g_pending[i];
        if (block) (void)hipEventSynchronize(p.e1);
        if (hipEventQuery(p.e1) != hipSuccess) {
            ++i;
            continue;
        }
        float ms = 0;
        (void)hipEventElapsedTime(&ms, p.e0, p.e1);
        char b[64];
        std::snprintf(b, sizeof b, ",\"span_us\":%.1f}", ms * 1000.0);
        writeLine(p.head + b);
        g_free_span.push_back(p.e0);
        g_free_span.push_back(p.e1);
        g_pending.erase(g_pending.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

}  // namespace

int g_mode = readMode();

void launchHookImpl(Stream s) {
    Thread& t = t_;
    t.c.launches += 1;
    t.last_stream = static_cast<hipStream_t>(s);
}

void copyHookImpl(int dir, std::uint64_t bytes) {
    Thread& t = t_;
    {
        if (dir == 0) {
            t.c.h2d_n += 1;
            t.c.h2d_b += bytes;
        } else if (dir == 1) {
            t.c.d2h_n += 1;
            t.c.d2h_b += bytes;
        } else if (dir == 2) {
            t.c.d2d_n += 1;
            t.c.d2d_b += bytes;
        } else if (dir == 3) {
            t.c.memsets += 1;
        }
    }
}

void syncHookImpl(double us) {
    Thread& t = t_;
    t.c.syncs += 1;
    t.c.sync_us += us;
}

double nowMicros() { return nowUs(); }

void graphHookImpl() { t_.c.graphs += 1; }

void beginRegionImpl(const char* kind, Stream s) {
    Thread& t = t_;
    if (t.active) {  // nested: the outer region keeps counting
        t.depth += 1;
        return;
    }
    std::lock_guard<std::mutex> lk(g_mu);
    static bool init = false;
    if (!init) {
        init = true;
        g_every = std::max(1, std::atoi(envStr("WHIRL_BUSY_EVERY", "8").c_str()));
    }
    drainPending(false);
    KindState& ks = g_kinds[kind];
    t.active = true;
    t.kind = kind;
    t.c0 = t.c;
    t.t0 = nowUs();
    t.t_mark = 0;
    t.marked = false;
    t.stream = static_cast<hipStream_t>(s ? s : t.last_stream);
    t.sampled = (ks.index % static_cast<std::uint64_t>(g_every)) == 0;
    ks.index += 1;
    t.e0 = t.e1 = nullptr;
    if (t.sampled && g_mode == 1 && t.stream) {
        t.e0 = spanEvent();
        t.e1 = spanEvent();
        (void)hipEventRecord(t.e0, t.stream);
    }
}

void markEndImpl() {
    Thread& t = t_;
    if (!t.active || t.marked) return;
    t.marked = true;
    t.t_mark = nowUs();
    if (t.e1) (void)hipEventRecord(t.e1, t.stream);
}

void endRegionImpl(const char* info) {
    Thread& t = t_;
    if (!t.active) return;
    if (t.depth > 0) {
        t.depth -= 1;
        return;
    }
    const double t1 = nowUs();
    std::lock_guard<std::mutex> lk(g_mu);
    t.active = false;
    const Counters& a = t.c0;
    const Counters& b = t.c;
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "{\"kind\":\"%s\",\"t0_us\":%.1f,\"host_us\":%.1f,\"enq_us\":%.1f,\"launch\":%llu,\"sync\":%llu,\"sync_us\":%.1f,"
                  "\"h2d_n\":%llu,\"h2d_b\":%llu,\"d2h_n\":%llu,\"d2h_b\":%llu,\"d2d_n\":%llu,\"d2d_b\":%llu,\"memset\":%llu,"
                  "\"graph\":%llu,\"mode\":%d,\"sampled\":%d",
                  t.kind, t.t0, t1 - t.t0, t.marked ? t.t_mark - t.t0 : -1.0,
                  static_cast<unsigned long long>(b.launches - a.launches), static_cast<unsigned long long>(b.syncs - a.syncs),
                  b.sync_us - a.sync_us, static_cast<unsigned long long>(b.h2d_n - a.h2d_n),
                  static_cast<unsigned long long>(b.h2d_b - a.h2d_b), static_cast<unsigned long long>(b.d2h_n - a.d2h_n),
                  static_cast<unsigned long long>(b.d2h_b - a.d2h_b), static_cast<unsigned long long>(b.d2d_n - a.d2d_n),
                  static_cast<unsigned long long>(b.d2d_b - a.d2d_b), static_cast<unsigned long long>(b.memsets - a.memsets),
                  static_cast<unsigned long long>(b.graphs - a.graphs), g_mode, t.sampled ? 1 : 0);
    std::string line = buf;
    if (info && *info) line += std::string(",") + info;
    if (t.e0) {
        if (!t.marked) (void)hipEventRecord(t.e1, t.stream);
        g_pending.push_back({t.e0, t.e1, line});
        t.e0 = t.e1 = nullptr;
        drainPending(false);
    } else {
        line += "}";
        writeLine(line);
    }
}

void flush() {
    if (!g_mode) return;
    std::lock_guard<std::mutex> lk(g_mu);
    drainPending(true);
    if (g_out) std::fflush(g_out);
}

}  // namespace whirl::hip::busy
