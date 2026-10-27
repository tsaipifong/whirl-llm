// GPU-side leak count (arch025-Q2), see include/whirl/leak_count.h. Written for WHIRL.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/leak_count.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <mutex>
#include <unordered_map>

namespace whirl::hip::leakcount {

namespace {

bool envOn() {
    const char* v = std::getenv("WHIRL_LEAK_CHECK");
    return v != nullptr && *v != 0 && std::strcmp(v, "0") != 0;
}

struct Counts {
    std::mutex mu;
    std::unordered_map<std::uint64_t, std::uint64_t> dev, host;  // live block -> bytes
    Snapshot now;
    int units = 0, leaks = 0;
    void reset() {
        dev.clear();
        host.clear();
        now = {};
        units = leaks = 0;
    }
};
Counts& counts() {
    static Counts* c = new Counts;  // never destroyed: hip::free may run from static destructors
    return *c;
}

void add(std::string& s, std::int64_t n, const char* one, const char* many) {
    if (n <= 0) return;
    if (!s.empty()) s += ", ";
    s += std::format("{} {}", n, n == 1 ? one : many);
}

}  // namespace

bool g_on = envOn();

void onDevAlloc(std::uint64_t base, std::uint64_t bytes) {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    c.dev[base] = bytes;
    c.now.dev += 1;
    c.now.dev_bytes += static_cast<std::int64_t>(bytes);
}

void onDevFree(std::uint64_t base) {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    const auto it = c.dev.find(base);
    if (it == c.dev.end()) return;  // not made while counting
    c.now.dev -= 1;
    c.now.dev_bytes -= static_cast<std::int64_t>(it->second);
    c.dev.erase(it);
}

void onHostAlloc(std::uint64_t base, std::uint64_t bytes) {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    c.host[base] = bytes;
    c.now.host += 1;
    c.now.host_bytes += static_cast<std::int64_t>(bytes);
}

void onHostFree(std::uint64_t base) {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    const auto it = c.host.find(base);
    if (it == c.host.end()) return;
    c.now.host -= 1;
    c.now.host_bytes -= static_cast<std::int64_t>(it->second);
    c.host.erase(it);
}

void onStream(int delta) {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    c.now.streams += delta;
}

void onEvent(int delta) {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    c.now.events += delta;
}

Snapshot snapshot() {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    return c.now;
}

std::string leaked(const Snapshot& b, const Snapshot& a) {
    std::string s;
    if (a.dev > b.dev) {
        add(s, a.dev - b.dev, "device allocation", "device allocations");
        s += std::format(" ({} bytes)", a.dev_bytes - b.dev_bytes);
    }
    if (a.host > b.host) {
        add(s, a.host - b.host, "pinned host allocation", "pinned host allocations");
        s += std::format(" ({} bytes)", a.host_bytes - b.host_bytes);
    }
    add(s, a.streams - b.streams, "stream", "streams");
    add(s, a.events - b.events, "event", "events");
    return s;
}

bool check(const char* unit, const Snapshot& before) {
    if (!g_on) return false;
    const std::string s = leaked(before, snapshot());
    Counts& c = counts();
    {
        std::lock_guard<std::mutex> lk(c.mu);
        c.units += 1;
        if (!s.empty()) c.leaks += 1;
    }
    if (s.empty()) return false;
    std::printf("LEAK_CHECK: %s: %s still alive\n", unit, s.c_str());
    std::fflush(stdout);
    return true;
}

int report() {
    if (!g_on) return 0;
    Counts& c = counts();
    int units = 0, leaks = 0;
    {
        std::lock_guard<std::mutex> lk(c.mu);
        units = c.units;
        leaks = c.leaks;
    }
    if (leaks == 0)
        std::printf("leak check: ok (%d units)\n", units);
    else
        std::printf("leak check: FAIL (%d of %d units leaked)\n", leaks, units);
    std::fflush(stdout);
    return leaks;
}

void testEnable(bool on) {
    Counts& c = counts();
    std::lock_guard<std::mutex> lk(c.mu);
    c.reset();
    g_on = on;
}

}  // namespace whirl::hip::leakcount
