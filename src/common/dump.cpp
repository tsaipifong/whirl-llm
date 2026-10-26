// Baseline dumps (arch025-S0a), see whirl/dump.h.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/dump.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <format>

namespace whirl::dump {

bool envOn(const char* name) {
    const std::string key = std::string("WHIRL_") + name;
#ifdef _WIN32
    char* v = nullptr;
    std::size_t n = 0;
    if (_dupenv_s(&v, &n, key.c_str()) != 0 || v == nullptr) return false;
    const std::string s(v);
    std::free(v);
#else
    const char* v = std::getenv(key.c_str());
    if (v == nullptr) return false;
    const std::string s(v);
#endif
    return !s.empty() && s != "0";
}

std::string kvLine(std::string_view site, const numerics::Request& r, bool moe, const numerics::KvCaps& caps,
                   std::optional<KvFormat> forced, const numerics::KvChoice& c) {
    return std::format("D kv: site={} mode={} items={:#x} custom={} moe={} caps=q8h:{},q8v:{},q4:{},fast_q8h:{} forced={} -> kv={} debug={} why={}",
                       site, numerics::modeName(r.mode), r.items, r.custom ? 1 : 0, moe ? 1 : 0, caps.q8h ? 1 : 0,
                       caps.q8v ? 1 : 0, caps.q4 ? 1 : 0, caps.fast_q8h ? 1 : 0,
                       forced ? kvFormatName(*forced) : "none", kvFormatName(c.kv), c.debug ? 1 : 0, c.why);
}

const numerics::KvChoice& kv(std::string_view site, const numerics::Request& r, bool moe, const numerics::KvCaps& caps,
                             std::optional<KvFormat> forced, const numerics::KvChoice& c) {
    if (envOn("DUMP_KV")) {
        const std::string l = kvLine(site, r, moe, caps, forced, c);
        std::fprintf(stderr, "%s\n", l.c_str());
        std::fflush(stderr);
    }
    return c;
}

namespace {
std::atomic<bool> g_dump_only{false};
}

void setDumpOnly(bool on) { g_dump_only.store(on); }
bool dumpOnly() { return g_dump_only.load(); }

}  // namespace whirl::dump
