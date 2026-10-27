// Baseline dumps (arch025-S0a), see whirl/dump.h.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/dump.h"

#include "platform/env.h"

#include <atomic>
#include <cstdio>
#include <format>

namespace whirl::dump {

// arch025 PL1b: the _dupenv_s read is platform::env::get (the same _dupenv_s call, moved); still read on every call
bool switchOn(const char* name) {
    const std::optional<std::string> s = platform::env::get(std::string("WHIRL_") + name);
    return s.has_value() && !s->empty() && *s != "0";
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
    if (switchOn("DUMP_KV")) {
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

// bug-cli-dumponly: only these whirl commands reach qwen35::loadOrTune (the --dump-only exit); keep in step with
// src/cli/main.cpp (cmdChat, cmdBench, cmdSeqtest, goldenSingle / goldenPair; serve = serveMain)
std::string dumpOnlyRefusal(std::string_view cmd) {
    static constexpr std::string_view honours[] = {"chat", "bench", "seqtest", "golden", "serve"};
    for (std::string_view h : honours)
        if (cmd == h) return {};
    return std::format("'{}' does not support --dump-only (it does not load and tune a model); use it with chat, bench, "
                       "seqtest, golden or serve",
                       cmd);
}

}  // namespace whirl::dump
