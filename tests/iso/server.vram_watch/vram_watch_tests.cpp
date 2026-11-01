// whirl-iso-server-vram-watch-tests: module server.vram_watch (arch025-engine S9; progress/arch025-engine.DESIGN.md v3
// 2.3 L4, 4 S9): src/server/engine_vram.h (class VramState, the former Engine members vram_watch_ .. vram_last_t_ of
// engine.h:265-274 at 6a9f6be without prefill_tok_cum_ / crlf_norm_; class VramWatch, the former Engine member functions
// setVramProbe / vramNote / vramIdleTick of engine_vram.cpp, plus vramHealth and the accessors) and
// src/server/engine_vram.cpp, built with only its declared dependencies (server.governor and what it declares, the WDDM
// watch policy mem.wddm_watch, core.shared) - not engine.h (ARCH-1 8.5 isolation).
// The checks pin what the move must keep, with the expected values read from the 6a9f6be bodies (engine_vram.cpp,
// engine.cpp health, engine_decode.cpp's spill argument, idleWait's msSince(vram_last_t_)):
// - no probe: vramNote "" without sampling, vramIdleTick nullopt, /health vram fields unset, not spilling;
// - setVramProbe: the probe sampled twice; the baseline is `base` when it is ok, else the first sample (vramNote returns
//   wddmwatch::format(sample, baseline)); /health gets the second sample; the idle cadence is cfg.idle_sample_s (60 when
//   0); a first sample that is not ok drops the probe (Engine's vram_probe_ is nullptr afterwards, through the reference);
//   others is installed into Engine's vram_others_ (read when a warning is printed);
// - the spill warning: with warn, a spill at startup is logged once (not when startup_warned); vramNote logs a later
//   spill as a WARNING with the others note appended; vram_spill_ / vramSpilling = warn && the watch spills (warn false:
//   the numbers only, never spilling);
// - F15 order with the governor on: vramNote hands the sample to govTick first, the warning goes to govSetWarn (not
//   logged by vramNote) and govApply logs it after its own lines; govReduced then makes the idle cadence 5 s;
// - vramIdleTick samples only when due; vramLastSample is the time of the last vramNote sample;
// - layout: VramWatch holds references only (12), is not copyable, trivially destroyed; the accessors' signatures.
// SPDX-License-Identifier: Apache-2.0
#include "device/null_backend.h"
#include "iso/fake_server_model.h"
#include "iso/iso_test.h"
#include "server/engine_vram.h"
#include "server/log.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

namespace s = whirl::server;
namespace memgov = whirl::memgov;
namespace wddmwatch = whirl::wddmwatch;
using W = s::VramWatch;

static_assert(std::is_trivially_destructible_v<W>);
static_assert(!std::is_copy_constructible_v<W> && !std::is_copy_assignable_v<W>);
static_assert(sizeof(W) == 12 * sizeof(void*));
static_assert(std::is_default_constructible_v<s::VramState>);
// what the other modules call (S9 rewrites of their call sites)
static_assert(std::is_same_v<decltype(&W::vramNote), std::string (W::*)()>);
static_assert(std::is_same_v<decltype(&W::vramIdleTick), std::optional<s::TimePoint> (W::*)()>);
static_assert(std::is_same_v<decltype(&W::vramHealth), void (W::*)(s::EngineApi::Health&)>);
static_assert(std::is_same_v<decltype(&W::vramSpilling), bool (W::*)() const>);
static_assert(std::is_same_v<decltype(&W::vramLastSample), s::TimePoint (W::*)() const>);

constexpr std::uint32_t KP = whirl::qwen35::kv_page;
constexpr std::uint64_t kRowA = 16, kRowB = 48;
constexpr std::uint64_t MiB = 1ull << 20;

std::vector<std::string> g_lines;
void hook(void*, std::string_view l) { g_lines.emplace_back(l); }
// the level of the first captured line containing `text` (' ' = none); Log::line writes "<stamp> <L> <msg>"
char levelOf(std::string_view text) {
    for (const std::string& l : g_lines)
        if (const auto p = l.find(text); p != std::string::npos && p >= 2) return l[l.find(' ', l.find(' ') + 1) + 1];
    return ' ';
}
std::size_t countOf(std::string_view text) {
    std::size_t n = 0;
    for (const std::string& l : g_lines) n += l.find(text) != std::string::npos;
    return n;
}
std::size_t indexOf(std::string_view text) {
    for (std::size_t i = 0; i < g_lines.size(); ++i)
        if (g_lines[i].find(text) != std::string::npos) return i;
    return g_lines.size();
}

// what Engine holds, built as Engine builds it (members in Engine's order; sc_, gv_ then vw_ last); two slots
struct Rig {
    whirl::device::NullBackend nb;
    whirl::test::FakeServerModel m{nb, 10};
    s::EngineOptions opt;
    std::uint32_t ctx = 4096;
    s::EngineThread owner;
    whirl::chat::ChatTokens ct;
    std::unique_ptr<s::CkptStore> ck;
    s::BatchStats bst;
    std::uint64_t conv_bytes = 0, ssm_bytes = 0;
    std::vector<s::Slot> slots;
    std::function<wddmwatch::Sample()> vram_probe;
    std::function<std::optional<wddmwatch::Others>()> vram_others;
    s::VramState vst;
    s::GovState gst;
    std::atomic<std::uint32_t> ctx_pub{0};
    std::vector<s::KvArr> kv;
    std::unique_ptr<s::PagePool> pool;
    std::unique_ptr<s::PrefixStore> ps;
    std::unique_ptr<s::TierBridge> tb;
    s::SlotPagesCounters cnt;
    std::unique_ptr<s::DrafterCtx> dc;
    std::unique_ptr<s::SlotPages> sc;
    std::unique_ptr<s::GovernorGlue> gv;
    std::unique_ptr<W> vw;
    std::vector<s::DevPtr> mem;
    // the probe the server installs (server_main: hip::wddmMemInfo); returns `cur` and counts the calls
    wddmwatch::Sample cur;
    int probes = 0, others_reads = 0;
    explicit Rig(std::uint32_t pages) {
        ck = std::make_unique<s::CkptStore>(nb, m, s::CkptStore::Layout{0, 0, false});
        slots.resize(2);
        for (std::uint32_t i = 0; i < 2; ++i) slots[i].id = i;
        pool = std::make_unique<s::PagePool>(nb, m);
        ps = std::make_unique<s::PrefixStore>(ct, s::PrefixStore::Config{});
        tb = std::make_unique<s::TierBridge>(nb, m, ct, *ps, kv, s::TierBridge::Config{});
        dc = std::make_unique<s::DrafterCtx>(s::DrafterCtx{bst, m, opt});
        sc = std::make_unique<s::SlotPages>(s::SlotPages{*ck, m, nb, opt, *pool, *ps, slots, *tb, kv, conv_bytes, ssm_bytes, cnt, *dc});
        gv = std::make_unique<s::GovernorGlue>(*ck, ctx, m, nb, opt, owner, *pool, *ps, slots, *tb, vram_probe, vram_others, ctx_pub, *sc, gst);
        vw = std::make_unique<W>(owner, vram_probe, vram_others, *gv, vst);
        m.pool_pages = pages;
        m.seq_pages = 64;
        mem = {nb.malloc(64 * KP * kRowA), nb.malloc(64 * KP * kRowB)};
        m.kv_arrs = {{mem[0], kRowA}, {mem[1], kRowB}};
        sc->initPool();
        g_lines.clear();
        s::Log::setConsole(false);
        s::Log::setHook(hook, nullptr);
    }
    ~Rig() {
        s::Log::setHook(nullptr, nullptr);
        vw.reset();
        gv.reset();
        sc.reset();
        tb.reset();
        ps.reset();
        pool.reset();
        for (s::DevPtr p : mem) nb.free(p);
    }
    std::function<wddmwatch::Sample()> probe() {
        return [this] { ++probes; return cur; };
    }
    std::function<std::optional<wddmwatch::Others>()> others() {
        return [this]() -> std::optional<wddmwatch::Others> { ++others_reads; return wddmwatch::Others{3ull << 30, 1ull << 30}; };
    }
    s::EngineApi::Health health() {
        s::EngineApi::Health h;
        vw->vramHealth(h);
        return h;
    }
};

wddmwatch::Sample smp(std::uint64_t local, std::uint64_t budget, std::uint64_t shared) {
    wddmwatch::Sample x;
    x.ok = true;
    x.local_usage = local;
    x.local_budget = budget;
    x.nonlocal_usage = shared;
    return x;
}

double secondsUntil(s::TimePoint t) { return std::chrono::duration<double>(t - s::Clock::now()).count(); }

void testNoProbe() {
    Rig r(4);
    ISO_CHECK(r.vw->vramNote().empty());
    ISO_CHECK(!r.vw->vramIdleTick().has_value());
    const s::EngineApi::Health h = r.health();
    ISO_CHECK(!h.vram_ok && !h.vram_spill && h.vram_local == 0 && h.vram_budget == 0 && h.vram_shared == 0);
    ISO_CHECK(!r.vw->vramSpilling());
    ISO_CHECK(r.vw->vramLastSample() == s::TimePoint{});
    // a probe that is installed with nothing to sample (nullptr): stays off
    r.vw->setVramProbe(nullptr, true, std::nullopt, {}, false, r.others());
    ISO_CHECK(!r.vram_probe && r.vram_others);  // others installed into Engine's vram_others_ even so
    ISO_CHECK(r.vw->vramNote().empty() && g_lines.empty());
}

void testSetProbe() {
    Rig r(4);
    r.cur = smp(100 * MiB, 1000 * MiB, 10 * MiB);
    wddmwatch::Config c;
    c.idle_sample_s = 5.0;
    r.vw->setVramProbe(r.probe(), true, std::nullopt, c, false, r.others());
    ISO_CHECK(r.probes == 2);  // the cost of one sample, taken twice
    ISO_CHECK(r.vram_probe && r.vram_others && r.others_reads == 0);
    ISO_CHECK(countOf("vram watch: ") == 1 && levelOf("vram watch: ") == 'I');
    ISO_CHECK(countOf("warns while GPU memory spills") == 1);
    s::EngineApi::Health h = r.health();
    ISO_CHECK(h.vram_ok && !h.vram_spill && h.vram_local == 100 * MiB && h.vram_budget == 1000 * MiB && h.vram_shared == 10 * MiB);
    ISO_CHECK(!r.vw->vramSpilling());
    // the next idle sample in cfg.idle_sample_s: not due now
    const std::optional<s::TimePoint> due = r.vw->vramIdleTick();
    ISO_CHECK(due.has_value() && secondsUntil(*due) > 3.0 && secondsUntil(*due) <= 5.0);
    ISO_CHECK(r.probes == 2);
    // vramNote: one sample, the text against the baseline (the first sample), vram_last_t_ now
    const s::TimePoint before = s::Clock::now();
    r.cur = smp(200 * MiB, 1000 * MiB, 20 * MiB);
    ISO_CHECK(r.vw->vramNote() == wddmwatch::format(r.cur, smp(100 * MiB, 1000 * MiB, 10 * MiB)));
    ISO_CHECK(r.probes == 3 && r.vw->vramLastSample() >= before);
    h = r.health();
    ISO_CHECK(h.vram_local == 200 * MiB && h.vram_shared == 20 * MiB);
    // idle_sample_s 0: 60 s
    Rig q(4);
    q.cur = smp(1, 2, 0);
    wddmwatch::Config c0;
    c0.idle_sample_s = 0.0;
    q.vw->setVramProbe(q.probe(), false, std::nullopt, c0);
    ISO_CHECK(secondsUntil(*q.vw->vramIdleTick()) > 55.0);
    ISO_CHECK(countOf("integrated GPU: numbers only, no warning") == 1);
}

void testBaseAndBadSample() {
    Rig r(4);
    const wddmwatch::Sample base = smp(50 * MiB, 1000 * MiB, 5 * MiB);
    r.cur = smp(100 * MiB, 1000 * MiB, 10 * MiB);
    r.vw->setVramProbe(r.probe(), true, base, {}, false, nullptr);
    ISO_CHECK(r.vw->vramNote() == wddmwatch::format(r.cur, base));  // base ok: the baseline is base
    Rig q(4);
    const wddmwatch::Sample bad;  // ok false
    q.cur = smp(100 * MiB, 1000 * MiB, 10 * MiB);
    q.vw->setVramProbe(q.probe(), true, bad, {}, false, nullptr);
    ISO_CHECK(q.vw->vramNote() == wddmwatch::format(q.cur, q.cur));  // base not ok: the first sample
    // the first sample not ok: the probe is dropped (Engine's vram_probe_), nothing published
    Rig z(4);
    z.cur = wddmwatch::Sample{};
    z.vw->setVramProbe(z.probe(), true, std::nullopt, {}, false, z.others());
    ISO_CHECK(z.probes == 2 && !z.vram_probe && z.vram_others);
    ISO_CHECK(z.vw->vramNote().empty() && z.probes == 2 && !z.vw->vramIdleTick().has_value());
    ISO_CHECK(!z.health().vram_ok && countOf("vram watch: ") == 0);
}

void testSpillWarnings() {
    const std::string others = wddmwatch::othersNote(wddmwatch::Others{3ull << 30, 1ull << 30});
    // a spill already there at startup: the startup line warned (startup_warned) -> not again; else warned here
    for (const bool startup_warned : {false, true}) {
        Rig r(4);
        r.cur = smp(1100 * MiB, 1000 * MiB, 10 * MiB);  // over the budget
        r.vw->setVramProbe(r.probe(), true, std::nullopt, {}, startup_warned, r.others());
        ISO_CHECK(countOf("over the Windows video memory (WDDM) budget") == (startup_warned ? 0u : 1u));
        ISO_CHECK(r.others_reads == 0);  // the startup check does not read others
        ISO_CHECK(r.health().vram_spill && r.vw->vramSpilling());
    }
    // a spill seen by vramNote: a WARNING with the others note; then the INFO line when it is over
    Rig r(4);
    r.cur = smp(100 * MiB, 1000 * MiB, 10 * MiB);
    r.vw->setVramProbe(r.probe(), true, std::nullopt, {}, false, r.others());
    ISO_CHECK(!r.vw->vramSpilling() && !r.health().vram_spill);
    r.cur = smp(100 * MiB, 1000 * MiB, 10 * MiB + 512 * MiB);  // shared grown by 512 MiB
    (void)r.vw->vramNote();
    ISO_CHECK(levelOf("shared GPU memory grew by 512 MiB") == 'W' && r.others_reads == 1);
    ISO_CHECK(countOf(others) == 1);
    ISO_CHECK(r.vw->vramSpilling() && r.health().vram_spill);
    r.cur = smp(100 * MiB, 1000 * MiB, 10 * MiB);
    (void)r.vw->vramNote();
    ISO_CHECK(levelOf("GPU memory spill is over") == 'I' && r.others_reads == 1);
    ISO_CHECK(!r.vw->vramSpilling() && !r.health().vram_spill);
    // warn false (UMA): the same spill, numbers only, never spilling
    Rig u(4);
    u.cur = smp(1100 * MiB, 1000 * MiB, 600 * MiB);
    u.vw->setVramProbe(u.probe(), false, smp(1, 2, 0), {}, false, u.others());
    (void)u.vw->vramNote();
    ISO_CHECK(countOf("(WDDM) budget") == 0 && countOf("shared GPU memory grew") == 0 && u.others_reads == 0);
    ISO_CHECK(!u.vw->vramSpilling() && !u.health().vram_spill && u.health().vram_ok);
}

// F15: govTick first, the warning through govSetWarn (logged by govApply after its own lines), govReduced -> 5 s
void testGovernorOrder() {
    Rig r(8);
    r.m.resize_ok = true;
    const std::uint64_t pool_b = 8ull * KP * (kRowA + kRowB);
    const std::uint64_t margin = pool_b / 2;
    memgov::Config c;
    c.per_tok = kRowA + kRowB;
    c.page = KP;
    c.pool_start = 8 * KP;
    c.pool_min = KP;
    c.ctx = 4096;
    c.recover_s = c.hold_s = c.settle_s = c.trust_s = c.osc_window_s = 0.0;
    c.min_grow = 0;
    c.min_shrink = 0;
    r.cur = smp(8 * pool_b, 8 * pool_b + 4 * margin, 0);  // room: nothing to do
    r.vw->setVramProbe(r.probe(), true, std::nullopt, {}, false, r.others());
    r.gv->setGovernor(c, [margin](std::uint64_t) { return margin; }, 0);
    ISO_CHECK(r.gv->govActive() && !r.gv->govReduced());
    // the budget drops (less than margin / 2 free) and the shared memory grew 512 MiB: a spill and a shrink
    r.cur = smp(8 * pool_b, 8 * pool_b + margin / 8, 512 * MiB);
    g_lines.clear();
    (void)r.vw->vramNote();
    ISO_CHECK(countOf("shared GPU memory grew by 512 MiB") == 0);  // not logged by vramNote: handed to the governor
    ISO_CHECK(r.others_reads == 1);                                 // the others note was built for it
    ISO_CHECK(r.gv->govReduced());                                  // govTick ran before the cadence was set
    const std::optional<s::TimePoint> due = r.vw->vramIdleTick();   // not due: returns the 5 s cadence
    ISO_CHECK(due.has_value() && secondsUntil(*due) > 3.0 && secondsUntil(*due) <= 5.0);
    ISO_CHECK(r.pool->pages() == 8);  // govTick only decides
    r.gv->govApply();
    ISO_CHECK(r.pool->pages() == 4);
    const std::size_t w = indexOf("shared GPU memory grew by 512 MiB");
    ISO_CHECK(w < g_lines.size() && levelOf("shared GPU memory grew by 512 MiB") == 'W');
    ISO_CHECK(indexOf("memory governor") < w);  // the governor's own lines first, the warning last
    ISO_CHECK(countOf(wddmwatch::othersNote(wddmwatch::Others{3ull << 30, 1ull << 30})) == 1);
}

void testIdleTick() {
    Rig r(4);
    r.cur = smp(100 * MiB, 1000 * MiB, 10 * MiB);
    wddmwatch::Config c;
    c.idle_sample_s = 1e-6;
    r.vw->setVramProbe(r.probe(), true, std::nullopt, c, false, nullptr);
    ISO_CHECK(r.probes == 2);
    const s::TimePoint before = s::Clock::now();
    // setVramProbe made the next sample due 1 us after its own: wait that out, or a fast build calls vramIdleTick
    // inside the microsecond and nothing is due (cpu_iso_server_vram_watch failed 3/3 that way, 2026-10-31)
    while (s::Clock::now() - before < std::chrono::microseconds(5)) {
    }
    const std::optional<s::TimePoint> due = r.vw->vramIdleTick();  // due now: one sample
    ISO_CHECK(r.probes == 3 && due.has_value() && *due >= before && r.vw->vramLastSample() >= before);
    Rig q(4);
    q.cur = r.cur;
    q.vw->setVramProbe(q.probe(), true, std::nullopt, {}, false, nullptr);
    (void)q.vw->vramIdleTick();  // a minute away: no sample
    ISO_CHECK(q.probes == 2 && q.vw->vramLastSample() == s::TimePoint{});
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.vram_watch",
                    {{"no_probe", testNoProbe},
                     {"set_probe", testSetProbe},
                     {"base_bad_sample", testBaseAndBadSample},
                     {"spill_warnings", testSpillWarnings},
                     {"governor_order", testGovernorOrder},
                     {"idle_tick", testIdleTick}},
                    argc, argv);
}
