// whirl-iso-server-governor-tests: module server.governor (arch025-engine S8; progress/arch025-engine.DESIGN.md v3 2.3 L3,
// 4 S8): src/server/engine_gov.h (class GovState, the former Engine members gov_ .. gov_json_ of engine.h:284-301 at
// 6871c14 plus gov_mu_; class GovernorGlue, the former Engine member functions of engine_gov.cpp and ctxForNew) and
// src/server/engine_gov.cpp, built with only its declared dependencies (server.slot_cache, the memory governor policy
// memgov::Governor, core.shared) - not engine.h (ARCH-1 8.5 isolation).
// The checks pin what the move must keep, with the expected values read from the 6871c14 bodies:
// - the initial state (no governor): not active, not reduced, admission not held, governorJson "", ctxForNew = ctx (ctx_new_
//   starts at 0xffffffff), govTick / govApply do nothing;
// - setGovernor: ctx_new_ = min(ctx, c.ctx or ctx), published to ctx_pub_, the status JSON (memgov::statusJson) with
//   last_action "none" and the pool / slot numbers of the moment;
// - govPublishCtx / govCtxPublished are the ctx_pub_ store / load; ctxForNew follows a later setCtx (ctx_ is a reference);
// - govAllIdle / govFreeable over the slots; govFreeCache drops the pages of idle slots only (sc_.freePages), not of a
//   running one, and records the last action;
// - govResize: the pool re-allocated through ServerModel::resizePool and sc_.initPool (pages of every slot dropped
//   first), "pool A -> B tokens"; a refused re-allocation keeps the pool, "re-allocation to N tokens failed";
// - govTick + govApply over a low WDDM budget with every slot idle: the pool shrinks 8 -> 4 pages (shrinks 1, reduced),
//   ctxForNew is the governor's, admission not held while idle; with a request running: no re-allocation, admission held,
//   the idle slot's prefix cache freed; govSetWarn is logged and cleared by govApply;
// - layout: GovernorGlue holds references only (32), is not copyable, trivially destroyed.
// SPDX-License-Identifier: Apache-2.0
#include "device/null_backend.h"
#include "iso/fake_server_model.h"
#include "iso/iso_test.h"
#include "server/engine_gov.h"

#include <atomic>
#include <cstdint>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace {

namespace s = whirl::server;
namespace memgov = whirl::memgov;
namespace wddmwatch = whirl::wddmwatch;
using G = s::GovernorGlue;

static_assert(std::is_trivially_destructible_v<G>);
static_assert(!std::is_copy_constructible_v<G> && !std::is_copy_assignable_v<G>);
static_assert(sizeof(G) == 32 * sizeof(void*));
static_assert(std::is_default_constructible_v<s::GovState>);
// the accessors the other modules call (S8 rewrites of their call sites)
static_assert(std::is_same_v<decltype(&G::govActive), bool (G::*)() const>);
static_assert(std::is_same_v<decltype(&G::govReduced), bool (G::*)() const>);
static_assert(std::is_same_v<decltype(&G::govHolding), bool (G::*)() const>);
static_assert(std::is_same_v<decltype(&G::govSetWarn), void (G::*)(const std::string&)>);
static_assert(std::is_same_v<decltype(&G::govCtxPublished), std::uint32_t (G::*)() const>);
static_assert(std::is_same_v<decltype(&G::govPublishCtx), void (G::*)(std::uint32_t)>);
static_assert(std::is_same_v<decltype(&G::ctxForNew), std::uint32_t (G::*)() const>);

constexpr std::uint32_t KP = whirl::qwen35::kv_page;
constexpr std::uint64_t kRowA = 16, kRowB = 48;  // bytes per token of the two pool arrays (64 per token)

// what Engine holds, built as Engine builds it (members in Engine's order; sc_ then gv_ last); two slots
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
    s::GovState gst;
    std::atomic<std::uint32_t> ctx_pub{0};
    std::vector<s::KvArr> kv;
    std::unique_ptr<s::PagePool> pool;
    std::unique_ptr<s::PrefixStore> ps;
    std::unique_ptr<s::TierBridge> tb;
    s::SlotPagesCounters cnt;
    std::unique_ptr<s::DrafterCtx> dc;
    std::unique_ptr<s::SlotPages> sc;
    std::unique_ptr<G> gv;
    std::vector<s::DevPtr> mem;
    explicit Rig(std::uint32_t pages) {
        ck = std::make_unique<s::CkptStore>(nb, m, s::CkptStore::Layout{0, 0, false});
        slots.resize(2);
        for (std::uint32_t i = 0; i < 2; ++i) slots[i].id = i;
        pool = std::make_unique<s::PagePool>(nb, m);
        ps = std::make_unique<s::PrefixStore>(ct, s::PrefixStore::Config{});
        tb = std::make_unique<s::TierBridge>(nb, m, ct, *ps, kv, s::TierBridge::Config{});
        dc = std::make_unique<s::DrafterCtx>(s::DrafterCtx{bst, m, opt});
        sc = std::make_unique<s::SlotPages>(s::SlotPages{*ck, m, nb, opt, *pool, *ps, slots, *tb, kv, conv_bytes, ssm_bytes, cnt, *dc});
        gv = std::make_unique<G>(*ck, ctx, m, nb, opt, owner, *pool, *ps, slots, *tb, vram_probe, vram_others, ctx_pub, *sc, gst);
        m.pool_pages = pages;
        m.seq_pages = 64;
        mem = {nb.malloc(64 * KP * kRowA), nb.malloc(64 * KP * kRowB)};  // room for a grow up to 64 pages
        m.kv_arrs = {{mem[0], kRowA}, {mem[1], kRowB}};
        sc->initPool();
    }
    ~Rig() {
        gv.reset();
        sc.reset();
        tb.reset();
        ps.reset();
        pool.reset();
        for (s::DevPtr p : mem) nb.free(p);
    }
};

// memgov::statusJson of the values govPublish reads (engine_gov.cpp:54-74 at 6871c14)
std::string json(bool reduced, std::uint32_t pool_tok, std::uint32_t pool_start, std::uint32_t ctx, std::uint32_t admitting,
                 std::uint32_t slots, bool held, std::uint64_t parked_mib, std::uint32_t shrinks, std::uint32_t grows,
                 const std::string& last) {
    return std::format("{{\"on\":true,\"reduced\":{},\"pool_tokens\":{},\"pool_tokens_start\":{},\"ctx_per_request\":{},"
                       "\"slots_admitting\":{},\"slots\":{},\"admission_held\":{},\"parked_mib\":{},\"shrinks\":{},"
                       "\"grows\":{},\"last_action\":\"{}\"}}",
                       reduced, pool_tok, pool_start, ctx, admitting, slots, held, parked_mib, shrinks, grows, last);
}

memgov::Config cfg(std::uint32_t pool_start, std::uint32_t ctx) {
    memgov::Config c;
    c.per_tok = kRowA + kRowB;
    c.page = KP;
    c.pool_start = pool_start;
    c.pool_min = KP;
    c.ctx = ctx;
    // every time rule 0 (as tests/pins engine --gov): a decision at each sample
    c.recover_s = c.hold_s = c.settle_s = c.trust_s = c.osc_window_s = 0.0;
    c.min_grow = 0;
    c.min_shrink = 0;
    return c;
}

void testInitial() {
    Rig r(4);
    ISO_CHECK(!r.gv->govActive());
    ISO_CHECK(!r.gv->govReduced());
    ISO_CHECK(!r.gv->govHolding());
    ISO_CHECK(r.gv->governorJson().empty());
    ISO_CHECK(r.gv->ctxForNew() == 4096);  // min(ctx_, 0xffffffff)
    ISO_CHECK(r.gv->govCtxPublished() == 0);
    wddmwatch::Sample smp;
    smp.ok = true;
    smp.local_budget = 1;
    smp.local_usage = 2;
    r.gv->govTick(smp);  // no governor: nothing decided
    r.gv->govApply();    // nothing pending
    ISO_CHECK(r.pool->pages() == 4 && r.gv->governorJson().empty() && !r.gv->govHolding());
}

void testCtxPublish() {
    Rig r(4);
    r.gv->govPublishCtx(77);
    ISO_CHECK(r.gv->govCtxPublished() == 77 && r.ctx_pub.load() == 77);
    r.ctx_pub.store(5);
    ISO_CHECK(r.gv->govCtxPublished() == 5);
    r.ctx = 1000;  // Engine::setCtx before the governor: ctx_ is a reference
    ISO_CHECK(r.gv->ctxForNew() == 1000);
}

void testSetGovernor() {
    Rig r(4);
    r.gv->setGovernor(cfg(4 * KP, 2048), [](std::uint64_t) { return std::uint64_t{0}; }, 0);
    ISO_CHECK(r.gv->govActive() && !r.gv->govReduced() && !r.gv->govHolding());
    ISO_CHECK(r.gv->ctxForNew() == 2048);  // min(ctx 4096, c.ctx 2048)
    ISO_CHECK(r.ctx_pub.load() == 2048 && r.gv->govCtxPublished() == 2048);
    ISO_CHECK(r.gv->governorJson() == json(false, 4 * KP, 4 * KP, 2048, 2, 2, false, 0, 0, 0, "none"));
    Rig q(4);
    q.gv->setGovernor(cfg(4 * KP, 0), nullptr, 0);  // c.ctx 0: the startup context
    ISO_CHECK(q.gv->ctxForNew() == 4096 && q.ctx_pub.load() == 4096);
    Rig w(4);
    w.gv->setGovernor(cfg(4 * KP, 9000), nullptr, 0);  // never above ctx_
    ISO_CHECK(w.gv->ctxForNew() == 4096);
}

void testIdleFreeable() {
    Rig r(4);
    ISO_CHECK(r.gv->govAllIdle());
    ISO_CHECK(r.gv->govFreeable() == 0);  // no checkpoint buffers (Layout 0), no shared table
    s::Job j;
    r.slots[1].job = &j;
    ISO_CHECK(!r.gv->govAllIdle());
    r.slots[1].job = nullptr;
    r.slots[0].phase = s::Phase::prefill;
    ISO_CHECK(!r.gv->govAllIdle());
}

void testFreeCache() {
    Rig r(4);
    r.gv->setGovernor(cfg(4 * KP, 0), nullptr, 0);
    s::Slot& a = r.slots[0];
    s::Slot& b = r.slots[1];
    ISO_CHECK(r.sc->ensurePages(a, KP) && r.sc->ensurePages(b, KP));
    a.cache_tokens.assign(3, 1u);
    s::Job j;
    b.job = &j;  // running: kept
    b.phase = s::Phase::decode;
    ISO_CHECK(r.pool->freeCount() == 2);
    r.gv->govFreeCache(false);
    ISO_CHECK(a.pages.empty() && b.pages.size() == 1 && r.pool->freeCount() == 3);
    r.gv->govPublish();
    ISO_CHECK(r.gv->governorJson() ==
              json(false, 4 * KP, 4 * KP, 4096, 2, 2, false, 0, 0, 0, "freed cache: 1 slot(s), 0 shared, 1 pages, 0 MiB checkpoint buffers"));
    b.job = nullptr;
    b.phase = s::Phase::idle;
}

void testResize() {
    Rig r(4);
    r.gv->setGovernor(cfg(4 * KP, 0), nullptr, 0);
    ISO_CHECK(r.sc->ensurePages(r.slots[0], KP));
    ISO_CHECK(!r.gv->govResize(4 * KP, true));  // same size: nothing
    ISO_CHECK(r.slots[0].pages.size() == 1);
    ISO_CHECK(!r.gv->govResize(2 * KP, true));  // the model refuses (FakeServerModel default): pool kept
    ISO_CHECK(r.pool->pages() == 4 && r.slots[0].pages.empty());  // the VRAM prefix cache went first
    r.gv->govPublish();
    ISO_CHECK(r.gv->governorJson().ends_with(std::format("\"last_action\":\"re-allocation to {} tokens failed\"}}", 2 * KP)));
    r.m.resize_ok = true;
    ISO_CHECK(r.sc->ensurePages(r.slots[1], KP));
    ISO_CHECK(r.gv->govResize(2 * KP, true));
    ISO_CHECK(r.pool->pages() == 2 && r.pool->freeCount() == 2 && r.slots[1].pages.empty());
    r.gv->govPublish();
    ISO_CHECK(r.gv->governorJson() == json(false, 2 * KP, 4 * KP, 4096, 2, 2, false, 0, 0, 0, std::format("pool {} -> {} tokens", 4 * KP, 2 * KP)));
}

void testTickApplyShrink() {
    Rig r(8);
    r.m.resize_ok = true;
    const std::uint64_t pool_b = 8ull * KP * (kRowA + kRowB);  // the whole pool's bytes (8 pages)
    const std::uint64_t margin = pool_b / 2;
    r.gv->setGovernor(cfg(8 * KP, 4096), [margin](std::uint64_t) { return margin; }, 0);
    // budget = usage + margin / 8: less than margin / 2 free (trigger); deficit = 7/8 margin = 3.5 pages -> 4 pages cut
    wddmwatch::Sample smp;
    smp.ok = true;
    smp.local_usage = 8 * pool_b;
    smp.local_budget = 8 * pool_b + margin / 8;
    r.gv->govSetWarn("spill warning (test)");
    r.gv->govTick(smp);
    ISO_CHECK(r.pool->pages() == 8);  // govTick only decides
    r.gv->govApply();
    ISO_CHECK(r.pool->pages() == 4);  // every slot idle: re-allocated to 4 pages now
    ISO_CHECK(r.gv->govReduced());
    ISO_CHECK(!r.gv->govHolding());   // nothing to hold while idle
    const std::uint32_t ctx_new = std::min<std::uint32_t>(4096, 4 * KP);  // memgov: min(c.ctx, the smaller pool)
    ISO_CHECK(r.gv->ctxForNew() == ctx_new && r.ctx_pub.load() == ctx_new);
    const std::string j = r.gv->governorJson();
    ISO_CHECK(j == json(true, 4 * KP, 8 * KP, ctx_new, 2, 2, false, 0, 1, 0, std::format("pool {} -> {} tokens", 8 * KP, 4 * KP)));
    r.gv->govApply();  // nothing pending: the status stays
    ISO_CHECK(r.gv->governorJson() == j);
}

void testTickHoldWhileRunning() {
    Rig r(8);
    r.m.resize_ok = true;
    const std::uint64_t pool_b = 8ull * KP * (kRowA + kRowB);
    const std::uint64_t margin = pool_b / 2;
    r.gv->setGovernor(cfg(8 * KP, 4096), [margin](std::uint64_t) { return margin; }, 0);
    s::Slot& a = r.slots[0];
    s::Slot& b = r.slots[1];
    ISO_CHECK(r.sc->ensurePages(a, KP) && r.sc->ensurePages(b, KP));
    s::Job j;
    a.job = &j;  // running
    a.phase = s::Phase::decode;
    wddmwatch::Sample smp;
    smp.ok = true;
    smp.local_usage = 8 * pool_b;
    smp.local_budget = 8 * pool_b + margin / 8;
    r.gv->govTick(smp);
    r.gv->govApply();
    ISO_CHECK(r.pool->pages() == 8);   // a request runs: no re-allocation
    ISO_CHECK(r.gv->govHolding());     // new requests queue until it finishes
    ISO_CHECK(a.pages.size() == 1 && b.pages.empty());  // the idle slot's prefix cache was freed, the running one kept
    const std::uint32_t ctx_new = std::min<std::uint32_t>(4096, 4 * KP);
    ISO_CHECK(r.gv->ctxForNew() == ctx_new);
    ISO_CHECK(r.gv->governorJson() ==
              json(true, 8 * KP, 8 * KP, ctx_new, 0, 2, true, 0, 0, 0, "freed cache: 1 slot(s), 0 shared, 1 pages, 0 MiB checkpoint buffers"));
    a.job = nullptr;
    a.phase = s::Phase::idle;
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.governor",
                    {{"initial", testInitial},
                     {"ctx_publish", testCtxPublish},
                     {"set_governor", testSetGovernor},
                     {"idle_freeable", testIdleFreeable},
                     {"free_cache", testFreeCache},
                     {"resize", testResize},
                     {"tick_apply_shrink", testTickApplyShrink},
                     {"tick_hold_running", testTickHoldWhileRunning}},
                    argc, argv);
}
