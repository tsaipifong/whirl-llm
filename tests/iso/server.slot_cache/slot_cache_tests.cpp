// whirl-iso-server-slot-cache-tests: module server.slot_cache, part S6 (src/server/engine_pages.h: SlotPagesCounters,
// the shared prefix checkpoint / KV page counters; arch025-engine S6, the former Engine members stat_spe_new_ ..
// stat_evictions_ of engine.h:339-340 at 421e22b, their six writes in engine_pages.cpp:112/135/170/242/278/283 and the
// six reads in engine.cpp:184-189 (Engine::stats)) compiled with only its own header (ARCH-1 8.5 isolation).
// The checks pin what the move must keep, with expected values written out: every counter starts at 0; each write
// method changes exactly its own counter by exactly the former statement's amount (+= 1, or += the uint32 token count,
// summed in 64 bits); the six reads return them. Layout pins (DESIGN v3 2.5: cnt_ sits where stat_spe_new_ was, so
// every Engine member keeps its offset): six uint64, 8-aligned, no padding, standard layout, trivially destroyed.
// Part S7 (arch025-engine S7: struct SlotPages, the former Engine member functions of engine_pages.cpp at f07e6a0 and
// Engine::initPool / kvTokBytes of engine.cpp:133-137 / engine.h:259-263, now src/server/engine_pages.cpp built here with
// its declared dependencies): an aggregate of references only (trivially destroyed, 13 references); initPool takes the
// model's pool arrays and pages, kvTokBytes sums the rows; pagesFor = the pages of n + draftRoom + 2 tokens;
// ensurePages / takePage evict an idle slot (freePages, dropCache, the eviction counter) or fail when nothing is
// evictable; poolAvailable = free + idle-owned pages minus what running prefills still need; dropSpe releases a shared
// checkpoint once; waitsForSys / saveCkpt guards; cacheCounters is the counters object Engine holds. Expected values are
// written out (read from the f07e6a0 bodies).
// SPDX-License-Identifier: Apache-2.0
#include "device/null_backend.h"
#include "iso/fake_server_model.h"
#include "iso/iso_test.h"
#include "server/engine_pages.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

namespace {

namespace s = whirl::server;
using C = s::SlotPagesCounters;
using P = s::SlotPages;

static_assert(sizeof(C) == 6 * sizeof(std::uint64_t));
static_assert(alignof(C) == alignof(std::uint64_t));
static_assert(std::is_standard_layout_v<C>);
static_assert(std::is_trivially_destructible_v<C>);
static_assert(std::is_trivially_copyable_v<C>);
// the token count of a shared prefix checkpoint (cache::SpeKey::n) is a uint32; the former statement added it to a
// uint64 counter
static_assert(std::is_same_v<decltype(&C::noteSpeTok), void (C::*)(std::uint32_t)>);

struct Want {
    std::uint64_t spe_new = 0, spe_hits = 0, spe_tok = 0, spe_evict = 0, unshare = 0, evictions = 0;
};

void same(const C& c, const Want& w) {
    ISO_CHECK(c.speNew() == w.spe_new);
    ISO_CHECK(c.speHits() == w.spe_hits);
    ISO_CHECK(c.speTok() == w.spe_tok);
    ISO_CHECK(c.speEvict() == w.spe_evict);
    ISO_CHECK(c.unshares() == w.unshare);
    ISO_CHECK(c.evictions() == w.evictions);
}

void testStart() {
    const C c;
    same(c, {});
    const C d{};
    same(d, {});
}

void testEachWrite() {
    C c;
    c.noteSpeNew();
    same(c, {1, 0, 0, 0, 0, 0});
    c.noteSpeHit();
    same(c, {1, 1, 0, 0, 0, 0});
    c.noteSpeTok(37);
    same(c, {1, 1, 37, 0, 0, 0});
    c.noteSpeEvict();
    same(c, {1, 1, 37, 1, 0, 0});
    c.noteUnshare();
    same(c, {1, 1, 37, 1, 1, 0});
    c.noteEviction();
    same(c, {1, 1, 37, 1, 1, 1});
}

void testCounts() {
    C c;
    for (int i = 0; i < 5; ++i) c.noteSpeNew();
    for (int i = 0; i < 3; ++i) c.noteSpeHit();
    for (int i = 0; i < 4; ++i) c.noteSpeEvict();
    for (int i = 0; i < 2; ++i) c.noteUnshare();
    for (int i = 0; i < 7; ++i) c.noteEviction();
    c.noteSpeTok(0);
    c.noteSpeTok(1000);
    c.noteSpeTok(24);
    same(c, {5, 3, 1024, 4, 2, 7});
}

void testTokensIn64Bits() {
    C c;
    const std::uint32_t big = std::numeric_limits<std::uint32_t>::max();
    c.noteSpeTok(big);
    c.noteSpeTok(big);
    c.noteSpeTok(2);
    same(c, {0, 0, 8589934592ull, 0, 0, 0});   // 2 * 4294967295 + 2 = 2^33: no 32-bit wrap
}

void testCopy() {
    C c;
    c.noteSpeNew();
    c.noteSpeTok(9);
    c.noteEviction();
    const C d = c;                            // a plain value (trivially copyable): Engine::stats reads, never copies
    same(d, {1, 0, 9, 0, 0, 1});
    c.noteEviction();
    same(d, {1, 0, 9, 0, 0, 1});
    same(c, {1, 0, 9, 0, 0, 2});
}

// ---- S7: SlotPages
static_assert(std::is_aggregate_v<P>);
static_assert(std::is_trivially_destructible_v<P>);
static_assert(sizeof(P) == 13 * sizeof(void*));
static_assert(std::is_same_v<decltype(P::ck_), s::CkptStore&> && std::is_same_v<decltype(P::m_), s::ServerModel&> &&
              std::is_same_v<decltype(P::ops_), whirl::device::Backend&> && std::is_same_v<decltype(P::opt_), const s::EngineOptions&> &&
              std::is_same_v<decltype(P::pool_), s::PagePool&> && std::is_same_v<decltype(P::ps_), s::PrefixStore&> &&
              std::is_same_v<decltype(P::slots_), std::vector<s::Slot>&> && std::is_same_v<decltype(P::tb_), s::TierBridge&> &&
              std::is_same_v<decltype(P::kv_arrays_), std::vector<s::KvArr>&> &&
              std::is_same_v<decltype(P::conv_bytes_), const std::uint64_t&> && std::is_same_v<decltype(P::ssm_bytes_), const std::uint64_t&> &&
              std::is_same_v<decltype(P::cnt_), C&> && std::is_same_v<decltype(P::dc_), s::DrafterCtx&>);

constexpr std::uint32_t KP = whirl::qwen35::kv_page;
constexpr std::uint64_t kRowA = 16, kRowB = 48;

// what Engine holds, built as Engine builds it (members in Engine's order; sc_ last); two slots, no shared checkpoint
// tables unless a test sizes ps.spes()
struct Rig {
    whirl::device::NullBackend nb;
    whirl::test::FakeServerModel m{nb, 10};
    s::EngineOptions opt;
    whirl::chat::ChatTokens ct;
    std::unique_ptr<s::CkptStore> ck;
    s::BatchStats bst;
    std::uint64_t conv_bytes = 0, ssm_bytes = 0;
    std::vector<s::Slot> slots;
    std::vector<s::KvArr> kv;
    std::unique_ptr<s::PagePool> pool;
    std::unique_ptr<s::PrefixStore> ps;
    std::unique_ptr<s::TierBridge> tb;
    C cnt;
    std::unique_ptr<s::DrafterCtx> dc;
    std::unique_ptr<P> sp;
    std::vector<s::DevPtr> mem;
    explicit Rig(std::uint32_t pages) {
        ck = std::make_unique<s::CkptStore>(nb, m, s::CkptStore::Layout{0, 0, false});
        slots.resize(2);
        for (std::uint32_t i = 0; i < 2; ++i) slots[i].id = i;
        pool = std::make_unique<s::PagePool>(nb, m);
        ps = std::make_unique<s::PrefixStore>(ct, s::PrefixStore::Config{});
        tb = std::make_unique<s::TierBridge>(nb, m, ct, *ps, kv, s::TierBridge::Config{});
        dc = std::make_unique<s::DrafterCtx>(s::DrafterCtx{bst, m, opt});
        sp = std::make_unique<P>(P{*ck, m, nb, opt, *pool, *ps, slots, *tb, kv, conv_bytes, ssm_bytes, cnt, *dc});
        m.pool_pages = pages;
        m.seq_pages = 64;  // pages one sequence may map
        mem = {nb.malloc(pages * KP * kRowA), nb.malloc(pages * KP * kRowB)};
        m.kv_arrs = {{mem[0], kRowA}, {mem[1], kRowB}};
    }
    ~Rig() {
        sp.reset();
        tb.reset();
        ps.reset();
        pool.reset();
        for (s::DevPtr p : mem) nb.free(p);
    }
};

void testInitPoolKvTokBytes() {
    Rig r(4);
    ISO_CHECK(r.sp->kvTokBytes() == 0);  // no pool arrays yet
    r.sp->initPool();
    ISO_CHECK(r.kv.size() == 2 && r.kv[0].base == r.mem[0] && r.kv[0].row == kRowA && r.kv[1].row == kRowB);
    ISO_CHECK(r.pool->pages() == 4 && r.pool->freeCount() == 4);
    ISO_CHECK(r.sp->kvTokBytes() == kRowA + kRowB);  // 64: the bytes one token takes in every pool array
    ISO_CHECK(&r.sp->cacheCounters() == &r.cnt);
}

void testPagesFor() {
    Rig r(4);
    // no MTP, no DFlash2: draftRoom 0, so the pages of n + 2 tokens
    ISO_CHECK(r.sp->pagesFor(0) == 1);
    ISO_CHECK(r.sp->pagesFor(KP - 2) == 1);
    ISO_CHECK(r.sp->pagesFor(KP - 1) == 2);
    ISO_CHECK(r.sp->pagesFor(2 * KP - 2) == 2);
    r.opt.use_mtp = true;
    r.opt.n_draft = 3;  // MTP: + n_draft
    ISO_CHECK(r.sp->pagesFor(KP - 5) == 1);
    ISO_CHECK(r.sp->pagesFor(KP - 4) == 2);
}

void testEnsureEvictsIdle() {
    Rig r(2);
    r.sp->initPool();
    s::Slot& a = r.slots[0];
    s::Slot& b = r.slots[1];
    ISO_CHECK(r.sp->ensurePages(a, KP));  // one page
    ISO_CHECK(a.pages.size() == 1 && r.pool->freeCount() == 1);
    a.cache_tokens.assign(5, 7u);
    a.tier_id = 9;
    a.tier_pages = 1;
    b.phase = s::Phase::prefill;
    ISO_CHECK(r.sp->ensurePages(b, 2 * KP));  // two pages: the idle slot a is evicted
    ISO_CHECK(b.pages.size() == 2);
    ISO_CHECK(a.pages.empty() && a.cache_tokens.empty() && a.tier_id == 0 && a.tier_pages == 0);
    ISO_CHECK(r.cnt.evictions() == 1 && r.cnt.unshares() == 0 && r.cnt.speEvict() == 0);
    // nothing left to evict (a holds no pages, b asks): no page
    ISO_CHECK(!r.sp->takePage(&b).has_value());
    ISO_CHECK(!r.sp->ensurePages(b, 3 * KP));
    ISO_CHECK(r.cnt.evictions() == 1);
}

void testPoolAvailable() {
    Rig r(4);
    r.sp->initPool();
    s::Slot& a = r.slots[0];
    s::Slot& b = r.slots[1];
    ISO_CHECK(r.sp->poolAvailable() == 4);
    ISO_CHECK(r.sp->ensurePages(a, KP));  // a idle, owns 1: still available (evictable)
    ISO_CHECK(r.sp->poolAvailable() == 4);
    s::Job j;
    b.phase = s::Phase::prefill;
    b.job = &j;
    b.n_prompt = 2 * KP;  // needs pagesFor(2 KP + 2) = 3 pages, holds 0
    ISO_CHECK(r.sp->poolAvailable() == 1);
    ISO_CHECK(r.sp->ensurePages(b, KP));  // holds 1 of the 3 now: 2 free + 1 idle-owned - 2 owed
    ISO_CHECK(r.sp->poolAvailable() == 1);
    b.n_prompt = 8 * KP;  // owes more than there is
    ISO_CHECK(r.sp->poolAvailable() == 0);
    b.job = nullptr;
}

void testDropSpe() {
    Rig r(4);
    r.sp->initPool();
    r.ps->spes().resize(1);
    s::Spe& e = r.ps->spes()[0];
    auto p = r.pool->takeFree();
    ISO_CHECK(p.has_value());
    e.pages = {*p};
    e.tokens.assign(3, 1u);
    e.valid = true;
    e.n = 3;
    e.tier_id = 5;
    r.sp->dropSpe(e);
    ISO_CHECK(!e.valid && e.pages.empty() && e.tokens.empty() && e.n == 0 && e.tier_id == 0);
    ISO_CHECK(r.pool->freeCount() == 4 && r.cnt.speEvict() == 1);
    r.sp->dropSpe(e);  // not valid: nothing, not counted again
    ISO_CHECK(r.pool->freeCount() == 4 && r.cnt.speEvict() == 1);
}

void testGuards() {
    Rig r(4);
    s::Slot& a = r.slots[0];
    a.cache_tokens.assign(2, 3u);
    r.sp->dropCache(a);
    ISO_CHECK(a.cache_tokens.empty());
    s::Job j;
    j.tokens.assign(40, 1u);
    ISO_CHECK(!r.sp->waitsForSys(j));  // no shared checkpoint table
    r.ps->spes().resize(1);
    j.params.cache_prompt = false;
    ISO_CHECK(!r.sp->waitsForSys(j));
    r.opt.no_prefix_cache = true;
    a.cache_tokens.assign(2, 3u);
    const std::uint64_t seq = a.ckpt_seq;
    r.sp->saveCkpt(a, 1, "x", std::nullopt, std::nullopt);  // no prefix cache: nothing saved
    ISO_CHECK(a.ckpt_seq == seq && a.cache_tokens.size() == 2);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.slot_cache",
                    {{"start", testStart},
                     {"each_write", testEachWrite},
                     {"counts", testCounts},
                     {"tokens_64", testTokensIn64Bits},
                     {"copy", testCopy},
                     {"pages_init_kvtok", testInitPoolKvTokBytes},
                     {"pages_for", testPagesFor},
                     {"pages_ensure_evicts_idle", testEnsureEvictsIdle},
                     {"pages_pool_available", testPoolAvailable},
                     {"pages_drop_spe", testDropSpe},
                     {"pages_guards", testGuards}},
                    argc, argv);
}
