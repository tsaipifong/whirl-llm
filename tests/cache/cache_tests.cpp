// Unit tests of the prefix cache module (src/cache) against a mock tier.
// SPDX-License-Identifier: Apache-2.0

#include "cache/fingerprint.h"
#include "cache/prefix_cache.h"
#include "mock_tier.h"
#include "tier/kv_tier.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <numeric>

using namespace whirl::cache;
using whirl::cache::test::MockTier;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(c)                                                              \
    do {                                                                      \
        if (c) {                                                              \
            ++g_pass;                                                         \
        } else {                                                              \
            ++g_fail;                                                         \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);          \
        }                                                                     \
    } while (0)

static std::vector<std::uint32_t> seq(std::size_t n, std::uint32_t base = 100) {
    std::vector<std::uint32_t> v(n);
    std::iota(v.begin(), v.end(), base);
    return v;
}

static Config cfgOn(bool lcp = true, std::size_t sys_min = 0) {
    Config c;
    c.enabled = true;
    c.lcp_on = lcp;
    c.sys_min = sys_min;
    c.kv_page = 64;
    return c;
}

// engine-like derived types (the View must see their bases)
struct Ck : CkptKey {
    std::uint64_t dev = 0xdead;
};
struct Sp : SpeKey {
    std::vector<int> pages{1, 2, 3};
};
struct Sl : SlotCache {
    int phase = 0;
};

static Ck ck(std::uint32_t pos, std::uint64_t s, bool logits = true) {
    Ck c;
    c.valid = true;
    c.pos = pos;
    c.seq = s;
    c.has_logits = logits;
    return c;
}

static Sp sp(std::vector<std::uint32_t> toks, std::uint64_t lu) {
    Sp s;
    s.valid = true;
    s.n = static_cast<std::uint32_t>(toks.size());
    s.tokens = std::move(toks);
    s.last_used = lu;
    return s;
}

static void testHelpers() {
    const auto a = seq(10), b = seq(6);
    CHECK(lcpLen(a, b) == 6);
    auto c = a;
    c[3] = 7;
    CHECK(lcpLen(a, c) == 3);
    CHECK(prefixEq(Tokens(a).subspan(0, 6), b));
    CHECK(!prefixEq(a, b));
    auto pc = makePrefixCache(cfgOn());
    const auto t1 = pc->touch(), t2 = pc->touch();
    CHECK(t1 == 1 && t2 == 2);
}

static void testMatchSlot() {
    auto pc = makePrefixCache(cfgOn());
    const auto cached = seq(1000);
    std::vector<Ck> cks = {ck(256, 1), ck(512, 2), ck(900, 3), Ck{}};
    // request diverges at 600: best checkpoint within the lcp is 512
    auto req = seq(1000);
    req[600] = 1;
    SlotMatch m = pc->matchSlot(cached, cks, req);
    CHECK(m.reuse == 512 && m.ck == 1 && m.lcp == 600);
    // full match at the request length: a checkpoint there needs logits
    const auto req900 = seq(900);
    cks[2].has_logits = false;
    m = pc->matchSlot(cached, cks, req900);
    CHECK(m.reuse == 512 && m.ck == 1);
    cks[2].has_logits = true;
    m = pc->matchSlot(cached, cks, req900);
    CHECK(m.reuse == 900 && m.ck == 2);
    // no common prefix
    m = pc->matchSlot(cached, cks, seq(50, 7));
    CHECK(m.reuse == 0 && m.ck == -1 && m.lcp == 0);
    // disabled: nothing, not even the lcp
    Config off = cfgOn();
    off.enabled = false;
    auto pd = makePrefixCache(off);
    m = pd->matchSlot(cached, cks, req);
    CHECK(m.reuse == 0 && m.ck == -1 && m.lcp == 0);
}

static void testCkptSlotFor() {
    auto pc = makePrefixCache(cfgOn());
    std::vector<Ck> cks = {ck(100, 5), ck(200, 3), ck(300, 7)};
    CHECK(pc->ckptSlotFor(cks, 200) == 1);  // same position: overwrite in place
    CHECK(pc->ckptSlotFor(cks, 400) == 1);  // oldest (lowest seq)
    cks.push_back(Ck{});
    CHECK(pc->ckptSlotFor(cks, 400) == 3);  // a free one first
    cks[0].valid = false;
    CHECK(pc->ckptSlotFor(cks, 400) == 0);  // first free one
    CHECK(pc->ckptSlotFor(cks, 300) == 2);  // the same position wins over a free one
}

static void testShared() {
    auto pc = makePrefixCache(cfgOn());
    const auto toks = seq(5000);
    std::vector<Sp> spes = {sp(seq(1024), 3), sp(seq(2048), 1), sp(seq(4096, 9), 2), Sp{}};
    CHECK(pc->matchShared(spes, toks, [](std::size_t) { return true; }) == 1);
    // off the chunk schedule: the shorter one
    CHECK(pc->matchShared(spes, toks, [](std::size_t a) { return a != 2048; }) == 0);
    // only proper prefixes match
    CHECK(pc->matchShared(spes, seq(2048), [](std::size_t) { return true; }) == 0);
    CHECK(pc->findShared(spes, Tokens(toks).subspan(0, 2048)) == 1);
    CHECK(pc->findShared(spes, Tokens(toks).subspan(0, 2047)) == -1);
    // insertion victim: a free entry first, else LRU, pinned skipped
    CHECK(pc->sharedVictim(spes) == 3);
    spes.pop_back();
    CHECK(pc->sharedVictim(spes) == 1);
    spes[1].pin = true;
    CHECK(pc->sharedVictim(spes) == 2);
    for (Sp& s : spes) s.pin = true;
    CHECK(pc->sharedVictim(spes) == -1);
    Config off = cfgOn();
    off.enabled = false;
    CHECK(makePrefixCache(off)->matchShared(spes, toks, [](std::size_t) { return true; }) == -1);
}

static void testEvict() {
    auto pc = makePrefixCache(cfgOn());
    std::vector<Sl> slots(3);
    slots[0].last_used = 5;
    slots[1].last_used = 2;
    slots[2].last_used = 9;
    std::vector<Sp> spes = {sp(seq(10), 4), sp(seq(20), 1)};
    auto all = [](std::size_t) { return true; };
    Victim v = pc->evictVictim(slots, all, spes);
    CHECK(v.kind == Victim::spe && v.index == 1);
    spes[1].pin = true;
    v = pc->evictVictim(slots, all, spes);
    CHECK(v.kind == Victim::slot && v.index == 1);
    v = pc->evictVictim(slots, [](std::size_t i) { return i != 1; }, spes);
    CHECK(v.kind == Victim::spe && v.index == 0);  // spe @4 older than slot 0 @5
    spes[0].valid = false;
    v = pc->evictVictim(slots, [](std::size_t i) { return i != 1; }, spes);
    CHECK(v.kind == Victim::slot && v.index == 0);
    v = pc->evictVictim(slots, [](std::size_t) { return false; }, spes);
    CHECK(v.kind == Victim::none);
    // ties: the first slot, and a shared entry only when strictly older
    slots[0].last_used = 1;
    slots[1].last_used = 1;
    spes[0].valid = true;
    spes[0].last_used = 1;
    v = pc->evictVictim(slots, all, spes);
    CHECK(v.kind == Victim::slot && v.index == 0);
}

static void testTier() {
    auto pc = makePrefixCache(cfgOn());
    MockTier t;
    CHECK(pc->findTierShared(seq(3000)) == 0);  // no tier attached
    pc->attachTier(&t);
    CHECK(pc->tier() == &t);
    auto toks = seq(8000);
    // multi-checkpoint entry (a spilled slot) and a shared 1-checkpoint entry
    t.add(11, seq(6000), {{2048, 1, 0, 4}, {4096, 1, 0, 1}, {6000, 1, 1, 0}});
    t.add(13, seq(3000), {{3000, 1, 0, 5}});
    auto req = toks;
    req[5000] = 1;
    TierResume r = pc->matchTierEntry(t.find(11), req);
    CHECK(r.reuse == 4096 && r.kind == 1);
    r = pc->matchTierEntry(t.find(11), seq(6000));
    CHECK(r.reuse == 6000 && r.kind == 0);  // has logits at the full length
    t.entries[0].cks[2].has_logits = 0;
    r = pc->matchTierEntry(t.find(11), seq(6000));
    CHECK(r.reuse == 4096);
    t.entries[0].n_tok = 3000;  // only the first n_tok tokens are valid
    r = pc->matchTierEntry(t.find(11), toks);
    CHECK(r.reuse == 2048 && r.kind == 4);
    CHECK(t.find(99) == nullptr);
    // a shared prefix already in the tier
    CHECK(pc->findTierShared(seq(3000)) == 13);
    CHECK(pc->findTierShared(seq(2999)) == 0);
    t.entries[1].cks[0].valid = 0;
    CHECK(pc->findTierShared(seq(3000)) == 0);
}

static void testLcpTarget() {
    auto pc = makePrefixCache(cfgOn());
    MockTier t;
    std::vector<Sl> slots(2);
    std::vector<Sp> spes(2);  // empty shared table entries (needed: lcp only with shared checkpoints on)
    const auto toks = seq(10000);
    auto floor64 = [](std::size_t lim) { return lim / 64 * 64; };
    CHECK(pc->lcpTarget(slots, spes, toks, 0, 0, floor64) == 0);  // nothing in common
    slots[1].cache_tokens = seq(5000);
    slots[1].cache_tokens.push_back(1);
    CHECK(pc->lcpTarget(slots, spes, toks, 0, 0, floor64) == 4992);
    CHECK(pc->lcpTarget(slots, spes, toks, 0, 4992, floor64) == 0);   // already the system split
    CHECK(pc->lcpTarget(slots, spes, toks, 4800, 0, floor64) == 0);   // not 4 pages past the reuse
    CHECK(pc->lcpTarget(slots, spes, toks, 4736, 0, floor64) == 4992);
    // the tier index and the shared entries count too
    pc->attachTier(&t);
    t.add(5, seq(7000), {});
    CHECK(pc->lcpTarget(slots, spes, toks, 0, 0, floor64) == 6976);
    spes[0] = sp(seq(9000), 1);
    CHECK(pc->lcpTarget(slots, spes, toks, 0, 0, floor64) == 8960);
    // at most the request length - 1
    CHECK(pc->lcpTarget(slots, spes, seq(9000), 0, 0, floor64) == 8960);
    // minimum: sys_min, else 2048
    std::vector<Sl> s2(1);
    s2[0].cache_tokens = seq(2000);
    std::vector<Sp> e2(1);
    CHECK(makePrefixCache(cfgOn())->lcpTarget(s2, e2, toks, 0, 0, floor64) == 0);
    CHECK(makePrefixCache(cfgOn(true, 1024))->lcpTarget(s2, e2, toks, 0, 0, floor64) == 1984);
    // off: lcp checkpoints off, no shared table, prefix cache off
    CHECK(makePrefixCache(cfgOn(false))->lcpTarget(slots, spes, toks, 0, 0, floor64) == 0);
    CHECK(pc->lcpTarget(slots, std::vector<Sp>{}, toks, 0, 0, floor64) == 0);
    Config off = cfgOn();
    off.enabled = false;
    CHECK(makePrefixCache(off)->lcpTarget(slots, spes, toks, 0, 0, floor64) == 0);
}

// the pre-R-1 server_main.cpp tierFingerprint (env list instead of the process block)
static std::uint64_t oldFingerprint(const FingerprintInputs& in, std::vector<std::pair<std::string, std::string>> env) {
    using whirl::tier::hashStr;
    std::uint64_t h = 0xcbf29ce484222325ull;
    h = hashStr(h, "whirl-kv-tier-1");
    h = hashStr(h, std::format("exe {:x} model {} {} {} kv {} mtp {} gdn {} ck {} pg {} batch {}", in.exe_id,
                               in.model_base, in.tensors, in.bytes, in.kv_name, in.use_mtp, in.n_gdn, in.ck_bytes,
                               in.page_bytes, static_cast<std::uint32_t>(in.max_batch)));
    h = hashStr(h, "numerics " + in.numerics);
    std::sort(env.begin(), env.end());
    for (const auto& [k, v] : env) {
        h = hashStr(h, k);
        h = hashStr(h, "=");
        h = hashStr(h, v);
        h = hashStr(h, ";");
    }
    return h;
}

static void testFingerprint() {
    FingerprintInputs in;
    in.exe_id = 0x1234abcd;
    in.model_base = "Qwen3.8-27B-Q4_K_M.gguf";
    in.tensors = 851;
    in.bytes = 16'000'000'000ull;
    in.kv_name = "f16";
    in.use_mtp = true;
    in.n_gdn = 48;
    in.ck_bytes = 1 << 20;
    in.page_bytes = 1 << 18;
    in.max_batch = 4096;
    in.numerics = "balance (items a,b)";
    const std::uint64_t base = tierFingerprint(in);
    CHECK(base == tierFingerprint(in));
    CHECK(base == oldFingerprint(in, {}));
    // numerics mode is part of the key
    FingerprintInputs p = in;
    p.numerics = "precise (items)";
    CHECK(tierFingerprint(p) != base);
    CHECK(tierFingerprint(p) == oldFingerprint(p, {}));
    // model identity / layout / mtp
    p = in;
    p.use_mtp = false;
    CHECK(tierFingerprint(p) != base);
    p = in;
    p.page_bytes += 1;
    CHECK(tierFingerprint(p) != base);
    // environment: WHIRL_* kept, order free, size / path / logging knobs ignored
    p = in;
    p.env = {{"WHIRL_KV_SSD_DIR", "D:\\x"}, {"PATH", "C:\\"}, {"WHIRL_PROFILE", "1"}, {"whirl_kv_ram_mb", "8"}};
    CHECK(tierFingerprint(p) == base);
    p.env = {{"WHIRL_B", "2"}, {"WHIRL_A", "1"}, {"OTHER", "x"}};
    const std::uint64_t e1 = tierFingerprint(p);
    CHECK(e1 != base);
    p.env = {{"WHIRL_A", "1"}, {"WHIRL_B", "2"}};
    CHECK(tierFingerprint(p) == e1);
    CHECK(e1 == oldFingerprint(in, {{"WHIRL_A", "1"}, {"WHIRL_B", "2"}}));
    p.env = {{"WHIRL_A", "1"}, {"WHIRL_B", "3"}};
    CHECK(tierFingerprint(p) != e1);
    p.env = {{"Whirl_Relax", "1"}};  // prefix check is case-insensitive, the key is hashed as written
    CHECK(tierFingerprint(p) != base);
    CHECK(fingerprintEnvKept("WHIRL_NUMERICS"));
    CHECK(fingerprintEnvKept("whirl_x"));
    CHECK(!fingerprintEnvKept("WHIRL_KV_TIER_MIN"));
    CHECK(!fingerprintEnvKept("WHIRL_TIMER_PROBE"));
    CHECK(!fingerprintEnvKept("WHIRLX"));
    // the process environment is already filtered
    for (const auto& [k, v] : fingerprintEnv()) CHECK(fingerprintEnvKept(k));
}

// CACHE-1: superseded-session rule (compact / compress yes; retry,
// regenerate, edit-last, other system prompt, unknown system prompt no)
static void testSupersede() {
    auto seq = [](std::size_t n, std::uint32_t base) {
        std::vector<std::uint32_t> v(n);
        for (std::size_t i = 0; i < n; ++i) v[i] = base + static_cast<std::uint32_t>(i);
        return v;
    };
    auto cat = [](std::vector<std::uint32_t> a, const std::vector<std::uint32_t>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    const auto sys = seq(3000, 1);
    const auto old_t = cat(sys, seq(20000, 100000));  // 23000 tokens
    // compact: system prompt + summary
    {
        const auto n = cat(sys, seq(1500, 500000));
        const Supersede s = supersedes(old_t, n, sys.size());
        CHECK(s.yes && s.keep == 3000);
    }
    // compress: first messages kept, middle dropped (mid-way divergence), shorter
    {
        auto n = old_t;
        n.resize(9000);
        n = cat(n, seq(4000, 600000));
        const Supersede s = supersedes(old_t, n, sys.size());
        CHECK(s.yes && s.keep == 9000);
    }
    // regenerate / retry: diverges at the last answer
    {
        auto n = old_t;
        n.resize(old_t.size() - 600);
        n = cat(n, seq(500, 700000));
        CHECK(!supersedes(old_t, n, sys.size()).yes);
    }
    // edit-last: the last user message changed (tail of 2500 tokens < max(4096, 25 %))
    {
        auto n = old_t;
        n.resize(old_t.size() - 2500);
        n = cat(n, seq(300, 710000));
        CHECK(!supersedes(old_t, n, sys.size()).yes);
    }
    // continuation (longer): never
    CHECK(!supersedes(old_t, cat(old_t, seq(100, 720000)), sys.size()).yes);
    // different system prompt (shares less than the system message)
    {
        auto n = seq(2990, 1);
        n = cat(n, seq(2000, 730000));
        CHECK(!supersedes(old_t, n, 3000).yes);
    }
    // unknown system message: never
    CHECK(!supersedes(old_t, cat(sys, seq(1500, 500000)), 0).yes);
    // not clearly shorter (new >= 75 % of old)
    CHECK(!supersedes(old_t, cat(sys, seq(16000, 740000)), sys.size()).yes);
    // short old session: the abandoned tail is under min_tail
    {
        const auto small_old = cat(sys, seq(3000, 750000));
        CHECK(!supersedes(small_old, cat(sys, seq(200, 760000)), sys.size()).yes);
    }
    // too short for the host tiers
    CHECK(tooShortForTier(2047, 2048) && !tooShortForTier(2048, 2048));
    // deferred marks: taken once per new entry, re-marking updates, forget drops
    SupersedeTracker tr;
    tr.mark(10, 1, 3000, 77);
    tr.mark(10, 2, 3000, 78);
    tr.mark(10, 1, 3100, 79);
    tr.mark(11, 1, 3000, 80);
    CHECK(tr.size() == 3);
    const auto p = tr.take(10);
    CHECK(p.size() == 2 && p[0].old_id == 1 && p[0].keep == 3100 && p[0].stamp == 79 && p[1].old_id == 2);
    CHECK(tr.take(10).empty() && tr.size() == 1);
    tr.forget(1);
    CHECK(tr.size() == 0);
}

int main() {
    testHelpers();
    testMatchSlot();
    testCkptSlotFor();
    testShared();
    testEvict();
    testTier();
    testLcpTarget();
    testFingerprint();
    testSupersede();
    std::printf("cache tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
