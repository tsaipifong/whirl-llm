// Unit tests of the prefix cache module (src/cache) against a mock tier.
// SPDX-License-Identifier: Apache-2.0

#include "../test_env.h"
#include "cache/fingerprint.h"
#include "cache/prefix_cache.h"
#include "mock_tier.h"
#include "tier/kv_tier.h"
#include "whirl/numerics.h"
#include "whirl/common.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cwchar>
#include <format>
#include <numeric>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>  // arch025 PL1: the reference copy of the base GetEnvironmentStringsW loop
#endif

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
    // TAIL-CK: a full set of >= ckpt_keep_prefix_min keeps its newest 'prefix' checkpoint
    std::vector<Ck> t = {ck(100, 5), ck(200, 3), ck(300, 7)};
    t[1].kind = "prefix";
    t[0].kind = "prompt-end";
    t[2].kind = "gen-end";
    CHECK(pc->ckptSlotFor(t, 400) == 0);  // oldest is the tail: the next oldest goes
    CHECK(pc->ckptSlotFor(t, 200) == 1);  // the same position: in place, also a 'prefix' one
    t[0].kind = "prefix";                 // two 'prefix' ones: the older (seq 3) goes, the newest stays
    CHECK(pc->ckptSlotFor(t, 400) == 1);
    t.push_back(Ck{});
    CHECK(pc->ckptSlotFor(t, 400) == 3);  // a free one first
    // fewer than ckpt_keep_prefix_min: plain oldest first (2 per slot keeps prompt-end + gen-end)
    std::vector<Ck> two = {ck(100, 3), ck(200, 4)};
    two[0].kind = "prefix";
    CHECK(pc->ckptSlotFor(two, 400) == 0);
    std::vector<Ck> one = {ck(100, 3)};
    one[0].kind = "prefix";
    CHECK(pc->ckptSlotFor(one, 400) == 0);
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
    // COLD-RESUME: an accept predicate skips refused checkpoints and falls back to the longest accepted one
    r = pc->matchTierEntry(t.find(11), req, [](std::uint32_t, std::uint8_t k) { return k != 1; });
    CHECK(r.reuse == 2048 && r.kind == 4);
    r = pc->matchTierEntry(t.find(11), req, [](std::uint32_t, std::uint8_t) { return false; });
    CHECK(r.reuse == 0);
    r = pc->matchTierEntry(t.find(11), req, [](std::uint32_t p, std::uint8_t) { return p % 2048 == 0; });
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
    // bug-s5a-cachemiss F2: the request's own slot / tier entry continued whole is not another session
    {
        std::vector<Sp> sp0(2);
        CHECK(pc->lcpTarget(slots, sp0, toks, 0, 0, floor64) == 6976);                  // tier entry 5 (7000 tok)
        CHECK(pc->lcpTarget(slots, sp0, toks, 0, 0, floor64, 9, 5) == 4992);            // own entry left out: slot 1
        CHECK(pc->lcpTarget(slots, sp0, toks, 0, 0, floor64, 1, 5) == 4992);            // slot 1 diverges at its end: counts
        CHECK(pc->lcpTarget(slots, sp0, toks, 0, 0, floor64, 1, 0) == 6976);            // own slot only
        std::vector<Sl> s3(2);
        s3[0].cache_tokens = seq(6000);                                                  // continued whole by toks
        CHECK(pc->lcpTarget(s3, sp0, toks, 0, 0, floor64, 0, 5) == 0);                  // own slot + own entry
        CHECK(pc->lcpTarget(s3, sp0, toks, 0, 0, floor64, 1, 5) == 5952);               // another slot's session counts
        CHECK(pc->lcpTarget(slots, sp0, seq(4000), 0, 0, floor64, 1, 5) == 3968);       // diverges inside them: counts
    }
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

// FIX-TEST-A (FR-TEST P2-3): the server's fingerprint assembly (cache::serverFingerprintInputs, called
// by server_main.cpp). Mutation M5 (server_main stopped filling gpu_arch / gpu_name) passed every test.
static void testServerFingerprintInputs() {
    ServerIdentity s;
    s.exe_id = 0x77;
    s.model_path = "C:\\models\\x\\Qwen3.8-27B-Q4_K_M.gguf";
    s.tensors = 851;
    s.bytes = 16'000'000'000ull;
    s.kv_name = "q8h";
    s.use_mtp = true;
    s.n_gdn = 48;
    s.ck_bytes = 1 << 20;
    s.page_bytes = 1 << 18;
    s.max_batch = 4096;
    s.numerics = "balance";
    s.gpu_arch = "gfx1201";
    s.gpu_name = "AMD Radeon AI PRO R9700";
    s.kernel_arch = "gfx1201";
    s.kernel_co_bytes = 123456;
    s.env = {{"WHIRL_A", "1"}};
    const FingerprintInputs in = serverFingerprintInputs(s);
    CHECK(in.gpu_arch == "gfx1201");
    CHECK(in.gpu_name == "AMD Radeon AI PRO R9700");
    CHECK(in.kernel_set == "gfx1201 co 123456");
    CHECK(in.model_base == "Qwen3.8-27B-Q4_K_M.gguf");
    CHECK(in.kv_name == "q8h" && in.numerics == "balance" && in.max_batch == 4096 && in.use_mtp && in.n_gdn == 48);
    CHECK(in.tensors == 851 && in.bytes == 16'000'000'000ull && in.ck_bytes == (1u << 20) && in.page_bytes == (1u << 18));
    CHECK(in.exe_id == 0x77 && in.env.size() == 1);
    CHECK(!in.legacy_no_gpu);
    const std::uint64_t fp = tierFingerprint(in);
    // a different GPU (same exe, same model) -> a different fingerprint, field by field
    ServerIdentity o = s;
    o.gpu_arch = "gfx1151";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.gpu_name = "AMD Radeon(TM) 8060S Graphics";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.kernel_arch = "gfx1151";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.kernel_co_bytes += 1;
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    o = s;
    o.kv_name = "f16";
    CHECK(tierFingerprint(serverFingerprintInputs(o)) != fp);
    // missing GPU identity -> error (no silent old format)
    auto throws = [](const ServerIdentity& x) {
        try {
            (void)serverFingerprintInputs(x);
        } catch (const std::invalid_argument&) {
            return true;
        }
        return false;
    };
    o = s;
    o.gpu_arch.clear();
    CHECK(throws(o));
    o = s;
    o.kernel_arch.clear();
    CHECK(throws(o));
    o = s;
    o.kernel_co_bytes = 0;
    CHECK(throws(o));
    o = s;
    o.gpu_name.clear();  // the name is informative only; the arch is the key
    CHECK(!throws(o));
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
    in.legacy_no_gpu = true;  // the format checks below compare with the pre-FIX-NUM hash
    const std::uint64_t base = tierFingerprint(in);
    CHECK(base == tierFingerprint(in));
    CHECK(base == oldFingerprint(in, {}));
    // numerics mode is part of the key
    FingerprintInputs p = in;
    p.numerics = "precise (items)";
    CHECK(tierFingerprint(p) != base);
    CHECK(tierFingerprint(p) == oldFingerprint(p, {}));
    // KV format is part of the key (FR-CACHE: corrupting kv_name went unnoticed)
    for (const char* kv : {"q8h", "q8", "q8v", "q4", ""}) {
        p = in;
        p.kv_name = kv;
        CHECK(tierFingerprint(p) != base);
    }
    p = in;
    p.max_batch = 2048;
    CHECK(tierFingerprint(p) != base);
    p = in;
    p.exe_id += 1;
    CHECK(tierFingerprint(p) != base);
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
    // FIX-NUM P2-2: the GPU code object is part of the key (one exe, R9700 + 8060S kernels)
    {
        FingerprintInputs a = in, b = in;
        a.legacy_no_gpu = b.legacy_no_gpu = false;
        a.gpu_arch = "gfx1201";
        b.gpu_arch = "gfx1151";
        a.kernel_set = b.kernel_set = "co 1";
        CHECK(tierFingerprint(a) != tierFingerprint(b));
        CHECK(tierFingerprint(a) != base && tierFingerprint(b) != base);
        CHECK(tierFingerprint(a) == tierFingerprint(a));
        // device name and kernel set (FR-CACHE P1-1)
        FingerprintInputs e = a;
        e.gpu_name = "AMD Radeon(TM) 8060S Graphics";
        CHECK(tierFingerprint(e) != tierFingerprint(a));
        e = a;
        e.kernel_set = "gfx1151 co 123";
        CHECK(tierFingerprint(e) != tierFingerprint(a));
        // FIX-TEST-A (FR-TEST P2-3): an empty GPU identity is an error, not the old format
        auto throws = [](const FingerprintInputs& x) {
            try {
                (void)tierFingerprint(x);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        };
        e = a;
        e.gpu_arch.clear();
        CHECK(throws(e));
        e = a;
        e.kernel_set.clear();
        CHECK(throws(e));
        e = a;
        e.gpu_arch.clear();
        e.gpu_name.clear();
        e.kernel_set.clear();
        CHECK(throws(e));  // all three empty: used to hash silently as the pre-FIX-NUM format
        e = a;
        e.legacy_no_gpu = true;  // legacy format with GPU fields set: contradictory
        CHECK(throws(e));
    }
    p.env = {{"Whirl_Relax", "1"}};  // prefix check is case-insensitive, the key is hashed as written
    CHECK(tierFingerprint(p) != base);
    CHECK(fingerprintEnvKept("WHIRL_NUMERICS"));
    CHECK(fingerprintEnvKept("whirl_x"));
    CHECK(!fingerprintEnvKept("WHIRL_KV_TIER_MIN"));
    CHECK(!fingerprintEnvKept("WHIRL_TIMER_PROBE"));
    // CACHEBAL: the cache-hit diagnostic compares, it does not change the stored KV format, so a verify run can
    // restore the entries a normal server wrote (the tier p1 gate does exactly that)
    CHECK(!fingerprintEnvKept("WHIRL_PREFIX_CACHE_VERIFY"));
    CHECK(!fingerprintEnvKept("WHIRL_LAUNCH_LOG"));  // ARCH-1 K-0 launch table (diagnostic)
    CHECK(!fingerprintEnvKept("WHIRL_LEAKCHECK_QUIET"));  // safe025 leak-check report switch (test builds only; rule I6)
#ifdef _WIN32
    {
        // I6: the switch on and off gives the same fingerprint environment and the same fingerprint
        const auto off = fingerprintEnv();
        _putenv_s("WHIRL_LEAKCHECK_QUIET", "1");
        const auto on = fingerprintEnv();
        _putenv_s("WHIRL_LEAKCHECK_QUIET", "");
        CHECK(on == off);
        auto q = p;
        q.env = off;
        const auto fp_off = tierFingerprint(q);
        q.env = on;
        CHECK(tierFingerprint(q) == fp_off);
    }
#endif
    CHECK(!fingerprintEnvKept("WHIRLX"));
    // the process environment is already filtered
    for (const auto& [k, v] : fingerprintEnv()) CHECK(fingerprintEnvKept(k));
}

// arch025-S0a (I6, REVIEW-ARCH025 A1): the baseline dumps are diagnostics; turning them on must not change the SSD
// fingerprint of the process environment (else a t2.ssd / restart case run with a dump on would miss the entries
// a run without it wrote). Control: an ordinary WHIRL_* variable does change it, so the env path is live.
static void setEnv(const char* k, const char* v) {
#ifdef _WIN32
    _putenv_s(k, v != nullptr ? v : "");
#else
    if (v != nullptr) setenv(k, v, 1); else unsetenv(k);
#endif
}
static void testFingerprintDumpEnv() {
    FingerprintInputs in;
    in.exe_id = 0x1234abcd;
    in.model_base = "Swift-1.5-27B-MXFP4.gguf";
    in.tensors = 851;
    in.bytes = 16'000'000'000ull;
    in.kv_name = "q8h";
    in.use_mtp = true;
    in.n_gdn = 48;
    in.ck_bytes = 1 << 20;
    in.page_bytes = 1 << 18;
    in.max_batch = 4096;
    in.numerics = "balance (items a,b)";
    in.gpu_arch = "gfx1201";
    in.gpu_name = "AMD Radeon AI PRO R9700";
    in.kernel_set = "gfx1201 co 123";
    const char* dumps[] = {"WHIRL_ALLOC_LOG",    "WHIRL_ALLOC_LOG_RAW",  "WHIRL_DUMP_EXEC",
                           "WHIRL_DUMP_KV",      "WHIRL_LAUNCH_LOG_RAW", "WHIRL_LAUNCH_LOG_RAW_MAX"};
    for (const char* k : dumps) setEnv(k, nullptr);
    in.env = fingerprintEnv();
    const std::uint64_t off = tierFingerprint(in);
    for (const char* k : dumps) CHECK(!fingerprintEnvKept(k));
    setEnv("WHIRL_ALLOC_LOG", "C:\tmp\alloc");
    setEnv("WHIRL_ALLOC_LOG_RAW", "server");
    setEnv("WHIRL_DUMP_EXEC", "1");
    setEnv("WHIRL_DUMP_KV", "1");
    setEnv("WHIRL_LAUNCH_LOG_RAW", "1");
    setEnv("WHIRL_LAUNCH_LOG_RAW_MAX", "1000");
    in.env = fingerprintEnv();
    CHECK(tierFingerprint(in) == off);
    setEnv("WHIRL_S0A_CONTROL", "1");  // not a skipped name: part of the key
    in.env = fingerprintEnv();
    CHECK(tierFingerprint(in) != off);
    setEnv("WHIRL_S0A_CONTROL", nullptr);
    for (const char* k : dumps) setEnv(k, nullptr);
    in.env = fingerprintEnv();
    CHECK(tierFingerprint(in) == off);
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
    // FIX-CACHE P2-4 task lineage (task_end: end of the new conversation's first message after the system one)
    {
        // compact / another conversation with the same system prompt (a subagent): shares only the system
        // prompt -> supersedes but is not cut
        const auto n = cat(sys, seq(1500, 500000));
        const Supersede s = supersedes(old_t, n, sys.size(), {}, 3600);
        CHECK(s.yes && s.keep == 3000 && !s.cut);
        // the first message shared too (compress): cut
        auto m = old_t;
        m.resize(9000);
        m = cat(m, seq(4000, 600000));
        const Supersede c = supersedes(old_t, m, sys.size(), {}, 3600);
        CHECK(c.yes && c.keep == 9000 && c.cut);
        // the first message only partly shared: not cut; task_end unknown (max): not cut; 0: the old rule
        auto q = old_t;
        q.resize(3400);
        q = cat(q, seq(1500, 610000));
        CHECK(supersedes(old_t, q, sys.size(), {}, 3600).yes && !supersedes(old_t, q, sys.size(), {}, 3600).cut);
        CHECK(!supersedes(old_t, m, sys.size(), {}, static_cast<std::size_t>(-1)).cut);
        CHECK(supersedes(old_t, n, sys.size()).cut);
        // marks carry it
        SupersedeTracker t2;
        t2.mark(5, 6, 3000, 1, false);
        const auto pp = t2.take(5);
        CHECK(pp.size() == 1 && !pp[0].cut);
    }
    // FIX-CACHE P3-3: near-end divergence (in-place spill allowed) uses the same threshold
    CHECK(nearEndDivergence(23000, 23000 - 600) && nearEndDivergence(23000, 23000 - 5000));  // 25 % of 23000 = 5750
    CHECK(!nearEndDivergence(23000, 3000) && !nearEndDivergence(23000, 23000 - 5750));
    CHECK(nearEndDivergence(6000, 3000) && !nearEndDivergence(9000, 3000));  // min_tail 4096
    CHECK(nearEndDivergence(5000, 6000));  // continues past the end
    // bug-s5a-cachemiss: restore classification (kind codes: gen-end / prompt-end < 4, system 4, prefix 5)
    CHECK(!restoreIsShared(1, 3876, 3876, 3909));  // gen-end: the session itself
    CHECK(!restoreIsShared(5, 3072, 3876, 3876));  // own tail checkpoint, the request continues the whole entry
    CHECK(!restoreIsShared(5, 3072, 3876, 3909));
    CHECK(restoreIsShared(5, 3072, 3876, 3875));   // diverges inside the entry: another question (doc_qa)
    CHECK(restoreIsShared(5, 3072, 3876, 3100));
    CHECK(restoreIsShared(4, 11433, 11433, 15017)); // a shared system entry (ends at its checkpoint)
    CHECK(restoreIsShared(5, 8192, 8192, 9000));    // a shared LCP prefix entry
    CHECK(!restoreIsShared(4, 2048, 6000, 6000));   // a session entry continued at its system checkpoint
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

// arch025-N2a: every WHIRL_* variable numerics::plan reads (Request::env, its A/B switches, Request::runtime) is kept by
// the tier fingerprint (not on its skip list), and changes Plan::bitsKey() in at least one context (value x model x
// mode x card x path) — so a bit-affecting variable can never be in the plan but invisible to bitsKey, or in bitsKey
// but skipped by the fingerprint. Exceptions with their owner: FD_SPLITS (attention splits: resolveFdSplits, the KV
// step N3; noteSwitches records it).
static void testFingerprintCoversBitsKey() {
    namespace nu = whirl::numerics;
    auto tgt = [](bool moe, bool s8060) {
        nu::Target t;
        t.gfx1151 = s8060;
        t.has_mxfp4 = true;
        t.moe = moe;
        t.moe_mxfp4 = moe;
        t.k_gemm8 = t.k_gemm8_moe = t.k_gemm8t = t.k_gdn_wmma = !s8060;
        t.k_kv_q8h = t.k_kv_q4 = t.mtp = t.k_spec_sample = t.k_gdn_v0 = true;
        t.fast_kv_q8h = s8060;
        t.strix_profile = t.k_gdn_sx = s8060;
        t.gdn_strix_shape = true;
        return t;
    };
    std::vector<std::string> names;
    for (const auto& e : nu::EnvValues::names()) names.push_back(e.name);
    for (std::size_t i = 0; nu::k_numerics_switches[i] != nullptr; ++i) names.push_back(nu::k_numerics_switches[i]);
    for (const auto& e : nu::RuntimeValues::names()) names.push_back(e.name);
    const std::vector<std::string> outside = {"FD_SPLITS"};
    const char* values[] = {"0", "1", "", "2", "5", "f16", "q4", "512", "0x3", "0.5", "64", "4096"};
    std::size_t n_checked = 0;
    for (const std::string& n : names) {
        CHECK(fingerprintEnvKept("WHIRL_" + n));
        if (std::find(outside.begin(), outside.end(), n) != outside.end()) continue;
        bool changes = false;
        for (const bool moe : {false, true})
            for (const bool s8060 : {false, true})
                for (const char* mode : {"precise", "balance", "fast"})
                    for (const char* v : values)
                        for (int path = 0; path < 3 && !changes; ++path) {
                            nu::Request r = nu::parseMode(mode, "t");
                            r.load_overrides = true;
                            if (path == 0) {
                                nu::Raw* f = r.env.find(n);
                                if (!f) continue;
                                *f = std::string(v);
                            } else {
                                r.front = path == 1 ? nu::Front::cli : nu::Front::server;
                                bool found = false;
                                for (const auto& e : nu::RuntimeValues::names())
                                    if (n == e.name) {
                                        r.runtime.*e.field = std::string(v);
                                        found = true;
                                    }
                                if (!found) continue;
                            }
                            nu::Request r0 = nu::parseMode(mode, "t");
                            r0.load_overrides = true;
                            r0.front = r.front;
                            const nu::Target t = tgt(moe, s8060);
                            try {
                                changes = nu::plan(r, t).bitsKey() != nu::plan(r0, t).bitsKey();
                            } catch (const std::exception&) {  // a value loadOrTune rejects (RELAX_K=f16): next
                            }
                        }
        if (!changes) std::printf("FAIL WHIRL_%s never changes numerics::Plan::bitsKey()\n", n.c_str());
        CHECK(changes);
        ++n_checked;
    }
    CHECK(n_checked + outside.size() == names.size());
    std::printf("  fingerprint covers bitsKey: %zu plan variables, %zu outside the plan\n", n_checked, outside.size());
}

// arch025 PL1 (high risk of the plan: a changed environment enumeration makes every SSD entry miss). Pins:
//  (1) fingerprintEnv() == a verbatim copy of the v0.2.4 / arch025-base loop (GetEnvironmentStringsW, narrow, the
//      '=' search from 1, fingerprintEnvKept), same pairs in the same order, on the live environment and with a fixed
//      set added (value with '=', non-ASCII value, lower-case name, a skipped name);
//  (2) the fixed set, filtered from fingerprintEnv(), gives exactly the literal pairs below;
//  (3) tierFingerprint of fixed inputs + that set == 0xe30c1ff63c026f60, computed independently by
//      scratch\a25_pl1\fp_expected.py (FNV-1a re-implementation; tierFingerprint itself is unchanged by PL1).
// arch025-S0a recorded no fingerprint value for arch025-base, so (1) is the base pin: the reference IS the base code.
#ifdef _WIN32
static std::vector<std::pair<std::string, std::string>> baseFingerprintEnv() {
    std::vector<std::pair<std::string, std::string>> kv;
    if (wchar_t* blk = GetEnvironmentStringsW()) {
        for (const wchar_t* p = blk; *p; p += wcslen(p) + 1) {
            const std::string s = whirl::narrow(p);
            const std::size_t eq = s.find('=', 1);
            if (eq == std::string::npos) continue;
            std::string k = s.substr(0, eq);
            if (fingerprintEnvKept(k)) kv.emplace_back(k, s.substr(eq + 1));
        }
        FreeEnvironmentStringsW(blk);
    }
    return kv;
}
#endif

static void testFingerprintPlatformEnv() {
#ifdef _WIN32
    CHECK(fingerprintEnv() == baseFingerprintEnv());  // the environment the test runs in
    _putenv_s("WHIRL_PL1T_B", "2");
    _putenv_s("WHIRL_PL1T_A", "1");
    _putenv_s("whirl_pl1t_lower", "x");
    _putenv_s("WHIRL_PL1T_EQ", "a=b");
    _wputenv_s(L"WHIRL_PL1T_U", L"\u00e9\u4e2d");
    _putenv_s("WHIRL_KV_SSD_DIR", "D:\\pl1");  // skipped name
    const auto got = fingerprintEnv();
    CHECK(got == baseFingerprintEnv());
    std::vector<std::pair<std::string, std::string>> mine;
    for (const auto& p : got) {
        std::string up = p.first;
        for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        if (up.rfind("WHIRL_PL1T_", 0) == 0 || up == "WHIRL_KV_SSD_DIR") mine.push_back(p);
    }
    std::vector<std::pair<std::string, std::string>> want = {
        {"WHIRL_PL1T_A", "1"}, {"WHIRL_PL1T_B", "2"}, {"WHIRL_PL1T_EQ", "a=b"}, {"WHIRL_PL1T_U", "\xc3\xa9\xe4\xb8\xad"},
        {"whirl_pl1t_lower", "x"}};
    auto sorted = mine;
    std::sort(sorted.begin(), sorted.end());
    CHECK(sorted == want);
    FingerprintInputs in;
    in.exe_id = 0x1234abcd;
    in.model_base = "Swift-1.5-27B-MXFP4.gguf";
    in.tensors = 851;
    in.bytes = 16'000'000'000ull;
    in.kv_name = "q8h";
    in.use_mtp = true;
    in.n_gdn = 48;
    in.ck_bytes = 1 << 20;
    in.page_bytes = 1 << 18;
    in.max_batch = 4096;
    in.numerics = "balance (items a,b)";
    in.gpu_arch = "gfx1201";
    in.gpu_name = "AMD Radeon AI PRO R9700";
    in.kernel_set = "gfx1201 co 123";
    in.env = mine;
    const std::uint64_t fp = tierFingerprint(in);
    if (fp != 0xe30c1ff63c026f60ull) std::printf("  PL1 fingerprint 0x%016llx (expected 0xe30c1ff63c026f60)\n",
                                                 static_cast<unsigned long long>(fp));
    CHECK(fp == 0xe30c1ff63c026f60ull);
    for (const char* k : {"WHIRL_PL1T_B", "WHIRL_PL1T_A", "whirl_pl1t_lower", "WHIRL_PL1T_EQ", "WHIRL_PL1T_U", "WHIRL_KV_SSD_DIR"})
        _putenv_s(k, "");
    CHECK(fingerprintEnv() == baseFingerprintEnv());
#endif
}

int main() {
    whirl::test::clearWhirlEnv();  // FR-TEST P3-11
    testFingerprintCoversBitsKey();
    testHelpers();
    testMatchSlot();
    testCkptSlotFor();
    testShared();
    testEvict();
    testTier();
    testLcpTarget();
    testFingerprint();
    testFingerprintDumpEnv();
    testFingerprintPlatformEnv();  // arch025 PL1
    try {
        testServerFingerprintInputs();
    } catch (const std::exception& e) {  // e.g. a field the assembly no longer fills: report, do not abort
        std::printf("testServerFingerprintInputs threw: %s\n", e.what());
        CHECK(false);
    }
    testSupersede();
    std::printf("cache tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
