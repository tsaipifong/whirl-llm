// Tests of the host prefix-cache tiers (src/tier/kv_tier) against a
// host-memory DeviceOps mock (no GPU): device addresses are host addresses,
// copies run immediately, events are always complete.
// SPDX-License-Identifier: Apache-2.0
//
// Usage: whirl-tier-tests [temp dir for the SSD tier]
//   (default: %WHIRL_TEST_TMP%\tier_test_tmp, else %TEMP%\whirl-tests\tier_test_tmp)

#include "../test_env.h"
#include "tier/kv_tier.h"
#include "tier/ram_size.h"
#include "tier/evict_policy.h"
#include "tier/idle_flush.h"
#include "cache/prefix_cache.h"

#include <malloc.h>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace whirl::tier;
namespace fs = std::filesystem;

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

// ---- mock ---------------------------------------------------------------------

class MockDeviceOps final : public DeviceOps {
public:
    std::atomic<long> live_events{0}, live_host{0}, live_streams{0};
    std::atomic<std::uint64_t> copies{0};
    // fault injection: copyAsync call number `throw_at` (counted from `copy_calls`) throws
    std::atomic<std::uint64_t> copy_calls{0}, throw_at{0};

    void copyAsync(DevPtr dst, DevPtr src, std::uint64_t bytes, Stream) override {
        if (const std::uint64_t n = ++copy_calls; n == throw_at.load()) throw std::runtime_error("injected copyAsync failure");
        std::memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), bytes);
        copies += 1;
    }
    void download(void* dst, DevPtr src, std::uint64_t bytes) override {
        std::memcpy(dst, reinterpret_cast<const void*>(src), bytes);
    }
    Stream streamCreateNonBlocking() override {
        live_streams += 1;
        return new int(0);
    }
    void streamDestroy(Stream s) override {
        live_streams -= 1;
        delete static_cast<int*>(s);
    }
    void streamSync(Stream) override {}
    void streamWaitEvent(Stream, Event) override {}
    Event eventCreate() override {
        live_events += 1;
        return new int(0);
    }
    void eventDestroy(Event e) override {
        live_events -= 1;
        delete static_cast<int*>(e);
    }
    void eventRecord(Event, Stream) override {}
    bool eventDone(Event) override { return true; }  // like HipDeviceOps: a query error reads as complete
    // fault injection: an asynchronous device error (the work before the event failed)
    std::atomic<bool> event_error{false};
    bool eventDoneChecked(Event) override {
        if (event_error.load()) throw std::runtime_error("injected asynchronous device error");
        return true;
    }
    // the IO thread waits here on a read job's fence: the gate lets a test
    // hold SSD reads back deterministically
    void eventSync(Event) override {
        std::unique_lock<std::mutex> lk(gate_m_);
        gate_cv_.wait(lk, [&] { return gate_open_; });
    }
    void* hostMalloc(std::uint64_t bytes) override {
        void* p = _aligned_malloc(static_cast<std::size_t>(bytes), 4096);
        if (p == nullptr) throw std::bad_alloc();
        live_host += 1;
        return p;
    }
    void hostFree(void* p) override {
        live_host -= 1;
        _aligned_free(p);
    }
    DevPtr malloc(std::uint64_t bytes) override {
        void* p = _aligned_malloc(static_cast<std::size_t>(bytes), 4096);
        if (p == nullptr) throw std::bad_alloc();
        return reinterpret_cast<DevPtr>(p);
    }
    void free(DevPtr p) override { _aligned_free(reinterpret_cast<void*>(p)); }
    void setDevice(int) override {}
    const char* lastErrorString() const override { return "mock"; }

    void closeGate() {
        std::lock_guard<std::mutex> lk(gate_m_);
        gate_open_ = false;
    }
    void openGate() {
        {
            std::lock_guard<std::mutex> lk(gate_m_);
            gate_open_ = true;
        }
        gate_cv_.notify_all();
    }

private:
    std::mutex gate_m_;
    std::condition_variable gate_cv_;
    bool gate_open_ = true;
};

// ---- helpers ------------------------------------------------------------------

static const Layout kLay = [] {
    Layout l;
    l.ck_bytes = (3ull << 20) + 12345;  // ckStride = 4 MiB (2 blocks)
    l.page_bytes = 1572864;             // 1.5 MiB
    l.page_tokens = 256;
    l.tok_cap = 8192;
    return l;
}();

static std::uint64_t splitmix(std::uint64_t& s) {
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static void fillBytes(void* p, std::uint64_t n, std::uint64_t seed) {
    auto* b = static_cast<std::uint8_t*>(p);
    std::uint64_t s = seed;
    std::uint64_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const std::uint64_t v = splitmix(s);
        std::memcpy(b + i, &v, 8);
    }
    for (; i < n; ++i) b[i] = static_cast<std::uint8_t>(splitmix(s));
}

static std::vector<std::uint32_t> makeTokens(std::size_t n, std::uint64_t seed) {
    std::vector<std::uint32_t> t(n);
    std::uint64_t s = seed;
    for (auto& x : t) x = static_cast<std::uint32_t>(splitmix(s) % 150000);
    return t;
}

// device-side copy of one slot: checkpoints + KV pages
struct DevSlot {
    MockDeviceOps* ops = nullptr;
    std::uint32_t nck = 0, pages = 0;
    std::vector<DevPtr> ck;
    DevPtr kv = 0;

    DevSlot(MockDeviceOps& o, std::uint32_t nck_, std::uint32_t pages_) : ops(&o), nck(nck_), pages(pages_) {
        for (std::uint32_t k = 0; k < nck; ++k) {
            ck.push_back(o.malloc(kLay.ck_bytes));
            std::memset(reinterpret_cast<void*>(ck.back()), 0, kLay.ck_bytes);
        }
        kv = o.malloc(kvBytes());
        std::memset(reinterpret_cast<void*>(kv), 0, kvBytes());
    }
    ~DevSlot() {
        for (auto p : ck) ops->free(p);
        ops->free(kv);
    }
    DevSlot(const DevSlot&) = delete;
    DevSlot& operator=(const DevSlot&) = delete;
    std::uint64_t kvBytes() const { return static_cast<std::uint64_t>(pages) * kLay.page_bytes; }
    void fill(std::uint64_t seed) {
        for (std::uint32_t k = 0; k < nck; ++k) fillBytes(reinterpret_cast<void*>(ck[k]), kLay.ck_bytes, seed * 100 + k);
        fillBytes(reinterpret_cast<void*>(kv), kvBytes(), seed * 100 + 99);
    }
    bool ckEq(const DevSlot& o, std::uint32_t k) const {
        return std::memcmp(reinterpret_cast<void*>(ck[k]), reinterpret_cast<void*>(o.ck[k]), kLay.ck_bytes) == 0;
    }
    bool ckZero(std::uint32_t k) const {
        const auto* b = reinterpret_cast<const std::uint8_t*>(ck[k]);
        return std::all_of(b, b + kLay.ck_bytes, [](std::uint8_t x) { return x == 0; });
    }
    bool kvEq(const DevSlot& o) const {
        return std::memcmp(reinterpret_cast<void*>(kv), reinterpret_cast<void*>(o.kv), kvBytes()) == 0;
    }
    // spans: the matched checkpoint and the KV pages first (main part), then
    // the other valid checkpoints (tail)
    std::vector<Span> spans(const Entry* e, std::uint32_t main_ck, std::size_t* n_main) const {
        std::vector<Span> v;
        v.push_back(Span{ck[main_ck], static_cast<std::uint64_t>(main_ck) * kLay.ckStride(), kLay.ck_bytes,
                         static_cast<std::uint8_t>(main_ck)});
        v.push_back(Span{kv, e->kvOff(kLay), kvBytes(), 0xff});
        *n_main = v.size();
        for (std::uint32_t k = 0; k < nck; ++k) {
            if (k == main_ck || e->ck[k].valid == 0) continue;
            v.push_back(Span{ck[k], static_cast<std::uint64_t>(k) * kLay.ckStride(), kLay.ck_bytes, static_cast<std::uint8_t>(k)});
        }
        return v;
    }
};

// spill a slot into a new entry (all checkpoints valid; ck k at position
// (k + 1) * n_tok / nck, the last one with logits)
static Entry* spill(Tier& t, const DevSlot& d, const std::vector<std::uint32_t>& toks) {
    Entry* e = t.newEntry(d.nck);
    e->tokens = toks;
    e->n_tok = static_cast<std::uint32_t>(toks.size());
    e->pages = d.pages;
    for (std::uint32_t k = 0; k < d.nck; ++k) {
        e->ck[k].pos = static_cast<std::uint32_t>((k + 1) * toks.size() / d.nck);
        e->ck[k].valid = 1;
        e->ck[k].has_logits = k + 1 == d.nck ? 1 : 0;
        e->ck[k].seq = k + 1;
    }
    std::size_t n_main = 0;
    const auto sp = d.spans(e, d.nck - 1, &n_main);
    for (const Span& s : sp) {
        const auto r = Tier::lbRange(s.off, s.len);
        if (!t.ensureBlocks(e, r[0], r[1])) {
            t.removeEntry(e);
            return nullptr;
        }
    }
    t.copyOut(e, sp);
    t.fence(e, nullptr);
    e->t_mod = std::chrono::steady_clock::now();
    return e;
}

static bool runRestore(Tier& t, Restore* r, int timeout_ms = 20000) {
    const auto dl = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!t.pumpRestore(r)) {
        if (std::chrono::steady_clock::now() > dl) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    while (!t.pumpTail(r)) {
        if (std::chrono::steady_clock::now() > dl) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return !r->failed;
}

static bool tickIdle(Tier& t, int timeout_ms = 20000) {
    const auto dl = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    // waiting only for write delays to expire is idle (the engine sleeps then)
    while (t.tick() && !t.waitOnly()) {
        if (std::chrono::steady_clock::now() > dl) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

static std::uint64_t fileBytesOf(std::uint32_t nck, std::uint32_t pages) {
    const std::uint64_t x = kLay.dataOff() + nck * kLay.ckStride() + static_cast<std::uint64_t>(pages) * kLay.page_bytes;
    return (x + sector - 1) / sector * sector;
}

static std::size_t countFiles(const fs::path& dir) {
    std::size_t n = 0;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec))
        if (de.path().extension() == ".wkv") ++n;
    return n;
}

static fs::path entryFile(const fs::path& dir, std::uint64_t fp, std::uint64_t id) {
    char name[64];
    std::snprintf(name, sizeof name, "%016llx-%016llx.wkv", static_cast<unsigned long long>(fp),
                  static_cast<unsigned long long>(id));
    return dir / name;
}

static std::vector<std::uint8_t> readFile(const fs::path& p) {
    std::vector<std::uint8_t> v;
    FILE* f = _wfopen(p.c_str(), L"rb");
    if (f == nullptr) return v;
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize(static_cast<std::size_t>(n));
    if (n > 0 && std::fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    std::fclose(f);
    return v;
}

static std::vector<std::uint32_t> prefixPlus(const std::vector<std::uint32_t>& a, std::size_t keep, std::size_t total,
                                             std::uint64_t seed) {
    std::vector<std::uint32_t> v(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(keep));
    auto extra = makeTokens(total - keep, seed);
    for (auto& x : extra) x += 200000;  // never equal to the original ids
    v.insert(v.end(), extra.begin(), extra.end());
    return v;
}

// ---- tests ------------------------------------------------------------------------

static void testLookup(MockDeviceOps& ops) {
    auto t = Tier::create(ops, 64ull << 20, 1, 0);
    Config c;
    t->start(c, kLay, 0x1234);
    CHECK(t->pinnedBytes() == (64ull << 20));
    CHECK(!t->hasSsd());
    DevSlot d(ops, 2, 2);
    d.fill(1);
    const auto toks = makeTokens(512, 7);
    Entry* e = spill(*t, d, toks);
    CHECK(e != nullptr);
    if (e == nullptr) return;
    CHECK((e->id & 1) == 1);
    CHECK(t->find(e->id) == e);
    CHECK(t->countRam() == 1);
    // ck0, ck1: 2 blocks each; KV 3 MiB at 8 MiB: 2 blocks
    CHECK(t->ramUsedBytes() == 6 * block_bytes);
    CHECK(e->ck[0].pos == 256 && e->ck[1].pos == 512);
    // verifySpans: the spilled bytes equal the device bytes
    std::size_t nm = 0;
    const auto sp = d.spans(e, 1, &nm);
    const auto v = t->verifySpans(e, sp);
    CHECK(v.bad == 0 && v.bytes == 2 * kLay.ck_bytes + d.kvBytes());

    auto longer = prefixPlus(toks, 512, 600, 3);
    auto m = t->lookup(longer);
    CHECK(m && m->e == e && m->reuse == 512 && m->ck == 1 && m->lcp == 512);
    // COLD-RESUME: a refused checkpoint (here: kind gen-end at 512) falls back to the next accepted one
    const std::uint8_t kind1 = e->ck[1].kind;
    e->ck[1].kind = 3;
    m = t->lookup(longer, [](std::uint32_t, std::uint8_t k) { return k != 3; });
    CHECK(m && m->reuse == 256 && m->ck == 0);
    m = t->lookup(longer, [](std::uint32_t, std::uint8_t) { return false; });
    CHECK(!m);
    m = t->lookup(longer);  // no predicate: the longest, as before
    CHECK(m && m->reuse == 512 && m->ck == 1);
    e->ck[1].kind = kind1;
    // full length: allowed only with logits
    m = t->lookup(toks);
    CHECK(m && m->reuse == 512 && m->ck == 1);
    e->ck[1].has_logits = 0;
    m = t->lookup(toks);
    CHECK(m && m->reuse == 256 && m->ck == 0 && m->lcp == 512);
    e->ck[1].has_logits = 1;
    // diverging at 300: longest checkpoint <= lcp
    m = t->lookup(prefixPlus(toks, 300, 700, 4));
    CHECK(m && m->reuse == 256 && m->ck == 0 && m->lcp == 300);
    // diverging at 100: no checkpoint
    m = t->lookup(prefixPlus(toks, 100, 700, 5));
    CHECK(!m);
    // invalid checkpoint skipped
    e->ck[1].valid = 0;
    m = t->lookup(longer);
    CHECK(m && m->ck == 0);
    e->ck[1].valid = 1;
    // shorter query than the checkpoint
    m = t->lookup(std::span<const std::uint32_t>(toks.data(), 400));
    CHECK(m && m->ck == 0 && m->reuse == 256);
    // tick without SSD: nothing busy
    CHECK(!t->tick());
    t->removeEntry(e);
    CHECK(t->entries.empty());
    CHECK(t->ramUsedBytes() == 0);
}

static void testEvictLru(MockDeviceOps& ops) {
    // 8 blocks; each entry: ck (2 blocks) + 2 pages (3 MiB at 4 MiB: 2 blocks)
    auto t = Tier::create(ops, 16ull << 20, 2, 0);
    t->start(Config{}, kLay, 1);
    DevSlot da(ops, 1, 2), db(ops, 1, 2), dc(ops, 1, 2);
    da.fill(11);
    db.fill(12);
    dc.fill(13);
    Entry* a = spill(*t, da, makeTokens(512, 1));
    Entry* b = spill(*t, db, makeTokens(512, 2));
    CHECK(a && b);
    if (!a || !b) return;
    const std::uint64_t ida = a->id, idb = b->id;
    CHECK(t->ramUsedBytes() == 8 * block_bytes);
    // c needs 4 blocks: the LRU entry (a, no SSD copy) is dropped
    Entry* c = spill(*t, dc, makeTokens(512, 3));
    CHECK(c != nullptr);
    CHECK(t->find(ida) == nullptr);
    CHECK(t->find(idb) == b);
    CHECK(t->stats.ram_evictions == 1 && t->stats.dropped == 1);
    // pinned / busy entries are not evicted
    b->pin = 1;
    c->io_busy = true;
    Entry* x = t->newEntry(1);
    CHECK(!t->ensureBlocks(x, 0, 2));
    CHECK(t->find(idb) == b && c != nullptr && t->find(c->id) == c);
    // more blocks than the arena has
    CHECK(!t->ensureBlocks(x, 0, 9));
    b->pin = 0;
    c->io_busy = false;
    // now b (older) goes
    CHECK(t->ensureBlocks(x, 0, 2));
    CHECK(t->find(idb) == nullptr && t->find(c->id) == c);
    CHECK(t->stats.ram_evictions == 2 && t->stats.dropped == 2);
    // an existing block is not allocated twice
    const auto before = t->ramUsedBytes();
    CHECK(t->ensureBlocks(x, 0, 2));
    CHECK(t->ramUsedBytes() == before);
}

static void testTrimDropCk(MockDeviceOps& ops) {
    auto t = Tier::create(ops, 64ull << 20, 3, 0);
    t->start(Config{}, kLay, 1);
    DevSlot d(ops, 2, 4);  // ck 4 blocks + KV 6 MiB (3 blocks)
    d.fill(21);
    Entry* e = spill(*t, d, makeTokens(1000, 9));
    CHECK(e != nullptr);
    if (!e) return;
    CHECK(e->blocks.size() == 7);
    CHECK(t->ramUsedBytes() == 7 * block_bytes);
    // shrink to one page: extent 8 + 1.5 MiB -> 5 blocks
    e->pages = 1;
    t->trim(e);
    CHECK(e->blocks.size() == 5 && e->dirty.size() == 5);
    CHECK(t->ramUsedBytes() == 5 * block_bytes);
    // drop checkpoint 1: blocks 2, 3
    t->dropCkBlocks(e, 1);
    CHECK(e->blocks[2] == no_block && e->blocks[3] == no_block);
    CHECK(e->dirty[2] == 0 && e->dirty[3] == 0);
    CHECK(e->blocks[0] != no_block && e->blocks[4] != no_block);
    CHECK(t->ramUsedBytes() == 3 * block_bytes);
    // dropping again is a no-op
    t->dropCkBlocks(e, 1);
    CHECK(t->ramUsedBytes() == 3 * block_bytes);
    auto r = Tier::lbRange(3 * block_bytes + 5, block_bytes);
    CHECK(r[0] == 3 && r[1] == 5);
}

static void testRestoreRam(MockDeviceOps& ops) {
    auto t = Tier::create(ops, 64ull << 20, 4, 0);
    t->start(Config{}, kLay, 1);
    DevSlot src(ops, 2, 5);
    src.fill(31);
    Entry* e = spill(*t, src, makeTokens(1200, 10));
    CHECK(e != nullptr);
    if (!e) return;
    DevSlot dst(ops, 2, 5);
    std::size_t nm = 0;
    const auto sp = dst.spans(e, 1, &nm);
    Restore* r = t->beginRestore(e, sp, nm);
    CHECK(r != nullptr);
    if (!r) return;
    CHECK(!r->from_ssd && r->job == nullptr);
    CHECK(e->pin == 1);
    // pieces: <= 1 MiB, the main part (ck1 + KV) first
    bool small = true;
    for (const auto& p : r->pieces) small = small && p.len <= max_piece;
    CHECK(small);
    std::size_t main_pieces = 0;
    for (std::size_t i = 0; i < 2; ++i) {
        const std::uint64_t len = sp[i].len;
        std::uint64_t off = sp[i].off, done = 0;
        while (done < len) {
            const std::uint64_t inner = (off + done) % block_bytes;
            done += std::min(std::min(len - done, block_bytes - inner), max_piece);
            ++main_pieces;
        }
    }
    CHECK(r->n_main == main_pieces);
    CHECK(r->pieces[0].ck == 1 && r->pieces[0].lb == 2);
    CHECK(r->pieces[r->n_main].ck == 0 && r->pieces[r->n_main].lb == 0);
    CHECK(r->bytes == 2 * kLay.ck_bytes + src.kvBytes());
    CHECK(runRestore(*t, r));
    const auto v = t->verifyPieces(r->pieces);
    CHECK(v.bad == 0 && v.bytes == r->bytes);
    t->endRestore(r);
    CHECK(e->pin == 0);
    CHECK(dst.ckEq(src, 0) && dst.ckEq(src, 1) && dst.kvEq(src));
    // restore of a range that is not in RAM: cannot
    t->dropCkBlocks(e, 0);
    DevSlot dst2(ops, 2, 5);
    const auto sp2 = dst2.spans(e, 1, &nm);
    CHECK(t->beginRestore(e, sp2, nm) == nullptr);
    CHECK(e->pin == 0);
}

struct SsdEnv {
    fs::path dir;
    Config cfg(std::uint64_t cap) const {
        Config c;
        const std::u8string u8 = dir.u8string();
        c.ssd_dir = std::string(reinterpret_cast<const char*>(u8.data()), u8.size());
        c.ssd_cap = cap;
        c.ssd_delay_ms = 0;
        c.ssd_seen_filter = false;  // the seen-once filter has its own test (testSeenOnce)
        return c;
    }
};

static void testSsd(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    const std::uint64_t fp1 = hashStr(0xcbf29ce484222325ull, "model-a");
    const std::uint64_t fp2 = hashStr(0xcbf29ce484222325ull, "model-b");
    CHECK(fp1 != fp2);
    const std::uint32_t nck = 2, pages = 7;  // 4 + 6 blocks; the file ends inside the last block
    const std::uint64_t fb = fileBytesOf(nck, pages);
    DevSlot src(ops, nck, pages);
    src.fill(41);
    const auto toks = makeTokens(1700, 11);
    std::uint64_t id_e = 0;
    std::vector<std::string> logs;
    std::atomic<int> wakes{0};
    {
        // 32 MiB: 16 blocks (one 10-block entry fits, not two)
        auto t = Tier::create(ops, 32ull << 20, 5, 0);
        t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
        t->setWake([&] { wakes += 1; });
        t->start(env.cfg(1ull << 30), kLay, fp1);
        CHECK(t->hasSsd());
        CHECK(fs::is_directory(dir));
        CHECK(t->ssdEntries() == 0 && t->ssd_bytes == 0);
        Entry* e = spill(*t, src, toks);
        CHECK(e != nullptr);
        if (!e) return;
        id_e = e->id;
        CHECK(e->blocks.size() == 10);
        // SSD write after ssd_delay_ms (0)
        CHECK(tickIdle(*t));
        CHECK(e->ssd.has_value());
        CHECK(!e->hasDirty() && !e->io_busy);
        // bytes actually written: header twice, tokens, block table, data region
        CHECK(t->stats.ssd_writes == 1 && t->stats.ssd_write_bytes == 3 * header_bytes + 8192 + (fb - kLay.dataOff()));
        CHECK(t->ssd_bytes == fb);
        CHECK(wakes.load() >= 1);
        CHECK(!logs.empty() && logs.back().find("written to SSD") != std::string::npos);
        const fs::path f = entryFile(dir, fp1, e->id);
        CHECK(fs::exists(f) && fs::file_size(f) == fb);
        {
            const auto bytes = readFile(f);
            CHECK(bytes.size() == fb);
            if (bytes.size() == fb) {
                Header h;
                std::memcpy(&h, bytes.data(), sizeof h);
                CHECK(std::memcmp(h.magic, "WHKVTIER", 8) == 0 && h.version == 2 && h.valid == 1);
                CHECK(h.fingerprint == fp1 && h.id == e->id && h.n_tok == 1700 && h.nck == nck && h.pages == pages);
                CHECK(h.tok_off == header_bytes && h.data_off == kLay.dataOff() && h.ck_stride == kLay.ckStride());
                CHECK(std::memcmp(bytes.data() + header_bytes, toks.data(), toks.size() * 4) == 0);
                // data region: ck1 at 1 * stride, KV at nck * stride
                CHECK(std::memcmp(bytes.data() + kLay.dataOff() + kLay.ckStride(), reinterpret_cast<void*>(src.ck[1]),
                                  kLay.ck_bytes) == 0);
                CHECK(std::memcmp(bytes.data() + kLay.dataOff() + e->kvOff(kLay), reinterpret_cast<void*>(src.kv),
                                  src.kvBytes()) == 0);
            }
        }
        // nothing more to write
        CHECK(!t->tick());
        CHECK(t->stats.ssd_writes == 1);

        // the RAM entry grows (dirty, newer than its SSD copy), then is evicted:
        // it falls back to the SSD snapshot
        DevSlot newer(ops, nck, pages);
        newer.fill(42);
        e->tokens.push_back(5);
        e->n_tok = 1701;
        std::size_t nm = 0;
        t->copyOut(e, newer.spans(e, 1, &nm));
        t->fence(e, nullptr);
        CHECK(e->hasDirty());
        DevSlot fsrc(ops, nck, pages);
        fsrc.fill(43);
        Entry* fe = spill(*t, fsrc, makeTokens(1700, 12));
        CHECK(fe != nullptr);
        CHECK(t->find(id_e) == e);
        CHECK(!e->in_ram && e->blocks.empty() && e->n_tok == 1700 && e->tokens.size() == 1700);
        CHECK(t->stats.ram_evictions == 1 && t->stats.dropped == 0);
        CHECK(t->countRam() == 1 && t->ssdEntries() == 1);

        // lookup finds the SSD-only entry; restore it from SSD (QD4 reads,
        // pieces pipelined as blocks arrive); fe (no SSD copy) is dropped
        auto m = t->lookup(prefixPlus(toks, 1700, 1800, 13));
        CHECK(m && m->e == e && m->ck == 1 && m->reuse == 1700);
        {
            DevSlot dst(ops, nck, pages);
            const auto sp = dst.spans(e, 1, &nm);
            Restore* r = t->beginRestore(e, sp, nm);
            CHECK(r != nullptr);
            if (r) {
                CHECK(r->from_ssd && r->job != nullptr);
                CHECK(e->in_ram && e->loading && e->io_busy);
                CHECK(t->stats.dropped == 1);
                CHECK(runRestore(*t, r));
                t->endRestore(r);
                CHECK(!e->loading && !e->io_busy && e->pin == 0);
                CHECK(dst.ckEq(src, 0) && dst.ckEq(src, 1) && dst.kvEq(src));
                CHECK(!e->hasDirty());
            }
        }
        // clean RAM entry with an SSD copy: no rewrite
        CHECK(!t->tick());
        CHECK(t->stats.ssd_writes == 1);

        // prefetch (no restore takes it): becomes a clean RAM entry
        Entry* g = spill(*t, fsrc, makeTokens(1700, 14));  // evicts e (falls back to SSD)
        CHECK(g != nullptr && !e->in_ram);
        if (g) t->removeEntry(g);
        CHECK(t->prefetch(e));
        CHECK(!t->prefetch(e));  // already loading
        CHECK(e->in_ram && e->loading && e->prefetching);
        // lookup sees a prefetching entry
        m = t->lookup(toks);
        CHECK(m && m->e == e);
        CHECK(tickIdle(*t));
        CHECK(e->in_ram && !e->loading && !e->prefetching && !e->io_busy);
        CHECK(!logs.empty() && logs.back().find("prefetched from SSD into RAM") != std::string::npos);
        {
            DevSlot dst(ops, nck, pages);
            const auto sp = dst.spans(e, 1, &nm);
            Restore* r = t->beginRestore(e, sp, nm);
            CHECK(r != nullptr && !r->from_ssd);
            if (r) {
                CHECK(runRestore(*t, r));
                t->endRestore(r);
                CHECK(dst.ckEq(src, 0) && dst.ckEq(src, 1) && dst.kvEq(src));
            }
        }

        // prefetch taken over by a restore (the read is held at its fence)
        g = spill(*t, fsrc, makeTokens(1700, 15));
        CHECK(g != nullptr && !e->in_ram);
        if (g) t->removeEntry(g);
        ops.closeGate();
        CHECK(t->prefetch(e));
        {
            DevSlot dst(ops, nck, pages);
            const auto sp = dst.spans(e, 1, &nm);
            Restore* r = t->beginRestore(e, sp, nm);
            CHECK(r != nullptr);
            if (r) {
                CHECK(r->from_ssd && r->job != nullptr && !e->prefetching);
                CHECK(r->cursor == 0);
                CHECK(!t->pumpRestore(r));
                ops.openGate();
                CHECK(runRestore(*t, r));
                t->endRestore(r);
                CHECK(dst.ckEq(src, 0) && dst.ckEq(src, 1) && dst.kvEq(src));
                CHECK(!e->io_busy && !e->loading && e->pin == 0);
            } else {
                ops.openGate();
            }
        }
        CHECK(!t->tick());

        // cutTail: the tail (ck0) is dropped before any of it was enqueued
        g = spill(*t, fsrc, makeTokens(1700, 16));
        CHECK(g != nullptr && !e->in_ram);
        if (g) t->removeEntry(g);
        ops.closeGate();
        {
            DevSlot dst(ops, nck, pages);
            const auto sp = dst.spans(e, 1, &nm);
            Restore* r = t->beginRestore(e, sp, nm);
            CHECK(r != nullptr);
            if (r) {
                CHECK(r->cursor == 0 && r->pieces.size() > r->n_main);
                Event cut = t->cutTail(r);
                CHECK(cut != nullptr && r->ev_cut == cut);
                CHECK(r->pieces.size() == r->n_main);
                ops.openGate();
                CHECK(runRestore(*t, r));
                t->endRestore(r);
                CHECK(dst.ckEq(src, 1) && dst.kvEq(src));
                CHECK(dst.ckZero(0));
            } else {
                ops.openGate();
            }
        }
        CHECK(!t->tick());
        CHECK(countFiles(dir) == 1);
    }
    CHECK(ops.live_events.load() == 0 && ops.live_host.load() == 0 && ops.live_streams.load() == 0);

    // startup index, same fingerprint: the entry is kept (SSD-only)
    {
        auto t = Tier::create(ops, 32ull << 20, 6, 0);
        t->start(env.cfg(1ull << 30), kLay, fp1);
        CHECK(t->ssdEntries() == 1 && t->entries.size() == 1);
        CHECK(t->ssd_bytes == fb);
        Entry* e = t->find(id_e);
        CHECK(e != nullptr);
        if (e) {
            CHECK(!e->in_ram && e->n_tok == 1700 && e->nck == nck && e->pages == pages && e->tokens == toks);
            CHECK(e->ck[1].valid == 1 && e->ck[1].has_logits == 1 && e->ck[1].pos == 1700);
            auto m = t->lookup(toks);
            CHECK(m && m->e == e && m->reuse == 1700);
            DevSlot dst(ops, nck, pages);
            std::size_t nm = 0;
            const auto sp = dst.spans(e, 1, &nm);
            Restore* r = t->beginRestore(e, sp, nm);
            CHECK(r != nullptr);
            if (r) {
                CHECK(runRestore(*t, r));
                t->endRestore(r);
                CHECK(dst.ckEq(src, 0) && dst.ckEq(src, 1) && dst.kvEq(src));
            }
        }
        CHECK(!t->tick());
    }

    // different fingerprint: foreign file, counted, evicted first by ssdRoom
    std::uint64_t id_h = 0;
    {
        auto t = Tier::create(ops, 32ull << 20, 7, 0);
        t->start(env.cfg(fb), kLay, fp2);  // room for exactly one file
        CHECK(t->ssdEntries() == 0 && t->entries.empty());
        CHECK(t->ssd_bytes == fb);
        CHECK(!t->lookup(toks));
        DevSlot hs(ops, nck, pages);
        hs.fill(51);
        Entry* h = spill(*t, hs, makeTokens(1700, 17));
        CHECK(h != nullptr);
        if (h) {
            id_h = h->id;
            CHECK(tickIdle(*t));
            CHECK(h->ssd.has_value());
            CHECK(t->stats.ssd_evictions == 1);
            CHECK(!fs::exists(entryFile(dir, fp1, id_e)));
            CHECK(fs::exists(entryFile(dir, fp2, id_h)));
            CHECK(t->ssd_bytes == fb);
            CHECK(countFiles(dir) == 1);
        }
    }

    // corrupted token checksum: the file is deleted at startup
    {
        const fs::path f = entryFile(dir, fp2, id_h);
        FILE* fh = _wfopen(f.c_str(), L"r+b");
        CHECK(fh != nullptr);
        if (fh) {
            std::fseek(fh, static_cast<long>(header_bytes), SEEK_SET);
            std::uint32_t x = 0;
            CHECK(std::fread(&x, 4, 1, fh) == 1);
            x ^= 0x5a5a;
            std::fseek(fh, static_cast<long>(header_bytes), SEEK_SET);
            CHECK(std::fwrite(&x, 4, 1, fh) == 1);
            std::fclose(fh);
        }
        auto t = Tier::create(ops, 16ull << 20, 8, 0);
        t->start(env.cfg(1ull << 30), kLay, fp2);
        CHECK(t->ssdEntries() == 0 && t->ssd_bytes == 0);
        CHECK(!fs::exists(f));
        CHECK(countFiles(dir) == 0);
    }

    // removeEntry deletes the SSD file
    {
        auto t = Tier::create(ops, 32ull << 20, 9, 0);
        t->start(env.cfg(1ull << 30), kLay, fp1);
        Entry* e = spill(*t, src, toks);
        CHECK(e != nullptr);
        if (e) {
            CHECK(tickIdle(*t));
            CHECK(countFiles(dir) == 1);
            t->removeEntry(e);
            CHECK(countFiles(dir) == 0 && t->ssd_bytes == 0 && t->stats.ssd_evictions == 1);
        }
    }
    CHECK(ops.live_events.load() == 0 && ops.live_host.load() == 0 && ops.live_streams.load() == 0);
    if (g_fail == 0) fs::remove_all(dir, ec);
}

// RAM-tier eviction policy (src/tier/evict_policy.h): written to SSD or dropped (TIER-EVICT)
static void testEvictPolicy() {
    EvictFacts b;  // a reused, never-written 4096-token entry, SSD tier on, shutdown mode
    b.ssd_on = true;
    b.shutdown_only = true;
    b.seen_filter = true;
    b.reused = true;
    b.n_tok = 4096;
    b.min_tokens = 2048;
    CHECK(evictWriteWanted(b));
    // shutdown mode: a one-shot prompt is never written, whatever the seen filter says
    EvictFacts f = b;
    f.reused = false;
    CHECK(!evictWriteWanted(f));
    f.seen_filter = false;
    CHECK(!evictWriteWanted(f));
    // always mode: seen-once filter decides
    f = b;
    f.shutdown_only = false;
    CHECK(evictWriteWanted(f));
    f.reused = false;
    CHECK(!evictWriteWanted(f));
    f.seen_filter = false;
    CHECK(evictWriteWanted(f));
    // never written: SSD off, demoted, empty, below min_tokens (both modes)
    for (bool sh : {true, false}) {
        f = b;
        f.shutdown_only = sh;
        f.ssd_on = false;
        CHECK(!evictWriteWanted(f));
        f = b;
        f.shutdown_only = sh;
        f.demoted = true;
        CHECK(!evictWriteWanted(f));
        f = b;
        f.shutdown_only = sh;
        f.n_tok = 0;
        f.min_tokens = 0;
        CHECK(!evictWriteWanted(f));
        f = b;
        f.shutdown_only = sh;
        f.n_tok = 2047;
        CHECK(!evictWriteWanted(f));
        f.n_tok = 2048;
        CHECK(evictWriteWanted(f));
        // current SSD copy: not rewritten; dirty since: rewritten
        f = b;
        f.shutdown_only = sh;
        f.on_ssd = true;
        CHECK(!evictWriteWanted(f));
        f.dirty = true;
        CHECK(evictWriteWanted(f));
    }
}

static void testHelpers() {
    CHECK(hashStr(0xcbf29ce484222325ull, "") == 0xcbf29ce484222325ull);
    CHECK(hashStr(0xcbf29ce484222325ull, "a") == 0xaf63dc4c8601ec8cull);  // FNV-1a 64 of "a"
    CHECK(exeIdentity() != 0);
    CHECK(kLay.ckStride() == (4ull << 20));
    CHECK(kLay.tokRegion() == 32768 && kLay.tblRegion() == 4096 && kLay.dataOff() == 40960);
}

// default size of the pinned-RAM tier (src/tier/ram_size.h)
static void testRamTierSize() {
    constexpr std::uint64_t G = 1ull << 30;
    const auto sz = [](std::uint64_t total, std::uint64_t avail, std::uint64_t floor_mb = 8192) {
        RamTierInput in;
        in.total_phys = total;
        in.avail_phys = avail;
        in.floor_mb = floor_mb;
        return ramTierSize(in);
    };
    // 1/4 of physical RAM, nearest GiB
    CHECK(sz(64 * G, 60 * G).mb == 16384);
    CHECK(sz(68341796864ull, 50 * G).mb == 16384);  // a "64 GB" machine: 63.65 GiB visible
    CHECK(sz(48 * G, 40 * G).mb == 12288);
    CHECK(!sz(64 * G, 60 * G).avail_limited);
    // floor: 8 GiB, or one full-length session if more
    CHECK(sz(16 * G, 0).mb == 8192);
    CHECK(sz(16 * G, 14 * G).mb == 7168 && sz(16 * G, 14 * G).avail_limited);  // 8 GiB would be over half of 14
    CHECK(sz(32 * G, 30 * G).mb == 8192);
    CHECK(sz(64 * G, 60 * G, 9216).mb == 16384);
    CHECK(sz(32 * G, 30 * G, 9216).mb == 9216);
    CHECK(sz(32 * G, 30 * G, 4096).mb == 8192);  // never under 8 GiB
    CHECK(sz(32 * G, 30 * G, 9216).reason.find("minimum") != std::string::npos);
    // cap 32 GiB; the floor wins over the cap
    CHECK(sz(128 * G, 120 * G).mb == 32768);
    CHECK(sz(256 * G, 250 * G).mb == 32768);
    CHECK(sz(256 * G, 250 * G).reason.find("capped") != std::string::npos);
    CHECK(sz(256 * G, 250 * G, 40960).mb == 40960);
    // availability: at most half of the RAM available at startup (whole GiB)
    CHECK(sz(64 * G, 20 * G).mb == 10240);
    CHECK(sz(64 * G, 20 * G).avail_limited);
    CHECK(sz(64 * G, 20 * G).reason.find("available") != std::string::npos);
    CHECK(sz(64 * G, 33 * G).mb == 16384 && !sz(64 * G, 33 * G).avail_limited);
    CHECK(sz(64 * G, 31 * G).mb == 15360);
    CHECK(sz(16 * G, 9 * G).mb == 4096);       // also below the floor: paging is worse
    CHECK(sz(16 * G, G + G / 2).mb == 0);      // half would be under 1 GiB: off
    CHECK(sz(16 * G, G + G / 2).avail_limited);
    CHECK(sz(16 * G, 2 * G).mb == 1024);
    // unknown RAM: the floor; unknown availability: no availability cap
    CHECK(sz(0, 0).mb == 8192);
    CHECK(sz(0, 0, 9216).mb == 9216);
    CHECK(sz(0, 6 * G).mb == 3072);
    CHECK(sz(64 * G, 0).mb == 16384);
    // explicit size: used as given (no caps), 0 = off
    {
        RamTierInput in;
        in.total_phys = 16 * G;
        in.avail_phys = 4 * G;
        in.explicit_mb = 20000;
        CHECK(ramTierSize(in).mb == 20000 && !ramTierSize(in).avail_limited);
        CHECK(ramTierSize(in).reason.find("--kv-ram-mb") != std::string::npos);
        in.explicit_mb = 0;
        CHECK(ramTierSize(in).mb == 0);
        in.explicit_mb = 512;
        in.explicit_src = "WHIRL_KV_RAM_MB";
        CHECK(ramTierSize(in).mb == 512 && ramTierSize(in).reason.find("WHIRL_KV_RAM_MB") != std::string::npos);
        in.integrated = true;
        CHECK(ramTierSize(in).mb == 512);  // explicit wins on an integrated GPU too
    }
    // integrated GPU (UMA): OS-visible RAM / 8, in [one entry, 16 GiB], <= 1/4 of available, off < 32 GB
    {
        const auto ig = [](std::uint64_t total, std::uint64_t avail, std::uint64_t entry_mb = 6144) {
            RamTierInput in;
            in.total_phys = total;
            in.avail_phys = avail;
            in.entry_mb = entry_mb;
            in.floor_mb = std::max<std::uint64_t>(8192, entry_mb);  // the discrete floor must not leak in
            in.integrated = true;
            return ramTierSize(in);
        };
        CHECK(ig(68341796864ull, 45 * G).mb == 8192);  // the 8060S box: 63.65 GiB visible -> 8 GiB
        CHECK(!ig(68341796864ull, 45 * G).avail_limited);
        CHECK(ig(68341796864ull, 45 * G).reason.find("1/8") != std::string::npos);
        CHECK(ig(96 * G, 80 * G).mb == 12288);
        CHECK(ig(128 * G, 120 * G).mb == 16384);
        CHECK(ig(256 * G, 250 * G).mb == 16384);  // cap
        CHECK(ig(256 * G, 250 * G).reason.find("capped") != std::string::npos);
        // floor: one max-context entry
        CHECK(ig(34 * G, 30 * G).mb == 6144);  // 34/8 = 4.25 -> 4 GiB, raised to 6
        CHECK(ig(34 * G, 30 * G).reason.find("max-context entry") != std::string::npos);
        CHECK(ig(34 * G, 30 * G, 7168).mb == 7168);
        CHECK(ig(256 * G, 250 * G, 20480).mb == 20480);  // floor beats the cap
        // availability: at most 1/4 of the RAM available at startup (whole GiB)
        CHECK(ig(64 * G, 24 * G).mb == 6144 && ig(64 * G, 24 * G).avail_limited);
        CHECK(ig(64 * G, 32 * G).mb == 8192 && !ig(64 * G, 32 * G).avail_limited);
        CHECK(ig(64 * G, 30 * G).mb == 7168 && ig(64 * G, 30 * G).avail_limited);
        // FIX-CACHE P3-4: 1/4 of the available RAM below one entry -> one entry if within half of it, else off
        CHECK(ig(64 * G, 20 * G).mb == 6144 && ig(64 * G, 20 * G).avail_limited);  // 1/4 = 5 GiB < 6, half = 10
        CHECK(ig(64 * G, 20 * G).reason.find("max-context entry") != std::string::npos);
        CHECK(ig(64 * G, 12 * G).mb == 6144);  // half = 6 GiB: still fits
        CHECK(ig(64 * G, 11 * G).mb == 0 && ig(64 * G, 11 * G).avail_limited);  // half = 5 GiB < one entry: off
        CHECK(ig(64 * G, 11 * G).reason.find("more than half") != std::string::npos);
        CHECK(ig(64 * G, 22 * G, 0).mb == 5120);  // entry size unknown: the 1/4 limit as before
        CHECK(ig(64 * G, 3 * G).mb == 0 && ig(64 * G, 3 * G).avail_limited);  // 1/4 under 1 GiB: off
        CHECK(ig(64 * G, 0).mb == 8192);  // availability unknown: no cap
        // under 32 GB of OS-visible RAM: off, with the reason
        CHECK(ig(16 * G, 12 * G).mb == 0);
        CHECK(ig(24 * G, 20 * G).mb == 0);
        CHECK(ig(24 * G, 20 * G).reason.find("under 32 GB") != std::string::npos);
        CHECK(ig(31999999999ull, 30 * G).mb == 0);
        CHECK(ig(32 * G, 28 * G).mb == 6144);          // a "32 GB" machine (32 GiB visible): 4 GiB -> entry floor 6
        CHECK(ig(32000000000ull, 28 * G).mb == 6144);  // exactly 32 GB: on
        CHECK(ig(0, 0).mb == 0);                       // unknown RAM: off
        // explicit wins on an integrated GPU, also under 32 GB; 0 turns it off
        RamTierInput in;
        in.total_phys = 128 * G;
        in.avail_phys = 100 * G;
        in.integrated = true;
        in.entry_mb = 6144;
        in.explicit_mb = 0;
        CHECK(ramTierSize(in).mb == 0 && ramTierSize(in).reason.find("--kv-ram-mb") != std::string::npos);
        in.explicit_mb = 4096;
        in.total_phys = 16 * G;
        CHECK(ramTierSize(in).mb == 4096);
        in.explicit_mb = 30000;
        in.avail_phys = 8 * G;
        CHECK(ramTierSize(in).mb == 30000);
        // the discrete rule is unchanged by entry_mb
        RamTierInput d;
        d.total_phys = 64 * G;
        d.avail_phys = 60 * G;
        d.entry_mb = 6144;
        CHECK(ramTierSize(d).mb == 16384);
    }
}

// ---- CACHE-1: block dedup, superseded sessions, SSD modes ------------------------------

// One server slot of the simulations: KV page j holds bytes derived from the
// tokens up to the end of page j (a token prefix always has the same KV
// bytes), checkpoints from the whole token list.
struct Sess {
    MockDeviceOps* ops;
    DevSlot d;
    std::vector<std::uint32_t> toks;
    Entry* e = nullptr;  // valid only right after spillSess (use id later)
    std::uint64_t id = 0;
    std::uint64_t seq = 0;
    Sess(MockDeviceOps& o, std::uint32_t max_pages) : ops(&o), d(o, 2, max_pages) {}
    std::uint32_t pages() const { return static_cast<std::uint32_t>((toks.size() + kLay.page_tokens - 1) / kLay.page_tokens); }
    static std::uint64_t sumOf(const std::vector<std::uint32_t>& t, std::size_t n) {
        std::uint64_t h = 1469598103934665603ull;
        for (std::size_t i = 0; i < n; ++i) h = (h ^ t[i]) * 1099511628211ull;
        return h;
    }
    void fillFrom(std::uint32_t pg0) {
        for (std::uint32_t j = pg0; j < pages(); ++j) {
            const std::size_t end = std::min<std::size_t>(toks.size(), static_cast<std::size_t>(j + 1) * kLay.page_tokens);
            fillBytes(reinterpret_cast<void*>(d.kv + static_cast<std::uint64_t>(j) * kLay.page_bytes), kLay.page_bytes, sumOf(toks, end));
        }
    }
};

// Spill the slot like Engine::spillSlot: in place when it still owns its
// entry (pages from the first changed one, the newer checkpoint), else a new
// entry. Returns true for a new entry.
static bool spillSess(Tier& t, Sess& s, std::size_t changed_from) {
    const std::uint32_t P = static_cast<std::uint32_t>(s.toks.size());
    const std::uint32_t npages = s.pages();
    CHECK(npages <= s.d.pages);
    if (npages > s.d.pages) return false;
    Entry* x = s.id != 0 ? t.find(s.id) : nullptr;
    const bool fresh = x == nullptr || !x->in_ram || x->n_tok == 0;
    std::uint32_t vp = 0;
    if (fresh) {
        x = t.newEntry(2);
    } else {
        vp = static_cast<std::uint32_t>(std::min<std::size_t>(changed_from, x->n_tok) / kLay.page_tokens);
    }
    s.fillFrom(vp);
    // keep the checkpoint this request resumed from, overwrite the other
    std::uint32_t k = 0;
    if (!fresh) {
        int keep = -1;
        for (int i = 0; i < 2; ++i)
            if (x->ck[i].valid != 0 && x->ck[i].pos <= changed_from && (keep < 0 || x->ck[i].pos > x->ck[keep].pos)) keep = i;
        k = keep == 0 ? 1u : 0u;
    }
    s.seq += 1;
    fillBytes(reinterpret_cast<void*>(s.d.ck[k]), kLay.ck_bytes, Sess::sumOf(s.toks, P) + k);
    std::vector<Span> sp;
    sp.push_back(Span{s.d.ck[k], static_cast<std::uint64_t>(k) * kLay.ckStride(), kLay.ck_bytes, static_cast<std::uint8_t>(k)});
    if (npages > vp)
        sp.push_back(Span{s.d.kv + static_cast<std::uint64_t>(vp) * kLay.page_bytes,
                          x->kvOff(kLay) + static_cast<std::uint64_t>(vp) * kLay.page_bytes,
                          static_cast<std::uint64_t>(npages - vp) * kLay.page_bytes, 0xff});
    for (const Span& q : sp) {
        const auto r = Tier::lbRange(q.off, q.len);
        if (!t.ensureBlocks(x, r[0], r[1])) {
            if (fresh) t.removeEntry(x);
            s.e = nullptr;
            s.id = 0;
            return false;
        }
    }
    // the other checkpoint is stale past the divergence
    const std::uint32_t o = 1 - k;
    if (x->ck[o].valid != 0 && x->ck[o].pos > changed_from) {
        t.dropCkBlocks(x, o);
        x->ck[o] = CkMeta{};
    }
    x->ck[k] = CkMeta{P, 1, 1, 0, 0, s.seq};
    t.copyOut(x, sp);
    x->tokens = s.toks;
    x->n_tok = P;
    x->pages = npages;
    t.trim(x);
    t.fence(x, nullptr);
    x->owner = 0;
    x->t_mod = std::chrono::steady_clock::now();
    if (!fresh) x->reused = true;  // Engine::spillSlot: an in-place update is a resumed session
    s.e = x;
    s.id = x->id;
    return fresh;
}

// What a restore or a resumed turn does to an entry (Tier restore, Engine::spillSlot): marks it reused. In
// shutdown mode only reused entries are written, also by the flush when WHIRL is closed (CLOSE-REUSED), so
// tests of that flush mark the sessions they expect on the SSD.
static void reuse(Tier& t, const Sess& s) {
    if (Entry* e = t.find(s.id)) e->reused = true;
}

// what Engine::supersedeOld does after a new entry's first spill
static std::uint64_t supersede(Tier& t, const Entry* x, std::size_t sys) {
    std::uint64_t freed = 0;
    for (int pass = 0; pass < 4; ++pass) {
        std::uint64_t f = 0;
        const std::vector<Entry*> all = t.entries;
        for (Entry* e : all) {
            if (e == x || e->n_tok == 0 || std::find(t.entries.begin(), t.entries.end(), e) == t.entries.end()) continue;
            const auto sp = whirl::cache::supersedes(whirl::cache::Tokens(e->tokens.data(), e->n_tok),
                                                     whirl::cache::Tokens(x->tokens.data(), x->n_tok), sys);
            if (sp.yes) f += t.trimTo(e, static_cast<std::uint32_t>(sp.keep));
        }
        freed += f;
        if (f == 0) break;
    }
    return freed;
}

// Engine::supersedeMark / supersedeApply (deferred: cut at the new
// conversation's second spill unless the old entry was updated since)
static std::uint64_t stampOf(const Entry* e) {
    return static_cast<std::uint64_t>(e->t_mod.time_since_epoch().count()) * 0x9e3779b97f4a7c15ull ^ e->n_tok;
}
static void supersedeMark(Tier& t, whirl::cache::SupersedeTracker& tr, const Entry* x, std::size_t sys) {
    for (Entry* e : t.entries) {
        if (e == x || e->n_tok == 0) continue;
        const auto sp = whirl::cache::supersedes(whirl::cache::Tokens(e->tokens.data(), e->n_tok),
                                                 whirl::cache::Tokens(x->tokens.data(), x->n_tok), sys);
        if (sp.yes) tr.mark(x->id, e->id, sp.keep, stampOf(e));
    }
}
static std::uint64_t supersedeApply(Tier& t, whirl::cache::SupersedeTracker& tr, const Entry* x) {
    auto pend = tr.take(x->id);
    std::uint64_t freed = 0;
    for (int pass = 0; pass < 4 && !pend.empty(); ++pass) {
        bool any = false;
        for (auto it = pend.begin(); it != pend.end();) {
            Entry* e = t.find(it->old_id);
            if (e == nullptr || stampOf(e) != it->stamp) {
                it = pend.erase(it);
                continue;
            }
            const std::uint64_t f = t.trimTo(e, static_cast<std::uint32_t>(it->keep));
            if (f == 0) {
                ++it;
                continue;
            }
            freed += f;
            any = true;
            it = pend.erase(it);
        }
        if (!any) break;
    }
    return freed;
}

static std::vector<std::uint32_t> freshToks(std::size_t n, std::uint64_t& seed) {
    auto v = makeTokens(n, seed++);
    for (auto& x : v) x += 300000;
    return v;
}

// Byte-compare a restore of entry e with the slot that produced it.
static bool restoreMatches(Tier& t, MockDeviceOps& ops, Entry* e, const Sess& s) {
    DevSlot dst(ops, 2, s.d.pages);
    std::uint32_t k = 0;
    for (std::uint32_t i = 0; i < 2; ++i)
        if (e->ck[i].valid != 0 && e->ck[i].pos == e->n_tok) k = i;
    std::vector<Span> sp;
    sp.push_back(Span{dst.ck[k], static_cast<std::uint64_t>(k) * kLay.ckStride(), kLay.ck_bytes, static_cast<std::uint8_t>(k)});
    sp.push_back(Span{dst.kv, e->kvOff(kLay), static_cast<std::uint64_t>(e->pages) * kLay.page_bytes, 0xff});
    Restore* r = t.beginRestore(e, sp, sp.size());
    if (r == nullptr) return false;
    const bool ok = runRestore(t, r);
    t.endRestore(r);
    if (!ok) return false;
    const bool ck_ok = std::memcmp(reinterpret_cast<void*>(dst.ck[k]), reinterpret_cast<void*>(s.d.ck[k]), kLay.ck_bytes) == 0;
    const bool kv_ok = std::memcmp(reinterpret_cast<void*>(dst.kv), reinterpret_cast<void*>(s.d.kv),
                                   static_cast<std::uint64_t>(e->pages) * kLay.page_bytes) == 0;
    return ck_ok && kv_ok;
}

static void testDedupRestart(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    const std::uint64_t fp = hashStr(0xcbf29ce484222325ull, "dedup");
    std::uint64_t seed = 900;
    const auto sys = freshToks(2560, seed);  // 10 pages = 15 MiB = 7 whole blocks
    Sess a(ops, 24), b(ops, 24);
    a.toks = sys;
    auto ta = freshToks(1500, seed);
    a.toks.insert(a.toks.end(), ta.begin(), ta.end());
    b.toks = sys;
    auto tb = freshToks(900, seed);
    b.toks.insert(b.toks.end(), tb.begin(), tb.end());
    std::uint64_t id_a = 0, id_b = 0;
    {
        auto t = Tier::create(ops, 256ull << 20, 7, 0);
        t->start(env.cfg(4ull << 30), kLay, fp);
        spillSess(*t, a, 0);
        CHECK(tickIdle(*t));
        const std::uint64_t w_a = t->stats.ssd_write_bytes;
        spillSess(*t, b, 0);
        CHECK(tickIdle(*t));
        const std::uint64_t w_b = t->stats.ssd_write_bytes - w_a;
        id_a = a.e->id;
        id_b = b.e->id;
        CHECK(b.e->ssd.has_value());
        std::size_t refs = 0;
        if (b.e->ssd)
            for (const BlkRef& r : b.e->ssd->tbl) refs += r.owner == id_a ? 1 : 0;
        CHECK(refs == 7);  // the shared system prompt: 7 whole blocks
        CHECK(t->stats.ssd_dedup_bytes == 7 * block_bytes);
        std::printf("dedup: entry a wrote %.1f MiB, entry b (same 2560-token prefix) wrote %.1f MiB, %.0f MiB shared\n",
                    w_a / 1048576.0, w_b / 1048576.0, t->stats.ssd_dedup_bytes / 1048576.0);
        CHECK(w_b + 6 * block_bytes <= w_a);
    }
    {
        // restart: both indexed, references verified, b restores from both files
        auto t = Tier::create(ops, 256ull << 20, 8, 0);
        t->start(env.cfg(4ull << 30), kLay, fp);
        CHECK(t->ssdEntries() == 2);
        Entry* eb = t->find(id_b);
        Entry* ea = t->find(id_a);
        CHECK(eb != nullptr && !eb->in_ram && ea != nullptr);
        if (eb && ea) CHECK(eb->lru <= ea->lru);  // owners stay newer than the files referencing them
        if (eb) CHECK(restoreMatches(*t, ops, eb, b));
        CHECK(tickIdle(*t));
        if (ea) CHECK(restoreMatches(*t, ops, ea, a));
        CHECK(tickIdle(*t));
    }
    {
        // the owner file vanishes (crash, manual delete): b is dropped at start, never restored
        fs::remove(entryFile(dir, fp, id_a), ec);
        auto t = Tier::create(ops, 256ull << 20, 9, 0);
        t->start(env.cfg(4ull << 30), kLay, fp);
        CHECK(t->ssdEntries() == 0);
        CHECK(countFiles(dir) == 0);
    }
    {
        // a half-written file (header still invalid) is ignored and deleted
        fs::remove_all(dir, ec);
        auto t = Tier::create(ops, 256ull << 20, 10, 0);
        t->start(env.cfg(4ull << 30), kLay, fp);
        Sess a2(ops, 24);
        a2.toks = a.toks;
        spillSess(*t, a2, 0);
        CHECK(tickIdle(*t));
        const fs::path f = entryFile(dir, fp, a2.e->id);
        t.reset();
        {
            FILE* fh = _wfopen(f.c_str(), L"r+b");
            CHECK(fh != nullptr);
            if (fh) {
                std::uint32_t zero = 0;
                std::fseek(fh, 12, SEEK_SET);  // Header::valid
                std::fwrite(&zero, 4, 1, fh);
                std::fclose(fh);
            }
        }
        auto t2 = Tier::create(ops, 256ull << 20, 10, 0);
        t2->start(env.cfg(4ull << 30), kLay, fp);
        CHECK(t2->ssdEntries() == 0 && !fs::exists(f));
    }
    {
        // dedup off: nothing shared
        fs::remove_all(dir, ec);
        auto t = Tier::create(ops, 256ull << 20, 10, 0);
        Config c = env.cfg(4ull << 30);
        c.dedup = false;
        t->start(c, kLay, fp);
        Sess a2(ops, 24), b2(ops, 24);
        a2.toks = a.toks;
        b2.toks = b.toks;
        spillSess(*t, a2, 0);
        CHECK(tickIdle(*t));
        spillSess(*t, b2, 0);
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_dedup_bytes == 0);
    }
    fs::remove_all(dir, ec);
}

static void testOwnerRewrite(MockDeviceOps& ops, const fs::path& dir) {
    // an owner rewritten in place under a block others reference: the
    // referencing file is dropped first (its RAM copy is written again)
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 950;
    auto t = Tier::create(ops, 256ull << 20, 11, 0);
    t->start(env.cfg(4ull << 30), kLay, 77);
    Sess a(ops, 24), b(ops, 24);
    a.toks = freshToks(3000, seed);
    b.toks = a.toks;
    b.toks.resize(2800);
    auto tb = freshToks(500, seed);
    b.toks.insert(b.toks.end(), tb.begin(), tb.end());
    spillSess(*t, a, 0);
    CHECK(tickIdle(*t));
    spillSess(*t, b, 0);
    CHECK(tickIdle(*t));
    CHECK(t->dependentsOf(a.e).size() == 1);
    // a regenerates from token 1000 (in place): blocks b references change
    a.toks.resize(1000);
    auto ta = freshToks(2500, seed);
    a.toks.insert(a.toks.end(), ta.begin(), ta.end());
    spillSess(*t, a, 1000);
    CHECK(tickIdle(*t));
    CHECK(b.e->ssd.has_value());  // written again
    // DEDUP-INV: b's write waits for a's rewrite and then shares only what a still has in common
    // (tokens [0, 1000): 3 pages = 2 whole blocks), never a changed block
    if (b.e->ssd && a.e->ssd) {
        const std::uint32_t a_lb0 = static_cast<std::uint32_t>(a.e->ssd->nck * kLay.ckStride() / block_bytes);
        std::size_t refs_a = 0, bad = 0;
        for (const BlkRef& r : b.e->ssd->tbl) {
            if (r.owner == 0) continue;
            bad += r.owner != a.id || r.owner_lb >= a_lb0 + 2 ? 1 : 0;
            refs_a += r.owner == a.id ? 1 : 0;
        }
        CHECK(bad == 0 && refs_a <= 2);
    }
    CHECK(restoreMatches(*t, ops, a.e, a));
    const std::uint64_t id_b = b.id;
    t.reset();
    auto t2 = Tier::create(ops, 256ull << 20, 12, 0);
    t2->start(env.cfg(4ull << 30), kLay, 77);
    CHECK(t2->ssdEntries() == 2);
    Entry* eb = t2->find(id_b);
    CHECK(eb != nullptr);
    if (eb) CHECK(restoreMatches(*t2, ops, eb, b));
    CHECK(tickIdle(*t2));
    t2.reset();
    fs::remove_all(dir, ec);
}

static void testModesLock(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 990;
    {
        // shutdown mode: nothing written until flushing (a reused entry: the only kind the close flush writes)
        auto t = Tier::create(ops, 128ull << 20, 13, 0);
        Config c = env.cfg(4ull << 30);
        c.ssd_shutdown_only = true;
        t->start(c, kLay, 5);
        Sess a(ops, 24);
        a.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        reuse(*t, a);
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
        // a second server on the same directory: SSD tier off
        auto t2 = Tier::create(ops, 32ull << 20, 14, 0);
        t2->start(c, kLay, 5);
        CHECK(!t2->hasSsd() && t2->ssdDirBusy());
        t2.reset();
        t->flushing = true;
        CHECK(tickIdle(*t));
        t->flushing = false;
        CHECK(t->stats.ssd_writes == 1 && countFiles(dir) == 1);
    }
    {
        // after the first server stopped, the lock is gone
        auto t = Tier::create(ops, 32ull << 20, 15, 0);
        t->start(env.cfg(4ull << 30), kLay, 5);
        CHECK(t->hasSsd() && t->ssdEntries() == 1);
    }
    fs::remove_all(dir, ec);
}

// MAIN-001: a truncated SSD file is never restored as valid KV. Cut at startup (header only, mid
// payload, last sector) the entry is not indexed and the file is removed; cut after the index the
// read job fails (restore failed -> cold prefill). A legal partial last block stays restorable.
static void testSsdTruncated(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    const std::uint64_t fp = hashStr(0xcbf29ce484222325ull, "model-trunc");
    const std::uint32_t nck = 2, pages = 7;  // the file ends inside its last block
    const std::uint64_t fb = fileBytesOf(nck, pages);
    DevSlot src(ops, nck, pages);
    src.fill(77);
    const auto toks = makeTokens(1700, 21);
    std::uint64_t id = 0;
    {
        auto t = Tier::create(ops, 32ull << 20, 5, 0);
        t->start(env.cfg(1ull << 30), kLay, fp);
        Entry* e = spill(*t, src, toks);
        CHECK(e != nullptr);
        if (!e) return;
        id = e->id;
        CHECK(tickIdle(*t));
    }
    const fs::path f = entryFile(dir, fp, id);
    CHECK(fs::exists(f) && fs::file_size(f) == fb);
    const fs::path bak = dir / L"backup.bin";
    fs::copy_file(f, bak, fs::copy_options::overwrite_existing, ec);
    const auto restoreFull = [&] {
        fs::remove(f, ec);
        fs::copy_file(bak, f, fs::copy_options::overwrite_existing, ec);
    };
    const std::uint64_t cuts[] = {header_bytes,                            // header only
                                  kLay.dataOff() + 4 * block_bytes,        // mid payload (block aligned)
                                  kLay.dataOff() + 4 * block_bytes + sector,  // mid block
                                  fb - sector};                           // last sector missing
    for (const std::uint64_t cut : cuts) {
        restoreFull();
        fs::resize_file(f, cut, ec);
        CHECK(!ec && fs::file_size(f) == cut);
        auto t = Tier::create(ops, 32ull << 20, 6, 0);
        t->start(env.cfg(1ull << 30), kLay, fp);
        CHECK(t->ssdEntries() == 0 && t->find(id) == nullptr);
        CHECK(!fs::exists(f));  // dropped like any other corrupt file
        CHECK(t->ssd_bytes == 0);
    }
    // cut after the index (the file changes under a running server): the restore fails cleanly
    for (const std::uint64_t cut : cuts) {
        if (cut < kLay.dataOff()) continue;  // the header read at restore time is not part of the job
        restoreFull();
        auto t = Tier::create(ops, 32ull << 20, 7, 0);
        t->start(env.cfg(1ull << 30), kLay, fp);
        Entry* e = t->find(id);
        CHECK(e != nullptr && t->ssdEntries() == 1);
        if (!e) continue;
        fs::resize_file(f, cut, ec);
        CHECK(!ec);
        DevSlot dst(ops, nck, pages);
        std::size_t nm = 0;
        const auto sp = dst.spans(e, 1, &nm);
        Restore* r = t->beginRestore(e, sp, nm);
        CHECK(r != nullptr);
        if (r) {
            CHECK(!runRestore(*t, r) && r->failed);
            t->endRestore(r);
        }
        tickIdle(*t);
    }
    // control: the intact file (partial last block) still restores byte-identical
    restoreFull();
    {
        auto t = Tier::create(ops, 32ull << 20, 8, 0);
        t->start(env.cfg(1ull << 30), kLay, fp);
        Entry* e = t->find(id);
        CHECK(e != nullptr && t->ssdEntries() == 1);
        if (e) {
            DevSlot dst(ops, nck, pages);
            std::size_t nm = 0;
            const auto sp = dst.spans(e, 1, &nm);
            Restore* r = t->beginRestore(e, sp, nm);
            CHECK(r != nullptr);
            if (r) {
                CHECK(runRestore(*t, r));
                t->endRestore(r);
                CHECK(dst.ckEq(src, 0) && dst.ckEq(src, 1) && dst.kvEq(src));
            }
        }
    }
    fs::remove_all(dir, ec);
}

// RESTORE-EXC: a throwing device copy during a restore marks it failed (cold prefill), it is never
// reported as done-and-valid; the tier stays consistent and the entry restores later.
static void testRestoreCopyThrows(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    const std::uint64_t fp = hashStr(0xcbf29ce484222325ull, "model-throw");
    const std::uint32_t nck = 2, pages = 7;
    DevSlot src(ops, nck, pages);
    src.fill(88);
    const auto toks = makeTokens(1700, 31);
    std::uint64_t id = 0;
    {
        auto t = Tier::create(ops, 32ull << 20, 5, 0);
        t->start(env.cfg(1ull << 30), kLay, fp);
        Entry* e = spill(*t, src, toks);
        CHECK(e != nullptr);
        if (!e) return;
        id = e->id;
        CHECK(tickIdle(*t));
    }
    auto t = Tier::create(ops, 32ull << 20, 6, 0);
    t->start(env.cfg(1ull << 30), kLay, fp);
    Entry* e = t->find(id);
    CHECK(e != nullptr);
    if (!e) return;
    {
        DevSlot dst(ops, nck, pages);
        std::size_t nm = 0;
        const auto sp = dst.spans(e, 1, &nm);
        ops.closeGate();  // the SSD read waits: no copy happens inside beginRestore
        Restore* r = t->beginRestore(e, sp, nm);
        CHECK(r != nullptr);
        if (!r) {
            ops.openGate();
            return;
        }
        // the 2nd copy of this restore throws
        ops.throw_at = ops.copy_calls.load() + 2;
        ops.openGate();
        const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(20);
        while (!t->pumpRestoreNoThrow(r) && std::chrono::steady_clock::now() < dl)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ops.throw_at = 0;
        CHECK(r->failed);
        t->endRestore(r);
        tickIdle(*t);
        // the tier is intact: the same entry restores byte-identical afterwards
        e = t->find(id);
        CHECK(e != nullptr);
        if (!e) return;
        DevSlot ok(ops, nck, pages);
        const auto sp2 = ok.spans(e, 1, &nm);
        Restore* r2 = t->beginRestore(e, sp2, nm);
        CHECK(r2 != nullptr);
        if (r2) {
            CHECK(runRestore(*t, r2));
            t->endRestore(r2);
            CHECK(ok.ckEq(src, 0) && ok.ckEq(src, 1) && ok.kvEq(src));
        }
    }
    fs::remove_all(dir, ec);
}

// TIER-LOCK-FAILOPEN: when the directory lock cannot be taken for a reason other than another
// server holding it (here: whirl.lock is a directory -> access denied), the SSD tier stays off
// instead of running unlocked.
static void testLockFailClosed(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / L"whirl.lock", ec);
    CHECK(!ec);
    SsdEnv env{dir};
    std::vector<std::string> logs;
    {
        auto t = Tier::create(ops, 32ull << 20, 5, 0);
        t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
        t->start(env.cfg(1ull << 30), kLay, 5);
        CHECK(!t->hasSsd());
        CHECK(!t->ssdDirBusy());  // not "another server": a different failure
        CHECK(!logs.empty() && logs.back().find("cannot lock") != std::string::npos);
        // the RAM tier still works
        DevSlot src(ops, 1, 2);
        src.fill(5);
        Entry* e = spill(*t, src, makeTokens(300, 3));
        CHECK(e != nullptr);
        CHECK(tickIdle(*t));
        CHECK(countFiles(dir) == 0);
    }
    fs::remove_all(dir, ec);
}

// TIER-DEP-CYCLE: A owns the shared prefix and B references it. A is then rewritten to B's
// content; its new blocks must not become references to B (A <-> B), otherwise dropping either
// one recursed forever. Also: a cycle that already exists is survivable (guard in dropSsdCopy).
static void testDedupCycle(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 970;
    const auto sys = freshToks(2560, seed);  // 10 pages
    const auto tz = freshToks(1500, seed);
    const auto ty = freshToks(1500, seed);
    Sess a(ops, 24), b(ops, 24);
    a.toks = sys;
    a.toks.insert(a.toks.end(), tz.begin(), tz.end());
    b.toks = sys;
    b.toks.insert(b.toks.end(), ty.begin(), ty.end());
    auto t = Tier::create(ops, 256ull << 20, 21, 0);
    t->start(env.cfg(4ull << 30), kLay, 91);
    spillSess(*t, a, 0);
    CHECK(tickIdle(*t));
    spillSess(*t, b, 0);
    CHECK(tickIdle(*t));
    const std::uint64_t id_a = a.id, id_b = b.id;
    CHECK(t->dependentsOf(a.e).size() == 1);  // B references A's prefix blocks
    // A regenerates its tail to exactly B's content (in place, from the end of the prefix)
    a.toks = b.toks;
    spillSess(*t, a, 2560);
    CHECK(tickIdle(*t));
    Entry* ea = t->find(id_a);
    Entry* eb = t->find(id_b);
    CHECK(ea != nullptr && eb != nullptr && ea->ssd.has_value() && eb->ssd.has_value());
    if (!ea || !eb || !ea->ssd || !eb->ssd) return;
    std::size_t a_refs_b = 0;
    for (const BlkRef& r : ea->ssd->tbl) a_refs_b += r.owner == id_b ? 1 : 0;
    CHECK(a_refs_b == 0);                          // no A -> B reference while B -> A exists
    CHECK(t->dependentsOf(eb).empty());            // nothing depends on B
    CHECK(t->dependentClosure(ea).size() <= 1);    // B at most; never A itself
    CHECK(restoreMatches(*t, ops, ea, a));
    CHECK(tickIdle(*t));
    eb = t->find(id_b);
    if (eb) CHECK(restoreMatches(*t, ops, eb, b));
    CHECK(tickIdle(*t));
    // force a cycle (what an older build could have left on disk) and remove an entry in it:
    // must terminate and leave a consistent tier
    ea = t->find(id_a);
    eb = t->find(id_b);
    if (ea && eb && ea->ssd && eb->ssd && !ea->ssd->tbl.empty() && !eb->ssd->tbl.empty()) {
        BlkRef ra = ea->ssd->tbl.back();
        ra.owner = id_b;
        ea->ssd->tbl.back() = ra;
        BlkRef rb = eb->ssd->tbl.back();
        rb.owner = id_a;
        eb->ssd->tbl.back() = rb;
        const std::vector<std::uint64_t> clo = t->dependentClosure(ea);  // one vector (begin/end of two temporaries was UB)
        CHECK(!clo.empty());
        CHECK(std::find(clo.begin(), clo.end(), id_a) != clo.end());
        t->removeEntry(ea);  // used to recurse until the stack overflowed
        CHECK(t->find(id_a) == nullptr);
    }
    t.reset();
    fs::remove_all(dir, ec);
}

static void testSupersedeTrim(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 1000;
    auto t = Tier::create(ops, 512ull << 20, 16, 0);
    t->start(env.cfg(8ull << 30), kLay, 6);
    const auto sys = freshToks(2048, seed);
    Sess old_s(ops, 40), regen(ops, 40), other(ops, 40), comp(ops, 40);
    old_s.toks = sys;
    auto h = freshToks(7000, seed);
    old_s.toks.insert(old_s.toks.end(), h.begin(), h.end());
    spillSess(*t, old_s, 0);
    CHECK(tickIdle(*t));
    const std::uint64_t id_old = old_s.e->id;
    // regenerate on another slot (diverges near the end): no supersede
    regen.toks = old_s.toks;
    regen.toks.resize(old_s.toks.size() - 300);
    auto r = freshToks(250, seed);
    regen.toks.insert(regen.toks.end(), r.begin(), r.end());
    spillSess(*t, regen, 0);
    CHECK(supersede(*t, regen.e, sys.size()) == 0);
    const std::uint64_t id_regen = regen.e->id;
    // a different system prompt, short: no supersede
    other.toks = freshToks(3000, seed);
    spillSess(*t, other, 0);
    CHECK(supersede(*t, other.e, 1500) == 0);
    CHECK(old_s.e->n_tok == 9048);
    CHECK(tickIdle(*t));
    // compact: system prompt + summary, much shorter
    comp.toks = sys;
    auto s2 = freshToks(1200, seed);
    comp.toks.insert(comp.toks.end(), s2.begin(), s2.end());
    spillSess(*t, comp, 0);
    const std::uint64_t freed = supersede(*t, comp.e, sys.size());
    CHECK(freed > 0);
    CHECK(tickIdle(*t));
    // the old session and its regenerate branch are cut (what comp references stays)
    for (const std::uint64_t id : {id_old, id_regen}) {
        Entry* e = t->find(id);
        if (e) CHECK(e->n_tok <= 2048 + kLay.page_tokens * 2);
    }
    CHECK(t->find(other.id) != nullptr && t->find(other.id)->n_tok == 3000);
    CHECK(restoreMatches(*t, ops, comp.e, comp));
    CHECK(tickIdle(*t));
    const std::uint64_t id_comp = comp.e->id;
    {
        // deferred rule: a short new session of another project (same system
        // prompt) is marked against comp... comp is shorter: nothing. A long
        // session P and a short new one N: P keeps working -> left alone
        whirl::cache::SupersedeTracker tr;
        Sess p(ops, 48), n(ops, 48);
        p.toks = sys;
        auto ph = freshToks(8000, seed);
        p.toks.insert(p.toks.end(), ph.begin(), ph.end());
        spillSess(*t, p, 0);
        CHECK(tickIdle(*t));
        n.toks = sys;
        auto nh = freshToks(900, seed);
        n.toks.insert(n.toks.end(), nh.begin(), nh.end());
        CHECK(spillSess(*t, n, 0));
        supersedeMark(*t, tr, n.e, sys.size());
        CHECK(tr.size() >= 1);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        auto more = freshToks(400, seed);
        const std::size_t pn = p.toks.size();
        p.toks.insert(p.toks.end(), more.begin(), more.end());
        CHECK(!spillSess(*t, p, pn));  // P's next turn (in place)
        n.toks.insert(n.toks.end(), more.begin(), more.end());
        CHECK(!spillSess(*t, n, 900 + sys.size()));
        CHECK(supersedeApply(*t, tr, n.e) == 0);
        CHECK(t->find(p.id) != nullptr && t->find(p.id)->n_tok == p.toks.size());
        CHECK(tickIdle(*t));
    }
    t.reset();
    // restart after the cut: consistent, comp restores bit-identically
    auto t2 = Tier::create(ops, 512ull << 20, 17, 0);
    t2->start(env.cfg(8ull << 30), kLay, 6);
    Entry* ec2 = t2->find(id_comp);
    CHECK(ec2 != nullptr);
    if (ec2) CHECK(restoreMatches(*t2, ops, ec2, comp));
    CHECK(tickIdle(*t2));
    t2.reset();
    fs::remove_all(dir, ec);
}

// Agent workflows: SSD bytes written / used and token hit rate, without
// (dedup off, no supersede) and with CACHE-1.
struct WfResult {
    double written_gib = 0, used_gib = 0, hit = 0;
};

static WfResult runWorkflow(MockDeviceOps& ops, const fs::path& dir, int kind, bool cache1) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    auto t = Tier::create(ops, 256ull << 20, 21, 0);
    Config c = env.cfg(3ull << 30);                    // SSD cap 3 GiB
    c.dedup = cache1;
    t->start(c, kLay, 31);
    std::uint64_t seed = 5000 + static_cast<std::uint64_t>(kind) * 100;
    const auto sys = freshToks(2600, seed);  // system prompt + tool definitions
    std::uint64_t hit_tok = 0, req_tok = 0;
    whirl::cache::SupersedeTracker tr;
    Sess s(ops, 96);
    auto request = [&](const std::vector<std::uint32_t>& toks, std::size_t changed_from, bool new_entry) {
        const auto m = t->lookup(toks);
        hit_tok += m ? m->reuse : 0;
        req_tok += toks.size();
        if (new_entry) s.id = 0;
        s.toks = toks;
        const bool fresh = spillSess(*t, s, changed_from);
        if (cache1 && s.e) {
            if (fresh) supersedeMark(*t, tr, s.e, sys.size());
            else supersedeApply(*t, tr, s.e);
        }
        tickIdle(*t);
    };
    std::vector<std::vector<std::uint32_t>> msgs;  // conversation after the system prompt
    auto build = [&] {
        std::vector<std::uint32_t> v = sys;
        for (const auto& m : msgs) v.insert(v.end(), m.begin(), m.end());
        return v;
    };
    for (int turn = 0; turn < 40; ++turn) {
        if (kind == 0) {
            // periodic compact every 10 turns: a new conversation = system prompt + summary
            if (turn > 0 && turn % 10 == 0) {
                msgs.clear();
                msgs.push_back(freshToks(800, seed));
                request(build(), sys.size(), true);
                continue;
            }
            msgs.push_back(freshToks(400 + static_cast<std::size_t>(turn % 3) * 150, seed));
            const auto cur = build();
            request(cur, cur.size() - msgs.back().size(), false);
        } else if (kind == 1) {
            // sliding window: tool outputs older than 3 turns cut to 60 tokens,
            // the oldest messages dropped beyond 6000 tokens
            msgs.push_back(freshToks(500, seed));
            std::size_t changed = sys.size();
            for (std::size_t i = 0; i < msgs.size(); ++i) {
                if (i + 3 < msgs.size() && msgs[i].size() > 60) {
                    msgs[i].resize(60);
                    break;
                }
                changed += msgs[i].size();
            }
            std::size_t total = 0;
            for (const auto& m : msgs) total += m.size();
            bool dropped = false;
            while (total > 6000) {
                total -= msgs.front().size();
                msgs.erase(msgs.begin());
                dropped = true;
            }
            if (dropped) changed = sys.size();
            const auto cur = build();
            request(cur, std::min(changed, cur.size()), false);
        } else {
            // regenerate: every 4th turn the last answer is retried twice
            msgs.push_back(freshToks(350, seed));
            auto cur = build();
            request(cur, cur.size() - msgs.back().size(), false);
            if (turn % 4 == 3) {
                for (int k = 0; k < 2; ++k) {
                    msgs.back() = freshToks(350, seed);
                    cur = build();
                    request(cur, cur.size() - msgs.back().size(), false);
                }
            }
        }
        // every 5th turn another project with the same tools starts a short session
        if (turn % 5 == 4) {
            Sess o(ops, 96);
            o.toks = sys;
            auto q = freshToks(600, seed);
            o.toks.insert(o.toks.end(), q.begin(), q.end());
            const auto m = t->lookup(o.toks);
            hit_tok += m ? m->reuse : 0;
            req_tok += o.toks.size();
            if (spillSess(*t, o, 0) && cache1 && o.e) supersedeMark(*t, tr, o.e, sys.size());
            tickIdle(*t);
        }
    }
    WfResult r;
    r.written_gib = static_cast<double>(t->stats.ssd_write_bytes) / 1073741824.0;
    r.used_gib = static_cast<double>(t->ssd_bytes) / 1073741824.0;
    r.hit = req_tok ? static_cast<double>(hit_tok) / static_cast<double>(req_tok) : 0;
    t.reset();
    fs::remove_all(dir, ec);
    return r;
}

static void testWorkflows(MockDeviceOps& ops, const fs::path& dir) {
    const char* names[] = {"periodic compact", "sliding window + tool output cut", "regenerate"};
    for (int k = 0; k < 3; ++k) {
        const WfResult b = runWorkflow(ops, dir, k, false);
        const WfResult a = runWorkflow(ops, dir, k, true);
        std::printf("workflow %-34s SSD written %.2f -> %.2f GiB (%+.0f%%), SSD used %.2f -> %.2f GiB, token hit rate %.1f%% -> %.1f%%\n",
                    names[k], b.written_gib, a.written_gib, b.written_gib > 0 ? (a.written_gib / b.written_gib - 1) * 100 : 0,
                    b.used_gib, a.used_gib, b.hit * 100, a.hit * 100);
        CHECK(a.written_gib <= b.written_gib);
        CHECK(a.hit + 0.02 >= b.hit);
    }
}


// ---- SSD-QUIET: write delay, eviction writes, cap, expiry, startup cleanup ----------------

// bytes the .wkv files of dir really occupy (sparse files: without the holes)
static std::uint64_t diskBytes(const fs::path& dir) {
    std::uint64_t n = 0;
    std::error_code ec;
    for (const auto& de : fs::directory_iterator(dir, ec)) {
        if (de.path().extension() != ".wkv") continue;
        DWORD hi = 0;
        const DWORD lo = GetCompressedFileSizeW(de.path().c_str(), &hi);
        n += (static_cast<std::uint64_t>(hi) << 32) | lo;
    }
    return n;
}

static void sleepMs(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

static void testQuietDefaults() {
    Config c;
    CHECK(c.ssd_delay_ms == 300000);
    CHECK(c.expire_s == 7ull * 24 * 3600 && c.expire_check_ms == 3600ull * 1000);
    CHECK(c.ssd_seen_filter);
    CHECK(!c.ssd_shutdown_only);  // the server's default mode (shutdown) is set by server_main
}

// An entry is written once unchanged for ssd_delay_ms; every change (or use) restarts the clock; while
// it only waits, tick() reports wait-only work and the due time.
static void testWriteDelay(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 3100;
    auto t = Tier::create(ops, 128ull << 20, 31, 0);
    Config c = env.cfg(4ull << 30);
    c.ssd_delay_ms = 600;
    t->start(c, kLay, 3);
    Sess a(ops, 24);
    a.toks = freshToks(3000, seed);
    spillSess(*t, a, 0);
    t->tick();
    CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
    CHECK(t->waitOnly() && t->nextDue().has_value());
    sleepMs(300);
    // a new request continues the conversation: the entry changes, the clock restarts
    a.toks.push_back(77);
    spillSess(*t, a, a.toks.size() - 1);
    sleepMs(400);  // 700 ms after the first spill, 400 ms after the change
    t->tick();
    CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
    CHECK(t->waitOnly());
    sleepMs(300);
    CHECK(tickIdle(*t));
    CHECK(t->stats.ssd_writes == 1 && countFiles(dir) == 1);
    CHECK(!t->nextDue().has_value());
    // a dirty entry that is served from RAM (restored) restarts its clock
    a.toks.push_back(78);
    spillSess(*t, a, a.toks.size() - 1);
    CHECK(t->find(a.id)->hasDirty());
    sleepMs(450);
    CHECK(restoreMatches(*t, ops, t->find(a.id), a));
    sleepMs(300);  // 750 ms after the spill, 300 ms after the restore
    t->tick();
    CHECK(t->stats.ssd_writes == 1);
    sleepMs(400);
    CHECK(tickIdle(*t));
    CHECK(t->stats.ssd_writes == 2);
    t.reset();
    fs::remove_all(dir, ec);
}

// RAM pressure before the delay: the evicted entry is written when it has >= min_tokens, else dropped.
static void testEvictionWrite(MockDeviceOps& ops, const fs::path& dir) {
    for (int mode = 0; mode < 3; ++mode) {  // 0: written, 1: below min_tokens (dropped), 2: shutdown mode (dropped)
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        std::uint64_t seed = 3200 + mode;
        auto t = Tier::create(ops, 64ull << 20, 32, 0);  // 32 blocks; a 3000-token entry takes 13
        Config c = env.cfg(4ull << 30);
        c.ssd_delay_ms = 600000;
        c.min_tokens = mode == 1 ? 5000 : 2048;
        c.ssd_shutdown_only = mode == 2;
        t->start(c, kLay, 4);
        Sess a(ops, 24), b(ops, 24), d(ops, 24);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        d.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        const std::uint64_t id_a = a.id;
        sleepMs(5);
        spillSess(*t, b, 0);
        t->tick();
        CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);  // not due
        // the third needs room: the LRU entry a goes
        bool kept = spillSess(*t, d, 0) && d.id != 0;
        if (mode == 0) {
            // a is written first (async); its blocks are free after that and the spill is retried
            CHECK(!kept && t->roomPending());
            CHECK(tickIdle(*t, 20000));
            kept = spillSess(*t, d, 0) && d.id != 0;
            CHECK(kept);
            CHECK(t->stats.ssd_writes >= 1);
            Entry* ea = t->find(id_a);
            CHECK(ea != nullptr && ea->ssd.has_value() && !ea->in_ram);  // evicted to its SSD copy
            CHECK(t->stats.dropped == 0);
            CHECK(countFiles(dir) == 1);
            // room for the restore (RAM is full of dirty entries, which would need writes first)
            if (Entry* eb = t->find(b.id)) t->removeEntry(eb);
            if (ea) CHECK(restoreMatches(*t, ops, ea, a));
            CHECK(tickIdle(*t));
        } else {
            CHECK(kept);
            CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
            CHECK(t->find(id_a) == nullptr && t->stats.dropped == 1);
        }
        t.reset();
        fs::remove_all(dir, ec);
    }
}

// FIX-RC4-A P1 (REVIEW-RC4-A P1-1): `always` mode, SSD cache at its cap, RAM tier full with a dirty entry
// worth a write. Restoring (or prefetching) an SSD-only entry makes room in RAM; the eviction write of the
// dirty victim makes room on SSD (LRU). The entry being restored is the SSD LRU entry at that point and
// must never be the one dropped (it was deleted under beginRestore: use after free).
static void testEvictionWriteSsdFull(MockDeviceOps& ops, const fs::path& dir) {
    for (int mode = 0; mode < 2; ++mode) {  // 0: beginRestore, 1: prefetch
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        std::uint64_t seed = 9100 + static_cast<std::uint64_t>(mode) * 10;
        auto t = Tier::create(ops, 72ull << 20, 91, 0);  // 36 blocks; a spilled 3000-token entry takes 11
        Config c = env.cfg(4ull << 30);
        c.ssd_delay_ms = 0;
        t->start(c, kLay, 9);
        Sess a(ops, 24), d(ops, 24), b(ops, 24), b2(ops, 24);
        a.toks = freshToks(3000, seed);
        d.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        b2.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        const std::uint64_t id_a = a.id;
        CHECK(tickIdle(*t));
        sleepMs(5);
        spillSess(*t, d, 0);
        CHECK(tickIdle(*t));  // a, d written (clean, SSD copies)
        CHECK(t->stats.ssd_writes == 2);
        t->cfg.ssd_delay_ms = 600000;                      // later entries are not due
        t->cfg.ssd_cap = t->ssd_bytes + t->ssd_bytes / 4;  // a + d fill the SSD tier
        sleepMs(5);
        spillSess(*t, b, 0);                               // dirty, not due
        CHECK(restoreMatches(*t, ops, t->find(d.id), d));  // d used: more recent than b
        sleepMs(5);
        spillSess(*t, b2, 0);                              // a (LRU, clean) drops to its SSD copy
        Entry* ea = t->find(id_a);
        CHECK(ea != nullptr && !ea->in_ram && ea->ssd.has_value());
        if (!ea || ea->in_ram) continue;
        if (mode == 0) {
            DevSlot dst(ops, 2, 24);
            std::vector<Span> sp;
            sp.push_back(Span{dst.ck[0], 0, kLay.ck_bytes, 0});
            sp.push_back(Span{dst.kv, ea->kvOff(kLay), static_cast<std::uint64_t>(ea->pages) * kLay.page_bytes, 0xff});
            Restore* r = t->beginRestore(ea, sp, sp.size());
            const bool alive = t->find(id_a) == ea;
            CHECK(alive);  // the restore target was deleted under beginRestore before FIX-RC4-A P1
            if (!alive) {
                std::printf("   P1: entry being restored was dropped by the room-making path (use after free)\n");
                std::fflush(stdout);
                std::_Exit(3);  // the heap is corrupt: do not continue
            }
            if (r) {
                CHECK(runRestore(*t, r));
                t->endRestore(r);
                CHECK(std::memcmp(reinterpret_cast<void*>(dst.ck[0]), reinterpret_cast<void*>(a.d.ck[0]), kLay.ck_bytes) == 0);
                CHECK(std::memcmp(reinterpret_cast<void*>(dst.kv), reinterpret_cast<void*>(a.d.kv),
                                  static_cast<std::uint64_t>(ea->pages) * kLay.page_bytes) == 0);
            } else {
                // not now: the eviction write is running (the engine prefills cold); a stays intact
                CHECK(t->roomPending());
                CHECK(ea->pin == 0 && !ea->in_ram && ea->ssd.has_value() && ea->blocks.empty());
                CHECK(tickIdle(*t));
                ea = t->find(id_a);
                CHECK(ea != nullptr);
                if (ea) CHECK(restoreMatches(*t, ops, ea, a));  // room now (the written victim is clean)
            }
        } else {
            bool started = t->prefetch(ea);
            const bool alive = t->find(id_a) == ea;
            CHECK(alive);
            if (!alive) {
                std::printf("   P1: entry being prefetched was dropped by the room-making path (use after free)\n");
                std::fflush(stdout);
                std::_Exit(3);
            }
            if (!started) {
                CHECK(t->roomPending());
                CHECK(ea->pin == 0 && !ea->in_ram && ea->ssd.has_value() && ea->blocks.empty());
                CHECK(tickIdle(*t));
                ea = t->find(id_a);
                CHECK(ea != nullptr);
                if (ea) started = t->prefetch(ea);
                CHECK(started);
            }
            if (started) {
                CHECK(tickIdle(*t));
                ea = t->find(id_a);
                CHECK(ea != nullptr && ea->in_ram);
                if (ea) CHECK(restoreMatches(*t, ops, ea, a));
            }
        }
        CHECK(t->ssd_bytes <= t->cfg.ssd_cap);
        CHECK(tickIdle(*t));
        for (const Entry* e : t->entries) CHECK(e->pin == 0 && e->ext_pin == 0);
        t.reset();
        fs::remove_all(dir, ec);
    }
}

// Shutdown mode: no write while running even with delay 0, a flush writes every reused entry (CLOSE-REUSED:
// only those; a one-shot prompt is not written, testCloseReusedTier); always mode: the flush (delay forced to
// 0 like Engine::flushTierOnStop) does not wait for the delay either.
static void testShutdownFlush(MockDeviceOps& ops, const fs::path& dir) {
    for (int mode = 0; mode < 2; ++mode) {
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        std::uint64_t seed = 3300 + mode;
        auto t = Tier::create(ops, 128ull << 20, 33, 0);
        Config c = env.cfg(4ull << 30);
        c.ssd_delay_ms = mode == 0 ? 0 : 600000;
        c.ssd_shutdown_only = mode == 0;
        t->start(c, kLay, 5);
        Sess a(ops, 24), b(ops, 24);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        spillSess(*t, b, 0);
        reuse(*t, a);
        reuse(*t, b);
        t->tick();
        sleepMs(50);
        t->tick();
        CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
        t->flushing = true;
        const std::uint64_t saved = t->cfg.ssd_delay_ms;
        t->cfg.ssd_delay_ms = 0;
        CHECK(tickIdle(*t));
        t->cfg.ssd_delay_ms = saved;
        CHECK(t->stats.ssd_writes == 2 && countFiles(dir) == 2);
        t.reset();
        fs::remove_all(dir, ec);
    }
}


// DEDUP-INV: entries that become due for a write in the same pass (the graceful shutdown flush writes every
// session at once; in `always` mode several sessions can pass the write delay on one tick) still share their
// common prefix: an entry whose prefix another entry is writing right now waits for that write instead of
// writing its own copy. Before the fix every startWrite of the pass saw no SSD base (the first file was still
// in flight) and the system prompt was written once per session.
static void testFlushDedup(MockDeviceOps& ops, const fs::path& dir) {
    for (int mode = 0; mode < 2; ++mode) {
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        const std::uint64_t fp = hashStr(0xcbf29ce484222325ull, "flushdedup");
        std::uint64_t seed = 3700 + static_cast<std::uint64_t>(mode) * 10;
        const auto sys = freshToks(2560, seed);  // 10 pages = 7 whole blocks
        Sess s0(ops, 24), s1(ops, 24), s2(ops, 24), u(ops, 24);
        Sess* fam[3] = {&s0, &s1, &s2};
        for (Sess* s : fam) {
            s->toks = sys;
            auto tail = freshToks(600 + 300 * static_cast<std::size_t>(s - &s0), seed);
            s->toks.insert(s->toks.end(), tail.begin(), tail.end());
        }
        u.toks = freshToks(2000, seed);  // unrelated: shares nothing, written alongside
        std::uint64_t ids[3] = {};
        {
            auto t = Tier::create(ops, 256ull << 20, 37, 0);
            Config c = env.cfg(4ull << 30);
            c.ssd_delay_ms = mode == 0 ? 600000 : 200;
            c.ssd_shutdown_only = mode == 0;
            c.min_tokens = 1024;  // u has 2000 tokens (the shutdown-mode close flush applies min_tokens: CLOSE-REUSED)
            t->start(c, kLay, fp);
            for (Sess* s : fam) spillSess(*t, *s, 0);
            spillSess(*t, u, 0);
            for (Sess* s : fam) reuse(*t, *s);  // sessions worth keeping (shutdown mode writes only reused ones)
            reuse(*t, u);
            for (int i = 0; i < 3; ++i) ids[i] = fam[i]->id;
            t->tick();
            CHECK(t->stats.ssd_writes == 0);
            if (mode == 0) {
                t->flushing = true;
                t->cfg.ssd_delay_ms = 0;
            } else {
                sleepMs(250);  // all four pass the write delay on the same tick
            }
            CHECK(tickIdle(*t));
            CHECK(t->stats.ssd_writes == 4);
            std::size_t refs = 0;
            for (std::uint64_t id : ids) {
                const Entry* e = t->find(id);
                CHECK(e != nullptr && e->ssd.has_value());
                if (e && e->ssd)
                    for (const BlkRef& r : e->ssd->tbl) refs += r.owner != 0 ? 1 : 0;
            }
            std::printf("flush dedup (%s): %zu shared block refs, %.0f MiB shared, %.1f MiB written\n",
                        mode == 0 ? "shutdown flush" : "always, same tick", refs, t->stats.ssd_dedup_bytes / 1048576.0,
                        t->stats.ssd_write_bytes / 1048576.0);
            CHECK(refs == 14);  // two of the three store the 7 system-prompt blocks as references
            CHECK(t->stats.ssd_dedup_bytes == 14 * block_bytes);
        }
        {
            // restart: all indexed, references verified, every session restores byte-exact
            auto t = Tier::create(ops, 256ull << 20, 38, 0);
            t->start(env.cfg(4ull << 30), kLay, fp);
            CHECK(t->ssdEntries() == 4);
            for (int i = 0; i < 3; ++i) {
                Entry* e = t->find(ids[i]);
                CHECK(e != nullptr);
                if (e) CHECK(restoreMatches(*t, ops, e, *fam[i]));
                CHECK(tickIdle(*t));
            }
        }
        fs::remove_all(dir, ec);
    }
}

// CB-FLAKE: the base write has finished on the IO thread but the tick that reaps it has not run yet (the
// engine's tick reaps first, then starts writes, while the IO thread keeps running). The entry sharing its
// prefix must still wait for it: the job is still in jobs_ (not reaped), the base has no SSD copy yet, so
// before the fix waitForDedupBase saw neither an in-flight write nor a file on disk and the second entry
// wrote the shared prefix in full (server gate sys: "0 of 2 SSD writes with shared blocks", ~1 in 24 runs
// on the R9700).
static void testDedupFinishedUnreaped(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    const std::uint64_t fp = hashStr(0xcbf29ce484222325ull, "dedupunreaped");
    std::uint64_t seed = 3900;
    const auto sys = freshToks(2560, seed);  // 10 pages = 7 whole blocks
    Sess a(ops, 24), b(ops, 24);
    for (Sess* s : {&a, &b}) {
        s->toks = sys;
        auto tail = freshToks(600 + 300 * static_cast<std::size_t>(s == &b), seed);
        s->toks.insert(s->toks.end(), tail.begin(), tail.end());
    }
    {
        auto t = Tier::create(ops, 256ull << 20, 39, 0);
        Config c = env.cfg(4ull << 30);
        c.ssd_delay_ms = 600000;
        c.ssd_shutdown_only = true;
        t->start(c, kLay, fp);
        spillSess(*t, a, 0);
        spillSess(*t, b, 0);
        reuse(*t, a);
        reuse(*t, b);
        t->tick();
        t->flushing = true;
        t->cfg.ssd_delay_ms = 0;
        t->tick();  // starts the first write; the other entry waits for it (DEDUP-INV)
        Entry* ea = t->find(a.id);
        Entry* eb = t->find(b.id);
        CHECK(ea != nullptr && eb != nullptr);
        if (ea && eb) {
            CHECK(ea->io_busy != eb->io_busy);  // exactly one write running
            Entry* first = ea->io_busy ? ea : eb;
            Entry* second = ea->io_busy ? eb : ea;
            CHECK(!first->ssd.has_value());
            // let the IO thread finish the write; no tick runs, so the job is finished but not reaped.
            // A fixed sleep could check too early, while the write is still in flight, and then pass without
            // exercising the race (an in-flight write was always waited for). REVIEW-022 F4: poll the job's own
            // finished flag (test hook) for up to 20 s, and fail if it never gets there; no timing assumption is left.
            bool written = false;
            for (int ms = 0; ms < 20000 && !written; ms += 10) {
                sleepMs(10);
                written = t->writeFinishedUnreaped(first->id);
            }
            std::printf("dedup base finished, not reaped: first write finished (job not reaped): %s\n", written ? "yes" : "NO");
            CHECK(written);
            CHECK(first->io_busy);  // not reaped yet
            const bool waits = t->waitForDedupBase(second);
            std::printf("dedup base finished, not reaped: second entry %s\n", waits ? "waits (shares the prefix)" : "does NOT wait");
            CHECK(waits);
        }
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 2);
        std::size_t refs = 0;
        for (std::uint64_t id : {a.id, b.id})
            if (const Entry* e = t->find(id); e && e->ssd)
                for (const BlkRef& r : e->ssd->tbl) refs += r.owner != 0 ? 1 : 0;
        std::printf("dedup base finished, not reaped: %zu shared block refs\n", refs);
        CHECK(refs == 7);  // the second file stores the 7 system-prompt blocks as references
    }
    fs::remove_all(dir, ec);
}

// Graceful flush order: most recently used first (a restore counts as use), so when the time bound runs
// out the oldest sessions are the ones not written. Writes run in submission order on the IO thread.
static void testFlushOrder(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 3600;
    auto t = Tier::create(ops, 256ull << 20, 36, 0);
    Config c = env.cfg(4ull << 30);
    c.ssd_delay_ms = 600000;
    c.ssd_shutdown_only = true;
    std::vector<std::string> logs;
    t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
    t->start(c, kLay, 6);
    Sess s0(ops, 24), s1(ops, 24), s2(ops, 24), s3(ops, 24);
    Sess* all[4] = {&s0, &s1, &s2, &s3};
    for (Sess* s : all) {
        s->toks = freshToks(3000, seed);
        spillSess(*t, *s, 0);
        sleepMs(5);
    }
    // creation order s0 s1 s2 s3; s1 is used last (restored), then s0 is updated in place
    CHECK(restoreMatches(*t, ops, t->find(s1.id), s1));
    sleepMs(5);
    s0.toks.push_back(99);
    spillSess(*t, s0, s0.toks.size() - 1);
    reuse(*t, s2);  // s0 (resumed) and s1 (restored) are reused already; the close flush writes only reused ones
    reuse(*t, s3);  // (the flag does not change the use order)
    t->tick();
    CHECK(t->stats.ssd_writes == 0);
    t->flushing = true;
    t->cfg.ssd_delay_ms = 0;
    CHECK(tickIdle(*t));
    CHECK(t->stats.ssd_writes == 4);
    // order of the "written to SSD" lines
    std::vector<std::uint64_t> order;
    for (const std::string& l : logs) {
        const auto p = l.find("entry ");
        if (l.find("written to SSD") == std::string::npos || p == std::string::npos) continue;
        order.push_back(std::strtoull(l.c_str() + p + 6, nullptr, 16));
    }
    const std::uint64_t want[4] = {s0.id, s1.id, s3.id, s2.id};
    CHECK(order.size() == 4);
    for (std::size_t i = 0; i < order.size() && i < 4; ++i) CHECK(order[i] == want[i]);
    t.reset();
    fs::remove_all(dir, ec);
}

// `always` mode seen-once filter: only entries whose prefix was reused (restored) or that were resumed
// (updated in place) are written; a stream of distinct one-shot prompts writes nothing. Shutdown mode's flush
// (when WHIRL is closed) writes only the reused entries too (CLOSE-REUSED; it used to write everything).
static void testSeenOnce(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 3700;
    auto mk = [&](std::uint64_t ram, std::uint64_t delay_ms, bool shutdown_only) {
        auto t = Tier::create(ops, ram, 37, 0);
        Config c = env.cfg(4ull << 30);
        c.ssd_seen_filter = true;
        c.ssd_delay_ms = delay_ms;
        c.ssd_shutdown_only = shutdown_only;
        t->start(c, kLay, 7);
        return t;
    };
    // 1. delay expiry: distinct prompts -> nothing; reused prefix and resumed session -> written
    {
        auto t = mk(256ull << 20, 200, false);
        std::vector<std::unique_ptr<Sess>> st;
        for (int i = 0; i < 6; ++i) {
            st.push_back(std::make_unique<Sess>(ops, 24));
            st.back()->toks = freshToks(3000, seed);
            spillSess(*t, *st.back(), 0);
        }
        sleepMs(300);
        CHECK(!t->tick());  // nothing to wait for or write
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
        CHECK(!t->waitOnly() && !t->nextDue().has_value());
        // one prefix is reused (restored), one session resumed (grew by appending)
        CHECK(restoreMatches(*t, ops, t->find(st[2]->id), *st[2]));
        st[4]->toks.push_back(55);
        spillSess(*t, *st[4], st[4]->toks.size() - 1);
        sleepMs(300);
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 2 && countFiles(dir) == 2);
        // restore path unchanged: evict the written one from RAM, restore from the SSD copy
        const std::uint64_t id2 = st[2]->id;
        t->removeEntry(t->find(st[0]->id));
        CHECK(t->find(id2)->ssd.has_value());
        // graceful flush in `always` mode keeps the filter
        t->flushing = true;
        t->cfg.ssd_delay_ms = 0;
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 2 && countFiles(dir) == 2);
        t.reset();
        fs::remove_all(dir, ec);
    }
    // 2. RAM pressure before the delay: a one-shot entry is dropped, a reused one is written then restored
    for (int mode = 0; mode < 2; ++mode) {  // 0: LRU victim never reused (dropped), 1: reused (written)
        auto t = mk(64ull << 20, 600000, false);  // 32 blocks; a 3000-token entry takes 13
        Sess a(ops, 24), b(ops, 24), d(ops, 24);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        d.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        const std::uint64_t id_a = a.id;
        if (mode == 1) CHECK(restoreMatches(*t, ops, t->find(a.id), a));
        sleepMs(5);
        spillSess(*t, b, 0);
        bool kept = spillSess(*t, d, 0) && d.id != 0;
        if (mode == 0) {
            CHECK(kept);
            CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
            CHECK(t->find(id_a) == nullptr && t->stats.dropped == 1);
        } else {
            // the reused LRU entry a is written (async); the unseen b is dropped for room
            CHECK(kept && t->stats.dropped == 1 && t->find(b.id) == nullptr);
            CHECK(tickIdle(*t, 20000));
            CHECK(t->stats.ssd_writes == 1 && countFiles(dir) == 1);
            Entry* ea = t->find(id_a);
            CHECK(ea != nullptr && ea->ssd.has_value());
            if (ea) CHECK(restoreMatches(*t, ops, ea, a));
            CHECK(tickIdle(*t));
        }
        t.reset();
        fs::remove_all(dir, ec);
    }
    // 3. shutdown mode: the flush writes only the reused entry (a restored), not the one-shot b (CLOSE-REUSED;
    //    before, it wrote everything in RAM, seen or not)
    {
        auto t = mk(256ull << 20, 600000, true);
        Sess a(ops, 24), b(ops, 24);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        spillSess(*t, b, 0);
        CHECK(restoreMatches(*t, ops, t->find(a.id), a));
        t->tick();
        CHECK(t->stats.ssd_writes == 0);
        t->flushing = true;
        t->cfg.ssd_delay_ms = 0;
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 1 && countFiles(dir) == 1);
        CHECK(t->find(a.id)->ssd.has_value() && !t->find(b.id)->ssd.has_value());
        t.reset();
        fs::remove_all(dir, ec);
    }
}

static void writeJunk(const fs::path& p, std::size_t n, std::uint64_t seed) {
    std::vector<std::uint8_t> v(n);
    if (n > 0) fillBytes(v.data(), n, seed);
    FILE* f = _wfopen(p.c_str(), L"wb");
    if (f) {
        if (n > 0) std::fwrite(v.data(), 1, n, f);
        std::fclose(f);
    }
}

// Startup: every kind of file in the directory is accounted for: corrupt / half-written files are deleted,
// valid files of another model count against the cap and go first, the directory never holds more than
// the cap (real bytes) and the log has one summary line.
static void testStartupCleanupCap(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 3400;
    const std::uint64_t fp_a = 41, fp_b = 42;
    std::uint64_t one_file = 0;
    std::uint64_t id_ours = 0;
    {
        // two entries of another model (fp_a)
        auto t = Tier::create(ops, 256ull << 20, 34, 0);
        t->start(env.cfg(4ull << 30), kLay, fp_a);
        Sess a(ops, 24), b(ops, 24);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        spillSess(*t, b, 0);
        CHECK(tickIdle(*t));
        one_file = t->ssd_bytes / 2;
    }
    {
        auto t = Tier::create(ops, 256ull << 20, 35, 0);
        t->start(env.cfg(4ull << 30), kLay, fp_b);
        CHECK(t->ssdEntries() == 0);  // the files of the other model are foreign: counted, not indexed
        Sess c(ops, 24);
        c.toks = freshToks(3000, seed);
        spillSess(*t, c, 0);
        CHECK(tickIdle(*t));
        id_ours = c.id;
    }
    const fs::path ours = entryFile(dir, fp_b, id_ours);
    CHECK(fs::exists(ours));
    // junk: not a cache file, zero length, a short file, a half-written one (valid = 0), a cut-short one
    writeJunk(dir / L"junk1.wkv", 100000, 1);
    writeJunk(dir / L"junk2.wkv", 0, 2);
    writeJunk(dir / L"junk3.wkv", 100, 3);
    const fs::path half = dir / L"000000000000002a-00000000000000aa.wkv";
    const fs::path cut = dir / L"000000000000002a-00000000000000bb.wkv";
    fs::copy_file(ours, half, fs::copy_options::overwrite_existing, ec);
    {
        FILE* fh = _wfopen(half.c_str(), L"r+b");
        CHECK(fh != nullptr);
        if (fh) {
            std::uint32_t zero = 0;
            std::fseek(fh, 12, SEEK_SET);  // Header::valid
            std::fwrite(&zero, 4, 1, fh);
            std::fclose(fh);
        }
    }
    fs::copy_file(ours, cut, fs::copy_options::overwrite_existing, ec);
    fs::resize_file(cut, fs::file_size(ours) / 3, ec);
    CHECK(countFiles(dir) == 8);  // 2 foreign + ours + 5 bad
    std::vector<std::string> logs;
    {
        // cap = 1.5 files: the bad files go, the other model's files go first, ours stays
        auto t = Tier::create(ops, 256ull << 20, 36, 0);
        t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
        Config c = env.cfg(one_file + one_file / 2);
        t->start(c, kLay, fp_b);
        CHECK(t->hasSsd());
        const Tier::Cleanup& cl = t->cleanup();
        CHECK(cl.bad_files == 5);
        CHECK(t->ssdEntries() == 1);
        CHECK(cl.cap_files == 2);
        CHECK(t->ssd_bytes <= c.ssd_cap);
        CHECK(diskBytes(dir) <= c.ssd_cap);
        CHECK(countFiles(dir) == 1 && fs::exists(ours));
        bool line = false;
        for (const auto& l : logs)
            if (l.find("SSD cache") != std::string::npos && l.find("removed at startup") != std::string::npos) line = true;
        CHECK(line);
    }
    {
        // a cap lower than what is there (--kv-ssd-gb lowered): evicted at start, not at the next write
        auto t = Tier::create(ops, 256ull << 20, 37, 0);
        Config c = env.cfg(one_file / 2);
        t->start(c, kLay, fp_b);
        CHECK(t->ssd_bytes <= c.ssd_cap && diskBytes(dir) <= c.ssd_cap && countFiles(dir) == 0);
    }
    fs::remove_all(dir, ec);
}

// The cap holds on every write and the LRU victim takes the files borrowing its blocks with it.
static void testCapLruDedup(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 3500;
    const std::uint64_t fp = 51;
    const auto sys = freshToks(2560, seed);
    Sess c(ops, 24), a(ops, 24), b(ops, 24);
    c.toks = freshToks(3000, seed);
    a.toks = sys;
    {
        auto x = freshToks(1000, seed);
        a.toks.insert(a.toks.end(), x.begin(), x.end());
    }
    b.toks = sys;
    {
        auto x = freshToks(700, seed);
        b.toks.insert(b.toks.end(), x.begin(), x.end());
    }
    std::uint64_t used = 0;
    {
        auto t = Tier::create(ops, 256ull << 20, 38, 0);
        t->start(env.cfg(4ull << 30), kLay, fp);
        spillSess(*t, c, 0);
        CHECK(tickIdle(*t));
        sleepMs(1100);  // distinct last-write seconds: the LRU order survives the restart
        spillSess(*t, a, 0);
        CHECK(tickIdle(*t));
        sleepMs(1100);
        spillSess(*t, b, 0);
        CHECK(tickIdle(*t));
        CHECK(t->dependentsOf(a.e).size() == 1);
        used = t->ssd_bytes;
        CHECK(diskBytes(dir) <= used);
    }
    const std::uint64_t cap = used + (3ull << 20);
    auto t = Tier::create(ops, 256ull << 20, 39, 0);
    Config cf = env.cfg(cap);
    t->start(cf, kLay, fp);
    CHECK(t->ssdEntries() == 3);
    Sess d(ops, 24), e2(ops, 24);
    d.toks = freshToks(3000, seed);
    e2.toks = freshToks(3000, seed);
    spillSess(*t, d, 0);
    CHECK(tickIdle(*t));
    CHECK(t->ssd_bytes <= cap && diskBytes(dir) <= cap);
    CHECK(t->find(c.id) == nullptr);  // the oldest went
    CHECK(t->find(a.id) != nullptr && t->find(b.id) != nullptr);
    spillSess(*t, e2, 0);
    CHECK(tickIdle(*t));
    CHECK(t->ssd_bytes <= cap && diskBytes(dir) <= cap);
    // a is evicted together with b (which borrows its blocks), never alone
    const bool a_gone = t->find(a.id) == nullptr, b_gone = t->find(b.id) == nullptr;
    CHECK(!(a_gone && !b_gone));
    const std::uint64_t cnt = countFiles(dir);
    t.reset();
    auto t2 = Tier::create(ops, 256ull << 20, 40, 0);
    t2->start(cf, kLay, fp);
    CHECK(t2->ssdEntries() == cnt);  // every file on disk is valid (no dangling reference)
    CHECK(t2->ssd_bytes <= cap && diskBytes(dir) <= cap);
    CHECK(t2->cleanup().bad_files == 0);
    t2.reset();
    fs::remove_all(dir, ec);
}

// Files unused for the expiry time are deleted (start-up and periodic); use refreshes them (also the
// owners of borrowed blocks).
static void testExpiry(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 3600;
    const std::uint64_t fp = 61;
    const std::int64_t day = 86400;
    std::int64_t offset = 0;
    auto cfg = [&](std::uint64_t cap) {
        Config c = env.cfg(cap);
        c.expire_check_ms = 0;
        c.clock_fn = [&] { return static_cast<std::int64_t>(std::time(nullptr)) + offset; };
        return c;
    };
    const auto sys = freshToks(2560, seed);
    Sess a(ops, 24), b(ops, 24), z(ops, 24);
    a.toks = sys;
    {
        auto x = freshToks(1000, seed);
        a.toks.insert(a.toks.end(), x.begin(), x.end());
    }
    b.toks = sys;
    {
        auto x = freshToks(700, seed);
        b.toks.insert(b.toks.end(), x.begin(), x.end());
    }
    z.toks = freshToks(3000, seed);
    std::uint64_t id_a = 0, id_b = 0, id_z = 0;
    {
        auto t = Tier::create(ops, 256ull << 20, 41, 0);
        t->start(cfg(4ull << 30), kLay, fp);
        spillSess(*t, a, 0);
        CHECK(tickIdle(*t));
        spillSess(*t, b, 0);
        CHECK(tickIdle(*t));
        spillSess(*t, z, 0);
        CHECK(tickIdle(*t));
        id_a = a.id;
        id_b = b.id;
        id_z = z.id;
        CHECK(t->dependentsOf(a.e).size() == 1);
    }
    {
        // 6 days later: all kept; b is used (restored): b and its owner a are fresh at day 6
        offset = 6 * day;
        auto t = Tier::create(ops, 256ull << 20, 42, 0);
        t->start(cfg(4ull << 30), kLay, fp);
        CHECK(t->ssdEntries() == 3 && t->cleanup().expired_files == 0);
        Entry* eb = t->find(id_b);
        CHECK(eb != nullptr);
        if (eb) CHECK(restoreMatches(*t, ops, eb, b));
        CHECK(tickIdle(*t));
    }
    {
        // day 8: z (unused for 8 days) expires at start; a and b were used at day 6
        offset = 8 * day;
        std::vector<std::string> logs;
        auto t = Tier::create(ops, 256ull << 20, 43, 0);
        t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
        t->start(cfg(4ull << 30), kLay, fp);
        CHECK(t->find(id_z) == nullptr);
        CHECK(t->find(id_a) != nullptr && t->find(id_b) != nullptr);
        CHECK(t->cleanup().expired_files == 1);
        CHECK(countFiles(dir) == 2);
        bool line = false;
        for (const auto& l : logs)
            if (l.find("1 expired") != std::string::npos) line = true;
        CHECK(line);
        // periodic: day 14 -> both expire on a tick
        offset = 14 * day;
        t->tick();
        CHECK(t->ssdEntries() == 0 && countFiles(dir) == 0 && t->ssd_bytes == 0);
    }
    {
        // files of another model expire too; expire_s = 0 turns expiry off
        fs::remove_all(dir, ec);
        offset = 0;
        {
            auto t = Tier::create(ops, 256ull << 20, 44, 0);
            t->start(cfg(4ull << 30), kLay, fp);
            Sess q(ops, 24);
            q.toks = freshToks(3000, seed);
            spillSess(*t, q, 0);
            CHECK(tickIdle(*t));
        }
        offset = 9 * day;
        {
            Config c = cfg(4ull << 30);
            c.expire_s = 0;
            auto t = Tier::create(ops, 256ull << 20, 45, 0);
            t->start(c, kLay, fp + 1);
            CHECK(countFiles(dir) == 1 && t->ssd_bytes > 0 && t->cleanup().expired_files == 0);
        }
        auto t = Tier::create(ops, 256ull << 20, 46, 0);
        t->start(cfg(4ull << 30), kLay, fp + 1);
        CHECK(countFiles(dir) == 0 && t->cleanup().expired_files == 1);
    }
    fs::remove_all(dir, ec);
}


// FIX-RC4-A P2-2: an SSD entry used from the RAM tier, unchanged (a shared system prompt / tool prefix),
// is never rewritten; its use must still reach the file (graceful shutdown flush) or it expires at a
// restart although it was used two days before. Both SSD modes.
static void testRamUsePersisted(MockDeviceOps& ops, const fs::path& dir) {
    for (int mode = 0; mode < 2; ++mode) {  // 0: always, 1: shutdown
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        std::uint64_t seed = 9200 + static_cast<std::uint64_t>(mode) * 10;
        const std::int64_t day = 86400;
        std::int64_t offset = 0;
        auto cfg = [&]() {
            Config c = env.cfg(4ull << 30);
            c.ssd_shutdown_only = mode == 1;
            c.clock_fn = [&] { return static_cast<std::int64_t>(std::time(nullptr)) + offset; };
            return c;
        };
        Sess a(ops, 24);
        a.toks = freshToks(3000, seed);
        std::uint64_t id_a = 0;
        {
            auto t = Tier::create(ops, 256ull << 20, 92, 0);
            t->start(cfg(), kLay, 77);
            spillSess(*t, a, 0);
            id_a = a.id;
            reuse(*t, a);  // a shared prefix used by others (shutdown mode's close flush writes only reused entries)
            t->flushing = true;
            CHECK(tickIdle(*t));
            CHECK(countFiles(dir) == 1);
        }
        {
            offset = 6 * day;  // restart; used from SSD at day 6, then from RAM at day 12
            auto t = Tier::create(ops, 256ull << 20, 93, 0);
            t->start(cfg(), kLay, 77);
            CHECK(restoreMatches(*t, ops, t->find(id_a), a));
            CHECK(tickIdle(*t));
            offset = 12 * day;
            Entry* ea = t->find(id_a);
            CHECK(ea != nullptr && ea->in_ram);
            CHECK(restoreMatches(*t, ops, ea, a));  // RAM restore at day 12
            CHECK(tickIdle(*t));
            t->flushing = true;  // graceful shutdown (Engine::flushTierOnStop)
            t->cfg.ssd_delay_ms = 0;
            CHECK(tickIdle(*t));
        }
        offset = 14 * day;  // last used 2 days ago
        auto t = Tier::create(ops, 256ull << 20, 94, 0);
        t->start(cfg(), kLay, 77);
        CHECK(t->find(id_a) != nullptr);  // expired before FIX-RC4-A (only the day-6 SSD use was on file)
        CHECK(t->cleanup().expired_files == 0);
        t.reset();
        fs::remove_all(dir, ec);
    }
}

// FIX-RC4-A P3-2: shutdown mode writes nothing to the SSD while running -- not the header / table rewrite
// of a trimmed (superseded) entry, not the "last used" time of an entry restored from SSD; both are written
// at the shutdown flush.
static void testShutdownNoWrites(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9300;
    const std::uint64_t fp = 78;
    std::int64_t offset = 0;
    auto cfg = [&]() {
        Config c = env.cfg(4ull << 30);
        c.ssd_shutdown_only = true;
        c.clock_fn = [&] { return static_cast<std::int64_t>(std::time(nullptr)) + offset; };
        return c;
    };
    Sess a(ops, 24);
    a.toks = freshToks(3000, seed);
    std::uint64_t id_a = 0;
    {
        auto t = Tier::create(ops, 256ull << 20, 95, 0);
        t->start(cfg(), kLay, fp);
        spillSess(*t, a, 0);
        id_a = a.id;
        auto more = freshToks(500, seed);
        a.toks.insert(a.toks.end(), more.begin(), more.end());
        spillSess(*t, a, 3000);  // in place: checkpoints at 3000 and 3500
        t->flushing = true;
        CHECK(tickIdle(*t));
        t->flushing = false;
        CHECK(t->stats.ssd_writes == 1);
        Entry* ea = t->find(id_a);
        CHECK(ea != nullptr && ea->ssd.has_value() && !ea->hasDirty());
        if (!ea) return;
        const fs::path f = entryFile(dir, fp, id_a);
        const auto sz0 = fs::file_size(f, ec);
        const auto mt0 = fs::last_write_time(f, ec);
        CHECK(t->trimTo(ea, 3200) > 0);  // superseded: cut to the checkpoint at 3000
        CHECK(ea->meta_dirty);
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 1);  // nothing written while running (was a meta-only rewrite)
        CHECK(fs::file_size(f, ec) == sz0 && fs::last_write_time(f, ec) == mt0);
        t->flushing = true;
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 2 && !ea->meta_dirty);  // the flush does it
        CHECK(fs::file_size(f, ec) < sz0);
    }
    {
        // restart, restore from SSD while running: the file's time is not touched until the flush
        offset = 2 * 86400;
        auto t = Tier::create(ops, 256ull << 20, 96, 0);
        t->start(cfg(), kLay, fp);
        Entry* ea = t->find(id_a);
        CHECK(ea != nullptr);
        if (!ea) return;
        const fs::path f = entryFile(dir, fp, id_a);
        const auto mt0 = fs::last_write_time(f, ec);
        a.toks.resize(3000);
        CHECK(restoreMatches(*t, ops, ea, a));
        CHECK(tickIdle(*t));
        CHECK(fs::last_write_time(f, ec) == mt0);
        t->flushing = true;
        CHECK(tickIdle(*t));
        CHECK(fs::last_write_time(f, ec) != mt0);  // persisted at the flush
    }
    fs::remove_all(dir, ec);
}

// TIER-EVICT (owner decision b): `shutdown` mode writes nothing while running, except a reused entry that RAM
// pressure has to evict: it is written (async; the spill that needed the room is retried) instead of dropped,
// so it survives a restart (INT-RC4F: 8060S, 8 GiB RAM tier, precise f16 KV -- the reused system checkpoint
// was pushed out of RAM and lost). One-shot prompts are still dropped, below min_tokens is dropped, a file
// that cannot fit the cap is skipped (one log line, dropped), and without RAM pressure nothing is written.
// Before the fix modes 0 and 1 dropped a (no file; the restart found nothing).
static void testShutdownEvictReused(MockDeviceOps& ops, const fs::path& dir) {
    // 0: reused LRU victim -> written, survives a restart; the one-shot b is dropped (not written)
    // 1: seen filter off: still only reused entries (b dropped, a written)
    // 2: cap smaller than a's file -> skipped with one log line, dropped, spill not blocked
    // 3: a reused but below min_tokens -> dropped
    // 4: no RAM pressure -> nothing written while running (the flush is unchanged)
    for (int mode = 0; mode < 5; ++mode) {
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        std::uint64_t seed = 9700 + static_cast<std::uint64_t>(mode) * 10;
        const std::uint64_t fp = 97;
        auto mkcfg = [&] {
            Config c = env.cfg(mode == 2 ? (8ull << 20) : (4ull << 30));
            c.ssd_seen_filter = mode != 1;
            c.ssd_delay_ms = 0;  // shutdown mode ignores the delay anyway
            c.ssd_shutdown_only = true;
            c.min_tokens = mode == 3 ? 5000 : 2048;
            return c;
        };
        std::vector<std::string> logs;
        Sess a(ops, 24), b(ops, 24), d(ops, 24);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        d.toks = freshToks(3000, seed);
        std::uint64_t id_a = 0;
        {
            // 32 blocks; a 3000-token entry takes 13 (mode 4: room for all three)
            auto t = Tier::create(ops, mode == 4 ? (256ull << 20) : (64ull << 20), 97, 0);
            t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
            t->start(mkcfg(), kLay, fp);
            spillSess(*t, a, 0);
            id_a = a.id;
            CHECK(restoreMatches(*t, ops, t->find(id_a), a));  // a is reused (its prefix restored once)
            CHECK(t->find(id_a) != nullptr && t->find(id_a)->reused);
            sleepMs(5);
            spillSess(*t, b, 0);  // one-shot
            CHECK(tickIdle(*t));
            CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);  // nothing while running
            // the third needs room (modes 0-3): the LRU entry a goes
            bool kept = spillSess(*t, d, 0) && d.id != 0;
            if (!kept && t->roomPending()) {  // a's write is running: retried once it frees a's blocks
                CHECK(tickIdle(*t, 20000));
                kept = spillSess(*t, d, 0) && d.id != 0;
            }
            CHECK(kept);
            CHECK(tickIdle(*t, 20000));
            Entry* ea = t->find(id_a);
            if (mode <= 1) {
                CHECK(t->stats.evict_writes == 1 && t->stats.ssd_writes == 1 && countFiles(dir) == 1);
                CHECK(ea != nullptr && ea->ssd.has_value());
                CHECK(t->find(b.id) == nullptr);  // the one-shot entry was dropped, never written
                std::size_t lines = 0;
                for (const auto& l : logs) lines += l.find("evicted from RAM: written to SSD") != std::string::npos ? 1 : 0;
                CHECK(lines == 1);
            } else if (mode == 4) {
                CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0 && t->stats.ram_evictions == 0);
                CHECK(ea != nullptr && ea->in_ram);
            } else {
                CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0 && t->stats.evict_writes == 0);
                CHECK(ea == nullptr);  // dropped as before
                if (mode == 2) {
                    std::size_t nofit = 0;
                    for (const auto& l : logs) nofit += l.find("no room under the cap") != std::string::npos ? 1 : 0;
                    CHECK(nofit == 1);
                }
            }
            // crash / kill: no flush (shutdown mode loses what is only in RAM)
        }
        if (mode <= 1) {
            // restart: a comes back from the SSD and restores bit-exactly
            auto t = Tier::create(ops, 64ull << 20, 98, 0);
            t->start(mkcfg(), kLay, fp);
            CHECK(t->ssdEntries() == 1);
            Entry* ea = t->find(id_a);
            CHECK(ea != nullptr && !ea->in_ram);
            if (ea) CHECK(restoreMatches(*t, ops, ea, a));
            CHECK(tickIdle(*t));
            CHECK(t->stats.ssd_writes == 0);
        }
        fs::remove_all(dir, ec);
    }
}

// TIER-IDLE: the idle-write policy (tier/idle_flush.h) with a fake clock. Fires once per idle period after
// after_ms of continuous idleness, never again while the server stays idle, again after a request ends the
// period; picks entries by the shutdown-mode eviction rule (reused, not demoted, >= min_tokens, not current).
static void testIdleFlushPolicy() {
    // when
    CHECK(idleFlushEnabled(true, true, 300000));
    CHECK(!idleFlushEnabled(false, true, 300000));  // always mode: entries are written after their delay
    CHECK(!idleFlushEnabled(true, false, 300000));  // no SSD tier
    CHECK(!idleFlushEnabled(true, true, 0));        // test override 0: off
    CHECK(idle_flush_ms_default == 300000);
    IdleFlushClock c;
    CHECK(!c.deadline(1000).has_value());  // not idle yet
    CHECK(!c.poll(10000, 1000));           // idle from t=10 s
    CHECK(c.idle() && !c.fired());
    CHECK(c.deadline(1000).has_value() && *c.deadline(1000) == 11000);
    CHECK(!c.poll(10999, 1000));
    CHECK(c.poll(11000, 1000));  // 1 s idle: fires
    CHECK(c.fired() && !c.deadline(1000).has_value());
    CHECK(!c.poll(11001, 1000));   // still idle: no second pass
    CHECK(!c.poll(500000, 1000));  // however long it stays idle
    c.activity();                  // a request: the period ends
    CHECK(!c.idle() && !c.fired());
    CHECK(!c.poll(600000, 1000));  // a new idle period starts now
    CHECK(!c.poll(600999, 1000));
    CHECK(c.poll(601000, 1000));  // and fires once again
    CHECK(!c.poll(700000, 1000));
    c.activity();
    CHECK(!c.poll(800000, 1000));
    c.activity();  // activity before the delay ran out: restarts the period
    CHECK(!c.poll(800900, 1000));
    CHECK(!c.poll(801000, 1000));  // only 100 ms into the new period
    CHECK(c.poll(801900, 1000));
    IdleFlushClock off;
    CHECK(!off.poll(0, 0) && !off.poll(1000000, 0) && !off.deadline(0).has_value());
    // what
    EvictFacts b;  // reused, never written, 4096 tokens, SSD on
    b.ssd_on = true;
    b.shutdown_only = true;
    b.reused = true;
    b.n_tok = 4096;
    b.min_tokens = 2048;
    CHECK(idleWriteWanted(b));
    EvictFacts f = b;
    f.reused = false;  // one-shot prompt: never, whatever the seen filter says
    CHECK(!idleWriteWanted(f));
    f.seen_filter = false;
    CHECK(!idleWriteWanted(f));
    f.shutdown_only = false;  // the rule is the shutdown one even if the caller passed always-mode facts
    CHECK(!idleWriteWanted(f));
    f = b;
    f.demoted = true;
    CHECK(!idleWriteWanted(f));
    f = b;
    f.n_tok = 2047;
    CHECK(!idleWriteWanted(f));
    f = b;
    f.ssd_on = false;
    CHECK(!idleWriteWanted(f));
    f = b;
    f.on_ssd = true;  // current on SSD: never rewritten
    CHECK(!idleWriteWanted(f));
    f.dirty = true;  // changed since its write (resumed): written again
    CHECK(idleWriteWanted(f));
}

// CLOSE-REUSED: the close-flush rule (evict_policy.h closeWriteWanted) is the shutdown-mode eviction rule,
// except that a demoted reused entry is still written (the close flush is its last chance).
static void testClosePolicy() {
    EvictFacts b;  // a reused, never-written 4096-token entry, SSD tier on, shutdown mode
    b.ssd_on = true;
    b.shutdown_only = true;
    b.seen_filter = true;
    b.reused = true;
    b.n_tok = 4096;
    b.min_tokens = 2048;
    CHECK(closeWriteWanted(b));
    EvictFacts f = b;
    f.reused = false;  // one-shot prompt: never, whatever the seen filter says
    CHECK(!closeWriteWanted(f));
    f.seen_filter = false;
    CHECK(!closeWriteWanted(f));
    f.shutdown_only = false;  // the rule is the shutdown one even if the caller passed always-mode facts
    CHECK(!closeWriteWanted(f));
    f = b;
    f.demoted = true;  // demoted but reused: written at close (not on eviction / idle)
    CHECK(closeWriteWanted(f) && !evictWriteWanted(f) && !idleWriteWanted(f));
    f.reused = false;  // demoted one-shot: never
    CHECK(!closeWriteWanted(f));
    f = b;
    f.ssd_on = false;
    CHECK(!closeWriteWanted(f));
    f = b;
    f.n_tok = 2047;
    CHECK(!closeWriteWanted(f));
    f = b;
    f.on_ssd = true;  // current on SSD: not rewritten; dirty since: rewritten
    CHECK(!closeWriteWanted(f));
    f.dirty = true;
    CHECK(closeWriteWanted(f));
}

// CLOSE-REUSED in the Tier: shutdown mode, the flush when WHIRL is closed (flushing, delay 0 like
// Engine::flushTierOnStop) writes the reused entries (restored, resumed, and a demoted reused one) and never
// the one-shot prompt; a restart restores the reused ones bit-exactly and has no trace of the one-shot one.
// Always mode keeps its seen-once filter at the flush (the one-shot one is written with the filter off).
static void testCloseReusedTier(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9700;
    const std::uint64_t fp = 97;
    std::vector<std::string> logs;
    Sess a(ops, 24), b(ops, 24), d(ops, 24), r(ops, 24);
    a.toks = freshToks(3000, seed);
    b.toks = freshToks(3000, seed);
    d.toks = freshToks(3000, seed);
    r.toks = freshToks(3000, seed);
    std::uint64_t id_a = 0, id_b = 0, id_d = 0, id_r = 0;
    auto mkcfg = [&] {
        Config c = env.cfg(4ull << 30);
        c.ssd_seen_filter = true;
        c.ssd_shutdown_only = true;
        c.ssd_delay_ms = 600000;
        return c;
    };
    {
        auto t = Tier::create(ops, 256ull << 20, 97, 0);
        t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
        t->start(mkcfg(), kLay, fp);
        spillSess(*t, a, 0);
        id_a = a.id;
        spillSess(*t, b, 0);
        id_b = b.id;
        spillSess(*t, d, 0);
        id_d = d.id;
        spillSess(*t, r, 0);
        id_r = r.id;
        CHECK(restoreMatches(*t, ops, t->find(id_a), a));  // a: restored (reused)
        auto more = freshToks(500, seed);                  // r: resumed in place (reused)
        r.toks.insert(r.toks.end(), more.begin(), more.end());
        spillSess(*t, r, 3000);
        CHECK(t->find(id_r) != nullptr && t->find(id_r)->reused);
        CHECK(restoreMatches(*t, ops, t->find(id_d), d));  // d: reused, then demoted
        t->demote(t->find(id_d));
        CHECK(!t->find(id_b)->reused);  // b: one-shot
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);  // nothing while running
        t->flushing = true;
        t->cfg.ssd_delay_ms = 0;
        CHECK(tickIdle(*t));
        t->flushing = false;
        CHECK(t->stats.ssd_writes == 3 && countFiles(dir) == 3);
        CHECK(t->find(id_a)->ssd.has_value() && t->find(id_r)->ssd.has_value() && t->find(id_d)->ssd.has_value());
        CHECK(!t->find(id_b)->ssd.has_value());  // the one-shot prompt is not written when WHIRL is closed
    }
    {
        // restart: the reused entries restore bit-exactly; the one-shot one is gone
        auto t = Tier::create(ops, 256ull << 20, 98, 0);
        t->start(mkcfg(), kLay, fp);
        CHECK(t->ssdEntries() == 3);
        CHECK(t->find(id_b) == nullptr);
        for (auto [id, s] : {std::pair<std::uint64_t, const Sess*>{id_a, &a}, {id_r, &r}, {id_d, &d}}) {
            Entry* e = t->find(id);
            CHECK(e != nullptr && !e->in_ram);
            if (e) CHECK(restoreMatches(*t, ops, e, *s));
        }
        CHECK(tickIdle(*t));
    }
    fs::remove_all(dir, ec);
    for (int filter = 1; filter >= 0; --filter) {
        // always mode: unchanged, the flush applies the seen-once filter
        auto t = Tier::create(ops, 256ull << 20, 99 + filter, 0);
        Config c = env.cfg(4ull << 30);
        c.ssd_seen_filter = filter == 1;
        c.ssd_shutdown_only = false;
        c.ssd_delay_ms = 600000;
        t->start(c, kLay, fp);
        Sess x(ops, 24);
        x.toks = freshToks(3000, seed);
        spillSess(*t, x, 0);
        CHECK(tickIdle(*t));
        t->flushing = true;
        t->cfg.ssd_delay_ms = 0;
        CHECK(tickIdle(*t));
        t->flushing = false;
        CHECK(t->stats.ssd_writes == (filter == 1 ? 0u : 1u));
        t.reset();
        fs::remove_all(dir, ec);
    }
}

// TIER-IDLE in the Tier: shutdown mode writes nothing while running; Tier::beginIdleWrite (the engine's idle
// clock fired) writes the reused entries, one write at a time, never the one-shot one, then ends the pass by
// itself; staying idle writes nothing more; a new pass rewrites only an entry that changed since (dirty);
// endIdleWrite (a request) stops starting writes after the one running; always mode ignores it.
static void testIdleWriteTier(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9900;
    const std::uint64_t fp = 99;
    auto mkcfg = [&](bool shutdown) {
        Config c = env.cfg(4ull << 30);
        c.ssd_seen_filter = true;
        c.ssd_shutdown_only = shutdown;
        c.ssd_delay_ms = shutdown ? 0 : 3600000;
        return c;
    };
    std::vector<std::string> logs;
    auto countLog = [&](const char* s) {
        std::size_t n = 0;
        for (const auto& l : logs) n += l.find(s) != std::string::npos ? 1 : 0;
        return n;
    };
    Sess a(ops, 24), b(ops, 24), d(ops, 24);
    a.toks = freshToks(3000, seed);
    b.toks = freshToks(3000, seed);
    d.toks = freshToks(3000, seed);
    std::uint64_t id_a = 0, id_b = 0, id_d = 0;
    {
        auto t = Tier::create(ops, 256ull << 20, 99, 0);
        t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
        t->start(mkcfg(true), kLay, fp);
        spillSess(*t, a, 0);
        id_a = a.id;
        spillSess(*t, b, 0);
        id_b = b.id;
        spillSess(*t, d, 0);
        id_d = d.id;
        CHECK(restoreMatches(*t, ops, t->find(id_a), a));  // a, d reused; b one-shot
        CHECK(restoreMatches(*t, ops, t->find(id_d), d));
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);  // shutdown mode: nothing while running
        // the idle pass: one write at a time
        CHECK(t->beginIdleWrite() == 2);
        CHECK(t->idleWriting() && t->stats.idle_passes == 1);
        t->tick();
        std::size_t busy = 0;
        for (const Entry* e : t->entries) busy += e->io_busy ? 1 : 0;
        CHECK(busy == 1);
        CHECK(tickIdle(*t, 20000));
        CHECK(!t->idleWriting());  // the pass ended by itself
        CHECK(t->stats.idle_writes == 2 && t->stats.ssd_writes == 2 && countFiles(dir) == 2);
        CHECK(t->find(id_a)->ssd.has_value() && t->find(id_d)->ssd.has_value());
        CHECK(!t->find(id_b)->ssd.has_value());  // the one-shot prompt is never written
        CHECK(countLog("written to SSD after the server sat idle") == 2 && countLog("idle write done") == 1);
        // still idle: nothing more
        for (int i = 0; i < 20; ++i) t->tick();
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_writes == 2);
        // a later idle period: everything current, nothing rewritten (no pass started)
        CHECK(t->beginIdleWrite() == 0);
        CHECK(tickIdle(*t));
        CHECK(!t->idleWriting() && t->stats.ssd_writes == 2 && t->stats.idle_passes == 1);
        // a resumed (changed) entry: written again by the next pass; the clean one is not
        auto more = freshToks(500, seed);
        a.toks.insert(a.toks.end(), more.begin(), more.end());
        spillSess(*t, a, 3000);
        CHECK(t->find(id_a)->hasDirty());
        CHECK(t->beginIdleWrite() == 1);
        CHECK(tickIdle(*t));
        CHECK(t->stats.idle_writes == 3 && t->stats.ssd_writes == 3 && !t->find(id_a)->hasDirty());
        CHECK(!t->find(id_b)->ssd.has_value());
        // interrupted: both a and d changed; a request arrives right after the first write started
        more = freshToks(500, seed);
        a.toks.insert(a.toks.end(), more.begin(), more.end());
        spillSess(*t, a, 3500);
        more = freshToks(500, seed);
        d.toks.insert(d.toks.end(), more.begin(), more.end());
        spillSess(*t, d, 3000);
        CHECK(t->beginIdleWrite() == 2);
        t->tick();
        t->endIdleWrite();
        CHECK(!t->idleWriting());
        CHECK(tickIdle(*t));
        CHECK(t->stats.idle_writes == 4 && t->stats.ssd_writes == 4);  // the running one finished, no other started
        CHECK(countLog("idle write stopped by a request") == 1);
        CHECK(t->find(id_a)->hasDirty() != t->find(id_d)->hasDirty());  // exactly one of them is still dirty
        // the next idle period writes the other
        t->beginIdleWrite();
        CHECK(tickIdle(*t));
        CHECK(t->stats.idle_writes == 5 && !t->find(id_a)->hasDirty() && !t->find(id_d)->hasDirty());
        CHECK(!t->find(id_b)->ssd.has_value() && countFiles(dir) == 2);
    }
    {
        // restart: the idle-written entries restore bit-exactly (same files and header protocol as any write)
        auto t = Tier::create(ops, 256ull << 20, 100, 0);
        t->start(mkcfg(true), kLay, fp);
        CHECK(t->ssdEntries() == 2);
        Entry* ea = t->find(id_a);
        CHECK(ea != nullptr && !ea->in_ram);
        if (ea) CHECK(restoreMatches(*t, ops, ea, a));
        CHECK(t->find(id_b) == nullptr);
        CHECK(tickIdle(*t));
    }
    fs::remove_all(dir, ec);
    {
        // always mode: the idle pass does not exist (beginIdleWrite is ignored)
        auto t = Tier::create(ops, 256ull << 20, 101, 0);
        t->start(mkcfg(false), kLay, fp);
        Sess x(ops, 24);
        x.toks = freshToks(3000, seed);
        spillSess(*t, x, 0);
        CHECK(restoreMatches(*t, ops, t->find(x.id), x));
        CHECK(t->beginIdleWrite() == 0);
        CHECK(!t->idleWriting());
        CHECK(tickIdle(*t));
        CHECK(t->stats.idle_writes == 0 && t->stats.ssd_writes == 0);  // its delay (1 h) has not run out
    }
    fs::remove_all(dir, ec);
}

// FIX-RC4-A P3-3: a cache file that cannot be deleted (another program holds it) stays counted against the
// cap -- at an LRU eviction, for a corrupt file at startup -- and a file that cannot be opened at startup is
// kept and counted (aged by its last-write time), not reported as expired and dropped from the accounting.
static void testDeleteFails(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9400;
    const std::uint64_t fp = 79;
    Sess a(ops, 24), b(ops, 24), c(ops, 24);
    a.toks = freshToks(3000, seed);
    b.toks = freshToks(3000, seed);
    c.toks = freshToks(3000, seed);
    const auto holdOpen = [](const fs::path& p, DWORD share) {
        return CreateFileW(p.c_str(), GENERIC_READ, share, nullptr, OPEN_EXISTING, 0, nullptr);
    };
    std::uint64_t cap = 0;
    {
        auto t = Tier::create(ops, 256ull << 20, 97, 0);
        t->start(env.cfg(4ull << 30), kLay, fp);
        spillSess(*t, a, 0);
        CHECK(tickIdle(*t));
        sleepMs(5);
        spillSess(*t, b, 0);
        CHECK(tickIdle(*t));
        cap = t->ssd_bytes + t->ssd_bytes / 4;  // 2.5 files
    }
    {
        // restart: a and b are SSD-only
        auto t = Tier::create(ops, 256ull << 20, 102, 0);
        t->start(env.cfg(cap), kLay, fp);
        CHECK(t->ssdEntries() == 2);
        // a (LRU) must go for c; its file is held by another program
        HANDLE h = holdOpen(entryFile(dir, fp, a.id), FILE_SHARE_READ);
        CHECK(h != INVALID_HANDLE_VALUE);
        sleepMs(5);
        spillSess(*t, c, 0);
        CHECK(tickIdle(*t));
        CHECK(fs::exists(entryFile(dir, fp, a.id)));
        CHECK(t->ssd_bytes >= diskBytes(dir));  // the file still on disk is still counted
        CHECK(diskBytes(dir) <= cap);           // so the cap holds (b went instead)
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    }
    {
        // startup: a file nobody can open, and a corrupt file that cannot be deleted
        const fs::path fc = entryFile(dir, fp, c.id);
        CHECK(fs::exists(fc));
        HANDLE h = holdOpen(fc, 0);
        CHECK(h != INVALID_HANDLE_VALUE);
        const fs::path junk = dir / L"junk.wkv";
        writeJunk(junk, 100000, 7);
        HANDLE hj = holdOpen(junk, FILE_SHARE_READ);
        auto t = Tier::create(ops, 256ull << 20, 98, 0);
        t->start(env.cfg(cap), kLay, fp);
        CHECK(t->cleanup().expired_files == 0);  // it was used minutes ago (was: last_used 0 -> "expired")
        CHECK(t->cleanup().unreadable_files == 1);
        CHECK(fs::exists(fc) && fs::exists(junk));
        CHECK(t->ssd_bytes >= diskBytes(dir));
        if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
        if (hj != INVALID_HANDLE_VALUE) CloseHandle(hj);
    }
    fs::remove_all(dir, ec);
}

// FIX-RC4-A P3-6: an asynchronous copy error (reported by the event query, not thrown by copyAsync) fails
// the restore (cold prefill); it used to read as "complete and valid".
static void testRestoreAsyncError(MockDeviceOps& ops) {
    auto t = Tier::create(ops, 64ull << 20, 99, 0);
    t->start(Config{}, kLay, 1);
    DevSlot src(ops, 2, 7);
    src.fill(55);
    Entry* e = spill(*t, src, makeTokens(1700, 41));
    CHECK(e != nullptr);
    if (!e) return;
    CHECK(tickIdle(*t));
    DevSlot dst(ops, 2, 7);
    std::size_t nm = 0;
    const auto sp = dst.spans(e, 1, &nm);
    ops.event_error = true;
    Restore* r = t->beginRestore(e, sp, nm);
    CHECK(r != nullptr);
    if (r) {
        bool done = false;
        for (int i = 0; i < 100 && !done; ++i) done = t->pumpRestoreNoThrow(r);
        CHECK(done && r->failed);
        t->endRestore(r);
    }
    ops.event_error = false;
    CHECK(e->pin == 0);
    DevSlot ok(ops, 2, 7);
    const auto sp2 = ok.spans(e, 1, &nm);
    Restore* r2 = t->beginRestore(e, sp2, nm);
    CHECK(r2 != nullptr);
    if (r2) {
        CHECK(runRestore(*t, r2));
        t->endRestore(r2);
        CHECK(ok.ckEq(src, 0) && ok.ckEq(src, 1) && ok.kvEq(src));
    }
}

// FIX-RC4-A P3-5: an idle engine sleeps until Tier::nextWake(); with nothing waiting for a write it must
// still wake for the periodic expiry check (nextDue() alone is empty then: it slept forever).
// P3-1: a spill larger than the whole RAM tier does not leave roomPending() set (busy loop).
static void testIdleWakeRoomPending(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    {
        auto t = Tier::create(ops, 64ull << 20, 100, 0);
        Config c = env.cfg(4ull << 30);
        c.expire_check_ms = 3600ull * 1000;
        t->start(c, kLay, 80);
        CHECK(tickIdle(*t));
        CHECK(!t->nextDue().has_value());
        const auto w = t->nextWake();
        CHECK(w.has_value());
        if (w) {
            const auto now = std::chrono::steady_clock::now();
            CHECK(*w > now && *w <= now + std::chrono::hours(1) + std::chrono::seconds(5));
        }
    }
    {
        auto t = Tier::create(ops, 64ull << 20, 101, 0);  // 32 blocks
        Config c = env.cfg(4ull << 30);
        c.ssd_delay_ms = 600000;
        c.expire_s = 0;
        t->start(c, kLay, 81);
        CHECK(!t->nextWake().has_value());  // expiry off and nothing waiting: nothing timed
        std::uint64_t seed = 9500;
        Sess a(ops, 24), b(ops, 24), d(ops, 24);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        d.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        sleepMs(5);
        spillSess(*t, b, 0);
        t->tick();
        spillSess(*t, d, 0);
        CHECK(t->roomPending());  // a's eviction write will free its blocks
        Entry* eb = t->find(b.id);
        CHECK(eb != nullptr);
        if (eb) {
            CHECK(!t->ensureBlocks(eb, 0, 100));  // more than the whole tier
            CHECK(!t->roomPending());
        }
        CHECK(tickIdle(*t));
    }
    fs::remove_all(dir, ec);
}

// FIX-RC4-A P3-7 (FLAKE-1 rest): a restore taking over a running prefetch removed the prefetch job from the
// job list with a swapRemove, so the writes queued after it were reaped -- and logged -- out of order.
static void testTakeoverKeepsJobOrder(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9600;
    const std::uint64_t fp = 82;
    Sess p(ops, 24);
    p.toks = freshToks(3000, seed);
    {
        auto t = Tier::create(ops, 256ull << 20, 103, 0);
        t->start(env.cfg(4ull << 30), kLay, fp);
        spillSess(*t, p, 0);
        CHECK(tickIdle(*t));
    }
    auto t = Tier::create(ops, 256ull << 20, 104, 0);
    std::vector<std::string> logs;
    t->log_fn = [&](std::string_view s) { logs.emplace_back(s); };
    t->start(env.cfg(4ull << 30), kLay, fp);
    Entry* ep = t->find(p.id);
    CHECK(ep != nullptr && !ep->in_ram);
    if (!ep) return;
    ops.closeGate();  // the prefetch read waits (the IO thread holds it)
    CHECK(t->prefetch(ep));
    Sess a(ops, 24), b(ops, 24), c(ops, 24);
    Sess* w[3] = {&a, &b, &c};
    for (Sess* s : w) {
        s->toks = freshToks(3000, seed);
        spillSess(*t, *s, 0);
        sleepMs(2);
    }
    t->tick();  // writes of a, b, c queued after the prefetch: jobs [p, a, b, c]
    DevSlot dst(ops, 2, 24);
    std::vector<Span> sp;
    sp.push_back(Span{dst.ck[0], 0, kLay.ck_bytes, 0});
    sp.push_back(Span{dst.kv, ep->kvOff(kLay), static_cast<std::uint64_t>(ep->pages) * kLay.page_bytes, 0xff});
    Restore* r = t->beginRestore(ep, sp, sp.size());  // takes the prefetch over
    CHECK(r != nullptr);
    ops.openGate();
    if (r) {
        CHECK(runRestore(*t, r));
        t->endRestore(r);
    }
    // let every write finish before reaping, so the reap order is the job-list order
    for (int i = 0; i < 2000 && countFiles(dir) < 4; ++i) sleepMs(5);
    sleepMs(1500);
    CHECK(tickIdle(*t));
    std::vector<std::uint64_t> order;
    for (const std::string& l : logs) {
        const auto q = l.find("entry ");
        if (l.find("written to SSD") == std::string::npos || q == std::string::npos) continue;
        order.push_back(std::strtoull(l.c_str() + q + 6, nullptr, 16));
    }
    CHECK(order.size() == 3);
    if (order.size() == 3) CHECK(order[0] == a.id && order[1] == b.id && order[2] == c.id);  // was c, a, b
    t.reset();
    fs::remove_all(dir, ec);
}

// FIX-RC4-A weak tests: (1) a reference cycle already on disk (an older build could write one) is removed
// at startup by validateRefs; (2) the owner of a borrowed block truncated -- at startup (the owner goes,
// the borrower with it) and under a running server (the borrower's restore fails on the short borrowed
// block: need = block_bytes).
static Header readHeader(const fs::path& f) {
    Header h{};
    const auto v = readFile(f);
    if (v.size() >= sizeof h) std::memcpy(&h, v.data(), sizeof h);
    return h;
}

static void testDedupStartupCycleOwnerTrunc(MockDeviceOps& ops, const fs::path& dir) {
    const std::uint64_t fp = 83;
    for (int mode = 0; mode < 3; ++mode) {  // 0: cycle on disk, 1: owner cut at startup, 2: owner cut while running
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        std::uint64_t seed = 9700 + static_cast<std::uint64_t>(mode) * 10;
        const auto sys = freshToks(2560, seed);
        Sess a(ops, 24), b(ops, 24);
        a.toks = sys;
        b.toks = sys;
        {
            auto x = freshToks(1500, seed);
            a.toks.insert(a.toks.end(), x.begin(), x.end());
            auto y = freshToks(1500, seed);
            b.toks.insert(b.toks.end(), y.begin(), y.end());
        }
        std::uint32_t borrowed_lb = 0;
        {
            auto t = Tier::create(ops, 256ull << 20, 105, 0);
            t->start(env.cfg(4ull << 30), kLay, fp);
            spillSess(*t, a, 0);
            CHECK(tickIdle(*t));
            spillSess(*t, b, 0);
            CHECK(tickIdle(*t));
            CHECK(t->dependentsOf(t->find(a.id)).size() == 1);
            Entry* eb = t->find(b.id);
            if (eb && eb->ssd)
                for (const BlkRef& r : eb->ssd->tbl)
                    if (r.owner == a.id) borrowed_lb = std::max(borrowed_lb, r.owner_lb);
            CHECK(borrowed_lb > 0);
        }
        const fs::path fa = entryFile(dir, fp, a.id), fb = entryFile(dir, fp, b.id);
        if (mode == 0) {
            // a's last (own) block becomes a reference to b's last (own) block: a -> b and b -> a
            const Header ha = readHeader(fa), hb = readHeader(fb);
            CHECK(ha.n_blk > 0 && hb.n_blk > 0);
            const auto vb = readFile(fb);
            BlkRef rb{};
            const std::uint64_t ob = hb.tbl_off + static_cast<std::uint64_t>(hb.n_blk - 1) * sizeof(BlkRef);
            if (vb.size() >= ob + sizeof rb) std::memcpy(&rb, vb.data() + ob, sizeof rb);
            CHECK(rb.owner == 0 && (rb.h0 != 0 || rb.h1 != 0));
            BlkRef ra = rb;
            ra.owner = b.id;
            ra.owner_lb = static_cast<std::uint32_t>(static_cast<std::uint64_t>(hb.nck) * kLay.ckStride() / block_bytes) + hb.n_blk - 1;
            FILE* f = _wfopen(fa.c_str(), L"r+b");
            CHECK(f != nullptr);
            if (f) {
                std::fseek(f, static_cast<long>(ha.tbl_off + static_cast<std::uint64_t>(ha.n_blk - 1) * sizeof(BlkRef)), SEEK_SET);
                std::fwrite(&ra, sizeof ra, 1, f);
                std::fclose(f);
            }
            std::vector<std::string> logs;
            auto t = Tier::create(ops, 256ull << 20, 106, 0);
            t->log_fn = [&](std::string_view l) { logs.emplace_back(l); };
            t->start(env.cfg(4ull << 30), kLay, fp);
            bool cyc = false;
            for (const auto& l : logs) cyc = cyc || l.find("reference cycle") != std::string::npos;
            CHECK(cyc);
            for (const Entry* e : t->entries) {
                const auto clo = t->dependentClosure(e);
                CHECK(std::find(clo.begin(), clo.end(), e->id) == clo.end());  // no cycle left
            }
            CHECK(countFiles(dir) == t->ssdEntries());
            CHECK(t->ssd_bytes == diskBytes(dir));
        } else if (mode == 1) {
            fs::resize_file(fa, kLay.dataOff() + static_cast<std::uint64_t>(borrowed_lb) * block_bytes + block_bytes / 2, ec);
            CHECK(!ec);
            auto t = Tier::create(ops, 256ull << 20, 107, 0);
            t->start(env.cfg(4ull << 30), kLay, fp);
            CHECK(t->find(a.id) == nullptr && t->find(b.id) == nullptr);  // the borrower goes with its owner
            CHECK(countFiles(dir) == 0 && t->ssd_bytes == 0);
        } else {
            auto t = Tier::create(ops, 256ull << 20, 108, 0);
            t->start(env.cfg(4ull << 30), kLay, fp);
            Entry* eb = t->find(b.id);
            CHECK(eb != nullptr && t->find(a.id) != nullptr);
            if (!eb) continue;
            // cut inside the block b borrows: a's own extent is not read, only b's borrowed block is short
            fs::resize_file(fa, kLay.dataOff() + static_cast<std::uint64_t>(borrowed_lb) * block_bytes + block_bytes / 2, ec);
            CHECK(!ec);
            DevSlot dst(ops, 2, 24);
            std::vector<Span> sp;
            sp.push_back(Span{dst.ck[0], 0, kLay.ck_bytes, 0});
            sp.push_back(Span{dst.kv, eb->kvOff(kLay), static_cast<std::uint64_t>(eb->pages) * kLay.page_bytes, 0xff});
            Restore* r = t->beginRestore(eb, sp, sp.size());
            CHECK(r != nullptr);
            if (r) {
                CHECK(!runRestore(*t, r) && r->failed);
                t->endRestore(r);
            }
            CHECK(tickIdle(*t));
            for (const Entry* e : t->entries) CHECK(e->pin == 0 && e->ext_pin == 0 && !e->io_busy);
        }
        fs::remove_all(dir, ec);
    }
}

// ---- SSD-QUIET: write amplification of a benchmark-like request stream -------------------
// Ornith-35B-A3B layout from the server log ("entry: 51.2 MiB per checkpoint, 9.0 MiB per 256-token
// page", f16 KV), scaled by 1/amp_div so the default run stays small (WHIRL_TIER_AMP=full: real sizes).

namespace amp {
struct Sim {
    MockDeviceOps& ops;
    Layout lay;
    std::uint32_t nck = 4;
    DevPtr ck_dev = 0, kv_dev = 0;
    std::uint32_t max_pages = 0;
    Sim(MockDeviceOps& o, int div, std::uint32_t max_tokens) : ops(o) {
        lay.ck_bytes = static_cast<std::uint64_t>(51.2 * 1048576.0) / div;
        lay.page_bytes = (9ull << 20) / div;
        lay.page_tokens = 256;
        lay.tok_cap = max_tokens;
        max_pages = (max_tokens + 255) / 256;
        ck_dev = o.malloc(lay.ck_bytes);
        kv_dev = o.malloc(static_cast<std::uint64_t>(max_pages) * lay.page_bytes);
        fillBytes(reinterpret_cast<void*>(ck_dev), lay.ck_bytes, 5);
        fillBytes(reinterpret_cast<void*>(kv_dev), static_cast<std::uint64_t>(max_pages) * lay.page_bytes, 6);
    }
    ~Sim() {
        ops.free(ck_dev);
        ops.free(kv_dev);
    }
    std::uint64_t logical(std::uint32_t n_tok, std::uint32_t valid_ck) const {
        return valid_ck * lay.ck_bytes + static_cast<std::uint64_t>((n_tok + 255) / 256) * lay.page_bytes;
    }
    // new entry: all `valid_ck` checkpoints and all pages are new
    Entry* spillNew(Tier& t, const std::vector<std::uint32_t>& toks, std::uint32_t valid_ck) {
        Entry* e = t.newEntry(nck);
        e->tokens = toks;
        e->n_tok = static_cast<std::uint32_t>(toks.size());
        e->pages = (e->n_tok + 255) / 256;
        std::vector<Span> sp;
        for (std::uint32_t k = 0; k < valid_ck; ++k) {
            e->ck[k] = CkMeta{static_cast<std::uint32_t>((k + 1) * toks.size() / valid_ck), 1, static_cast<std::uint8_t>(k + 1 == valid_ck), 0, 0, k + 1};
            sp.push_back(Span{ck_dev, static_cast<std::uint64_t>(k) * lay.ckStride(), lay.ck_bytes, static_cast<std::uint8_t>(k)});
        }
        sp.push_back(Span{kv_dev, e->kvOff(lay), static_cast<std::uint64_t>(e->pages) * lay.page_bytes, 0xff});
        for (const Span& s : sp) {
            const auto r = Tier::lbRange(s.off, s.len);
            if (!t.ensureBlocks(e, r[0], r[1])) {
                t.removeEntry(e);
                return nullptr;
            }
        }
        t.copyOut(e, sp);
        t.fence(e, nullptr);
        e->t_mod = std::chrono::steady_clock::now();
        return e;
    }
    // a conversation turn: `add` tokens appended; pages from the old last (partial) page are new, the
    // oldest checkpoint slot is replaced by one at the new end (what a request that resumed from the
    // newest checkpoint produces)
    void growTurn(Tier& t, Entry* e, std::size_t add, std::uint64_t seed, std::uint32_t slot) {
        const std::uint32_t old_pages = e->pages;
        auto more = makeTokens(add, seed);
        e->tokens.insert(e->tokens.end(), more.begin(), more.end());
        e->n_tok = static_cast<std::uint32_t>(e->tokens.size());
        e->pages = (e->n_tok + 255) / 256;
        const std::uint32_t vp = old_pages > 0 ? old_pages - 1 : 0;  // the partial last page changes
        std::vector<Span> sp;
        sp.push_back(Span{ck_dev, static_cast<std::uint64_t>(slot) * lay.ckStride(), lay.ck_bytes, static_cast<std::uint8_t>(slot)});
        sp.push_back(Span{kv_dev + static_cast<std::uint64_t>(vp) * lay.page_bytes, e->kvOff(lay) + static_cast<std::uint64_t>(vp) * lay.page_bytes,
                          static_cast<std::uint64_t>(e->pages - vp) * lay.page_bytes, 0xff});
        for (const Span& s : sp) {
            const auto r = Tier::lbRange(s.off, s.len);
            CHECK(t.ensureBlocks(e, r[0], r[1]));
        }
        e->ck[slot] = CkMeta{e->n_tok, 1, 1, 0, 0, 100 + static_cast<std::uint64_t>(e->n_tok)};
        t.copyOut(e, sp);
        t.fence(e, nullptr);
        e->t_mod = std::chrono::steady_clock::now();
    }
};
}  // namespace amp

static void testWriteAmplification(MockDeviceOps& ops, const fs::path& dir) {
    const char* full = std::getenv("WHIRL_TIER_AMP");
    const int div = full != nullptr && std::strcmp(full, "full") == 0 ? 1 : 8;
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    amp::Sim sim(ops, div, 36000);
    // A: distinct prompts of 2k..32k tokens, one request each (4 checkpoints), written at once (delay 0)
    {
        auto t = Tier::create(ops, 1ull << 30, 71, 0);
        t->start(env.cfg(64ull << 30), sim.lay, 9);
        const std::uint32_t lens[] = {2048, 4096, 8192, 16384, 32768};
        std::uint64_t seed = 7100;
        std::printf("amp A: distinct prompts (scale 1/%d): tokens, logical MiB, written MiB, ratio\n", div);
        std::uint64_t req_no = 0;
        for (std::uint32_t n : lens) {
            const std::uint64_t w0 = t->stats.ssd_write_bytes;
            Entry* e = sim.spillNew(*t, freshToks(n, seed), 4);
            CHECK(e != nullptr);
            if (!e) continue;
            CHECK(tickIdle(*t, 120000));
            const std::uint64_t w = t->stats.ssd_write_bytes - w0;
            const std::uint64_t lg = sim.logical(n, 4);
            const std::uint64_t fb = e->ssd ? e->ssd->file_bytes : 0;
            std::printf("  %6u tok  logical %8.1f  file %8.1f  written %8.1f  ratio %.3f (to file %.3f)\n", n, lg / 1048576.0, fb / 1048576.0,
                        w / 1048576.0, static_cast<double>(w) / static_cast<double>(lg), static_cast<double>(w) / static_cast<double>(fb ? fb : 1));
            // the file is the logical size rounded up to whole 2 MiB strides; the writes add only headers
            CHECK(fb > 0 && static_cast<double>(w) <= 1.001 * static_cast<double>(fb));
            CHECK(t->stats.ssd_writes == ++req_no);  // one write per request, never again
        }
        // the same prompts again are RAM hits: nothing to write
        const std::uint64_t w_all = t->stats.ssd_write_bytes;
        CHECK(tickIdle(*t));
        CHECK(t->stats.ssd_write_bytes == w_all);
    }
    fs::remove_all(dir, ec);
    // B: one conversation, 16 turns of +700 tokens from 8k, a checkpoint replaced per turn: the old
    // 2 s behaviour writes after every turn, a long delay once at the end
    std::uint64_t per_turn = 0, coalesced = 0, final_logical = 0;
    for (int variant = 0; variant < 2; ++variant) {
        auto t = Tier::create(ops, 1ull << 30, 72, 0);
        Config c = env.cfg(64ull << 30);
        c.ssd_delay_ms = variant == 0 ? 0 : 3600000;
        t->start(c, sim.lay, 9);
        std::uint64_t seed = 7200;
        Entry* e = sim.spillNew(*t, freshToks(8192, seed), 3);
        CHECK(e != nullptr);
        if (!e) return;
        for (int turn = 0; turn < 16; ++turn) {
            if (variant == 0) CHECK(tickIdle(*t, 120000));
            sim.growTurn(*t, e, 700, seed + turn, static_cast<std::uint32_t>(turn % 3));
        }
        // the delay passes / graceful stop
        t->flushing = true;
        t->cfg.ssd_delay_ms = 0;
        CHECK(tickIdle(*t, 120000));
        (variant == 0 ? per_turn : coalesced) = t->stats.ssd_write_bytes;
        final_logical = sim.logical(e->n_tok, 3);
        t.reset();
        fs::remove_all(dir, ec);
    }
    std::printf("amp B: 16-turn conversation (final entry %.1f MiB): written per turn %.1f MiB (x%.2f), coalesced %.1f MiB (x%.2f)\n",
                final_logical / 1048576.0, per_turn / 1048576.0, static_cast<double>(per_turn) / static_cast<double>(final_logical),
                coalesced / 1048576.0, static_cast<double>(coalesced) / static_cast<double>(final_logical));
    CHECK(coalesced <= final_logical + final_logical / 10);  // stride padding only
    CHECK(per_turn > 2 * coalesced);
}

// A drive with a volatile write cache, for power-loss fault injection (FIX-HDR0). Writes stay in
// the "cache" until a flush barrier puts them on the media in order. The power fails at write
// number crash_at: that write and every later one are lost, and of the writes still cached the
// adversarial subset reaches the media: those at or past persist_from (the data region), none
// before it (header, token ids, block table). That is a legal outcome for a drive that reorders
// unflushed writes, and the worst one for an in-place rewrite.
class PowerLossSim final : public WriteSim {
public:
    int crash_at = 0;  // 1-based; 0: the power never fails
    std::uint64_t persist_from = 0;
    int writes = 0, flushes = 0;
    ~PowerLossSim() override { drop(); }
    bool write(void* fh, const std::uint8_t* buf, std::uint64_t len, std::uint64_t off) override {
        if (dead_) return true;  // lost with the power (the dying process never learns)
        if (++writes == crash_at) {
            powerFail();
            return true;
        }
        auto* p = static_cast<std::uint8_t*>(_aligned_malloc(static_cast<std::size_t>(len), 4096));
        std::memcpy(p, buf, static_cast<std::size_t>(len));
        pending_.push_back(Pending{fh, off, len, p});
        return true;
    }
    bool flush(void* fh) override {
        if (dead_) return true;
        flushes += 1;
        bool ok = true;
        for (const Pending& p : pending_) ok = realWrite(p) && ok;
        drop();
        return ok && FlushFileBuffers(static_cast<HANDLE>(fh)) != 0;
    }
    bool dead() const override { return dead_; }

private:
    struct Pending {
        void* fh;
        std::uint64_t off, len;
        std::uint8_t* buf;
    };
    std::vector<Pending> pending_;
    bool dead_ = false;
    static bool realWrite(const Pending& p) {
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(p.off);
        ov.OffsetHigh = static_cast<DWORD>(p.off >> 32);
        DWORD w = 0;
        return WriteFile(static_cast<HANDLE>(p.fh), p.buf, static_cast<DWORD>(p.len), &w, &ov) != 0 && w == p.len;
    }
    void powerFail() {
        for (const Pending& p : pending_)
            if (p.off >= persist_from) realWrite(p);
        drop();
        dead_ = true;
    }
    void drop() {
        for (const Pending& p : pending_) _aligned_free(p.buf);
        pending_.clear();
    }
};

// Power loss at every write of (a) an in-place rewrite of a valid file and (b) a fresh file's
// first write: the next start must never restore bytes the file's header does not describe
// (the old version or a cold miss; the new version only when nothing was lost). Before
// FIX-HDR0, (a) kept the old valid header over half-new data and restored wrong KV.
static void testPowerLoss(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    SsdEnv env{dir};
    int crash_runs = 0, wrong = 0, cold = 0, old_ok = 0;
    for (int n = 1; n < 64; ++n) {
        fs::remove_all(dir, ec);
        std::uint64_t seed = 3100;
        Sess a1(ops, 24), a2(ops, 24);
        a1.toks = freshToks(3000, seed);
        std::uint64_t id = 0;
        PowerLossSim sim;
        sim.crash_at = n;
        sim.persist_from = kLay.dataOff();
        {
            auto t = Tier::create(ops, 256ull << 20, 31, 0);
            t->start(env.cfg(4ull << 30), kLay, 78);
            spillSess(*t, a1, 0);
            CHECK(tickIdle(*t));
            CHECK(a1.e != nullptr && a1.e->ssd.has_value());
            id = a1.id;
            // regenerate from token 1000, in place (same entry, same file)
            a2.toks = a1.toks;
            a2.toks.resize(1000);
            auto tail = freshToks(2200, seed);
            a2.toks.insert(a2.toks.end(), tail.begin(), tail.end());
            a2.id = a1.id;
            a2.seq = a1.seq;
            a2.fillFrom(0);
            g_write_sim.store(&sim);
            CHECK(!spillSess(*t, a2, 1000));
            CHECK(tickIdle(*t));
            if (!sim.dead()) g_write_sim.store(nullptr);
            t.reset();  // the process is gone (a dead sim keeps the files untouched)
            g_write_sim.store(nullptr);
        }
        const bool crashed = sim.dead();
        {
            auto t = Tier::create(ops, 256ull << 20, 32, 0);
            t->start(env.cfg(4ull << 30), kLay, 78);
            Entry* e = t->find(id);
            if (e == nullptr) {
                cold += 1;
            } else if (e->n_tok == a1.toks.size() && restoreMatches(*t, ops, e, a1)) {
                old_ok += 1;
            } else if (!crashed && e->n_tok == a2.toks.size() && restoreMatches(*t, ops, e, a2)) {
            } else {
                wrong += 1;
                std::printf("  in-place rewrite, power lost at write %d: restored bytes are wrong\n", n);
            }
            CHECK(!crashed || e == nullptr || e->n_tok == a1.toks.size());
            CHECK(tickIdle(*t));
        }
        if (!crashed) {
            CHECK(n > 3 && sim.flushes >= 2);  // ran to the end: the rewrite took n - 1 writes
            break;
        }
        crash_runs += 1;
    }
    std::printf("  in-place: %d crash points, cold %d, old version %d, wrong %d\n", crash_runs, cold, old_ok, wrong);
    CHECK(crash_runs >= 4);
    CHECK(wrong == 0);
    // (b) a fresh file: nothing to restore until the whole file is on the media
    int fresh_runs = 0, fresh_bad = 0;
    for (int n = 1; n < 64; ++n) {
        fs::remove_all(dir, ec);
        std::uint64_t seed = 3200;
        Sess b(ops, 24);
        b.toks = freshToks(2500, seed);
        std::uint64_t id = 0;
        PowerLossSim sim;
        sim.crash_at = n;
        sim.persist_from = kLay.dataOff();
        {
            auto t = Tier::create(ops, 256ull << 20, 33, 0);
            t->start(env.cfg(4ull << 30), kLay, 78);
            g_write_sim.store(&sim);
            CHECK(spillSess(*t, b, 0));
            id = b.id;
            CHECK(tickIdle(*t));
            if (!sim.dead()) g_write_sim.store(nullptr);
            t.reset();
            g_write_sim.store(nullptr);
        }
        const bool crashed = sim.dead();
        {
            auto t = Tier::create(ops, 256ull << 20, 34, 0);
            t->start(env.cfg(4ull << 30), kLay, 78);
            Entry* e = t->find(id);
            if (crashed && e != nullptr) {
                fresh_bad += 1;
                std::printf("  fresh file, power lost at write %d: entry restored\n", n);
            }
            if (!crashed) CHECK(e != nullptr && restoreMatches(*t, ops, e, b));
            CHECK(tickIdle(*t));
        }
        if (!crashed) break;
        fresh_runs += 1;
    }
    std::printf("  fresh: %d crash points, restored %d\n", fresh_runs, fresh_bad);
    CHECK(fresh_runs >= 3);
    CHECK(fresh_bad == 0);
    fs::remove_all(dir, ec);
}

// FIX-CACHE P2-1 (FR-CACHE): RAM entries whose files do not all fit the SSD cap, `always` mode (mode 0) and
// the shutdown-mode flush (mode 1). Before: writing b dropped a's SSD copy (a RAM entry), a became dirty and
// immediately due, its write dropped b's copy, ... an endless rewrite loop (the engine never idle, the SSD
// written continuously). Now: a write never drops the SSD copy of a RAM entry; b is skipped (logged once),
// the cap holds, the tier goes idle.
static void testCapTightRamEntries(MockDeviceOps& ops, const fs::path& dir) {
    for (int mode = 0; mode < 2; ++mode) {
        std::error_code ec;
        fs::remove_all(dir, ec);
        SsdEnv env{dir};
        std::uint64_t seed = 9100 + static_cast<std::uint64_t>(mode) * 10;
        auto t = Tier::create(ops, 256ull << 20, 91, 0);
        const std::uint64_t f1 = fileBytesOf(2, (3000 + kLay.page_tokens - 1) / kLay.page_tokens);
        Config c = env.cfg(f1 + f1 / 2);  // one 3000-token file fits, two do not
        c.ssd_shutdown_only = mode == 1;
        std::size_t nofit_logs = 0;
        t->log_fn = [&](std::string_view m) {
            if (m.find("not written to SSD") != std::string_view::npos) nofit_logs += 1;
        };
        t->start(c, kLay, 91);
        Sess a(ops, 32), b(ops, 32);
        a.toks = freshToks(3000, seed);
        b.toks = freshToks(3000, seed);
        spillSess(*t, a, 0);
        if (mode == 0) CHECK(tickIdle(*t));
        sleepMs(5);
        spillSess(*t, b, 0);
        reuse(*t, a);  // the shutdown-mode close flush writes only reused entries (CLOSE-REUSED)
        reuse(*t, b);
        if (mode == 1) t->flushing = true;
        const bool idle = tickIdle(*t, 3000);
        CHECK(idle);
        std::printf("  mode %d: idle %d, ssd_writes %llu, ssd_bytes %llu (cap %llu), nofit logs %zu\n", mode, idle ? 1 : 0,
                    static_cast<unsigned long long>(t->stats.ssd_writes), static_cast<unsigned long long>(t->ssd_bytes),
                    static_cast<unsigned long long>(t->cfg.ssd_cap), nofit_logs);
        CHECK(t->stats.ssd_writes == 1);
        CHECK(t->ssd_bytes <= t->cfg.ssd_cap && diskBytes(dir) <= t->cfg.ssd_cap);
        CHECK(nofit_logs == 1);
        CHECK(countFiles(dir) == 1);
        // more ticks: the skipped entry is not retried (nothing freed, same size), no new log
        for (int i = 0; i < 20; ++i) t->tick();
        CHECK(t->stats.ssd_writes == 1 && nofit_logs == 1);
        // room freed (the written entry removed): the skipped one is written now
        Entry* written = t->find(a.id) && t->find(a.id)->ssd ? t->find(a.id) : t->find(b.id);
        CHECK(written != nullptr && written->ssd.has_value());
        if (written) t->removeEntry(written);
        CHECK(tickIdle(*t, 3000));
        CHECK(t->stats.ssd_writes == 2 && countFiles(dir) == 1);
        CHECK(t->ssd_bytes <= t->cfg.ssd_cap && diskBytes(dir) <= t->cfg.ssd_cap);
        t.reset();
        fs::remove_all(dir, ec);
    }
}

// FIX-CACHE P2-1 (FR-CACHE, cap overrun): b borrows blocks of the LRU SSD-only entry a; b's write makes room
// by dropping a, which drops b's own copy too. Before: the room was made for fb - old (b's old copy) but the
// reservation charged the whole new file, so ssd_bytes ended up to one old copy over the cap. Now the room
// is made again for the whole file; here it cannot be (c, in RAM, is newer than a: its copy is not dropped
// for a write), so b's write is skipped and the cap holds.
static void testCapOwnCopyDropped(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9150;
    const std::uint64_t fp = 94;
    const auto sys = freshToks(2560, seed);
    Sess a(ops, 32), b(ops, 32), c(ops, 32);
    a.toks = sys;
    {
        auto x = freshToks(1000, seed);
        a.toks.insert(a.toks.end(), x.begin(), x.end());
    }
    b.toks = sys;
    {
        auto x = freshToks(700, seed);
        b.toks.insert(b.toks.end(), x.begin(), x.end());
    }
    c.toks = freshToks(3000, seed);
    {
        auto t = Tier::create(ops, 256ull << 20, 94, 0);
        t->start(env.cfg(8ull << 30), kLay, fp);
        spillSess(*t, a, 0);
        CHECK(tickIdle(*t));
        sleepMs(1100);  // distinct last-write seconds: the LRU order survives the restart
        spillSess(*t, b, 0);
        CHECK(tickIdle(*t));
        sleepMs(1100);
        spillSess(*t, c, 0);
        CHECK(tickIdle(*t));
        CHECK(t->dependentsOf(a.e).size() == 1);
    }
    auto t = Tier::create(ops, 256ull << 20, 95, 0);
    t->start(env.cfg(8ull << 30), kLay, fp);
    CHECK(t->ssdEntries() == 3);
    Entry* eb = t->find(b.id);
    CHECK(eb != nullptr && restoreMatches(*t, ops, eb, b));  // b back in RAM (clean); raises a (b borrows its blocks)
    Entry* ec0 = t->find(c.id);
    CHECK(ec0 != nullptr && restoreMatches(*t, ops, ec0, c));  // c newer than a: a is the LRU victim
    eb = t->find(b.id);
    CHECK(eb != nullptr && eb->in_ram && eb->ssd.has_value());
    if (eb == nullptr || !eb->in_ram || !eb->ssd) return;
    const std::size_t bn = b.toks.size();
    auto more = freshToks(600, seed);
    b.toks.insert(b.toks.end(), more.begin(), more.end());
    CHECK(!spillSess(*t, b, bn));  // in place, appended: dirty
    eb = t->find(b.id);
    const std::uint64_t fb = (kLay.dataOff() + eb->extent(kLay) + sector - 1) / sector * sector;
    const std::uint64_t old0 = eb->ssd->alloc_bytes;
    const std::uint64_t c_alloc = t->find(c.id)->ssd->alloc_bytes;
    t->cfg.ssd_cap = c_alloc + fb - 1;
    CHECK(t->ssd_bytes + fb - old0 > t->cfg.ssd_cap);  // a (LRU) must go for b's write
    t->tick();  // b's write: dropping a drops b's own copy
    std::printf("  ssd_bytes %llu (cap %llu, fb %llu, b old %llu), a %s, c %s\n", static_cast<unsigned long long>(t->ssd_bytes),
                static_cast<unsigned long long>(t->cfg.ssd_cap), static_cast<unsigned long long>(fb),
                static_cast<unsigned long long>(old0), t->find(a.id) ? "kept" : "gone", t->find(c.id) ? "kept" : "gone");
    CHECK(t->ssd_bytes <= t->cfg.ssd_cap);
    CHECK(tickIdle(*t, 3000));
    CHECK(t->ssd_bytes <= t->cfg.ssd_cap && diskBytes(dir) <= t->cfg.ssd_cap);
    eb = t->find(b.id);
    CHECK(eb != nullptr && !eb->ssd.has_value() && eb->in_ram);  // skipped: stays in RAM only
    CHECK(t->find(c.id) != nullptr && t->find(c.id)->ssd.has_value());
    if (eb) CHECK(restoreMatches(*t, ops, eb, b));
    t.reset();
    fs::remove_all(dir, ec);
}

// FIX-CACHE P2-4: Tier::demote (an entry that shares only its system prompt with a newer conversation): kept,
// the first RAM eviction victim, dropped instead of written to SSD; any use undoes it.
static void testDemote(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9300;
    auto t = Tier::create(ops, 72ull << 20, 97, 0);  // 36 blocks; a spilled 3000-token entry takes 11
    Config c = env.cfg(4ull << 30);
    c.ssd_delay_ms = 600000;  // nothing due: a RAM eviction would write an entry now
    t->start(c, kLay, 97);
    Sess a(ops, 24), b(ops, 24), cc(ops, 24), d(ops, 24);
    a.toks = freshToks(3000, seed);
    b.toks = freshToks(3000, seed);
    cc.toks = freshToks(3000, seed);
    d.toks = freshToks(3000, seed);
    spillSess(*t, a, 0);
    spillSess(*t, b, 0);
    spillSess(*t, cc, 0);
    CHECK(t->countRam() == 3);
    t->demote(t->find(b.id));
    CHECK(t->stats.demoted == 1 && t->find(b.id)->demoted);
    spillSess(*t, d, 0);  // needs room: b goes first (not a, the LRU), dropped without an SSD write
    CHECK(t->find(b.id) == nullptr);
    CHECK(t->find(a.id) != nullptr && t->find(a.id)->in_ram);
    CHECK(t->find(d.id) != nullptr && t->find(d.id)->in_ram);
    CHECK(tickIdle(*t));
    CHECK(t->stats.ssd_writes == 0 && countFiles(dir) == 0);
    // a use undoes it
    Entry* ea = t->find(a.id);
    t->demote(ea);
    CHECK(ea->demoted);
    CHECK(restoreMatches(*t, ops, ea, a));
    CHECK(!t->find(a.id)->demoted);
    // an in-place update too
    Entry* ed = t->find(d.id);
    t->demote(ed);
    const std::size_t dn = d.toks.size();
    auto more = freshToks(200, seed);
    d.toks.insert(d.toks.end(), more.begin(), more.end());
    CHECK(!spillSess(*t, d, dn));
    CHECK(!t->find(d.id)->demoted);
    t.reset();
    fs::remove_all(dir, ec);
}

// FIX-CACHE P2-2 (FR-CACHE): a restart with another per-slot context (tok_cap: --ctx, --parallel, or the
// VRAM-based auto-shrink) changes the file layout (dataOff). Before: the files of the same fingerprint were
// deleted as corrupt. Now: kept like foreign files (counted against the cap, evicted first), and used again
// by a server with the old context.
static void testTokCapRestart(MockDeviceOps& ops, const fs::path& dir) {
    std::error_code ec;
    fs::remove_all(dir, ec);
    SsdEnv env{dir};
    std::uint64_t seed = 9200;
    Sess a(ops, 32);
    a.toks = freshToks(3000, seed);
    std::uint64_t bytes1 = 0;
    {
        auto t = Tier::create(ops, 256ull << 20, 92, 0);
        t->start(env.cfg(8ull << 30), kLay, 92);
        spillSess(*t, a, 0);
        CHECK(tickIdle(*t));
        CHECK(t->ssdEntries() == 1);
        bytes1 = t->ssd_bytes;
    }
    for (int k = 0; k < 2; ++k) {  // 0: smaller context, 1: larger
        Layout l2 = kLay;
        l2.tok_cap = k == 0 ? kLay.tok_cap / 2 : kLay.tok_cap * 2;
        auto t2 = Tier::create(ops, 256ull << 20, 93, 0);
        std::vector<std::string> logs;
        t2->log_fn = [&](std::string_view m) { logs.emplace_back(m); };
        t2->start(env.cfg(8ull << 30), l2, 92);
        std::printf("  tok_cap %u -> %u: entries %zu, bad_files %llu, layout_files %llu, files on disk %zu\n", kLay.tok_cap, l2.tok_cap,
                    t2->ssdEntries(), static_cast<unsigned long long>(t2->cleanup().bad_files),
                    static_cast<unsigned long long>(t2->cleanup().layout_files), countFiles(dir));
        CHECK(t2->cleanup().bad_files == 0);
        CHECK(t2->cleanup().layout_files == 1);
        CHECK(countFiles(dir) == 1);
        CHECK(t2->ssdEntries() == 0);           // not usable with this layout
        CHECK(t2->ssd_bytes > 0 && t2->ssd_bytes == diskBytes(dir));  // counted against the cap (on-disk bytes)
        CHECK(!t2->lookup(a.toks).has_value());
        bool logged = false;
        for (const auto& l : logs) logged = logged || l.find("another per-slot context") != std::string::npos;
        CHECK(logged);
        if (k == 0) {
            // evicted first: a new entry that needs the room replaces it
            const std::uint64_t f1 = (l2.dataOff() + 2 * kLay.ckStride() +
                                      static_cast<std::uint64_t>((2500 + kLay.page_tokens - 1) / kLay.page_tokens) * kLay.page_bytes + sector - 1) /
                                     sector * sector;
            t2.reset();
            auto t3 = Tier::create(ops, 256ull << 20, 94, 0);
            t3->start(env.cfg(std::max(bytes1, f1) + f1 / 2), l2, 92);
            Sess b(ops, 32);
            b.toks = freshToks(2500, seed);
            spillSess(*t3, b, 0);
            CHECK(tickIdle(*t3));
            CHECK(t3->ssdEntries() == 1 && countFiles(dir) == 1);
            CHECK(t3->ssd_bytes <= t3->cfg.ssd_cap);
            CHECK(t3->find(a.id) == nullptr && t3->find(b.id) != nullptr);
            t3.reset();
            // re-create a's file for the next round
            fs::remove_all(dir, ec);
            auto t4 = Tier::create(ops, 256ull << 20, 95, 0);
            t4->start(env.cfg(8ull << 30), kLay, 92);
            a.id = 0;
            spillSess(*t4, a, 0);
            CHECK(tickIdle(*t4));
        }
    }
    // back to the original context: the file is used again
    auto t5 = Tier::create(ops, 256ull << 20, 96, 0);
    t5->start(env.cfg(8ull << 30), kLay, 92);
    CHECK(t5->ssdEntries() == 1 && t5->cleanup().layout_files == 0 && t5->cleanup().bad_files == 0);
    Entry* ea = t5->find(a.id);
    CHECK(ea != nullptr);
    if (ea) CHECK(restoreMatches(*t5, ops, ea, a));
    t5.reset();
    fs::remove_all(dir, ec);
}

// WHIRL_TIER_ONLY=<test name>: run only that test (regression proofs). A name that matches no test is
// a usage error (exit 2; FIX-TEST-A, FR-TEST P3-3: a typo used to run 1 check in main and pass).
static int g_only_matched = 0;
static bool want(const char* name) {
    const char* only = std::getenv("WHIRL_TIER_ONLY");
    const bool all = only == nullptr || *only == 0;
    if (!all && std::strcmp(only, name) == 0) ++g_only_matched;
    return all || std::strcmp(only, name) == 0;
}

int main(int argc, char** argv) {
    whirl::test::clearWhirlEnv({L"WHIRL_TIER_ONLY", L"WHIRL_TIER_AMP"});  // FR-TEST P3-11
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // default: WHIRL_TEST_TMP, else %TEMP%\whirl-tests; this test uses <base>\tier_test_tmp
    fs::path base;
    if (const wchar_t* t = _wgetenv(L"WHIRL_TEST_TMP"); t && *t) base = fs::path(t);
    else base = fs::temp_directory_path() / L"whirl-tests";
    fs::path dir = argc > 1 ? fs::path(argv[1]) : base / L"tier_test_tmp";
    MockDeviceOps ops;
    if (want("testHelpers")) {
        std::printf("-- %s\n", "testHelpers");
        testHelpers();
    }
    if (want("testEvictPolicy")) {
        std::printf("-- %s\n", "testEvictPolicy");
        testEvictPolicy();
    }
    if (want("testRamTierSize")) {
        std::printf("-- %s\n", "testRamTierSize");
        testRamTierSize();
    }
    if (want("testLookup")) {
        std::printf("-- %s\n", "testLookup");
        testLookup(ops);
    }
    if (want("testEvictLru")) {
        std::printf("-- %s\n", "testEvictLru");
        testEvictLru(ops);
    }
    if (want("testTrimDropCk")) {
        std::printf("-- %s\n", "testTrimDropCk");
        testTrimDropCk(ops);
    }
    if (want("testRestoreRam")) {
        std::printf("-- %s\n", "testRestoreRam");
        testRestoreRam(ops);
    }
    CHECK(ops.live_events.load() == 0 && ops.live_host.load() == 0 && ops.live_streams.load() == 0);
    if (want("testSsd")) {
        std::printf("-- %s\n", "testSsd");
        testSsd(ops, dir);
    }
    if (want("testDedupRestart")) {
        std::printf("-- %s\n", "testDedupRestart");
        testDedupRestart(ops, dir);
    }
    if (want("testOwnerRewrite")) {
        std::printf("-- %s\n", "testOwnerRewrite");
        testOwnerRewrite(ops, dir);
    }
    if (want("testModesLock")) {
        std::printf("-- %s\n", "testModesLock");
        testModesLock(ops, dir);
    }
    if (want("testSsdTruncated")) {
        std::printf("-- %s\n", "testSsdTruncated");
        testSsdTruncated(ops, dir);
    }
    if (want("testRestoreCopyThrows")) {
        std::printf("-- %s\n", "testRestoreCopyThrows");
        testRestoreCopyThrows(ops, dir);
    }
    if (want("testLockFailClosed")) {
        std::printf("-- %s\n", "testLockFailClosed");
        testLockFailClosed(ops, dir);
    }
    if (want("testDedupCycle")) {
        std::printf("-- %s\n", "testDedupCycle");
        testDedupCycle(ops, dir);
    }
    if (want("testSupersedeTrim")) {
        std::printf("-- %s\n", "testSupersedeTrim");
        testSupersedeTrim(ops, dir);
    }
    if (want("testWorkflows")) {
        std::printf("-- %s\n", "testWorkflows");
        testWorkflows(ops, dir);
    }
    if (want("testQuietDefaults")) {
        std::printf("-- %s\n", "testQuietDefaults");
        testQuietDefaults();
    }
    if (want("testWriteDelay")) {
        std::printf("-- %s\n", "testWriteDelay");
        testWriteDelay(ops, dir);
    }
    if (want("testEvictionWrite")) {
        std::printf("-- %s\n", "testEvictionWrite");
        testEvictionWrite(ops, dir);
    }
    if (want("testEvictionWriteSsdFull")) {
        std::printf("-- %s\n", "testEvictionWriteSsdFull");
        testEvictionWriteSsdFull(ops, dir);
    }
    if (want("testShutdownFlush")) {
        std::printf("-- %s\n", "testShutdownFlush");
        testShutdownFlush(ops, dir);
    }
    if (want("testDedupFinishedUnreaped")) {
        std::printf("-- %s\n", "testDedupFinishedUnreaped");
        testDedupFinishedUnreaped(ops, dir);
    }
    if (want("testFlushDedup")) {
        std::printf("-- %s\n", "testFlushDedup");
        testFlushDedup(ops, dir);
    }
    if (want("testFlushOrder")) {
        std::printf("-- %s\n", "testFlushOrder");
        testFlushOrder(ops, dir);
    }
    if (want("testSeenOnce")) {
        std::printf("-- %s\n", "testSeenOnce");
        testSeenOnce(ops, dir);
    }
    if (want("testIdleFlushPolicy")) {
        std::printf("-- %s\n", "testIdleFlushPolicy");
        testIdleFlushPolicy();
    }
    if (want("testIdleWriteTier")) {
        std::printf("-- %s\n", "testIdleWriteTier");
        testIdleWriteTier(ops, dir);
    }
    if (want("testClosePolicy")) {
        std::printf("-- %s\n", "testClosePolicy");
        testClosePolicy();
    }
    if (want("testCloseReusedTier")) {
        std::printf("-- %s\n", "testCloseReusedTier");
        testCloseReusedTier(ops, dir);
    }
    if (want("testShutdownEvictReused")) {
        std::printf("-- %s\n", "testShutdownEvictReused");
        testShutdownEvictReused(ops, dir);
    }
    if (want("testStartupCleanupCap")) {
        std::printf("-- %s\n", "testStartupCleanupCap");
        testStartupCleanupCap(ops, dir);
    }
    if (want("testCapLruDedup")) {
        std::printf("-- %s\n", "testCapLruDedup");
        testCapLruDedup(ops, dir);
    }
    if (want("testExpiry")) {
        std::printf("-- %s\n", "testExpiry");
        testExpiry(ops, dir);
    }
    if (want("testRamUsePersisted")) {
        std::printf("-- %s\n", "testRamUsePersisted");
        testRamUsePersisted(ops, dir);
    }
    if (want("testShutdownNoWrites")) {
        std::printf("-- %s\n", "testShutdownNoWrites");
        testShutdownNoWrites(ops, dir);
    }
    if (want("testDeleteFails")) {
        std::printf("-- %s\n", "testDeleteFails");
        testDeleteFails(ops, dir);
    }
    if (want("testRestoreAsyncError")) {
        std::printf("-- %s\n", "testRestoreAsyncError");
        testRestoreAsyncError(ops);
    }
    if (want("testIdleWakeRoomPending")) {
        std::printf("-- %s\n", "testIdleWakeRoomPending");
        testIdleWakeRoomPending(ops, dir);
    }
    if (want("testTakeoverKeepsJobOrder")) {
        std::printf("-- %s\n", "testTakeoverKeepsJobOrder");
        testTakeoverKeepsJobOrder(ops, dir);
    }
    if (want("testDedupStartupCycleOwnerTrunc")) {
        std::printf("-- %s\n", "testDedupStartupCycleOwnerTrunc");
        testDedupStartupCycleOwnerTrunc(ops, dir);
    }
    if (want("testWriteAmplification")) {
        std::printf("-- %s\n", "testWriteAmplification");
        testWriteAmplification(ops, dir);
    }
    if (want("testDemote")) {
        std::printf("-- %s\n", "testDemote");
        testDemote(ops, dir);
    }
    if (want("testTokCapRestart")) {
        std::printf("-- testTokCapRestart\n");
        testTokCapRestart(ops, dir);
    }
    if (want("testCapTightRamEntries")) {
        std::printf("-- %s\n", "testCapTightRamEntries");
        testCapTightRamEntries(ops, dir);
    }
    if (want("testCapOwnCopyDropped")) {
        std::printf("-- testCapOwnCopyDropped\n");
        testCapOwnCopyDropped(ops, dir);
    }
    if (want("testPowerLoss")) {
        std::printf("-- %s\n", "testPowerLoss");
        testPowerLoss(ops, dir);
    }
    std::printf("tier tests: %d passed, %d failed\n", g_pass, g_fail);
    if (const char* only = std::getenv("WHIRL_TIER_ONLY"); only && *only && g_only_matched == 0) {
        std::printf("tier tests: WHIRL_TIER_ONLY=%s matches no test\n", only);
        return 2;
    }
    if (g_pass + g_fail == 0) {
        std::printf("tier tests: 0 checks ran: refusing to report success\n");
        return 2;
    }
    return g_fail == 0 ? 0 : 1;
}
