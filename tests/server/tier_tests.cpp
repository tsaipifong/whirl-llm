// Tests of the host prefix-cache tiers (src/tier/kv_tier) against a
// host-memory DeviceOps mock (no GPU): device addresses are host addresses,
// copies run immediately, events are always complete.
// SPDX-License-Identifier: Apache-2.0
//
// Usage: whirl-tier-tests [temp dir for the SSD tier]
//   (default: %WHIRL_TEST_TMP%\tier_test_tmp, else %TEMP%\whirl-tests\tier_test_tmp)

#include "tier/kv_tier.h"
#include "tier/ram_size.h"

#include <malloc.h>

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

    void copyAsync(DevPtr dst, DevPtr src, std::uint64_t bytes, Stream) override {
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
    bool eventDone(Event) override { return true; }
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
    while (t.tick()) {
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
        CHECK(t->stats.ssd_writes == 1 && t->stats.ssd_write_bytes == 10 * block_bytes);
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
                CHECK(std::memcmp(h.magic, "WHKVTIER", 8) == 0 && h.version == 1 && h.valid == 1);
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

static void testHelpers() {
    CHECK(hashStr(0xcbf29ce484222325ull, "") == 0xcbf29ce484222325ull);
    CHECK(hashStr(0xcbf29ce484222325ull, "a") == 0xaf63dc4c8601ec8cull);  // FNV-1a 64 of "a"
    CHECK(exeIdentity() != 0);
    CHECK(kLay.ckStride() == (4ull << 20));
    CHECK(kLay.tokRegion() == 32768 && kLay.dataOff() == 36864);
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
    // integrated GPU (UMA): off by default
    {
        RamTierInput in;
        in.total_phys = 128 * G;
        in.avail_phys = 100 * G;
        in.integrated = true;
        CHECK(ramTierSize(in).mb == 0);
        CHECK(ramTierSize(in).reason.find("integrated") != std::string::npos);
    }
}

int main(int argc, char** argv) {
    // default: WHIRL_TEST_TMP, else %TEMP%\whirl-tests; this test uses <base>\tier_test_tmp
    fs::path base;
    if (const wchar_t* t = _wgetenv(L"WHIRL_TEST_TMP"); t && *t) base = fs::path(t);
    else base = fs::temp_directory_path() / L"whirl-tests";
    fs::path dir = argc > 1 ? fs::path(argv[1]) : base / L"tier_test_tmp";
    MockDeviceOps ops;
    testHelpers();
    testRamTierSize();
    testLookup(ops);
    testEvictLru(ops);
    testTrimDropCk(ops);
    testRestoreRam(ops);
    CHECK(ops.live_events.load() == 0 && ops.live_host.load() == 0 && ops.live_streams.load() == 0);
    testSsd(ops, dir);
    std::printf("tier tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
