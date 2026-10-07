// Host tiers for the server's prefix cache (VRAM -> pinned RAM -> SSD).
// SPDX-License-Identifier: Apache-2.0
//
// See kv_tier.h for the design. Windows only for the SSD tier (unbuffered
// positional file I/O through <windows.h>).

#include "kv_tier.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <new>
#include <utility>

namespace whirl::tier {

namespace {

constexpr std::uint64_t alignUp(std::uint64_t x, std::uint64_t a) { return (x + a - 1) / a * a; }

std::uint64_t satSub(std::uint64_t a, std::uint64_t b) { return a > b ? a - b : 0; }

template <class F>
struct ScopeExit {
    F f;
    ~ScopeExit() { f(); }
};
template <class F>
ScopeExit(F) -> ScopeExit<F>;

// UTF-8 -> UTF-16 (throws on invalid UTF-8, like utf8ToUtf16LeAllocZ).
std::wstring utf8ToWide(std::string_view s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) throw TierError("kv tier: invalid UTF-8 path");
    std::wstring w(static_cast<std::size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

template <class... A>
std::string fmt(const char* f, A... a) {
    char buf[320];
    const int n = std::snprintf(buf, sizeof buf, f, a...);
    if (n < 0) return {};
    return std::string(buf, static_cast<std::size_t>(std::min<int>(n, static_cast<int>(sizeof buf) - 1)));
}

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

std::int64_t unixNow() { return static_cast<std::int64_t>(std::time(nullptr)); }

std::uint64_t tokSum(const std::uint32_t* toks, std::size_t n) {
    std::uint64_t h = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < n; ++i) {
        h ^= toks[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

// sector-aligned heap buffer (unbuffered I/O)
struct AlignedFree {
    void operator()(std::uint8_t* p) const { ::operator delete[](p, std::align_val_t{4096}); }
};
using AlignedBuf = std::unique_ptr<std::uint8_t[], AlignedFree>;
AlignedBuf alignedAlloc(std::size_t n) {
    return AlignedBuf(static_cast<std::uint8_t*>(::operator new[](n, std::align_val_t{4096})));
}

OVERLAPPED ovAt(std::uint64_t off) {
    OVERLAPPED ov{};
    ov.Offset = static_cast<DWORD>(off);
    ov.OffsetHigh = static_cast<DWORD>(off >> 32);
    return ov;
}

bool writeAt(HANDLE fh, const std::uint8_t* buf, std::uint64_t len, std::uint64_t off) {
    std::uint64_t done = 0;
    while (done < len) {
        const DWORD n = static_cast<DWORD>(std::min<std::uint64_t>(len - done, 64ull << 20));
        OVERLAPPED ov = ovAt(off + done);
        DWORD w = 0;
        if (WriteFile(fh, buf + done, n, &w, &ov) == 0 || w == 0) return false;
        done += w;
    }
    return true;
}

bool readAt(HANDLE fh, std::uint8_t* buf, std::uint64_t len, std::uint64_t off) {
    std::uint64_t done = 0;
    while (done < len) {
        const DWORD n = static_cast<DWORD>(std::min<std::uint64_t>(len - done, 64ull << 20));
        OVERLAPPED ov = ovAt(off + done);
        DWORD r = 0;
        if (ReadFile(fh, buf + done, n, &r, &ov) == 0 || r == 0) return false;
        done += r;
    }
    return true;
}

}  // namespace

// ---- IO jobs (IO thread) ----------------------------------------------------

enum class JobKind { write, read };

struct Job {
    JobKind kind = JobKind::read;
    std::uint64_t entry_id = 0;
    std::wstring path;
    // write: header (valid = 0 / 1 copies) and tokens, sector-aligned buffers
    AlignedBuf hdr0, hdr1, toks;
    std::uint64_t toks_len = 0;
    std::uint64_t tok_off = 0;
    std::uint64_t data_off = 0;
    std::uint64_t file_bytes = 0;
    // (logical block, arena block address) to write / read, ascending
    std::vector<std::uint32_t> lbs;
    std::vector<std::uint64_t> ptrs;
    std::atomic<std::uint32_t> done{0};
    std::atomic<bool> finished{false};
    std::atomic<bool> failed{false};
    // read: tier-stream work enqueued before the job (copies that may still
    // touch the arena blocks it fills) must finish before the file is read
    Event fence = nullptr;
    // SSD -> RAM read started ahead of a restore (queued request); cleared
    // when a restore takes the job over (it then runs at restore priority)
    std::atomic<bool> prefetch{false};
    // write: block table (rows; cand: a candidate reference, kept when the
    // block hashes the same, else the block is written) and what was written
    std::vector<BlkRef> rows;
    std::vector<std::uint8_t> cand;
    std::uint32_t kv_lb0 = 0;
    AlignedBuf tbl;
    std::uint64_t tbl_off = 0;
    std::uint64_t reserved = 0;  // ssd_bytes reserved at submit
    std::uint64_t written = 0, dedup = 0;
    bool meta_only = false;
    // read: source file per job block (paths[src[i]], logical block src_lb[i])
    // and the owner entries pinned (ext_pin) while the job runs
    std::vector<std::wstring> paths;
    std::vector<std::uint16_t> src;
    std::vector<std::uint32_t> src_lb;
    std::vector<std::uint64_t> owners;
    // write snapshot
    std::optional<SsdSnap> snap;
    std::chrono::steady_clock::time_point t0{};
};

// ---- Layout / Entry -----------------------------------------------------------

std::uint64_t Layout::ckStride() const { return alignUp(ck_bytes, block_bytes); }
std::uint64_t Layout::tokRegion() const { return alignUp(static_cast<std::uint64_t>(tok_cap) * 4, sector); }
std::uint64_t Layout::maxKvBlocks() const {
    const std::uint64_t pt = page_tokens > 0 ? page_tokens : 1;
    const std::uint64_t pages = (static_cast<std::uint64_t>(tok_cap) + pt - 1) / pt;
    return (pages * page_bytes + block_bytes - 1) / block_bytes + 1;
}
std::uint64_t Layout::tblRegion() const { return alignUp(maxKvBlocks() * sizeof(BlkRef), sector); }
std::uint64_t Layout::dataOff() const { return header_bytes + tokRegion() + tblRegion(); }

// 4 independent multiply-xorshift lanes over 64-bit words, folded to 128 bits.
std::array<std::uint64_t, 2> blockHash(const void* p, std::uint64_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    std::uint64_t a0 = 0x9e3779b97f4a7c15ull ^ n, a1 = 0xbf58476d1ce4e5b9ull, a2 = 0x94d049bb133111ebull, a3 = 0xd6e8feb86659fd93ull;
    constexpr std::uint64_t m0 = 0xff51afd7ed558ccdull, m1 = 0xc4ceb9fe1a85ec53ull;
    std::uint64_t i = 0;
    auto rd = [&](std::uint64_t o) {
        std::uint64_t v;
        std::memcpy(&v, b + o, 8);
        return v;
    };
    for (; i + 32 <= n; i += 32) {
        a0 = (a0 ^ rd(i)) * m0;
        a1 = (a1 ^ rd(i + 8)) * m1;
        a2 = (a2 ^ rd(i + 16)) * m0;
        a3 = (a3 ^ rd(i + 24)) * m1;
        a0 ^= a0 >> 29;
        a1 ^= a1 >> 31;
        a2 ^= a2 >> 27;
        a3 ^= a3 >> 33;
    }
    for (; i < n; ++i) a0 = (a0 ^ b[i]) * m0;
    auto fin = [](std::uint64_t z) {
        z = (z ^ (z >> 33)) * 0xff51afd7ed558ccdull;
        z = (z ^ (z >> 33)) * 0xc4ceb9fe1a85ec53ull;
        return z ^ (z >> 33);
    };
    std::uint64_t h0 = fin(a0 ^ fin(a1 + 0x632be59bd9b4e019ull) ^ (a2 * 3));
    std::uint64_t h1 = fin(a3 ^ fin(a2 + 0x8cb92ba72f3d8dd7ull) ^ (a1 * 5) ^ (a0 >> 7));
    if (h0 == 0 && h1 == 0) h1 = 1;  // (0, 0) means "unknown"
    return {h0, h1};
}

std::uint64_t Entry::kvOff(const Layout& l) const { return static_cast<std::uint64_t>(nck) * l.ckStride(); }
std::uint64_t Entry::extent(const Layout& l) const {
    return kvOff(l) + static_cast<std::uint64_t>(pages) * l.page_bytes;
}
bool Entry::hasDirty() const {
    for (auto d : dirty)
        if (d) return true;
    return false;
}

// ---- Tier: create / start -----------------------------------------------------

Tier::Tier(DeviceOps& ops, std::uint64_t seed, int device) : ops_(ops), rng_(seed), dev_(device) {}

std::unique_ptr<Tier> Tier::create(DeviceOps& ops, std::uint64_t ram_bytes, std::uint64_t seed, int device) {
    std::unique_ptr<Tier> t(new Tier(ops, seed, device));
    t->cfg.ram_bytes = ram_bytes;
    t->stream = ops.streamCreateNonBlocking();
    // pinned arena (512 MiB chunks: one huge pinned allocation can be refused)
    std::uint64_t left = alignUp(ram_bytes, block_bytes);
    while (left > 0) {
        const std::uint64_t n = std::min(left, chunk_bytes);
        void* buf = ops.hostMalloc(n);
        t->chunks_.push_back(buf);
        t->chunk_len_.push_back(n);
        left -= n;
    }
    std::uint32_t nb = 0;
    for (auto len : t->chunk_len_) nb += static_cast<std::uint32_t>(len / block_bytes);
    t->n_blocks_ = nb;
    t->free_blocks_.reserve(nb);
    std::uint32_t b = nb;
    while (b > 0) {
        b -= 1;
        t->free_blocks_.push_back(b);
    }
    // a few events up front (no allocations later on the hot path)
    for (int i = 0; i < 16; ++i) t->ev_pool_.push_back(ops.eventCreate());
    return t;
}

Tier::~Tier() {
    {
        std::lock_guard<std::mutex> lk(q_mutex_);
        stop_ = true;
    }
    q_cond_.notify_all();
    if (io_thread_.joinable()) io_thread_.join();
    if (dir_lock_ != nullptr) CloseHandle(static_cast<HANDLE>(dir_lock_));
    dir_lock_ = nullptr;
    // every job still known (finished or queued); restores must have ended
    std::vector<Job*> all = jobs_;
    for (Job* j : q_reads_)
        if (std::find(all.begin(), all.end(), j) == all.end()) all.push_back(j);
    for (Job* j : q_writes_)
        if (std::find(all.begin(), all.end(), j) == all.end()) all.push_back(j);
    q_reads_.clear();
    q_writes_.clear();
    jobs_.clear();
    for (Job* j : all) freeJob(j);
    // entries: memory only (their SSD files stay for the next start)
    for (Entry* e : entries) {
        if (e->ev != nullptr) poolEvent(e->ev);
        delete e;
    }
    entries.clear();
    for (Event ev : ev_pool_) {
        try {
            ops_.eventDestroy(ev);
        } catch (...) {
        }
    }
    ev_pool_.clear();
    for (void* c : chunks_) {
        try {
            ops_.hostFree(c);
        } catch (...) {
        }
    }
    chunks_.clear();
    if (stream != nullptr) {
        try {
            ops_.streamDestroy(stream);
        } catch (...) {
        }
    }
}

void Tier::start(const Config& cfg_in, const Layout& lay_in, std::uint64_t fingerprint) {
    Config c = cfg_in;
    c.ram_bytes = cfg.ram_bytes;
    cfg = c;
    lay = lay_in;
    fingerprint_ = fingerprint;
    if (cfg.ssd_dir) {
        std::wstring dw = utf8ToWide(*cfg.ssd_dir);
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(dw), ec);
        // one server per directory: a second one (same fingerprint) would take
        // the first one's files for its own and evict them under it
        const std::wstring lp = dw + L"\\whirl.lock";
        HANDLE lh = CreateFileW(lp.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (lh == INVALID_HANDLE_VALUE) {
            if (GetLastError() == ERROR_SHARING_VIOLATION) {
                dir_busy_ = true;
                if (log_fn) log_fn("kv tier: the SSD directory is in use by another WHIRL server; SSD tier off for this one");
                return;
            }
        } else {
            const std::string pid = std::to_string(GetCurrentProcessId());
            DWORD w = 0;
            WriteFile(lh, pid.data(), static_cast<DWORD>(pid.size()), &w, nullptr);
            dir_lock_ = lh;
        }
        ssd_dir_w_ = std::move(dw);
        index();
        io_thread_ = std::thread([this] { ioThread(); });
    }
}

std::uint64_t Tier::pinnedBytes() const {
    std::uint64_t s = 0;
    for (auto len : chunk_len_) s += len;
    return s;
}

void Tier::setWake(std::function<void()> f) { wake_fn_ = std::move(f); }

std::uint64_t Tier::blockAddr(std::uint32_t b) const {
    const std::uint32_t per = static_cast<std::uint32_t>(chunk_bytes / block_bytes);
    return reinterpret_cast<std::uint64_t>(chunks_[b / per]) + static_cast<std::uint64_t>(b % per) * block_bytes;
}

Entry* Tier::find(std::uint64_t id) {
    for (Entry* e : entries)
        if (e->id == id) return e;
    return nullptr;
}

bool Tier::contains(const Entry* e) const { return std::find(entries.begin(), entries.end(), e) != entries.end(); }

std::uint64_t Tier::ramUsedBytes() const {
    return static_cast<std::uint64_t>(n_blocks_ - static_cast<std::uint32_t>(free_blocks_.size())) * block_bytes;
}

std::size_t Tier::countRam() const {
    std::size_t n = 0;
    for (const Entry* e : entries) n += e->in_ram ? 1 : 0;
    return n;
}

void Tier::touch(Entry* e) {
    lru_seq_ += 1;
    e->lru = lru_seq_;
    touchOwners(e);
}

Event Tier::newEvent() {
    if (!ev_pool_.empty()) {
        Event ev = ev_pool_.back();
        ev_pool_.pop_back();
        return ev;
    }
    return ops_.eventCreate();
}

void Tier::poolEvent(Event ev) {
    try {
        ev_pool_.push_back(ev);
    } catch (...) {
        ops_.eventDestroy(ev);
    }
}

bool Tier::gpuIdle(Entry* e) {
    if (!e->gpu_busy) return true;
    if (ops_.eventDone(e->ev)) e->gpu_busy = false;
    return !e->gpu_busy;
}

// ---- entries ------------------------------------------------------------------

Entry* Tier::newEntry(std::uint32_t nck) {
    auto e = std::make_unique<Entry>();
    e->id = rng_() | 1;
    e->nck = nck;
    e->in_ram = true;
    e->ev = newEvent();
    entries.push_back(e.get());
    Entry* p = e.release();
    touch(p);
    return p;
}

void Tier::releaseBlocks(Entry* e, std::size_t from_lb) {
    if (from_lb >= e->blocks.size()) return;
    for (std::size_t i = from_lb; i < e->blocks.size(); ++i)
        if (e->blocks[i] != no_block) free_blocks_.push_back(e->blocks[i]);
    e->blocks.resize(from_lb);
    e->dirty.resize(from_lb);
}

void Tier::removeEntry(Entry* e) {
    if (e->ssd) {
        for (Entry* d : dependentsOf(e)) dropSsdCopy(d);
        if (!contains(e)) return;
    }
    releaseBlocks(e, 0);
    if (e->ssd) deleteFile(e->id, e->ssd->alloc_bytes);
    e->ssd.reset();
    gc_due_ = true;
    if (e->ev != nullptr) poolEvent(e->ev);
    e->ev = nullptr;
    auto it = std::find(entries.begin(), entries.end(), e);
    if (it != entries.end()) entries.erase(it);
    delete e;
}

// Drop the RAM copy: falls back to the SSD snapshot if there is one, else the
// entry is gone. Caller checked evictable().
void Tier::dropRam(Entry* e) {
    stats.ram_evictions += 1;
    if (e->ssd) {
        releaseBlocks(e, 0);
        e->in_ram = false;
        e->owner.reset();
        const SsdSnap& sn = *e->ssd;
        e->n_tok = sn.n_tok;
        e->nck = sn.nck;
        e->pages = sn.pages;
        e->ck = sn.ck;
        try {
            e->tokens = sn.tokens;
        } catch (...) {
            removeEntry(e);
        }
        return;
    }
    stats.dropped += 1;
    removeEntry(e);
}

bool Tier::evictable(Entry* e, Entry* keep) {
    return e != keep && e->in_ram && e->pin == 0 && !e->io_busy && gpuIdle(e);
}

// Free RAM until `blocks` arena blocks are free (LRU). false: not possible.
bool Tier::makeRoom(std::size_t blocks, Entry* keep) {
    while (free_blocks_.size() < blocks) {
        Entry* victim = nullptr;
        for (Entry* e : entries) {
            if (!evictable(e, keep)) continue;
            if (victim == nullptr || e->lru < victim->lru) victim = e;
        }
        if (victim == nullptr) return false;
        dropRam(victim);
    }
    return true;
}

bool Tier::ensureBlocks(Entry* e, std::size_t lb0, std::size_t lb1) {
    std::size_t missing = 0;
    for (std::size_t lb = lb0; lb < lb1; ++lb) {
        if (lb >= e->blocks.size() || e->blocks[lb] == no_block) missing += 1;
    }
    if (missing == 0) return true;
    if (missing > n_blocks_) return false;
    if (!makeRoom(missing, e)) return false;
    while (e->blocks.size() < lb1) {
        e->blocks.push_back(no_block);
        e->dirty.push_back(0);
    }
    for (std::size_t lb = lb0; lb < lb1; ++lb) {
        if (e->blocks[lb] == no_block) {
            e->blocks[lb] = free_blocks_.back();
            free_blocks_.pop_back();
        }
    }
    return true;
}

std::array<std::size_t, 2> Tier::lbRange(std::uint64_t off, std::uint64_t len) {
    return {static_cast<std::size_t>(off / block_bytes),
            static_cast<std::size_t>((off + len + block_bytes - 1) / block_bytes)};
}

void Tier::trim(Entry* e) {
    const std::size_t lb_end = static_cast<std::size_t>((e->extent(lay) + block_bytes - 1) / block_bytes);
    releaseBlocks(e, lb_end);
}

void Tier::dropCkBlocks(Entry* e, std::uint32_t k) {
    const auto r = lbRange(static_cast<std::uint64_t>(k) * lay.ckStride(), lay.ck_bytes);
    const std::size_t end = std::min(r[1], e->blocks.size());
    for (std::size_t lb = r[0]; lb < end; ++lb) {
        if (e->blocks[lb] != no_block) {
            free_blocks_.push_back(e->blocks[lb]);
            e->blocks[lb] = no_block;
            e->dirty[lb] = 0;
        }
    }
}

// ---- copies (tier stream) -------------------------------------------------------

std::uint64_t Tier::copyOut(Entry* e, std::span<const Span> spans) {
    std::uint64_t bytes = 0;
    for (const Span& s : spans) {
        std::uint64_t done = 0;
        while (done < s.len) {
            const std::uint64_t off = s.off + done;
            const std::size_t lb = static_cast<std::size_t>(off / block_bytes);
            const std::uint64_t inner = off % block_bytes;
            const std::uint64_t n = std::min(std::min(s.len - done, block_bytes - inner), max_piece);
            ops_.copyAsync(blockAddr(e->blocks[lb]) + inner, s.dev + done, n, stream);
            e->dirty[lb] = 1;
            done += n;
        }
        bytes += s.len;
    }
    return bytes;
}

void Tier::fence(Entry* e, Event extra) {
    ops_.eventRecord(e->ev, stream);
    e->gpu_busy = true;
    if (extra != nullptr) ops_.eventRecord(extra, stream);
}

// ---- lookup ---------------------------------------------------------------------

std::optional<Match> Tier::lookup(std::span<const std::uint32_t> toks) {
    std::optional<Match> best;
    for (Entry* e : entries) {
        if (e->n_tok == 0) continue;
        if (e->loading && !e->prefetching) continue;
        if (e->in_ram && !gpuIdle(e)) continue;  // a spill into it is still running
        if (!e->in_ram && e->io_busy) continue;
        const auto& et = e->tokens;
        std::size_t lcp = 0;
        const std::size_t lim = std::min(et.size(), toks.size());
        while (lcp < lim && et[lcp] == toks[lcp]) lcp += 1;
        std::optional<std::uint32_t> ck;
        for (std::uint32_t k = 0; k < e->nck; ++k) {
            const CkMeta& c = e->ck[k];
            if (c.valid == 0 || c.pos == 0 || c.pos > lcp || c.pos > toks.size()) continue;
            if (c.pos == toks.size() && c.has_logits == 0) continue;
            if (!ck || c.pos > e->ck[*ck].pos) ck = k;
        }
        if (!ck) continue;
        const std::uint32_t k = *ck;
        const std::uint32_t r = e->ck[k].pos;
        if (!best || r > best->reuse || (r == best->reuse && e->in_ram && !best->e->in_ram))
            best = Match{e, r, k, lcp};
    }
    return best;
}

// ---- restore ----------------------------------------------------------------------

Restore* Tier::beginRestore(Entry* e, std::span<const Span> spans, std::size_t n_main_spans) {
    auto r = std::make_unique<Restore>();
    r->entry_id = e->id;
    r->t0 = std::chrono::steady_clock::now();
    r->from_ssd = !e->in_ram || e->prefetching;
    bool job_new = false;
    if (e->prefetching) {
        // an SSD -> RAM prefetch of this entry is running (or just finished):
        // take it over; its blocks arrive in ascending order
        Job* found = nullptr;
        for (std::size_t i = 0; i < jobs_.size(); ++i) {
            Job* j = jobs_[i];
            if (j->kind == JobKind::read && j->entry_id == e->id) {
                found = j;
                jobs_[i] = jobs_.back();  // swapRemove
                jobs_.pop_back();
                break;
            }
        }
        if (found == nullptr) return nullptr;
        found->prefetch.store(false, std::memory_order_release);
        e->prefetching = false;
        r->job = found;
    } else if (!e->in_ram) {
        if (!ssd_dir_w_ || !refsOk(e)) return nullptr;
        // blocks for the whole extent (the entry becomes a RAM entry, clean)
        const std::size_t lb_end = static_cast<std::size_t>((e->extent(lay) + block_bytes - 1) / block_bytes);
        if (!ensureBlocks(e, 0, lb_end)) return nullptr;
        // blocks of invalid checkpoint slots are not needed
        for (std::uint32_t k = 0; k < e->nck; ++k)
            if (e->ck[k].valid == 0) dropCkBlocks(e, k);
        job_new = true;
    } else {
        for (const Span& s : spans) {
            const auto rg = lbRange(s.off, s.len);
            for (std::size_t lb = rg[0]; lb < rg[1]; ++lb) {
                if (lb >= e->blocks.size() || e->blocks[lb] == no_block) return nullptr;
            }
        }
    }
    // pieces in span order: the request's part first (matched checkpoint, KV
    // pages), then the other checkpoints
    for (std::size_t si = 0; si < spans.size(); ++si) {
        const Span& s = spans[si];
        if (si == n_main_spans) r->n_main = r->pieces.size();
        std::uint64_t done = 0;
        while (done < s.len) {
            const std::uint64_t off = s.off + done;
            const std::size_t lb = static_cast<std::size_t>(off / block_bytes);
            const std::uint64_t inner = off % block_bytes;
            const std::uint64_t n = std::min(std::min(s.len - done, block_bytes - inner), max_piece);
            r->pieces.push_back(Piece{s.dev + done, blockAddr(e->blocks[lb]) + inner, static_cast<std::uint32_t>(n),
                                      static_cast<std::uint32_t>(lb), s.ck});
            done += n;
        }
        r->bytes += s.len;
    }
    if (n_main_spans >= spans.size()) r->n_main = r->pieces.size();
    if (job_new) {
        // the file is read in the order the pieces need its blocks (the
        // request's part first), then the rest of the extent
        auto job = std::make_unique<Job>();
        job->kind = JobKind::read;
        job->entry_id = e->id;
        job->path = filePath(e->id);
        const std::size_t nb = e->blocks.size();
        std::vector<std::uint8_t> seen(nb, 0);
        for (const Piece& p : r->pieces) {
            if (seen[p.lb]) continue;
            seen[p.lb] = 1;
            job->lbs.push_back(p.lb);
            job->ptrs.push_back(blockAddr(e->blocks[p.lb]));
        }
        for (std::size_t lb = 0; lb < nb; ++lb) {
            const std::uint32_t b = e->blocks[lb];
            if (b == no_block || seen[lb]) continue;
            job->lbs.push_back(static_cast<std::uint32_t>(lb));
            job->ptrs.push_back(blockAddr(b));
        }
        job->data_off = lay.dataOff();
        readSources(job.get(), e);
        job->fence = newEvent();
        ops_.eventRecord(job->fence, stream);
        e->in_ram = true;
        e->io_busy = true;
        e->loading = true;
        for (auto& d : e->dirty) d = 0;
        r->job = job.release();
        submit(r->job);
    }
    if (r->job != nullptr) {
        r->pos_of_lb.assign(e->blocks.size(), 0xffffffffu);
        const auto& lbs = r->job->lbs;
        for (std::size_t k = 0; k < lbs.size(); ++k) {
            if (lbs[k] < r->pos_of_lb.size()) r->pos_of_lb[lbs[k]] = static_cast<std::uint32_t>(k);
        }
    }
    r->ev = newEvent();
    r->ev_main = newEvent();
    e->pin += 1;
    touch(e);
    Restore* rp = r.release();
    pumpRestore(rp);
    return rp;
}

bool Tier::prefetch(Entry* e) {
    if (e->in_ram || e->io_busy || e->loading || !e->ssd || !ssd_dir_w_ || !refsOk(e)) return false;
    const std::size_t lb_end = static_cast<std::size_t>((e->extent(lay) + block_bytes - 1) / block_bytes);
    if (!ensureBlocks(e, 0, lb_end)) return false;
    for (std::uint32_t k = 0; k < e->nck; ++k)
        if (e->ck[k].valid == 0) dropCkBlocks(e, k);
    auto job = std::make_unique<Job>();
    job->kind = JobKind::read;
    job->entry_id = e->id;
    job->path = filePath(e->id);
    job->t0 = std::chrono::steady_clock::now();
    for (std::size_t lb = 0; lb < e->blocks.size(); ++lb) {
        const std::uint32_t b = e->blocks[lb];
        if (b == no_block) continue;
        job->lbs.push_back(static_cast<std::uint32_t>(lb));
        job->ptrs.push_back(blockAddr(b));
    }
    job->data_off = lay.dataOff();
    readSources(job.get(), e);
    job->fence = newEvent();
    job->prefetch.store(true, std::memory_order_release);
    ops_.eventRecord(job->fence, stream);
    e->in_ram = true;
    e->io_busy = true;
    e->loading = true;
    e->prefetching = true;
    for (auto& d : e->dirty) d = 0;
    touch(e);
    jobs_.push_back(job.get());
    Job* jp = job.release();
    submit(jp);
    return true;
}

bool Tier::pumpRestore(Restore* r) {
    if (r->failed) return true;
    if (find(r->entry_id) == nullptr) {
        r->failed = true;
        return true;
    }
    std::uint32_t done_n = 0xffffffffu;
    if (r->job != nullptr) {
        if (r->job->failed.load(std::memory_order_acquire)) {
            r->failed = true;
            return true;
        }
        done_n = r->job->done.load(std::memory_order_acquire);
    }
    while (r->cursor < r->pieces.size()) {
        if (!r->main_rec && r->cursor == r->n_main) {
            ops_.eventRecord(r->ev_main, stream);
            r->main_rec = true;
        }
        const Piece& p = r->pieces[r->cursor];
        if (r->job != nullptr && (p.lb >= r->pos_of_lb.size() || r->pos_of_lb[p.lb] >= done_n)) break;
        ops_.copyAsync(p.dev, p.host, p.len, stream);
        r->cursor += 1;
    }
    if (!r->main_rec) {
        if (r->cursor < r->n_main) return false;
        ops_.eventRecord(r->ev_main, stream);
        r->main_rec = true;
        return false;
    }
    return ops_.eventDone(r->ev_main);
}

Tier::Verify Tier::verifyPieces(std::span<const Piece> pieces) {
    std::vector<std::uint8_t> scratch(max_piece);
    Verify v;
    for (const Piece& p : pieces) {
        ops_.download(scratch.data(), p.dev, p.len);
        const auto* host = reinterpret_cast<const std::uint8_t*>(p.host);
        if (std::memcmp(scratch.data(), host, p.len) != 0) v.bad += 1;
        v.bytes += p.len;
    }
    return v;
}

Tier::Verify Tier::verifySpans(const Entry* e, std::span<const Span> spans) {
    std::vector<std::uint8_t> scratch(max_piece);
    Verify v;
    for (const Span& s : spans) {
        std::uint64_t done = 0;
        while (done < s.len) {
            const std::uint64_t off = s.off + done;
            const std::size_t lb = static_cast<std::size_t>(off / block_bytes);
            const std::uint64_t inner = off % block_bytes;
            const std::size_t n = static_cast<std::size_t>(std::min(std::min(s.len - done, block_bytes - inner), max_piece));
            ops_.download(scratch.data(), s.dev + done, n);
            const auto* host = reinterpret_cast<const std::uint8_t*>(blockAddr(e->blocks[lb]) + inner);
            if (std::memcmp(scratch.data(), host, n) != 0) v.bad += 1;
            done += n;
        }
        v.bytes += s.len;
    }
    return v;
}

Event Tier::cutTail(Restore* r) {
    pumpRestore(r);
    r->pieces.resize(std::max(r->cursor, r->n_main));
    if (r->ev_cut == nullptr) r->ev_cut = newEvent();
    ops_.eventRecord(r->ev_cut, stream);
    return r->ev_cut;
}

bool Tier::pumpTail(Restore* r) {
    if (r->failed) return true;
    pumpRestore(r);
    if (r->failed) return true;
    if (r->cursor < r->pieces.size()) return false;
    if (r->job != nullptr) {
        if (!r->job->finished.load(std::memory_order_acquire)) return false;
    }
    if (!r->enqueued) {
        Entry* e = find(r->entry_id);
        if (e == nullptr) {
            r->failed = true;
            return true;
        }
        ops_.eventRecord(r->ev, stream);
        ops_.eventRecord(e->ev, stream);
        e->gpu_busy = true;
        r->enqueued = true;
        return false;
    }
    return ops_.eventDone(r->ev);
}

void Tier::endRestore(Restore* r) {
    if (Entry* e = find(r->entry_id)) {
        if (e->pin > 0) e->pin -= 1;
        if (r->job != nullptr) {
            if (r->job->finished.load(std::memory_order_acquire)) {
                e->io_busy = false;
                e->loading = false;
                if (r->job->failed.load(std::memory_order_acquire)) {
                    // the read failed: the file is unusable
                    removeEntry(e);
                } else if (e->ssd_stale) {
                    e->ssd_stale = false;
                    dropSsdCopy(e);
                }
            }
        }
    }
    if (r->job != nullptr) {
        if (r->job->finished.load(std::memory_order_acquire)) {
            freeJob(r->job);
        } else {
            // still running (failed restore): reaped later by tick
            try {
                jobs_.push_back(r->job);
            } catch (...) {
            }
        }
    }
    if (r->ev != nullptr) poolEvent(r->ev);
    if (r->ev_main != nullptr) poolEvent(r->ev_main);
    if (r->ev_cut != nullptr) poolEvent(r->ev_cut);
    delete r;
}

// ---- SSD ----------------------------------------------------------------------------

std::wstring Tier::filePath(std::uint64_t id) const {
    const std::string s = *cfg.ssd_dir + fmt("\\%016llx-%016llx.wkv", static_cast<unsigned long long>(fingerprint_),
                                             static_cast<unsigned long long>(id));
    return utf8ToWide(s);
}

// ---- block dedup bookkeeping ----------------------------------------------------------

std::vector<Entry*> Tier::dependentsOf(const Entry* e) const {
    std::vector<Entry*> v;
    for (Entry* d : entries) {
        if (d == e || !d->ssd) continue;
        for (const BlkRef& r : d->ssd->tbl) {
            if (r.owner == e->id) {
                v.push_back(d);
                break;
            }
        }
    }
    return v;
}

namespace {
bool jobRefs(const Job* j, std::uint64_t id) {
    if (j->kind != JobKind::write || j->finished.load(std::memory_order_acquire)) return false;
    for (const BlkRef& r : j->rows)
        if (r.owner == id) return true;
    return false;
}
}  // namespace

// Its SSD copy can be dropped now (no IO on it or on blocks it owns).
bool Tier::ssdDroppable(const Entry* e, int depth) const {
    if (e->io_busy || e->pin > 0 || e->ext_pin > 0 || depth > 8) return false;
    for (const Job* j : jobs_)
        if (jobRefs(j, e->id)) return false;
    for (const Entry* d : dependentsOf(e))
        if (!ssdDroppable(d, depth + 1)) return false;
    return true;
}

// Drop e's SSD copy (and first every SSD copy referencing it). An SSD-only
// entry is removed; a RAM entry becomes dirty (written again later). An entry
// with IO running is marked stale and dropped when its job completes.
void Tier::dropSsdCopy(Entry* e) {
    if (!e->ssd || !contains(e)) return;
    for (Entry* d : dependentsOf(e)) dropSsdCopy(d);
    if (!contains(e) || !e->ssd) return;
    if (e->io_busy) {
        e->ssd_stale = true;
        return;
    }
    if (!e->in_ram) {
        removeEntry(e);
        return;
    }
    deleteFile(e->id, e->ssd->alloc_bytes);
    e->ssd.reset();
    e->meta_dirty = false;
    for (std::size_t lb = 0; lb < e->blocks.size(); ++lb) e->dirty[lb] = e->blocks[lb] != no_block ? 1 : 0;
    gc_due_ = true;
}

// Owners of blocks e references are at least as recent as e (LRU drops
// referencing files before the files they reference).
void Tier::touchOwners(const Entry* e) {
    if (!e->ssd) return;
    std::uint64_t last = 0;
    for (const BlkRef& r : e->ssd->tbl) {
        if (r.owner == 0 || r.owner == last) continue;
        last = r.owner;
        if (Entry* o = find(r.owner)) o->lru = std::max(o->lru, e->lru);
    }
}

// Entries left with no usable checkpoint (trimmed to the blocks others
// reference) go once nothing references them.
void Tier::gcOwners() {
    gc_due_ = false;
    bool again = true;
    while (again) {
        again = false;
        for (Entry* e : entries) {
            bool ck = false;
            for (std::uint32_t k = 0; k < e->nck; ++k) ck = ck || e->ck[k].valid != 0;
            if (ck && e->n_tok > 0) continue;
            if (e->io_busy || e->pin > 0 || e->ext_pin > 0 || e->loading) continue;
            if (e->in_ram && !gpuIdle(e)) continue;
            if (!dependentsOf(e).empty()) continue;
            bool refd = false;
            for (const Job* j : jobs_) refd = refd || jobRefs(j, e->id);
            if (refd) continue;
            removeEntry(e);
            again = true;
            break;
        }
    }
}

// Superseded session: cut e to tokens [0, keep) -- rounded down to its last
// checkpoint at or below keep, up to the KV pages other files reference.
std::uint64_t Tier::trimTo(Entry* e, std::uint32_t keep) {
    if (!contains(e) || e->pin > 0 || e->io_busy || e->loading || e->ext_pin > 0 || e->n_tok == 0) return 0;
    if (e->in_ram && !gpuIdle(e)) return 0;
    for (const Job* j : jobs_)
        if (jobRefs(j, e->id)) return 0;
    const std::uint32_t kv_lb0 = static_cast<std::uint32_t>(e->kvOff(lay) / block_bytes);
    std::uint64_t need_pages = 0;
    for (const Entry* d : dependentsOf(e)) {
        for (const BlkRef& r : d->ssd->tbl) {
            if (r.owner != e->id || r.owner_lb < kv_lb0) continue;
            const std::uint64_t end = static_cast<std::uint64_t>(r.owner_lb - kv_lb0 + 1) * block_bytes;
            need_pages = std::max(need_pages, (end + lay.page_bytes - 1) / lay.page_bytes);
        }
    }
    std::uint32_t ck_pos = 0;
    for (std::uint32_t k = 0; k < e->nck; ++k)
        if (e->ck[k].valid != 0 && e->ck[k].pos <= keep) ck_pos = std::max(ck_pos, e->ck[k].pos);
    const std::uint32_t n_new =
        std::max<std::uint32_t>(ck_pos, static_cast<std::uint32_t>(std::min<std::uint64_t>(e->n_tok, need_pages * lay.page_tokens)));
    if (n_new >= e->n_tok) return 0;
    const std::uint64_t before = (e->in_ram ? static_cast<std::uint64_t>(e->blocks.size()) * block_bytes : 0) +
                                 (e->ssd ? e->ssd->alloc_bytes : 0);
    stats.superseded += 1;
    if (n_new == 0) {
        removeEntry(e);
        stats.superseded_bytes += before;
        return before;
    }
    const std::uint32_t pt = lay.page_tokens;
    auto cut = [&](std::uint32_t& n_tok, std::uint32_t& pages, std::array<CkMeta, max_ck>& ck, std::uint32_t nck,
                   std::vector<std::uint32_t>& toks, bool ram) {
        for (std::uint32_t k = 0; k < nck; ++k) {
            if (ck[k].valid != 0 && ck[k].pos > n_new) {
                if (ram) dropCkBlocks(e, k);
                ck[k] = CkMeta{};
            }
        }
        n_tok = std::min(n_tok, n_new);
        pages = (n_tok + pt - 1) / pt;
        if (toks.size() > n_tok) toks.resize(n_tok);
    };
    cut(e->n_tok, e->pages, e->ck, e->nck, e->tokens, e->in_ram);
    if (e->in_ram) trim(e);
    e->owner.reset();
    if (e->ssd) {
        SsdSnap& sn = *e->ssd;
        cut(sn.n_tok, sn.pages, sn.ck, sn.nck, sn.tokens, false);
        const std::uint64_t kvo = static_cast<std::uint64_t>(sn.nck) * lay.ckStride();
        const std::size_t n_blk = sn.pages == 0 ? 0 : lbRange(kvo, static_cast<std::uint64_t>(sn.pages) * lay.page_bytes)[1] - kvo / block_bytes;
        if (sn.tbl.size() > n_blk) sn.tbl.resize(n_blk);
        std::size_t n_ref = 0;
        for (const BlkRef& r : sn.tbl) n_ref += r.owner != 0 ? 1 : 0;
        const std::uint64_t fb = alignUp(lay.dataOff() + kvo + static_cast<std::uint64_t>(sn.pages) * lay.page_bytes, sector);
        const std::uint64_t alloc = satSub(fb, static_cast<std::uint64_t>(n_ref) * block_bytes);
        ssd_bytes = satSub(ssd_bytes, satSub(sn.alloc_bytes, alloc));
        sn.alloc_bytes = std::min(sn.alloc_bytes, alloc);
        sn.file_bytes = fb;
        e->meta_dirty = true;
    }
    gc_due_ = true;
    const std::uint64_t after = (e->in_ram ? static_cast<std::uint64_t>(e->blocks.size()) * block_bytes : 0) +
                                (e->ssd ? e->ssd->alloc_bytes : 0);
    stats.superseded_bytes += satSub(before, after);
    return satSub(before, after);
}

std::uint64_t Tier::allocBytes(const Entry* e, std::size_t n_ref) const {
    return satSub(fileBytes(e), static_cast<std::uint64_t>(n_ref) * block_bytes);
}

bool Tier::refsOk(const Entry* e) const {
    if (!e->ssd) return true;
    for (const BlkRef& r : e->ssd->tbl) {
        if (r.owner == 0) continue;
        const Entry* o = nullptr;
        for (const Entry* x : entries)
            if (x->id == r.owner) o = x;
        if (o == nullptr || !o->ssd || o->ssd_stale) return false;
    }
    return true;
}

// Source file of each block of a read job (referenced blocks come from their
// owner's file); pins the owners until the job is freed.
void Tier::readSources(Job* j, const Entry* e) {
    j->paths.assign(1, filePath(e->id));
    j->src.clear();
    j->src_lb.clear();
    std::vector<std::uint64_t> ids{e->id};
    const std::uint32_t lb0 = e->ssd ? static_cast<std::uint32_t>(static_cast<std::uint64_t>(e->ssd->nck) * lay.ckStride() / block_bytes) : 0;
    for (const std::uint32_t lb : j->lbs) {
        std::uint16_t si = 0;
        std::uint32_t sl = lb;
        if (e->ssd && lb >= lb0 && lb - lb0 < e->ssd->tbl.size() && e->ssd->tbl[lb - lb0].owner != 0) {
            const BlkRef& r = e->ssd->tbl[lb - lb0];
            auto it = std::find(ids.begin(), ids.end(), r.owner);
            if (it == ids.end()) {
                ids.push_back(r.owner);
                j->paths.push_back(filePath(r.owner));
                it = ids.end() - 1;
            }
            si = static_cast<std::uint16_t>(it - ids.begin());
            sl = r.owner_lb;
        }
        j->src.push_back(si);
        j->src_lb.push_back(sl);
    }
    for (std::size_t i = 1; i < ids.size(); ++i) {
        if (Entry* o = find(ids[i])) {
            o->ext_pin += 1;
            j->owners.push_back(ids[i]);
        }
    }
}

void Tier::deleteFile(std::uint64_t id, std::uint64_t size) {
    if (!ssd_dir_w_) return;
    std::wstring p;
    try {
        p = filePath(id);
    } catch (...) {
        return;
    }
    DeleteFileW(p.c_str());
    ssd_bytes = satSub(ssd_bytes, size);
    stats.ssd_evictions += 1;
}

std::uint64_t Tier::fileBytes(const Entry* e) const { return alignUp(lay.dataOff() + e->extent(lay), sector); }

// Make room on SSD for `need` more bytes (LRU: foreign files, SSD-only
// entries, SSD copies of RAM entries). false: cannot.
bool Tier::ssdRoom(std::uint64_t need, Entry* keep) {
    while (ssd_bytes + need > cfg.ssd_cap) {
        // oldest foreign file first
        if (!foreign_.empty()) {
            std::size_t oi = 0;
            for (std::size_t i = 0; i < foreign_.size(); ++i)
                if (foreign_[i].last_used < foreign_[oi].last_used) oi = i;
            Foreign f = std::move(foreign_[oi]);
            foreign_.erase(foreign_.begin() + static_cast<std::ptrdiff_t>(oi));
            DeleteFileW(f.path.c_str());
            ssd_bytes = satSub(ssd_bytes, f.size);
            stats.ssd_evictions += 1;
            continue;
        }
        Entry* victim = nullptr;
        for (Entry* e : entries) {
            if (e == keep || !e->ssd || !ssdDroppable(e)) continue;
            if (victim == nullptr || e->lru < victim->lru) victim = e;
        }
        if (victim == nullptr) return false;
        dropSsdCopy(victim);
    }
    return true;
}

void Tier::submit(Job* j) {
    {
        std::lock_guard<std::mutex> lk(q_mutex_);
        try {
            (j->kind == JobKind::read ? q_reads_ : q_writes_).push_back(j);
        } catch (...) {
            j->failed.store(true, std::memory_order_release);
            j->finished.store(true, std::memory_order_release);
            return;
        }
    }
    q_cond_.notify_one();
}

void Tier::freeJob(Job* j) {
    for (std::uint64_t id : j->owners)
        if (Entry* o = find(id))
            if (o->ext_pin > 0) o->ext_pin -= 1;
    j->owners.clear();
    if (j->fence != nullptr) poolEvent(j->fence);
    delete j;
}

void Tier::makeHeader(const Entry* e, bool valid, std::uint8_t* buf) const {
    std::memset(buf, 0, header_bytes);
    Header h{};
    std::memcpy(h.magic, file_magic, sizeof h.magic);
    h.version = file_version;
    h.valid = valid ? 1u : 0u;
    h.fingerprint = fingerprint_;
    h.id = e->id;
    h.n_tok = e->n_tok;
    h.nck = e->nck;
    h.pages = e->pages;
    h._r = 0;
    h.tok_off = header_bytes;
    h.data_off = lay.dataOff();
    h.ck_stride = lay.ckStride();
    h.page_bytes = lay.page_bytes;
    h.last_used = unixNow();
    h.tok_sum = tokSum(e->tokens.data(), e->n_tok);
    for (std::size_t k = 0; k < max_ck; ++k) h.ck[k] = e->ck[k];
    h.tbl_off = header_bytes + lay.tokRegion();
    h.n_blk = e->pages == 0 ? 0u
                            : static_cast<std::uint32_t>(lbRange(e->kvOff(lay), static_cast<std::uint64_t>(e->pages) * lay.page_bytes)[1] -
                                                         e->kvOff(lay) / block_bytes);
    std::memcpy(buf, &h, sizeof h);
}

// Queue an SSD write of e's dirty blocks (main thread; e is idle on the GPU).
// KV blocks that repeat a block already on SSD (same token prefix, same
// content hash) become references instead of writes.
void Tier::startWrite(Entry* e) {
    const std::uint64_t fb = fileBytes(e);
    if (fb > cfg.ssd_cap) return;
    const std::uint64_t kvo = e->kvOff(lay);
    const std::uint32_t kv_lb0 = static_cast<std::uint32_t>(kvo / block_bytes);
    const std::size_t lb_end = static_cast<std::size_t>((e->extent(lay) + block_bytes - 1) / block_bytes);
    const std::size_t n_blk =
        e->pages == 0 ? 0 : lbRange(kvo, static_cast<std::uint64_t>(e->pages) * lay.page_bytes)[1] - kv_lb0;
    auto isDirty = [&](std::size_t lb) { return lb < e->blocks.size() && e->blocks[lb] != no_block && e->dirty[lb] != 0; };
    // blocks others reference must not change under them: wait for reads of
    // e's blocks and for writes referencing e; drop the references to blocks
    // this write changes (or cuts off)
    if (e->ext_pin > 0) return;
    for (const Job* j : jobs_)
        if (jobRefs(j, e->id)) return;
    if (e->ssd) {
        std::vector<Entry*> hit;
        for (Entry* d : dependentsOf(e)) {
            for (const BlkRef& r : d->ssd->tbl) {
                if (r.owner != e->id) continue;
                if (r.owner_lb >= lb_end || isDirty(r.owner_lb)) {
                    if (d->io_busy) return;
                    hit.push_back(d);
                    break;
                }
            }
        }
        for (Entry* d : hit) dropSsdCopy(d);
    }
    const std::uint64_t old0 = e->ssd ? e->ssd->alloc_bytes : 0;
    if (!ssdRoom(satSub(fb, old0), e)) return;
    const std::uint64_t old = e->ssd ? e->ssd->alloc_bytes : 0;  // ssdRoom may have dropped it
    auto j = std::make_unique<Job>();
    j->kind = JobKind::write;
    j->entry_id = e->id;
    j->path = filePath(e->id);
    j->meta_only = !e->hasDirty();
    // block table: unchanged blocks keep their row
    j->kv_lb0 = kv_lb0;
    j->rows.assign(n_blk, BlkRef{});
    j->cand.assign(n_blk, 0);
    if (e->ssd) {
        const auto& ot = e->ssd->tbl;
        for (std::size_t k = 0; k < std::min(n_blk, ot.size()); ++k)
            if (!isDirty(kv_lb0 + k)) j->rows[k] = ot[k];
    }
    // reference candidates: the SSD entry sharing the longest token prefix;
    // only blocks wholly inside the shared full pages
    if (cfg.dedup && n_blk > 0) {
        const Entry* base = nullptr;
        std::size_t best = 0;
        for (const Entry* b : entries) {
            if (b == e || !b->ssd || b->io_busy || b->ssd_stale || b->ssd->tbl.empty()) continue;
            const auto& bt = b->ssd->tokens;
            const std::size_t lim = std::min<std::size_t>(bt.size(), e->n_tok);
            std::size_t l = 0;
            while (l < lim && bt[l] == e->tokens[l]) l += 1;
            if (l > best) {
                best = l;
                base = b;
            }
        }
        if (base != nullptr && lay.page_tokens > 0) {
            const std::uint64_t shared = static_cast<std::uint64_t>(best / lay.page_tokens) * lay.page_bytes;
            const std::size_t ncand = std::min<std::size_t>({static_cast<std::size_t>(shared / block_bytes), n_blk, base->ssd->tbl.size()});
            const std::uint32_t b_lb0 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(base->ssd->nck) * lay.ckStride() / block_bytes);
            for (std::size_t k = 0; k < ncand; ++k) {
                if (!isDirty(kv_lb0 + k)) continue;
                BlkRef r = base->ssd->tbl[k];
                if (r.h0 == 0 && r.h1 == 0) continue;
                if (r.owner == 0) {
                    r.owner = base->id;
                    r.owner_lb = b_lb0 + static_cast<std::uint32_t>(k);
                }
                if (r.owner == e->id) continue;
                const Entry* o = r.owner == base->id ? base : find(r.owner);
                if (o == nullptr || !o->ssd || o->io_busy || o->ssd_stale) continue;
                j->rows[k] = r;
                j->cand[k] = 1;
            }
        }
    }
    j->hdr0 = alignedAlloc(header_bytes);
    j->hdr1 = alignedAlloc(header_bytes);
    makeHeader(e, false, j->hdr0.get());
    makeHeader(e, true, j->hdr1.get());
    const std::uint64_t tb = alignUp(static_cast<std::uint64_t>(e->n_tok) * 4, sector);
    j->toks_len = std::max(tb, sector);
    j->toks = alignedAlloc(static_cast<std::size_t>(j->toks_len));
    std::memset(j->toks.get(), 0, static_cast<std::size_t>(j->toks_len));
    std::memcpy(j->toks.get(), e->tokens.data(), static_cast<std::size_t>(e->n_tok) * 4);
    j->tok_off = header_bytes;
    j->tbl_off = header_bytes + lay.tokRegion();
    j->data_off = lay.dataOff();
    j->file_bytes = fb;
    for (std::size_t lb = 0; lb < e->blocks.size(); ++lb) {
        if (!isDirty(lb)) continue;
        j->lbs.push_back(static_cast<std::uint32_t>(lb));
        j->ptrs.push_back(blockAddr(e->blocks[lb]));
    }
    SsdSnap sn;
    sn.n_tok = e->n_tok;
    sn.nck = e->nck;
    sn.pages = e->pages;
    sn.ck = e->ck;
    sn.tokens.assign(e->tokens.begin(), e->tokens.begin() + e->n_tok);
    sn.file_bytes = fb;
    j->snap = std::move(sn);
    e->io_busy = true;
    e->meta_dirty = false;
    j->t0 = std::chrono::steady_clock::now();
    j->reserved = fb;
    ssd_bytes = ssd_bytes - old + fb;
    jobs_.push_back(j.get());
    Job* jp = j.release();
    submit(jp);
}

bool Tier::tick() {
    bool busy = false;
    // finished jobs
    std::size_t i = 0;
    while (i < jobs_.size()) {
        Job* j = jobs_[i];
        if (!j->finished.load(std::memory_order_acquire)) {
            busy = true;
            i += 1;
            continue;
        }
        jobs_[i] = jobs_.back();  // swapRemove
        jobs_.pop_back();
        bool failed = j->failed.load(std::memory_order_acquire);
        if (j->kind == JobKind::read) {
            if (Entry* e = find(j->entry_id)) {
                e->io_busy = false;
                e->loading = false;
                const bool pf = j->prefetch.load(std::memory_order_acquire) && e->prefetching;
                e->prefetching = false;
                if (pf && !failed) {
                    // prefetched: a clean RAM entry now
                    if (log_fn) {
                        log_fn(fmt("kv tier: entry %016llx (%u tok) prefetched from SSD into RAM in %.0f ms",
                                   static_cast<unsigned long long>(e->id), e->n_tok, msSince(j->t0)));
                    }
                } else if (failed || e->pin == 0) {
                    // abandoned or failed read: the RAM blocks are incomplete
                    if (e->ssd) {
                        releaseBlocks(e, 0);
                        e->in_ram = false;
                    } else {
                        removeEntry(e);
                    }
                }
                if (contains(e) && e->ssd_stale && !e->io_busy) {
                    e->ssd_stale = false;
                    dropSsdCopy(e);
                }
            }
        } else if (j->kind == JobKind::write) {
            stats.ssd_write_bytes += j->written;
            if (Entry* e = find(j->entry_id)) {
                e->io_busy = false;
                // references still valid? (an owner dropped or rewritten meanwhile)
                bool refs_ok = !e->ssd_stale;
                for (std::size_t k = 0; refs_ok && k < j->rows.size(); ++k) {
                    const BlkRef& r = j->rows[k];
                    if (r.owner == 0) continue;
                    const Entry* o = find(r.owner);
                    if (o == nullptr || !o->ssd || o->ssd_stale) {
                        refs_ok = false;
                        break;
                    }
                    const std::uint32_t o_lb0 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(o->ssd->nck) * lay.ckStride() / block_bytes);
                    const std::size_t ok_ = r.owner_lb >= o_lb0 ? r.owner_lb - o_lb0 : o->ssd->tbl.size();
                    if (ok_ >= o->ssd->tbl.size() || o->ssd->tbl[ok_].owner != 0 || o->ssd->tbl[ok_].h0 != r.h0 ||
                        o->ssd->tbl[ok_].h1 != r.h1)
                        refs_ok = false;
                }
                if (!failed && !refs_ok) {
                    DeleteFileW(j->path.c_str());
                    failed = true;
                }
                e->ssd_stale = false;
                if (failed) {
                    if (log_fn) log_fn("kv tier: SSD write failed or superseded; the entry stays in RAM only");
                    ssd_bytes = satSub(ssd_bytes, j->reserved);
                    if (e->ssd)
                        for (Entry* d : dependentsOf(e)) dropSsdCopy(d);
                    e->ssd.reset();
                    e->meta_dirty = false;
                    for (std::size_t lb = 0; lb < e->blocks.size(); ++lb)
                        e->dirty[lb] = e->blocks[lb] != no_block ? 1 : 0;
                    if (!e->in_ram) removeEntry(e);
                    gc_due_ = true;
                } else {
                    std::size_t n_ref = 0;
                    for (const BlkRef& r : j->rows) n_ref += r.owner != 0 ? 1 : 0;
                    j->snap->tbl = std::move(j->rows);
                    const std::uint64_t alloc = allocBytes(e, n_ref);
                    j->snap->alloc_bytes = alloc;
                    ssd_bytes = satSub(ssd_bytes, j->reserved) + alloc;
                    e->ssd = std::move(j->snap);
                    j->snap.reset();
                    for (auto lb : j->lbs) {
                        if (lb < e->dirty.size()) e->dirty[lb] = 0;
                    }
                    stats.ssd_writes += 1;
                    stats.ssd_dedup_bytes += j->dedup;
                    touchOwners(e);
                    if (log_fn && !j->meta_only) {
                        log_fn(fmt("kv tier: entry %016llx (%u tok) written to SSD: %.0f MiB in %.0f ms (file %.0f MiB, %.0f MiB "
                                   "shared with other entries; SSD written in all %.2f GiB)",
                                   static_cast<unsigned long long>(e->id), e->n_tok, static_cast<double>(j->written) / 1048576.0,
                                   msSince(j->t0), static_cast<double>(alloc) / 1048576.0,
                                   static_cast<double>(n_ref * block_bytes) / 1048576.0,
                                   static_cast<double>(stats.ssd_write_bytes) / 1073741824.0));
                    }
                }
            }
        }
        freeJob(j);
    }
    if (ssd_dir_w_) {
        const auto now = std::chrono::steady_clock::now();
        // startWrite may evict SSD-only entries (ssdRoom): iterate over a copy
        // and skip entries removed meanwhile
        const std::vector<Entry*> snapshot = entries;
        const bool writes_on = !cfg.ssd_shutdown_only || flushing;
        for (Entry* e : snapshot) {
            if (!contains(e)) continue;
            if (e->meta_dirty && e->ssd && !e->io_busy && e->pin == 0) {
                // trimmed (superseded session): header / table rewrite, file cut
                try {
                    startWrite(e);
                } catch (...) {
                }
                busy = busy || e->io_busy;  // deferred: retried on a later tick
                continue;
            }
            if (!writes_on) continue;
            if (!e->in_ram || e->io_busy || e->pin > 0 || e->n_tok == 0) continue;
            if (!gpuIdle(e)) {
                busy = true;
                continue;
            }
            if (!e->hasDirty() && e->ssd) continue;
            const auto age = std::chrono::duration_cast<std::chrono::milliseconds>(now - e->t_mod).count();
            const std::uint64_t age_ms = age > 0 ? static_cast<std::uint64_t>(age) : 0;
            if (age_ms < cfg.ssd_delay_ms) {
                busy = true;
                continue;
            }
            try {
                startWrite(e);
            } catch (...) {
            }
            busy = busy || e->io_busy;
        }
    } else {
        for (Entry* e : entries) {
            if (!gpuIdle(e)) busy = true;
        }
    }
    if (gc_due_) gcOwners();
    return busy;
}

// ---- startup index ----------------------------------------------------------------

void Tier::indexFile(const std::wstring& pw, std::uint64_t size, std::vector<IndexItem>& order) {
    bool del = false;
    ScopeExit del_guard{[&] {
        if (del) DeleteFileW(pw.c_str());
    }};
    HANDLE fh = CreateFileW(pw.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return;
    ScopeExit close_guard{[&] { CloseHandle(fh); }};
    try {
        alignas(16) std::uint8_t hdr_buf[header_bytes];
        if (!readAt(fh, hdr_buf, header_bytes, 0)) return;
        Header hd;
        std::memcpy(&hd, hdr_buf, sizeof hd);
        if (std::memcmp(hd.magic, file_magic, sizeof hd.magic) != 0) return;
        // allocated bytes (sparse files: shared blocks are holes)
        {
            DWORD hi = 0;
            const DWORD lo = GetCompressedFileSizeW(pw.c_str(), &hi);
            if (lo != INVALID_FILE_SIZE || GetLastError() == NO_ERROR)
                size = std::min(size, (static_cast<std::uint64_t>(hi) << 32) | lo);
        }
        if (hd.version != file_version || hd.fingerprint != fingerprint_) {
            // another build / format: counted against the cap, dropped first
            foreign_.push_back(Foreign{pw, size, hd.last_used});
            ssd_bytes += size;
            return;
        }
        if (hd.valid == 0 || hd.n_tok == 0 || hd.n_tok > lay.tok_cap || hd.nck > max_ck || hd.ck_stride != lay.ckStride() ||
            hd.page_bytes != lay.page_bytes || hd.data_off != lay.dataOff()) {
            del = true;
            return;
        }
        std::vector<std::uint32_t> toks(hd.n_tok);
        if (!readAt(fh, reinterpret_cast<std::uint8_t*>(toks.data()), static_cast<std::uint64_t>(hd.n_tok) * 4, hd.tok_off))
            return;
        if (tokSum(toks.data(), toks.size()) != hd.tok_sum) {
            del = true;
            return;
        }
        const std::uint64_t kv_off = static_cast<std::uint64_t>(hd.nck) * lay.ckStride();
        const std::size_t n_blk =
            hd.pages == 0 ? 0 : lbRange(kv_off, static_cast<std::uint64_t>(hd.pages) * lay.page_bytes)[1] - kv_off / block_bytes;
        if (hd.n_blk != n_blk || n_blk > lay.maxKvBlocks() || hd.tbl_off != header_bytes + lay.tokRegion()) {
            del = true;
            return;
        }
        std::vector<BlkRef> tbl(n_blk);
        if (n_blk > 0 && !readAt(fh, reinterpret_cast<std::uint8_t*>(tbl.data()), n_blk * sizeof(BlkRef), hd.tbl_off)) return;
        auto e = std::make_unique<Entry>();
        e->id = hd.id;
        e->nck = hd.nck;
        e->n_tok = hd.n_tok;
        e->pages = hd.pages;
        for (std::size_t k = 0; k < max_ck; ++k) e->ck[k] = hd.ck[k];
        e->in_ram = false;
        e->tokens = toks;
        try {
            e->ev = newEvent();
        } catch (...) {
            e->ev = nullptr;
        }
        SsdSnap sn;
        sn.n_tok = hd.n_tok;
        sn.nck = hd.nck;
        sn.pages = hd.pages;
        sn.ck = e->ck;
        sn.tokens = std::move(toks);
        sn.file_bytes = size;
        sn.alloc_bytes = size;
        sn.tbl = std::move(tbl);
        e->ssd = std::move(sn);
        entries.push_back(e.get());
        Entry* ep = e.release();
        try {
            order.push_back(IndexItem{ep, hd.last_used});
        } catch (...) {
        }
        ssd_bytes += size;
    } catch (const std::bad_alloc&) {
        return;
    }
}

void Tier::index() {
    const std::wstring& dir = *ssd_dir_w_;
    const std::wstring pat = dir + L"\\*.wkv";
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW(pat.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    ScopeExit close_guard{[&] { FindClose(h); }};
    std::vector<IndexItem> order;
    while (true) {
        const std::wstring full = dir + L"\\" + fd.cFileName;
        const std::uint64_t size = (static_cast<std::uint64_t>(fd.nFileSizeHigh) << 32) | fd.nFileSizeLow;
        indexFile(full, size, order);
        if (FindNextFileW(h, &fd) == 0) break;
    }
    validateRefs();
    order.erase(std::remove_if(order.begin(), order.end(), [&](const IndexItem& o) { return !contains(o.e); }), order.end());
    // LRU order of the indexed entries = their last write time
    std::stable_sort(order.begin(), order.end(), [](const IndexItem& a, const IndexItem& b) { return a.t < b.t; });
    for (const IndexItem& o : order) touch(o.e);
}

// Startup: a file referencing a block that is not (or no longer) in its
// owner's file with the same hash is dropped, until nothing changes.
void Tier::validateRefs() {
    bool again = true;
    while (again) {
        again = false;
        for (Entry* e : entries) {
            if (!e->ssd) continue;
            bool ok = true;
            for (const BlkRef& r : e->ssd->tbl) {
                if (r.owner == 0) continue;
                const Entry* o = find(r.owner);
                if (o == nullptr || o == e || !o->ssd) {
                    ok = false;
                    break;
                }
                const std::uint32_t o_lb0 = static_cast<std::uint32_t>(static_cast<std::uint64_t>(o->ssd->nck) * lay.ckStride() / block_bytes);
                const std::size_t k = r.owner_lb >= o_lb0 ? r.owner_lb - o_lb0 : o->ssd->tbl.size();
                if (k >= o->ssd->tbl.size() || o->ssd->tbl[k].owner != 0 || o->ssd->tbl[k].h0 != r.h0 || o->ssd->tbl[k].h1 != r.h1) {
                    ok = false;
                    break;
                }
            }
            if (ok) continue;
            if (log_fn)
                log_fn(fmt("kv tier: SSD entry %016llx references a block no longer on SSD; dropped",
                           static_cast<unsigned long long>(e->id)));
            removeEntry(e);  // also drops entries referencing its blocks
            again = true;
            break;
        }
    }
}

std::size_t Tier::ssdEntries() const {
    std::size_t n = 0;
    for (const Entry* e : entries) n += e->ssd ? 1 : 0;
    return n;
}

// ---- IO thread ----------------------------------------------------------------------

void Tier::ioThread() {
    try {
        ops_.setDevice(dev_);
    } catch (...) {
    }
    while (true) {
        Job* j = nullptr;
        {
            std::unique_lock<std::mutex> lk(q_mutex_);
            q_cond_.wait(lk, [&] { return stop_ || !q_reads_.empty() || !q_writes_.empty(); });
            if (stop_) return;
            std::size_t ri = 0;
            for (std::size_t i = 0; i < q_reads_.size(); ++i) {
                if (!q_reads_[i]->prefetch.load(std::memory_order_acquire)) {
                    ri = i;
                    break;
                }
            }
            if (!q_reads_.empty()) {
                j = q_reads_[ri];
                q_reads_.erase(q_reads_.begin() + static_cast<std::ptrdiff_t>(ri));
            } else {
                j = q_writes_.front();
                q_writes_.erase(q_writes_.begin());
            }
        }
        runJob(j);
        if (wake_fn_) wake_fn_();
    }
}

// A read job waiting while a write runs (served between the write's blocks).
Job* Tier::pendingRead() {
    std::lock_guard<std::mutex> lk(q_mutex_);
    if (q_reads_.empty()) return nullptr;
    Job* j = q_reads_.front();
    q_reads_.erase(q_reads_.begin());
    return j;
}

// A restore's read waiting while a prefetch read runs.
Job* Tier::pendingRestoreRead() {
    std::lock_guard<std::mutex> lk(q_mutex_);
    for (std::size_t i = 0; i < q_reads_.size(); ++i) {
        if (!q_reads_[i]->prefetch.load(std::memory_order_acquire)) {
            Job* j = q_reads_[i];
            q_reads_.erase(q_reads_.begin() + static_cast<std::ptrdiff_t>(i));
            return j;
        }
    }
    return nullptr;
}

void Tier::runJob(Job* j) {
    switch (j->kind) {
    case JobKind::read: {
        if (j->fence != nullptr) {
            try {
                ops_.eventSync(j->fence);
            } catch (...) {
            }
        }
        if (j->paths.empty()) j->paths.push_back(j->path);
        std::vector<HANDLE> fhs;
        bool open_ok = true;
        for (const std::wstring& pth : j->paths) {
            HANDLE h = CreateFileW(pth.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
                                   FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
            if (h == INVALID_HANDLE_VALUE) {
                open_ok = false;
                break;
            }
            fhs.push_back(h);
        }
        auto fhOf = [&](std::size_t i) { return fhs[i < j->src.size() ? j->src[i] : 0]; };
        auto lbOf = [&](std::size_t i) { return i < j->src_lb.size() ? j->src_lb[i] : j->lbs[i]; };
        if (!open_ok) {
            for (HANDLE h : fhs) CloseHandle(h);
            j->failed.store(true, std::memory_order_release);
        } else {
            // read_qd reads in flight, completed (and reported) in order
            OVERLAPPED ov[read_qd] = {};
            HANDLE evs[read_qd] = {};
            bool eof[read_qd] = {};
            bool ok = true;
            for (auto& h : evs) {
                h = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                if (h == nullptr) ok = false;
            }
            const std::size_t n_lbs = j->lbs.size();
            std::size_t issued = 0;
            std::size_t done = 0;
            while (ok && done < n_lbs) {
                while (issued < n_lbs && issued - done < read_qd) {
                    const std::size_t k = issued % read_qd;
                    const std::uint64_t off = j->data_off + static_cast<std::uint64_t>(lbOf(issued)) * block_bytes;
                    ov[k] = ovAt(off);
                    ov[k].hEvent = evs[k];
                    eof[k] = false;
                    ResetEvent(evs[k]);
                    if (ReadFile(fhOf(issued), reinterpret_cast<void*>(j->ptrs[issued]), static_cast<DWORD>(block_bytes), nullptr, &ov[k]) == 0) {
                        const DWORD err = GetLastError();
                        if (err == ERROR_HANDLE_EOF) {
                            eof[k] = true;  // past the end of the file
                        } else if (err != ERROR_IO_PENDING) {
                            ok = false;
                            break;
                        }
                    }
                    issued += 1;
                }
                if (!ok) break;
                const std::size_t k = done % read_qd;
                if (!eof[k]) {
                    DWORD r = 0;
                    if (GetOverlappedResult(fhOf(done), &ov[k], &r, TRUE) == 0 && GetLastError() != ERROR_HANDLE_EOF) {
                        ok = false;
                        done += 1;
                        break;
                    }
                }
                done += 1;
                j->done.store(static_cast<std::uint32_t>(done), std::memory_order_release);
                // a prefetch yields to a restore's read
                if (j->prefetch.load(std::memory_order_acquire)) {
                    if (Job* rj = pendingRestoreRead()) runJob(rj);
                }
            }
            // reads still in flight (failure): let them land before the buffers go
            for (; done < issued; ++done) {
                const std::size_t k = done % read_qd;
                if (eof[k]) continue;
                DWORD r = 0;
                GetOverlappedResult(fhOf(done), &ov[k], &r, TRUE);
            }
            if (!ok) j->failed.store(true, std::memory_order_release);
            for (HANDLE h : evs)
                if (h != nullptr) CloseHandle(h);
            for (HANDLE h : fhs) CloseHandle(h);
        }
        j->finished.store(true, std::memory_order_release);
        break;
    }
    case JobKind::write: {
        // sparse: blocks stored in another file (dedup) are holes here
        HANDLE fh = CreateFileW(j->path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr, OPEN_ALWAYS,
                                FILE_FLAG_NO_BUFFERING, nullptr);
        bool ok = fh != INVALID_HANDLE_VALUE;
        if (ok) {
            DWORD br = 0;
            DeviceIoControl(fh, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &br, nullptr);
            ok = writeAt(fh, j->hdr0.get(), header_bytes, 0) && writeAt(fh, j->toks.get(), j->toks_len, j->tok_off);
            j->written += header_bytes + j->toks_len;
            if (ok) {
                for (std::size_t i = 0; i < j->lbs.size(); ++i) {
                    if (Job* rj = pendingRead()) runJob(rj);
                    const std::uint32_t lb = j->lbs[i];
                    const std::uint64_t off = j->data_off + static_cast<std::uint64_t>(lb) * block_bytes;
                    const std::uint64_t n = std::min(block_bytes, satSub(j->file_bytes, off));
                    if (n == 0) continue;
                    const auto* buf = reinterpret_cast<const std::uint8_t*>(j->ptrs[i]);
                    if (lb >= j->kv_lb0 && lb - j->kv_lb0 < j->rows.size()) {
                        // KV block: hash it; a candidate reference with the same hash is kept
                        const std::size_t k = lb - j->kv_lb0;
                        const auto h = blockHash(buf, n);
                        if (j->cand[k] != 0 && n == block_bytes && j->rows[k].h0 == h[0] && j->rows[k].h1 == h[1]) {
                            j->dedup += n;
                            continue;
                        }
                        j->rows[k] = BlkRef{0, 0, 0, h[0], h[1]};
                    }
                    if (!writeAt(fh, buf, alignUp(n, sector), off)) {
                        ok = false;
                        break;
                    }
                    j->written += alignUp(n, sector);
                }
            }
            if (ok) {
                const std::uint64_t tl = std::max(alignUp(j->rows.size() * sizeof(BlkRef), sector), sector);
                j->tbl = alignedAlloc(static_cast<std::size_t>(tl));
                std::memset(j->tbl.get(), 0, static_cast<std::size_t>(tl));
                if (!j->rows.empty()) std::memcpy(j->tbl.get(), j->rows.data(), j->rows.size() * sizeof(BlkRef));
                ok = writeAt(fh, j->tbl.get(), tl, j->tbl_off);
                j->written += tl;
            }
            ok = ok && FlushFileBuffers(fh) != 0;
            ok = ok && writeAt(fh, j->hdr1.get(), header_bytes, 0);
            j->written += header_bytes;
            if (ok) {
                LARGE_INTEGER dist;
                dist.QuadPart = static_cast<LONGLONG>(j->file_bytes);
                LARGE_INTEGER np;
                if (SetFilePointerEx(fh, dist, &np, FILE_BEGIN) != 0) SetEndOfFile(fh);
                ok = FlushFileBuffers(fh) != 0;
            }
            CloseHandle(fh);
        }
        if (!ok) {
            DeleteFileW(j->path.c_str());
            j->failed.store(true, std::memory_order_release);
        }
        j->finished.store(true, std::memory_order_release);
        break;
    }
    }
}

// ---- process memory declaration -------------------------------------------------------

void declarePinned(const char* local_app_data, std::uint64_t bytes) {
    if (local_app_data == nullptr) return;
    try {
        const std::filesystem::path dir =
            std::filesystem::path(utf8ToWide(local_app_data)) / L"whirl" / L"pinned";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        if (ec) return;
        const std::filesystem::path path = dir / (std::to_wstring(GetCurrentProcessId()) + L".txt");
        FILE* f = _wfopen(path.c_str(), L"wb");
        if (f == nullptr) return;
        const std::string line = fmt("pinned_mib %llu\n", static_cast<unsigned long long>(bytes >> 20));
        std::fwrite(line.data(), 1, line.size(), f);
        std::fclose(f);
    } catch (...) {
    }
}

std::uint64_t hashStr(std::uint64_t h0, std::string_view s) {
    std::uint64_t h = h0;
    for (char c : s) {
        h ^= static_cast<std::uint8_t>(c);
        h *= 0x100000001b3ull;
    }
    return h;
}

std::uint64_t exeIdentity() {
    wchar_t buf[1024];
    const DWORD n = GetModuleFileNameW(nullptr, buf, 1024 - 1);
    if (n == 0) return 0;
    buf[n] = 0;
    WIN32_FILE_ATTRIBUTE_DATA info;
    if (GetFileAttributesExW(buf, GetFileExInfoStandard, &info) == 0) return 0;
    const std::uint64_t size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | info.nFileSizeLow;
    const std::uint64_t wt =
        (static_cast<std::uint64_t>(info.ftLastWriteTime.dwHighDateTime) << 32) | info.ftLastWriteTime.dwLowDateTime;
    return size * 0x9e3779b97f4a7c15ull ^ wt;
}

}  // namespace whirl::tier
