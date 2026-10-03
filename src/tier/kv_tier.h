// Host tiers for the server's prefix cache (VRAM -> pinned RAM -> SSD).
// SPDX-License-Identifier: Apache-2.0
//
// An entry is a snapshot of one idle slot: its token ids up to its last
// checkpoint P, its prefix-cache checkpoints (DeltaNet conv / ssm state, MTP
// hidden row, logits row) and the KV rows of positions 0 .. P-1 (whole pages,
// every attention layer and the MTP layer, every array of the KV format).
// Restoring copies the very bytes back, so a restored slot is bit-identical
// to the slot that never left VRAM.
//
// Layout of an entry (the same in RAM and in the file's data region):
// checkpoint k at k * ck_stride, KV page j at nck * ck_stride + j * page_bytes.
// RAM: a pinned host arena of 2 MiB blocks (allocated once at create, in
// 512 MiB chunks); an entry maps its logical 2 MiB blocks to arena blocks.
// All GPU <-> host copies run on one non-blocking stream (FIFO), in pieces of
// at most 1 MiB so that the decode loop's small synchronous readbacks are
// never queued behind a long transfer.
// SSD: one file per entry (4 KiB header, token ids, data region), written by
// an IO thread with unbuffered positional writes (dirty blocks only; the
// header is invalidated first and rewritten last), read back the same way
// (read_qd overlapped reads in flight).
//
// Threading: the main (engine) thread owns all entry bookkeeping and calls
// every Tier method; the IO thread only moves file bytes and reports progress
// through atomics. The wake callback (setWake) is invoked from the IO thread
// after each job; set it before start().
//
// All device memory / stream / event / pinned-host operations go through
// DeviceOps (device_ops.h); this file never includes HIP headers.

#pragma once

#include "device_ops.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

namespace whirl::tier {

constexpr std::uint64_t block_bytes = 2ull << 20;
constexpr std::uint64_t max_piece = 1ull << 20;
constexpr std::size_t max_ck = 8;
// SSD reads in flight per read job
constexpr std::size_t read_qd = 4;
// pinned arena allocation unit (one huge pinned allocation can be refused)
constexpr std::uint64_t chunk_bytes = 512ull << 20;
constexpr std::uint32_t no_block = 0xffffffffu;
constexpr char file_magic[8] = {'W', 'H', 'K', 'V', 'T', 'I', 'E', 'R'};
constexpr std::uint32_t file_version = 1;
constexpr std::uint64_t header_bytes = 4096;
constexpr std::uint64_t sector = 4096;

struct TierError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct CkMeta {
    std::uint32_t pos = 0;
    std::uint8_t valid = 0, has_logits = 0, kind = 0, _p = 0;
    std::uint64_t seq = 0;
};
static_assert(sizeof(CkMeta) == 16);

// On-disk file header (first bytes of the 4 KiB header sector).
struct Header {
    char magic[8];
    std::uint32_t version;
    std::uint32_t valid;
    std::uint64_t fingerprint;
    std::uint64_t id;
    std::uint32_t n_tok;
    std::uint32_t nck;
    std::uint32_t pages;
    std::uint32_t _r;
    std::uint64_t tok_off;
    std::uint64_t data_off;
    std::uint64_t ck_stride;
    std::uint64_t page_bytes;
    std::int64_t last_used;
    std::uint64_t tok_sum;
    CkMeta ck[max_ck];
};
static_assert(sizeof(Header) == 224);
static_assert(std::is_standard_layout_v<Header>);

// Layout constants of this model / server (bytes).
struct Layout {
    std::uint64_t ck_bytes = 0, page_bytes = 0;
    std::uint32_t page_tokens = 0;
    std::uint32_t tok_cap = 0;  // max tokens per entry (context per slot)

    std::uint64_t ckStride() const;
    std::uint64_t tokRegion() const;
    std::uint64_t dataOff() const;
};

struct Config {
    std::uint64_t ram_bytes = 0;
    std::optional<std::string> ssd_dir;  // UTF-8; nullopt: no SSD tier
    std::uint64_t ssd_cap = 0;
    // entries shorter than this are not kept (re-prefill is cheap)
    std::uint32_t min_tokens = 2048;
    // an entry is written to SSD once unchanged for this long
    std::uint64_t ssd_delay_ms = 2000;
};

// Meta of the entry's copy on SSD (a RAM eviction of a newer, dirty entry
// falls back to it; the file stays self-consistent until the next write).
struct SsdSnap {
    std::uint32_t n_tok = 0, nck = 0, pages = 0;
    std::array<CkMeta, max_ck> ck{};
    std::vector<std::uint32_t> tokens;
    std::uint64_t file_bytes = 0;
};

struct Entry {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> tokens;
    std::uint32_t n_tok = 0, nck = 0;
    std::array<CkMeta, max_ck> ck{};
    std::uint32_t pages = 0;  // KV pages holding data (ceil(n_tok / page_tokens))
    // logical 2 MiB block -> arena block (no_block: not in RAM)
    std::vector<std::uint32_t> blocks;
    std::vector<std::uint8_t> dirty;
    bool in_ram = false;
    std::optional<SsdSnap> ssd;
    Event ev = nullptr;  // last tier-stream operation on this entry (spill / restore)
    bool gpu_busy = false;
    // an SSD write or read job holds the entry (no RAM eviction / in-place update)
    bool io_busy = false;
    std::uint32_t pin = 0;  // restores in progress reading it
    // being read from SSD into RAM (not usable for another restore yet)
    bool loading = false;
    // the read is a prefetch no restore owns yet (a restore may take it over)
    bool prefetching = false;
    // slot whose VRAM content extends this entry (in-place updates)
    std::optional<std::uint32_t> owner;
    std::uint64_t lru = 0;
    std::chrono::steady_clock::time_point t_mod{};
    bool removed = false;

    std::uint64_t kvOff(const Layout& l) const;
    std::uint64_t extent(const Layout& l) const;
    bool hasDirty() const;
};

// (device address, logical offset in the entry, bytes)
struct Span {
    DevPtr dev = 0;
    std::uint64_t off = 0, len = 0;
    std::uint8_t ck = 0xff;
};
struct Piece {
    DevPtr dev = 0;
    std::uint64_t host = 0;
    std::uint32_t len = 0, lb = 0;
    std::uint8_t ck = 0xff;
};

struct Match {
    Entry* e = nullptr;
    std::uint32_t reuse = 0;
    std::uint32_t ck = 0;
    std::size_t lcp = 0;
};

struct Job;  // IO job (defined in kv_tier.cpp)

struct Restore {
    std::uint64_t entry_id = 0;
    std::vector<Piece> pieces;
    std::size_t cursor = 0;
    // pieces [0, n_main) are what the request needs (the matched checkpoint +
    // KV pages); the rest (the slot's other checkpoints) follow on the same
    // stream after the request has started (tail)
    std::size_t n_main = 0;
    Event ev_main = nullptr;
    bool main_rec = false;
    // tail cut short (saveCkpt): the main stream waits for what was enqueued
    Event ev_cut = nullptr;
    // SSD read: index of each logical block in the read job's order (0xffffffff: not read)
    std::vector<std::uint32_t> pos_of_lb;
    Job* job = nullptr;  // SSD read job (nullptr: entry already in RAM)
    bool enqueued = false;
    Event ev = nullptr;
    std::chrono::steady_clock::time_point t0;
    bool from_ssd = false;
    std::uint64_t bytes = 0;
    bool failed = false;
};

struct Stats {
    std::uint64_t spills = 0, spill_bytes = 0, restores_ram = 0, restores_ssd = 0, restore_bytes = 0,
                  ssd_writes = 0, ssd_write_bytes = 0, ram_evictions = 0, ssd_evictions = 0, dropped = 0;
};

class Tier {
public:
    // Arena + stream (everything that touches device memory), before the KV
    // pool is sized. Throws on failure.
    static std::unique_ptr<Tier> create(DeviceOps& ops, std::uint64_t ram_bytes, std::uint64_t seed, int device);
    // Stops the IO thread (joins it after its current job), frees the
    // entries, jobs, events, arena and stream. End every Restore first.
    ~Tier();
    Tier(const Tier&) = delete;
    Tier& operator=(const Tier&) = delete;

    // Layout, SSD index and IO thread (after the KV format is known).
    void start(const Config& cfg, const Layout& lay, std::uint64_t fingerprint);

    std::uint64_t pinnedBytes() const;
    void setWake(std::function<void()> f);

    Entry* find(std::uint64_t id);
    std::uint64_t ramUsedBytes() const;
    std::size_t countRam() const;
    std::size_t ssdEntries() const;

    // GPU work on e finished? (clears gpu_busy)
    bool gpuIdle(Entry* e);
    Entry* newEntry(std::uint32_t nck);
    // Remove an entry entirely (RAM blocks, SSD file, index).
    void removeEntry(Entry* e);

    // Arena blocks for logical blocks [lb0, lb1) of e (allocating what is missing).
    bool ensureBlocks(Entry* e, std::size_t lb0, std::size_t lb1);
    // Blocks covering a logical range.
    static std::array<std::size_t, 2> lbRange(std::uint64_t off, std::uint64_t len);
    // Free blocks past the entry's extent (after it shrank).
    void trim(Entry* e);
    // Release the blocks of checkpoint slot k (invalid checkpoint).
    void dropCkBlocks(Entry* e, std::uint32_t k);

    // Device -> entry (spill). Blocks must exist.
    std::uint64_t copyOut(Entry* e, std::span<const Span> spans);
    // Mark the end of a batch of copies on e (record its event).
    void fence(Entry* e, Event extra /*nullable*/);

    // Longest usable prefix of toks among the entries (RAM or SSD).
    std::optional<Match> lookup(std::span<const std::uint32_t> toks);

    // Start restoring entry e (nullptr: cannot). Poll with pumpRestore.
    Restore* beginRestore(Entry* e, std::span<const Span> spans, std::size_t n_main_spans);
    // Start reading SSD-only entry e into RAM ahead of its restore.
    bool prefetch(Entry* e);
    // Enqueue the pieces whose blocks are in RAM; true once the request's
    // part (pieces [0, n_main)) is done on the GPU (or the restore failed).
    bool pumpRestore(Restore* r);

    struct Verify {
        std::uint64_t bytes = 0;
        std::size_t bad = 0;
    };
    Verify verifyPieces(std::span<const Piece> pieces);
    Verify verifySpans(const Entry* e, std::span<const Span> spans);

    // Stop the tail where it is; returns the event after the enqueued pieces.
    Event cutTail(Restore* r);
    // true once everything (tail pieces, the SSD read) is done.
    bool pumpTail(Restore* r);
    // Finish (or abandon) a restore: unpin, free its resources.
    void endRestore(Restore* r);
    // Main-thread housekeeping: finished SSD jobs, due SSD writes. true while
    // there is background work.
    bool tick();

    bool hasSsd() const { return ssd_dir_w_.has_value(); }

    // public state read by the engine
    Config cfg;
    Layout lay;
    Stats stats;
    Stream stream = nullptr;
    std::vector<Entry*> entries;
    std::uint64_t ssd_bytes = 0;
    std::function<void(std::string_view)> log_fn;

private:
    struct Foreign {
        std::wstring path;
        std::uint64_t size = 0;
        std::int64_t last_used = 0;
    };
    struct IndexItem {
        Entry* e;
        std::int64_t t;
    };

    Tier(DeviceOps& ops, std::uint64_t seed, int device);

    std::uint64_t blockAddr(std::uint32_t b) const;
    void touch(Entry* e);
    Event newEvent();
    void poolEvent(Event ev);
    void releaseBlocks(Entry* e, std::size_t from_lb);
    void dropRam(Entry* e);
    bool evictable(Entry* e, Entry* keep);
    bool makeRoom(std::size_t blocks, Entry* keep);
    bool contains(const Entry* e) const;

    std::wstring filePath(std::uint64_t id) const;
    void deleteFile(std::uint64_t id, std::uint64_t size);
    std::uint64_t fileBytes(const Entry* e) const;
    bool ssdRoom(std::uint64_t need, Entry* keep);
    void submit(Job* j);
    void freeJob(Job* j);
    void makeHeader(const Entry* e, bool valid, std::uint8_t* buf) const;
    void startWrite(Entry* e);
    void indexFile(const std::wstring& full, std::uint64_t size, std::vector<IndexItem>& order);
    void index();

    void ioThread();
    Job* pendingRead();
    Job* pendingRestoreRead();
    void runJob(Job* j);

    DeviceOps& ops_;
    std::uint64_t fingerprint_ = 0;
    std::vector<void*> chunks_;
    std::vector<std::uint64_t> chunk_len_;
    std::uint32_t n_blocks_ = 0;
    std::vector<std::uint32_t> free_blocks_;
    std::uint64_t lru_seq_ = 0;
    std::optional<std::wstring> ssd_dir_w_;
    std::vector<Foreign> foreign_;
    // IO thread
    std::mutex q_mutex_;
    std::condition_variable q_cond_;
    std::vector<Job*> q_reads_;
    std::vector<Job*> q_writes_;
    bool stop_ = false;  // guarded by q_mutex_
    std::thread io_thread_;
    std::vector<Job*> jobs_;
    std::function<void()> wake_fn_;
    std::mt19937_64 rng_;
    std::vector<Event> ev_pool_;
    int dev_ = 0;  // device (the IO thread waits on tier-stream events)
};

// Write %LOCALAPPDATA%\whirl\pinned\<pid>.txt = the pinned host bytes this
// process allocated on purpose (the RAM tier). local_app_data: UTF-8, nullable.
void declarePinned(const char* local_app_data, std::uint64_t bytes);
// FNV-1a over a string (fingerprints).
std::uint64_t hashStr(std::uint64_t h, std::string_view s);
// Identity of the running executable (size + last write time): an SSD entry
// written by another build is not restored (its numerics may differ).
std::uint64_t exeIdentity();

}  // namespace whirl::tier
