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
// SSD: one sparse file per entry (4 KiB header, token ids, block table, data
// region), written by an IO thread with unbuffered positional writes (dirty
// blocks only; the header is invalidated first, behind a flush barrier when the
// file already existed, and the valid header is written last behind another, so a
// power loss never leaves a valid header over other bytes), read back
// the same way (read_qd overlapped reads in flight).
// Block dedup (CACHE-1): a KV block (2 MiB of the data region) that holds the
// same token prefix as a block already on SSD and hashes the same (128-bit
// content hash) is not written; the block table records its owner (entry id,
// logical block) and the read job reads it from the owner's file. Owners are
// always files (references are resolved to the owner at write time, never
// chained). Dropping an owner's SSD copy (LRU, rewrite of a referenced block,
// removal) first drops every entry referencing it; at startup a reference
// whose owner is missing or hashes differently drops the referencing file.
// Startup checks a file's header, size, token checksum and block references
// (owner present, same hash); with the crash-safe write protocol below, a crash,
// power loss, truncation or stale file is a miss (re-prefill), never wrong KV
// bytes. The KV / checkpoint data is not checksummed on read: silent corruption
// inside a file (bit rot, a failing disk) is not detected.
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
#include "evict_policy.h"

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
constexpr std::uint32_t file_version = 2;
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
    std::uint64_t tbl_off;  // block table (n_blk BlkRef, one per KV block)
    std::uint32_t n_blk;
    std::uint32_t _r2;
};
static_assert(sizeof(Header) == 240);

// Block table row of one KV block (logical block kv_lb0 + k of the entry).
struct BlkRef {
    std::uint64_t owner = 0;     // 0: stored in this file; else the owner entry id
    std::uint32_t owner_lb = 0;  // logical block in the owner's file
    std::uint32_t _r = 0;
    std::uint64_t h0 = 0, h1 = 0;  // content hash (0, 0: unknown, never shared)
};
static_assert(sizeof(BlkRef) == 32);

// 128-bit content hash of a block (dedup).
std::array<std::uint64_t, 2> blockHash(const void* p, std::uint64_t n);
static_assert(std::is_standard_layout_v<Header>);

// Layout constants of this model / server (bytes).
struct Layout {
    std::uint64_t ck_bytes = 0, page_bytes = 0;
    std::uint32_t page_tokens = 0;
    std::uint32_t tok_cap = 0;  // max tokens per entry (context per slot)

    std::uint64_t ckStride() const;
    std::uint64_t tokRegion() const;
    std::uint64_t maxKvBlocks() const;
    std::uint64_t tblRegion() const;
    std::uint64_t dataOff() const;
};

struct Config {
    std::uint64_t ram_bytes = 0;
    std::optional<std::string> ssd_dir;  // UTF-8; nullopt: no SSD tier
    std::uint64_t ssd_cap = 0;
    // entries shorter than this are not kept (re-prefill is cheap)
    std::uint32_t min_tokens = 2048;
    // an entry is written to SSD once unchanged for this long (the clock restarts whenever the
    // entry changes or is restored from RAM; 5 min: a conversation that continues within it is
    // served from RAM and never costs a disk write)
    std::uint64_t ssd_delay_ms = 300000;
    // --kv-ssd-mode shutdown: only reused entries (Entry::reused) are ever written: while
    // `flushing` is set (when WHIRL is closed, CLOSE-REUSED), when RAM pressure has to evict one
    // (TIER-EVICT), or after the server sat idle (TIER-IDLE); a one-shot prompt never is
    bool ssd_shutdown_only = false;
    // `always` mode: an entry is written (delay expired, RAM pressure, flush) only if it was reused
    // (its prefix was restored at least once) or is a session that was resumed (grew by appending);
    // a one-shot prompt nobody came back to costs no disk write. Not applied in shutdown-only mode.
    bool ssd_seen_filter = true;
    // SSD files not used for this long are deleted (startup, then every expire_check_ms); 0: never
    std::uint64_t expire_s = 7ull * 24 * 3600;
    std::uint64_t expire_check_ms = 3600ull * 1000;
    // unix seconds "now" (nullptr: the system clock); a test hook
    std::function<std::int64_t()> clock_fn;
    // share identical KV blocks between SSD files (WHIRL_KV_SSD_DEDUP=0: off)
    bool dedup = true;
};

// Meta of the entry's copy on SSD (a RAM eviction of a newer, dirty entry
// falls back to it; the file stays self-consistent until the next write).
struct SsdSnap {
    std::uint32_t n_tok = 0, nck = 0, pages = 0;
    std::array<CkMeta, max_ck> ck{};
    std::vector<std::uint32_t> tokens;
    std::uint64_t file_bytes = 0;   // logical file size
    std::uint64_t alloc_bytes = 0;  // bytes counted against the cap (shared blocks excluded)
    std::vector<BlkRef> tbl;        // one row per KV block
    std::int64_t stamped = 0;       // last use the file records (header last_used / last-write time), unix s
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
    std::int64_t used = 0;  // unix seconds of the last use (spill, restore; SSD files: persisted)
    std::chrono::steady_clock::time_point t_mod{};
    bool removed = false;
    // read jobs of other entries reading blocks this entry owns
    std::uint32_t ext_pin = 0;
    // metadata changed (trimTo): header / table rewrite due
    bool meta_dirty = false;
    // an owner of a block it references was dropped while its write ran
    bool ssd_stale = false;
    // restored at least once, resumed in place (a multi-turn session), or a shared prefix checkpoint
    // (system message) a request attached from VRAM; see Config::ssd_seen_filter, ssd_shutdown_only
    bool reused = false;
    // its SSD write was skipped: a file of nofit_fb bytes did not fit the cap (Tier::startWrite). Not
    // retried until its size changes or SSD room was freed since (Tier::ssd_gen_ moved past nofit_gen).
    std::uint64_t nofit_fb = 0, nofit_gen = 0;
    // Tier::demote: the first RAM / SSD eviction victim, dropped (not written) on RAM eviction; any use clears it
    bool demoted = false;

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
                  ssd_writes = 0, ssd_write_bytes = 0, ram_evictions = 0, ssd_evictions = 0, dropped = 0,
                  ssd_dedup_bytes = 0, superseded = 0, superseded_bytes = 0, demoted = 0,
                  evict_writes = 0,  // shutdown mode: writes of reused entries RAM had to evict (TIER-EVICT)
                  idle_writes = 0,   // shutdown mode: writes of reused entries after the server sat idle (TIER-IDLE)
                  idle_passes = 0;   // idle write passes started (one per idle period at most)
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
    // Remove an entry entirely (RAM blocks, SSD file, index); entries
    // referencing its SSD blocks lose their SSD copy first.
    void removeEntry(Entry* e);
    // Superseded session (compact / compress): keep only tokens [0, keep)
    // (rounded up to what referencing files need, down to its checkpoints),
    // in RAM and on SSD. Returns the bytes freed (0: nothing / not now).
    std::uint64_t trimTo(Entry* e, std::uint32_t keep);
    // Probably superseded but not provably dead (another conversation shares only its system prompt):
    // kept whole, evicted first (RAM: dropped to its SSD copy or gone, never written; SSD: first LRU
    // victim). Any later use (restore, in-place update) undoes it.
    void demote(Entry* e);
    // Entries whose SSD block table references blocks e owns.
    std::vector<Entry*> dependentsOf(const Entry* e) const;
    // Ids of every entry that depends on e directly or through other entries (e itself only if
    // the dependency graph has a cycle through it).
    std::vector<std::uint64_t> dependentClosure(const Entry* e) const;
    // DEDUP-INV: an entry about to be written whose prefix another entry is writing right now (more
    // whole blocks shared with that in-flight write than with any file already on the SSD): it waits
    // for that write, so entries due in the same pass (the shutdown flush) share the prefix too.
    bool waitForDedupBase(const Entry* e) const;
    // test hook (REVIEW-022 F4): entry_id has a write job the IO thread has finished and no tick has reaped yet
    bool writeFinishedUnreaped(std::uint64_t entry_id) const;

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
    // accept (optional): a checkpoint (position, kind code) may be resumed from; the longest accepted one
    // per entry is the match (COLD-RESUME: --precise accepts only cold-exact checkpoints)
    std::optional<Match> lookup(std::span<const std::uint32_t> toks,
                                const std::function<bool(std::uint32_t, std::uint8_t)>& accept = {});

    // Start restoring entry e (nullptr: cannot). Poll with pumpRestore.
    Restore* beginRestore(Entry* e, std::span<const Span> spans, std::size_t n_main_spans);
    // Start reading SSD-only entry e into RAM ahead of its restore.
    bool prefetch(Entry* e);
    // Enqueue the pieces whose blocks are in RAM; true once the request's
    // part (pieces [0, n_main)) is done on the GPU (or the restore failed).
    bool pumpRestore(Restore* r);
    // The same, but an exception from the device layer marks the restore failed (a half-copied
    // checkpoint must never be taken for a valid one) and counts as done.
    bool pumpRestoreNoThrow(Restore* r);

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
    bool pumpTailNoThrow(Restore* r);  // exception -> r->failed, done
    // Finish (or abandon) a restore: unpin, free its resources.
    void endRestore(Restore* r);
    // Main-thread housekeeping: finished SSD jobs, due SSD writes. true while
    // there is background work.
    bool tick();
    // the only background work tick() reported is waiting for write delays to expire
    bool waitOnly() const { return wait_only_; }
    // earliest time an entry becomes due for its SSD write (nullopt: none waiting)
    std::optional<std::chrono::steady_clock::time_point> nextDue() const { return next_due_; }
    // when tick() has timed work next: a due SSD write, or the periodic SSD expiry check (an idle
    // engine must wake for it too, or an idle server never expires files)
    std::optional<std::chrono::steady_clock::time_point> nextWake() const {
        std::optional<std::chrono::steady_clock::time_point> w = next_due_;
        if (ssd_dir_w_ && cfg.expire_s > 0 && cfg.expire_check_ms > 0 && (!w || next_expiry_ < *w)) w = next_expiry_;
        return w;
    }
    // RAM evictions started SSD writes that will free blocks when they finish (retry the spill then)
    bool roomPending() const { return room_pending_; }

    bool hasSsd() const { return ssd_dir_w_.has_value(); }
    // What the SSD cache cleanup removed (startup and periodic expiry) and what it holds now.
    struct Cleanup {
        std::uint64_t files = 0, bytes = 0;  // on disk now (after the startup cleanup)
        std::uint64_t expired_files = 0, expired_bytes = 0;
        std::uint64_t bad_files = 0, bad_bytes = 0;          // corrupt, half-written, old format, unreadable
        std::uint64_t cap_files = 0, cap_bytes = 0;          // evicted (LRU) to fit the cap
        // could not be opened (locked by another program): kept, counted against the cap, evicted first
        std::uint64_t unreadable_files = 0, unreadable_bytes = 0;
        // written with another per-slot context (tok_cap: other file layout): kept, counted, evicted first
        std::uint64_t layout_files = 0, layout_bytes = 0;
    };
    const Cleanup& cleanup() const { return cleanup_; }
    // the SSD directory is used by another process (SSD tier disabled)
    bool ssdDirBusy() const { return dir_busy_; }

    // graceful shutdown: write due entries now (also in ssd_shutdown_only mode)
    bool flushing = false;

    // TIER-IDLE (shutdown mode; policy in idle_flush.h): the engine has been idle long enough. tick() then
    // writes the entries idleWriteWanted picks, one write at a time, and ends the pass by itself when none
    // is left. endIdleWrite: a request arrived -- no further write is started (one already running finishes).
    // Returns the entries it will write (0: nothing to do, no pass started).
    std::size_t beginIdleWrite();
    void endIdleWrite();
    bool idleWriting() const { return idle_writing_; }

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
        bool tried = false;  // ssdRoom: deleting it failed in this pass (locked)
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
    bool ensureBlocksHeld(Entry* e, std::size_t lb_end);
    bool contains(const Entry* e) const;

    std::wstring filePath(std::uint64_t id) const;
    void deleteFile(std::uint64_t id, std::uint64_t size);
    void keepForeign(const std::wstring& path, std::uint64_t counted);
    void forgetForeign(const std::wstring& path);
    std::uint64_t fileBytes(const Entry* e) const;
    // ram_copies: SSD copies of entries also in RAM may go too (only enforceCap: a write never drops
    // them, the RAM entry would just be written again -- an endless rewrite loop under a tight cap)
    bool ssdRoom(std::uint64_t need, Entry* keep, bool ram_copies = false);
    void noFit(Entry* e, std::uint64_t fb);
    // bumped whenever SSD room is freed (a file deleted or cut, an entry dropped to SSD-only)
    std::uint64_t ssd_gen_ = 0;
    void submit(Job* j);
    void freeJob(Job* j);
    void makeHeader(const Entry* e, bool valid, std::uint8_t* buf) const;
    void startWrite(Entry* e);
    void indexFile(const std::wstring& full, std::uint64_t size, std::vector<IndexItem>& order);
    void index();
    void validateRefs();
    bool ssdDroppable(const Entry* e, int depth = 0) const;
    void dropSsdCopy(Entry* e);
    std::vector<std::uint64_t> dropping_;  // ids whose dropSsdCopy is running (a dependency cycle must not recurse forever)
    void touchOwners(const Entry* e);
    void gcOwners();
    std::uint64_t allocBytes(const Entry* e, std::size_t n_ref) const;
    bool refsOk(const Entry* e) const;
    void readSources(Job* j, const Entry* e);

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
    void* dir_lock_ = nullptr;  // HANDLE of <ssd dir>\whirl.lock (delete on close)
    bool dir_busy_ = false;
    bool gc_due_ = false;
    Cleanup cleanup_;
    bool wait_only_ = false;
    bool room_pending_ = false;
    std::optional<std::chrono::steady_clock::time_point> next_due_;
    std::int64_t nowUnix() const;
    void markUsed(Entry* e);
    void stampFile(std::uint64_t id, std::int64_t unix_s);
    void persistUse();
    void expireSweep();
    void enforceCap();
    std::uint64_t foreignFiles_() const { return foreign_.size(); }
    std::chrono::steady_clock::time_point next_expiry_{};
    bool seenOk(const Entry* e) const { return cfg.ssd_shutdown_only || !cfg.ssd_seen_filter || e->reused; }
    EvictFacts evictFacts(const Entry* v) const;
    bool evictWanted(const Entry* v) const;
    bool evictWrite(Entry* v);
    bool idleWanted(const Entry* v) const;
    bool closeWanted(const Entry* v) const;
    // TIER-IDLE pass state
    bool idle_writing_ = false;
    std::vector<std::uint64_t> idle_skip_;  // entries whose write could not start in this pass (not retried)
    std::uint64_t idle_pass_writes_ = 0, idle_pass_bytes0_ = 0;
    std::chrono::steady_clock::time_point idle_t0_{};
};

// Write %LOCALAPPDATA%\whirl\pinned\<pid>.txt = the pinned host bytes this
// process allocated on purpose (the RAM tier). local_app_data: UTF-8, nullable.
void declarePinned(const char* local_app_data, std::uint64_t bytes);
// FNV-1a over a string (fingerprints).
std::uint64_t hashStr(std::uint64_t h, std::string_view s);
// Identity of the running executable (size + last write time): an SSD entry
// written by another build is not restored (its numerics may differ).
std::uint64_t exeIdentity();

// CPU-test hook (power-loss fault injection; nullptr in the product). When set, the SSD write
// job's positional writes and flush barriers go through it, so a test can model the drive's
// volatile write cache: writes not yet covered by a flush may reach the media in any order, or
// not at all. While dead() (the simulated power is off) no file is deleted, truncated or
// re-stamped. Set it only while no IO job is running.
struct WriteSim {
    virtual ~WriteSim() = default;
    virtual bool write(void* fh, const std::uint8_t* buf, std::uint64_t len, std::uint64_t off) = 0;
    virtual bool flush(void* fh) = 0;
    virtual bool dead() const = 0;
};
extern std::atomic<WriteSim*> g_write_sim;

}  // namespace whirl::tier
