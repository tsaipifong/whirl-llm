// Tier bridge of the server (arch024 M5: spill half S6b, restore half S7). Between the engine's slots / shared prefix
// checkpoints and the host tier (tier::Tier): which state of an idle slot goes to the RAM tier and in what span order
// (checkpoints, then KV pages of every pool array), the spill fence, the superseded-session rule (CACHE-1: mark on a
// conversation's first spill, cut / demote on its second), the bounded flush at shutdown and the TIER-IDLE write; and
// which tier entry is worth restoring into a slot, which of its checkpoints and in what order (the request's part,
// then the tail), the pumps, the tail cut, verify, the tier counters and the log lines. The data movement (copyOut,
// fences, the IO thread, SSD files) stays in tier::Tier; the scheduling (when a slot is idle, when the loop is idle,
// whether a request fits, the slot's pages, its phase, startJob, the loops over the slots) stays in Engine.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Moved out of Engine (S6b: waitSpill / ckSpans / pageSpans / spillSlot / spillSpe /
// supersedeMark / supersedeApply / flushTierOnStop / idleFlushPoll / idleFlushDue; S7: the tier parts of tryRestore /
// beginRestore / finishTail / cutTail / pollRestores / tierTick / prefetchFor / waitsForTier / logTier) with the same
// rules and the same log lines. Prior art: progress/arch024-S6b.PRIOR_ART.md (llama.cpp: state_write_data picks the
// pieces, llama_io_write_i the destination; server_prompt_cache is an object beside the slots; SGLang HiCache / vLLM
// offloading: the write policy lives in a cache controller, not in the scheduler) and progress/arch024-S7.PRIOR_ART.md
// (llama.cpp server_prompt_cache::load picks and writes back, the slot clears on failure; vLLM KV connector: the
// scheduler allocates the blocks, the connector loads and reports completion).

#pragma once

#include "cache/ckpt_store.h"
#include "cache/prefix_cache.h"
#include "device/backend.h"
#include "device/handles.h"
#include "cache/cache_model.h"  // not server/backend.h: the narrow model interface (arch025-lane2 D2.2)
#include "server/prefix_store.h"
#include "whirl/chat_tokens.h"
#include "tier/idle_flush.h"
#include "tier/kv_tier.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <span>
#include <vector>

namespace whirl::server {

// The host-tier part of a slot (arch024 M5): the tier entry it last spilled to, the spill fence, and the restore
// under way (S7: restore, rs_map, rs_tail, rs_shared).
struct SlotTier {
    std::uint64_t tier_id = 0;
    std::uint32_t tier_pages = 0;
    device::EventH spill_ev;  // fence of the slot's last spill (tier stream)
    bool spill_busy = false;
    std::uint32_t spill_upto = 0;
    bool spill_wanted = false;
    tier::Restore* restore = nullptr;
    std::array<std::int32_t, tier::max_ck> rs_map{};  // entry checkpoint -> slot checkpoint (-1: not restored)
    std::array<bool, tier::max_ck> rs_tail{};         // entry checkpoints still arriving on the tier stream (item T3)
    bool rs_shared = false;                           // restoring a shared prefix entry (item S)
};

class TierBridge {
public:
    struct Config {
        bool verify = false;     // WHIRL_TIER_VERIFY: each spill / restore is read back and compared (one log line per attempt)
        bool supersede = true;   // CACHE-1 superseded-session cut (WHIRL_KV_SUPERSEDE=0: off)
        std::uint64_t idle_flush_ms = tier::idle_flush_ms_default;  // TIER-IDLE (0: off)
        std::uint64_t flush_ms = 10000;       // bound of the shutdown flush
        std::uint64_t flush_long_ms = 60000;  // the same with --kv-ssd-mode shutdown (every reused entry is written then)
        std::uint32_t min_gain = 512;         // a restore brings at least this many tokens over the VRAM prefix (TIER_MIN_GAIN)
    };
    // the Engine steps inside a restore, called where they were (the slot's cached prefix, the DFlash2 drafter
    // context, a VRAM shared checkpoint made from a restored shared entry)
    struct RestoreSteps {
        std::function<void()> drop_cache;                 // failed: Engine::dropCache
        std::function<void(std::uint32_t)> drafter_reset;  // restored n_tok tokens: DrafterCtx::dflReset (Engine::dc_)
        std::function<void(Ckpt&, std::uint64_t)> share;   // restored shared entry (its checkpoint, entry id): Engine::createSpe
    };
    enum class Pump { pending, failed, restored };
    // tier counters for other threads (Engine::stats / statsShared), published by tick()
    struct Published {
        std::uint64_t ssd_written = 0, ssd_dedup = 0, ssd_used = 0, ssd_entries = 0, superseded = 0;
    };

    // m: the model (main stream, state geometry); kv: the pool arrays (Engine's, set by initPool)
    TierBridge(device::Backend& ops, CacheModel& m, const chat::ChatTokens& ct, PrefixStore& ps,
               const std::vector<KvArr>& kv, const Config& cfg);
    // arch024 M8: the engine thread that may call the entry points below (nullptr: unchecked; docs/dev/threads.md)
    void setOwner(const EngineThread* o) { owner_ = o; }
    const EngineThread* ownerThread() const { return owner_; }

    const Config& config() const { return cfg_; }
    tier::Tier* tier() const { return tier_; }
    void attach(tier::Tier* t) {  // nullptr: none
        ownerCheck(owner_, "TierBridge::attach");
        tier_ = t;
    }

    // ---- spill fences
    // the main stream waits for the slot's last spill, which may still read its checkpoint buffers (item S: also
    // after an eviction cleared spill_busy)
    void waitSpill(SlotTier& st);
    // other places that order the main stream after a spill (page eviction, shared prefix replacement) count here
    void countSpillWait() {
        ownerCheck(owner_, "TierBridge::countSpillWait");
        spill_waits_ += 1;
    }
    std::uint64_t spillWaits() const { return spill_waits_; }

    // ---- span lists (the order of the pieces in a tier entry; spills and restores)
    std::vector<tier::Span>& spans() { return spans_; }
    // a checkpoint's pieces at base: per DeltaNet layer conv, ssm; then the hidden row and the logits
    void ckSpans(const Ckpt& c, std::uint64_t base, std::uint8_t tag = 0xff);
    // one KV page at base: the page's rows of every pool array in kvArrays() order (K, V, then q8h / q4 scales)
    void pageSpans(std::int32_t phys, std::uint64_t base);

    // ---- spills
    // item G: an idle slot's new content (pages, checkpoints) to the RAM tier, async on the tier stream; a new
    // entry, or in place when the slot continues the entry it last spilled to. id: the slot's number (logs, entry
    // owner); idle: the slot is idle (nothing is spilled otherwise)
    void spillSlot(std::uint32_t id, bool idle, SlotPrefix& sp, SlotTier& st);
    // item S: a new shared prefix checkpoint to the RAM tier (once; one already there is reused)
    void spillSpe(Spe& s);

    // ---- CACHE-1 superseded sessions (spillSlot calls these: mark on a new entry, apply on an in-place update)
    void dropContained(std::uint32_t slot_id, tier::Entry* x);
    void supersedeMark(std::uint32_t slot_id, tier::Entry* x);
    void supersedeApply(std::uint32_t slot_id, tier::Entry* x);
    std::size_t supersedePending() const { return supersede_.size(); }  // marks not yet applied (tests)

    // ---- restores (item G; item T3: the request's part first, the slot's other checkpoints follow as the tail)
    // the tier entry worth restoring into a slot holding sp for toks (over the slot's VRAM prefix and the shared
    // checkpoints, by the gain rule), or none. The caller checked that the request fits
    std::optional<tier::Match> restoreMatch(SlotPrefix& sp, std::span<const std::uint32_t> toks);
    // after the scheduler gave the slot its pages for mt: choose the checkpoints (the matched one, then the others by
    // position), build the spans and start the copies. false: the tier did not start it. t0: when the restore began
    bool startRestore(std::uint64_t job_id, std::uint32_t id, std::span<const std::uint32_t> toks, SlotPrefix& sp,
                      SlotTier& st, const tier::Match& mt, std::chrono::steady_clock::time_point t0);
    // pump the main part of the slot's restore (an exception marks it failed, never a valid one)
    Pump pumpRestore(SlotTier& st);
    // the main part failed (copies may still be in flight): fence, the slot's cache goes, the restore ends
    void restoreFailed(std::uint64_t job_id, std::uint32_t id, SlotTier& st, const RestoreSteps& steps);
    // the main part is in: the slot's tokens, checkpoints and tier entry, stats, verify, log; the tail goes on
    void restoreDone(std::uint64_t job_id, std::uint32_t id, SlotPrefix& sp, SlotTier& st, const RestoreSteps& steps);
    // the tail: pump it (wait: until it is in), then its checkpoints become valid and the restore ends
    void finishTail(std::uint32_t id, SlotPrefix& sp, SlotTier& st, bool wait);
    // the slot writes its checkpoints again (saveCkpt): what is in counts, the rest of the tail is dropped
    void cutTail(std::uint32_t id, SlotPrefix& sp, SlotTier& st);
    // a queued request and a slot restoring (st): the prefix that restore brings for toks when it reaches the tier's
    // minimum and the schedule, else 0 (the scheduler compares it with what VRAM has)
    std::uint32_t queuedReuse(const SlotTier& st, std::span<const std::uint32_t> toks) const;
    // a queued request: the SSD entry a prefetch into RAM would serve (not in RAM, not loading, on the schedule)
    std::optional<tier::Match> prefetchCandidate(std::span<const std::uint32_t> toks);
    void prefetch(std::uint64_t job_id, tier::Entry* x);

    // ---- the tier's background work (SSD writes, reaps) and the published counters; true: still busy
    bool tick();
    Published published() const;
    // the "kv tier |" summary line; cow_pages / q_waits: the page pool's counters
    void logStats(std::uint64_t cow_pages, std::uint64_t q_waits) const;

    // ---- shutdown: the tier's pending writes run now (no SSD delay), tick() until it is not busy or the bound
    // (flush_ms, flush_long_ms in shutdown mode) passed; true: writes still pending at the bound
    bool flushOnStop(const std::function<bool()>& tick);

    // ---- TIER-IDLE (policy: tier/idle_flush.h), from runLoop's idle wait. building: a request is being built
    void idlePoll(bool building);
    std::optional<std::chrono::steady_clock::time_point> idleDue() const;
    void idleActivity() {
        ownerCheck(owner_, "TierBridge::idleActivity");
        idle_clock_.activity();
    }

private:
    device::Backend& ops_;
    CacheModel& m_;
    const chat::ChatTokens& ct_;
    PrefixStore& ps_;
    const std::vector<KvArr>& kv_;
    Config cfg_;
    std::uint64_t conv_bytes_ = 0, ssm_bytes_ = 0;
    tier::Tier* tier_ = nullptr;
    std::vector<tier::Span> spans_;
    std::uint64_t spill_waits_ = 0;
    cache::SupersedeTracker supersede_;
    tier::IdleFlushClock idle_clock_;
    std::atomic<std::uint64_t> pub_ssd_written_{0}, pub_ssd_dedup_{0}, pub_ssd_used_{0}, pub_ssd_entries_{0}, pub_superseded_{0};
    const EngineThread* owner_ = nullptr;
};

}  // namespace whirl::server
