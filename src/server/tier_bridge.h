// Tier bridge of the server (arch024 M5, first half: spill). Between the engine's slots / shared prefix checkpoints
// and the host tier (tier::Tier): which state of an idle slot goes to the RAM tier and in what span order (checkpoints,
// then KV pages of every pool array), the spill fence, the superseded-session rule (CACHE-1: mark on a conversation's
// first spill, cut / demote on its second), the bounded flush at shutdown and the TIER-IDLE write. The data movement
// (copyOut, fences, the IO thread, SSD files) stays in tier::Tier; the scheduling (when a slot is idle, when the loop
// is idle) stays in Engine. The restore half (tryRestore, beginRestore, pollRestores, tierTick, ...) is S7.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Moved out of Engine (waitSpill / ckSpans / pageSpans / spillSlot / spillSpe / supersedeMark /
// supersedeApply / flushTierOnStop / idleFlushPoll / idleFlushDue) with the same rules and the same log lines. Prior
// art: progress/arch024-S6b.PRIOR_ART.md (llama.cpp: state_write_data picks the pieces, llama_io_write_i the
// destination; server_prompt_cache is an object beside the slots; SGLang HiCache / vLLM offloading: the write policy
// lives in a cache controller, not in the scheduler).

#pragma once

#include "cache/ckpt_store.h"
#include "cache/prefix_cache.h"
#include "device/backend.h"
#include "device/handles.h"
#include "server/backend.h"
#include "server/prefix_store.h"
#include "server/tokens.h"
#include "tier/idle_flush.h"
#include "tier/kv_tier.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <vector>

namespace whirl::server {

// The host-tier part of a slot (arch024 M5): the tier entry it last spilled to and the spill fence. S7 adds the
// restore fields (restore, rs_map, rs_tail) here.
struct SlotTier {
    std::uint64_t tier_id = 0;
    std::uint32_t tier_pages = 0;
    device::EventH spill_ev;  // fence of the slot's last spill (tier stream)
    bool spill_busy = false;
    std::uint32_t spill_upto = 0;
    bool spill_wanted = false;
};

class TierBridge {
public:
    struct Config {
        bool verify = false;     // WHIRL_TIER_VERIFY: each spill is read back and compared (one log line per attempt)
        bool supersede = true;   // CACHE-1 superseded-session cut (WHIRL_KV_SUPERSEDE=0: off)
        std::uint64_t idle_flush_ms = tier::idle_flush_ms_default;  // TIER-IDLE (0: off)
        std::uint64_t flush_ms = 10000;       // bound of the shutdown flush
        std::uint64_t flush_long_ms = 60000;  // the same with --kv-ssd-mode shutdown (every reused entry is written then)
    };

    // m: the model (main stream, state geometry); kv: the pool arrays (Engine's, set by initPool)
    TierBridge(device::Backend& ops, ServerModel& m, const ChatTokens& ct, PrefixStore& ps,
               const std::vector<KvArr>& kv, const Config& cfg);

    const Config& config() const { return cfg_; }
    tier::Tier* tier() const { return tier_; }
    void attach(tier::Tier* t) { tier_ = t; }  // nullptr: none

    // ---- spill fences
    // the main stream waits for the slot's last spill, which may still read its checkpoint buffers (item S: also
    // after an eviction cleared spill_busy)
    void waitSpill(SlotTier& st);
    // other places that order the main stream after a spill (page eviction, shared prefix replacement) count here
    void countSpillWait() { spill_waits_ += 1; }
    std::uint64_t spillWaits() const { return spill_waits_; }

    // ---- span lists (the order of the pieces in a tier entry; also used by the restore half)
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
    void supersedeMark(std::uint32_t slot_id, tier::Entry* x);
    void supersedeApply(std::uint32_t slot_id, tier::Entry* x);
    std::size_t supersedePending() const { return supersede_.size(); }  // marks not yet applied (tests)

    // ---- shutdown: the tier's pending writes run now (no SSD delay), tick() until it is not busy or the bound
    // (flush_ms, flush_long_ms in shutdown mode) passed; true: writes still pending at the bound
    bool flushOnStop(const std::function<bool()>& tick);

    // ---- TIER-IDLE (policy: tier/idle_flush.h), from runLoop's idle wait. building: a request is being built
    void idlePoll(bool building);
    std::optional<std::chrono::steady_clock::time_point> idleDue() const;
    void idleActivity() { idle_clock_.activity(); }

private:
    device::Backend& ops_;
    ServerModel& m_;
    const ChatTokens& ct_;
    PrefixStore& ps_;
    const std::vector<KvArr>& kv_;
    Config cfg_;
    std::uint64_t conv_bytes_ = 0, ssm_bytes_ = 0;
    tier::Tier* tier_ = nullptr;
    std::vector<tier::Span> spans_;
    std::uint64_t spill_waits_ = 0;
    cache::SupersedeTracker supersede_;
    tier::IdleFlushClock idle_clock_;
};

}  // namespace whirl::server
