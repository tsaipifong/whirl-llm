// Server request slot (arch025-engine S1, moved verbatim out of engine.h): the per-request state the scheduler, the
// prefill / decode phases and the host-tier glue work on (SlotData; its bases SlotPrefix: pages and checkpoints,
// SlotTier: the tier entry and the spill fence), Slot's F1 wait on its last spill, and parkSlotCkpts (the memory
// governor's checkpoint park of one idle slot, moved out of engine_gov.cpp). No Engine: every engine component
// takes Slot& from here (progress/arch025-engine.DESIGN.md, layer L0).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cache/ckpt_store.h"
#include "engine_api.h"
#include "prefix_store.h"
#include "protocol.h"
#include "sampling/sampler.h"
#include "spec/policy.h"
#include "stream.h"
#include "tier_bridge.h"
#include "whirl/model_types.h"
#include "whirl/wddm_watch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace whirl::server {

// rows of one decode / verify step of a slot (engine.h re-exports it; was declared there)
constexpr std::uint32_t max_small_batch = qwen35::max_small_batch;
using sampling::Sampler;

enum class Phase { idle, prefill, decode, restore };

// cache_tokens, last_used: cache::SlotCache; pages, checkpoints and checkpoint points: SlotPrefix (arch024 M4);
// the tier entry and the spill fence: SlotTier (M5). The members; Slot adds the F1 wait (below).
struct SlotData : SlotPrefix, SlotTier {
    std::uint32_t id = 0;
    Phase phase = Phase::idle;
    Job* job = nullptr;
    Sampler sm;
    bool greedy = true;
    bool ignore_eos = false;
    std::uint32_t max_tokens = 0;
    std::uint32_t n_prompt = 0;
    std::size_t pf_off = 0;
    TimePoint tp0{};
    TimePoint td0{};
    double prompt_ms = 0;
    Out out;
    StreamState st;
    std::uint32_t n_gen = 0;
    std::string finish = "length";
    bool done = false;
    bool stopped_str = false;
    bool pool_cut = false;  // FR-SRV P3-8: ended because the KV pool had no page left (decode)
    std::uint32_t n_after_stop = 0;  // FR-SRV P3-7: generated tokens after the stop string (same cycle)
    std::uint32_t cycles = 0;
    std::uint32_t drafted = 0;
    std::uint32_t accepted = 0;
    std::uint32_t relaxed = 0;  // relaxacc: drafts kept by the relaxed rule (not the exact one)
    double relax_gap = 0;       // relaxacc greedy: sum of ln p(argmax) - ln p(kept draft)
    std::uint32_t pos = 0;
    std::uint32_t next = 0;
    // host tiers (tier_id, tier_pages, spill_ev, spill_busy, spill_upto, spill_wanted, restore, rs_map, rs_tail,
    // rs_shared: SlotTier)
    std::optional<TimePoint> tp0_pre;
    std::array<std::uint32_t, max_small_batch> pend{};
    std::size_t n_pend = 0;
    std::uint32_t pend_pos = 0;
    std::uint32_t last_acc = 0;
    std::uint32_t last_row = 0;
    bool logits_ok = false;
    float acc_ema = 0;
    qwen35::DraftAccept dacc;
    qwen35::SlotAccept sacc;      // WHIRL_SLOT_DRAFTS=2: per-slot acceptance with pool shrinkage
    std::uint32_t d_prev = 0;     // MTP drafts of this slot in its last MTP cycle
    qwen35::NgramPolicy ng;
    std::uint32_t ng_cycles = 0;
    // DFlash2 (--draft dflash): the drafter context of this slot holds positions dfl_lo .. dfl_hi - 1
    // as one unbroken run (rows below dfl_lo may exist but follow a gap); acceptance by draft
    // position and the mean draft count of its blocks
    std::uint32_t dfl_lo = 0, dfl_hi = 0;
    qwen35::DraftAccept dfl_acc;
    float dfl_nd = 0;
    std::uint32_t dfl_cycles = 0;
    std::array<std::uint32_t, max_small_batch + 1> batch{};
    std::size_t n_batch = 0;
    std::uint32_t last_log_n = 0;
    TimePoint last_log_t{};
    // WDDM-LOG: sharp decode slowdown within the request (checked at each "gen:" report)
    wddmwatch::DecodeWatch dec_watch;
    // v0.2.2.2 memory governor: the checkpoint buffers were freed (VRAM back to Windows); the slot takes no new
    // request until they are restored
    bool parked = false;

    std::uint64_t reqId() const { return job ? job->id : 0; }
};

// F1 (arch024 M1 / M2): a slot that goes away or is overwritten (vector clear / resize, exception unwinding,
// move-assign) first waits for its last spill, which may still read the checkpoint buffers; only then are the
// members (Ckpt's DevBuf / PinnedBuf) destroyed. All five special members are written out: a destructor alone would
// suppress the implicit moves, and slots_.resize needs the noexcept move constructor.
struct Slot : SlotData {
    Slot() = default;
    Slot(Slot&&) noexcept = default;  // resources change hands, nothing is freed: no wait
    Slot& operator=(Slot&& o) noexcept {
        if (this != &o) {
            waitSpill();
            SlotData::operator=(std::move(o));
        }
        return *this;
    }
    ~Slot() { waitSpill(); }
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;

private:
    // a destructor cannot throw: an event error (device lost) leaves nothing to wait for
    void waitSpill() noexcept {
        try {
            spill_ev.waitIfPending();
        } catch (...) {
        }
    }
};

// govFreeCache's checkpoint part for one idle slot: F1 passes the slot's spill fence to CkptStore::park, then the
// slot holds no checkpoint buffer (parked). Returns the VRAM freed. A free function so the F1 tests drive this call.
std::uint64_t parkSlotCkpts(CkptStore& cs, Slot& sl);

}  // namespace whirl::server
