// Prefix store of the server (arch024 M4): the shared prefix checkpoint table (spes_), the prefix cache's decisions
// (cache::PrefixCache) and the tier index view, the chunk schedule of a prompt (system / think-open / image splits,
// the schedule floor, the TAIL-CK point) and the COLD-RESUME rule, so "where may a request resume, and which slot
// takes it" is answered in one place. Host-only: the device side (copies, page mapping, spills) stays with Engine.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Moved out of Engine (sysSplit / imgSplits / schedFloor / onSchedule / resumeOk / speMatch /
// lcpTarget / cacheMatch / pickSlot, the TAIL-CK point of startJob) with the same rules. include/whirl/
// prefill_schedule.h is shared with the CLI and stays as it is (arch024 PLAN C6). Prior art:
// progress/arch024-S5a.PRIOR_ART.md (llama.cpp get_available_slot: LCP similarity, then LRU; vLLM / SGLang: the
// prefix lookup is an object beside the scheduler).

#pragma once

#include "cache/ckpt_store.h"
#include "cache/prefix_cache.h"
#include "device/handles.h"
#include "server/engine_thread.h"
#include "server/tokens.h"
#include "whirl/prefill_schedule.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace whirl::tier {
class Tier;
}

namespace whirl::server {

// checkpoint kind <-> code (tier entries and cache::ResumePred): 1 prompt-end, 2 think-open, 3 gen-end, 4 system,
// 5 prefix, 0 other ("restored")
std::uint8_t ckKindCode(std::string_view k);
const char* ckKindStr(std::uint8_t c);

// The prefix-cache part of a slot (arch024 M4): its KV pages, its checkpoints and the checkpoint points of the
// request it prefills. cache::SlotCache (cache_tokens, last_used) stays the base: PrefixCache reads slots through
// View<const SlotCache>.
struct SlotPrefix : cache::SlotCache {
    std::vector<std::int32_t> pages;
    std::vector<Ckpt> ckpts;
    std::uint64_t ckpt_seq = 0;
    std::uint32_t reuse = 0;
    std::optional<DevPtr> ck_hid;
    std::uint32_t pf_split = 0;
    std::uint32_t sys_split = 0;
    std::uint32_t lcp_ck = 0;
    // vision: end of the prompt's last image (after <|vision_end|>), a chunk split
    // where a shared 'prefix' checkpoint is kept (0 = none)
    std::uint32_t img_ck = 0;
    // TAIL-CK: slot checkpoint at the start of the prompt's last schedule chunk (0 = none)
    std::uint32_t tail_ck = 0;
    bool on_sched = false;
};

// Shared prefix checkpoint (item S).
// index fields (valid, n, kind, tokens, last_used, pin, uses): cache::SpeKey
struct SpeData : cache::SpeKey {
    std::vector<std::int32_t> pages;
    Ckpt ck;
    std::uint64_t tier_id = 0;
    device::EventH spill_ev;
    bool spill_busy = false;
    device::EventH ready_ev;
};

// F1 as for Slot: waits for the spill (reads ck) and the ready fence (the copy into ck) before ck is freed.
struct Spe : SpeData {
    Spe() = default;
    Spe(Spe&&) noexcept = default;
    Spe& operator=(Spe&& o) noexcept {
        if (this != &o) {
            waitFences();
            SpeData::operator=(std::move(o));
        }
        return *this;
    }
    ~Spe() { waitFences(); }
    Spe(const Spe&) = delete;
    Spe& operator=(const Spe&) = delete;

private:
    void waitFences() noexcept {
        try {
            spill_ev.waitIfPending();
            ready_ev.waitIfPending();
        } catch (...) {
        }
    }
};

class PrefixStore {
public:
    struct Config {
        bool enabled = true;          // !--no-prefix-cache
        bool lcp_on = false;          // LCP checkpoints
        bool cold_resume = false;     // COLD-RESUME (--precise): resume only from cold-exact states
        std::uint32_t sys_min = 0;    // system-message split minimum (0 = off)
        bool tail_ck = true;          // TAIL-CK
        std::uint32_t tail_ck_min = sched::tail_ck_min_default;
        std::uint32_t kv_page = 64;   // tokens per KV page
    };
    // vision (attachVision): image-end splits of the schedule
    struct Vision {
        bool on = false;              // an mmproj is loaded
        std::uint32_t vision_end_id = 0;
        std::uint32_t ck_min = sched::img_split_min_default;
    };
    struct Match {
        std::uint32_t reuse = 0;
        Ckpt* ck = nullptr;
        std::size_t lcp = 0;
    };

    PrefixStore(const ChatTokens& ct, const Config& cfg);
    // arch024 M8: the engine thread that may call the entry points below (nullptr: unchecked; docs/dev/threads.md)
    void setOwner(const EngineThread* o) { owner_ = o; }
    const EngineThread* ownerThread() const { return owner_; }

    const Config& config() const { return cfg_; }
    cache::PrefixCache& cache() { return *pc_; }
    const cache::Tier* tierView() const { return tier_view_.get(); }
    void attachTier(tier::Tier* t);  // nullptr: none
    void setVision(const Vision& v) {
        ownerCheck(owner_, "PrefixStore::setVision");
        vis_ = v;
    }

    // ---- the chunk schedule of a prompt (functions of the tokens and the settings)
    std::uint32_t sysSplit(std::span<const std::uint32_t> toks) const;
    // vision: the image-end splits of toks (position after the <|vision_end|> that follows a
    // run of image ids), those >= ck_min; a function of the tokens only, so every prompt
    // with the same prefix gets the same splits there (at most 32)
    std::vector<std::uint32_t> imgSplits(std::span<const std::uint32_t> toks) const;
    std::size_t schedFloor(std::span<const std::uint32_t> toks, std::size_t lim) const;
    bool onSchedule(std::span<const std::uint32_t> toks, std::size_t a) const;
    // COLD-RESUME: may a request with these tokens resume from a checkpoint (position, kind code)?
    bool resumeOk(std::span<const std::uint32_t> toks, std::uint32_t pos, std::uint8_t kind) const;
    // TAIL-CK: the slot checkpoint point of a prompt that resumes at sl.reuse with the split points already set in sl
    // (on_sched, pf_split, sys_split, lcp_ck, img_ck), or 0 (none). Only with at least cache::ckpt_keep_prefix_min
    // checkpoints in the slot, so prompt-end / gen-end stay.
    std::uint32_t tailCk(const SlotPrefix& sl, std::span<const std::uint32_t> toks) const;

    // ---- shared prefix checkpoints
    std::vector<Spe>& spes() { return spes_; }
    const std::vector<Spe>& spes() const { return spes_; }
    Spe* speMatch(std::span<const std::uint32_t> toks);
    // memory governor (arch024 M7, engine_gov.cpp): the shared prefix checkpoints' buffers parked / back.
    // waitSpeFences: every entry's spill and ready fences (before the pool's quarantine is reclaimed and the buffers go).
    void waitSpeFences();
    // parkSpes: every entry's checkpoint buffers released with its fences (F1) and the table dropped; the bytes freed.
    std::uint64_t parkSpes(CkptStore& cs);
    // unparkSpes: a new table of n entries with fresh checkpoint buffers; the bytes allocated. On a failure the
    // buffers allocated so far are released ({}: just allocated, no spill has read them), the table is dropped and
    // the error rethrown.
    std::uint64_t unparkSpes(CkptStore& cs, std::size_t n);

    // ---- lookup over the slots
    Match cacheMatch(SlotPrefix& sl, std::span<const std::uint32_t> toks) const;
    std::uint32_t lcpTarget(cache::View<const cache::SlotCache> slots, std::span<const std::uint32_t> toks,
                            std::uint32_t reuse, std::uint32_t sys, std::size_t self_slot = static_cast<std::size_t>(-1),
                            std::uint64_t self_tier = 0) const;
    // the free slot (avail(slot)) that resumes the longest prefix (its own checkpoints, or a shared one), ties to
    // the least recently used; nullptr: none free
    template <class S, class Avail>
    S* pickSlot(std::span<S> slots, std::span<const std::uint32_t> toks, Avail avail) {
        ownerCheck(owner_, "PrefixStore::pickSlot");
        S* best = nullptr;
        std::uint32_t best_reuse = 0;
        const std::uint32_t spe_n = [&] {
            Spe* s = speMatch(toks);
            return s ? s->n : 0u;
        }();
        for (S& sl : slots) {
            if (!avail(sl)) continue;
            const std::uint32_t r = std::max(cacheMatch(sl, toks).reuse, spe_n);
            if (!best || r > best_reuse || (r == best_reuse && sl.last_used < best->last_used)) {
                best = &sl;
                best_reuse = r;
            }
        }
        return best;
    }

private:
    const ChatTokens& ct_;
    Config cfg_;
    Vision vis_;
    // prefix cache decisions (lookup / insertion / eviction); tier_view_: the tier seen through cache::Tier
    std::unique_ptr<cache::PrefixCache> pc_;
    std::unique_ptr<cache::Tier> tier_view_;
    std::vector<Spe> spes_;
    const EngineThread* owner_ = nullptr;
};

}  // namespace whirl::server
