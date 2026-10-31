// Server engine, the scheduler's side of the host tiers (arch025-engine S15, module server.tier_glue): attachTier,
// waitsForTier / prefetchFor, tryRestore / restoreSteps / pollRestores, tierTick and flushTierOnStop. References only -
// core.shared members of Engine, the queue state of jq_ under its old names, drafter_ctx, slot_end and slot_start;
// bodies verbatim from engine_tier.cpp (DESIGN v3 L8).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cache/page_pool.h"
#include "device/backend.h"
#include "prefix_store.h"
#include "tier_bridge.h"
#include "server/engine_api.h"
#include "server/engine_drafter.h"
#include "server/engine_end.h"
#include "server/engine_gov.h"
#include "server/engine_pages.h"
#include "server/engine_options.h"
#include "server/engine_queue.h"
#include "server/engine_slot.h"
#include "server/engine_start.h"
#include "tier/kv_tier.h"
#include "whirl/thread_annotations.h"

#include <condition_variable>
#include <cstdint>
#include <string>
#include <vector>

namespace whirl::server {

class TierGlue {
public:
    TierGlue(device::Backend& ops, EngineOptions& opt, std::vector<Slot>& slots, PagePool& pool, PrefixStore& ps,
             TierBridge& tb, DrafterCtx& dc, SlotEnd& se, SlotStart& ss, JobQueue& jq, SlotPages& sc,
             GovernorGlue& gv)
        : ops_(ops), opt_(opt), slots_(slots), pool_(pool), ps_(ps), tb_(tb), dc_(dc), se_(se), ss_(ss),
          jq_(jq), sc_(sc), gv_(gv) {}
    TierGlue(const TierGlue&) = delete;
    TierGlue& operator=(const TierGlue&) = delete;

    void attachTier(tier::Tier* t);
    bool waitsForTier(Job& job);
    void prefetchFor(Job& job);
    bool tryRestore(Slot& sl, Job& job);
    void pollRestores();
    bool tierTick();
    TierBridge::RestoreSteps restoreSteps(Slot& sl);
    void flushTierOnStop();

private:
    // slot_end / slot_start, called by name from the bodies above
    void failSlot(Slot& sl, const std::string& err) { se_.failSlot(sl, err); }
    void releaseSlot(Slot& sl) { se_.releaseSlot(sl); }
    bool startJob(Slot& sl, Job& job) { return ss_.startJob(sl, job); }

    device::Backend& ops_;
    EngineOptions& opt_;
    std::vector<Slot>& slots_;
    PagePool& pool_;
    PrefixStore& ps_;
    TierBridge& tb_;
    DrafterCtx& dc_;
    SlotEnd& se_;
    SlotStart& ss_;
    JobQueue& jq_;  // the tier wake (jqTierWake, arch025-engine S10b)
    SlotPages& sc_;
    GovernorGlue& gv_;
};

}  // namespace whirl::server
