// Server engine, the prefill phase (arch025-engine S12, module server.prefill): chunk length, the decode floor
// (floorProtected / noteFloor), checkpoint splits, prefillStep / prefillGroup / finishPrefill and the loop's
// runPrefill. References only - core.shared members of Engine, n_building_ of jq_, drafter_ctx, slot_cache, slot_end
// and the placed state fl_ / prefill_tok_cum_; bodies verbatim from engine_prefill.cpp (DESIGN v3 L6).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "cache/page_pool.h"
#include "stream.h"
#include "decode_floor.h"
#include "device/backend.h"
#include "prefix_store.h"
#include "server/engine_api.h"
#include "server/engine_drafter.h"
#include "server/engine_end.h"
#include "server/engine_options.h"
#include "server/engine_pages.h"
#include "server/engine_queue.h"
#include "server/engine_sampling.h"
#include "server/engine_slot.h"
#include "server/engine_state.h"
#include "whirl/chat_tokens.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace whirl::server {

class PrefillPhase {
public:
    PrefillPhase(ServerModel& m, device::Backend& ops, const chat::ChatTokens& ct, EngineOptions& opt, SpecState& spec,
                 FloorState& fl, SampleBuffers& sb, std::vector<Slot>& slots, BatchStats& bst, std::uint64_t& prefill_tok_cum,
                 PrefixStore& ps, DrafterCtx& dc, SlotPages& sc, SlotEnd& se, JobQueue& jq, PagePool& pool,
                 const ResponseCtx& rcx)
        : m_(m), ops_(ops), ct_(ct), opt_(opt), spec_(spec), fl_(fl), sb_(sb), slots_(slots), bst_(bst),
          prefill_tok_cum_(prefill_tok_cum), ps_(ps), dc_(dc), sc_(sc), se_(se), n_building_(jq.n_building_), pool_(pool), rcx_(rcx) {}
    PrefillPhase(const PrefillPhase&) = delete;
    PrefillPhase& operator=(const PrefillPhase&) = delete;

    std::size_t chunkLen(const Slot& sl) const;
    std::vector<DecodeFloor::SlotTok> floorProtected() const;
    void maybeSplitCkpt(Slot& sl, DevPtr hid_row);
    void prefillStep(Slot& sl);
    std::size_t prefillGroup(Slot& first, std::size_t row_budget);
    void finishPrefill(Slot& sl);
    void runPrefill(Slot* pf, bool& idle_wait, std::size_t& pf_rows, bool& floor_pf);
    void noteFloor(TimePoint tl2, TimePoint tl3, std::size_t pf_rows, bool floor_pf);
    // arch025-engine S12b: the scheduler's two uses of the floor state
    void floorLogPeriod();
    bool floorOn() const { return fl_.floor.on(); }

private:
    // slot_end, called by name from the bodies above
    void failSlot(Slot& sl, const std::string& err) { se_.failSlot(sl, err); }
    void finishJob(Slot& sl) { se_.finishJob(sl); }

    ServerModel& m_;
    device::Backend& ops_;
    const chat::ChatTokens& ct_;
    EngineOptions& opt_;
    SpecState& spec_;
    FloorState& fl_;
    SampleBuffers& sb_;
    std::vector<Slot>& slots_;
    BatchStats& bst_;
    std::uint64_t& prefill_tok_cum_;
    PrefixStore& ps_;
    DrafterCtx& dc_;
    SlotPages& sc_;
    SlotEnd& se_;
    const std::atomic<std::uint32_t>& n_building_;
    PagePool& pool_;
    const ResponseCtx& rcx_;
};

}  // namespace whirl::server
