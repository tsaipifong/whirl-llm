// Server engine, the start of a request (arch025-engine S14, module server.slot_start): startJob (prompt render and
// tokens, cache / prefix / checkpoint lookup, vision, the slot's sampling and draft state). References only -
// core.shared members of Engine, drafter_ctx, governor, prefill, slot_cache and the placed state ids_ / crlf_norm_;
// body verbatim from engine.cpp (DESIGN v3 L7).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "cache/ckpt_store.h"
#include "cache/page_pool.h"
#include "device/backend.h"
#include "prefix_store.h"
#include "stream.h"
#include "tier_bridge.h"
#include "server/engine_api.h"
#include "server/engine_drafter.h"
#include "server/engine_gov.h"
#include "server/engine_options.h"
#include "server/engine_pages.h"
#include "server/engine_prefill.h"
#include "server/engine_sampling.h"
#include "server/engine_slot.h"
#include "server/engine_state.h"
#include "server/engine_vision.h"
#include "whirl/chat_tokens.h"

#include <cstdint>
#include <vector>

namespace whirl::server {

class SlotStart {
public:
    SlotStart(ServerModel& m, device::Backend& ops, const chat::ChatTokens& ct, EngineOptions& opt, std::uint32_t& ctx,
              CkptStore& ck, SpecState& spec, SampleBuffers& sb, std::vector<Slot>& slots, IdGen& ids,
              std::vector<std::uint32_t>& crlf_norm, PagePool& pool, PrefixStore& ps, TierBridge& tb, ImageEmbeddings& vis,
              DrafterCtx& dc, SlotPages& sc, GovernorGlue& gv, PrefillPhase& pf)
        : m_(m), ops_(ops), ct_(ct), opt_(opt), ctx_(ctx), ck_(ck), spec_(spec), sb_(sb), slots_(slots), ids_(ids),
          crlf_norm_(crlf_norm), pool_(pool), ps_(ps), tb_(tb), vis_(vis), dc_(dc), sc_(sc), gv_(gv), pf_(pf) {}
    SlotStart(const SlotStart&) = delete;
    SlotStart& operator=(const SlotStart&) = delete;

    bool startJob(Slot& sl, Job& job);

private:
    // prefill, called by name from the body above
    void finishPrefill(Slot& sl) { pf_.finishPrefill(sl); }

    ServerModel& m_;
    device::Backend& ops_;
    const chat::ChatTokens& ct_;
    EngineOptions& opt_;
    std::uint32_t& ctx_;
    CkptStore& ck_;
    SpecState& spec_;
    SampleBuffers& sb_;
    std::vector<Slot>& slots_;
    IdGen& ids_;
    std::vector<std::uint32_t>& crlf_norm_;
    PagePool& pool_;
    PrefixStore& ps_;
    TierBridge& tb_;
    ImageEmbeddings& vis_;
    DrafterCtx& dc_;
    SlotPages& sc_;
    GovernorGlue& gv_;
    PrefillPhase& pf_;
};

}  // namespace whirl::server
