// Server engine, the decode / speculative cycle (arch025-engine S13, module server.spec_cycle): decodeCycle, its
// Cycle and the thirteen phases in call order, the draft budget (pickDrafts / splitDrafts / draftCap) and the relax
// rules. References only - core.shared members of Engine, drafter_ctx, slot_end and the placed state spec_on_ (read;
// allocState writes it once before the loop) / prefill_tok_cum_ (read); bodies verbatim from engine_decode.cpp
// (DESIGN v3 L7).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "device/backend.h"
#include "sampling/acceptor.h"
#include "server/engine_api.h"
#include "server/engine_drafter.h"
#include "server/engine_end.h"
#include "server/engine_options.h"
#include "server/engine_pages.h"
#include "server/engine_vram.h"
#include "server/engine_sampling.h"
#include "server/engine_slot.h"
#include "server/engine_state.h"
#include "whirl/chat_tokens.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace whirl::server {

class SpecCycle {
public:
    SpecCycle(ServerModel& m, device::Backend& ops, const chat::ChatTokens& ct, EngineOptions& opt, std::uint32_t& ctx,
              SpecState& spec, SampleBuffers& sb, bool& spec_on, std::vector<Slot>& slots, BatchStats& bst,
              std::uint64_t& prefill_tok_cum, DrafterCtx& dc, SlotEnd& se, SlotPages& sc, VramWatch& vw)
        : m_(m), ops_(ops), ct_(ct), opt_(opt), ctx_(ctx), spec_(spec), sb_(sb), spec_on_(spec_on), slots_(slots),
          bst_(bst), prefill_tok_cum_(prefill_tok_cum), dc_(dc), se_(se), sc_(sc), vw_(vw) {}
    SpecCycle(const SpecCycle&) = delete;
    SpecCycle& operator=(const SpecCycle&) = delete;

    std::uint32_t pickDrafts(std::span<Slot* const> act);
    bool relaxRepeats(const Slot& sl, std::span<const std::uint32_t> dr, std::uint32_t acc) const;
    void decodeCycle(std::span<Slot* const> act_in);
    // decodeCycle's phases in call order (arch025-E2: extracted verbatim). Cycle = the values one cycle passes from
    // phase to phase (engine_decode.cpp). sampleRows holds the cycle's one host sync (readCtl).
    struct Cycle;
    bool relaxes(const Cycle& cy, std::size_t k) const;
    std::uint32_t verifyRoom(const Slot& sl, std::uint32_t lim) const;
    void dropOutOfRoom(Cycle& cy, std::span<Slot* const> act_in);
    void draftLimits(Cycle& cy);
    void ngramDrafts(Cycle& cy);
    void dflashDrafts(Cycle& cy);
    void mtpDrafts(Cycle& cy);
    void layoutBatch(Cycle& cy);
    void draftSampling(Cycle& cy);
    void enqueueVerify(Cycle& cy);
    void sampleRows(Cycle& cy);
    void trainCycle(Cycle& cy);
    void cycleStats(Cycle& cy);
    void acceptSlot(Cycle& cy, std::size_t k, const sampling::Acceptor& acceptor);
    void finishCycle(Cycle& cy);
    void splitDrafts(std::span<Slot* const> act, std::uint32_t nd, std::uint32_t rows, std::span<std::uint32_t> nd_m);
    std::uint32_t draftCap(std::uint32_t A);
    // v0252-hostwait (PLAN V1-61, EngineOptions::cycle_sync): set the mode up once, from Engine::allocState after the
    // sampling buffers: unknown bits dropped with a warning and the effective mode written back to opt_.cycle_sync (bit 1
    // adds bit 0), the pinned readback mirrors + event in sb_ (SampleBuffers::allocPinned, bit 0), the model's early
    // submission (ServerModel::setKick, bits 2 / 3). cycle_sync 0: nothing is allocated or called (the old cycle).
    // v0252-hostwait2 (REVIEW-v0252-hostwait D1): no member of its own - the resources live in SampleBuffers (sb_) and in
    // the model, the counters in BatchStats (bst_): references only, as the module rule says.
    void initSync();

private:
    // slot_end, called by name from the bodies above
    void finishJob(Slot& sl) { se_.finishJob(sl); }
    // v0252-hostwait: the cycle's readbacks and its one wait (sampleRows): bit 0 enqueues the control words
    // (ServerModel::readCtlAsync) and the sampling / relaxacc readbacks into sb_'s pinned mirrors on the model stream,
    // records sb_.readbackEvent() and waits once (bit 1: polls it); otherwise readCtl + the blocking downloads, as before.
    // Returns whether the pinned path ran (the caller copies the pinned data into the host vectors after its profile
    // timestamp).
    bool readbackEnqueueWait(bool any_sampling, bool any_relax, std::size_t ctl_n, std::size_t relax_n);
    void spinWait(device::Event e);
    // v0252-hostwait bits 2 / 3: the model hands the launches it queued on its stream to the GPU (ServerModel::kick)
    void kick();

    ServerModel& m_;
    device::Backend& ops_;
    const chat::ChatTokens& ct_;
    EngineOptions& opt_;
    std::uint32_t& ctx_;
    SpecState& spec_;
    SampleBuffers& sb_;
    bool& spec_on_;
    std::vector<Slot>& slots_;
    BatchStats& bst_;
    std::uint64_t& prefill_tok_cum_;
    DrafterCtx& dc_;
    SlotEnd& se_;
    SlotPages& sc_;
    VramWatch& vw_;
};

}  // namespace whirl::server
