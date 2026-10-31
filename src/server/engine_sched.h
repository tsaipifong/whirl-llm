// Server engine, the scheduler (arch025-engine S15b, module server.scheduler): runLoop and its phases idleWait /
// admitQueued / startAdmitted, and logBatchStats. It only calls the components below it each loop iteration
// (decodeCycle, runPrefill, the tier glue, the end and start of a request). References only - core.shared members of
// Engine, the queue state of jq_ under its old names (read; locked as jq_.q_mutex_), every component below and the placed state last_tick_; bodies
// verbatim from engine.cpp (DESIGN v3 L9, 2.2).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "cache/page_pool.h"
#include "prefix_store.h"
#include "tier_bridge.h"
#include "server/engine_api.h"
#include "server/engine_cycle.h"
#include "server/engine_end.h"
#include "server/engine_gov.h"
#include "server/engine_options.h"
#include "server/engine_pages.h"
#include "server/engine_prefill.h"
#include "server/engine_queue.h"
#include "server/engine_slot.h"
#include "server/engine_start.h"
#include "server/engine_state.h"
#include "server/engine_thread.h"
#include "server/engine_tierglue.h"
#include "server/engine_vision.h"
#include "server/engine_vram.h"
#include "whirl/thread_annotations.h"
#include "whirl/wddm_watch.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace whirl::server {

class Scheduler {
public:
    Scheduler(ServerModel& m, EngineOptions& opt, EngineThread& owner, SpecState& spec, std::vector<Slot>& slots,
              BatchStats& bst, std::function<wddmwatch::Sample()>& vram_probe, PagePool& pool, TimePoint& last_tick,
              PrefixStore& ps, TierBridge& tb, ImageEmbeddings& vis, SlotPages& sc, GovernorGlue& gv, VramWatch& vw,
              JobQueue& jq, SlotEnd& se, PrefillPhase& pf, SpecCycle& cyc, SlotStart& ss, TierGlue& tg)
        : m_(m), opt_(opt), owner_(owner), spec_(spec), slots_(slots), bst_(bst), vram_probe_(vram_probe), pool_(pool),
          last_tick_(last_tick), ps_(ps), tb_(tb), vis_(vis), sc_(sc), gv_(gv), vw_(vw), jq_(jq), se_(se), pf_(pf),
          cyc_(cyc), ss_(ss), tg_(tg), q_cond_(jq.q_cond_), queue_(jq.queue_), n_building_(jq.n_building_),
          stop_(jq.stop_), loop_calls_(jq.loop_calls_), tier_wake_(jq.tier_wake_), wake_seq_(jq.wake_seq_) {}
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    // Main loop (returns after stop()); see Engine::runLoop
    void runLoop();
    void logBatchStats(bool force);

private:
    // runLoop's phases in call order (arch025-E2: extracted verbatim)
    bool idleWait(whirl::UniqueLock& lk, bool& was_busy) WH_REQUIRES(jq_.q_mutex_);
    std::vector<Job*> admitQueued(std::uint32_t n_busy) WH_REQUIRES(jq_.q_mutex_);  // q_mutex_ held by the caller
    void startAdmitted(const std::vector<Job*>& admit);

    // the components below, called by name from the bodies above
    void runLoopCalls(whirl::UniqueLock& lk) WH_REQUIRES(jq_.q_mutex_) { jq_.runLoopCalls(lk); }
    void abortQueued(whirl::UniqueLock& lk) WH_REQUIRES(jq_.q_mutex_) { jq_.abortQueued(lk); }
    void decodeCycle(std::span<Slot* const> act_in) { cyc_.decodeCycle(act_in); }
    void failSlot(Slot& sl, const std::string& err) { se_.failSlot(sl, err); }
    void releaseSlot(Slot& sl) { se_.releaseSlot(sl); }
    void reapDisconnected() { se_.reapDisconnected(); }
    void runPrefill(Slot* pf, bool& idle_wait, std::size_t& pf_rows, bool& floor_pf) {
        pf_.runPrefill(pf, idle_wait, pf_rows, floor_pf);
    }
    void noteFloor(TimePoint tl2, TimePoint tl3, std::size_t pf_rows, bool floor_pf) {
        pf_.noteFloor(tl2, tl3, pf_rows, floor_pf);
    }
    bool startJob(Slot& sl, Job& job) { return ss_.startJob(sl, job); }
    bool waitsForTier(Job& job) { return tg_.waitsForTier(job); }
    void prefetchFor(Job& job) { tg_.prefetchFor(job); }
    bool tryRestore(Slot& sl, Job& job) { return tg_.tryRestore(sl, job); }
    void pollRestores() { tg_.pollRestores(); }
    bool tierTick() { return tg_.tierTick(); }
    void flushTierOnStop() { tg_.flushTierOnStop(); }

    ServerModel& m_;
    EngineOptions& opt_;
    EngineThread& owner_;
    SpecState& spec_;
    std::vector<Slot>& slots_;
    BatchStats& bst_;
    std::function<wddmwatch::Sample()>& vram_probe_;
    PagePool& pool_;
    TimePoint& last_tick_;
    PrefixStore& ps_;
    TierBridge& tb_;
    ImageEmbeddings& vis_;
    SlotPages& sc_;
    GovernorGlue& gv_;
    VramWatch& vw_;
    JobQueue& jq_;
    SlotEnd& se_;
    PrefillPhase& pf_;
    SpecCycle& cyc_;
    SlotStart& ss_;
    TierGlue& tg_;
    std::condition_variable& q_cond_;
    // JobQueue's state the loop reads, under its old names, guarded by jq_.q_mutex_ (the lock the loop holds; S10b:
    // the writes are JobQueue's methods), so the analysis checks every access here
    std::deque<Job*>& queue_ WH_GUARDED_BY(jq_.q_mutex_);
    std::atomic<std::uint32_t>& n_building_;
    bool& stop_ WH_GUARDED_BY(jq_.q_mutex_);
    std::vector<std::function<void()>>& loop_calls_ WH_GUARDED_BY(jq_.q_mutex_);
    bool& tier_wake_ WH_GUARDED_BY(jq_.q_mutex_);
    std::uint64_t& wake_seq_ WH_GUARDED_BY(jq_.q_mutex_);
};

}  // namespace whirl::server
