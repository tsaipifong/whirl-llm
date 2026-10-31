// Server engine, the end of a request (arch025-engine S11, module server.slot_end): finishJob (the response, the
// cache and tier hand-off, the request log), releaseSlot, failSlot and the disconnected-client reaper. References
// only - core.shared members of Engine, the queue state of jq_ under its old names, slot_cache, vram_watch and the
// placed state last_reap_; bodies verbatim from engine.cpp / engine_queue.cpp (DESIGN v3 L5).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend.h"
#include "device/backend.h"
#include "server/engine_api.h"
#include "server/engine_options.h"
#include "server/engine_pages.h"
#include "server/engine_queue.h"
#include "server/engine_sampling.h"
#include "server/engine_slot.h"
#include "server/engine_vision.h"
#include "server/engine_vram.h"
#include "tier_bridge.h"
#include "whirl/thread_annotations.h"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace whirl::server {

class SlotEnd {
public:
    SlotEnd(ServerModel& m, device::Backend& ops, EngineOptions& opt, SampleBuffers& sb, SlotPages& sc,
            std::vector<Slot>& slots, TierBridge& tb, ImageEmbeddings& vis, VramWatch& vw, TimePoint& last_reap,
            JobQueue& jq)
        : m_(m), ops_(ops), opt_(opt), sb_(sb), sc_(sc), slots_(slots), tb_(tb), vis_(vis), vw_(vw),
          last_reap_(last_reap), q_mutex_(jq.q_mutex_), q_cond_(jq.q_cond_), queue_(jq.queue_), n_active_(jq.n_active_) {}
    SlotEnd(const SlotEnd&) = delete;
    SlotEnd& operator=(const SlotEnd&) = delete;

    void finishJob(Slot& sl);
    void releaseSlot(Slot& sl);
    void failSlot(Slot& sl, const std::string& err);
    // FR-SRV P2-2: queued / running requests whose client disconnected end (slot freed)
    void reapDisconnected();

private:
    ServerModel& m_;
    device::Backend& ops_;
    EngineOptions& opt_;
    SampleBuffers& sb_;
    SlotPages& sc_;
    std::vector<Slot>& slots_;
    TierBridge& tb_;
    ImageEmbeddings& vis_;
    VramWatch& vw_;
    TimePoint& last_reap_;
    whirl::Mutex& q_mutex_;
    std::condition_variable& q_cond_;
    std::deque<Job*>& queue_;
    std::uint32_t& n_active_;
};

}  // namespace whirl::server
