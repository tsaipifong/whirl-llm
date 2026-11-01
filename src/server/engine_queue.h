// Server engine, the request queue (arch025-engine S10, module server.job_queue): the queue state (the former Engine
// members q_mutex_ .. wake_seq_, in that order) and the methods the connection threads call (submitAndWait,
// beginBuilding / endBuilding, nextJobId, stop, runOnLoop, jqHealth) plus every write of the queue state by the other
// modules (S10b: runLoopCalls / abortQueued / jqAdmit / jqTierWakeClear with the caller's lock, jqStartAdmitted /
// jqRelease / jqReap / jqTierWake take it themselves). q_mutex_ guards the queue state; the scheduler names it
// jq_.q_mutex_ (DESIGN v3 L1, 4 S10 / S10b). The loop's condition waits are methods too (arch025-engreview E2, DESIGN
// v3 10.2: jqWaitTierBusy / jqWaitVision / jqWaitUntil / jqWaitIdle with the caller's lock, jqWaitBurst takes it).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine_api.h"
#include "engine_slot.h"
#include "whirl/thread_annotations.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

namespace whirl::server {

class JobQueue {
public:
    explicit JobQueue(std::vector<Slot>& slots) : slots_(slots) {}
    JobQueue(const JobQueue&) = delete;
    JobQueue& operator=(const JobQueue&) = delete;

    std::uint64_t nextJobId() { return next_id_.fetch_add(1); }
    void submitAndWait(Job& job);
    void beginBuilding() { n_building_.fetch_add(1); }
    void endBuilding();
    void stop();
    void runOnLoop(std::function<void()> fn, std::chrono::milliseconds timeout);
    void jqHealth(EngineApi::Health& h);  // Engine::health: the q_mutex_ lock and the active / queued / slots reads

    // the loop's queue steps (Scheduler, with jq_.q_mutex_ held; arch025-engine S10b, bodies verbatim)
    void runLoopCalls(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);
    void abortQueued(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);
    void jqAdmit(std::size_t qi, std::vector<Job*>& admit) WH_REQUIRES(q_mutex_);  // admitQueued: queue_[qi] starts
    void jqTierWakeClear() WH_REQUIRES(q_mutex_) { tier_wake_ = false; }  // idleWait
    // the loop's condition waits (Scheduler::idleWait, the caller's lock; arch025-engreview E2, wait statements verbatim)
    void jqWaitTierBusy(whirl::UniqueLock& lk, std::uint64_t seq) WH_REQUIRES(q_mutex_);  // tier work in flight: 5 ms
    void jqWaitVision(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);  // the vision encoder's idle timeout: 100 ms
    void jqWaitUntil(whirl::UniqueLock& lk, std::chrono::steady_clock::time_point due) WH_REQUIRES(q_mutex_);  // a due time
    void jqWaitIdle(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);  // nothing due
    // the same, taking the lock themselves
    void jqStartAdmitted(Job* job, std::size_t& n_requeued);  // startAdmitted: back to the queue (item T3)
    void jqRelease(Job* job);  // releaseSlot: the request is done, one active less
    void jqReap();             // reapDisconnected: queued requests whose client disconnected are dropped
    void jqTierWake();         // attachTier: the tier IO thread's wake callback
    void jqWaitBurst(std::uint64_t wake_seq0);  // runLoop: burst gather / restore, 1 ms or a wake_seq_ bump (E2)

    // the queue state (Engine's loop, release and tier paths read and write it under q_mutex_)
    whirl::Mutex q_mutex_;
    std::condition_variable q_cond_;
    std::deque<Job*> queue_ WH_GUARDED_BY(q_mutex_);
    std::uint32_t n_active_ WH_GUARDED_BY(q_mutex_) = 0;
    std::atomic<std::uint32_t> n_building_{0};
    std::atomic<std::uint64_t> next_id_{1};
    bool stop_ WH_GUARDED_BY(q_mutex_) = false;
    std::vector<std::function<void()>> loop_calls_ WH_GUARDED_BY(q_mutex_);  // runOnLoop (tests)
    bool tier_wake_ WH_GUARDED_BY(q_mutex_) = false;
    std::uint64_t wake_seq_ WH_GUARDED_BY(q_mutex_) = 0;  // bumped by a queued request, a finished build, the tier IO thread

private:
    std::vector<Slot>& slots_;
};

}  // namespace whirl::server
