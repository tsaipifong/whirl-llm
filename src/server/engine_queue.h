// Server engine, the request queue (arch025-engine S10, module server.job_queue): the queue state (the former Engine
// members q_mutex_ .. wake_seq_, in that order) and the methods the connection threads call (submitAndWait,
// beginBuilding / endBuilding, nextJobId, stop, runOnLoop) plus the loop's runLoopCalls / abortQueued. q_mutex_
// guards the queue state. Engine keeps references under the old names until the scheduler / slot_end / tier_glue
// modules take their parts (DESIGN v3 L1).
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
    void runLoopCalls(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);
    void abortQueued(whirl::UniqueLock& lk) WH_REQUIRES(q_mutex_);

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
