// Server engine, the request queue (arch025-N4e, from engine.cpp): stop, the test hook runOnLoop, submitAndWait /
// endBuilding for the connection threads, jqHealth, and the queue steps of the other modules (arch025-engine S10b:
// runLoopCalls, abortQueued, jqAdmit, jqStartAdmitted, jqRelease, jqReap, jqTierWake) and the loop's condition waits
// (arch025-engreview E2: jqWaitBurst, jqWaitTierBusy, jqWaitVision, jqWaitUntil, jqWaitIdle). q_mutex_ guards the queue
// state (engine_queue.h).
// SPDX-License-Identifier: Apache-2.0

// arch025-regb1 (N4-reg b1): the component's own header, not engine.h / engine_internal.h (no include edge back to server.engine)
#include "server/engine_queue.h"

#include "cache/tier_adapter.h"
#include "log.h"
#include "whirl/dflash_policy.h"
#include "spec/spec_rules.h"  // the speculative-decoding rules shared with the CLI (arch025-R12cli)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <future>
#include <stdexcept>
#include <thread>

namespace whirl::server {

void JobQueue::stop() {
    whirl::MutexLock lk(q_mutex_);
    stop_ = true;
    q_cond_.notify_all();
}

void JobQueue::runOnLoop(std::function<void()> fn, std::chrono::milliseconds timeout) {
    auto done = std::make_shared<std::promise<void>>();
    // REVIEW-ARCH4 P3-5: 0 queued, 1 claimed by the loop, 2 cancelled by a timed-out caller. fn usually captures the
    // caller's locals by reference, so it may only run while the caller waits: the loop claims the call before
    // running it, and a timed-out caller either cancels it (never runs) or, when already claimed, waits for it.
    auto state = std::make_shared<std::atomic<int>>(0);
    std::future<void> f = done->get_future();
    {
        whirl::MutexLock lk(q_mutex_);
        loop_calls_.push_back([fn = std::move(fn), done, state] {
            int queued = 0;
            if (!state->compare_exchange_strong(queued, 1)) return;  // cancelled: the caller is gone
            try {
                fn();
                done->set_value();
            } catch (...) {
                done->set_exception(std::current_exception());
            }
        });
        ++wake_seq_;
        q_cond_.notify_all();
    }
    if (f.wait_for(timeout) != std::future_status::ready) {
        int queued = 0;
        if (state->compare_exchange_strong(queued, 2)) throw std::runtime_error("Engine::runOnLoop: the engine loop did not run the call");
        f.wait();  // claimed just now: fn is running on the engine thread with the caller's references
    }
    f.get();
}

// called with lk held; runs the queued test calls without the lock (unlocks / relocks through WH_UNLOCK_HELD /
// WH_RELOCK_HELD, so the body is checked: arch025-tsaidle)
void JobQueue::runLoopCalls(whirl::UniqueLock& lk) {
    if (loop_calls_.empty()) return;
    std::vector<std::function<void()>> calls;
    calls.swap(loop_calls_);
    WH_UNLOCK_HELD(lk, q_mutex_);
    for (auto& f : calls) f();
    WH_RELOCK_HELD(lk, q_mutex_);
}

void JobQueue::abortQueued(whirl::UniqueLock& lk) {
    // shutting down: queued requests are answered 503
    for (Job* job : queue_) {
        const std::string r = errorResponse(503, "server is shutting down");
        if (job->w) {
            job->w->write(r);
            job->w->flush();
        }
        job->done = true;
    }
    queue_.clear();
    q_cond_.notify_all();
    (void)lk;
}

// admitQueued (engine_sched.cpp): queue_[qi] starts this iteration (q_mutex_ held by the caller)
void JobQueue::jqAdmit(std::size_t qi, std::vector<Job*>& admit) {
    admit.push_back(queue_[qi]);
    queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(qi));
    n_active_ += 1;
}

// startAdmitted (engine_sched.cpp, item T3): back to the queue, behind the ones requeued before it
void JobQueue::jqStartAdmitted(Job* job, std::size_t& n_requeued) {
    whirl::MutexLock g(q_mutex_);
    queue_.insert(queue_.begin() + static_cast<std::ptrdiff_t>(n_requeued), job);
    ++n_requeued;
    n_active_ -= 1;
}

// releaseSlot (engine_end.cpp): the request is done, one active request less
void JobQueue::jqRelease(Job* job) {
    whirl::MutexLock lk(q_mutex_);
    job->done = true;
    n_active_ -= 1;
    q_cond_.notify_all();
}

// reapDisconnected (engine_end.cpp): queued requests whose client disconnected
void JobQueue::jqReap() {
    // queued: dropped without a response (nobody reads it)
    whirl::MutexLock lk(q_mutex_);
    for (std::size_t i = 0; i < queue_.size();) {
        Job* job = queue_[i];
        if (job->w == nullptr || job->w->alive()) {
            ++i;
            continue;
        }
        logW("req {} | client disconnected while queued; request dropped", job->id);
        queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(i));
        job->done = true;
        q_cond_.notify_all();
    }
}

// attachTier (engine_tier.cpp): the tier IO thread's wake callback
void JobQueue::jqTierWake() {
    whirl::MutexLock lk(q_mutex_);
    tier_wake_ = true;
    ++wake_seq_;
    q_cond_.notify_all();
}

// runLoop (engine_sched.cpp): burst gather / host-tier restore: 1 ms, sooner at a wake_seq_ bump (arch025-engreview E2)
void JobQueue::jqWaitBurst(std::uint64_t wake_seq0) {
    whirl::UniqueLock lw(q_mutex_);
    q_cond_.wait_for(lw, std::chrono::milliseconds(1), [&]() WH_REQUIRES(q_mutex_) { return wake_seq_ != wake_seq0; });
}

// idleWait (engine_sched.cpp, q_mutex_ held by the caller): the four waits of the idle loop, verbatim
void JobQueue::jqWaitTierBusy(whirl::UniqueLock& lk, std::uint64_t seq) {
    q_cond_.wait_for(lk, std::chrono::milliseconds(5),
                     [&]() WH_REQUIRES(q_mutex_) { return wake_seq_ != seq || !queue_.empty() || stop_ || !loop_calls_.empty(); });
}

void JobQueue::jqWaitVision(whirl::UniqueLock& lk) {
    q_cond_.wait_for(lk, std::chrono::milliseconds(100), [&]() WH_REQUIRES(q_mutex_) { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });
}

void JobQueue::jqWaitUntil(whirl::UniqueLock& lk, std::chrono::steady_clock::time_point due) {
    q_cond_.wait_until(lk, due, [&]() WH_REQUIRES(q_mutex_) { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });
}

void JobQueue::jqWaitIdle(whirl::UniqueLock& lk) {
    q_cond_.wait(lk, [&]() WH_REQUIRES(q_mutex_) { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });
}

void JobQueue::submitAndWait(Job& job) {
    whirl::UniqueLock lk(q_mutex_);
    if (stop_) {
        // FR-SRV P3-1: shutting down (the loop may be flushing the tier, or gone): nobody would run
        // this job; answered here instead of waiting forever on a finished engine
        n_building_.fetch_sub(1);
        ++wake_seq_;
        q_cond_.notify_all();
        lk.unlock();
        if (job.w) {
            job.w->write(errorResponse(503, "server is shutting down"));
            job.w->flush();
        }
        return;
    }
    const std::size_t free = slots_.size() - std::min<std::size_t>(slots_.size(), n_active_);
    const std::size_t waiting = queue_.size();
    queue_.push_back(&job);
    n_building_.fetch_sub(1);
    ++wake_seq_;
    q_cond_.notify_all();
    lk.unlock();
    if (waiting >= free)
        logI("req {} | queued ({} request(s) waiting ahead, all {} slots busy)", job.id, waiting - free, slots_.size());
    lk.lock();
    q_cond_.wait(lk, [&] { return job.done; });
}

// Engine::health: the q_mutex_ lock and the three reads (arch025-engine S16, verbatim from engine.cpp)
void JobQueue::jqHealth(EngineApi::Health& h) {
    whirl::MutexLock lk(q_mutex_);
    h.active = n_active_;
    h.queued = queue_.size();
    h.slots = slots_.size();
}

void JobQueue::endBuilding() {
    // a connection that did not queue a request (bad request, probe): a burst
    // gather waiting for it ends now
    whirl::MutexLock lk(q_mutex_);
    n_building_.fetch_sub(1);
    ++wake_seq_;
    q_cond_.notify_all();
}

}  // namespace whirl::server
