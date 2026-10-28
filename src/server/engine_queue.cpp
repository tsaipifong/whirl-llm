// Server engine, the request queue (arch025-N4e, from engine.cpp): stop, the test hook runOnLoop, submitAndWait /
// endBuilding for the connection threads, abortQueued at shutdown and the disconnected-client reaper. q_mutex_ guards
// the queue state (engine.h).
// SPDX-License-Identifier: Apache-2.0

#include "engine.h"
#include "engine_internal.h"

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

void Engine::stop() {
    whirl::MutexLock lk(q_mutex_);
    stop_ = true;
    q_cond_.notify_all();
}

void Engine::runOnLoop(std::function<void()> fn, std::chrono::milliseconds timeout) {
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
void Engine::runLoopCalls(whirl::UniqueLock& lk) {
    if (loop_calls_.empty()) return;
    std::vector<std::function<void()>> calls;
    calls.swap(loop_calls_);
    WH_UNLOCK_HELD(lk, q_mutex_);
    for (auto& f : calls) f();
    WH_RELOCK_HELD(lk, q_mutex_);
}

void Engine::reapDisconnected() {
    if (msSince(last_reap_) < 50) return;
    last_reap_ = Clock::now();
    {
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
    for (Slot& sl : slots_) {
        if (sl.job == nullptr || sl.job->w == nullptr || sl.st.gone) continue;
        if (sl.phase != Phase::prefill && sl.phase != Phase::decode) continue;
        if (sl.job->w->alive()) continue;
        if (sl.phase == Phase::decode) {
            sl.st.fail();  // the decode cycle finishes it ("client disconnected")
            continue;
        }
        // prefilling: nothing was sent yet; the partial prefill is dropped
        logW("req {} | slot {} | client disconnected during prefill ({} of {} prompt tokens); slot freed", sl.job->id, sl.id,
             sl.reuse + sl.pf_off, sl.n_prompt);
        dropCache(sl);
        releaseSlot(sl);
    }
}

void Engine::submitAndWait(Job& job) {
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

void Engine::endBuilding() {
    // a connection that did not queue a request (bad request, probe): a burst
    // gather waiting for it ends now
    whirl::MutexLock lk(q_mutex_);
    n_building_.fetch_sub(1);
    ++wake_seq_;
    q_cond_.notify_all();
}

void Engine::abortQueued(whirl::UniqueLock& lk) {
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

}  // namespace whirl::server
