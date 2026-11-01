// whirl-iso-server-job-queue-tests: module server.job_queue (arch025-engine S10 / S10b, progress/arch025-engine.DESIGN.md
// v3 2.3 L1): src/server/engine_queue.h (class JobQueue, the queue state q_mutex_ .. wake_seq_) and
// src/server/engine_queue.cpp, built with only its declared dependencies (core.shared: server.slot for Slot, the protocol's
// errorResponse for the 503 text) - not engine.h (ARCH-1 8.5 isolation, arch025-regb2 / PLAN N4-reg b2).
// The checks pin what the queue's callers rely on, read from engine_queue.cpp:
// - layout: JobQueue holds a reference to Engine's slots, is not copyable; the constructor takes exactly the DESIGN v3
//   3.1 reference list (slots_);
// - nextJobId counts from 1; beginBuilding / endBuilding count n_building_ and endBuilding bumps wake_seq_;
// - submitAndWait after stop(): answered 503 on the connection, not queued, n_building_ released (FR-SRV P3-1);
// - jqAdmit / jqStartAdmitted / jqRelease: the active count and the requeue order (item T3);
// - jqReap drops only queued requests whose client is gone; abortQueued answers every queued one 503;
// - jqHealth reads active / queued / slots; runOnLoop runs through runLoopCalls, and a timed-out call that the loop never
//   claimed throws and never runs; jqTierWake sets tier_wake_ and bumps wake_seq_.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/engine_queue.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

namespace s = whirl::server;
using Q = s::JobQueue;

static_assert(!std::is_copy_constructible_v<Q> && !std::is_copy_assignable_v<Q>);
static_assert(std::is_constructible_v<Q, std::vector<s::Slot>&>);
static_assert(!std::is_default_constructible_v<Q>);
static_assert(std::is_same_v<decltype(&Q::nextJobId), std::uint64_t (Q::*)()>);
static_assert(std::is_same_v<decltype(&Q::submitAndWait), void (Q::*)(s::Job&)>);
static_assert(std::is_same_v<decltype(&Q::jqHealth), void (Q::*)(s::EngineApi::Health&)>);

class FakeConn final : public s::Conn {
public:
    bool write(std::string_view d) override {
        out.append(d);
        return true;
    }
    bool flush() override {
        ++flushes;
        return true;
    }
    bool alive() override { return live; }
    std::string out;
    int flushes = 0;
    bool live = true;
};

void testIdsAndBuilding() {
    std::vector<s::Slot> slots(2);
    Q q(slots);
    ISO_CHECK(q.nextJobId() == 1);
    ISO_CHECK(q.nextJobId() == 2);
    q.beginBuilding();
    q.beginBuilding();
    ISO_CHECK(q.n_building_.load() == 2);
    std::uint64_t seq0;
    {
        whirl::MutexLock lk(q.q_mutex_);
        seq0 = q.wake_seq_;
    }
    q.endBuilding();
    ISO_CHECK(q.n_building_.load() == 1);
    whirl::MutexLock lk(q.q_mutex_);
    ISO_CHECK(q.wake_seq_ == seq0 + 1);
}

void testSubmitAfterStop() {
    std::vector<s::Slot> slots(1);
    Q q(slots);
    q.stop();
    FakeConn c;
    s::Job j;
    j.w = &c;
    q.beginBuilding();
    q.submitAndWait(j);  // returns at once: nobody would run it
    ISO_CHECK(q.n_building_.load() == 0);
    ISO_CHECK(c.out.find("503") != std::string::npos && c.out.find("server is shutting down") != std::string::npos);
    ISO_CHECK(c.flushes == 1);
    whirl::MutexLock lk(q.q_mutex_);
    ISO_CHECK(q.queue_.empty() && q.stop_);
}

void testAdmitRequeueRelease() {
    std::vector<s::Slot> slots(2);
    Q q(slots);
    s::Job a, b, c;
    a.id = 1;
    b.id = 2;
    c.id = 3;
    std::vector<s::Job*> admit;
    {
        whirl::MutexLock lk(q.q_mutex_);
        q.queue_ = {&a, &b, &c};
        q.jqAdmit(1, admit);  // b starts
        q.jqAdmit(0, admit);  // a starts
        ISO_CHECK(q.n_active_ == 2 && q.queue_.size() == 1 && q.queue_[0] == &c);
    }
    ISO_CHECK(admit.size() == 2 && admit[0] == &b && admit[1] == &a);
    std::size_t n_requeued = 0;
    q.jqStartAdmitted(&b, n_requeued);  // back to the queue, in front
    q.jqStartAdmitted(&a, n_requeued);  // behind b, in front of c
    {
        whirl::MutexLock lk(q.q_mutex_);
        ISO_CHECK(n_requeued == 2 && q.n_active_ == 0);
        ISO_CHECK(q.queue_.size() == 3 && q.queue_[0] == &b && q.queue_[1] == &a && q.queue_[2] == &c);
        q.n_active_ = 1;
    }
    q.jqRelease(&c);
    whirl::MutexLock lk(q.q_mutex_);
    ISO_CHECK(c.done && q.n_active_ == 0);
}

void testReapAndAbort() {
    std::vector<s::Slot> slots(1);
    Q q(slots);
    FakeConn live, gone;
    gone.live = false;
    s::Job a, b, n;
    a.w = &live;
    b.w = &gone;  // n.w == nullptr: kept (no client to ask)
    {
        whirl::MutexLock lk(q.q_mutex_);
        q.queue_ = {&a, &b, &n};
    }
    q.jqReap();
    {
        whirl::MutexLock lk(q.q_mutex_);
        ISO_CHECK(q.queue_.size() == 2 && q.queue_[0] == &a && q.queue_[1] == &n);
    }
    ISO_CHECK(b.done && !a.done && !n.done && gone.out.empty());
    whirl::UniqueLock lk(q.q_mutex_);
    q.abortQueued(lk);
    ISO_CHECK(q.queue_.empty() && a.done && n.done);
    ISO_CHECK(live.out.find("503") != std::string::npos && live.flushes == 1);
}

void testHealthAndTierWake() {
    std::vector<s::Slot> slots(3);
    Q q(slots);
    s::Job a;
    {
        whirl::MutexLock lk(q.q_mutex_);
        q.queue_ = {&a};
        q.n_active_ = 2;
    }
    s::EngineApi::Health h;
    q.jqHealth(h);
    ISO_CHECK(h.active == 2 && h.queued == 1 && h.slots == 3);
    q.jqTierWake();
    whirl::UniqueLock lk(q.q_mutex_);
    ISO_CHECK(q.tier_wake_ && q.wake_seq_ == 1);
    q.jqTierWakeClear();
    ISO_CHECK(!q.tier_wake_);
}

void testRunOnLoop() {
    std::vector<s::Slot> slots(1);
    Q q(slots);
    // a loop that never runs the call: the caller times out, the call is cancelled and never runs
    bool ran = false;
    bool threw = false;
    try {
        q.runOnLoop([&] { ran = true; }, std::chrono::milliseconds(1));
    } catch (const std::runtime_error&) {
        threw = true;
    }
    ISO_CHECK(threw);
    {
        whirl::UniqueLock lk(q.q_mutex_);
        ISO_CHECK(q.loop_calls_.size() == 1);
        q.runLoopCalls(lk);  // claims the cancelled call: it does not run
        ISO_CHECK(q.loop_calls_.empty());
    }
    ISO_CHECK(!ran);
    // a loop thread that runs the queued calls: the caller returns after fn ran
    std::thread loop([&] {
        for (int i = 0; i < 2000; ++i) {
            whirl::UniqueLock lk(q.q_mutex_);
            if (!q.loop_calls_.empty()) {
                q.runLoopCalls(lk);
                return;
            }
            q.jqWaitUntil(lk, std::chrono::steady_clock::now() + std::chrono::milliseconds(5));
        }
    });
    int v = 0;
    q.runOnLoop([&] { v = 7; }, std::chrono::seconds(10));
    loop.join();
    ISO_CHECK(v == 7);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.job_queue",
                    {{"ids_building", testIdsAndBuilding},
                     {"submit_after_stop", testSubmitAfterStop},
                     {"admit_requeue_release", testAdmitRequeueRelease},
                     {"reap_abort", testReapAndAbort},
                     {"health_tier_wake", testHealthAndTierWake},
                     {"run_on_loop", testRunOnLoop}},
                    argc, argv);
}
