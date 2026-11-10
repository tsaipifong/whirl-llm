// Server engine, the scheduler (arch025-engine S15b, module server.scheduler): runLoop and its phases idleWait /
// admitQueued / startAdmitted, and logBatchStats; bodies verbatim from engine.cpp (S16 file move).
// SPDX-License-Identifier: Apache-2.0

// arch025-regb1 (N4-reg b1): the component's own header, not engine.h / engine_internal.h (no include edge back to server.engine)
#include "server/engine_sched.h"
#include "server/engine_prof.h"

#include "cache/tier_adapter.h"
#include "log.h"
#include "whirl/dflash_policy.h"
#include "spec/spec_rules.h"// the speculative-decoding rules shared with the CLI (arch025-R12cli)

#include <algorithm>
#include <cmath>
#include <cstring>
#include <ctime>
#include <limits>
#include <future>
#include <stdexcept>
#include <thread>

namespace whirl::server {

void Scheduler::logBatchStats(bool force) {
    const double el = msSince(bst_.t0);
    if (!force && el < 5000) return;
    if (bst_.cycles > 0 || bst_.prefill_tok > 0) {
        std::uint32_t n_dec = 0, n_pf = 0;
        for (const Slot& sl : slots_) {
            if (sl.phase == Phase::decode) ++n_dec;
            else if (sl.phase == Phase::prefill || sl.phase == Phase::restore) ++n_pf;
        }
        const double cyc = static_cast<double>(std::max<std::uint64_t>(bst_.cycles, 1));
        // FR-SRV P2-5: "drafts/cycle" is the MTP draft count; literal drafts (DFlash2 / n-gram) are
        // reported next to it (with --draft dflash the MTP figure is always 0)
        const std::string lit = bst_.lit_drafts > 0
                                    ? std::format(", {:.2f} literal drafts/cycle (DFlash2 / n-gram)", static_cast<double>(bst_.lit_drafts) / cyc)
                                    : std::string();
        // WDDM-LOG: this process's video memory at the same cadence (issues #5 / #6)
        const std::string vram = vw_.vramNote();
        logI("batch | slots busy {}/{} (decode {}, prefill {}) | {} cycles, {:.2f} slots/cycle, {:.2f} rows/cycle, {:.2f} "
             "drafts/cycle{} | aggregate tg {:.1f} tok/s, prefill {:.0f} tok/s over {:.1f} s{}{}",
             n_dec + n_pf, slots_.size(), n_dec, n_pf, bst_.cycles, static_cast<double>(bst_.slots) / cyc,
             static_cast<double>(bst_.rows) / cyc, static_cast<double>(bst_.drafts) / cyc, lit,
             static_cast<double>(bst_.tokens) * 1000.0 / el, static_cast<double>(bst_.prefill_tok) * 1000.0 / el, el / 1000.0,
             vram.empty() ? "" : " | ", vram);
        // v0252-hostwait2 (REVIEW-v0252-hostwait B4): what WHIRL_CYCLE_SYNC really did, counted since start (the GPU
        // runners check these, not only the configured value of the "decode: cycle sync" line)
        if (opt_.cycle_sync != 0) {
            const ServerModel::KickCount kc = m_.kickCount();
            logI("batch | cycle sync {}: {} pinned waits ({} event polls), {} blocking readCtl, {} early submissions asked, "
                 "model kicks {} after MTP steps + {} inside forwards (since start)",
                 opt_.cycle_sync, bst_.cs_pinned, bst_.cs_polls, bst_.cs_readctl, bst_.cs_kicks, kc.calls, kc.forward);
        }
    }
    bst_.t0 = Clock::now();
    bst_.tokens = bst_.cycles = bst_.rows = bst_.slots = bst_.drafts = bst_.lit_drafts = 0;
    if (bst_.seg_batches > 0)
        logI("batch | segmented prefill: {} forwards, {} chunks ({:.2f} chunks/forward)", bst_.seg_batches, bst_.seg_chunks,
             static_cast<double>(bst_.seg_chunks) / static_cast<double>(bst_.seg_batches));
    bst_.prefill_tok = bst_.seg_batches = bst_.seg_chunks = 0;
    pf_.floorLogPeriod();
}

void Scheduler::runLoop() {
    // arch024 M8: this thread owns the engine state until runLoop returns (startup before it: no owner, allowed)
    struct Owned {
        EngineThread& o;
        explicit Owned(EngineThread& t) : o(t) { o.enter(); }
        ~Owned() { o.leave(); }
    } owned(owner_);
    bst_.t0 = Clock::now();
    bool was_busy = false;
    for (;;) {
        const TimePoint tl0 = Clock::now();
        // v0.2.2.2 memory governor: its pending actions (safe here: no forward in flight)
        gv_.govApply();
        // ---- admit
        std::uint32_t n_busy = 0;
        for (const Slot& sl : slots_)
            if (sl.phase != Phase::idle) ++n_busy;
        whirl::UniqueLock lk(jq_.q_mutex_);
        runLoopCalls(lk);
        if (stop_ && n_busy == 0) {
            abortQueued(lk);
            lk.unlock();
            flushTierOnStop();
            lk.lock();
            runLoopCalls(lk);
            return;
        }
        // shutting down: no new requests are admitted while the running ones finish
        if (stop_ && !queue_.empty()) abortQueued(lk);
        if (n_busy == 0 && queue_.empty() && idleWait(lk, was_busy)) return;
        const std::uint64_t wake_seq0 = wake_seq_;  // a later arrival ends this iteration's wait (below)
        std::vector<Job*> admit = admitQueued(n_busy);
        // item T4: requests left in the queue (all slots busy, or waiting)
        std::vector<Job*> pf_jobs;
        if (tb_.tier() != nullptr) {
            for (Job* qj : queue_) {
                if (pf_jobs.size() == 8) break;
                if (qj->pf_checked) continue;
                pf_jobs.push_back(qj);
            }
        }
        lk.unlock();
        was_busy = true;
        for (Job* qj : pf_jobs) prefetchFor(*qj);
        startAdmitted(admit);
        const TimePoint tl1 = Clock::now();
        // ---- host-tier restores in progress / tier housekeeping (<= every 2 ms)
        if (tb_.tier() != nullptr) {
            pollRestores();
            if (msSince(last_tick_) >= 2) {
                tierTick();
                last_tick_ = Clock::now();
            }
        }
        reapDisconnected();
        // ---- one prefill chunk (oldest request first), batched with other
        // prefilling requests' chunks when that is bit-identical
        const TimePoint tl2 = Clock::now();
        Slot* pf = nullptr;
        for (Slot& sl : slots_) {
            if (sl.phase != Phase::prefill) continue;
            if (!pf || sl.job->id < pf->job->id) pf = &sl;
        }
        bool idle_wait = false;
        std::size_t pf_rows = 0;
        bool floor_pf = false;
        runPrefill(pf, idle_wait, pf_rows, floor_pf);
        // ---- one decode cycle over every decoding slot
        std::vector<Slot*> act;
        for (Slot& sl : slots_)
            if (sl.phase == Phase::decode) act.push_back(&sl);
        bool restoring = false;
        for (const Slot& sl : slots_) restoring = restoring || sl.phase == Phase::restore;
        const TimePoint tl3 = Clock::now();
        if (!act.empty()) {
            try {
                decodeCycle(act);
            } catch (const std::exception& ex) {
                for (Slot* sl : act)
                    if (sl->phase == Phase::decode) failSlot(*sl, ex.what());
            }
            if (pf_.floorOn()) noteFloor(tl2, tl3, pf_rows, floor_pf);
        } else if (idle_wait || (restoring && pf == nullptr)) {
            // burst gather / host-tier restore: poll again in 1 ms, sooner when a
            // request is queued, a connection finishes building one, or the tier
            // IO thread reports progress
            jq_.jqWaitBurst(wake_seq0);
        }
        if (opt_.loop_log && (!act.empty() || restoring || pf != nullptr)) {
            const TimePoint tl4 = Clock::now();
            auto d = [](TimePoint a, TimePoint b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
            logI("loop | admit {:.2f} | tier {:.2f} | pf {:.2f} | dec {:.2f} ms | A {} | restoring {} | prefill {}", d(tl0, tl1),
                 d(tl1, tl2), d(tl2, tl3), d(tl3, tl4), act.size(), restoring, pf != nullptr);
        }
        logBatchStats(false);
        vis_.idle();
    }
}

// arch025-E2: runLoop's phases (extracted verbatim; same call order, same lock state at every call)
// idle: no request running or queued. Called with lk held; true = the engine stops (runLoop returns).
// arch025-tsaidle: the body unlocks and relocks the caller's lock through WH_UNLOCK_HELD / WH_RELOCK_HELD (they name
// q_mutex_), so the body is checked by the analysis again, as inside runLoop before E2 (WH_NO_THREAD_SAFETY_ANALYSIS here
// let a dropped relock or a guarded read after the unlock pass; the wait predicates are lambdas, checked either way)
// arch025-engreview E2: the four condition waits are JobQueue's jqWaitTierBusy / jqWaitVision / jqWaitUntil / jqWaitIdle
bool Scheduler::idleWait(whirl::UniqueLock& lk, bool& was_busy) {
    if (was_busy) {
        WH_UNLOCK_HELD(lk, jq_.q_mutex_);
        logBatchStats(true);
        if (spec_.prof) spec_.prof->print(m_);
        was_busy = false;
        WH_RELOCK_HELD(lk, jq_.q_mutex_);
    }
    // idle: tier background work is polled; otherwise sleep until a
    // request arrives or the tier IO thread wakes us
    bool logged = false;
    while (queue_.empty()) {
        if (!loop_calls_.empty()) {
            runLoopCalls(lk);
            continue;
        }
        if (stop_) {
            abortQueued(lk);
            WH_UNLOCK_HELD(lk, jq_.q_mutex_);
            flushTierOnStop();
            WH_RELOCK_HELD(lk, jq_.q_mutex_);
            runLoopCalls(lk);
            return true;
        }
        // WDDM-LOG: /health "vram" and the spill-over line stay current while idle (one sample a minute)
        std::optional<TimePoint> vram_due;
        if (vram_probe_) {
            WH_UNLOCK_HELD(lk, jq_.q_mutex_);
            vram_due = vw_.vramIdleTick();
            gv_.govApply();
            WH_RELOCK_HELD(lk, jq_.q_mutex_);
            if (!queue_.empty() || stop_) continue;
        }
        if (tb_.tier() != nullptr) {
            const std::uint64_t seq = wake_seq_;
            WH_UNLOCK_HELD(lk, jq_.q_mutex_);
            tb_.idlePoll(n_building_.load() > 0);
            const bool busy = tierTick();
            if (busy) logged = false;
            if (!busy && !logged) {
                tb_.logStats(pool_.stats().cow_pages, pool_.stats().q_waits);
                logged = true;
            }
            WH_RELOCK_HELD(lk, jq_.q_mutex_);
            if (busy) {
                // tier work in flight: polled every 5 ms, but a request (or the
                // tier IO thread) ends the wait at once (a sleep here delayed a
                // request by up to one Windows timer tick, 10-16 ms)
                jq_.jqWaitTierBusy(lk, seq);
                continue;
            }
        }
        // vision: while the encoder is loaded, wake every 100 ms for its idle timeout
        // (a timed wait on the queue condition, not a sleep: a request that arrives
        // meanwhile starts at once; the prototype slept 100 ms here)
        if (vis_.wakeNeeded()) {
            WH_UNLOCK_HELD(lk, jq_.q_mutex_);
            vis_.idle();
            WH_RELOCK_HELD(lk, jq_.q_mutex_);
            jq_.jqTierWakeClear();
            if (queue_.empty() && !stop_)
                jq_.jqWaitVision(lk);
            continue;
        }
        jq_.jqTierWakeClear();
        std::optional<std::chrono::steady_clock::time_point> wake;
        if (tb_.tier() != nullptr) wake = tb_.tier()->nextWake();
        if (const auto idle_due = tb_.idleDue(); idle_due && (!wake || *idle_due < *wake)) wake = idle_due;  // TIER-IDLE
        if (vram_due && (!wake || *vram_due < *wake)) wake = *vram_due;  // WDDM-LOG idle sample
        if (wake) {
            // an SSD write is due at a known time, the periodic SSD expiry check, or the idle write: wake
            // then (re-checked every minute)
            const auto due = std::min(*wake + std::chrono::milliseconds(2), std::chrono::steady_clock::now() + std::chrono::seconds(60));
            jq_.jqWaitUntil(lk, due);
        } else {
            jq_.jqWaitIdle(lk);
        }
    }
    // a request arrived: the idle period ends, an idle write stops starting new entries (TIER-IDLE)
    tb_.idleActivity();
    if (tb_.tier() != nullptr) tb_.tier()->endIdleWrite();
    bst_.t0 = Clock::now();
    // v0.2.2.2: the governor sees the budget before the first request after an idle period (one probe sample,
    // ~0.3 ms) when the last sample is older than 5 s, so a pool that must shrink does so before the request runs
    if (gv_.govActive() && vram_probe_ && msSince(vw_.vramLastSample()) > 5000.0) {
        WH_UNLOCK_HELD(lk, jq_.q_mutex_);
        (void)vw_.vramNote();
        gv_.govApply();
        WH_RELOCK_HELD(lk, jq_.q_mutex_);
    }
    return false;
}

// the queued requests that start this iteration (q_mutex_ held by the caller)
std::vector<Job*> Scheduler::admitQueued(std::uint32_t n_busy) {
    std::vector<Job*> admit;
    std::size_t qi = 0;
    std::optional<std::size_t> pool_avail;  // FR-SRV P2-3 (computed when a request is about to start)
    // v0.2.2.2 memory governor: parked slots take no request; while a smaller pool waits for the running requests,
    // none is admitted (they queue)
    std::size_t admit_cap = slots_.size();
    if (gv_.govActive()) {
        std::size_t n_parked = 0;
        for (const Slot& sl : slots_) n_parked += sl.parked ? 1 : 0;
        admit_cap = gv_.govHolding() && n_busy > 0 ? n_busy : slots_.size() - n_parked;
    }
    while (qi < queue_.size() && n_busy + admit.size() < admit_cap) {
        // item S: a new session whose system message another request is
        // prefilling right now waits for that checkpoint
        if (sc_.waitsForSys(*queue_[qi]) || waitsForTier(*queue_[qi])) {
            ++qi;
            continue;
        }
        // FR-SRV P2-3: a prompt the KV pool cannot hold next to the running requests waits in the
        // queue (in order: later requests wait behind it) instead of failing mid-prefill. Alone it
        // always starts (startJob rejects a prompt larger than the whole pool).
        {
            if (!pool_avail) pool_avail = sc_.poolAvailable();
            Job& qj = *queue_[qi];
            const std::size_t need = sc_.pagesFor(qj.tokens.size());
            if (n_busy + admit.size() > 0 && need <= pool_.pages() && need > *pool_avail) {
                if (!qj.pool_wait_logged) {
                    qj.pool_wait_logged = true;
                    logI("req {} | queued: the KV pool cannot hold this prompt next to the running requests ({} pages needed, "
                         "{} free or evictable); it starts when they finish",
                         qj.id, need, *pool_avail);
                }
                break;
            }
            *pool_avail -= std::min(*pool_avail, need);
        }
        jq_.jqAdmit(qi, admit);
    }
    return admit;
}

void Scheduler::startAdmitted(const std::vector<Job*>& admit) {
    std::size_t n_requeued = 0;
    for (Job* job : admit) {
        // item T3: admitted together with a request whose restore just
        // started on the same prefix: back to the queue, wait for it
        if (waitsForTier(*job)) {
            jq_.jqStartAdmitted(job, n_requeued);
            continue;
        }
        Slot& sl = *ps_.pickSlot(std::span<Slot>(slots_), job->tokens,
                                        [](const Slot& o) { return o.phase == Phase::idle && !o.parked; });
        const double wait_ms = msSince(job->t_arrive);
        if (wait_ms > 50) logI("req {} | slot {} | starting after {:.0f} ms in queue", job->id, sl.id, wait_ms);
        else logI("req {} | slot {} | assigned", job->id, sl.id);
        if (tryRestore(sl, *job)) continue;
        sl.phase = Phase::prefill;  // reserve before the next pick
        bool ok = true;
        try {
            ok = startJob(sl, *job);
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
            ok = true;
        }
        if (!ok) {
            sl.job = job;
            releaseSlot(sl);
        }
    }
}

}  // namespace whirl::server
