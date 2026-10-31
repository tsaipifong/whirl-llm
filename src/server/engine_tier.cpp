// Server engine, host tier glue (arch025-N4f, from engine.cpp): the scheduler's half of the RAM / SSD tiers (attach,
// restore admission and steps, the tick, the shutdown flush); the tier half is TierBridge (tb_).
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

void Engine::attachTier(tier::Tier* t) {
    tb_.attach(t);
    pool_.setQuarantineStream(t ? t->stream() : nullptr);
    ps_.attachTier(t);
    if (t) {
        t->setWake([this] {
            whirl::MutexLock lk(q_mutex_);
            tier_wake_ = true;
            ++wake_seq_;
            q_cond_.notify_all();
        });
        t->log_fn = [](std::string_view msg) { logI("{}", msg); };
    }
}

bool Engine::waitsForTier(Job& job) {
    if (!tb_.tier()) return false;
    if (!job.params.cache_prompt || opt_.no_prefix_cache) return false;
    const std::span<const std::uint32_t> toks = job.tokens;
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::restore) continue;
        const std::uint32_t reuse = tb_.queuedReuse(sl, toks);
        if (reuse == 0) continue;
        std::uint32_t vr = 0;
        if (Spe* s = ps_.speMatch(toks)) vr = s->n;
        for (Slot& o : slots_)
            if (o.phase == Phase::idle) vr = std::max(vr, ps_.cacheMatch(o, toks).reuse);
        if (reuse <= vr || reuse - vr < 512) continue;
        if (!job.tier_wait_logged) {
            logI("req {} | waiting for the restore of its prefix ({} tok) into slot {}", job.id, reuse, sl.id);
            job.tier_wait_logged = true;
        }
        return true;
    }
    return false;
}

void Engine::prefetchFor(Job& job) {
    if (!tb_.tier()) return;
    if (job.pf_checked) return;
    job.pf_checked = true;
    if (!job.params.cache_prompt || opt_.no_prefix_cache) return;
    const std::span<const std::uint32_t> toks = job.tokens;
    const auto mt = tb_.prefetchCandidate(toks);
    if (!mt) return;
    std::uint32_t vr = 0;
    if (Spe* s = ps_.speMatch(toks)) vr = s->n;
    for (Slot& o : slots_) vr = std::max(vr, ps_.cacheMatch(o, toks).reuse);
    if (mt->reuse <= vr || mt->reuse - vr < 512) return;
    tb_.prefetch(job.id, mt->e);
}

// ---------------------------------------------------------------------------
// item G: host tiers

bool Engine::tryRestore(Slot& sl, Job& job) {
    if (!tb_.tier()) return false;
    if (!job.params.cache_prompt || opt_.no_prefix_cache) return false;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t nd_max = dc_.draftRoom();
    if (toks.size() + nd_max + 2 >= ctxForNew() || toks.size() + nd_max + 2 >= static_cast<std::uint64_t>(pool_.pages()) * kv_page)
        return false;
    const auto mt = tb_.restoreMatch(sl, toks);
    if (!mt) return false;
    try {
        tier::Entry* x = mt->e;
        const TimePoint t0 = Clock::now();
        // the slot's own VRAM content goes (it is in the tier if it was worth keeping)
        sc_.freePages(sl);
        sl.job = &job;
        sl.phase = Phase::restore;
        if (!sc_.ensurePages(sl, x->n_tok)) {
            logW("req {} | slot {} | kv tier: not restoring {} tok: the kv pool has no room ({} of {} pages for this slot, {} "
                 "free, {} quarantined batches)",
                 job.id, sl.id, x->n_tok, sl.pages.size(), (x->n_tok + kv_page - 1) / kv_page, pool_.freeCount(),
                 pool_.batches());
            sl.phase = Phase::prefill;
            return false;
        }
        if (!tb_.startRestore(job.id, sl.id, job.tokens, sl, sl, *mt, t0)) {
            sl.phase = Phase::prefill;
            return false;
        }
        sl.tp0_pre = t0;
        return true;
    } catch (const std::exception& ex) {
        logW("req {} | slot {} | kv tier: restore failed to start: {} ({})", job.id, sl.id, ex.what(), ops_.lastErrorString());
        return false;
    }
}

// the Engine steps TierBridge calls inside a restore (in the order they always ran)
TierBridge::RestoreSteps Engine::restoreSteps(Slot& sl) {
    TierBridge::RestoreSteps s;
    s.drop_cache = [this, &sl] { sc_.dropCache(sl); };
    s.drafter_reset = [this, &sl](std::uint32_t n_tok) { dc_.dflReset(sl, n_tok); };
    s.share = [this, &sl](Ckpt& c, std::uint64_t entry_id) {
        sc_.createSpe(sl, sl.cache_tokens, c.pos, c.kind, std::nullopt, &c, entry_id);
    };
    return s;
}

void Engine::pollRestores() {
    if (!tb_.tier()) return;
    for (Slot& sl : slots_) {
        if (!sl.restore) continue;
        if (sl.phase != Phase::restore) {
            tb_.finishTail(sl.id, sl, sl, false);
            continue;
        }
        const TierBridge::Pump p = tb_.pumpRestore(sl);
        if (p == TierBridge::Pump::pending) continue;
        Job& job = *sl.job;
        if (p == TierBridge::Pump::failed) tb_.restoreFailed(job.id, sl.id, sl, restoreSteps(sl));
        else tb_.restoreDone(job.id, sl.id, sl, sl, restoreSteps(sl));
        sl.phase = Phase::prefill;
        bool ok = true;
        try {
            ok = startJob(sl, job);
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
            ok = true;
        }
        if (!ok) {
            sl.job = &job;
            releaseSlot(sl);
        }
    }
}

// the tier's background work (TierBridge::tick), then the page pool's quarantine and the deferred spills
bool Engine::tierTick() {
    if (!tb_.tier()) return false;
    bool busy = tb_.tick();
    if (!pool_.quarantineEmpty()) {
        pool_.reclaim(false);
        busy = busy || !pool_.quarantineEmpty();
    }
    for (Slot& sl : slots_) {
        if (sl.phase == Phase::idle && sl.spill_wanted) {
            try {
                tb_.spillSlot(sl.id, sl.phase == Phase::idle, sl, sl);
            } catch (const std::exception&) {
            }
            // the spill (or its retry) leaves background work: the SSD write after it
            busy = true;
        }
    }
    return busy;
}

// Shutdown (stop()): the bounded tier flush (TierBridge::flushOnStop), then the tier line
void Engine::flushTierOnStop() {
    tb_.flushOnStop([this] { return tierTick(); });
    tb_.logStats(pool_.stats().cow_pages, pool_.stats().q_waits);
}

}  // namespace whirl::server
