// Server engine, KV pages and checkpoints (arch025-N4f, from engine.cpp): the slots' checkpoints, page allocation /
// eviction / copy-on-write over pool_, the shared prefix checkpoints over ps_, and the pool arithmetic of admission.
// arch025-engine S7: these are SlotPages's (module server.slot_cache, engine_pages.h); only the definition lines changed
// (Engine:: -> SlotPages::, SlotEvictor holds SlotPages&); initPool (from engine.cpp) is the last one.
// arch025-engine S8 (N4-S7c, S7 review consider-3): the includes nothing here used (whirl/dflash_policy.h, spec/spec_rules.h,
// <cmath> <cstring> <ctime> <future> <thread>) are gone, and with them two false layering edges.
// SPDX-License-Identifier: Apache-2.0

#include "server/engine_pages.h"

#include "cache/tier_adapter.h"
#include "log.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace whirl::server {

// ---------------------------------------------------------------------------
// checkpoints (of the currently selected slot)

void SlotPages::saveCkpt(Slot& sl, std::uint32_t pos, const char* kind, std::optional<DevPtr> hid,
                      std::optional<DevPtr> logits) {
    if (opt_.no_prefix_cache || sl.ckpts.empty()) return;
    // item T3: a restore tail still arriving: done -> its checkpoints are
    // valid; else it stops where it is and the main stream waits for the
    // enqueued copies
    if (sl.restore != nullptr && sl.phase != Phase::restore) tb_.cutTail(sl.id, sl, sl);
    // a spill to the host tier may still be reading the checkpoint buffers
    tb_.waitSpill(sl);
    Ckpt* slot = &sl.ckpts[ps_.cache().ckptSlotFor(sl.ckpts, pos)];
    ck_.save(*slot, hid, logits);
    sl.ckpt_seq += 1;
    slot->valid = true;
    slot->pos = pos;
    slot->seq = sl.ckpt_seq;
    slot->kind = kind;
}

void SlotPages::dropCache(Slot& sl) {
    for (Ckpt& c : sl.ckpts) c.valid = false;
    sl.cache_tokens.clear();
    dc_.dflReset(sl, 0);
}

void SlotPages::freePages(Slot& sl) {
    const bool busy = sl.spill_busy && !ops_.eventDone(sl.spill_ev.get());
    sl.spill_busy = false;
    if (sl.spill_wanted) logW("slot {} | kv tier: deferred spill dropped (pages evicted)", sl.id);
    sl.spill_wanted = false;
    sl.tier_id = 0;
    sl.tier_pages = 0;
    dropCache(sl);
    if (busy) {
        try {
            pool_.quarantine(sl.pages);
        } catch (const std::exception&) {
            // cannot track them: order the main stream after the spill instead
            tb_.countSpillWait();
            try {
                ops_.streamWaitEvent(m_.stream(), sl.spill_ev.get());
            } catch (const std::exception&) {
            }
            pool_.unref(sl.pages);
        }
    } else {
        pool_.unref(sl.pages);
    }
    sl.pages.clear();
}

// ---------------------------------------------------------------------------
// pool pages (item S: reference counts)

struct SlotPages::SlotEvictor final : PagePool::Evictor {
    SlotPages& e;
    Slot* sl;
    SlotEvictor(SlotPages& e_, Slot* sl_) : e(e_), sl(sl_) {}
    bool evictOne() override { return e.evictFor(sl); }
};

// the prefix cache's victim for a page sl (nullptr: none) needs: a shared checkpoint dropped or an idle slot's pages
// freed (true), or nothing to evict (false)
bool SlotPages::evictFor(Slot* sl) {
    const cache::Victim v = ps_.cache().evictVictim(
        slots_,
        [&](std::size_t i) {
            const Slot& o = slots_[i];
            return !((sl != nullptr && &o == sl) || o.phase != Phase::idle || o.pages.empty());
        },
        ps_.spes());
    Slot* victim = v.kind == cache::Victim::slot ? &slots_[v.index] : nullptr;
    Spe* vspe = v.kind == cache::Victim::spe ? &ps_.spes()[v.index] : nullptr;
    if (vspe) {
        logI("kv pool full: dropping shared {} checkpoint @{} ({} pages)", vspe->kind, vspe->n, vspe->pages.size());
        dropSpe(*vspe);
        return true;
    }
    if (!victim) return false;
    if (sl) {
        logI("kv pool full: evicting idle slot {} ({} pages, {} cached tokens) for slot {}", victim->id,
             victim->pages.size(), victim->cache_tokens.size(), sl->id);
    } else {
        logI("kv pool full: evicting idle slot {} ({} pages, {} cached tokens)", victim->id, victim->pages.size(),
             victim->cache_tokens.size());
    }
    cnt_.noteEviction();
    freePages(*victim);
    return true;
}

std::optional<std::int32_t> SlotPages::takePage(Slot* sl) {
    SlotEvictor ev(*this, sl);
    return pool_.take(ev);
}

void SlotPages::cowPages(Slot& sl, std::uint32_t reuse, bool spill) {
    const std::size_t keep = reuse == 0 ? 0 : (reuse - 1) / kv_page;
    SlotEvictor ev(*this, &sl);
    switch (pool_.cow(sl.pages, sl.id, reuse, spill, ev)) {
    case PagePool::Cow::unchanged:
        return;
    case PagePool::Cow::no_page:
        if (!spill) throw KvPoolFull();
        tb_.waitSpill(sl);
        return;
    case PagePool::Cow::done:
        break;
    }
    if (!spill) cnt_.noteUnshare();
    sl.tier_pages = std::min(sl.tier_pages, static_cast<std::uint32_t>(keep));
}

bool SlotPages::ensurePages(Slot& sl, std::uint64_t n_tok) {
    SlotEvictor ev(*this, &sl);
    return pool_.ensure(sl.pages, sl.id, n_tok, ev);
}

// ---------------------------------------------------------------------------
// item S: shared prefix checkpoints

void SlotPages::dropSpe(Spe& s) {
    if (!s.valid) return;
    const bool busy = s.spill_busy && !ops_.eventDone(s.spill_ev.get());
    s.spill_busy = false;
    if (busy) {
        try {
            pool_.quarantine(s.pages);
        } catch (const std::exception&) {
            tb_.countSpillWait();
            try {
                ops_.streamWaitEvent(m_.stream(), s.spill_ev.get());
            } catch (const std::exception&) {
            }
            pool_.unref(s.pages);
        }
    } else {
        pool_.unref(s.pages);
    }
    s.pages.clear();
    s.tokens.clear();
    s.valid = false;
    s.n = 0;
    s.tier_id = 0;
    cnt_.noteSpeEvict();
}

void SlotPages::createSpe(Slot& sl, std::span<const std::uint32_t> toks_all, std::uint32_t pos, const char* kind,
                       std::optional<DevPtr> hid_row, const Ckpt* src, std::uint64_t tier_id) {
    if (ps_.spes().empty() || opt_.no_prefix_cache || pos == 0 || pos > toks_all.size()) return;
    const auto toks = toks_all.subspan(0, pos);
    const std::uint64_t use = ps_.cache().touch();
    if (const int i = ps_.cache().findShared(ps_.spes(), toks); i >= 0) {
        ps_.spes()[static_cast<std::size_t>(i)].last_used = use;
        return;
    }
    const std::size_t last = (pos - 1) / kv_page;
    if (sl.pages.size() <= last) return;
    const int vi = ps_.cache().sharedVictim(ps_.spes());
    if (vi < 0) return;
    Spe& s = ps_.spes()[static_cast<std::size_t>(vi)];
    if (s.valid) logI("shared prefix: replacing {} checkpoint @{} (used {}x)", s.kind, s.n, s.uses);
    dropSpe(s);
    s.pin = true;
    struct Unpin {
        Spe& s;
        ~Unpin() { s.pin = false; }
    } unpin{s};
    s.tokens.reserve(pos);
    s.pages.reserve(last + 1);
    auto np = takePage(&sl);
    if (!np) return;
    pool_.copy(*np, sl.pages[last]);
    for (std::size_t i = 0; i < last; ++i) {
        pool_.ref(sl.pages[i]);
        s.pages.push_back(sl.pages[i]);
    }
    s.pages.push_back(*np);
    // a spill of the previous content may still read the state buffers
    if (s.spill_ev && !ops_.eventDone(s.spill_ev.get())) {
        tb_.countSpillWait();
        ops_.streamWaitEvent(m_.stream(), s.spill_ev.get());
    }
    Ckpt& c = s.ck;
    const auto& cfg = m_.cfg();
    if (src) {
        for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
            if (c.conv[i] == 0) continue;
            ops_.copyAsync(c.conv[i], src->conv[i], conv_bytes_, m_.stream());
            ops_.copyAsync(c.ssm[i], src->ssm[i], ssm_bytes_, m_.stream());
        }
        ops_.copyAsync(c.hid, src->hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
    } else {
        m_.selectSeq(sl.id);
        m_.commitSeq(m_.curSeq());
        const auto conv = m_.convState();
        const auto ssm = m_.ssmState();
        for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
            if (ssm[i] == 0) continue;
            ops_.copyAsync(c.conv[i], conv[i], conv_bytes_, m_.stream());
            ops_.copyAsync(c.ssm[i], ssm[i], ssm_bytes_, m_.stream());
        }
        if (hid_row) ops_.copyAsync(c.hid, *hid_row, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
    }
    c.valid = true;
    c.pos = pos;
    c.seq = 1;
    c.kind = kind;
    c.has_logits = false;
    s.tokens.assign(toks.begin(), toks.end());
    s.n = pos;
    s.kind = kind;
    s.valid = true;
    s.last_used = use;
    s.uses = 0;
    s.tier_id = tier_id;
    cnt_.noteSpeNew();
    logI("req {} | slot {} | shared prefix: kept {} checkpoint @{} ({} pages shared, 1 copied){}", sl.reqId(), sl.id,
         kind, pos, last, src ? " from the restored entry" : "");
    if (tier_id == 0) {
        try {
            tb_.spillSpe(s);
        } catch (const std::exception& ex) {
            logW("shared prefix: tier spill failed: {} ({})", ex.what(), ops_.lastErrorString());
        }
    }
}

void SlotPages::attachSpe(Slot& sl, Spe& s) {
    s.pin = true;
    struct Unpin {
        Spe& s;
        ~Unpin() { s.pin = false; }
    } unpin{s};
    freePages(sl);
    const std::size_t last = s.pages.size() - 1;
    sl.pages.reserve(last + 1);
    sl.cache_tokens.reserve(s.n);
    auto np = takePage(&sl);
    if (!np) throw KvPoolFull();
    for (std::size_t i = 0; i < last; ++i) {
        pool_.ref(s.pages[i]);
        sl.pages.push_back(s.pages[i]);
    }
    sl.pages.push_back(*np);
    pool_.copy(*np, s.pages[last]);
    m_.mapPages(sl.id, 0, sl.pages);
    sl.cache_tokens.insert(sl.cache_tokens.end(), s.tokens.begin(), s.tokens.end());
    // another lineage's KV: no drafter context for it
    dc_.dflReset(sl, s.n);
    s.last_used = ps_.cache().touch();
    s.uses += 1;
    cnt_.noteSpeHit();
    // the checkpoint's RAM tier entry was reused (served from VRAM here, so no tier restore marked it):
    // worth an SSD write under the seen-once filter, and on eviction in shutdown mode (TIER-EVICT)
    if (tb_.tier() != nullptr && s.tier_id != 0)
        if (tier::Entry* x = tb_.tier()->find(s.tier_id)) x->reused = true;
    cnt_.noteSpeTok(s.n);
}

bool SlotPages::waitsForSys(Job& job) {
    if (ps_.spes().empty() || opt_.no_prefix_cache || !job.params.cache_prompt) return false;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t b = ps_.sysSplit(toks);
    if (b == 0) return false;
    if (Spe* s = ps_.speMatch(toks); s && s->n >= b) return false;
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::prefill || !sl.on_sched || sl.sys_split != b) continue;
        if (sl.reuse + sl.pf_off >= b) continue;
        const auto& jt = sl.job->tokens;
        if (jt.size() < b || !std::equal(jt.begin(), jt.begin() + b, toks.begin())) continue;
        if (!job.sys_wait_logged) {
            logI("req {} | waiting for the system-message checkpoint @{} that slot {} is prefilling", job.id, b, sl.id);
            job.sys_wait_logged = true;
        }
        return true;
    }
    return false;
}

std::size_t SlotPages::pagesFor(std::uint64_t n_tok) const {
    return PagePool::pagesFor(n_tok + dc_.draftRoom() + 2);
}

std::size_t SlotPages::poolAvailable() const {
    std::size_t n = pool_.reclaimable(), owed = 0;
    auto own = [&](const std::vector<std::int32_t>& pages) { n += pool_.soleOwned(pages); };
    for (const Slot& sl : slots_) {
        if (sl.phase == Phase::idle) {
            own(sl.pages);  // evictable (its prefix cache is dropped)
        } else if (sl.phase == Phase::prefill && sl.job) {
            const std::size_t want = pagesFor(sl.n_prompt);
            if (want > sl.pages.size()) owed += want - sl.pages.size();
        }
    }
    for (const Spe& s : ps_.spes()) own(s.pages);
    return n > owed ? n - owed : 0;
}

void SlotPages::initPool() {
    const std::uint32_t n = m_.poolPages();
    kv_arrays_ = m_.kvArrays();
    pool_.init(n, kv_arrays_);
}

}  // namespace whirl::server
