// Server engine, PRECISE-IDLE (feat026-pidle): the background re-prefill of a finished reply in --precise, so the
// conversation's next turn resumes close to the end of the reply with a state bitwise equal to a cold prefill. The
// schedule and the start choice are pure (server/precise_idle.h, with the proof); here: when it runs (runLoop's idle
// wait, one chunk per call), the slot bookkeeping (checkpoints above the start invalidated before any KV row is
// rewritten, copy on write against a running spill or shared pages, the tier's valid page count, the cached tokens
// = the rows the KV holds), the checkpoint, and stopping when a request arrives.
//
// feat026-pidle-r1 (review progress/feat026-pidle-acc.review.md): it is a speculative job nobody asked for, so it
// never costs anyone else anything: it takes KV pages only from the free / quarantined ones (never evicts another
// conversation's cache or a shared checkpoint; skipped before anything changes when they do not suffice), nothing it
// does throws out of runLoop (a failure ends the job, at worst dropping this slot's own cache), and it is not started
// for a reply to a thinking-block prompt (the template drops the reasoning from the history: the checkpoint would
// never be used, while prompt-end / think-open / gen-end of the slot would be lost).
//
// feat026-pidle-r2 (research/pidle/2026-10-26-think-template-reuse.md): the reply is a token prefix of the next prompt
// with thinking off (both templates), and with thinking on only when the client sends the reasoning back
// (reasoning_content); a client that drops it makes the next prompt diverge one token before the end of this prompt.
// So a thinking reply is re-prefilled when this prompt's history shows reasoning sent back (an earlier assistant turn
// with a non-empty think block: ChatTokens::reasoningInHistory), and skipped otherwise (first turn, empty blocks
// only). A variant-a request with preserve_thinking=false is always skipped (the template strips the blocks of
// earlier turns, so the next prompt diverges before the reply). A wrong guess costs only idle GPU time: the next
// turn resumes from a checkpoint only where its tokens match (output bitwise either way).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Prior art: progress/feat026-pidle.PRIOR_ART.md, progress/feat026-pidle-r1.PRIOR_ART.md.

#include "engine.h"

#include "log.h"
#include "precise_idle.h"
#include "whirl/vismap.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace whirl::server {

namespace {

// pages for the idle job come only from the free list and the quarantine (a spill's pages, after a GPU-side wait):
// never another slot's cache or a shared checkpoint (Engine::SlotEvictor would evict them)
struct NoEvict final : PagePool::Evictor {
    bool evictOne() override { return false; }
};

// feat026-pidle-r2 (review r1-batch-b): runs f when the scope is left by a throw / an early exit while armed
template <class F>
struct OnAbort {
    F f;
    bool armed = true;
    ~OnAbort() {
        if (!armed) return;
        try {
            f();
        } catch (...) {
        }
    }
};
template <class F>
OnAbort(F) -> OnAbort<F>;

}  // namespace

bool Engine::pidleOn() const {
    // COLD-RESUME only (balance / fast resume from gen-end and never need it); not with DFlash2 (its drafter context
    // follows every prefill row; left out of the first version), not without a prefix cache
    return opt_.precise_idle && opt_.cold_resume && !opt_.no_prefix_cache && !opt_.use_dflash;
}

void Engine::pidleMark(Slot& sl) {
    sl.pidle_want = pidleOn() && !sl.ckpts.empty() && !sl.cache_tokens.empty();
    sl.pidle_t = Clock::now();
    if (!sl.pidle_want) return;
    // feat026-pidle-r2: preserve_thinking=false (variant a): the next prompt re-renders the earlier turns without
    // their think blocks (thinking on or off), so it diverges before this reply
    // (feat026-pidle-r3: not for a reply that ends in a tool call: that assistant message is after the last real user
    // query, so the template keeps its think block and the next prompt holds the reply as generated)
    if (sl.pidle_hist_dropped && !sl.pidle_tool_end) {
        sl.pidle_want = false;
        pidle_skipped_ += 1;
        logI("slot {} | precise idle: skipped: preserve_thinking=false (the next prompt drops the think blocks)", sl.id);
        return;
    }
    // a thinking-block prompt ("<think>\n" at its end): the next prompt holds this reply only if the client sends the
    // reasoning back, which this prompt's history tells (feat026-pidle-r2). Without it (first turn, or the earlier
    // replies came back as empty blocks) the next prompt re-renders the reply as "<think>" "\n\n" "</think>" ... and
    // diverges at N - 1. (A predicted re-rendering "P[:-1] + \n\n</think>\n\n + content" could be checkpointed
    // instead: follow-up.)
    const std::size_t np = std::min<std::size_t>(sl.n_prompt, sl.cache_tokens.size());
    const std::span<const std::uint32_t> prompt = std::span<const std::uint32_t>(sl.cache_tokens).first(np);
    if (ct_.thinkOpenSplit(prompt) > 0) {
        if (ct_.reasoningInHistory(prompt)) {
            logI("slot {} | precise idle: thinking reply kept: the history carries reasoning (the client sends it back)",
                 sl.id);
            return;
        }
        sl.pidle_want = false;
        pidle_skipped_ += 1;
        logI("slot {} | precise idle: skipped: the prompt opens a thinking block and its history carries no reasoning "
             "(the next prompt will not hold this reply)",
             sl.id);
    }
}

std::optional<TimePoint> Engine::pidleDue() const {
    if (!pidleOn() || pidle_.active) return std::nullopt;
    std::optional<TimePoint> due;
    for (const Slot& sl : slots_) {
        if (!sl.pidle_want || sl.phase != Phase::idle || sl.parked) continue;
        const TimePoint t = sl.pidle_t + std::chrono::milliseconds(opt_.precise_idle_ms);
        if (!due || t < *due) due = t;
    }
    // a connection is building a request: look again shortly (it either arrives, which ends the wait, or is refused)
    if (due && n_building_.load() > 0) due = std::max(*due, Clock::now() + std::chrono::milliseconds(20));
    // the memory governor is acting: look again later (no busy wait on a start that cannot happen now)
    if (due && (gov_pending_ || gov_hold_)) due = std::max(*due, Clock::now() + std::chrono::milliseconds(100));
    return due;
}

bool Engine::pidleStep() {
    // runLoop calls this with its queue lock released: nothing may leave here as an exception (review A2)
    try {
        return pidleRun();
    } catch (const std::exception& ex) {
        logW("precise idle: stopped: {} ({})", ex.what(), ops_.lastErrorString());
    } catch (...) {
        logW("precise idle: stopped: unknown error");
    }
    if (pidle_.active) {
        // the rows from `from` on may be half written: the slot's cache goes
        pidle_.active = false;
        pidle_cancelled_ += 1;
        try {
            dropCache(slots_[pidle_.slot]);
        } catch (...) {
        }
    }
    return false;
}

bool Engine::pidleRun() {
    if (!pidleOn()) return false;
    if (!pidle_.active) {
        if (n_building_.load() > 0 || gov_pending_ || gov_hold_) return false;
        // the slot whose reply ended last (the conversation most likely to continue)
        Slot* pick = nullptr;
        const TimePoint now = Clock::now();
        for (Slot& sl : slots_) {
            if (!sl.pidle_want || sl.phase != Phase::idle || sl.parked) continue;
            if (now < sl.pidle_t + std::chrono::milliseconds(opt_.precise_idle_ms)) continue;
            if (!pick || sl.pidle_t > pick->pidle_t) pick = &sl;
        }
        if (!pick) return false;
        Slot& sl = *pick;
        sl.pidle_want = false;
        // a host-tier restore tail still arriving into this slot: leave it (its checkpoints become valid when done)
        if (sl.restore != nullptr) {
            pidle_skipped_ += 1;
            return false;
        }
        const std::vector<std::uint32_t>& V = sl.cache_tokens;
        const std::uint32_t sys = ps_.sysSplit(V);
        std::vector<std::uint32_t> imgs = ps_.imgSplits(V);
        const pidle::Sched sc{V.size(), sys, imgs};
        const pidle::Plan pl = pidle::plan(sl.ckpts, sc);
        if (!pl.ok()) {
            pidle_skipped_ += 1;
            return false;
        }
        // vision: image tokens need their embeddings (visPrepare): not re-prefilled here
        if (std::any_of(V.begin() + pl.from, V.begin() + pl.to, [](std::uint32_t t) { return t >= qwen35::image_id_base; })) {
            pidle_skipped_ += 1;
            return false;
        }
        // pages a spill still reads, or that are shared, are replaced (copy on write), as startJob does
        bool spill = false;
        if (sl.spill_busy && pl.from < sl.spill_upto) {
            if (ops_.eventDone(sl.spill_ev.get())) sl.spill_busy = false;
            else spill = true;
        }
        const bool cow = spill || pool_.sharedFrom(sl.pages, pl.from);
        // the pages the job needs (copy of the page holding row from - 1 + the pages up to `to`), checked before
        // anything changes; only free / quarantined pages count, plus the slot's own pages the copy on write releases
        // (review C3: a speculative job never evicts another conversation's cache or a shared checkpoint)
        const std::size_t keep = (pl.from - 1) / kv_page;  // pl.from > 0 (plan)
        const std::size_t want = PagePool::pagesFor(pl.to);
        const std::size_t have = sl.pages.size();
        // (review r1-batch-b: freeable, not reclaimable: a quarantined page another holder keeps never comes back)
        std::size_t need = 0, avail = pool_.freeable();
        if (cow && keep < have) {
            need = std::max<std::size_t>(1, want > keep ? want - keep : 0);
            // the copy's page is taken before the old pages go; after it, the old pages only this slot held are free
            // (or quarantined behind the spill)
            if (avail >= 1) avail += pool_.soleOwned(std::span<const std::int32_t>(sl.pages).subspan(keep));
        } else {
            need = want > have ? want - have : 0;
        }
        if (need > avail) {
            pidle_skipped_ += 1;
            logI("slot {} | precise idle: skipped: needs {} free KV pages, {} free (another conversation's cache is never "
                 "evicted for it)",
                 sl.id, need, pool_.freeable());
            return false;
        }
        // review r1-batch-b: from the copy on write on the slot's rows and checkpoints change; a throw before the job
        // is set up (pidle_.active) drops the slot's cache, so cache_tokens / checkpoints never describe rows the KV
        // no longer holds (pidleStep's catch only cleans up an active job)
        OnAbort drop{[this, &sl] { dropCache(sl); }};
        if (cow) {
            NoEvict ne;
            switch (pool_.cow(sl.pages, sl.id, pl.from, spill, ne)) {
            case PagePool::Cow::no_page:  // (the check above makes this unexpected) nothing changed yet: skip
                drop.armed = false;
                pidle_skipped_ += 1;
                logI("slot {} | precise idle: skipped: no free KV page for the copy on write", sl.id);
                return false;
            case PagePool::Cow::done:
                if (!spill) stat_unshare_ += 1;
                break;
            case PagePool::Cow::unchanged:
                break;
            }
        }
        stealFreePagesNow();  // (test seam, inert)
        // every page up to `to` before any checkpoint is invalidated (review r1-batch-b: a page count that was too high
        // must not cost the slot its prompt-end / think-open / gen-end); the chunks then find their pages there
        {
            NoEvict ne;
            const std::size_t pages_before = sl.pages.size();  // after the copy on write
            if (!pool_.ensure(sl.pages, sl.id, pl.to, ne)) {
                // feat026-pidle-r3: ensure leaves the pages it did take in sl.pages; a skipped job gives them back
                // (they hold no rows: the slot would keep pages for nothing, shortened to the cache it has)
                if (sl.pages.size() > pages_before) {
                    pool_.unref(std::span<const std::int32_t>(sl.pages).subspan(pages_before));
                    sl.pages.resize(pages_before);
                }
                drop.armed = false;
                pidle_skipped_ += 1;
                if (cow) {
                    // the copy on write released the pages after row from: the rows above it are gone
                    for (Ckpt& c : sl.ckpts)
                        if (c.valid && c.pos > pl.from) c.valid = false;
                    sl.tier_pages = std::min(sl.tier_pages, pl.from / kv_page);
                    if (sl.cache_tokens.size() > pl.from) sl.cache_tokens.resize(pl.from);
                }
                logI("slot {} | precise idle: skipped: no free KV page for rows up to {} ({} free)", sl.id, pl.to,
                     pool_.freeCount());
                return false;
            }
        }
        // KV rows from `from` on are rewritten (or, after the copy on write, gone): every checkpoint above it
        // describes the old rows (prompt-end, think-open, gen-end of the last request) and goes before the first row
        // is written (as startJob does for the rows above its reuse point; llama.cpp server-context.cpp erases the
        // checkpoints past pos_next the same way)
        for (Ckpt& c : sl.ckpts)
            if (c.valid && c.pos > pl.from) c.valid = false;
        // the host tier's copy of this slot is valid only below `from` now
        sl.tier_pages = std::min(sl.tier_pages, pl.from / kv_page);
        const Ckpt& c0 = sl.ckpts[static_cast<std::size_t>(pl.ck)];
        m_.selectSeq(sl.id);
        if (test_pidle_load_fail_) {
            test_pidle_load_fail_ = false;
            throw std::runtime_error("test: checkpoint load failed");
        }
        ck_.load(c0);
        sl.ck_hid = c0.hid;
        pidle_.active = true;
        pidle_.slot = sl.id;
        pidle_.from = pl.from;
        pidle_.to = pl.to;
        pidle_.cur = pl.from;
        pidle_.last_n = 0;
        pidle_.toks = V;
        pidle_.sys = sys;
        pidle_.imgs = std::move(imgs);
        pidle_.t0 = now;
        drop.armed = false;  // the job owns the clean-up from here (pidleStep / pidleStop)
        // review C2: the cached tokens are the rows the KV holds for them (startJob's rule, llama.cpp keep_first):
        // [0, from) now, [0, cur) as the chunks run (set at the end / when stopped)
        sl.cache_tokens.resize(pl.from);
        pidle_started_ += 1;
        logI("slot {} | precise idle: re-prefilling {} .. {} of the {} cached tokens on the cold schedule (from the '{}' checkpoint) "
             "for the next turn",
             sl.id, pl.from, pl.to, pidle_.toks.size(), c0.kind);
    }
    Slot& sl = slots_[pidle_.slot];
    // the memory governor wants a safe point / parked the slot: stop here (as for a request)
    if (sl.parked || gov_pending_ || gov_hold_ || sl.phase != Phase::idle) {
        pidleCancel();
        return false;
    }
    const pidle::Sched sc{pidle_.toks.size(), pidle_.sys, pidle_.imgs};
    const std::size_t n = pidle::chunkLen(sc, pidle_.cur, pidle_.to);
    // the pages of this chunk, from free / quarantined pages only (normally all there: the check at the start)
    {
        NoEvict ne;
        if (n == 0 || !pool_.ensure(sl.pages, sl.id, pidle_.cur + n, ne)) {
            pidleStop("no free KV page for the next chunk");
            return false;
        }
    }
    try {
        m_.selectSeq(sl.id);
        const std::span<const std::uint32_t> toks = std::span<const std::uint32_t>(pidle_.toks).subspan(pidle_.from);
        const std::size_t off = pidle_.cur - pidle_.from;
        if (opt_.use_mtp) m_.prefillMtpChunk(toks, pidle_.from, off, n, sl.ck_hid);
        else m_.forward(toks.subspan(off, n), pidle_.cur);
    } catch (const std::exception& ex) {
        // the rows from `from` on may be half written: the slot's cache goes
        logW("slot {} | precise idle: stopped: {} ({}); dropping the slot's prefix cache", sl.id, ex.what(), ops_.lastErrorString());
        pidle_.active = false;
        pidle_cancelled_ += 1;
        dropCache(sl);
        return true;
    }
    pidle_.cur += static_cast<std::uint32_t>(n);
    pidle_.last_n = n;
    pidle_rows_ += n;
    if (pidle_.cur < pidle_.to) return true;
    const std::uint64_t E = m_.cfg().n_embd;
    saveCkpt(sl, pidle_.to, "prefix", opt_.use_mtp ? std::optional<DevPtr>(m_.hn() + static_cast<std::uint64_t>(n - 1) * E * 4) : std::nullopt,
             std::nullopt);
    sl.cache_tokens.assign(pidle_.toks.begin(), pidle_.toks.begin() + pidle_.to);
    pidle_.active = false;
    pidle_done_ += 1;
    // the tier's copy is short of the new content: spill it again (tierTick)
    if (tb_.tier() != nullptr) sl.spill_wanted = true;
    logI("slot {} | precise idle: checkpoint 'prefix' @{} kept ({} rows in {:.0f} ms); a next turn continuing these tokens resumes there",
         sl.id, pidle_.to, pidle_.to - pidle_.from, msSince(pidle_.t0));
    return true;
}

void Engine::pidleCancel() { pidleStop("a request arrived"); }

void Engine::pidleStop(const char* why) {
    if (!pidle_.active) return;
    Slot& sl = slots_[pidle_.slot];
    pidle_.active = false;
    pidle_cancelled_ += 1;
    // the chunks that ran end at a point of the idle schedule: keep a checkpoint there (same rule as `to`)
    // (not into the buffers of a slot the governor parked)
    if (pidle_.cur > pidle_.from && pidle_.last_n > 0 && !sl.parked) {
        const std::uint64_t E = m_.cfg().n_embd;
        m_.selectSeq(sl.id);
        saveCkpt(sl, pidle_.cur, "prefix",
                 opt_.use_mtp ? std::optional<DevPtr>(m_.hn() + static_cast<std::uint64_t>(pidle_.last_n - 1) * E * 4) : std::nullopt,
                 std::nullopt);
        if (tb_.tier() != nullptr) sl.spill_wanted = true;
    }
    // the rows written so far are the slot's (review C2)
    sl.cache_tokens.assign(pidle_.toks.begin(), pidle_.toks.begin() + pidle_.cur);
    logI("slot {} | precise idle: {}; stopped at {} of {} .. {}{}", sl.id, why, pidle_.cur, pidle_.from, pidle_.to,
         pidle_.cur > pidle_.from ? " (checkpoint kept there)" : "");
}

}  // namespace whirl::server
