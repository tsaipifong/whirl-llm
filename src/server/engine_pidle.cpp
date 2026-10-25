// Server engine, PRECISE-IDLE (feat026-pidle): the background re-prefill of a finished reply in --precise, so the
// conversation's next turn resumes close to the end of the reply with a state bitwise equal to a cold prefill. The
// schedule and the start choice are pure (server/precise_idle.h, with the proof); here: when it runs (runLoop's idle
// wait, one chunk per call), the slot bookkeeping (checkpoints above the start invalidated before any KV row is
// rewritten, copy on write against a running spill or shared pages, the tier's valid page count), the checkpoint, and
// stopping when a request arrives.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Prior art: progress/feat026-pidle.PRIOR_ART.md.

#include "engine.h"

#include "log.h"
#include "precise_idle.h"
#include "whirl/vismap.h"

#include <algorithm>
#include <chrono>

namespace whirl::server {

bool Engine::pidleOn() const {
    // COLD-RESUME only (balance / fast resume from gen-end and never need it); not with DFlash2 (its drafter context
    // follows every prefill row; left out of the first version), not without a prefix cache
    return opt_.precise_idle && opt_.cold_resume && !opt_.no_prefix_cache && !opt_.use_dflash;
}

void Engine::pidleMark(Slot& sl) {
    sl.pidle_want = pidleOn() && !sl.ckpts.empty() && !sl.cache_tokens.empty();
    sl.pidle_t = Clock::now();
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
        // KV rows from `from` on are rewritten: every checkpoint above it describes the old rows (prompt-end,
        // think-open, gen-end of the last request) and goes before the first row is written
        for (Ckpt& c : sl.ckpts)
            if (c.valid && c.pos > pl.from) c.valid = false;
        // pages a spill still reads, or that are shared, are replaced (copy on write), as startJob does
        bool spill = false;
        if (sl.spill_busy && pl.from < sl.spill_upto) {
            if (ops_.eventDone(sl.spill_ev.get())) sl.spill_busy = false;
            else spill = true;
        }
        if (spill || pool_.sharedFrom(sl.pages, pl.from)) cowPages(sl, pl.from, spill);
        // the host tier's copy of this slot is valid only below `from` now
        sl.tier_pages = std::min(sl.tier_pages, pl.from / kv_page);
        const Ckpt& c0 = sl.ckpts[static_cast<std::size_t>(pl.ck)];
        m_.selectSeq(sl.id);
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
        pidle_started_ += 1;
        logI("slot {} | precise idle: re-prefilling {} .. {} of the {} cached tokens on the cold schedule (from the '{}' checkpoint) "
             "for the next turn",
             sl.id, pl.from, pl.to, V.size(), c0.kind);
    }
    Slot& sl = slots_[pidle_.slot];
    // the memory governor wants a safe point / parked the slot: stop here (as for a request)
    if (sl.parked || gov_pending_ || gov_hold_ || sl.phase != Phase::idle) {
        pidleCancel();
        return false;
    }
    const pidle::Sched sc{pidle_.toks.size(), pidle_.sys, pidle_.imgs};
    const std::size_t n = pidle::chunkLen(sc, pidle_.cur, pidle_.to);
    try {
        if (n == 0 || !ensurePages(sl, pidle_.cur + n)) throw std::runtime_error("no pages");
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
    pidle_.active = false;
    pidle_done_ += 1;
    // the tier's copy is short of the new content: spill it again (tierTick)
    if (tb_.tier() != nullptr) sl.spill_wanted = true;
    logI("slot {} | precise idle: checkpoint 'prefix' @{} kept ({} rows in {:.0f} ms); a next turn continuing these tokens resumes there",
         sl.id, pidle_.to, pidle_.to - pidle_.from, msSince(pidle_.t0));
    return true;
}

void Engine::pidleCancel() {
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
    logI("slot {} | precise idle: a request arrived; stopped at {} of {} .. {}{}", sl.id, pidle_.cur, pidle_.from, pidle_.to,
         pidle_.cur > pidle_.from ? " (checkpoint kept there)" : "");
}

}  // namespace whirl::server
