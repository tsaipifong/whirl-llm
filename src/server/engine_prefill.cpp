// Server engine, prefill (arch025-N4f, from engine.cpp): chunk lengths, checkpoint splits, one slot's step, batched
// prefill groups, the end of a prompt, runLoop's prefill phase and the decode floor bookkeeping.
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

std::size_t Engine::chunkLen(const Slot& sl) const {
    const std::size_t cur = sl.reuse + sl.pf_off;
    const std::vector<std::uint32_t> imgs = ps_.imgSplits(sl.job->tokens);
    const std::size_t N = sl.job->tokens.size();
    // With the decode floor on, any decoding slot keeps forwards at one chunk, also
    // an unprotected one of the same burst: next to a long prompt a 2048-row forward
    // doubled its stall (96k: 1.2 -> 3.3 s, ~1.5 tok/s) for ~4% faster prefill.
    // With the floor off (--decode-min-tps 0, throughput first) chunks merge anyway.
    std::size_t exec = opt_.prefill_exec;
    if (exec > prefill_chunk && fl_.floor.on())
        for (const Slot& o : slots_)
            if (&o != &sl && o.phase == Phase::decode) exec = prefill_chunk;
    // a shared-prefix checkpoint / the TAIL-CK tail checkpoint is kept there: no forward crosses it
    const std::array<std::size_t, 2> stops = {sl.lcp_ck, sl.tail_ck};
    return sched::execLen(N, sl.sys_split, sl.pf_split, imgs, cur, exec, stops, m_.maxBatch());
}

void Engine::maybeSplitCkpt(Slot& sl, DevPtr hid_row) {
    const std::size_t cur = sl.reuse + sl.pf_off;
    // item S: shared prefix checkpoints (a failure only loses the checkpoint)
    if (sl.on_sched && ((sl.sys_split > 0 && cur == sl.sys_split) || (sl.lcp_ck > 0 && cur == sl.lcp_ck) || (sl.img_ck > 0 && cur == sl.img_ck))) {
        const char* kind = cur == sl.sys_split ? "system" : "prefix";
        try {
            createSpe(sl, sl.job->tokens, static_cast<std::uint32_t>(cur), kind,
                      opt_.use_mtp ? std::optional<DevPtr>(hid_row) : std::nullopt, nullptr, 0);
        } catch (const std::exception& ex) {
            logW("req {} | slot {} | shared prefix: checkpoint @{} not kept: {} ({})", sl.reqId(), sl.id, cur, ex.what(),
                 ops_.lastErrorString());
        }
    }
    // TAIL-CK: slot checkpoint at the start of the last schedule chunk (cold-exact)
    if (sl.tail_ck > 0 && cur == sl.tail_ck) {
        m_.selectSeq(sl.id);
        saveCkpt(sl, sl.tail_ck, "prefix", opt_.use_mtp ? std::optional<DevPtr>(hid_row) : std::nullopt, std::nullopt);
    }
    if (sl.pf_split == 0 || cur != sl.pf_split) return;
    m_.selectSeq(sl.id);
    saveCkpt(sl, sl.pf_split, "think-open", opt_.use_mtp ? std::optional<DevPtr>(hid_row) : std::nullopt, std::nullopt);
}

void Engine::prefillStep(Slot& sl) {
    Job& job = *sl.job;
    const std::span<const std::uint32_t> toks = std::span<const std::uint32_t>(job.tokens).subspan(sl.reuse);
    const std::size_t n = chunkLen(sl);
    if (!ensurePages(sl, sl.reuse + sl.pf_off + n)) throw KvPoolFull();
    m_.selectSeq(sl.id);
    if (opt_.use_mtp) {
        m_.prefillMtpChunk(toks, sl.reuse, sl.pf_off, n, sl.ck_hid);
    } else {
        m_.forward(toks.subspan(sl.pf_off, n), sl.reuse + static_cast<std::uint32_t>(sl.pf_off));
    }
    dc_.dflIngest(sl, static_cast<std::uint32_t>(n), 0, static_cast<std::uint32_t>(n), sl.reuse + static_cast<std::uint32_t>(sl.pf_off));
    sl.pf_off += n;
    bst_.prefill_tok += n;
    prefill_tok_cum_ += n;
    maybeSplitCkpt(sl, m_.hn() + static_cast<std::uint64_t>(n - 1) * m_.cfg().n_embd * 4);
    if (sl.pf_off == toks.size()) finishPrefill(sl);
}

// Returns the rows run (0 when nothing ran). row_budget caps the rows of the
// whole forward; the oldest request's chunk always runs (decode floor).
std::size_t Engine::prefillGroup(Slot& first, std::size_t row_budget) {
    const std::size_t n0 = chunkLen(first);
    const std::size_t small_max = max_small_batch;
    auto solo = [&] {
        try {
            prefillStep(first);
        } catch (const std::exception& ex) {
            failSlot(first, ex.what());
        }
    };
    if (!opt_.seg_prefill || n0 <= small_max || !m_.canSegment()) {
        solo();
        return n0;
    }
    const std::size_t cap = std::min<std::size_t>(m_.maxBatch(), std::max(row_budget, n0));
    // the other prefilling slots, oldest first
    std::vector<Slot*> cand;
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::prefill || &sl == &first) continue;
        std::size_t at = cand.size();
        cand.push_back(&sl);
        while (at > 0 && cand[at - 1]->job->id > sl.job->id) {
            cand[at] = cand[at - 1];
            --at;
        }
        cand[at] = &sl;
    }
    std::vector<Slot*> grp{&first};
    std::size_t total = n0;
    for (Slot* sl : cand) {
        if (grp.size() == max_small_batch) break;
        const std::size_t n = chunkLen(*sl);
        // a chunk that cannot join waits for its own turn (it is then the oldest)
        if (n <= small_max || total + n > cap) continue;
        grp.push_back(sl);
        total += n;
    }
    if (grp.size() == 1) {
        solo();
        return n0;
    }
    // pool pages for every chunk of the group (a slot that cannot get them fails alone)
    {
        std::vector<Slot*> kept;
        for (Slot* sl : grp) {
            bool ok = false;
            try {
                ok = ensurePages(*sl, sl->reuse + sl->pf_off + chunkLen(*sl));
            } catch (const std::exception&) {
                ok = false;
            }
            if (!ok) {
                failSlot(*sl, "KvPoolFull");
                continue;
            }
            kept.push_back(sl);
        }
        grp = std::move(kept);
        if (grp.empty()) return 0;
    }
    std::vector<PSeg> segs(grp.size());
    for (std::size_t k = 0; k < grp.size(); ++k) {
        Slot& sl = *grp[k];
        segs[k].seq = sl.id;
        segs[k].tokens = std::span<const std::uint32_t>(sl.job->tokens).subspan(sl.reuse);
        segs[k].pos0 = sl.reuse;
        segs[k].off = sl.pf_off;
        segs[k].n = chunkLen(sl);
        if (opt_.use_mtp) segs[k].prev_hidden = sl.ck_hid;
    }
    try {
        m_.prefillSegs(segs, opt_.use_mtp);
    } catch (const std::exception& ex) {
        for (Slot* sl : grp) failSlot(*sl, ex.what());
        return 0;
    }
    bst_.prefill_tok += total;
    prefill_tok_cum_ += total;
    bst_.seg_batches += 1;
    bst_.seg_chunks += grp.size();
    std::uint64_t r0 = 0;
    for (std::size_t k = 0; k < grp.size(); ++k) {
        Slot& sl = *grp[k];
        dc_.dflIngest(sl, static_cast<std::uint32_t>(total), static_cast<std::uint32_t>(r0), static_cast<std::uint32_t>(segs[k].n),
                  sl.reuse + static_cast<std::uint32_t>(sl.pf_off));
        sl.pf_off += segs[k].n;
        r0 += segs[k].n;
        try {
            maybeSplitCkpt(sl, m_.hn() + (r0 - 1) * m_.cfg().n_embd * 4);
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
            continue;
        }
        if (sl.pf_off != segs[k].tokens.size()) continue;
        try {
            m_.segLogitsToFront(static_cast<std::uint32_t>(k));
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
            continue;
        }
        try {
            finishPrefill(sl);
        } catch (const std::exception& ex) {
            failSlot(sl, ex.what());
        }
    }
    return static_cast<std::size_t>(r0);
}

void Engine::finishPrefill(Slot& sl) {
    Job& job = *sl.job;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t N = sl.n_prompt;
    const auto& cfg = m_.cfg();
    const std::uint32_t V = cfg.n_vocab;
    m_.selectSeq(sl.id);
    std::uint32_t first;
    if (sl.greedy) {
        first = m_.argmax();
    } else {
        sb_.fetchCands(sl.sm, 1);
        sb_.prepareRow(sl.sm, 0);
        sl.sm.pos_idx = 0;
        first = sampling::sampleRow(sb_, sl.sm, std::nullopt, 0);
    }
    if (opt_.verify_cache && sl.reuse > 0) {
        // diagnostic: compare the cached-prefix logits with a from-scratch prefill
        std::vector<float> cached(V);
        ops_.download(cached.data(), m_.logits(), static_cast<std::uint64_t>(V) * 4);
        {
            const bool busy = sl.spill_busy && !ops_.eventDone(sl.spill_ev.get());
            if (busy || pool_.sharedFrom(sl.pages, 0)) {
                cowPages(sl, 0, busy);
                if (!ensurePages(sl, N)) throw KvPoolFull();
            }
        }
        m_.selectSeq(sl.id);
        m_.reset();
        const std::size_t v_sys = ps_.sysSplit(toks);
        const std::size_t v_th = ct_.thinkOpenSplit(toks);
        const std::vector<std::uint32_t> v_imgs = ps_.imgSplits(toks);
        std::size_t off = 0;
        while (off < toks.size()) {
            const std::size_t n = prefillChunkLen(sched::schedNext(toks.size(), v_sys, v_th, v_imgs, off) - off);
            if (opt_.use_mtp) m_.prefillMtpChunk(toks, 0, off, n, std::nullopt);
            else m_.forward(toks.subspan(off, n), static_cast<std::uint32_t>(off));
            off += n;
        }
        std::vector<float>& fresh = sb_.rowHost();
        ops_.download(fresh.data(), m_.logits(), static_cast<std::uint64_t>(V) * 4);
        float mc = -std::numeric_limits<float>::infinity(), mf = mc;
        std::size_t ac = 0, af = 0;
        float maxd = 0;
        std::size_t ndiff = 0;  // logits whose bits differ (max abs diff prints 4 decimals, KL can underflow)
        for (std::size_t j = 0; j < V; ++j) {
            if (std::memcmp(&cached[j], &fresh[j], sizeof(float)) != 0) ++ndiff;
            if (cached[j] > mc) {
                mc = cached[j];
                ac = j;
            }
            if (fresh[j] > mf) {
                mf = fresh[j];
                af = j;
            }
            maxd = std::max(maxd, std::fabs(cached[j] - fresh[j]));
        }
        double zc = 0, zf = 0;
        for (std::size_t j = 0; j < V; ++j) {
            zc += std::exp(static_cast<double>(cached[j] - mc));
            zf += std::exp(static_cast<double>(fresh[j] - mf));
        }
        double kl = 0;
        for (std::size_t j = 0; j < V; ++j) {
            const double pf = std::exp(static_cast<double>(fresh[j] - mf)) / zf;
            const double pc = std::exp(static_cast<double>(cached[j] - mc)) / zc;
            if (pf > 0) kl += pf * (std::log(pf) - std::log(std::max(pc, 1e-300)));
        }
        logI("req {} | slot {} | prefix-cache verify: cached vs from-scratch prefill logits: max abs diff {:.4f}, KL {:.2e}, "
             "top1 {} ({} vs {}), bitwise differing logits {}",
             job.id, sl.id, maxd, kl, ac == af ? "same" : "DIFFERENT", ac, af, ndiff);
        // continue from the from-scratch state (the prompt-end checkpoint below is saved from it)
        if (sl.greedy) {
            first = m_.argmax();
        } else {
            sb_.fetchCands(sl.sm, 1);
            sb_.prepareRow(sl.sm, 0);
            first = sampling::sampleRow(sb_, sl.sm, std::nullopt, 0);
        }
    }
    sl.prompt_ms = msSince(sl.tp0);
    const std::uint32_t n_new = N - sl.reuse;
    // (thinking-mode prompts keep the think-open checkpoint at N - 1 instead)
    if (sl.reuse < N && ct_.thinkOpenSplit(toks) == 0)
        saveCkpt(sl, N, "prompt-end", opt_.use_mtp ? std::optional<DevPtr>(m_.mtpH()) : std::nullopt, m_.logits());
    logI("req {} | slot {} | prefill: {} tok in {:.1f} ms ({:.1f} tok/s)", job.id, sl.id, n_new, sl.prompt_ms,
         sl.prompt_ms > 0 ? static_cast<double>(n_new) * 1000.0 / sl.prompt_ms : 0.0);

    sl.out = Out{};
    sl.out.chat = job.kind == JobKind::chat;
    sl.out.think_open = job.kind == JobKind::chat && job.think;
    sl.out.tools_on = job.tools_on;
    sl.out.stops = job.params.stop;
    sl.st = StreamState{};
    sl.st.job = &job;
    sl.st.cx = &rcx_;
    sl.st.begin();
    sl.pos = N;
    sl.next = first;
    sl.pend[0] = first;
    sl.n_pend = 1;
    sl.pend_pos = N;
    sl.last_acc = 0;
    sl.last_row = 0;
    sl.logits_ok = true;
    sl.td0 = Clock::now();
    sl.last_log_t = sl.td0;
    sl.last_log_n = 0;
    sl.dec_watch.reset();
    sl.done = emit(ct_, sl, first);
    flushOut(sl);
    sl.phase = Phase::decode;
    if (sl.done || sl.st.gone) finishJob(sl);
}

// Decode floor: the slots it protects. Every decoding slot is protected (a stream
// someone is reading), except one that arrived in the same burst as a request
// still prefilling (arrival times within the gather window): simultaneous
// arrivals do not hold one another back, so a burst merges its prefill forwards
// at full throughput. Arrival order does not matter otherwise: a conversation's
// next turn that arrives while earlier subagent prompts prefill is protected too.
// Empty when the floor is off or nothing is prefilling.
std::vector<DecodeFloor::SlotTok> Engine::floorProtected() const {
    std::vector<DecodeFloor::SlotTok> dec_toks;
    if (!fl_.floor.on()) return dec_toks;
    const double gather_ms = opt_.gather_ms > 0 ? opt_.gather_ms : 30.0;
    const auto gather_dur = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double, std::milli>(gather_ms));
    bool has_pf = false;
    for (const Slot& sl : slots_) has_pf = has_pf || (sl.phase == Phase::prefill && sl.job);
    if (!has_pf) return dec_toks;
    for (const Slot& sl : slots_) {
        if (sl.phase != Phase::decode || !sl.job) continue;
        bool same_burst = false;
        for (const Slot& pf : slots_) {
            if (pf.phase != Phase::prefill || !pf.job) continue;
            const TimePoint a = sl.job->t_arrive, b = pf.job->t_arrive;
            same_burst = same_burst || (a > b ? a - b : b - a) <= gather_dur;
        }
        if (!same_burst) dec_toks.push_back({sl.id, sl.reqId(), sl.n_gen});
    }
    return dec_toks;
}

// one prefill chunk (oldest request first), with the decode floor's budget
void Engine::runPrefill(Slot* pf, bool& idle_wait, std::size_t& pf_rows, bool& floor_pf) {
    // decode floor: the slots it protects this iteration
    const std::vector<DecodeFloor::SlotTok> dec_toks = floorProtected();
    if (fl_.floor.on() && dec_toks.empty()) fl_.floor.idle();
    auto floorNow = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - fl_.epoch).count(); };
    if (pf) {
        // a burst in flight: hold the first chunk briefly so the other
        // requests join its segmented forward
        const bool gather = opt_.seg_prefill && pf->pf_off == 0 && opt_.gather_ms > 0 && n_building_.load() > 0 &&
                            msSince(pf->tp0) < opt_.gather_ms;
        if (gather) {
            idle_wait = true;
        } else if (!dec_toks.empty() && !fl_.floor.prefillAllowed(floorNow(), dec_toks)) {
            // decoding slots are below the floor in this period: a decode cycle alone
            ++fl_.waits;
        } else {
            std::size_t budget = static_cast<std::size_t>(-1);
            if (!dec_toks.empty()) {
                fl_.floor.startPeriod(floorNow(), dec_toks);
                const DecodeFloor::Plan pl = fl_.floor.plan(chunkLen(*pf), m_.maxBatch());
                budget = pl.rows;
                floor_pf = true;
                ++fl_.periods;
                if (msSince(fl_.log_t) >= 10000) {
                    fl_.log_t = Clock::now();
                    if (pl.cycles > 0)
                        logI("decode floor | {:.0f} tok/s per decoding slot ({} decoding): prefill forward <= {} rows, ~{} decode "
                             "cycles between forwards{} (cycle {:.1f} ms, prefill {:.0f} rows/s)",
                             fl_.floor.minTps(), dec_toks.size(), pl.rows, pl.cycles,
                             pl.reachable ? "" : " (floor out of reach: prefill keeps 20% of the time)", fl_.floor.cycleMs(),
                             fl_.floor.rowsPerMs() * 1000.0);
                    else
                        logI("decode floor | {:.0f} tok/s per decoding slot ({} decoding): prefill forward <= {} rows "
                             "(not measured yet)",
                             fl_.floor.minTps(), dec_toks.size(), pl.rows);
                }
            }
            pf_rows = prefillGroup(*pf, budget);
            if (spec_.prof) spec_.prof->last_end.reset();
            // FR-SRV P3-12: chunks of <= max_small_batch rows cannot join a segmented forward;
            // they ran only once their request was the oldest, so a short follow-up (a few new
            // tokens on a cached prefix) waited for a whole 128k prefill. They run now, alone
            // (the same forward they would get later: outputs unchanged).
            for (Slot& o : slots_) {
                if (&o == pf || o.phase != Phase::prefill || o.job == nullptr) continue;
                if (chunkLen(o) > max_small_batch) continue;
                try {
                    prefillStep(o);
                } catch (const std::exception& ex) {
                    failSlot(o, ex.what());
                }
            }
        }
    }
}

void Engine::noteFloor(TimePoint tl2, TimePoint tl3, std::size_t pf_rows, bool floor_pf) {
    // measurements: the cycle synchronizes the stream, so [tl2, now] covers
    // this iteration's prefill forward too
    const TimePoint te = Clock::now();
    const double it_ms = std::chrono::duration<double, std::milli>(te - tl2).count();
    const double cyc_ms = std::chrono::duration<double, std::milli>(te - tl3).count();
    std::vector<DecodeFloor::SlotTok> now_dec;
    for (const Slot& sl : slots_)
        if (sl.phase == Phase::decode) now_dec.push_back({sl.id, sl.reqId(), sl.n_gen});
    fl_.floor.noteCycle(cyc_ms, pf_rows > 0, now_dec);
    if (floor_pf && pf_rows > 0) fl_.floor.notePrefill(pf_rows, it_ms - std::min(fl_.floor.cycleMs(), 0.5 * it_ms));
}

}  // namespace whirl::server
