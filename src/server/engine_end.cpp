// Server engine, the end of a request (arch025-engine S11, module server.slot_end): finishJob, releaseSlot, failSlot
// and the disconnected-client reaper; bodies verbatim from engine.cpp / engine_queue.cpp (S16 file move).
// SPDX-License-Identifier: Apache-2.0

#include "engine.h"
#include "engine_internal.h"

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

void SlotEnd::finishJob(Slot& sl) {
    Job& job = *sl.job;
    const std::uint64_t E = m_.cfg().n_embd;
    const std::uint32_t N = sl.n_prompt;
    m_.selectSeq(sl.id);
    ops_.streamSync(m_.stream());
    const double gen_ms = msSince(sl.td0);
    if (sl.stopped_str) sl.finish = "stop";
    // ---- checkpoint after the last token the trunk has seen
    if (sl.pos > N && sl.pos == sl.cache_tokens.size() && sl.logits_ok) {
        const std::optional<DevPtr> last_hid =
            opt_.use_mtp ? std::optional<DevPtr>(m_.mtpH() + static_cast<std::uint64_t>(sl.last_acc) * E * 4) : std::nullopt;
        sc_.saveCkpt(sl, sl.pos, "gen-end", last_hid, sb_.logitsRow(sl.last_row));
        // MTP KV rows for the accepted drafts of the last cycle (normally
        // written at the start of the next cycle)
        if (opt_.use_mtp && sl.n_pend > 1)
            m_.mtpEnqueue(m_.mtpH(), std::span<const std::uint32_t>(sl.pend.data(), sl.n_pend - 1), sl.pend_pos);
    } else if (sl.pos == sl.cache_tokens.size()) {
        // no gen-end checkpoint; the MTP KV rows of the last cycle are still needed
        if (opt_.use_mtp && sl.n_pend > 1)
            m_.mtpEnqueue(m_.mtpH(), std::span<const std::uint32_t>(sl.pend.data(), sl.n_pend - 1), sl.pend_pos);
    } else {
        logW("req {} | slot {} | cache bookkeeping mismatch (pos {}, tokens {}); dropping prefix cache", job.id, sl.id, sl.pos,
             sl.cache_tokens.size());
        sc_.dropCache(sl);
    }
    ops_.streamSync(m_.stream());

    // ---- final output
    Final fin = sl.st.finish(sl.out, sl.finish);
    if (sl.pool_cut && fin.reason == "length") fin.cause = "kv_pool_exhausted";
    const double tot_ms = msSince(sl.tp0);
    const std::uint32_t n_new = N - sl.reuse;
    Timings tim;
    tim.cache_n = sl.reuse;
    tim.prompt_n = n_new;
    tim.prompt_ms = sl.prompt_ms;
    // FR-SRV P3-7: tokens generated after the stop string in the same cycle are not completion tokens
    const std::uint32_t n_out = sl.n_gen - std::min(sl.n_gen, sl.n_after_stop);
    tim.predicted_n = n_out;
    tim.predicted_ms = gen_ms;
    tim.draft_n = sl.drafted;
    tim.draft_n_accepted = sl.accepted;
    sl.st.end(sl.out, fin, tim, N);

    // ---- summary (llama.cpp style)
    const double pn = n_new, gn = n_out;
    logI("req {} | slot {} | prompt eval time = {:>10.2f} ms / {:>5} tokens ({:>8.2f} ms per token, {:>8.2f} tokens per second)",
         job.id, sl.id, sl.prompt_ms, n_new, n_new > 0 ? sl.prompt_ms / pn : 0.0,
         sl.prompt_ms > 0 ? pn * 1000.0 / sl.prompt_ms : 0.0);
    logI("req {} | slot {} |        eval time = {:>10.2f} ms / {:>5} tokens ({:>8.2f} ms per token, {:>8.2f} tokens per second)",
         job.id, sl.id, gen_ms, n_out, n_out > 0 ? gen_ms / gn : 0.0, gen_ms > 0 ? gn * 1000.0 / gen_ms : 0.0);
    logI("req {} | slot {} |       total time = {:>10.2f} ms / {:>5} tokens", job.id, sl.id, tot_ms, n_new + n_out);
    if (opt_.use_mtp) {
        logI("req {} | slot {} | MTP: drafted {}, accepted {}, acceptance rate {:.1f}%, {:.2f} tokens per cycle ({} verify "
             "cycles)",
             job.id, sl.id, sl.drafted, sl.accepted, sl.drafted > 0 ? 100.0 * sl.accepted / sl.drafted : 0.0,
             sl.cycles > 0 ? static_cast<double>(sl.accepted + sl.cycles) / sl.cycles : 0.0, sl.cycles);
        if (opt_.ngram) logI("req {} | slot {} | n-gram drafts in {} of {} cycles", job.id, sl.id, sl.ng_cycles, sl.cycles);
    }
    // (whatever the draft source: with DFlash2 drafts it must stay 0, FR-DFL P2-3)
    if (opt_.relax.on && (opt_.use_mtp || opt_.use_dflash))
        logI("req {} | slot {} | relaxacc: {} of {} accepted drafts kept by the relaxed rule (greedy: mean ln p_max/p_kept {:.3f})",
             job.id, sl.id, sl.relaxed, sl.accepted, sl.relaxed > 0 ? sl.relax_gap / sl.relaxed : 0.0);
    if (!sl.greedy && (sl.sm.big_used > 0 || sl.sm.full_used > 0))
        logI("req {} | slot {} | sampler fallbacks: top-{} {}x, full row {}x", job.id, sl.id, k_big, sl.sm.big_used,
             sl.sm.full_used);
    logI("req {} | slot {} | finish_reason {}{}, completion {} tok, reasoning {} B, content {} B, tool_calls {}, cache now {} "
         "tok",
         job.id, sl.id, fin.reason, sl.st.gone ? " (client disconnected)" : "", n_out, fin.reason_len, fin.content_len,
         fin.n_tools, sl.cache_tokens.size());
    if (const std::string vram = vw_.vramNote(); !vram.empty()) logI("req {} | slot {} | {}", job.id, sl.id, vram);
    releaseSlot(sl);
    // item G: copy the idle slot's new content to the host tier (async)
    try {
        tb_.spillSlot(sl.id, sl.phase == Phase::idle, sl, sl);
    } catch (const std::exception& ex) {
        logW("slot {} | kv tier: spill failed: {} ({})", sl.id, ex.what(), ops_.lastErrorString());
    }
}

void SlotEnd::releaseSlot(Slot& sl) {
    if (sl.restore != nullptr && sl.phase != Phase::restore) tb_.finishTail(sl.id, sl, sl, true);
    Job* job = sl.job;
    if (!job) return;
    vis_.done(sl.id, job->vis.get());
    {
        whirl::MutexLock lk(q_mutex_);
        job->done = true;
        n_active_ -= 1;
        q_cond_.notify_all();
    }
    sl.job = nullptr;
    sl.phase = Phase::idle;
}

void SlotEnd::failSlot(Slot& sl, const std::string& err) {
    Job* job = sl.job;
    if (!job) return;
    if (sl.restore != nullptr && sl.phase != Phase::restore) tb_.finishTail(sl.id, sl, sl, true);
    // FR-SRV P2-3: the KV pool ran out while prefilling (decoding requests grew into the pages the
    // admission counted on): a retryable 503, never a 500
    const bool pool_full = err == "KvPoolFull" && sl.phase != Phase::decode;
    if (pool_full) logW("req {} | slot {} | KV pool full during prefill: answered 503 (retry later)", job->id, sl.id);
    else logE("req {} | slot {} | failed: {} ({})", job->id, sl.id, err, ops_.lastErrorString());
    if (sl.phase == Phase::decode && job->params.stream) {
        sl.st.fail();
    } else {
        const std::string r = pool_full ? errorResponse(503, "the KV cache pool is full (other requests hold every page); retry later",
                                                        "Retry-After: 5\r\n")
                                        : errorResponse(500, err);
        if (job->w) {
            job->w->write(r);
            job->w->flush();
        }
    }
    sc_.dropCache(sl);
    releaseSlot(sl);
}

void SlotEnd::reapDisconnected() {
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
        sc_.dropCache(sl);
        releaseSlot(sl);
    }
}

}  // namespace whirl::server
