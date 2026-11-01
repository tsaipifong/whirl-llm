// Server engine, the start of a request (arch025-engine S14, module server.slot_start): startJob; body verbatim from
// engine.cpp (S16 file move).
// SPDX-License-Identifier: Apache-2.0

// arch025-regb1 (N4-reg b1): the component's own header, not engine.h / engine_internal.h (no include edge back to server.engine)
#include "server/engine_start.h"
#include "server/engine_output.h"

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

bool SlotStart::startJob(Slot& sl, Job& job) {
    const Params& p = job.params;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t N = static_cast<std::uint32_t>(toks.size());
    const std::uint32_t nd_max = dc_.draftRoom();
    const auto& cfg = m_.cfg();
    sl.job = &job;
    sl.last_used = ps_.cache().touch();
    // (v0.2.2.2: ctxForNew = the context per request, lowered by the memory governor for new requests only)
    const std::uint32_t ctx_new = gv_.ctxForNew();
    if (N + nd_max + 2 >= ctx_new || static_cast<std::uint64_t>(N) + nd_max + 2 >= static_cast<std::uint64_t>(pool_.pages()) * kv_page) {
        if (ctx_new < ctx_ && N + nd_max + 2 < ctx_)
            logW("req {} | prompt of {} tokens does not fit the context per request of {} tokens (lowered from {} by the memory "
                 "governor while the WDDM budget is low)",
                 job.id, N, ctx_new, ctx_);
        else if (!opt_.ctx_note.empty())
            logW("req {} | prompt of {} tokens does not fit the context per request of {} tokens ({})", job.id, N, ctx_, opt_.ctx_note);
        writeError(job, 400, "prompt is too long for the context size");
        return false;
    }
    const std::uint32_t room = ctx_new - N - nd_max - 2;
    sl.max_tokens = std::min(p.max_tokens.value_or(room), room);
    sl.greedy = p.temperature <= 0 || !std::isfinite(1.0f / p.temperature);
    sl.ignore_eos = p.ignore_eos;
    sl.n_prompt = N;
    std::uint64_t seed = p.seed;
    if (!p.seed_given) seed = ids_.next();
    sl.sm = Sampler{};
    sl.sm.temp = p.temperature;
    sl.sm.inv_t = sl.greedy ? 1.0f : 1.0f / p.temperature;
    sl.sm.top_k = p.top_k;
    sl.sm.top_p = p.top_p;
    sl.sm.min_p = p.min_p;
    sl.sm.seed = seed;
    sl.sm.race = opt_.spec_sample;
    sl.sm.cands = sb_.cands();
    if (sl.greedy) {
        logI("req {} | slot {} | sampling: greedy (temperature 0), max_tokens {}{}, stop strings {}{}", job.id, sl.id,
             sl.max_tokens, p.max_tokens ? "" : " (ctx limit)", p.stop.size(), p.ignore_eos ? ", ignore_eos" : "");
    } else {
        logI("req {} | slot {} | sampling: temperature {:.2f}, top_k {}, top_p {:.3f}, min_p {:.3f}, seed {}{}, max_tokens "
             "{}{}, stop strings {}{}",
             job.id, sl.id, p.temperature, p.top_k, p.top_p, p.min_p, seed, p.seed_given ? "" : " (random)", sl.max_tokens,
             p.max_tokens ? "" : " (ctx limit)", p.stop.size(), p.ignore_eos ? ", ignore_eos" : "");
    }
    if (p.presence_penalty != 0 || p.frequency_penalty != 0) {
        logW("req {} | slot {} | presence_penalty {:.2f} / frequency_penalty {:.2f} are accepted but not applied", job.id,
             sl.id, p.presence_penalty, p.frequency_penalty);
    }

    // ---- prefix cache lookup (this slot's KV region and checkpoints)
    PrefixStore::Match mt = p.cache_prompt ? ps_.cacheMatch(sl, toks) : PrefixStore::Match{};
    // item S: a shared prefix checkpoint longer than the slot's own reusable prefix
    Spe* spe_hit = nullptr;
    if (p.cache_prompt) {
        if (Spe* s = ps_.speMatch(toks); s && s->n > mt.reuse) spe_hit = s;
    }
    const std::uint32_t sys = ps_.sysSplit(toks);
    const std::uint32_t lcp_t = ps_.lcpTarget(slots_, toks, spe_hit ? spe_hit->n : mt.reuse, sys, sl.id, sl.tier_id);
    if (spe_hit) {
        sc_.attachSpe(sl, *spe_hit);
        mt = PrefixStore::Match{spe_hit->n, &spe_hit->ck, std::max<std::size_t>(mt.lcp, spe_hit->n)};
    }
    sl.reuse = mt.reuse;
    // the state at reuse is the cold one (a shared checkpoint is cold-exact)
    sl.on_sched = mt.reuse == 0 || spe_hit != nullptr ||
                  (mt.ck != nullptr && ckKindCode(mt.ck->kind) >= 4 && ps_.onSchedule(toks, mt.reuse)) ||
                  (opt_.cold_resume && mt.ck != nullptr);  // COLD-RESUME: every accepted checkpoint is cold-exact
    // positions >= reuse are rewritten: pages a spill still reads, or that
    // are shared, are replaced (copy on write), never waited for
    bool spill = false;
    if (sl.spill_busy && mt.reuse < sl.spill_upto) {
        if (ops_.eventDone(sl.spill_ev.get())) sl.spill_busy = false;
        else spill = true;
    }
    if (spill || pool_.sharedFrom(sl.pages, mt.reuse)) sc_.cowPages(sl, mt.reuse, spill);
    sl.tier_pages = std::min(sl.tier_pages, mt.reuse / kv_page);
    // a request that reuses nothing of the slot starts a new lineage
    if (mt.reuse == 0) {
        sl.tier_id = 0;
        sl.tier_pages = 0;
    }
    for (Ckpt& c : sl.ckpts)
        if (c.valid && c.pos > mt.reuse) c.valid = false;
    sl.cache_tokens.resize(std::min<std::size_t>(sl.cache_tokens.size(), mt.reuse));
    if (opt_.use_dflash) {
        // the slot's own prefix keeps its drafter rows below reuse (attachSpe reset them for a
        // shared checkpoint); positions >= reuse are rewritten
        if (mt.reuse == 0) {
            dc_.dflReset(sl, 0);
        } else {
            // (FR-DFL P3-3: only rows the context ring still holds count; none left -> cold at reuse)
            const ::whirl::dflash::Trim tr =
                ::whirl::dflash::trimForReuse(::whirl::dflash::Range{sl.dfl_lo, sl.dfl_hi}, mt.reuse, m_.dflashRing());
            if (tr.reset) {
                dc_.dflReset(sl, mt.reuse);
            } else {
                m_.dflashTrim(sl.id, mt.reuse);
                sl.dfl_lo = tr.r.lo;
                sl.dfl_hi = tr.r.hi;
            }
        }
    }
    {
        const std::size_t need_p =
            static_cast<std::size_t>((static_cast<std::uint64_t>(N) + nd_max + 2 + kv_page - 1) / kv_page);
        if (sl.pages.size() > need_p) {
            const std::span<const std::int32_t> extra(sl.pages.data() + need_p, sl.pages.size() - need_p);
            if (sl.spill_busy && !ops_.eventDone(sl.spill_ev.get())) pool_.quarantine(extra);
            else pool_.unref(extra);
            sl.pages.resize(need_p);
            sl.tier_pages = std::min(sl.tier_pages, static_cast<std::uint32_t>(need_p));
        }
    }
    if (opt_.no_prefix_cache) {
        logI("req {} | slot {} | prompt {} tok, prefix cache disabled, prefill {} new", job.id, sl.id, N, N);
    } else if (mt.ck) {
        logI("req {} | slot {} | prompt {} tok, common prefix {}, reused {} tok (checkpoint '{}' @{}), prefill {} new",
             job.id, sl.id, N, mt.lcp, mt.reuse, mt.ck->kind, mt.ck->pos, N - mt.reuse);
    } else {
        logI("req {} | slot {} | prompt {} tok, common prefix {}, reused 0 tok (no usable checkpoint), prefill {} new",
             job.id, sl.id, N, mt.lcp, N);
    }
    sl.tp0 = sl.tp0_pre ? *sl.tp0_pre : Clock::now();
    sl.tp0_pre.reset();
    // vision: encode (or reuse) the images whose tokens are prefilled
    vis_.prepare(sl.id, job.vis.get(), job.id, opt_.verify_cache ? 0 : mt.reuse);
    m_.selectSeq(sl.id);
    sl.ck_hid.reset();
    if (mt.ck) {
        const Ckpt& c = *mt.ck;
        ck_.load(c);
        sl.ck_hid = c.hid;
        if (spe_hit != nullptr && opt_.use_mtp) {
            // own copy of the MTP hidden row: the shared checkpoint may be
            // replaced before this slot's first chunk runs
            const DevPtr own = sl.ckpts[0].hid;
            tb_.waitSpill(sl);
            ops_.copyAsync(own, c.hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
            sl.ck_hid = own;
        }
        if (mt.reuse == N) {
            ops_.copyAsync(m_.logits(), c.logits, static_cast<std::uint64_t>(cfg.n_vocab) * 4, m_.stream());
            if (opt_.use_mtp) ops_.copyAsync(m_.mtpH(), c.hid, static_cast<std::uint64_t>(cfg.n_embd) * 4, m_.stream());
        }
    } else {
        m_.reset();
    }
    sl.cache_tokens.insert(sl.cache_tokens.end(), toks.begin() + mt.reuse, toks.end());
    sl.pf_off = 0;
    {
        const std::uint32_t sp = static_cast<std::uint32_t>(ct_.thinkOpenSplit(toks));
        sl.pf_split = sp > mt.reuse ? sp : 0;
        sl.sys_split = sys > mt.reuse ? sys : 0;
        sl.lcp_ck = (sl.on_sched && lcp_t > mt.reuse && lcp_t != sl.sys_split) ? lcp_t : 0;
        if (sl.lcp_ck > 0)
            logI("req {} | slot {} | shared prefix: common prefix with an earlier session; keeping a 'prefix' checkpoint @{}",
                 job.id, sl.id, sl.lcp_ck);
        // vision: a shared checkpoint at the end of the last image (follow-up questions
        // about the same image in new conversations start there)
        sl.img_ck = 0;
        if (sl.on_sched && job.vis) {
            const std::vector<std::uint32_t> imgs = ps_.imgSplits(toks);
            if (!imgs.empty()) {
                const std::uint32_t b = imgs.back();
                if (b > mt.reuse && b != sl.sys_split && b != sl.lcp_ck) {
                    sl.img_ck = b;
                    logI("req {} | slot {} | shared prefix: keeping a 'prefix' checkpoint at the end of the image @{}", job.id, sl.id, b);
                }
            }
        }
        // TAIL-CK: the same long document with another question (one message each, REL3) shares
        // everything up to the question; keep a slot checkpoint at the start of the prompt's last
        // schedule chunk (257..1280 tokens before the end), so the next request resumes there instead
        // of re-prefilling the whole document. Only on the cold schedule (kind 'prefix': reused only
        // where it is on the new prompt's schedule too, as for the shared checkpoints), and only with
        // enough slots that prompt-end / gen-end stay (EngineOptions::tail_ck, tail_ck_slots).
        sl.tail_ck = ps_.tailCk(sl, toks);
        if (sl.tail_ck > 0)
            logI("req {} | slot {} | keeping a tail checkpoint @{} ({} tokens before the prompt end)", job.id, sl.id,
                 sl.tail_ck, N - sl.tail_ck);
    }
    sl.phase = Phase::prefill;
    // reset per-request counters
    sl.n_gen = 0;
    sl.finish = "length";
    sl.done = false;
    sl.stopped_str = false;
    sl.pool_cut = false;
    sl.n_after_stop = 0;
    sl.cycles = 0;
    sl.drafted = 0;
    sl.accepted = 0;
    sl.relaxed = 0;
    sl.relax_gap = 0;
    sl.n_batch = 0;
    sl.acc_ema = static_cast<float>(opt_.n_draft);
    sl.dacc = qwen35::DraftAccept{};
    sl.sacc = qwen35::SlotAccept{};
    sl.d_prev = 0;
    sl.ng.verify_slope = opt_.ngram_slope;
    sl.ng.reset();
    sl.ng.ng.norm = crlf_norm_;
    sl.ng_cycles = 0;
    sl.dfl_acc = qwen35::DraftAccept{};
    sl.dfl_nd = static_cast<float>(m_.dflashMaxDrafts());
    sl.dfl_cycles = 0;
    // relaxacc: a solo request starts from the same draft-policy state every time (the kept
    // tokens depend on the drafts), so its output depends only on prompt, seed and settings
    if ((opt_.timing_reset || opt_.relax.on) && (opt_.use_mtp || opt_.use_dflash)) {
        bool others = false;
        for (Slot& o : slots_) others = others || (&o != &sl && o.phase == Phase::decode);
        if (!others) {
            for (auto& t : spec_.timing) t = qwen35::DraftTiming{};
            for (auto& c : spec_.ccost) c.reset();
            spec_.ccost_primed.fill(false);
            spec_.dpool = qwen35::DraftAccept{};
            for (auto& t : spec_.dtiming) t = qwen35::DraftTiming{};
            spec_.prev_nd = 0;
            spec_.policy_cycles = 0;
        }
    }
    if (mt.reuse == N) finishPrefill(sl);
    return true;
}

}  // namespace whirl::server
