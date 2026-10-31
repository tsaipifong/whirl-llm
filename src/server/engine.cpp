// Reimplements the WHIRL Zig research prototype's server.zig (Engine, Slot, Spe,
// CycleProf; the decode / prefill / prefix-cache / host-tier logic).
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

// ---------------------------------------------------------------------------
// construction

namespace {
// the prefix store's settings (arch024 M4); the constructor body changes none of these options
PrefixStore::Config storeConfig(const EngineOptions& o) {
    PrefixStore::Config c;
    c.enabled = !o.no_prefix_cache;
    c.lcp_on = o.lcp_on;
    c.cold_resume = o.cold_resume;
    c.sys_min = o.sys_min;
    c.tail_ck = o.tail_ck;
    c.tail_ck_min = o.tail_ck_min;
    c.kv_page = kv_page;
    return c;
}

// the tier bridge's settings (arch024 M5); the constructor body changes none of these options
TierBridge::Config bridgeConfig(const EngineOptions& o) {
    TierBridge::Config c;
    c.verify = o.tier_verify;
    c.supersede = o.kv_supersede;
    c.idle_flush_ms = o.kv_idle_flush_ms;
    c.flush_ms = Engine::shutdown_flush_ms;
    c.flush_long_ms = Engine::shutdown_flush_long_ms;
    c.min_gain = o.tier_min_gain;
    return c;
}
}  // namespace

Engine::Engine(ServerModel& model, device::Backend& ops, const Tokenizer& tok, const EngineOptions& opt)
    : m_(model), ops_(ops), tok_(tok), ct_(tok), opt_(opt), ctx_(opt.ctx),
      ck_(ops, model, CkptStore::Layout{model.convBytes(), model.ssmBytes(), opt.ckpt_host}), ids_(opt.seed),
      rcx_{opt_.model_name, tok_, ids_}, pool_(ops, model), ps_(ct_, storeConfig(opt_)),
      tb_(ops, model, ct_, ps_, kv_arrays_, bridgeConfig(opt_)) {
    if (opt_.parallel < 1 || opt_.parallel > gdn_max_seg) throw std::runtime_error("parallel must be 1..16");
    if (opt_.n_ck == 0 || opt_.no_prefix_cache) opt_.n_spe = 0;
    if (opt_.no_prefix_cache) opt_.n_ck = 0;
    // FR-CACHE P2-3: a tier entry, Slot::rs_map / rs_tail and beginRestore's `chosen` hold at most max_ck
    // checkpoints (server_main clamps WHIRL_SERVE_CKPTS already; this guards every other caller)
    static_assert(std::tuple_size_v<decltype(Slot::rs_map)> == tier::max_ck);
    static_assert(std::tuple_size_v<decltype(Slot::rs_tail)> == tier::max_ck);
    static_assert(std::tuple_size_v<decltype(tier::Entry::ck)> == tier::max_ck);
    if (opt_.n_ck > tier::max_ck) {
        logW("prefix checkpoints per slot {} above the maximum {}; using {}", opt_.n_ck, tier::max_ck, tier::max_ck);
        opt_.n_ck = tier::max_ck;
    }
    if (opt_.profile > 0) {
        spec_.prof = std::make_unique<CycleProf>();
        spec_.prof->events = opt_.profile != 2;
    }
    conv_bytes_ = m_.convBytes();
    ssm_bytes_ = m_.ssmBytes();
    spec_.vrows = std::clamp(m_.verifyRows(), max_small_batch, qwen35::max_verify_rows);
    // FR-SRV P3-9: n-gram drafts run with MTP and with DFlash2
    if (opt_.ngram && (opt_.use_mtp || opt_.use_dflash)) crlf_norm_ = chat::ChatTokens::crlfToLf(tok_);
    // FR-SRV P2-5: --draft dflash without DeltaNet replay: literal drafts are capped by the snapshot
    // sets that exist, so none allocated = no draft ever (server_main allocates them before sizing
    // the KV pool; this covers every other caller)
    if (opt_.use_dflash && !m_.gdnReplay() && m_.snapSets() == 0) {
        m_.ensureSnapshots(std::max<std::uint32_t>(m_.dflashMaxDrafts(), 1));
        logI("DFlash2 without DeltaNet replay: {} snapshot sets for literal drafts", m_.snapSets());
    }
    fl_.floor.configure(opt_.decode_min_tps);
    fl_.epoch = Clock::now();
    ctx_pub_.store(ctx_, std::memory_order_relaxed);
    ck_.setOwner(&owner_);
    pool_.setOwner(&owner_);
    ps_.setOwner(&owner_);
    tb_.setOwner(&owner_);
}

Engine::~Engine() {
    // the checkpoint buffers (Ckpt), events (slots, shared checkpoints, quarantine) and sampling buffers release
    // themselves when the members are destroyed; Slot / Spe wait for their spill / ready fences first (F1)
}

Engine::StateMem Engine::allocStateMeasured() {
    StateMem r;
    r.before = ops_.memInfo();
    allocState();
    r.after = ops_.memInfo();
    return r;
}

void Engine::allocState() {
    slots_.resize(opt_.parallel);
    m_.selectSeq(0);
    for (std::uint32_t si = 0; si < opt_.parallel; ++si) {
        Slot& sl = slots_[si];
        sl.id = si;
        sl.rs_map.fill(-1);
        sl.ckpts.resize(opt_.n_ck);
        ck_.unpark(sl.ckpts, true);
    }
    ps_.spes().resize(opt_.n_spe);
    for (Spe& s : ps_.spes()) ck_.init(s.ck, false);
    // sampling buffers
    sb_.alloc(opt_.parallel, opt_.relax.on, opt_.relax.on ? opt_.relax.topk() : 0);
    if (opt_.relax.on) {
        logI("decode: relaxed draft acceptance (relaxacc, fast): greedy top-{} with p >= {:.2f} p_max; sampling typical "
             "acceptance eps {:.2f} delta {:.2f} - output differs from exact speculative decoding",
             opt_.relax.topk(), opt_.relax.alpha, opt_.relax.eps, opt_.relax.delta);
    }
    spec_on_ = opt_.spec_sample && opt_.use_mtp && m_.draftSampleOk() && !opt_.relax.on;
    if (opt_.spec_sample && opt_.use_mtp)
        logI("sampling: speculative sampling of MTP drafts (specsample) {}", spec_on_ ? "on for temperature > 0 (race-coupled drafts; the text for a seed does not depend on drafts)" : opt_.relax.on ? "off (relaxacc typical acceptance instead)" : "unavailable (no draft_sample_rows kernel): exact-match acceptance");
}

void Engine::initPool() {
    const std::uint32_t n = m_.poolPages();
    kv_arrays_ = m_.kvArrays();
    pool_.init(n, kv_arrays_);
}

// the vision settings of the prefix store + the image embeddings (here, not in engine_vision.cpp: that file must not
// include engine.h, or server.image_embeddings and server.engine include each other)
void Engine::attachVision(const VisionConfig& vc) {
    ps_.setVision({vc.vis != nullptr, vc.vision_end_id, vc.ck_min});
    vis_.attach(vc);
}

EngineFacts Engine::facts() const {
    EngineFacts f;
    f.model_name = opt_.model_name;
    f.model_file = opt_.model_file;
    f.tmpl = opt_.tmpl;
    f.tok = &tok_;
    f.use_mtp = opt_.use_mtp;
    f.numerics_json = opt_.numerics_json;
    f.startup_json = opt_.startup_json;
    f.kv_ssd_mode = opt_.kv_ssd_mode;
    f.vision = vis_.config();
    const auto& cfg = m_.cfg();
    f.arch = cfg.archName();
    f.n_vocab = cfg.n_vocab;
    f.n_embd = cfg.n_embd;
    f.n_layer = cfg.n_layer;
    return f;
}

Engine::Health Engine::health() {
    Health h;
    {
        whirl::MutexLock lk(q_mutex_);
        h.active = n_active_;
        h.queued = queue_.size();
        h.slots = slots_.size();
    }
    whirl::MutexLock lk(vram_mu_);
    h.vram_ok = vram_last_.ok;
    h.vram_spill = vram_spill_;
    h.vram_local = vram_last_.local_usage;
    h.vram_budget = vram_last_.local_budget;
    h.vram_shared = vram_last_.nonlocal_usage;
    return h;
}

Engine::Stats Engine::stats() const {
    Stats s;
    s.spe_new = stat_spe_new_;
    s.spe_hits = stat_spe_hits_;
    s.spe_tok = stat_spe_tok_;
    s.spe_evict = stat_spe_evict_;
    s.unshare = stat_unshare_;
    s.evictions = stat_evictions_;
    s.cow_pages = pool_.stats().cow_pages;
    s.q_waits = pool_.stats().q_waits;
    s.spill_waits = tb_.spillWaits();
    s.vis_enc = vis_.encoded();
    s.vis_hit = vis_.hits();
    const TierBridge::Published tp = tb_.published();
    s.ssd_written = tp.ssd_written;
    s.ssd_dedup = tp.ssd_dedup;
    s.ssd_used = tp.ssd_used;
    s.ssd_entries = tp.ssd_entries;
    s.superseded = tp.superseded;
    s.dfl_cycles = bst_.dfl_cycles;
    s.dfl_drafted = bst_.dfl_drafted;
    s.dfl_accepted = bst_.dfl_accepted;
    s.dfl_cold = bst_.dfl_cold;
    s.dfl_gap = bst_.dfl_gap;
    return s;
}

Engine::Stats Engine::statsShared() const {
    Stats s;
    const TierBridge::Published tp = tb_.published();
    s.ssd_written = tp.ssd_written;
    s.ssd_dedup = tp.ssd_dedup;
    s.ssd_used = tp.ssd_used;
    s.ssd_entries = tp.ssd_entries;
    s.superseded = tp.superseded;
    return s;
}

// ---------------------------------------------------------------------------
// request lifecycle

bool Engine::startJob(Slot& sl, Job& job) {
    const Params& p = job.params;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t N = static_cast<std::uint32_t>(toks.size());
    const std::uint32_t nd_max = dc_.draftRoom();
    const auto& cfg = m_.cfg();
    sl.job = &job;
    sl.last_used = ps_.cache().touch();
    // (v0.2.2.2: ctxForNew = the context per request, lowered by the memory governor for new requests only)
    const std::uint32_t ctx_new = ctxForNew();
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
        attachSpe(sl, *spe_hit);
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
    if (spill || pool_.sharedFrom(sl.pages, mt.reuse)) cowPages(sl, mt.reuse, spill);
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

void Engine::finishJob(Slot& sl) {
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
        saveCkpt(sl, sl.pos, "gen-end", last_hid, sb_.logitsRow(sl.last_row));
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
        dropCache(sl);
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
    if (const std::string vram = vramNote(); !vram.empty()) logI("req {} | slot {} | {}", job.id, sl.id, vram);
    releaseSlot(sl);
    // item G: copy the idle slot's new content to the host tier (async)
    try {
        tb_.spillSlot(sl.id, sl.phase == Phase::idle, sl, sl);
    } catch (const std::exception& ex) {
        logW("slot {} | kv tier: spill failed: {} ({})", sl.id, ex.what(), ops_.lastErrorString());
    }
}

// ---------------------------------------------------------------------------

void Engine::releaseSlot(Slot& sl) {
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

void Engine::failSlot(Slot& sl, const std::string& err) {
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
    dropCache(sl);
    releaseSlot(sl);
}

void Engine::logBatchStats(bool force) {
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
        const std::string vram = vramNote();
        logI("batch | slots busy {}/{} (decode {}, prefill {}) | {} cycles, {:.2f} slots/cycle, {:.2f} rows/cycle, {:.2f} "
             "drafts/cycle{} | aggregate tg {:.1f} tok/s, prefill {:.0f} tok/s over {:.1f} s{}{}",
             n_dec + n_pf, slots_.size(), n_dec, n_pf, bst_.cycles, static_cast<double>(bst_.slots) / cyc,
             static_cast<double>(bst_.rows) / cyc, static_cast<double>(bst_.drafts) / cyc, lit,
             static_cast<double>(bst_.tokens) * 1000.0 / el, static_cast<double>(bst_.prefill_tok) * 1000.0 / el, el / 1000.0,
             vram.empty() ? "" : " | ", vram);
    }
    bst_.t0 = Clock::now();
    bst_.tokens = bst_.cycles = bst_.rows = bst_.slots = bst_.drafts = bst_.lit_drafts = 0;
    if (bst_.seg_batches > 0)
        logI("batch | segmented prefill: {} forwards, {} chunks ({:.2f} chunks/forward)", bst_.seg_batches, bst_.seg_chunks,
             static_cast<double>(bst_.seg_chunks) / static_cast<double>(bst_.seg_batches));
    bst_.prefill_tok = bst_.seg_batches = bst_.seg_chunks = 0;
    if (fl_.periods > 0)
        logI("batch | decode floor: {} prefill forwards while decoding, {} decode cycles held prefill back", fl_.periods,
             fl_.waits);
    fl_.periods = fl_.waits = 0;
}

void Engine::runLoop() {
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
        govApply();
        // ---- admit
        std::uint32_t n_busy = 0;
        for (const Slot& sl : slots_)
            if (sl.phase != Phase::idle) ++n_busy;
        whirl::UniqueLock lk(q_mutex_);
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
            if (fl_.floor.on()) noteFloor(tl2, tl3, pf_rows, floor_pf);
        } else if (idle_wait || (restoring && pf == nullptr)) {
            // burst gather / host-tier restore: poll again in 1 ms, sooner when a
            // request is queued, a connection finishes building one, or the tier
            // IO thread reports progress
            whirl::UniqueLock lw(q_mutex_);
            q_cond_.wait_for(lw, std::chrono::milliseconds(1), [&]() WH_REQUIRES(q_mutex_) { return wake_seq_ != wake_seq0; });
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
bool Engine::idleWait(whirl::UniqueLock& lk, bool& was_busy) {
    if (was_busy) {
        WH_UNLOCK_HELD(lk, q_mutex_);
        logBatchStats(true);
        if (spec_.prof) spec_.prof->print(m_);
        was_busy = false;
        WH_RELOCK_HELD(lk, q_mutex_);
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
            WH_UNLOCK_HELD(lk, q_mutex_);
            flushTierOnStop();
            WH_RELOCK_HELD(lk, q_mutex_);
            runLoopCalls(lk);
            return true;
        }
        // WDDM-LOG: /health "vram" and the spill-over line stay current while idle (one sample a minute)
        std::optional<TimePoint> vram_due;
        if (vram_probe_) {
            WH_UNLOCK_HELD(lk, q_mutex_);
            vram_due = vramIdleTick();
            govApply();
            WH_RELOCK_HELD(lk, q_mutex_);
            if (!queue_.empty() || stop_) continue;
        }
        if (tb_.tier() != nullptr) {
            const std::uint64_t seq = wake_seq_;
            WH_UNLOCK_HELD(lk, q_mutex_);
            tb_.idlePoll(n_building_.load() > 0);
            const bool busy = tierTick();
            if (busy) logged = false;
            if (!busy && !logged) {
                tb_.logStats(pool_.stats().cow_pages, pool_.stats().q_waits);
                logged = true;
            }
            WH_RELOCK_HELD(lk, q_mutex_);
            if (busy) {
                // tier work in flight: polled every 5 ms, but a request (or the
                // tier IO thread) ends the wait at once (a sleep here delayed a
                // request by up to one Windows timer tick, 10-16 ms)
                q_cond_.wait_for(lk, std::chrono::milliseconds(5),
                                 [&]() WH_REQUIRES(q_mutex_) { return wake_seq_ != seq || !queue_.empty() || stop_ || !loop_calls_.empty(); });
                continue;
            }
        }
        // vision: while the encoder is loaded, wake every 100 ms for its idle timeout
        // (a timed wait on the queue condition, not a sleep: a request that arrives
        // meanwhile starts at once; the prototype slept 100 ms here)
        if (vis_.wakeNeeded()) {
            WH_UNLOCK_HELD(lk, q_mutex_);
            vis_.idle();
            WH_RELOCK_HELD(lk, q_mutex_);
            tier_wake_ = false;
            if (queue_.empty() && !stop_)
                q_cond_.wait_for(lk, std::chrono::milliseconds(100), [&]() WH_REQUIRES(q_mutex_) { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });
            continue;
        }
        tier_wake_ = false;
        std::optional<std::chrono::steady_clock::time_point> wake;
        if (tb_.tier() != nullptr) wake = tb_.tier()->nextWake();
        if (const auto idle_due = tb_.idleDue(); idle_due && (!wake || *idle_due < *wake)) wake = idle_due;  // TIER-IDLE
        if (vram_due && (!wake || *vram_due < *wake)) wake = *vram_due;  // WDDM-LOG idle sample
        if (wake) {
            // an SSD write is due at a known time, the periodic SSD expiry check, or the idle write: wake
            // then (re-checked every minute)
            const auto due = std::min(*wake + std::chrono::milliseconds(2), std::chrono::steady_clock::now() + std::chrono::seconds(60));
            q_cond_.wait_until(lk, due, [&]() WH_REQUIRES(q_mutex_) { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });
        } else {
            q_cond_.wait(lk, [&]() WH_REQUIRES(q_mutex_) { return !queue_.empty() || tier_wake_ || stop_ || !loop_calls_.empty(); });
        }
    }
    // a request arrived: the idle period ends, an idle write stops starting new entries (TIER-IDLE)
    tb_.idleActivity();
    if (tb_.tier() != nullptr) tb_.tier()->endIdleWrite();
    bst_.t0 = Clock::now();
    // v0.2.2.2: the governor sees the budget before the first request after an idle period (one probe sample,
    // ~0.3 ms) when the last sample is older than 5 s, so a pool that must shrink does so before the request runs
    if (gov_ && vram_probe_ && msSince(vram_last_t_) > 5000.0) {
        WH_UNLOCK_HELD(lk, q_mutex_);
        (void)vramNote();
        govApply();
        WH_RELOCK_HELD(lk, q_mutex_);
    }
    return false;
}

// the queued requests that start this iteration (q_mutex_ held by the caller)
std::vector<Job*> Engine::admitQueued(std::uint32_t n_busy) {
    std::vector<Job*> admit;
    std::size_t qi = 0;
    std::optional<std::size_t> pool_avail;  // FR-SRV P2-3 (computed when a request is about to start)
    // v0.2.2.2 memory governor: parked slots take no request; while a smaller pool waits for the running requests,
    // none is admitted (they queue)
    std::size_t admit_cap = slots_.size();
    if (gov_) {
        std::size_t n_parked = 0;
        for (const Slot& sl : slots_) n_parked += sl.parked ? 1 : 0;
        admit_cap = gov_hold_ && n_busy > 0 ? n_busy : slots_.size() - n_parked;
    }
    while (qi < queue_.size() && n_busy + admit.size() < admit_cap) {
        // item S: a new session whose system message another request is
        // prefilling right now waits for that checkpoint
        if (waitsForSys(*queue_[qi]) || waitsForTier(*queue_[qi])) {
            ++qi;
            continue;
        }
        // FR-SRV P2-3: a prompt the KV pool cannot hold next to the running requests waits in the
        // queue (in order: later requests wait behind it) instead of failing mid-prefill. Alone it
        // always starts (startJob rejects a prompt larger than the whole pool).
        {
            if (!pool_avail) pool_avail = poolAvailable();
            Job& qj = *queue_[qi];
            const std::size_t need = pagesFor(qj.tokens.size());
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
        admit.push_back(queue_[qi]);
        queue_.erase(queue_.begin() + static_cast<std::ptrdiff_t>(qi));
        n_active_ += 1;
    }
    return admit;
}

void Engine::startAdmitted(const std::vector<Job*>& admit) {
    std::size_t n_requeued = 0;
    for (Job* job : admit) {
        // item T3: admitted together with a request whose restore just
        // started on the same prefix: back to the queue, wait for it
        if (waitsForTier(*job)) {
            whirl::MutexLock g(q_mutex_);
            queue_.insert(queue_.begin() + static_cast<std::ptrdiff_t>(n_requeued), job);
            ++n_requeued;
            n_active_ -= 1;
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
