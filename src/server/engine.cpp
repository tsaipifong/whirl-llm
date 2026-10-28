// Reimplements the WHIRL Zig research prototype's server.zig (Engine, Slot, Spe,
// CycleProf; the decode / prefill / prefix-cache / host-tier logic).
// SPDX-License-Identifier: Apache-2.0

#include "engine.h"

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

namespace {

using cache::lcpLen;
using cache::prefixEq;

struct KvPoolFull : std::runtime_error {
    KvPoolFull() : std::runtime_error("KvPoolFull") {}
};

}  // namespace

// ---------------------------------------------------------------------------
// WHIRL_PROFILE decode-cycle profile (index = decoding slots)

struct CycleProf {
    static constexpr std::size_t n = gdn_max_seg + 1;
    std::array<std::uint64_t, n> cycles{}, rows{}, drafts{}, tokens{};
    std::array<double, n> wall_ms{}, host_ms{}, gap_ms{}, post_ms{}, enq_ms{};
    std::optional<TimePoint> last_end;
    std::optional<TimePoint> t_ready;
    bool events = true;

    void print(ServerModel& m) {
        for (std::size_t a = 1; a < n; ++a) {
            if (cycles[a] == 0) continue;
            const double c = static_cast<double>(cycles[a]);
            std::array<double, ServerModel::n_prof_classes> pm{}, pv{};
            m.profileTake(a, pm, pv);
            double tm = 0, tv = 0;
            for (double v : pm) tm += v;
            for (double v : pv) tv += v;
            logI("profile | slots {} | {} cycles, {:.2f} rows, {:.2f} drafts/slot, {:.2f} tok/cycle | wall {:.2f} ms/cycle "
                 "(pre-launch host {:.2f}, gap before cycle {:.2f}) | GPU mtp {:.2f} + verify {:.2f} ms | {:.1f} tok/s in "
                 "cycles",
                 a, cycles[a], static_cast<double>(rows[a]) / c, static_cast<double>(drafts[a]) / c / static_cast<double>(a),
                 static_cast<double>(tokens[a]) / c, wall_ms[a] / c, host_ms[a] / c, gap_ms[a] / c, tm / c, tv / c,
                 static_cast<double>(tokens[a]) * 1000.0 / (wall_ms[a] + gap_ms[a]));
            for (std::size_t i = 0; i < ServerModel::n_prof_classes; ++i) {
                if (pm[i] > 0 || pv[i] > 0)
                    logI("profile | slots {} |   {:<9} mtp {:>6.3f}  verify {:>6.3f} ms/cycle", a, m.profileClassName(i),
                         pm[i] / c, pv[i] / c);
            }
            logI("profile | slots {} |   host: enqueue {:.3f} ms, after readCtl {:.3f} ms per cycle", a, enq_ms[a] / c,
                 post_ms[a] / c);
            cycles[a] = rows[a] = drafts[a] = tokens[a] = 0;
            wall_ms[a] = host_ms[a] = gap_ms[a] = post_ms[a] = enq_ms[a] = 0;
        }
        last_end.reset();
    }
};

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
        prof_ = std::make_unique<CycleProf>();
        prof_->events = opt_.profile != 2;
    }
    conv_bytes_ = m_.convBytes();
    ssm_bytes_ = m_.ssmBytes();
    vrows_ = std::clamp(m_.verifyRows(), max_small_batch, qwen35::max_verify_rows);
    // FR-SRV P3-9: n-gram drafts run with MTP and with DFlash2
    if (opt_.ngram && (opt_.use_mtp || opt_.use_dflash)) crlf_norm_ = chat::ChatTokens::crlfToLf(tok_);
    // FR-SRV P2-5: --draft dflash without DeltaNet replay: literal drafts are capped by the snapshot
    // sets that exist, so none allocated = no draft ever (server_main allocates them before sizing
    // the KV pool; this covers every other caller)
    if (opt_.use_dflash && !m_.gdnReplay() && m_.snapSets() == 0) {
        m_.ensureSnapshots(std::max<std::uint32_t>(m_.dflashMaxDrafts(), 1));
        logI("DFlash2 without DeltaNet replay: {} snapshot sets for literal drafts", m_.snapSets());
    }
    floor_.configure(opt_.decode_min_tps);
    floor_epoch_ = Clock::now();
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

void Engine::stop() {
    whirl::MutexLock lk(q_mutex_);
    stop_ = true;
    q_cond_.notify_all();
}

void Engine::runOnLoop(std::function<void()> fn, std::chrono::milliseconds timeout) {
    auto done = std::make_shared<std::promise<void>>();
    // REVIEW-ARCH4 P3-5: 0 queued, 1 claimed by the loop, 2 cancelled by a timed-out caller. fn usually captures the
    // caller's locals by reference, so it may only run while the caller waits: the loop claims the call before
    // running it, and a timed-out caller either cancels it (never runs) or, when already claimed, waits for it.
    auto state = std::make_shared<std::atomic<int>>(0);
    std::future<void> f = done->get_future();
    {
        whirl::MutexLock lk(q_mutex_);
        loop_calls_.push_back([fn = std::move(fn), done, state] {
            int queued = 0;
            if (!state->compare_exchange_strong(queued, 1)) return;  // cancelled: the caller is gone
            try {
                fn();
                done->set_value();
            } catch (...) {
                done->set_exception(std::current_exception());
            }
        });
        ++wake_seq_;
        q_cond_.notify_all();
    }
    if (f.wait_for(timeout) != std::future_status::ready) {
        int queued = 0;
        if (state->compare_exchange_strong(queued, 2)) throw std::runtime_error("Engine::runOnLoop: the engine loop did not run the call");
        f.wait();  // claimed just now: fn is running on the engine thread with the caller's references
    }
    f.get();
}

// called with lk held; runs the queued test calls without the lock (unlocks / relocks through WH_UNLOCK_HELD /
// WH_RELOCK_HELD, so the body is checked: arch025-tsaidle)
void Engine::runLoopCalls(whirl::UniqueLock& lk) {
    if (loop_calls_.empty()) return;
    std::vector<std::function<void()>> calls;
    calls.swap(loop_calls_);
    WH_UNLOCK_HELD(lk, q_mutex_);
    for (auto& f : calls) f();
    WH_RELOCK_HELD(lk, q_mutex_);
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
    f.vision = vis_cfg_;
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
    s.vis_enc = stat_vis_enc_;
    s.vis_hit = stat_vis_hit_;
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
// checkpoints (of the currently selected slot)

void Engine::saveCkpt(Slot& sl, std::uint32_t pos, const char* kind, std::optional<DevPtr> hid,
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

void Engine::dropCache(Slot& sl) {
    for (Ckpt& c : sl.ckpts) c.valid = false;
    sl.cache_tokens.clear();
    dflReset(sl, 0);
}

// ---------------------------------------------------------------------------
// DFlash2 drafter context (--draft dflash). Its rows come from the target features captured by
// each forward / verify; a slot's context is valid for the slot's own lineage only (same tokens at
// the same positions). Restores from a shared checkpoint or a host tier bring KV and state but no
// features: the context starts empty at the restore point and the slot drafts with DFlash2 again
// once dfl_warm rows were ingested after it (re-running the target over earlier positions would
// rewrite KV rows the restored state already holds).

std::uint32_t Engine::draftRoom() const {
    if (opt_.use_mtp) return opt_.n_draft;
    if (opt_.use_dflash) return m_.dflashMaxDrafts();
    return 0;
}

void Engine::dflReset(Slot& sl, std::uint32_t at) {
    if (!opt_.use_dflash) return;
    m_.dflashTrim(sl.id, 0);
    sl.dfl_lo = sl.dfl_hi = at;
}

// Rows r0 .. r0 + n - 1 of the latest forward / verify of run_rows rows are positions pos0 .. of
// slot sl: ingest those whose features were captured.
void Engine::dflIngest(Slot& sl, std::uint32_t run_rows, std::uint32_t r0, std::uint32_t n, std::uint32_t pos0) {
    if (!opt_.use_dflash || n == 0) return;
    ::whirl::dflash::Range rg{sl.dfl_lo, sl.dfl_hi};
    const ::whirl::dflash::Ingest g = ::whirl::dflash::planIngest(rg, m_.captureN(), run_rows, r0, n, pos0);
    bst_.dfl_gap += g.gap;
    if (g.any) m_.dflashIngest(sl.id, g.cap_row, g.n, g.pos);
    sl.dfl_lo = rg.lo;
    sl.dfl_hi = rg.hi;
}

bool Engine::dflReady(const Slot& sl) const {
    return opt_.use_dflash && ::whirl::dflash::ready(::whirl::dflash::Range{sl.dfl_lo, sl.dfl_hi}, sl.pos, opt_.dfl_warm);
}

void Engine::freePages(Slot& sl) {
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

struct Engine::SlotEvictor final : PagePool::Evictor {
    Engine& e;
    Slot* sl;
    SlotEvictor(Engine& e_, Slot* sl_) : e(e_), sl(sl_) {}
    bool evictOne() override { return e.evictFor(sl); }
};

// the prefix cache's victim for a page sl (nullptr: none) needs: a shared checkpoint dropped or an idle slot's pages
// freed (true), or nothing to evict (false)
bool Engine::evictFor(Slot* sl) {
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
    stat_evictions_ += 1;
    freePages(*victim);
    return true;
}

std::optional<std::int32_t> Engine::takePage(Slot* sl) {
    SlotEvictor ev(*this, sl);
    return pool_.take(ev);
}

void Engine::cowPages(Slot& sl, std::uint32_t reuse, bool spill) {
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
    if (!spill) stat_unshare_ += 1;
    sl.tier_pages = std::min(sl.tier_pages, static_cast<std::uint32_t>(keep));
}

bool Engine::ensurePages(Slot& sl, std::uint64_t n_tok) {
    SlotEvictor ev(*this, &sl);
    return pool_.ensure(sl.pages, sl.id, n_tok, ev);
}

// ---------------------------------------------------------------------------
// item S: shared prefix checkpoints

void Engine::dropSpe(Spe& s) {
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
    stat_spe_evict_ += 1;
}

void Engine::createSpe(Slot& sl, std::span<const std::uint32_t> toks_all, std::uint32_t pos, const char* kind,
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
    stat_spe_new_ += 1;
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

void Engine::attachSpe(Slot& sl, Spe& s) {
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
    dflReset(sl, s.n);
    s.last_used = ps_.cache().touch();
    s.uses += 1;
    stat_spe_hits_ += 1;
    // the checkpoint's RAM tier entry was reused (served from VRAM here, so no tier restore marked it):
    // worth an SSD write under the seen-once filter, and on eviction in shutdown mode (TIER-EVICT)
    if (tb_.tier() != nullptr && s.tier_id != 0)
        if (tier::Entry* x = tb_.tier()->find(s.tier_id)) x->reused = true;
    stat_spe_tok_ += s.n;
}

bool Engine::waitsForSys(Job& job) {
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
// sampling

// ---------------------------------------------------------------------------
// request lifecycle

bool Engine::emit(Slot& sl, std::uint32_t t) {
    if (!sl.ignore_eos && (t == ct_.im_end || t == ct_.endoftext)) {
        sl.finish = "stop";
        return true;
    }
    sl.batch[sl.n_batch] = t;
    sl.n_batch += 1;
    sl.n_gen += 1;
    if (sl.n_gen >= sl.max_tokens) {
        sl.finish = "length";
        return true;
    }
    return false;
}

void Engine::flushOut(Slot& sl) {
    std::uint32_t n_after = 0;
    if (sl.st.push(sl.out, std::span<const std::uint32_t>(sl.batch.data(), sl.n_batch), &n_after)) {
        sl.done = true;
        sl.stopped_str = true;
        sl.n_after_stop = n_after;
    }
    sl.n_batch = 0;
}

bool Engine::startJob(Slot& sl, Job& job) {
    const Params& p = job.params;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t N = static_cast<std::uint32_t>(toks.size());
    const std::uint32_t nd_max = draftRoom();
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
            dflReset(sl, 0);
        } else {
            // (FR-DFL P3-3: only rows the context ring still holds count; none left -> cold at reuse)
            const ::whirl::dflash::Trim tr =
                ::whirl::dflash::trimForReuse(::whirl::dflash::Range{sl.dfl_lo, sl.dfl_hi}, mt.reuse, m_.dflashRing());
            if (tr.reset) {
                dflReset(sl, mt.reuse);
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
    visPrepare(sl, job, opt_.verify_cache ? 0 : mt.reuse);
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
            for (auto& t : timing_) t = qwen35::DraftTiming{};
            for (auto& c : ccost_) c.reset();
            ccost_primed_.fill(false);
            dpool_ = qwen35::DraftAccept{};
            for (auto& t : dtiming_) t = qwen35::DraftTiming{};
            prev_nd_ = 0;
            policy_cycles_ = 0;
        }
    }
    if (mt.reuse == N) finishPrefill(sl);
    return true;
}

std::size_t Engine::chunkLen(const Slot& sl) const {
    const std::size_t cur = sl.reuse + sl.pf_off;
    const std::vector<std::uint32_t> imgs = ps_.imgSplits(sl.job->tokens);
    const std::size_t N = sl.job->tokens.size();
    // With the decode floor on, any decoding slot keeps forwards at one chunk, also
    // an unprotected one of the same burst: next to a long prompt a 2048-row forward
    // doubled its stall (96k: 1.2 -> 3.3 s, ~1.5 tok/s) for ~4% faster prefill.
    // With the floor off (--decode-min-tps 0, throughput first) chunks merge anyway.
    std::size_t exec = opt_.prefill_exec;
    if (exec > prefill_chunk && floor_.on())
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
    dflIngest(sl, static_cast<std::uint32_t>(n), 0, static_cast<std::uint32_t>(n), sl.reuse + static_cast<std::uint32_t>(sl.pf_off));
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
        dflIngest(sl, static_cast<std::uint32_t>(total), static_cast<std::uint32_t>(r0), static_cast<std::uint32_t>(segs[k].n),
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
    sl.done = emit(sl, first);
    flushOut(sl);
    sl.phase = Phase::decode;
    if (sl.done || sl.st.gone) finishJob(sl);
}

// relaxacc repetition guard: would keeping draft dr[acc] (after sl.next and dr[0..acc)) end in
// an n-gram already in the slot's recent tokens?
// (the rule is shared with the CLI: spec::rules::relaxRepeats, src/spec/spec_rules.h)
bool Engine::relaxRepeats(const Slot& sl, std::span<const std::uint32_t> dr, std::uint32_t acc) const {
    return spec::rules::relaxRepeats(std::span<const std::uint32_t>(sl.cache_tokens), sl.next, dr, acc);
}

std::uint32_t Engine::pickDrafts(std::span<Slot* const> act) {
    if (!opt_.use_mtp) return 0;
    const std::uint32_t A = static_cast<std::uint32_t>(act.size());
    std::uint32_t nd = draftCap(A);
    if (nd == 0) return 0;
    if (opt_.mtp_auto) {
        std::vector<const qwen35::DraftAccept*> accs;
        std::vector<qwen35::DraftAccept> eff;
        eff.reserve(act.size());
        for (Slot* sl : act) {
            if (opt_.slot_drafts == 2 && act.size() >= 2) {  // one slot: exactly as before
                eff.push_back(sl->sacc.effective(dpool_));
                accs.push_back(&eff.back());
            } else {
                accs.push_back(&sl->dacc);
            }
        }
        const std::uint32_t pick =
            qwen35::pickDrafts(accs, timing_[std::min(A, gdn_max_seg)], nd, opt_.relax.on ? policy_cycles_ : n_cycles_, prev_nd_);
        prev_nd_ = pick;
        return pick;
    }
    if (opt_.mtp_adapt > 0) nd = spec::rules::adaptDrafts(act, [](const Slot* sl) { return sl->acc_ema; }, opt_.mtp_adapt, nd);
    return nd;
}

// arch025-E2: decodeCycle's phases. Cycle holds the values one decode cycle passes from one phase to the next
// (they were decodeCycle's locals; value-initialized per cycle, never kept across cycles).
struct Engine::Cycle {
    std::uint64_t E = 0;
    TimePoint t_cycle0{};
    std::vector<Slot*> act;  // the slots with context room
    std::size_t A = 0;
    std::uint32_t Au = 0;
    // n-gram drafts per slot (history = cache_tokens + next) when they promise
    // more tokens per ms than MTP / DFlash2 drafts. Literal drafts (n-gram, DFlash2) go to the
    // verify from the host: ng_n / ng_d, with dsrc telling which drafter made them.
    std::array<std::uint32_t, gdn_max_seg> ng_n{}, ng_ml{};
    std::array<std::array<std::uint32_t, qwen35::max_ng_drafts>, gdn_max_seg> ng_d{};
    std::array<std::uint8_t, gdn_max_seg> dsrc{};  // 0 MTP / none, 1 n-gram, 2 DFlash2
    std::size_t n_ng = 0;
    std::uint32_t snap_cap = 0, lit_cap = 0;
    std::uint32_t dfl_lim = 0;  // DFlash2: drafts per slot within the shared verify rows / snapshot sets
    TimePoint td0{};
    std::size_t n_dfl = 0;
    std::uint32_t dfl_max_n = 0;
    std::vector<Slot*> act_mtp;  // the MTP slots
    std::size_t n_mtp = 0;
    std::uint32_t nd = 0;
    std::array<std::uint32_t, gdn_max_seg> nd_m{};  // per-slot counts (MTP slots, in act_mtp order)
    std::uint32_t nd_max = 0;  // MTP steps = the largest
    TimePoint tc0{};
    std::size_t pa = 0;
    CycleProf* p = nullptr;
    std::array<MSeg, gdn_max_seg> msegs{};
    std::array<VSeg, gdn_max_seg> vsegs{};
    std::array<std::uint32_t, gdn_max_seg> row_of{}, snap_of{}, nd_of{};
    std::array<std::size_t, qwen35::max_drafts + 1> n_step{};
    std::uint32_t rows = 0;  // the verify's rows
    std::uint32_t rk = 0;
};

// relaxacc (fast) applies to MTP drafts only: one rule for every site below (FR-DFL P2-3)
bool Engine::relaxes(const Cycle& cy, std::size_t k) const {
    return whirl::dflash::relaxAppliesTo(static_cast<whirl::dflash::DraftKind>(cy.dsrc[k]), opt_.relax.on);
}

// the verify writes KV rows pos .. pos + drafts: stay inside the pages mapped
// for the slot and the context
std::uint32_t Engine::verifyRoom(const Slot& sl, std::uint32_t lim) const {
    const std::uint64_t mapped = static_cast<std::uint64_t>(sl.pages.size()) * kv_page;
    const std::uint64_t a1 = mapped > static_cast<std::uint64_t>(sl.pos) + 1 ? mapped - (sl.pos + 1ull) : 0;
    const std::uint64_t a2 = spec::rules::ctxRoom(sl.pos, ctx_);
    return static_cast<std::uint32_t>(std::min({static_cast<std::uint64_t>(lim), a1, a2}));
}

void Engine::decodeCycle(std::span<Slot* const> act_in) {
    Cycle cy;
    const auto& cfg = m_.cfg();
    cy.E = cfg.n_embd;
    cy.t_cycle0 = Clock::now();
    dropOutOfRoom(cy, act_in);
    cy.A = cy.act.size();
    if (cy.A == 0) return;
    draftLimits(cy);
    ngramDrafts(cy);
    dflashDrafts(cy);
    mtpDrafts(cy);
    layoutBatch(cy);
    draftSampling(cy);
    enqueueVerify(cy);
    sampleRows(cy);
    trainCycle(cy);
    cycleStats(cy);
    const sampling::Acceptor acceptor(opt_.relax);
    for (std::size_t k = 0; k < cy.A; ++k) acceptSlot(cy, k, acceptor);
    finishCycle(cy);
}

void Engine::dropOutOfRoom(Cycle& cy, std::span<Slot* const> act_in) {
    // slots out of context room end here
    const std::uint32_t nd_room = draftRoom();
    for (Slot* sl : act_in) {
        const bool room_ok = spec::rules::decodeRoom(sl->pos, nd_room, ctx_) && ensurePages(*sl, sl->pos + nd_room + 2);
        if (!room_ok) {
            if (spec::rules::decodeRoom(sl->pos, nd_room, ctx_))
            {
                logW("req {} | slot {} | kv pool exhausted at {} tokens (every page belongs to a running request); "
                     "finish_reason length, finish_cause kv_pool_exhausted",
                     sl->reqId(), sl->id, sl->pos);
                sl->pool_cut = true;
            }
            sl->finish = "length";
            sl->logits_ok = false;  // other work ran since its last cycle
            finishJob(*sl);
            continue;
        }
        cy.act.push_back(sl);
    }
}

void Engine::draftLimits(Cycle& cy) {
    // replay keeps no snapshot sets: the verify row budget is the only shared limit
    cy.snap_cap = m_.gdnReplay() ? vrows_ : std::min(qwen35::gdn_max_snap, m_.snapSets());
    // literal drafts (n-gram, DFlash2) per slot: the snapshot budget they had before CONC-8S
    cy.lit_cap = (m_.gdnReplay() || opt_.ng_snap_sets == 0) ? cy.snap_cap : std::min(cy.snap_cap, opt_.ng_snap_sets);
    cy.Au = static_cast<std::uint32_t>(cy.A);
    if (opt_.use_dflash) {
        cy.dfl_lim = ::whirl::dflash::draftLimit(m_.dflashMaxDrafts() + 1, qwen35::max_ng_drafts, cy.lit_cap, vrows_, cy.Au, m_.fusedDecode());
    }
}

void Engine::ngramDrafts(Cycle& cy) {
    const qwen35::DraftTiming& dtm = dtiming_[std::min<std::size_t>(cy.A, gdn_max_seg)];
    if (opt_.ngram && (opt_.use_mtp || opt_.use_dflash)) {
        const std::uint32_t lim = spec::rules::ngramDraftCap(cy.lit_cap, vrows_, cy.Au, opt_.ngram_max, m_.fusedDecode());
        const qwen35::DraftTiming& tm = opt_.use_dflash ? dtm : timing_[std::min<std::size_t>(cy.A, gdn_max_seg)];
        const std::uint32_t nd0 = spec::rules::ngramAltDrafts(prev_nd_, draftCap(cy.Au));
        for (std::size_t k = 0; k < cy.A; ++k) {
            if (lim == 0) break;
            Slot& sl = *cy.act[k];
            sl.cache_tokens.push_back(sl.next);
            struct Pop {
                std::vector<std::uint32_t>& v;
                ~Pop() { v.pop_back(); }
            } pop{sl.cache_tokens};
            const std::uint32_t lim_k = verifyRoom(sl, lim);
            if (lim_k == 0) continue;
            const auto mt = sl.ng.propose(sl.cache_tokens, opt_.ngram_min, std::span<std::uint32_t>(cy.ng_d[k].data(), lim_k));
            if (mt.n == 0) continue;
            // the alternative's tokens per ms: MTP with nd0 drafts, or DFlash2 with its usual draft
            // count (a slot whose drafter context is not ready verifies one row)
            float alt_score = 0;
            if (opt_.use_dflash) {
                const std::uint32_t dn = dflReady(sl) ? verifyRoom(sl, std::min(cy.dfl_lim, static_cast<std::uint32_t>(std::lround(sl.dfl_nd)))) : 0;
                if (auto t = dtm.estimate(dn)) alt_score = sl.dfl_acc.expected(dn) / *t;
            } else if (opt_.slot_drafts == 2 && cy.A >= 2) {
                // against this slot's own last MTP allocation and its effective acceptance (one slot: as before)
                const std::uint32_t d = sl.d_prev > 0 ? sl.d_prev : nd0;
                if (auto t = tm.estimate(d)) alt_score = sl.sacc.effective(dpool_).expected(d) / *t;
            } else {
                alt_score = spec::rules::mtpScore(sl.dacc, tm, nd0);
            }
            cy.ng_ml[k] = mt.mlen;
            cy.ng_n[k] = opt_.ngram_force ? static_cast<std::uint32_t>(mt.n) : sl.ng.choose(mt.n, mt.mlen, alt_score, tm);
            if (cy.ng_n[k] > 0) {
                cy.n_ng += 1;
                cy.dsrc[k] = 1;
            }
        }
    }
}

void Engine::dflashDrafts(Cycle& cy) {
    // DFlash2 drafts for the slots without n-gram drafts (their time counts in the DFlash2 cycle time)
    cy.td0 = Clock::now();
    if (opt_.use_dflash && cy.dfl_lim > 0) {
        for (std::size_t k = 0; k < cy.A; ++k) {
            if (cy.dsrc[k] != 0) continue;
            Slot& sl = *cy.act[k];
            if (!dflReady(sl)) {
                bst_.dfl_cold += 1;
                continue;
            }
            const std::uint32_t lim_k = verifyRoom(sl, cy.dfl_lim);
            if (lim_k == 0) continue;
            const std::vector<std::uint32_t> d = m_.dflashDraft(sl.id, sl.next, sl.pos, opt_.dfl_p_min, opt_.dfl_n_min);
            sl.dfl_nd = 0.8f * sl.dfl_nd + 0.2f * static_cast<float>(d.size());
            const std::uint32_t n = std::min(lim_k, static_cast<std::uint32_t>(d.size()));
            std::copy(d.begin(), d.begin() + n, cy.ng_d[k].begin());
            cy.ng_n[k] = n;
            cy.dsrc[k] = 2;
            cy.n_dfl += 1;
            cy.dfl_max_n = std::max(cy.dfl_max_n, n);
        }
    }
}

void Engine::mtpDrafts(Cycle& cy) {
    // MTP drafts for the other slots
    for (std::size_t k = 0; k < cy.A; ++k)
        if (cy.ng_n[k] == 0) cy.act_mtp.push_back(cy.act[k]);
    cy.n_mtp = cy.act_mtp.size();
    cy.nd = cy.n_mtp > 0 ? pickDrafts(cy.act_mtp) : 0;
    {
        // keep the verify within vrows_ rows (and the snapshot sets without replay)
        std::uint32_t ng_tot = 0;
        for (std::size_t k = 0; k < cy.A; ++k) ng_tot += cy.ng_n[k];
        while (cy.nd > 0 && (static_cast<std::uint32_t>(cy.A) + ng_tot + static_cast<std::uint32_t>(cy.n_mtp) * cy.nd > vrows_ ||
                          ng_tot + static_cast<std::uint32_t>(cy.n_mtp) * cy.nd > cy.snap_cap))
            cy.nd -= 1;
    }
    cy.nd_max = cy.nd;
    if (cy.n_mtp > 0) {
        std::uint32_t ng_tot = 0;
        for (std::size_t k = 0; k < cy.A; ++k) ng_tot += cy.ng_n[k];
        const std::uint32_t used = static_cast<std::uint32_t>(cy.A) + ng_tot;
        const std::uint32_t rows_free = vrows_ > used ? vrows_ - used : 0;
        const std::uint32_t snap_free = cy.snap_cap > ng_tot ? cy.snap_cap - ng_tot : 0;
        if (opt_.slot_drafts == 2 && cy.nd > 0) {
            const std::size_t ia = std::min<std::size_t>(cy.A, gdn_max_seg);
            qwen35::CycleCost& cc = ccost_[ia];
            if (!ccost_primed_[ia] && cc.samples() == 0) {
                float ctx_k = 0;
                for (Slot* s : cy.act) ctx_k += static_cast<float>(s->pos) / 1024.0f;
                cc.prior(timing_[ia], static_cast<std::uint32_t>(cy.A), ctx_k / static_cast<float>(cy.A));
                ccost_primed_[ia] = true;
            }
            std::vector<qwen35::DraftAccept> eff;
            eff.reserve(cy.A);
            std::vector<DraftPlanSlot> ps(cy.A);
            for (std::size_t k = 0; k < cy.A; ++k) {
                Slot& s = *cy.act[k];
                ps[k].pos = s.pos;
                ps[k].d_prev = s.d_prev;
                if (cy.ng_n[k] > 0) {
                    ps[k].ng_rows = cy.ng_n[k] + 1;
                    ps[k].ng_e = s.ng.acc[qwen35::NgramPolicy::bucket(cy.ng_ml[k])].expected(cy.ng_n[k]);
                } else {
                    eff.push_back(s.sacc.effective(dpool_));
                    ps[k].acc = &eff.back();
                }
            }
            const std::uint32_t cap = !m_.fusedDecode() ? 1u : std::min({opt_.n_draft, qwen35::max_drafts, max_small_batch - 1});
            const std::vector<std::uint32_t> d = planSlotDrafts(ps, cc, cy.nd, cap, vrows_, snap_free);
            for (std::size_t i = 0; i < cy.n_mtp; ++i) cy.nd_m[i] = d[i];
        } else {
            splitDrafts(cy.act_mtp, cy.nd, std::min(rows_free, snap_free), std::span<std::uint32_t>(cy.nd_m.data(), cy.n_mtp));
        }
        for (std::size_t i = 0; i < cy.n_mtp; ++i) cy.act_mtp[i]->d_prev = cy.nd_m[i];
        cy.nd_max = 0;
        for (std::size_t i = 0; i < cy.n_mtp; ++i) cy.nd_max = std::max(cy.nd_max, cy.nd_m[i]);
    }
}

void Engine::layoutBatch(Cycle& cy) {
    cy.tc0 = Clock::now();
    cy.pa = std::min<std::size_t>(cy.A, gdn_max_seg);
    cy.p = prof_.get();
    if (cy.p && cy.p->events) m_.profileStart(0, cy.pa);
    // MTP steps r >= 1 run on the MTP slots with more than r drafts: MTP slots
    // sorted by draft count (descending) at the front of the batch
    {
        std::array<std::size_t, gdn_max_seg> ord{};
        for (std::size_t i = 0; i < cy.n_mtp; ++i) ord[i] = i;
        for (std::size_t i = 1; i < cy.n_mtp; ++i) {
            std::size_t j = i;
            while (j > 0 && cy.nd_m[ord[j - 1]] < cy.nd_m[ord[j]]) {
                std::swap(ord[j - 1], ord[j]);
                --j;
            }
        }
        for (std::size_t q = 0; q < cy.n_mtp; ++q) {
            Slot& s = *cy.act_mtp[ord[q]];
            cy.msegs[q] = MSeg{s.id, std::span<const std::uint32_t>(s.pend.data(), s.n_pend), s.pend_pos, s.pos};
        }
        for (std::size_t r = 0; r <= qwen35::max_drafts; ++r) {
            std::size_t c = 0;
            for (std::size_t i = 0; i < cy.n_mtp; ++i)
                if (cy.nd_m[i] > r) ++c;
            cy.n_step[r] = c;
        }
    }
    {
        // n-gram slots after the MTP slots in the MTP batch (their pend rows only)
        std::size_t ig = cy.n_mtp, im = 0;
        std::uint32_t row = 0, snap = 0;
        for (std::size_t k = 0; k < cy.A; ++k) {
            Slot& sl = *cy.act[k];
            const MSeg ms{sl.id, std::span<const std::uint32_t>(sl.pend.data(), sl.n_pend), sl.pend_pos, sl.pos};
            if (cy.ng_n[k] == 0) {
                cy.nd_of[k] = cy.nd_m[im];
                ++im;
            } else {
                cy.msegs[ig] = ms;
                ++ig;
                cy.nd_of[k] = cy.ng_n[k];
            }
            cy.vsegs[k] = VSeg{sl.id, sl.next, cy.nd_of[k], sl.pos,
                            cy.ng_n[k] > 0 ? std::span<const std::uint32_t>(cy.ng_d[k].data(), cy.ng_n[k])
                                        : std::span<const std::uint32_t>()};
            cy.row_of[k] = row;
            cy.snap_of[k] = snap;
            row += cy.nd_of[k] + 1;
            snap += cy.nd_of[k];
        }
    }
}

void Engine::draftSampling(Cycle& cy) {
    if (spec_on_ && cy.nd_max > 0) {
        // specsample: MTP slots of sampling requests draw their drafts from q by the target's race
        for (std::size_t k = 0; k < cy.A; ++k) {
            const Slot& sl = *cy.act[k];
            spec::DraftSample ds;
            ds.on = !sl.greedy && cy.ng_n[k] == 0;
            if (ds.on) {
                ds.top_k = sl.sm.top_k;
                ds.inv_t = sl.sm.inv_t;
                ds.top_p = static_cast<float>(sl.sm.top_p);
                ds.min_p = static_cast<float>(sl.sm.min_p);
                ds.seed = sl.sm.seed;
                ds.pos0 = sl.n_gen;
            }
            m_.setDraftSample(sl.id, ds);
        }
    }
}

void Engine::enqueueVerify(Cycle& cy) {
    if (cy.nd_max > 0) {
        m_.setDraftCutoff(opt_.p_min, std::min(opt_.n_min, cy.nd_max));
        struct Reset {
            ServerModel& m;
            ~Reset() { m.setDraftCutoff(0, 0); }
        } reset{m_};
        m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data(), cy.A), 0, true);
        for (std::uint32_t r = 1; r < cy.nd_max; ++r)
            m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data(), cy.n_step[r]), r, true);
    } else if (opt_.use_mtp) {
        // no MTP drafts this cycle: still feed the pend rows to the MTP block so
        // its KV cache stays complete for later drafting cycles
        m_.mtpBatchStepEx(std::span<const MSeg>(cy.msegs.data(), cy.A), 0, false);
    }
    if (cy.p && cy.p->events) m_.profileStart(1, cy.pa);
    cy.rows = m_.verifyBatchEnqueue(std::span<const VSeg>(cy.vsegs.data(), cy.A));
    if (cy.p) m_.profileStop();
}

void Engine::sampleRows(Cycle& cy) {
    bool any_sampling = false, any_relax = false;
    cy.rk = opt_.relax.topk();
    for (std::size_t k = 0; k < cy.A; ++k) {
        Slot& sl = *cy.act[k];
        if (sl.greedy) {
            if (relaxes(cy, k) && cy.nd_of[k] > 0) {
                any_relax = true;
                sb_.launchTopk(cy.row_of[k], cy.nd_of[k], cy.rk, 1.0f, sb_.relaxDev(), max_rows, cy.row_of[k]);
            }
            continue;
        }
        any_sampling = true;
        sb_.launchTopk(cy.row_of[k], cy.nd_of[k] + 1, k_small, sl.sm.inv_t, sb_.sampDev(), max_rows, cy.row_of[k]);
    }
    const TimePoint t_enq = Clock::now();
    m_.readCtl(std::span<std::int32_t>(sb_.ctlHost().data(), slots_.size() * qwen35::ctl_words));
    if (cy.p) {
        cy.p->t_ready = Clock::now();
        cy.p->enq_ms[cy.pa] += std::chrono::duration<double, std::milli>(t_enq - cy.tc0).count();
    }
    if (any_sampling) ops_.download(sb_.sampHost().data(), sb_.sampDev(), SampleBuffers::sampBytes());
    if (any_relax) ops_.download(sb_.relaxHost().data(), sb_.relaxDev(), sb_.relaxHost().size());
}

void Engine::trainCycle(Cycle& cy) {
    // relaxacc: the kept tokens depend on which drafts were proposed, so the draft-count policy
    // must not see wall-clock times (run-to-run jitter would change the output). It sees a
    // deterministic cycle cost instead: a base per slot, + ngram_slope per extra verify row (the
    // per-device verify-row slope, 0.12 on the 8060S), + 3% per sequential MTP draft step.
    const auto synthMs = [&](std::uint32_t steps) {
        float rows = 0;
        for (std::size_t k = 0; k < cy.A; ++k) rows += static_cast<float>(cy.nd_of[k] + 1);
        const float a = static_cast<float>(cy.A);
        return spec::rules::synthCycleMs(a, rows, opt_.ngram_slope, steps);
    };
    const float cycle_ms = opt_.relax.on ? synthMs(cy.n_ng == cy.A ? 0 : cy.nd_max) : static_cast<float>(msSince(cy.tc0));
    if (opt_.slot_drafts == 2 && opt_.use_mtp) {
        // every cycle (uniform, per-slot, with n-gram slots) trains the cost model of A slots;
        // a per-slot cycle reaches timing_ as the time the uniform split would have taken
        qwen35::CycleCost& cc = ccost_[std::min<std::size_t>(cy.A, gdn_max_seg)];
        qwen35::CycleFeat x, xu;
        bool uni = cy.n_ng == 0;
        for (std::size_t k = 0; k < cy.A; ++k) {
            const float ctx_k = static_cast<float>(cy.act[k]->pos) / 1024.0f;
            x.rows += static_cast<float>(cy.nd_of[k] + 1);
            x.row_ctx += static_cast<float>(cy.nd_of[k] + 1) * ctx_k;
            xu.rows += static_cast<float>(cy.nd + 1);
            xu.row_ctx += static_cast<float>(cy.nd + 1) * ctx_k;
            uni = uni && cy.nd_of[k] == cy.nd;
        }
        x.steps = static_cast<float>(cy.nd_max);
        xu.steps = static_cast<float>(cy.nd);
        if (cy.nd > 0 && cy.n_ng == 0) {
            if (uni) timing_[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.nd, cycle_ms);
            else if (cc.samples() >= 16)
                timing_[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.nd, cycle_ms * cc.predict(xu) / cc.predict(x));
        }
        cc.observe(x, cycle_ms);
    } else if (cy.nd > 0 && cy.n_ng == 0) {
        // swaps keep the rows of the common count (an extra step at most)
        std::uint32_t tot = 0;
        for (std::size_t i = 0; i < cy.n_mtp; ++i) tot += cy.nd_m[i];
        if (tot == cy.nd * static_cast<std::uint32_t>(cy.n_mtp))
            timing_[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.nd, cycle_ms);
    }
    if (cy.n_ng == cy.A) {
        // n-gram cycle times (every slot drafted from its history): per slot
        std::uint32_t ng_max_n = 0;
        for (std::size_t k = 0; k < cy.A; ++k) ng_max_n = std::max(ng_max_n, cy.ng_n[k]);
        const float ms = opt_.relax.on ? cycle_ms : static_cast<float>(msSince(cy.tc0));
        for (Slot* sl : cy.act) sl->ng.timing.update(ng_max_n, ms);
    }
    if (opt_.use_dflash && cy.n_ng == 0) {
        // DFlash2 cycle times (drafting included; 0 drafts: a plain verify)
        dtiming_[std::min<std::size_t>(cy.A, gdn_max_seg)].update(cy.dfl_max_n, static_cast<float>(msSince(cy.td0)));
    }
}

void Engine::cycleStats(Cycle& cy) {
    if (cy.p) {
        if (cy.p->events) m_.profileFlush(cy.pa);
        cy.p->cycles[cy.pa] += 1;
        cy.p->rows[cy.pa] += cy.rows;
        cy.p->drafts[cy.pa] += cy.rows - cy.A;
        cy.p->wall_ms[cy.pa] += msSince(cy.t_cycle0);
        cy.p->host_ms[cy.pa] += msSince(cy.t_cycle0) - msSince(cy.tc0);
        if (cy.p->last_end) cy.p->gap_ms[cy.pa] += std::chrono::duration<double, std::milli>(cy.t_cycle0 - *cy.p->last_end).count();
    }
    if (opt_.trace_nd) logI("trace | cycle {} | slots {} | nd {} | n-gram {} | dflash {} | {:.2f} ms", n_cycles_, cy.A, cy.nd, cy.n_ng, cy.n_dfl, msSince(cy.tc0));
    n_cycles_ += 1;
    policy_cycles_ += 1;
    bst_.cycles += 1;
    bst_.rows += cy.rows;
    bst_.slots += cy.A;
    bst_.drafts += cy.nd;
    for (std::size_t k = 0; k < cy.A; ++k) bst_.lit_drafts += cy.ng_n[k];
}

void Engine::acceptSlot(Cycle& cy, std::size_t k, const sampling::Acceptor& acceptor) {
    Slot& sl = *cy.act[k];
    const std::int32_t* ctl = sb_.ctlHost().data() + static_cast<std::size_t>(sl.id) * qwen35::ctl_words;
    const std::uint32_t row0 = cy.row_of[k];
    const std::uint32_t nd_k = cy.nd_of[k];
    const bool is_ng = cy.ng_n[k] > 0;  // literal drafts (n-gram or DFlash2)
    // MTP: drafts (and the device p-min cutoff) from the control area; n-gram: literal
    const std::uint32_t nd_dev =
        is_ng ? nd_k : (nd_k > 0 ? std::min(nd_k, static_cast<std::uint32_t>(std::max(0, ctl[qwen35::ctl_nd]))) : 0);
    std::array<std::uint32_t, qwen35::max_ng_drafts> dr{};
    for (std::uint32_t r = 0; r < nd_k; ++r)
        dr[r] = is_ng ? cy.ng_d[k][r] : static_cast<std::uint32_t>(ctl[qwen35::ctl_drafts + r]);
    if (cy.dsrc[k] == 1) sl.ng_cycles += 1;
    sl.cycles += 1;
    sl.drafted += nd_dev;
    // draft acceptance (src/sampling/acceptor.h): greedy / relaxacc greedy, or the sampler's draw at every
    // output index (inverse CDF / specsample race / relaxacc typical)
    const sampling::Drafts drafts{std::span<const std::uint32_t>(dr.data(), dr.size()), nd_dev, row0};
    const sampling::RepeatGuard repeats = [&](std::uint32_t a) { return relaxRepeats(sl, dr, a); };
    const sampling::Accepted ac =
        sl.greedy ? acceptor.greedy(ctl + qwen35::ctl_rows, drafts, relaxes(cy, k), sampling::RelaxRows{sb_.relaxHost().data(), max_rows, cy.rk},
                                    repeats, sl.relaxed, sl.relax_gap)
                  : acceptor.sample(sb_, sl.sm, drafts, relaxes(cy, k), sl.n_gen, repeats, sl.relaxed);
    std::uint32_t acc = ac.acc;
    std::uint32_t chosen = ac.chosen;
    sl.accepted += acc;
    if (cy.dsrc[k] == 0 && opt_.use_mtp) {
        // n-gram cycles stay out of the MTP acceptance models
        sl.acc_ema = spec::rules::accEma(sl.acc_ema, acc);
        sl.dacc.update(nd_dev, acc);
        if (opt_.slot_drafts == 2 && nd_dev > 0) {
            sl.sacc.observe(nd_dev, acc);
            dpool_.update(nd_dev, acc);
        }
    } else if (cy.dsrc[k] == 2) {
        sl.dfl_acc.update(nd_dev, acc);
        sl.dfl_cycles += 1;
        bst_.dfl_cycles += 1;
        bst_.dfl_drafted += nd_dev;
        bst_.dfl_accepted += acc;
    }
    // an accepted end-of-turn draft ends the reply: the trunk keeps only the
    // rows before it and the stop token becomes `chosen`
    if (!sl.ignore_eos) {
        for (std::uint32_t r = 0; r < acc; ++r) {
            if (dr[r] == ct_.im_end || dr[r] == ct_.endoftext) {
                chosen = dr[r];
                acc = r;
                break;
            }
        }
    }
    // same for max_tokens: the last emitted token is not fed to the trunk
    {
        const std::uint32_t budget = sl.max_tokens > sl.n_gen ? sl.max_tokens - sl.n_gen : 0;
        if (budget >= 1 && acc + 1 > budget) {
            chosen = dr[budget - 1];
            acc = budget - 1;
        }
    }
    // the verify ran nd + 1 rows of this slot: keep acc + 1 of them
    if (m_.gdnReplay()) m_.setPending(sl.id, acc + 1);
    else if (acc < nd_k) m_.restoreSeqSnapshot(sl.id, cy.snap_of[k] + acc);
    // DFlash2 context: the kept rows' features (whatever drafted them; rejected rows never enter)
    dflIngest(sl, cy.rows, row0, acc + 1, sl.pos);
    // trunk now holds next, dr[0..acc) at pos .. pos + acc
    sl.cache_tokens.push_back(sl.next);
    sl.cache_tokens.insert(sl.cache_tokens.end(), dr.begin(), dr.begin() + acc);
    const std::uint32_t n_before = sl.n_gen;
    for (std::uint32_t r = 0; r < acc; ++r)
        if (!sl.done) sl.done = emit(sl, dr[r]);
    if (!sl.done) sl.done = emit(sl, chosen);
    bst_.tokens += sl.n_gen - n_before;
    if (opt_.use_mtp)
        ops_.copyAsync(m_.seqHid(sl.id), m_.hn() + static_cast<std::uint64_t>(row0) * cy.E * 4, (acc + 1) * cy.E * 4,
                       m_.stream());
    for (std::uint32_t r = 0; r < acc; ++r) sl.pend[r] = dr[r];
    sl.pend[acc] = chosen;
    sl.n_pend = acc + 1;
    sl.pend_pos = sl.pos + 1;
    sl.next = chosen;
    sl.pos += acc + 1;
    sl.last_acc = acc;
    sl.last_row = row0 + acc;
    sl.logits_ok = true;
    flushOut(sl);
    if (sl.n_gen >= sl.last_log_n + 100 || msSince(sl.last_log_t) >= 5000) {
        const double el = msSince(sl.td0);
        if (opt_.use_mtp || opt_.use_dflash) {
            logI("req {} | slot {} | gen: n_gen {}, tg {:.2f} t/s, draft acceptance {:.1f}% ({:.2f} tok/cycle)",
                 sl.reqId(), sl.id, sl.n_gen, static_cast<double>(sl.n_gen) * 1000.0 / el,
                 sl.drafted > 0 ? 100.0 * sl.accepted / sl.drafted : 0.0,
                 static_cast<double>(sl.accepted + sl.cycles) / sl.cycles);
        } else {
            logI("req {} | slot {} | gen: n_gen {}, tg {:.2f} t/s", sl.reqId(), sl.id, sl.n_gen,
                 static_cast<double>(sl.n_gen) * 1000.0 / el);
        }
        sl.last_log_n = sl.n_gen;
        sl.last_log_t = Clock::now();
        // WDDM-LOG: a sharp drop against the request's early speed; the spill is named as the likely
        // cause only while one is seen (vram_watch_ state from the last status-line sample)
        if (const auto ev = sl.dec_watch.report(sl.n_gen, el, sl.cycles, vram_warn_ && vram_watch_.spilling(), prefill_tok_cum_)) {
            if (ev->warn) logW("req {} | slot {} | {}", sl.reqId(), sl.id, ev->text);
            else logI("req {} | slot {} | {}", sl.reqId(), sl.id, ev->text);
        }
    }
}

void Engine::finishCycle(Cycle& cy) {
    if (cy.p) {
        std::uint64_t tk = 0;
        for (Slot* sl : cy.act) tk += sl->last_acc + 1;
        cy.p->tokens[cy.pa] += tk;
    }
    // finish after every slot of the cycle is processed (their logits rows are
    // still intact for the gen-end checkpoints)
    bool any_done = false;
    for (Slot* sl : cy.act) {
        if (sl->done || sl->st.gone) {
            any_done = true;
            finishJob(*sl);
        }
    }
    if (cy.p) {
        if (cy.p->t_ready) cy.p->post_ms[cy.pa] += msSince(*cy.p->t_ready);
        if (any_done) cy.p->last_end.reset();
        else cy.p->last_end = Clock::now();
    }
}

void Engine::splitDrafts(std::span<Slot* const> act, std::uint32_t nd, std::uint32_t rows, std::span<std::uint32_t> nd_m) {
    for (auto& x : nd_m) x = nd;
    if (opt_.slot_drafts != 1 || !opt_.mtp_auto || act.size() < 2 || nd == 0) return;
    const std::uint32_t cap = std::min(opt_.n_draft, qwen35::max_drafts);
    std::array<const qwen35::DraftAccept*, gdn_max_seg> acc{};
    for (std::size_t k = 0; k < act.size(); ++k) acc[k] = &act[k]->dacc;
    splitDraftsByGain(std::span<const qwen35::DraftAccept* const>(acc.data(), act.size()), nd, rows, cap, nd_m);
}

std::uint32_t Engine::draftCap(std::uint32_t A) {
    return slotDraftCap(opt_.batch_drafts[std::min(A, gdn_max_seg)], opt_.n_draft, m_.fusedDecode(), m_.gdnReplay(), A, vrows_);
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
// item G: host tiers

bool Engine::tryRestore(Slot& sl, Job& job) {
    if (!tb_.tier()) return false;
    if (!job.params.cache_prompt || opt_.no_prefix_cache) return false;
    const std::span<const std::uint32_t> toks = job.tokens;
    const std::uint32_t nd_max = draftRoom();
    if (toks.size() + nd_max + 2 >= ctxForNew() || toks.size() + nd_max + 2 >= static_cast<std::uint64_t>(pool_.pages()) * kv_page)
        return false;
    const auto mt = tb_.restoreMatch(sl, toks);
    if (!mt) return false;
    try {
        tier::Entry* x = mt->e;
        const TimePoint t0 = Clock::now();
        // the slot's own VRAM content goes (it is in the tier if it was worth keeping)
        freePages(sl);
        sl.job = &job;
        sl.phase = Phase::restore;
        if (!ensurePages(sl, x->n_tok)) {
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
    s.drop_cache = [this, &sl] { dropCache(sl); };
    s.drafter_reset = [this, &sl](std::uint32_t n_tok) { dflReset(sl, n_tok); };
    s.share = [this, &sl](Ckpt& c, std::uint64_t entry_id) {
        createSpe(sl, sl.cache_tokens, c.pos, c.kind, std::nullopt, &c, entry_id);
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

// ---------------------------------------------------------------------------

void Engine::releaseSlot(Slot& sl) {
    if (sl.restore != nullptr && sl.phase != Phase::restore) tb_.finishTail(sl.id, sl, sl, true);
    Job* job = sl.job;
    if (!job) return;
    visDone(sl, *job);
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

std::size_t Engine::pagesFor(std::uint64_t n_tok) const {
    return PagePool::pagesFor(n_tok + draftRoom() + 2);
}

std::size_t Engine::poolAvailable() const {
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

void Engine::reapDisconnected() {
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
        dropCache(sl);
        releaseSlot(sl);
    }
}

void Engine::writeError(Job& job, int status, const std::string& msg) {
    logW("req {} | {} {}", job.id, status, msg);
    const std::string r = errorResponse(status, msg);
    if (job.w) {
        job.w->write(r);
        job.w->flush();
    }
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
    if (stat_floor_periods_ > 0)
        logI("batch | decode floor: {} prefill forwards while decoding, {} decode cycles held prefill back", stat_floor_periods_,
             stat_floor_waits_);
    stat_floor_periods_ = stat_floor_waits_ = 0;
}

void Engine::submitAndWait(Job& job) {
    whirl::UniqueLock lk(q_mutex_);
    if (stop_) {
        // FR-SRV P3-1: shutting down (the loop may be flushing the tier, or gone): nobody would run
        // this job; answered here instead of waiting forever on a finished engine
        n_building_.fetch_sub(1);
        ++wake_seq_;
        q_cond_.notify_all();
        lk.unlock();
        if (job.w) {
            job.w->write(errorResponse(503, "server is shutting down"));
            job.w->flush();
        }
        return;
    }
    const std::size_t free = slots_.size() - std::min<std::size_t>(slots_.size(), n_active_);
    const std::size_t waiting = queue_.size();
    queue_.push_back(&job);
    n_building_.fetch_sub(1);
    ++wake_seq_;
    q_cond_.notify_all();
    lk.unlock();
    if (waiting >= free)
        logI("req {} | queued ({} request(s) waiting ahead, all {} slots busy)", job.id, waiting - free, slots_.size());
    lk.lock();
    q_cond_.wait(lk, [&] { return job.done; });
}

void Engine::endBuilding() {
    // a connection that did not queue a request (bad request, probe): a burst
    // gather waiting for it ends now
    whirl::MutexLock lk(q_mutex_);
    n_building_.fetch_sub(1);
    ++wake_seq_;
    q_cond_.notify_all();
}

void Engine::abortQueued(whirl::UniqueLock& lk) {
    // shutting down: queued requests are answered 503
    for (Job* job : queue_) {
        const std::string r = errorResponse(503, "server is shutting down");
        if (job->w) {
            job->w->write(r);
            job->w->flush();
        }
        job->done = true;
    }
    queue_.clear();
    q_cond_.notify_all();
    (void)lk;
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
    if (!floor_.on()) return dec_toks;
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
            if (floor_.on()) noteFloor(tl2, tl3, pf_rows, floor_pf);
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
        visIdle();
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
        if (prof_) prof_->print(m_);
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
        if (visWakeNeeded()) {
            WH_UNLOCK_HELD(lk, q_mutex_);
            visIdle();
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

// one prefill chunk (oldest request first), with the decode floor's budget
void Engine::runPrefill(Slot* pf, bool& idle_wait, std::size_t& pf_rows, bool& floor_pf) {
    // decode floor: the slots it protects this iteration
    const std::vector<DecodeFloor::SlotTok> dec_toks = floorProtected();
    if (floor_.on() && dec_toks.empty()) floor_.idle();
    auto floorNow = [&] { return std::chrono::duration<double, std::milli>(Clock::now() - floor_epoch_).count(); };
    if (pf) {
        // a burst in flight: hold the first chunk briefly so the other
        // requests join its segmented forward
        const bool gather = opt_.seg_prefill && pf->pf_off == 0 && opt_.gather_ms > 0 && n_building_.load() > 0 &&
                            msSince(pf->tp0) < opt_.gather_ms;
        if (gather) {
            idle_wait = true;
        } else if (!dec_toks.empty() && !floor_.prefillAllowed(floorNow(), dec_toks)) {
            // decoding slots are below the floor in this period: a decode cycle alone
            ++stat_floor_waits_;
        } else {
            std::size_t budget = static_cast<std::size_t>(-1);
            if (!dec_toks.empty()) {
                floor_.startPeriod(floorNow(), dec_toks);
                const DecodeFloor::Plan pl = floor_.plan(chunkLen(*pf), m_.maxBatch());
                budget = pl.rows;
                floor_pf = true;
                ++stat_floor_periods_;
                if (msSince(floor_log_t_) >= 10000) {
                    floor_log_t_ = Clock::now();
                    if (pl.cycles > 0)
                        logI("decode floor | {:.0f} tok/s per decoding slot ({} decoding): prefill forward <= {} rows, ~{} decode "
                             "cycles between forwards{} (cycle {:.1f} ms, prefill {:.0f} rows/s)",
                             floor_.minTps(), dec_toks.size(), pl.rows, pl.cycles,
                             pl.reachable ? "" : " (floor out of reach: prefill keeps 20% of the time)", floor_.cycleMs(),
                             floor_.rowsPerMs() * 1000.0);
                    else
                        logI("decode floor | {:.0f} tok/s per decoding slot ({} decoding): prefill forward <= {} rows "
                             "(not measured yet)",
                             floor_.minTps(), dec_toks.size(), pl.rows);
                }
            }
            pf_rows = prefillGroup(*pf, budget);
            if (prof_) prof_->last_end.reset();
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
    floor_.noteCycle(cyc_ms, pf_rows > 0, now_dec);
    if (floor_pf && pf_rows > 0) floor_.notePrefill(pf_rows, it_ms - std::min(floor_.cycleMs(), 0.5 * it_ms));
}

}  // namespace whirl::server
