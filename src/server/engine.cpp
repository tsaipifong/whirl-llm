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
    // v0252-hostwait (PLAN V1-61): the decode cycle's wait / submission mode (cycle_sync 0: nothing allocated or called)
    cyc_.initSync();
}

void Engine::initPool() {
    sc_.initPool();
}

// the memory governor (arch025-engine S8): GovernorGlue gv_ (module server.governor)
void Engine::setGovernor(const memgov::Config& c, std::function<std::uint64_t(std::uint64_t)> margin, std::uint64_t ram_tier_mb,
                         std::uint64_t vram_total) {
    gv_.setGovernor(c, std::move(margin), ram_tier_mb, vram_total);
}

std::string Engine::governorJson() {
    return gv_.governorJson();
}

// the video memory watch (arch025-engine S9): VramWatch vw_ (module server.vram_watch)
void Engine::setVramProbe(std::function<wddmwatch::Sample()> probe, bool warn, std::optional<wddmwatch::Sample> base,
                          wddmwatch::Config cfg, bool startup_warned, std::function<std::optional<wddmwatch::Others>()> others) {
    vw_.setVramProbe(std::move(probe), warn, std::move(base), std::move(cfg), startup_warned, std::move(others));
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
    jq_.jqHealth(h);
    vw_.vramHealth(h);
    return h;
}

Engine::Stats Engine::stats() const {
    Stats s;
    s.spe_new = sc_.cacheCounters().speNew();
    s.spe_hits = sc_.cacheCounters().speHits();
    s.spe_tok = sc_.cacheCounters().speTok();
    s.spe_evict = sc_.cacheCounters().speEvict();
    s.unshare = sc_.cacheCounters().unshares();
    s.evictions = sc_.cacheCounters().evictions();
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

// ---------------------------------------------------------------------------

}  // namespace whirl::server
