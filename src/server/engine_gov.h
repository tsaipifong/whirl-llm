// Server engine, module server.governor (arch025-engine S8; progress/arch025-engine.DESIGN.md v3 2.3 L3, 2.5, 4 S8):
// the v0.2.2.2 memory governor's state and its glue.
// - class GovState: the former Engine members gov_ .. gov_json_ (engine.h:284-301 at 6871c14), declared here verbatim,
//   in that order, with those initial values, plus the new last member gov_mu_. Engine holds it as gst_ where gov_ was
//   (DESIGN v3 2.5). The members are private (DESIGN v3 1.5 A1); the only class that reaches them is GovernorGlue, the
//   same module's logic (friend), which binds a reference to each.
// - F14 (DESIGN v3 4 S8, v2 2.3): gov_json_ is guarded by gov_mu_ (before: vram_mu_, the vram watch's lock). No lock was
//   ever held together with the other (engine_gov.cpp governorJson / govPublish, engine_vram.cpp setVramProbe / vramNote
//   and engine.cpp health each take one lock and read one group), so the split changes no lock order.
// - class GovernorGlue: the former Engine member functions setGovernor, governorJson, govPublish, govAllIdle,
//   govFreeable, govTick, govDrain, govFreeCache, govRestoreCache, govResize, govApply (engine_gov.cpp, bodies verbatim;
//   only governorJson / govPublish lock gov_mu_ instead of vram_mu_) and the header-inline ctxForNew (declarations moved
//   verbatim from engine.h:152-159 / 229-239), plus the small accessors the other modules use instead of the members
//   (govActive, govReduced, govHolding, govSetWarn, govCtxPublished, govPublishCtx; each body is the former statement or
//   expression). It holds references only (private, under their own names, so the bodies read as before): the core.shared
//   members ck_ ctx_ m_ ops_ opt_ owner_ pool_ ps_ slots_ tb_ vram_probe_ (DESIGN v3 3.1 refs), the governor's placed
//   state that Engine keeps where it was (vram_others_, ctx_pub_), the slot_cache holder sc_ (a lower module) and gst_'s
//   members. Engine holds it as gv_, its last member (destroyed first; trivial destructor).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "server/engine_pages.h"   // SlotPages (server.slot_cache); Slot, Spe, CkptStore, PagePool, PrefixStore,
                                   // TierBridge, ServerModel, EngineOptions, device::Backend through its headers
#include "server/engine_thread.h"  // EngineThread
#include "whirl/mem_governor.h"
#include "whirl/thread_annotations.h"
#include "whirl/wddm_watch.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace whirl::server {

class GovState {
    friend class GovernorGlue;  // server.governor's own logic (DESIGN v3 1.5 A1: no other class reaches these)

    // v0.2.2.2 memory governor (main thread; gov_json_ under gov_mu_)
    std::optional<memgov::Governor> gov_;
    std::function<std::uint64_t(std::uint64_t)> gov_margin_;
    memgov::Plan gov_plan_;
    bool gov_pending_ = false;
    bool gov_hold_ = false;       // new requests queue (a smaller pool waits for the running ones)
    bool spe_parked_ = false;     // shared prefix checkpoint buffers freed
    std::uint64_t gov_parked_ = 0;  // checkpoint VRAM freed (to restore)
    std::uint64_t gov_ram_mb_ = 0;
    std::uint64_t gov_vram_total_ = 0;
    bool gov_grow_denied_logged_ = false;
    std::uint64_t gov_restored_ = 0;  // checkpoint bytes the last govRestoreCache allocated
    std::uint64_t gov_resize_before_ = 0;  // local usage read right before the last re-allocation
    std::int64_t gov_resize_delta_ = 0;    // bytes it added (> 0) or freed (< 0)
    std::uint32_t ctx_new_ = 0xffffffffu;  // context per request for new requests (governor); running requests keep ctx_
    std::string gov_warn_;        // the spill warning, logged after the governor's actions
    std::string gov_last_;
    std::string gov_json_ WH_GUARDED_BY(gov_mu_);
    whirl::Mutex gov_mu_;  // gov_json_ for governorJson() (HTTP threads); F14: its own lock, not the vram watch's
};

class GovernorGlue {
public:
    GovernorGlue(CkptStore& ck, const std::uint32_t& ctx, ServerModel& m, device::Backend& ops, const EngineOptions& opt,
                 const EngineThread& owner, PagePool& pool, PrefixStore& ps, std::vector<Slot>& slots, TierBridge& tb,
                 const std::function<wddmwatch::Sample()>& vram_probe,
                 const std::function<std::optional<wddmwatch::Others>()>& vram_others, std::atomic<std::uint32_t>& ctx_pub,
                 SlotPages& sc, GovState& g)
        : ck_(ck), ctx_(ctx), m_(m), ops_(ops), opt_(opt), owner_(owner), pool_(pool), ps_(ps), slots_(slots), tb_(tb),
          vram_probe_(vram_probe), vram_others_(vram_others), ctx_pub_(ctx_pub), sc_(sc), gov_(g.gov_), gov_margin_(g.gov_margin_),
          gov_plan_(g.gov_plan_), gov_pending_(g.gov_pending_), gov_hold_(g.gov_hold_), spe_parked_(g.spe_parked_),
          gov_parked_(g.gov_parked_), gov_ram_mb_(g.gov_ram_mb_), gov_vram_total_(g.gov_vram_total_),
          gov_grow_denied_logged_(g.gov_grow_denied_logged_), gov_restored_(g.gov_restored_),
          gov_resize_before_(g.gov_resize_before_), gov_resize_delta_(g.gov_resize_delta_), ctx_new_(g.ctx_new_),
          gov_warn_(g.gov_warn_), gov_last_(g.gov_last_), gov_json_(g.gov_json_), gov_mu_(g.gov_mu_) {}
    GovernorGlue(const GovernorGlue&) = delete;
    GovernorGlue& operator=(const GovernorGlue&) = delete;

    // v0.2.2.2 memory governor (after setVramProbe, before runLoop): acts on the samples of the vram probe when the WDDM
    // budget drops while the server runs (whirl/mem_governor.h). margin: what must stay free under a given budget
    // (poolbudget::defaultMargin); ram_tier_mb: the RAM tier in use (0: none), for its log line.
    // vram_total: the GPU's dedicated VRAM (0: unknown); a grow is capped so WHIRL + what other programs use now (vram others
    // probe) + the margin fit it
    void setGovernor(const memgov::Config& c, std::function<std::uint64_t(std::uint64_t)> margin, std::uint64_t ram_tier_mb,
                     std::uint64_t vram_total = 0);
    std::string governorJson();
    // v0.2.2.2 memory governor
    void govTick(const wddmwatch::Sample& s);   // after each sample: decide (and log the decisions)
    void govApply();                            // at a safe point of runLoop (no forward in flight): act
    bool govAllIdle() const;
    std::uint64_t govFreeable() const;
    void govFreeCache(bool park);
    void govRestoreCache();
    bool govResize(std::uint32_t tokens, bool shrink);
    void govDrain();
    void govPublish();
    std::uint32_t ctxForNew() const { return std::min(ctx_, ctx_new_); }
    // for the other modules (arch025-engine S8; each body is the former statement / expression at its call site)
    bool govActive() const { return gov_.has_value(); }                    // idleWait, admitQueued, vramNote: `gov_`
    bool govReduced() const { return gov_ && gov_->reduced(); }            // vramNote
    bool govHolding() const { return gov_hold_; }                          // admitQueued
    void govSetWarn(const std::string& t) { gov_warn_ = t; }               // vramNote
    std::uint32_t govCtxPublished() const { return ctx_pub_.load(std::memory_order_relaxed); }  // Engine::ctxPublished
    void govPublishCtx(std::uint32_t c) { ctx_pub_.store(c, std::memory_order_relaxed); }       // Engine::setCtx

private:
    CkptStore& ck_;
    const std::uint32_t& ctx_;
    ServerModel& m_;
    device::Backend& ops_;
    const EngineOptions& opt_;
    const EngineThread& owner_;
    PagePool& pool_;
    PrefixStore& ps_;
    std::vector<Slot>& slots_;
    TierBridge& tb_;
    const std::function<wddmwatch::Sample()>& vram_probe_;
    const std::function<std::optional<wddmwatch::Others>()>& vram_others_;
    std::atomic<std::uint32_t>& ctx_pub_;
    SlotPages& sc_;
    std::optional<memgov::Governor>& gov_;
    std::function<std::uint64_t(std::uint64_t)>& gov_margin_;
    memgov::Plan& gov_plan_;
    bool& gov_pending_;
    bool& gov_hold_;
    bool& spe_parked_;
    std::uint64_t& gov_parked_;
    std::uint64_t& gov_ram_mb_;
    std::uint64_t& gov_vram_total_;
    bool& gov_grow_denied_logged_;
    std::uint64_t& gov_restored_;
    std::uint64_t& gov_resize_before_;
    std::int64_t& gov_resize_delta_;
    std::uint32_t& ctx_new_;
    std::string& gov_warn_;
    std::string& gov_last_;
    std::string& gov_json_ WH_GUARDED_BY(gov_mu_);
    whirl::Mutex& gov_mu_;
};

}  // namespace whirl::server
