// Server engine, module server.vram_watch (arch025-engine S9; progress/arch025-engine.DESIGN.md v3 2.3 L4, 2.5, 4 S9):
// the WDDM video memory watch's state and its logic (WDDM-LOG, issues #5 / #6).
// - class VramState: the former Engine members vram_watch_ .. vram_last_t_ (engine.h:265-274 at 6a9f6be, without
//   prefill_tok_cum_ / crlf_norm_, which stood between them and now follow vst_ in Engine; DESIGN v3 2.5 table), declared
//   here verbatim, in that order, with those initial values. Engine holds it as vst_ where vram_watch_ was. The members
//   are private (DESIGN v3 1.5 A1); the only class that reaches them is VramWatch, the same module's logic (friend),
//   which binds a reference to each.
// - class VramWatch: the former Engine member functions setVramProbe, vramNote, vramIdleTick (engine_vram.cpp, bodies
//   verbatim; F15: vramNote keeps its order in one function - sample, vram_last_t_, gv_.govTick, the spill warning
//   (gv_.govSetWarn while the governor is on, else logW), the locked publish, gv_.govReduced for the idle cadence), plus
//   vramHealth (engine.cpp health's vram_mu_ lock and its five reads, verbatim) and the small accessors the other modules
//   use instead of the members (vramSpilling, vramLastSample; each body is the former expression at its call site). It
//   holds references only (private, under their own names, so the bodies read as before): the core.shared members
//   owner_ vram_probe_ (DESIGN v3 3.1 refs), the governor's placed state vram_others_ (written only by setVramProbe, the
//   startup installer, DESIGN v3 2.6), the governor's holder gv_ (a lower module) and vst_'s members. Engine holds it as
//   vw_, its last member (destroyed first; trivial destructor).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "server/engine_api.h"     // EngineApi::Health, Clock, TimePoint
#include "server/engine_gov.h"     // GovernorGlue (server.governor)
#include "server/engine_thread.h"  // EngineThread
#include "whirl/thread_annotations.h"
#include "whirl/wddm_watch.h"

#include <functional>
#include <optional>
#include <string>

namespace whirl::server {

class VramState {
    friend class VramWatch;  // server.vram_watch's own logic (DESIGN v3 1.5 A1: no other class reaches these)

    wddmwatch::Watch vram_watch_;
    TimePoint vram_idle_next_{};    // next idle sample (once a minute while no request runs)
    double vram_idle_s_ = 60.0;
    bool vram_warn_ = false;
    whirl::Mutex vram_mu_;  // vram_last_ / vram_spill_ for health() (HTTP threads)
    wddmwatch::Sample vram_last_ WH_GUARDED_BY(vram_mu_);
    bool vram_spill_ WH_GUARDED_BY(vram_mu_) = false;
    TimePoint vram_last_t_{};  // last probe sample
};

class VramWatch {
public:
    VramWatch(const EngineThread& owner, std::function<wddmwatch::Sample()>& vram_probe,
              std::function<std::optional<wddmwatch::Others>()>& vram_others, GovernorGlue& gv, VramState& v)
        : owner_(owner), vram_probe_(vram_probe), vram_others_(vram_others), gv_(gv), vram_watch_(v.vram_watch_),
          vram_idle_next_(v.vram_idle_next_), vram_idle_s_(v.vram_idle_s_), vram_warn_(v.vram_warn_), vram_mu_(v.vram_mu_),
          vram_last_(v.vram_last_), vram_spill_(v.vram_spill_), vram_last_t_(v.vram_last_t_) {}
    VramWatch(const VramWatch&) = delete;
    VramWatch& operator=(const VramWatch&) = delete;

    // WDDM-LOG (issues #5 / #6): before runLoop, after everything sized at startup is allocated. The
    // probe (hip::wddmMemInfo on Windows) is sampled for each "batch |" status line, at each request
    // end and once a minute while idle, never per decode cycle. base: the baseline for the non-local
    // growth (the server passes the read the KV pool was sized from, so a spill already there at startup
    // is not hidden); nullopt = the first sample. warn = false (UMA iGPU, where "shared" is the same RAM)
    // logs the numbers without the warning. startup_warned: the startup line already printed the spill
    // warning (the watch counts it as its first). others: read only when a spill warning is printed.
    void setVramProbe(std::function<wddmwatch::Sample()> probe, bool warn, std::optional<wddmwatch::Sample> base = std::nullopt,
                      wddmwatch::Config cfg = {}, bool startup_warned = false,
                      std::function<std::optional<wddmwatch::Others>()> others = nullptr);
    // samples the WDDM probe (warns through vram_watch_); "" without a probe
    std::string vramNote();
    // idle: one sample a minute so /health and the spill-over line stay current; returns the next due time
    std::optional<TimePoint> vramIdleTick();
    // for the other modules (arch025-engine S9; each body is the former statement / expression at its call site)
    void vramHealth(EngineApi::Health& h);  // Engine::health: the vram_mu_ lock and the five vram reads
    bool vramSpilling() const { return vram_warn_ && vram_watch_.spilling(); }  // engine_decode.cpp DecodeWatch report
    TimePoint vramLastSample() const { return vram_last_t_; }                   // idleWait: msSince(vram_last_t_)

private:
    const EngineThread& owner_;
    std::function<wddmwatch::Sample()>& vram_probe_;
    std::function<std::optional<wddmwatch::Others>()>& vram_others_;
    GovernorGlue& gv_;
    wddmwatch::Watch& vram_watch_;
    TimePoint& vram_idle_next_;
    double& vram_idle_s_;
    bool& vram_warn_;
    whirl::Mutex& vram_mu_;
    wddmwatch::Sample& vram_last_ WH_GUARDED_BY(vram_mu_);
    bool& vram_spill_ WH_GUARDED_BY(vram_mu_);
    TimePoint& vram_last_t_;
};

}  // namespace whirl::server
