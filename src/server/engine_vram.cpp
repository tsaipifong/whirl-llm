// Server engine, WDDM video memory watch (WDDM-LOG, issues #5 / #6): the probe set up at startup (setVramProbe), one
// sample per "batch |" line / request end (vramNote: hands the sample to the memory governor's govTick, then the spill
// warning) and once a minute while idle (vramIdleTick). The state is Engine's (vram_* members, vram_mu_ for health()).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Moved verbatim out of engine_gov.cpp (arch025-N4d / N10 engine_gov split; before that out of
// engine.cpp, arch024 S9).

#include "engine.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <format>

namespace whirl::server {

void Engine::setVramProbe(std::function<wddmwatch::Sample()> probe, bool warn, std::optional<wddmwatch::Sample> base,
                          wddmwatch::Config cfg, bool startup_warned, std::function<std::optional<wddmwatch::Others>()> others) {
    owner_.check("Engine::setVramProbe");
    vram_probe_ = std::move(probe);
    vram_others_ = std::move(others);
    vram_warn_ = warn;
    if (!vram_probe_) return;
    // the cost of one sample (taken twice: the first also builds the DXGI factory's caches)
    const TimePoint t0 = Clock::now();
    const wddmwatch::Sample first = vram_probe_();
    const double first_us = msSince(t0) * 1000.0;
    const TimePoint t1 = Clock::now();
    const wddmwatch::Sample s = vram_probe_();
    const double next_us = msSince(t1) * 1000.0;
    const wddmwatch::Sample b = base && base->ok ? *base : first;
    vram_watch_ = wddmwatch::Watch(b, cfg);
    vram_idle_s_ = cfg.idle_sample_s > 0.0 ? cfg.idle_sample_s : 60.0;
    if (!first.ok || !b.ok) {
        vram_probe_ = nullptr;
        return;
    }
    logI("vram watch: {} | sampled with each 'batch |' line, at request end and once a minute while idle ({:.0f} us per sample, "
         "first {:.0f} us){}",
         wddmwatch::format(s, b), next_us, first_us,
         warn ? "; warns while GPU memory spills (local usage over the budget, or shared grown by more than 256 MiB since the KV "
                "pool was sized, not counting pinned host memory WHIRL allocated)"
              : "; integrated GPU: numbers only, no warning");
    if (warn) {
        // a spill already there now: the startup line warned (counted as the watch's first warning), else warn here
        const auto ev = vram_watch_.check(s, std::chrono::duration<double>(Clock::now().time_since_epoch()).count());
        if (ev && !startup_warned) logW("{}", ev->text);
    }
    {
        whirl::MutexLock lk(vram_mu_);
        vram_last_ = s;
        vram_spill_ = warn && vram_watch_.spilling();
    }
    vram_idle_next_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(vram_idle_s_));
}

std::string Engine::vramNote() {
    if (!vram_probe_) return {};
    const wddmwatch::Sample s = vram_probe_();
    vram_last_t_ = Clock::now();
    // v0.2.2.2: the memory governor decides first; its actions run at the next safe point of the loop (govApply), and the
    // spill warning is logged after them (the warning comes last)
    govTick(s);
    if (vram_warn_) {
        const double now_s = std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
        if (const auto ev = vram_watch_.check(s, now_s)) {
            if (ev->warn) {
                std::string t = ev->text;
                if (vram_others_)
                    if (const auto o = vram_others_()) t += wddmwatch::othersNote(*o);
                if (gov_) gov_warn_ = t;
                else logW("{}", t);
            } else {
                logI("{}", ev->text);
            }
        }
    }
    {
        whirl::MutexLock lk(vram_mu_);
        vram_last_ = s;
        vram_spill_ = vram_warn_ && vram_watch_.spilling();
    }
    // while the governor has reduced something, idle samples every 5 s (recovery, the re-allocation when idle)
    const double idle_s = gov_ && gov_->reduced() ? std::min(vram_idle_s_, 5.0) : vram_idle_s_;
    vram_idle_next_ = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(idle_s));
    return wddmwatch::format(s, vram_watch_.baseline());
}

std::optional<TimePoint> Engine::vramIdleTick() {
    owner_.check("Engine::vramIdleTick");
    if (!vram_probe_) return std::nullopt;
    if (Clock::now() >= vram_idle_next_) (void)vramNote();  // logs only a spill warning / the spill-over line
    return vram_idle_next_;
}

}  // namespace whirl::server
