// Server engine, memory governor glue (arch024 M7): the v0.2.2.2 memory governor's decisions and actions (govTick /
// govApply / govFreeCache / govRestoreCache / govResize / govDrain / govPublish); the WDDM video memory watch that
// feeds it (setVramProbe / vramNote / vramIdleTick) is engine_vram.cpp (arch025-N4d / N10 engine_gov split). The
// policy is memgov::Governor (src/common/mem_governor.cpp, unchanged); the
// resources are changed through their modules: CkptStore park / unpark / release (slot and shared checkpoints),
// PrefixStore waitSpeFences / parkSpes / unparkSpes (the shared prefix table), PagePool reclaim and initPool (the
// page bookkeeping), ServerModel::resizePool (the pool arrays).
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Moved out of engine.cpp (arch024 S9) with the same rules, order and log lines. Prior art:
// progress/arch024-S9.PRIOR_ART.md (llama.cpp ggml_backend_dev_memory behind the device interface; vLLM profiles
// once at startup, no runtime governor).

#include "engine.h"

#include "log.h"

#include <algorithm>
#include <chrono>
#include <format>
#include <thread>

namespace whirl::server {

// ---------------------------------------------------------------------------
// v0.2.2.2 memory governor (whirl/mem_governor.h): decisions at each probe sample (govTick), actions at a safe point
// of runLoop (govApply: no forward in flight). The KV pool changes only while every slot is idle, so a running
// request's KV is never touched.

namespace {
constexpr double kGiB = 1073741824.0;
constexpr double kMiB = 1048576.0;
double monoS() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
}  // namespace

void Engine::setGovernor(const memgov::Config& c, std::function<std::uint64_t(std::uint64_t)> margin, std::uint64_t ram_tier_mb,
                         std::uint64_t vram_total) {
    owner_.check("Engine::setGovernor");
    gov_.emplace(c);
    gov_vram_total_ = vram_total;
    gov_margin_ = std::move(margin);
    gov_ram_mb_ = ram_tier_mb;
    ctx_new_ = std::min(ctx_, c.ctx > 0 ? c.ctx : ctx_);
    ctx_pub_.store(ctxForNew(), std::memory_order_relaxed);
    gov_last_ = "none";
    govPublish();
}

std::string Engine::governorJson() {
    whirl::MutexLock lk(vram_mu_);
    return gov_json_;
}

void Engine::govPublish() {
    if (!gov_) return;
    memgov::Status st;
    st.on = true;
    st.reduced = gov_->reduced();
    st.pool_tokens = pool_.pages() * kv_page;
    st.pool_tokens_start = gov_->config().pool_start;
    st.ctx_per_request = ctxForNew();
    st.slots = static_cast<std::uint32_t>(slots_.size());
    std::uint32_t parked = 0;
    for (const Slot& sl : slots_) parked += sl.parked ? 1 : 0;
    st.slots_admitting = gov_hold_ ? 0 : st.slots - parked;
    st.admission_held = gov_hold_;
    st.parked_mib = gov_parked_ >> 20;
    st.shrinks = gov_->shrinks();
    st.grows = gov_->grows();
    st.last_action = gov_last_;
    ctx_pub_.store(ctxForNew(), std::memory_order_relaxed);
    whirl::MutexLock lk(vram_mu_);
    gov_json_ = memgov::statusJson(st);
}

bool Engine::govAllIdle() const {
    for (const Slot& sl : slots_)
        if (sl.phase != Phase::idle || sl.job != nullptr || sl.restore != nullptr) return false;
    return true;
}

std::uint64_t Engine::govFreeable() const {
    std::uint64_t b = 0;
    bool pf = false;
    for (const Slot& sl : slots_) {
        pf = pf || sl.phase == Phase::prefill || sl.phase == Phase::restore;
        if (sl.phase == Phase::idle && sl.job == nullptr && sl.restore == nullptr && !sl.parked) b += sl.ckpts.size() * ck_.bytes(true);
    }
    if (!pf && !spe_parked_) b += ps_.spes().size() * ck_.bytes(false);
    return b;
}

void Engine::govTick(const wddmwatch::Sample& s) {
    owner_.check("Engine::govTick");
    if (!gov_ || !s.ok || s.local_budget == 0) return;
    memgov::Input in;
    in.now_s = monoS();
    in.local = s.local_usage;
    in.budget = s.local_budget;
    in.margin = gov_margin_ ? gov_margin_(s.local_budget) : 0;
    in.pool = pool_.pages() * kv_page;
    in.all_idle = govAllIdle();
    in.parked = gov_parked_;
    in.freeable = in.all_idle ? 0 : govFreeable();
    const memgov::Plan p = gov_->tick(in);
    const double free_gib = (static_cast<double>(in.budget) - static_cast<double>(in.local)) / kGiB;
    if (p.episode_start)
        logW("memory governor: {:.2f} GiB free under the Windows video memory (WDDM) budget of {:.2f} GiB (WHIRL uses {:.2f} GiB), "
             "less than half of the {} MiB margin (another program likely took VRAM); acting in this order: free cache, "
             "queue new requests, context for new requests, a smaller KV pool when every slot is idle",
             free_gib, in.budget / kGiB, in.local / kGiB, in.margin >> 20);
    if (p.ram_tier_note && gov_ram_mb_ > 0)
        logI("memory governor: RAM tier kept at {} MiB (untouched: it is pinned system RAM, not VRAM, and does not count against "
             "the VRAM budget; 17 GiB pinned left the R9700 budget unchanged in our test)",
             gov_ram_mb_);
    if (p.hold_admission && !gov_hold_)
        logW("memory governor: new requests queue until the running request(s) finish, then the KV pool is re-allocated smaller "
             "({} -> {} tokens, {:.2f} -> {:.2f} GiB)",
             in.pool, p.pool_target.value_or(in.pool), static_cast<double>(m_.poolPages()) * kv_page * kvTokBytes() / kGiB,
             static_cast<double>(p.pool_target.value_or(in.pool)) * kvTokBytes() / kGiB);
    if (p.ctx_new != ctxForNew())
        logI("memory governor: context for new requests {} -> {} tokens ({}; running requests keep theirs)", ctxForNew(), p.ctx_new,
             p.ctx_new < ctxForNew() ? "the smaller KV pool holds no more" : "the KV pool holds it again");
    gov_plan_ = p;
    gov_pending_ = true;
}

void Engine::govDrain() {
    owner_.check("Engine::govDrain");
    // tier IO that reads or writes the pool or the checkpoint buffers: spills (device -> pinned copies on the tier stream)
    for (Slot& sl : slots_) {
        sl.spill_ev.waitIfPending();
        sl.spill_busy = false;
    }
    for (Spe& sp : ps_.spes()) {
        sp.spill_ev.waitIfPending();
        sp.ready_ev.waitIfPending();
        sp.spill_busy = false;
    }
    pool_.reclaim(true);
    ops_.streamSync(m_.stream());
    if (tb_.tier() != nullptr) ops_.streamSync(tb_.tier()->stream());
}

void Engine::govFreeCache(bool park) {
    owner_.check("Engine::govFreeCache");
    std::size_t n_sl = 0, n_pages = 0, n_spe = 0, n_park = 0;
    std::uint64_t ck_bytes = 0;
    bool pf = false;
    for (const Slot& sl : slots_) pf = pf || sl.phase == Phase::prefill || sl.phase == Phase::restore;
    for (Slot& sl : slots_) {
        if (sl.phase != Phase::idle || sl.job != nullptr || sl.restore != nullptr) continue;
        if (!sl.pages.empty()) {
            ++n_sl;
            n_pages += sl.pages.size();
            freePages(sl);
        }
        if (park && !sl.parked && !sl.ckpts.empty()) {
            ck_bytes += parkSlotCkpts(ck_, sl);
            ++n_park;
        }
    }
    if (!pf) {
        for (Spe& sp : ps_.spes()) {
            if (!sp.valid) continue;
            ++n_spe;
            n_pages += sp.pages.size();
            dropSpe(sp);
        }
        if (park && !spe_parked_ && !ps_.spes().empty()) {
            ps_.waitSpeFences();
            pool_.reclaim(true);
            ck_bytes += ps_.parkSpes(ck_);  // destroys the events (device::EventH)
            spe_parked_ = true;
        }
    }
    gov_parked_ += ck_bytes;
    if (n_sl == 0 && n_spe == 0 && ck_bytes == 0) return;
    std::string t = std::format("memory governor: freed reusable cache: prefix cache of {} idle slot(s) and {} shared prefix "
                                "checkpoint(s) dropped from the KV pool ({} pages; their content went to the RAM tier at request "
                                "end when it had room)",
                                n_sl, n_spe, n_pages);
    if (ck_bytes > 0)
        t += std::format("; {:.0f} MiB of checkpoint buffers freed now (VRAM back to Windows; until they are restored {} idle slot(s) "
                         "take no new request{})",
                         ck_bytes / kMiB, n_park, spe_parked_ ? " and no shared prefix checkpoint is kept" : "");
    else
        t += "; the pool's VRAM itself returns to Windows only when the pool is re-allocated smaller";
    logI("{}", t);
    gov_last_ = std::format("freed cache: {} slot(s), {} shared, {} pages, {:.0f} MiB checkpoint buffers", n_sl, n_spe, n_pages, ck_bytes / kMiB);
}

void Engine::govRestoreCache() {
    owner_.check("Engine::govRestoreCache");
    gov_restored_ = 0;
    if (gov_parked_ == 0) return;
    std::uint64_t bytes = 0;  // of the slots / shared checkpoints restored completely
    std::size_t n = 0;
    Slot* cur = nullptr;      // the slot being restored (its partial buffers are freed on failure)
    try {
        for (Slot& sl : slots_) {
            if (!sl.parked) continue;
            cur = &sl;
            // a parked slot holds no checkpoint buffer (govFreeCache cleared them; a failed restore frees its partial ones)
            sl.ckpts.clear();
            sl.ckpts.resize(opt_.n_ck);
            ck_.unpark(sl.ckpts, true);
            bytes += sl.ckpts.size() * ck_.bytes(true);
            sl.parked = false;
            cur = nullptr;
            ++n;
        }
        if (spe_parked_) {
            // a failure part way frees what the shared checkpoints got (PrefixStore::unparkSpes) and rethrows
            bytes += ps_.unparkSpes(ck_, opt_.n_spe);
            spe_parked_ = false;
        }
    } catch (const std::exception& ex) {
        // REVIEW-GOV-1 P2-2: free what the unfinished slot / shared checkpoints got, so they stay parked holding nothing and
        // the next try starts clean (before arch024 S3 initCkpt overwrote raw pointers: a retry over partial buffers leaked them)
        // {}: these buffers were just allocated, no spill has read them (F1; arch024 S3 test ckpt_restore_fail)
        if (cur != nullptr) {
            ck_.park(cur->ckpts, {});
            cur->ckpts.clear();
        }
        gov_parked_ = gov_parked_ > bytes ? gov_parked_ - bytes : 0;
        gov_restored_ = bytes;
        logW("memory governor: restoring the checkpoint buffers failed after {:.0f} MiB ({}); the slots not restored stay parked and "
             "are tried again later",
             bytes / kMiB, ex.what());
        return;
    }
    gov_parked_ = 0;
    gov_restored_ = bytes;
    logI("memory governor: checkpoint buffers restored ({:.0f} MiB, {} slot(s){}): every slot takes requests again", bytes / kMiB, n,
         opt_.n_spe > 0 ? " and the shared prefix checkpoints" : "");
}

bool Engine::govResize(std::uint32_t tokens, bool shrink) {
    owner_.check("Engine::govResize");
    const std::uint32_t old = pool_.pages() * kv_page;
    if (tokens == old) return false;
    const TimePoint t0 = Clock::now();
    const wddmwatch::Sample before = vram_probe_ ? vram_probe_() : wddmwatch::Sample{};
    // 1. the tier IO that reads or writes the pool, then the VRAM prefix cache (its content is in the RAM tier when it had
    // room: the next request restores from there)
    govDrain();
    std::size_t n_sl = 0, n_spe = 0;
    for (Slot& sl : slots_) {
        if (sl.pages.empty() && sl.cache_tokens.empty()) continue;
        ++n_sl;
        freePages(sl);
    }
    for (Spe& sp : ps_.spes()) {
        if (!sp.valid) continue;
        ++n_spe;
        dropSpe(sp);
    }
    pool_.reclaim(true);
    // 2. the pool itself (the decode path of the server keeps no captured graph: model.use_graph is off in whirl-server)
    bool ok = false;
    try {
        ok = m_.resizePool(tokens);
    } catch (const std::exception& ex) {
        logE("memory governor: the KV pool could not be allocated again ({}): {}", ex.what(), ops_.lastErrorString());
        throw;
    }
    // 3. page bookkeeping and the tier's view of the pool arrays (the per-token layout is unchanged)
    initPool();
    const double ms = msSince(t0);
    // the WDDM usage follows a free over ~0.1-0.3 s: read it until two reads 50 ms apart agree (at most 1 s; every slot is
    // idle, a request that arrives meanwhile waits at most that long)
    wddmwatch::Sample after = vram_probe_ ? vram_probe_() : wddmwatch::Sample{};
    for (int i = 0; i < 20 && after.ok; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const wddmwatch::Sample a2 = vram_probe_();
        const bool same = a2.ok && a2.local_usage == after.local_usage;
        after = a2;
        if (same) break;
    }
    const std::uint32_t now_tok = pool_.pages() * kv_page;
    if (!ok) {
        logW("memory governor: the KV pool could not be re-allocated to {} tokens; kept at {} tokens ({:.0f} ms)", tokens, now_tok, ms);
        gov_last_ = std::format("re-allocation to {} tokens failed", tokens);
        return false;
    }
    gov_resize_before_ = before.ok ? before.local_usage : 0;
    gov_resize_delta_ = (static_cast<std::int64_t>(now_tok) - static_cast<std::int64_t>(old)) * static_cast<std::int64_t>(kvTokBytes());
    std::string vr;
    if (before.ok && after.ok)
        vr = std::format("; WHIRL's VRAM {:.2f} -> {:.2f} GiB, budget {:.2f} GiB", before.local_usage / kGiB, after.local_usage / kGiB,
                         after.local_budget / kGiB);
    logW("memory governor: KV pool re-allocated {} -> {} tokens ({:.2f} -> {:.2f} GiB) in {:.0f} ms with every slot idle; VRAM "
         "prefix cache of {} slot(s) and {} shared checkpoint(s) dropped first (the RAM tier keeps what it had){}",
         old, now_tok, static_cast<double>(old) * kvTokBytes() / kGiB, static_cast<double>(now_tok) * kvTokBytes() / kGiB, ms, n_sl,
         n_spe, vr);
    gov_last_ = std::format("pool {} -> {} tokens", old, now_tok);
    return true;
}

void Engine::govApply() {
    owner_.check("Engine::govApply");
    if (!gov_ || !gov_pending_) return;
    gov_pending_ = false;
    const memgov::Plan p = gov_plan_;
    const bool idle = govAllIdle();
    const bool resize_now = p.pool_target && idle && *p.pool_target != pool_.pages() * kv_page;
    // 1. free reusable cache (idle prefix cache; while requests run, the checkpoint buffers of idle slots / shared ones)
    if (p.free_cache) govFreeCache(!resize_now && !idle);
    // 3. / 4. admission and context for new requests
    const bool hold = p.hold_admission && !idle;
    if (!hold && gov_hold_ && !resize_now) logI("memory governor: new requests are admitted again (the budget has room)");
    gov_hold_ = hold;
    ctx_new_ = p.ctx_new;
    // 5. the pool, with every slot idle
    bool resized = false;
    std::optional<std::uint32_t> target = resize_now ? p.pool_target : std::nullopt;
    if (target && !p.shrink && gov_vram_total_ > 0 && vram_others_ && vram_probe_) {
        // a grow: the budget alone can read high again while another program still holds VRAM (R9700: 31.02 GiB with a 4 GiB
        // hog alive once WHIRL had shrunk), so cap it to what fits next to what the other programs use now (one PDH read,
        // ~0.3 s, only here, while idle)
        const auto o = vram_others_();
        const wddmwatch::Sample s = vram_probe_();
        if (o && s.ok) {
            const std::uint64_t margin_full = gov_margin_ ? gov_margin_(s.local_budget) : 0;
            const std::uint32_t pool = pool_.pages() * kv_page;
            const std::uint64_t page_b = kvTokBytes() * kv_page;
            const auto addFor = [&](std::uint64_t m) {
                const std::uint64_t used = s.local_usage + o->now + m;
                const std::uint64_t room = gov_vram_total_ > used ? gov_vram_total_ - used : 0;
                return page_b ? static_cast<std::uint32_t>(room / page_b) * kv_page : 0u;
            };
            // the governor plans the last step straight to the startup pool with endMargin free: the cap allows the same for
            // that step only (a partial grow keeps the full margin)
            std::uint64_t margin = margin_full;
            std::uint32_t capped = std::min(*target, pool + addFor(margin_full));
            if (*target >= gov_->config().pool_start && capped < *target && pool + addFor(memgov::endMargin(margin_full)) >= *target) {
                margin = memgov::endMargin(margin_full);
                capped = *target;
            }
            const std::uint64_t min_add = std::min<std::uint64_t>(gov_->config().min_grow / std::max<std::uint64_t>(kvTokBytes(), 1),
                                                                  *target - pool);
            if (capped <= pool || capped - pool < min_add) {
                if (!gov_grow_denied_logged_)
                    logI("memory governor: the KV pool stays at {} tokens for now: other programs use {:.2f} GiB of this GPU's {:.2f} GiB "
                         "(WHIRL {:.2f} GiB, margin {} MiB), although the WDDM budget ({:.2f} GiB) would allow more; checked again later",
                         pool, o->now / kGiB, gov_vram_total_ / kGiB, s.local_usage / kGiB, margin >> 20, s.local_budget / kGiB);
                gov_grow_denied_logged_ = true;
                gov_->growDenied(monoS());
                target.reset();
            } else {
                target = capped;
            }
        }
    }
    if (target) {
        resized = govResize(*target, p.shrink);
        if (resized && p.shrink) gov_hold_ = false;
    }
    if (gov_parked_ > 0 && (p.restore_cache || resized)) govRestoreCache();
    if (resized) {
        // the usage this should give, for the governor while the WDDM counters catch up (they show a free late)
        const std::int64_t exp = static_cast<std::int64_t>(gov_resize_before_) + gov_resize_delta_ + static_cast<std::int64_t>(gov_restored_);
        gov_->resized(pool_.pages() * kv_page, !p.shrink, monoS(), gov_resize_before_ > 0 && exp > 0 ? static_cast<std::uint64_t>(exp) : 0);
        ctx_new_ = gov_->ctxNew();
        if (!p.shrink) gov_grow_denied_logged_ = false;
    }
    if (p.episode_end) {
        logI("memory governor: back to the startup settings (KV pool {} tokens, context {} tokens per request, {} slot(s))",
             pool_.pages() * kv_page, ctxForNew(), slots_.size());
        gov_last_ = "back to the startup settings";
    }
    // 6. the warning last
    if (!gov_warn_.empty()) {
        logW("{}{}", gov_warn_,
             resized ? " (the KV pool was re-allocated smaller just now: see the line above)"
                     : (p.warn_slow ? " The running request(s) stay slower until they finish: their KV cannot move." : ""));
        gov_warn_.clear();
    }
    govPublish();
}

}  // namespace whirl::server
