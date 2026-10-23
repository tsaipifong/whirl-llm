// Tier bridge (arch024 M5: spill half S6b, restore half S7): see tier_bridge.h.
// SPDX-License-Identifier: Apache-2.0
#include "server/tier_bridge.h"

#include "server/log.h"
#include "whirl/model.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>

namespace whirl::server {

namespace {

constexpr std::uint32_t kv_page = qwen35::kv_page;
constexpr std::uint8_t kNoTag = 0xff;
using Clock = std::chrono::steady_clock;

double msSince(Clock::time_point t0) { return std::chrono::duration<double, std::milli>(Clock::now() - t0).count(); }

std::int64_t steadyMs(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(t.time_since_epoch()).count();
}

// CACHE-1: changes when the entry is updated in place (its session went on); not on LRU touches (entries
// referencing its blocks touch it)
std::uint64_t tierStamp(const tier::Entry* e) {
    return static_cast<std::uint64_t>(e->t_mod.time_since_epoch().count()) * 0x9e3779b97f4a7c15ull ^ e->n_tok;
}

}  // namespace

TierBridge::TierBridge(device::Backend& ops, ServerModel& m, const ChatTokens& ct, PrefixStore& ps,
                       const std::vector<KvArr>& kv, const Config& cfg)
    : ops_(ops), m_(m), ct_(ct), ps_(ps), kv_(kv), cfg_(cfg), conv_bytes_(m.convBytes()), ssm_bytes_(m.ssmBytes()) {}

// ---------------------------------------------------------------------------
// item G: host tiers

void TierBridge::waitSpill(SlotTier& st) {
    // (also after an eviction cleared spill_busy: the spill may still read the
    // checkpoint buffers; item S)
    if (!st.spill_ev) return;
    st.spill_busy = false;
    if (ops_.eventDone(st.spill_ev.get())) return;
    spill_waits_ += 1;
    ops_.streamWaitEvent(m_.stream(), st.spill_ev.get());
}

void TierBridge::ckSpans(const Ckpt& c, std::uint64_t base, std::uint8_t tag) {
    const auto& cfg = m_.cfg();
    std::uint64_t off = base;
    for (std::uint32_t i = 0; i < cfg.n_layer; ++i) {
        if (c.conv[i] == 0) continue;
        spans_.push_back({c.conv[i], off, conv_bytes_, tag});
        off += conv_bytes_;
        spans_.push_back({c.ssm[i], off, ssm_bytes_, tag});
        off += ssm_bytes_;
    }
    const std::uint64_t hb = static_cast<std::uint64_t>(cfg.n_embd) * 4;
    spans_.push_back({c.hid, off, hb, tag});
    off += hb;
    spans_.push_back({c.logits, off, static_cast<std::uint64_t>(cfg.n_vocab) * 4, tag});
}

void TierBridge::pageSpans(std::int32_t phys, std::uint64_t base) {
    std::uint64_t off = base;
    for (const KvArr& a : kv_) {
        const std::uint64_t len = static_cast<std::uint64_t>(kv_page) * a.row;
        spans_.push_back({a.base + static_cast<std::uint64_t>(phys) * len, off, len, kNoTag});
        off += len;
    }
}

void TierBridge::spillSpe(Spe& s) {
    tier::Tier* t = tier_;
    if (!t) return;
    if (s.n < t->cfg.min_tokens) return;
    // already there (restored from it, or written before a restart)
    // (s.tokens.size() == s.n)
    if (const std::uint64_t id = ps_.cache().findTierShared(s.tokens); id != 0) {
        s.tier_id = id;
        return;
    }
    const tier::Layout lay = t->lay;
    tier::Entry* x = t->newEntry(1);
    const std::uint32_t npages = static_cast<std::uint32_t>(s.pages.size());
    const std::uint64_t kv_off = x->kvOff(lay);
    bool ok = true;
    {
        const auto r = tier::Tier::lbRange(0, lay.ck_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (ok) {
        const auto r = tier::Tier::lbRange(kv_off, static_cast<std::uint64_t>(npages) * lay.page_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (!ok) {
        logW("shared prefix: no room for {} tokens in the RAM tier; not kept there", s.n);
        t->removeEntry(x);
        return;
    }
    // the state / page copies into the checkpoint run on the main stream
    // (mid-prefill): the tier stream starts after them
    if (!s.ready_ev) s.ready_ev = device::EventH::create(ops_);
    ops_.eventRecord(s.ready_ev.get(), m_.stream());
    ops_.streamWaitEvent(t->stream(), s.ready_ev.get());
    spans_.clear();
    ckSpans(s.ck, 0);
    for (std::size_t j = 0; j < s.pages.size(); ++j) pageSpans(s.pages[j], kv_off + j * lay.page_bytes);
    const std::uint64_t bytes = t->copyOut(x, spans_);
    x->tokens.insert(x->tokens.end(), s.tokens.begin(), s.tokens.end());
    x->n_tok = s.n;
    x->pages = npages;
    x->ck[0] = tier::CkMeta{s.n, 1, 0, ckKindCode(s.kind), 0, 1};
    t->trim(x);
    x->t_mod = Clock::now();
    x->owner.reset();
    if (!s.spill_ev) s.spill_ev = device::EventH::create(ops_);
    t->fence(x, s.spill_ev.get());
    s.spill_busy = true;
    s.tier_id = x->id;
    t->stats.spills += 1;
    t->stats.spill_bytes += bytes;
    logI("shared prefix: spill {} checkpoint @{} to RAM ({:.1f} MiB)", s.kind, s.n, static_cast<double>(bytes) / 1048576.0);
}

void TierBridge::spillSlot(std::uint32_t id, bool idle, SlotPrefix& sp, SlotTier& st) {
    tier::Tier* t = tier_;
    if (!t) return;
    if (!idle || sp.pages.empty()) return;
    std::uint32_t P = 0;
    for (const Ckpt& c : sp.ckpts)
        if (c.valid && c.pos > P) P = c.pos;
    if (P < t->cfg.min_tokens || P > sp.cache_tokens.size()) return;
    const std::uint32_t npages = (P + kv_page - 1) / kv_page;
    if (npages > sp.pages.size()) return;
    tier::Entry* x = nullptr;
    std::uint32_t vp = 0;
    bool fresh = true;
    if (st.tier_id != 0) {
        if (tier::Entry* old = t->find(st.tier_id)) {
            // in place only when this slot's content continues the entry or diverged near its end
            // (retry / regenerate / edit-last: the old tail is abandoned). A long tail is another
            // branch -- a conversation that shares only a prefix (a subagent with the same system
            // prompt on the main agent's slot, or another slot that restored a shorter checkpoint of
            // it): a new entry, the old branch stays in the tiers (FR-CACHE P3-3).
            const bool near_end = cache::nearEndDivergence(old->n_tok, static_cast<std::size_t>(st.tier_pages) * kv_page);
            if (old->in_ram && old->owner && *old->owner == id && old->pin == 0 && !old->loading &&
                old->nck == sp.ckpts.size() && near_end) {
                if (old->io_busy) {
                    // an SSD write is reading it: retry when that is done
                    st.spill_wanted = true;
                    return;
                }
                x = old;
                fresh = false;
                x->reused = true;  // the session grew by appending (resumed): worth an SSD write
                vp = std::min(st.tier_pages, old->n_tok / kv_page);
            } else if (!near_end && old->owner && *old->owner == id) {
                old->owner.reset();  // this slot no longer extends it (another branch now)
            }
        }
    }
    if (fresh) {
        x = t->newEntry(static_cast<std::uint32_t>(sp.ckpts.size()));
        // a session restored from the tier earlier is a multi-turn one: its successor entry is worth keeping
        if (st.tier_id != 0)
            if (const tier::Entry* prev = t->find(st.tier_id)) x->reused = prev->reused;
    }
    const tier::Layout lay = t->lay;
    const std::uint64_t stride = lay.ckStride();
    const std::uint64_t kv_off = x->kvOff(lay);
    // RAM blocks: changed checkpoints, pages vp ..
    bool ok = true;
    for (std::size_t j = 0; j < sp.ckpts.size(); ++j) {
        const Ckpt& c = sp.ckpts[j];
        if (!c.valid || c.pos > P) continue;
        const auto r = tier::Tier::lbRange(j * stride, lay.ck_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (ok && npages > vp) {
        const auto r = tier::Tier::lbRange(kv_off + static_cast<std::uint64_t>(vp) * lay.page_bytes,
                                           static_cast<std::uint64_t>(npages - vp) * lay.page_bytes);
        if (!t->ensureBlocks(x, r[0], r[1])) ok = false;
    }
    if (!ok && fresh && t->roomPending()) {
        // RAM pressure: older entries are being written to the SSD tier first; their blocks free up when
        // those writes finish (the slot keeps its state meanwhile): retry then
        t->removeEntry(x);
        st.spill_wanted = true;
        return;
    }
    if (!ok) {
        logW("slot {} | kv tier: no room for {} tokens in the RAM tier ({} MiB); not kept", id, P, t->cfg.ram_bytes >> 20);
        if (fresh) t->removeEntry(x);
        else t->trim(x);
        if (!fresh) st.tier_id = x->id;
        return;
    }
    // spans: older checkpoints first (the next request overwrites them first), then pages
    spans_.clear();
    const std::size_t nck = std::min(sp.ckpts.size(), tier::max_ck);
    std::array<std::size_t, tier::max_ck> order{};
    for (std::size_t j = 0; j < nck; ++j) order[j] = j;
    std::sort(order.begin(), order.begin() + nck,
              [&](std::size_t a, std::size_t b) { return sp.ckpts[a].seq < sp.ckpts[b].seq; });
    std::uint64_t ck_bytes = 0;
    for (std::size_t oi = 0; oi < nck; ++oi) {
        const std::size_t j = order[oi];
        const Ckpt& c = sp.ckpts[j];
        if (!c.valid || c.pos > P) {
            if (x->ck[j].valid != 0) t->dropCkBlocks(x, static_cast<std::uint32_t>(j));
            x->ck[j] = tier::CkMeta{};
            continue;
        }
        const bool same = x->ck[j].valid != 0 && x->ck[j].pos == c.pos && x->ck[j].seq == c.seq;
        if (!same) {
            ckSpans(c, j * stride);
            ck_bytes += lay.ck_bytes;
        }
        x->ck[j] = tier::CkMeta{c.pos, 1, static_cast<std::uint8_t>(c.has_logits ? 1 : 0), ckKindCode(c.kind), 0, c.seq};
    }
    for (std::uint32_t pg = vp; pg < npages; ++pg) pageSpans(sp.pages[pg], kv_off + static_cast<std::uint64_t>(pg) * lay.page_bytes);
    const std::uint64_t bytes = t->copyOut(x, spans_);
    x->tokens.assign(sp.cache_tokens.begin(), sp.cache_tokens.begin() + P);
    x->n_tok = P;
    x->pages = npages;
    t->trim(x);
    x->t_mod = Clock::now();
    x->owner = id;
    if (!st.spill_ev) st.spill_ev = device::EventH::create(ops_);
    t->fence(x, st.spill_ev.get());
    if (cfg_.verify) {
        // REVIEW-ARCH4 P3-4: every verify attempt leaves exactly one line (byte-identical, differ or error), so the
        // release gate can count attempts against spills / restores and fail on a verify that threw
        bool sync_ok = true;
        try {
            ops_.eventSync(st.spill_ev.get());
        } catch (const std::exception& ex) {
            sync_ok = false;
            logE("slot {} | kv tier verify: spill of {} tok: error (event sync: {})", id, P, ex.what());
        }
        if (sync_ok) {
            try {
                const auto v = t->verifySpans(x, spans_);
                if (v.bad > 0) logE("slot {} | kv tier verify: spill of {} tok: {} pieces differ", id, P, v.bad);
                else
                    logI("slot {} | kv tier verify: spill of {} tok byte-identical ({:.1f} MiB)", id, P,
                         static_cast<double>(v.bytes) / 1048576.0);
            } catch (const std::exception& ex) {
                logE("slot {} | kv tier verify: spill of {} tok: error ({})", id, P, ex.what());
            }
        }
    }
    st.spill_busy = true;
    st.spill_upto = P;
    st.spill_wanted = false;
    st.tier_id = x->id;
    st.tier_pages = P / kv_page;
    t->stats.spills += 1;
    t->stats.spill_bytes += bytes;
    logI("slot {} | kv tier: spill {} tok to RAM ({}, pages {}..{}, checkpoints {:.0f} MiB, {:.1f} MiB in all; RAM tier {:.2f} "
         "/ {:.2f} GiB)",
         id, P, fresh ? "new entry" : "in place", vp, npages, static_cast<double>(ck_bytes) / 1048576.0,
         static_cast<double>(bytes) / 1048576.0, static_cast<double>(t->ramUsedBytes()) / 1073741824.0,
         static_cast<double>(t->cfg.ram_bytes) / 1073741824.0);
    if (fresh) supersedeMark(id, x);
    else supersedeApply(id, x);
}

// CACHE-1 superseded sessions (agent compact / compress: same system prompt,
// early or mid-way divergence, clearly shorter; cache::supersedes). The new
// conversation's first spill marks the old entries; its second spill cuts
// those not used since down to the shared prefix in RAM and on SSD (blocks
// other files reference stay). Retry / regenerate / edit-last and other
// system prompts never match; a parallel session that keeps using its entry
// is left alone.
void TierBridge::supersedeMark(std::uint32_t slot_id, tier::Entry* x) {
    tier::Tier* t = tier_;
    if (!t || !x || !cfg_.supersede) return;
    const cache::Tokens nt(x->tokens.data(), x->n_tok);
    const std::size_t sys = ct_.sysBoundary(nt);
    if (sys == 0 || sys >= nt.size()) return;
    // FR-CACHE P2-4: cut only an entry of the same task lineage (it shares the first message after the
    // system prompt too); one sharing only the system prompt is demoted, never cut
    const std::size_t task = ct_.nextTurn(nt, sys);
    const std::size_t task_end = task > sys ? task : std::numeric_limits<std::size_t>::max();
    for (tier::Entry* e : t->entries) {
        if (e == x || e->n_tok == 0) continue;
        const cache::Tokens ot(e->tokens.data(), std::min<std::size_t>(e->n_tok, e->tokens.size()));
        const cache::Supersede sp = cache::supersedes(ot, nt, sys, {}, task_end);
        if (!sp.yes) continue;
        supersede_.mark(x->id, e->id, sp.keep, tierStamp(e), sp.cut);
        logI("slot {} | kv tier: entry {:016x} ({} tok) looks superseded by this conversation (shares {} tok); {} after its next "
             "turn unless used again",
             slot_id, e->id, e->n_tok, sp.keep, sp.cut ? "cut" : "evicted first (shares only the system prompt: not cut)");
    }
}

void TierBridge::supersedeApply(std::uint32_t slot_id, tier::Entry* x) {
    tier::Tier* t = tier_;
    if (!t || !x) return;
    std::vector<cache::SupersedeTracker::Pending> pend = supersede_.take(x->id);
    // several passes: a superseded branch referencing blocks of another
    // superseded entry holds them until it is cut itself
    for (int pass = 0; pass < 4 && !pend.empty(); ++pass) {
        bool any = false;
        for (auto it = pend.begin(); it != pend.end();) {
            tier::Entry* e = t->find(it->old_id);
            if (e == nullptr || tierStamp(e) != it->stamp) {  // gone, or used again
                it = pend.erase(it);
                continue;
            }
            if (!it->cut) {
                // a compact, or another conversation with the same system prompt (a subagent while the
                // main agent waits): kept whole, but the first RAM / SSD eviction victim (no SSD write for it)
                t->demote(e);
                logI("slot {} | kv tier: entry {:016x} ({} tok) not used since a conversation with the same system prompt "
                     "started: kept, evicted first under RAM / SSD pressure",
                     slot_id, it->old_id, e->n_tok);
                it = pend.erase(it);
                continue;
            }
            const std::uint32_t old_n = e->n_tok;
            const std::uint64_t freed = t->trimTo(e, static_cast<std::uint32_t>(it->keep));
            if (freed == 0) {
                ++it;
                continue;
            }
            any = true;
            logI("slot {} | kv tier: entry {:016x} ({} tok) superseded (compact / compress): cut to the {} shared tok, {:.0f} MiB freed",
                 slot_id, it->old_id, old_n, it->keep, static_cast<double>(freed) / 1048576.0);
            it = pend.erase(it);
        }
        if (!any) break;
    }
}

// Shutdown (Engine::stop()): finished sessions still waiting for their spill go to the RAM
// tier, and the RAM entries not yet current on the SSD tier are written now instead
// of after the usual delay (shutdown mode: reused entries only, CLOSE-REUSED -- the
// rule lives in Tier::tick / evict_policy.h; always mode: its seen-once filter), so a
// graceful stop (Ctrl+C, Ctrl+Break, console close) does not lose the last turns. Bounded by
// flush_ms; runs on the engine thread like every other tier step (tick: Engine::tierTick, which calls tick()).
bool TierBridge::flushOnStop(const std::function<bool()>& tick) {
    tier::Tier* t = tier_;
    if (!t) return false;
    const std::uint64_t saved_delay = t->cfg.ssd_delay_ms;
    t->cfg.ssd_delay_ms = 0;
    t->flushing = true;
    const std::uint64_t flush_ms = t->cfg.ssd_shutdown_only ? cfg_.flush_long_ms : cfg_.flush_ms;
    const auto t0 = Clock::now();
    const std::uint64_t w0 = t->stats.ssd_writes;
    bool busy = tick();
    if (busy) logI("shutdown: writing pending prefix-cache entries to the tiers (up to {} s)", flush_ms / 1000);
    while (busy && msSince(t0) < static_cast<double>(flush_ms)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        busy = tick();
    }
    t->cfg.ssd_delay_ms = saved_delay;
    t->flushing = false;
    if (busy)
        logW("shutdown: tier writes still pending after {} s; exiting without them", flush_ms / 1000);
    else if (t->stats.ssd_writes != w0 || msSince(t0) > 1.0)
        logI("shutdown: tier writes done in {:.0f} ms ({} SSD writes)", msSince(t0), t->stats.ssd_writes - w0);
    return busy;
}

// TIER-IDLE (policy: tier/idle_flush.h). Called from the idle wait of runLoop: once the server has been idle
// for idle_flush_ms, the tier writes its reused entries to SSD in the background (shutdown mode only).
void TierBridge::idlePoll(bool building) {
    tier::Tier* t = tier_;
    if (!t || !tier::idleFlushEnabled(t->cfg.ssd_shutdown_only, t->hasSsd(), cfg_.idle_flush_ms)) return;
    if (building) {  // a request is being built: not idle
        idle_clock_.activity();
        t->endIdleWrite();
        return;
    }
    if (idle_clock_.poll(steadyMs(std::chrono::steady_clock::now()), cfg_.idle_flush_ms)) {
        if (const std::size_t n = t->beginIdleWrite(); n > 0)
            logI("kv tier: idle for {:g} s: writing {} reused prefix-cache entries to SSD in the background (shutdown mode; "
                 "a request stops it)",
                 static_cast<double>(cfg_.idle_flush_ms) / 1000.0, n);
    }
}

std::optional<std::chrono::steady_clock::time_point> TierBridge::idleDue() const {
    const tier::Tier* t = tier_;
    if (!t || !tier::idleFlushEnabled(t->cfg.ssd_shutdown_only, t->hasSsd(), cfg_.idle_flush_ms)) return std::nullopt;
    const auto d = idle_clock_.deadline(cfg_.idle_flush_ms);
    if (!d) return std::nullopt;
    return std::chrono::steady_clock::time_point(std::chrono::milliseconds(*d));
}

// ---------------------------------------------------------------------------
// restores (arch024-S7: moved out of Engine::tryRestore / beginRestore / finishTail / cutTail / pollRestores /
// waitsForTier / prefetchFor / tierTick / logTier; Engine keeps the scheduler's part around these calls)

std::optional<tier::Match> TierBridge::restoreMatch(SlotPrefix& sp, std::span<const std::uint32_t> toks) {
    tier::Tier* t = tier_;
    if (!t) return std::nullopt;
    const auto mt = ps_.config().cold_resume
                        ? t->lookup(toks, [&](std::uint32_t p, std::uint8_t k) { return ps_.resumeOk(toks, p, k); })
                        : t->lookup(toks);
    if (!mt) return std::nullopt;
    // item S: a shared prefix entry off this prompt's schedule would not give
    // the cold result; one already in VRAM needs no restore
    if (mt->e->ck[mt->ck].kind >= 4 && !ps_.onSchedule(toks, mt->reuse)) return std::nullopt;
    std::uint32_t vr = ps_.cacheMatch(sp, toks).reuse;
    if (Spe* s = ps_.speMatch(toks)) vr = std::max(vr, s->n);
    if (mt->reuse <= vr) return std::nullopt;
    const std::uint32_t gain = mt->reuse - vr;
    if (gain < cfg_.min_gain || (cfg_.min_gain >= 512 && gain < mt->reuse / 40)) return std::nullopt;
    return mt;
}

bool TierBridge::startRestore(std::uint64_t job_id, std::uint32_t id, std::span<const std::uint32_t> toks, SlotPrefix& sp,
                              SlotTier& st, const tier::Match& mt, Clock::time_point t0) {
    tier::Tier* t = tier_;
    tier::Entry* x = mt.e;
    const tier::Layout lay = t->lay;
    // checkpoints: the matched one first, then the others by position, up to the slot's count
    std::array<bool, tier::max_ck> chosen{};
    st.rs_map.fill(-1);
    chosen[mt.ck] = true;
    for (std::size_t n_sel = 1; n_sel < sp.ckpts.size(); ++n_sel) {
        std::optional<std::size_t> best;
        for (std::size_t k = 0; k < x->nck; ++k) {
            if (chosen[k] || x->ck[k].valid == 0) continue;
            if (!best || x->ck[k].pos > x->ck[*best].pos) best = k;
        }
        if (!best) break;
        chosen[*best] = true;
    }
    std::int32_t nj = 0;
    for (std::size_t k = 0; k < x->nck; ++k) {
        if (!chosen[k]) continue;
        st.rs_map[k] = nj;
        ++nj;
    }
    // item T3: the request's part first (the matched checkpoint and the KV
    // pages); the slot's other checkpoints follow on the tier stream (tail)
    spans_.clear();
    ckSpans(sp.ckpts[static_cast<std::size_t>(st.rs_map[mt.ck])], static_cast<std::uint64_t>(mt.ck) * lay.ckStride());
    const std::uint64_t kv_off = x->kvOff(lay);
    for (std::uint32_t pg = 0; pg < x->pages; ++pg)
        pageSpans(sp.pages[pg], kv_off + static_cast<std::uint64_t>(pg) * lay.page_bytes);
    const std::size_t n_main = spans_.size();
    st.rs_tail.fill(false);
    for (std::size_t k = 0; k < x->nck; ++k) {
        if (!chosen[k] || k == mt.ck) continue;
        ckSpans(sp.ckpts[static_cast<std::size_t>(st.rs_map[k])], static_cast<std::uint64_t>(k) * lay.ckStride(),
                static_cast<std::uint8_t>(k));
        st.rs_tail[k] = true;
    }
    tier::Restore* r = t->beginRestore(x, spans_, n_main);
    if (!r) return false;
    r->t0 = t0;
    st.restore = r;
    // item S: a shared prefix entry is never updated in place by a slot
    st.rs_shared = x->ck[mt.ck].kind >= 4;
    if (!st.rs_shared) x->owner = id;
    logI("req {} | slot {} | kv tier: restoring {} tok from {} (checkpoint '{}' @{}, {:.1f} MiB; VRAM prefix was {})", job_id,
         id, x->n_tok, r->from_ssd ? "SSD" : "RAM", ckKindStr(x->ck[mt.ck].kind), mt.reuse,
         static_cast<double>(r->bytes) / 1048576.0, ps_.cacheMatch(sp, toks).reuse);
    return true;
}

void TierBridge::finishTail(std::uint32_t id, SlotPrefix& sp, SlotTier& st, bool wait) {
    tier::Tier* t = tier_;
    if (!t) return;
    tier::Restore* r = st.restore;
    if (!r) return;
    for (;;) {
        // (an exception marks the restore failed: the tail checkpoints stay invalid)
        const bool done = t->pumpTailNoThrow(r);
        if (done) break;
        if (!wait) return;
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    st.restore = nullptr;
    tier::Entry* x = t->find(r->entry_id);
    if (cfg_.verify && !r->failed && x != nullptr && r->cursor > r->n_main) {
        // (pieces of a checkpoint saveCkpt has overwritten since differ by design)
        std::size_t n_bad = 0, n = 0, n_err = 0;
        for (std::size_t i = r->n_main; i < r->cursor; ++i) {
            const tier::Piece& pc = r->pieces[i];
            if (pc.ck == kNoTag || !st.rs_tail[pc.ck]) continue;
            const std::int32_t j = st.rs_map[pc.ck];
            if (j >= 0 && sp.ckpts[static_cast<std::size_t>(j)].valid) continue;
            ++n;
            try {
                n_bad += t->verifyPieces(std::span<const tier::Piece>(&pc, 1)).bad;
            } catch (const std::exception&) {
                ++n_err;
            }
        }
        if (n_err > 0) logE("slot {} | kv tier verify: restore tail: error ({} of {} pieces could not be verified, {} differ)", id, n_err, n, n_bad);
        else if (n_bad > 0) logE("slot {} | kv tier verify: restore tail: {} of {} pieces differ", id, n_bad, n);
        else logI("slot {} | kv tier verify: restore tail byte-identical ({} pieces)", id, n);
    }
    if (r->failed || x == nullptr) {
        // copies may still be in flight into the tail buffers: order the main stream after them
        try {
            ops_.eventRecord(r->ev, t->stream());
            ops_.streamWaitEvent(m_.stream(), r->ev);
        } catch (const std::exception&) {
        }
    } else if (!sp.cache_tokens.empty()) {
        for (std::size_t k = 0; k < x->nck; ++k) {
            if (!st.rs_tail[k]) continue;
            const std::int32_t j = st.rs_map[k];
            if (j < 0) continue;
            Ckpt& c = sp.ckpts[static_cast<std::size_t>(j)];
            const tier::CkMeta cm = x->ck[k];
            if (c.valid || cm.valid == 0 || cm.pos > sp.reuse || cm.pos > sp.cache_tokens.size()) continue;
            c.valid = true;
            c.pos = cm.pos;
            c.seq = cm.seq;
            c.kind = ckKindStr(cm.kind);
            c.has_logits = cm.has_logits != 0;
        }
    }
    st.rs_tail.fill(false);
    t->endRestore(r);
}

void TierBridge::cutTail(std::uint32_t id, SlotPrefix& sp, SlotTier& st) {
    finishTail(id, sp, st, false);
    tier::Restore* r = st.restore;
    if (!r) return;
    tier::Tier* t = tier_;
    t->pumpRestoreNoThrow(r);  // (a throw marks the restore failed: the tail checkpoints stay invalid)
    for (std::size_t i = r->cursor; i < r->pieces.size(); ++i)
        if (r->pieces[i].ck != kNoTag) st.rs_tail[r->pieces[i].ck] = false;
    device::Event ev = nullptr;
    try {
        ev = t->cutTail(r);
    } catch (const std::exception&) {
        finishTail(id, sp, st, true);
        return;
    }
    try {
        ops_.streamWaitEvent(m_.stream(), ev);
    } catch (const std::exception&) {
        finishTail(id, sp, st, true);
    }
}

TierBridge::Pump TierBridge::pumpRestore(SlotTier& st) {
    tier::Tier* t = tier_;
    tier::Restore* r = st.restore;
    // an exception from the copies marks the restore failed (cold prefill), never a valid one
    const bool done = t->pumpRestoreNoThrow(r);
    if (!done) return Pump::pending;
    if (r->failed || t->find(r->entry_id) == nullptr) return Pump::failed;
    return Pump::restored;
}

void TierBridge::restoreFailed(std::uint64_t job_id, std::uint32_t id, SlotTier& st, const RestoreSteps& steps) {
    tier::Tier* t = tier_;
    tier::Restore* r = st.restore;
    logW("req {} | slot {} | kv tier: restore failed; prefilling from scratch", job_id, id);
    // copies already enqueued may still write the slot's pages / checkpoints
    try {
        ops_.eventRecord(r->ev, t->stream());
        ops_.streamWaitEvent(m_.stream(), r->ev);
    } catch (const std::exception&) {
    }
    st.restore = nullptr;
    st.rs_tail.fill(false);
    steps.drop_cache();
    t->endRestore(r);
}

void TierBridge::restoreDone(std::uint64_t job_id, std::uint32_t id, SlotPrefix& sp, SlotTier& st, const RestoreSteps& steps) {
    tier::Tier* t = tier_;
    tier::Restore* r = st.restore;
    tier::Entry& en = *t->find(r->entry_id);
    sp.cache_tokens.assign(en.tokens.begin(), en.tokens.begin() + en.n_tok);
    // the tier holds KV and state, not drafter features
    steps.drafter_reset(en.n_tok);
    for (Ckpt& c : sp.ckpts) c.valid = false;
    std::uint64_t max_seq = 0;
    for (std::size_t k = 0; k < en.nck; ++k) {
        const std::int32_t j = st.rs_map[k];
        if (j < 0) continue;
        max_seq = std::max(max_seq, en.ck[k].seq);
        if (st.rs_tail[k]) continue;  // item T3: valid once the tail is in
        Ckpt& c = sp.ckpts[static_cast<std::size_t>(j)];
        const tier::CkMeta cm = en.ck[k];
        c.valid = true;
        c.pos = cm.pos;
        c.seq = cm.seq;
        c.kind = ckKindStr(cm.kind);
        c.has_logits = cm.has_logits != 0;
        max_seq = std::max(max_seq, cm.seq);
    }
    sp.ckpt_seq = std::max(sp.ckpt_seq, max_seq);
    if (st.rs_shared) {
        st.tier_id = 0;
        st.tier_pages = 0;
    } else {
        st.tier_id = en.id;
        st.tier_pages = en.n_tok / kv_page;
        en.owner = id;
    }
    const double ms = msSince(r->t0);
    if (r->from_ssd) t->stats.restores_ssd += 1;
    else t->stats.restores_ram += 1;
    t->stats.restore_bytes += r->bytes;
    std::uint64_t main_bytes = 0;
    for (std::size_t i = 0; i < r->n_main; ++i) main_bytes += r->pieces[i].len;
    if (cfg_.verify) {
        try {
            const auto v = t->verifyPieces(std::span<const tier::Piece>(r->pieces.data(), r->n_main));
            if (v.bad > 0)
                logE("req {} | slot {} | kv tier verify: restore of {} tok: {} of {} pieces differ from the entry", job_id,
                     id, en.n_tok, v.bad, r->n_main);
            else
                logI("req {} | slot {} | kv tier verify: restore of {} tok byte-identical ({:.1f} MiB)", job_id, id,
                     en.n_tok, static_cast<double>(v.bytes) / 1048576.0);
        } catch (const std::exception& ex) {
            logE("req {} | slot {} | kv tier verify: restore of {} tok: error ({})", job_id, id, en.n_tok, ex.what());
        }
    }
    logI("req {} | slot {} | kv tier: restored {} tok from {} in {:.1f} ms ({:.2f} GB/s){}", job_id, id, en.n_tok,
         r->from_ssd ? "SSD" : "RAM", ms, static_cast<double>(main_bytes) / (ms * 1e6),
         r->n_main < r->pieces.size() ? "; other checkpoints follow" : "");
    // item S: a restored shared prefix entry becomes a VRAM shared checkpoint again
    if (st.rs_shared && en.nck >= 1) {
        for (Ckpt& c : sp.ckpts) {
            if (!c.valid || ckKindCode(c.kind) < 4) continue;
            try {
                steps.share(c, en.id);
            } catch (const std::exception& ex) {
                logW("req {} | slot {} | shared prefix: not kept from the restored entry: {}", job_id, id, ex.what());
            }
            break;
        }
    }
}

std::uint32_t TierBridge::queuedReuse(const SlotTier& st, std::span<const std::uint32_t> toks) const {
    const tier::Tier* t = tier_;
    tier::Restore* r = st.restore;
    if (!t || !r) return 0;
    const cache::Tier::Entry x = ps_.tierView()->find(r->entry_id);
    if (!x) return 0;
    const auto [reuse, kind] =
        ps_.config().cold_resume ? ps_.cache().matchTierEntry(x, toks, [&](std::uint32_t p, std::uint8_t k) { return ps_.resumeOk(toks, p, k); })
                                 : ps_.cache().matchTierEntry(x, toks);
    if (reuse < t->cfg.min_tokens) return 0;
    if (kind >= 4 && !ps_.onSchedule(toks, reuse)) return 0;
    return reuse;
}

std::optional<tier::Match> TierBridge::prefetchCandidate(std::span<const std::uint32_t> toks) {
    tier::Tier* t = tier_;
    if (!t) return std::nullopt;
    const auto mt = ps_.config().cold_resume
                        ? t->lookup(toks, [&](std::uint32_t p, std::uint8_t k) { return ps_.resumeOk(toks, p, k); })
                        : t->lookup(toks);
    if (!mt) return std::nullopt;
    tier::Entry* x = mt->e;
    if (x->in_ram || x->loading || x->io_busy) return std::nullopt;
    if (x->ck[mt->ck].kind >= 4 && !ps_.onSchedule(toks, mt->reuse)) return std::nullopt;
    return mt;
}

void TierBridge::prefetch(std::uint64_t job_id, tier::Entry* x) {
    bool ok = false;
    try {
        ok = tier_->prefetch(x);
    } catch (const std::exception&) {
        ok = false;
    }
    if (ok) logI("req {} | kv tier: prefetching {} tok from SSD into RAM while queued", job_id, x->n_tok);
}

bool TierBridge::tick() {
    tier::Tier* t = tier_;
    if (!t) return false;
    bool busy = t->tick();
    // only waiting for write delays (minutes): not polled every few ms; the idle loop sleeps until due
    if (t->waitOnly()) busy = false;
    pub_ssd_written_.store(t->stats.ssd_write_bytes, std::memory_order_relaxed);
    pub_ssd_dedup_.store(t->stats.ssd_dedup_bytes, std::memory_order_relaxed);
    pub_ssd_used_.store(t->ssd_bytes, std::memory_order_relaxed);
    pub_superseded_.store(t->stats.superseded, std::memory_order_relaxed);
    pub_ssd_entries_.store(t->ssdEntries(), std::memory_order_relaxed);
    return busy;
}

TierBridge::Published TierBridge::published() const {
    Published p;
    p.ssd_written = pub_ssd_written_.load(std::memory_order_relaxed);
    p.ssd_dedup = pub_ssd_dedup_.load(std::memory_order_relaxed);
    p.ssd_used = pub_ssd_used_.load(std::memory_order_relaxed);
    p.ssd_entries = pub_ssd_entries_.load(std::memory_order_relaxed);
    p.superseded = pub_superseded_.load(std::memory_order_relaxed);
    return p;
}

void TierBridge::logStats(std::uint64_t cow_pages, std::uint64_t q_waits) const {
    const tier::Tier* t = tier_;
    if (!t) return;
    const tier::Stats& s = t->stats;
    logI("kv tier | RAM {:.2f} / {:.2f} GiB in {} entries, SSD {:.2f} GiB in {} entries | spills {} ({:.0f} MiB), restores RAM "
         "{} / SSD {} ({:.0f} MiB), SSD writes {} ({:.0f} MiB written since start, {:.0f} MiB shared instead of written), RAM "
         "evictions {}, dropped {}, superseded {} (demoted {}), spill waits {}, pages copied on write {}, quarantine waits {}",
         static_cast<double>(t->ramUsedBytes()) / 1073741824.0, static_cast<double>(t->cfg.ram_bytes) / 1073741824.0,
         t->countRam(), static_cast<double>(t->ssd_bytes) / 1073741824.0, t->ssdEntries(), s.spills,
         static_cast<double>(s.spill_bytes) / 1048576.0, s.restores_ram, s.restores_ssd,
         static_cast<double>(s.restore_bytes) / 1048576.0, s.ssd_writes, static_cast<double>(s.ssd_write_bytes) / 1048576.0,
         static_cast<double>(s.ssd_dedup_bytes) / 1048576.0, s.ram_evictions, s.dropped, s.superseded, s.demoted, spill_waits_, cow_pages, q_waits);
}

}  // namespace whirl::server
