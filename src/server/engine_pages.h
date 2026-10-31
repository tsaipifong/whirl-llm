// Server engine, module server.slot_cache (arch025-engine S6; progress/arch025-engine.DESIGN.md v3 2.3 L2, 2.5, 4 S6):
// SlotPagesCounters, the shared prefix checkpoint and KV page counters - the former Engine members stat_spe_new_ ..
// stat_evictions_ (engine.h:339-340 at 421e22b), declared here verbatim, in that order, with those initial values.
// Engine holds it as cnt_ where stat_spe_new_ was: six uint64, 8-aligned, no padding, so every Engine member keeps its
// offset (pinned by tests/iso/server.slot_cache and tests/pins/engine_member_order.txt). The members are private
// (DESIGN v3 1.5 A1): the slot_cache functions in engine_pages.cpp change them through one write method per former
// write statement (the method body is that statement), Engine::stats reads them through the const reads. Written and
// read on the engine thread only.
// arch025-engine S7: SlotPages, the KV pages and checkpoints of the slots - the former Engine member functions saveCkpt
// .. waitsForSys, pagesFor, poolAvailable, initPool (definitions in engine_pages.cpp, bodies verbatim) and the
// header-inline kvTokBytes (declarations moved verbatim from engine.h:169-184 / 238-241 / 259-263 at f07e6a0), plus
// cacheCounters() for Engine::stats. An aggregate of references only (DESIGN v3 2.1 rules 1 / 2, 3.1): the core.shared
// members it uses, under their own names, slot_cache's placed state that Engine keeps where it was (kv_arrays_,
// conv_bytes_, ssm_bytes_, cnt_) and the drafter context dc_. Engine holds it as sc_, its last member (2.5).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "cache/page_pool.h"         // PagePool
#include "server/engine_common.h"   // kv_page, KvPoolFull (server.engine_types)
#include "server/engine_drafter.h"  // DrafterCtx (server.drafter_ctx); ServerModel, EngineOptions, Slot, Spe, Job,
                                    // CkptStore, PrefixStore, TierBridge through its headers

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace whirl::server {

class SlotPagesCounters {
public:
    // writes (engine_pages.cpp: evictFor, cowPages (not when spilling), dropSpe, createSpe, attachSpe x 2)
    void noteEviction() { stat_evictions_ += 1; }
    void noteUnshare() { stat_unshare_ += 1; }
    void noteSpeEvict() { stat_spe_evict_ += 1; }
    void noteSpeNew() { stat_spe_new_ += 1; }
    void noteSpeHit() { stat_spe_hits_ += 1; }
    void noteSpeTok(std::uint32_t n) { stat_spe_tok_ += n; }
    // reads (Engine::stats)
    std::uint64_t speNew() const { return stat_spe_new_; }
    std::uint64_t speHits() const { return stat_spe_hits_; }
    std::uint64_t speTok() const { return stat_spe_tok_; }
    std::uint64_t speEvict() const { return stat_spe_evict_; }
    std::uint64_t unshares() const { return stat_unshare_; }
    std::uint64_t evictions() const { return stat_evictions_; }

private:
    std::uint64_t stat_spe_new_ = 0, stat_spe_hits_ = 0, stat_spe_tok_ = 0, stat_spe_evict_ = 0, stat_unshare_ = 0;
    std::uint64_t stat_evictions_ = 0;
};

struct SlotPages {
    // checkpoints
    void saveCkpt(Slot& sl, std::uint32_t pos, const char* kind, std::optional<DevPtr> hid, std::optional<DevPtr> logits);
    void dropCache(Slot& sl);
    void freePages(Slot& sl);
    // pages (arch024 M3: the bookkeeping is pool_'s; these pass the slot's pages and the evictor)
    struct SlotEvictor;  // PagePool::Evictor over the idle slots and shared checkpoints (prefix cache's choice)
    bool evictFor(Slot* sl);
    std::optional<std::int32_t> takePage(Slot* sl);
    void cowPages(Slot& sl, std::uint32_t reuse, bool spill);
    bool ensurePages(Slot& sl, std::uint64_t n_tok);
    // shared prefix checkpoints (the table, the lookups and the schedule: ps_)
    void dropSpe(Spe& s);
    void createSpe(Slot& sl, std::span<const std::uint32_t> toks_all, std::uint32_t pos, const char* kind,
                   std::optional<DevPtr> hid_row, const Ckpt* src, std::uint64_t tier_id);
    void attachSpe(Slot& sl, Spe& s);
    bool waitsForSys(Job& job);
    // FR-SRV P2-3: pool pages a prompt of n_tok tokens needs; pages free or evictable now minus
    // what the running prefills still need
    std::size_t pagesFor(std::uint64_t n_tok) const;
    std::size_t poolAvailable() const;
    // ... and, once the model's KV pool exists, the page bookkeeping.
    void initPool();
    std::uint64_t kvTokBytes() const {
        std::uint64_t b = 0;
        for (const KvArr& a : kv_arrays_) b += a.row;
        return b;
    }
    // the counters for Engine::stats (DESIGN v3 2.2: the core reads no component data)
    const SlotPagesCounters& cacheCounters() const { return cnt_; }

    CkptStore& ck_;
    ServerModel& m_;
    device::Backend& ops_;
    const EngineOptions& opt_;
    PagePool& pool_;
    PrefixStore& ps_;
    std::vector<Slot>& slots_;
    TierBridge& tb_;
    std::vector<KvArr>& kv_arrays_;
    const std::uint64_t& conv_bytes_;
    const std::uint64_t& ssm_bytes_;
    SlotPagesCounters& cnt_;
    DrafterCtx& dc_;
};

}  // namespace whirl::server
