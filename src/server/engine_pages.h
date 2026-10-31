// Server engine, module server.slot_cache (arch025-engine S6; progress/arch025-engine.DESIGN.md v3 2.3 L2, 2.5, 4 S6):
// SlotPagesCounters, the shared prefix checkpoint and KV page counters - the former Engine members stat_spe_new_ ..
// stat_evictions_ (engine.h:339-340 at 421e22b), declared here verbatim, in that order, with those initial values.
// Engine holds it as cnt_ where stat_spe_new_ was: six uint64, 8-aligned, no padding, so every Engine member keeps its
// offset (pinned by tests/iso/server.slot_cache and tests/pins/engine_member_order.txt). The members are private
// (DESIGN v3 1.5 A1): the slot_cache functions in engine_pages.cpp change them through one write method per former
// write statement (the method body is that statement), Engine::stats reads them through the const reads. Written and
// read on the engine thread only. S7 adds SlotPages (sc_) to this header.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

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

}  // namespace whirl::server
