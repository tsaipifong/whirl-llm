// KV pool pages of the server (arch024 M3): which pages are free, how many holders each page has (a shared prefix
// is held by several slots / shared checkpoints), the pages a RAM tier spill still reads (quarantine), copy-on-write
// of a shared or spilling tail, and growing a page list for more tokens.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL. Moved out of Engine (initPool's page part, takeFreePage / refPage / unrefPages / sharedFrom /
// copyPage / takePage / quarantinePages / reclaimPages / cowPages / ensurePages, pagesFor / poolAvailable's page
// part) without a change of order, copies or counters. Prior art: progress/arch024-S4.PRIOR_ART.md (vLLM BlockPool
// ref_cnt + free queue, llama.cpp find_slot, SGLang lock_ref eviction).
//
// PagePool does not know Slot or Spe: when no page is free it asks an Evictor (the engine's prefix cache picks an
// idle slot or a shared checkpoint and frees its pages) and tries again. Quarantined pages come back only after
// their event (recorded on the tier stream after the spill copies) is done, or - reclaim(true) - with the model
// stream ordered after it on the GPU (streamWaitEvent: the page is reused in stream order, not freed on the host).

#pragma once

#include "device/backend.h"
#include "device/handles.h"
#include "server/backend.h"
#include "server/engine_thread.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace whirl::server {

struct QBatch {
    std::vector<std::int32_t> pages;
    device::EventH ev;
};

class PagePool {
public:
    // frees the pages of one holder chosen by the caller; false: nothing left to evict
    class Evictor {
    public:
        virtual bool evictOne() = 0;

    protected:
        ~Evictor() = default;
    };
    struct Stats {
        std::uint64_t cow_pages = 0;  // pages a COW left to a spill (quarantined)
        std::uint64_t q_waits = 0;    // quarantine batches the model stream waited for (reclaim(true))
    };
    enum class Cow { unchanged, no_page, done };

    PagePool(device::Backend& ops, ServerModel& m) : ops_(ops), m_(m) {}
    // arch024 M8: the engine thread that may call the entry points below (nullptr: unchecked; docs/dev/threads.md)
    void setOwner(const EngineThread* o) { owner_ = o; }
    const EngineThread* ownerThread() const { return owner_; }

    // n pages, every one free (taken in increasing order), no holder; arrs: the pool arrays a page copy covers
    void init(std::uint32_t n, std::vector<KvArr> arrs);
    std::uint32_t pages() const { return pages_; }
    std::size_t freeCount() const { return free_.size(); }
    std::size_t batches() const { return quarantine_.size(); }
    bool quarantineEmpty() const { return quarantine_.empty(); }
    std::uint16_t refCount(std::int32_t p) const { return ref_[static_cast<std::size_t>(p)]; }
    const Stats& stats() const { return stats_; }

    // a free page with one holder (nullopt: none free)
    std::optional<std::int32_t> takeFree();
    // a page: free, else reclaim the done quarantine, else ev.evictOne(), else wait (on the GPU) for the quarantine
    std::optional<std::int32_t> take(Evictor& ev);
    void ref(std::int32_t p) {
        ownerCheck(owner_, "PagePool::ref");
        ref_[static_cast<std::size_t>(p)] += 1;
    }
    // one holder fewer each; free at 0
    void unref(std::span<const std::int32_t> pages);
    // a page from index (reuse - 1) / kv_page on has another holder
    bool sharedFrom(std::span<const std::int32_t> pages, std::uint32_t reuse) const;
    // page src into page dst in every pool array, on the model stream
    void copy(std::int32_t dst, std::int32_t src);
    // the stream quarantine records its event on (the RAM tier's stream); nullptr: no tier (quarantine throws)
    void setQuarantineStream(device::Stream s) {
        ownerCheck(owner_, "PagePool::setQuarantineStream");
        q_stream_ = s;
    }
    // pages a spill still reads: held until an event recorded now on the quarantine stream is done
    void quarantine(std::span<const std::int32_t> pages);
    // the quarantine batches whose event is done back (one holder fewer each); force: all of them, the model stream
    // ordered after each pending event first
    void reclaim(bool force);
    // copy-on-write of pages[(reuse - 1) / kv_page ...] (seq: its sequence for mapPages): the first of them copied to
    // a new page when reuse > 0, the rest released (spill: quarantined). no_page: no new page (pages unchanged).
    Cow cow(std::vector<std::int32_t>& pages, std::uint32_t seq, std::uint32_t reuse, bool spill, Evictor& ev);
    // pages for n_tok tokens of sequence seq (mapped); false: fewer (the pages taken so far stay in pages)
    bool ensure(std::vector<std::int32_t>& pages, std::uint32_t seq, std::uint64_t n_tok, Evictor& ev);
    // pages of n_tok tokens
    static std::size_t pagesFor(std::uint64_t n_tok) { return static_cast<std::size_t>((n_tok + qwen35::kv_page - 1) / qwen35::kv_page); }
    // free pages plus quarantined ones
    std::size_t reclaimable() const;
    // feat026-pidle-r2 (review r1-batch-b): free pages plus the quarantined ones whose every holder is the
    // quarantine (a quarantined page another slot or a shared checkpoint still holds never becomes free)
    std::size_t freeable() const;
    // pages of the list with no other holder
    std::size_t soleOwned(std::span<const std::int32_t> pages) const;

private:
    device::Backend& ops_;
    ServerModel& m_;
    device::Stream q_stream_ = nullptr;
    std::uint32_t pages_ = 0;
    std::vector<std::int32_t> free_;
    std::vector<std::uint16_t> ref_;
    std::vector<QBatch> quarantine_;
    std::vector<KvArr> arrs_;
    Stats stats_;
    const EngineThread* owner_ = nullptr;
};

}  // namespace whirl::server
