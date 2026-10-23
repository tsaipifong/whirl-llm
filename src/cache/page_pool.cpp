// KV pool pages (arch024 M3); see page_pool.h.
// SPDX-License-Identifier: Apache-2.0

#include "cache/page_pool.h"

#include <stdexcept>

namespace whirl::server {

using qwen35::kv_page;

void PagePool::init(std::uint32_t n, std::vector<KvArr> arrs) {
    ownerCheck(owner_, "PagePool::init");
    pages_ = n;
    ref_.assign(pages_, 0);
    free_.clear();
    free_.reserve(pages_);
    // every page free (popped in increasing order)
    for (std::uint32_t pg = pages_; pg > 0;) {
        --pg;
        free_.push_back(static_cast<std::int32_t>(pg));
    }
    arrs_ = std::move(arrs);
}

std::optional<std::int32_t> PagePool::takeFree() {
    ownerCheck(owner_, "PagePool::takeFree");
    if (free_.empty()) return std::nullopt;
    const std::int32_t p = free_.back();
    free_.pop_back();
    ref_[static_cast<std::size_t>(p)] = 1;
    return p;
}

void PagePool::unref(std::span<const std::int32_t> pages) {
    ownerCheck(owner_, "PagePool::unref");
    for (std::int32_t p : pages) {
        std::uint16_t& r = ref_[static_cast<std::size_t>(p)];
        r -= 1;
        if (r == 0) free_.push_back(p);
    }
}

bool PagePool::sharedFrom(std::span<const std::int32_t> pages, std::uint32_t reuse) const {
    const std::size_t keep = reuse == 0 ? 0 : (reuse - 1) / kv_page;
    if (keep >= pages.size()) return false;
    for (std::size_t i = keep; i < pages.size(); ++i)
        if (ref_[static_cast<std::size_t>(pages[i])] > 1) return true;
    return false;
}

void PagePool::copy(std::int32_t dst_p, std::int32_t src_p) {
    ownerCheck(owner_, "PagePool::copy");
    const std::uint64_t src = static_cast<std::uint64_t>(src_p);
    const std::uint64_t dst = static_cast<std::uint64_t>(dst_p);
    for (const KvArr& a : arrs_) {
        const std::uint64_t len = static_cast<std::uint64_t>(kv_page) * a.row;
        ops_.copyAsync(a.base + dst * len, a.base + src * len, len, m_.stream());
    }
}

std::optional<std::int32_t> PagePool::take(Evictor& ev) {
    ownerCheck(owner_, "PagePool::take");
    for (;;) {
        if (free_.empty() && !quarantine_.empty()) reclaim(false);
        if (auto p = takeFree()) return p;
        if (ev.evictOne()) continue;
        // only pages a spill still reads: wait for it on the GPU
        if (!quarantine_.empty()) {
            reclaim(true);
            continue;
        }
        return std::nullopt;
    }
}

void PagePool::quarantine(std::span<const std::int32_t> pages) {
    ownerCheck(owner_, "PagePool::quarantine");
    if (pages.empty()) return;
    if (q_stream_ == nullptr) throw std::runtime_error("NoTier");
    device::EventH ev = device::EventH::create(ops_);
    ops_.eventRecord(ev.get(), q_stream_);  // a throw destroys ev
    quarantine_.push_back({std::vector<std::int32_t>(pages.begin(), pages.end()), std::move(ev)});
}

void PagePool::reclaim(bool force) {
    ownerCheck(owner_, "PagePool::reclaim");
    std::size_t i = 0;
    while (i < quarantine_.size()) {
        QBatch& q = quarantine_[i];
        bool done = ops_.eventDone(q.ev.get());
        if (!done && force) {
            stats_.q_waits += 1;
            try {
                ops_.streamWaitEvent(m_.stream(), q.ev.get());
            } catch (const std::exception&) {
            }
            done = true;
        }
        if (!done) {
            ++i;
            continue;
        }
        unref(q.pages);
        quarantine_.erase(quarantine_.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

PagePool::Cow PagePool::cow(std::vector<std::int32_t>& pages, std::uint32_t seq, std::uint32_t reuse, bool spill,
                            Evictor& ev) {
    ownerCheck(owner_, "PagePool::cow");
    const std::size_t keep = reuse == 0 ? 0 : (reuse - 1) / kv_page;
    if (keep >= pages.size()) return Cow::unchanged;
    std::optional<std::int32_t> newp;
    if (reuse > 0) {
        newp = take(ev);
        if (!newp) return Cow::no_page;
        copy(*newp, pages[keep]);
    }
    const std::span<const std::int32_t> old(pages.data() + keep, pages.size() - keep);
    if (spill) quarantine(old);
    else unref(old);
    if (spill) stats_.cow_pages += pages.size() - keep;
    pages.resize(keep);
    if (newp) {
        pages.push_back(*newp);
        m_.mapPages(seq, static_cast<std::uint32_t>(keep), std::span<const std::int32_t>(pages.data() + keep, 1));
    }
    return Cow::done;
}

bool PagePool::ensure(std::vector<std::int32_t>& pages, std::uint32_t seq, std::uint64_t n_tok, Evictor& ev) {
    ownerCheck(owner_, "PagePool::ensure");
    const std::size_t need = pagesFor(n_tok);
    if (need <= pages.size()) return true;
    if (need > m_.seqPages()) return false;
    const std::size_t first = pages.size();
    pages.reserve(need);
    while (pages.size() < need) {
        auto p = take(ev);
        if (!p) break;
        pages.push_back(*p);
    }
    m_.mapPages(seq, static_cast<std::uint32_t>(first),
                std::span<const std::int32_t>(pages.data() + first, pages.size() - first));
    return pages.size() >= need;
}

std::size_t PagePool::reclaimable() const {
    std::size_t n = free_.size();
    for (const QBatch& q : quarantine_) n += q.pages.size();
    return n;
}

std::size_t PagePool::soleOwned(std::span<const std::int32_t> pages) const {
    std::size_t n = 0;
    for (std::int32_t pg : pages)
        if (ref_[static_cast<std::size_t>(pg)] == 1) ++n;
    return n;
}

}  // namespace whirl::server
