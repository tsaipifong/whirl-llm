// Prefix cache decisions (moved out of src/server/engine.cpp, refactor R-1).
// SPDX-License-Identifier: Apache-2.0

#include "cache/prefix_cache.h"

#include <algorithm>
#include <limits>

namespace whirl::cache {

std::size_t lcpLen(Tokens a, Tokens b) {
    const std::size_t n = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

bool prefixEq(Tokens a, Tokens b) { return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin()); }

Supersede supersedes(Tokens old_t, Tokens new_t, std::size_t sys, const SupersedeRule& r) {
    Supersede s;
    const std::size_t lo = old_t.size(), ln = new_t.size();
    if (sys == 0 || lo == 0 || ln == 0) return s;
    const std::size_t l = lcpLen(old_t, new_t);
    if (l < sys) return s;  // not the same system prompt
    if (static_cast<double>(ln) > r.shorter * static_cast<double>(lo)) return s;
    const std::size_t tail = lo - l;
    const double need = std::max(static_cast<double>(r.min_tail), r.tail_frac * static_cast<double>(lo));
    if (static_cast<double>(tail) < need) return s;  // near-end divergence: retry / regenerate / edit
    s.yes = true;
    s.keep = l;
    return s;
}

void SupersedeTracker::mark(std::uint64_t new_id, std::uint64_t old_id, std::size_t keep, std::uint64_t stamp) {
    for (Item& it : items_) {
        if (it.new_id == new_id && it.p.old_id == old_id) {
            it.p = {old_id, keep, stamp};
            return;
        }
    }
    if (items_.size() >= 256) items_.erase(items_.begin());
    items_.push_back({new_id, {old_id, keep, stamp}});
}

std::vector<SupersedeTracker::Pending> SupersedeTracker::take(std::uint64_t new_id) {
    std::vector<Pending> v;
    std::size_t w = 0;
    for (std::size_t i = 0; i < items_.size(); ++i) {
        if (items_[i].new_id == new_id) v.push_back(items_[i].p);
        else items_[w++] = items_[i];
    }
    items_.resize(w);
    return v;
}

void SupersedeTracker::forget(std::uint64_t id) {
    std::erase_if(items_, [&](const Item& it) { return it.new_id == id || it.p.old_id == id; });
}

namespace {

class PrefixCacheImpl final : public PrefixCache {
public:
    explicit PrefixCacheImpl(const Config& cfg) : cfg_(cfg) {}

    const Config& config() const override { return cfg_; }
    void attachTier(const Tier* t) override { tier_ = t; }
    const Tier* tier() const override { return tier_; }
    std::uint64_t touch() override { return ++use_seq_; }

    SlotMatch matchSlot(Tokens cached, View<const CkptKey> cks, Tokens toks) const override {
        if (!cfg_.enabled) return {};
        const std::size_t lcp = lcpLen(cached, toks);
        int ck = -1;
        for (std::size_t i = 0; i < cks.size(); ++i) {
            const CkptKey& c = cks[i];
            if (!c.valid || c.pos > lcp || c.pos > toks.size() || c.pos == 0) continue;
            if (c.pos == toks.size() && !c.has_logits) continue;
            if (ck < 0 || c.pos > cks[static_cast<std::size_t>(ck)].pos) ck = static_cast<int>(i);
        }
        return {ck >= 0 ? cks[static_cast<std::size_t>(ck)].pos : 0, ck, lcp};
    }

    int matchShared(View<const SpeKey> spes, Tokens toks, const PosPred& on_schedule) const override {
        if (!cfg_.enabled) return -1;
        int best = -1;
        for (std::size_t i = 0; i < spes.size(); ++i) {
            const SpeKey& s = spes[i];
            if (!s.valid || s.n >= toks.size()) continue;
            if (best >= 0 && s.n <= spes[static_cast<std::size_t>(best)].n) continue;
            if (!prefixEq(s.tokens, toks.subspan(0, s.n))) continue;
            if (!on_schedule(s.n)) continue;
            best = static_cast<int>(i);
        }
        return best;
    }

    int findShared(View<const SpeKey> spes, Tokens prefix) const override {
        for (std::size_t i = 0; i < spes.size(); ++i) {
            const SpeKey& s = spes[i];
            if (s.valid && s.n == prefix.size() && prefixEq(s.tokens, prefix)) return static_cast<int>(i);
        }
        return -1;
    }

    TierResume matchTierEntry(Tier::Entry e, Tokens toks) const override {
        const Tier& t = *tier_;
        const Tokens all = t.tokens(e);
        const Tokens et = all.subspan(0, std::min<std::size_t>(t.nTok(e), all.size()));
        const std::size_t lcp = lcpLen(et, toks);
        TierResume r;
        const std::uint32_t nck = t.nck(e);
        for (std::uint32_t k = 0; k < nck; ++k) {
            const TierCk c = t.ck(e, k);
            if (c.valid == 0 || c.pos == 0 || c.pos > lcp || c.pos > toks.size()) continue;
            if (c.pos == toks.size() && c.has_logits == 0) continue;
            if (c.pos > r.reuse) {
                r.reuse = c.pos;
                r.kind = c.kind;
            }
        }
        return r;
    }

    std::uint64_t findTierShared(Tokens prefix) const override {
        if (!tier_) return 0;
        const Tier& t = *tier_;
        const std::size_t n = prefix.size();
        for (std::size_t i = 0; i < t.entryCount(); ++i) {
            const Tier::Entry e = t.entryAt(i);
            if (t.nck(e) != 1) continue;
            const TierCk c = t.ck(e, 0);
            if (c.valid == 0 || c.pos != n || t.nTok(e) != n) continue;
            const Tokens et = t.tokens(e);
            if (et.size() < n || !std::equal(prefix.begin(), prefix.end(), et.begin())) continue;
            return t.id(e);
        }
        return 0;
    }

    std::uint32_t lcpTarget(View<const SlotCache> slots, View<const SpeKey> spes, Tokens toks, std::uint32_t reuse,
                            std::uint32_t sys, const FloorFn& floor) const override {
        if (!cfg_.lcp_on || spes.empty() || !cfg_.enabled) return 0;
        std::size_t L = 0;
        for (std::size_t i = 0; i < slots.size(); ++i) L = std::max(L, lcpLen(slots[i].cache_tokens, toks));
        for (std::size_t i = 0; i < spes.size(); ++i)
            if (spes[i].valid) L = std::max(L, lcpLen(spes[i].tokens, toks));
        if (tier_)
            for (std::size_t i = 0; i < tier_->entryCount(); ++i) L = std::max(L, lcpLen(tier_->tokens(tier_->entryAt(i)), toks));
        const std::size_t mn = cfg_.sys_min > 0 ? cfg_.sys_min : 2048;
        if (L < mn || L < 2) return 0;
        const std::size_t a = floor(std::min(L, toks.size() - 1));
        if (a < mn || a == sys || a < reuse + static_cast<std::size_t>(cfg_.kv_page) * 4) return 0;
        return static_cast<std::uint32_t>(a);
    }

    std::size_t ckptSlotFor(View<const CkptKey> cks, std::uint32_t pos) const override {
        std::size_t slot = 0;
        for (std::size_t i = 0; i < cks.size(); ++i) {
            const CkptKey& c = cks[i];
            if (c.valid && c.pos == pos) {
                slot = i;
                break;
            }
            if (!c.valid) {
                if (cks[slot].valid) slot = i;
            } else if (cks[slot].valid && c.seq < cks[slot].seq) {
                slot = i;
            }
        }
        return slot;
    }

    int sharedVictim(View<const SpeKey> spes) const override {
        int v = -1;
        for (std::size_t i = 0; i < spes.size(); ++i) {
            const SpeKey& s = spes[i];
            if (s.pin) continue;
            if (!s.valid) return static_cast<int>(i);
            if (v < 0 || s.last_used < spes[static_cast<std::size_t>(v)].last_used) v = static_cast<int>(i);
        }
        return v;
    }

    Victim evictVictim(View<const SlotCache> slots, const SlotPred& evictable, View<const SpeKey> spes) const override {
        Victim v;
        std::uint64_t lu = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (!evictable(i)) continue;
            if (slots[i].last_used < lu) {
                v = {Victim::slot, i};
                lu = slots[i].last_used;
            }
        }
        for (std::size_t i = 0; i < spes.size(); ++i) {
            const SpeKey& s = spes[i];
            if (!s.valid || s.pin) continue;
            if (s.last_used < lu) {
                v = {Victim::spe, i};
                lu = s.last_used;
            }
        }
        return v;
    }

private:
    Config cfg_;
    const Tier* tier_ = nullptr;
    std::uint64_t use_seq_ = 0;
};

}  // namespace

std::unique_ptr<PrefixCache> makePrefixCache(const Config& cfg) { return std::make_unique<PrefixCacheImpl>(cfg); }

}  // namespace whirl::cache
