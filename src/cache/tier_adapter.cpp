// SPDX-License-Identifier: Apache-2.0

#include "cache/tier_adapter.h"

#include "tier/kv_tier.h"

namespace whirl::cache {

namespace {

class TierAdapter final : public Tier {
public:
    explicit TierAdapter(tier::Tier& t) : t_(t) {}
    std::uint32_t minTokens() const override { return t_.cfg.min_tokens; }
    std::size_t entryCount() const override { return t_.entries.size(); }
    Entry entryAt(std::size_t i) const override { return t_.entries[i]; }
    Entry find(std::uint64_t id) const override { return t_.find(id); }
    std::uint64_t id(Entry e) const override { return x(e).id; }
    Tokens tokens(Entry e) const override { return x(e).tokens; }
    std::uint32_t nTok(Entry e) const override { return x(e).n_tok; }
    std::uint32_t nck(Entry e) const override { return x(e).nck; }
    TierCk ck(Entry e, std::uint32_t k) const override {
        const tier::CkMeta& c = x(e).ck[k];
        return {c.pos, c.valid, c.has_logits, c.kind};
    }

private:
    static const tier::Entry& x(Entry e) { return *static_cast<const tier::Entry*>(e); }
    tier::Tier& t_;
};

}  // namespace

std::unique_ptr<Tier> makeTierAdapter(tier::Tier& t) { return std::make_unique<TierAdapter>(t); }

}  // namespace whirl::cache
