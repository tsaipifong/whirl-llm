// In-memory cache::Tier for the prefix cache unit tests.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cache/prefix_cache.h"

#include <deque>

namespace whirl::cache::test {

struct MockEntry {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> tokens;
    std::uint32_t n_tok = 0;
    std::vector<TierCk> cks;
};

class MockTier final : public Tier {
public:
    std::uint32_t min_tokens = 2048;
    std::deque<MockEntry> entries;  // stable addresses

    MockEntry& add(std::uint64_t id, std::vector<std::uint32_t> toks, std::vector<TierCk> cks) {
        MockEntry e;
        e.id = id;
        e.n_tok = static_cast<std::uint32_t>(toks.size());
        e.tokens = std::move(toks);
        e.cks = std::move(cks);
        entries.push_back(std::move(e));
        return entries.back();
    }

    std::uint32_t minTokens() const override { return min_tokens; }
    std::size_t entryCount() const override { return entries.size(); }
    Entry entryAt(std::size_t i) const override { return &entries[i]; }
    Entry find(std::uint64_t id) const override {
        for (const MockEntry& e : entries)
            if (e.id == id) return &e;
        return nullptr;
    }
    std::uint64_t id(Entry e) const override { return x(e).id; }
    Tokens tokens(Entry e) const override { return x(e).tokens; }
    std::uint32_t nTok(Entry e) const override { return x(e).n_tok; }
    std::uint32_t nck(Entry e) const override { return static_cast<std::uint32_t>(x(e).cks.size()); }
    TierCk ck(Entry e, std::uint32_t k) const override { return x(e).cks[k]; }

private:
    static const MockEntry& x(Entry e) { return *static_cast<const MockEntry*>(e); }
};

}  // namespace whirl::cache::test
