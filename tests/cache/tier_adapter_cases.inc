// arch025-n8b2 (PLAN N8 batch 2): cases of cache.tier_adapter (src/cache/tier_adapter.cpp: the prefix cache's cache::Tier view of the
// KV host tier tier::Tier). Written new (no case of the adapter existed: its pins were the server test testTierBridgeRestore); the
// tier is the REAL tier.kv over the NullBackend (host memory), entries made by Tier::newEntry. Included by tests/iso/cache.tier_adapter/.
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "cache/tier_adapter.h"
#include "device/null_backend.h"
#include "tier/kv_tier.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace tier_adapter_cases {

namespace tk = whirl::tier;
namespace wc = whirl::cache;

struct Fixture {
    whirl::device::NullBackend ops;
    std::unique_ptr<tk::Tier> tier = tk::Tier::create(ops, 8 * tk::block_bytes, 42, 0);
    tk::Entry* a = nullptr;
    tk::Entry* b = nullptr;
    Fixture() {
        a = tier->newEntry(2);
        a->tokens = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
        a->n_tok = 8;  // tokens may hold more than the stored prefix: the adapter reports n_tok, not tokens.size()
        a->ck[0] = {.pos = 4, .valid = 1, .has_logits = 0, .kind = 2};
        a->ck[1] = {.pos = 8, .valid = 0, .has_logits = 1, .kind = 3};
        b = tier->newEntry(1);
        b->tokens = {7, 8, 9};
        b->n_tok = 3;
        b->ck[0] = {.pos = 3, .valid = 1, .has_logits = 1, .kind = 1};
    }
};

void testTierAdapterEntries() {
    Fixture f;
    const auto t = wc::makeTierAdapter(*f.tier);
    CHECK(t->entryCount() == 2);
    CHECK(t->entryAt(0) == f.a && t->entryAt(1) == f.b);  // the tier's order (the prefix cache scans it front to back)
    CHECK(t->id(t->entryAt(0)) == f.a->id && t->id(t->entryAt(1)) == f.b->id);
    CHECK(t->find(f.b->id) == f.b && t->find(f.a->id) == f.a);
    CHECK(t->find(f.a->id + f.b->id + 2) == nullptr);  // ids are odd: the sum + 2 is even, no entry has it
    // a live view, not a copy: an entry made after the adapter is seen
    tk::Entry* c = f.tier->newEntry(0);
    CHECK(t->entryCount() == 3 && t->entryAt(2) == c && t->find(c->id) == c);
}

void testTierAdapterFields() {
    Fixture f;
    const auto t = wc::makeTierAdapter(*f.tier);
    const wc::Tier::Entry a = t->find(f.a->id);
    CHECK(t->nTok(a) == 8 && t->nck(a) == 2);
    const wc::Tokens ta = t->tokens(a);
    CHECK(ta.size() == 10 && ta[0] == 11 && ta[9] == 20 && ta.data() == f.a->tokens.data());  // no copy
    const wc::TierCk c0 = t->ck(a, 0), c1 = t->ck(a, 1);
    CHECK(c0.pos == 4 && c0.valid == 1 && c0.has_logits == 0 && c0.kind == 2);
    CHECK(c1.pos == 8 && c1.valid == 0 && c1.has_logits == 1 && c1.kind == 3);
    const wc::Tier::Entry b = t->find(f.b->id);
    const wc::TierCk cb = t->ck(b, 0);
    CHECK(t->nTok(b) == 3 && t->nck(b) == 1 && cb.pos == 3 && cb.valid == 1 && cb.has_logits == 1 && cb.kind == 1);
}

void testTierAdapterMinTokens() {
    Fixture f;
    const auto t = wc::makeTierAdapter(*f.tier);
    CHECK(t->minTokens() == tk::Config{}.min_tokens);  // 2048 before start()
    f.tier->cfg.min_tokens = 512;                       // read through the tier, every call
    CHECK(t->minTokens() == 512);
}

}  // namespace tier_adapter_cases
