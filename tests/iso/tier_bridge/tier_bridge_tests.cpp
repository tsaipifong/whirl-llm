// whirl-iso-tier-bridge-tests: src/server/tier_bridge.cpp built alone (ARCH-1 8.5, arch025-lane2 leg 4) with its declared
// dependencies: the prefix store (and what it declares), the checkpoint buffers through the narrow model interface
// (cache/cache_model.h, iso::FakeCacheModel), the KV host tier (tier.kv) on the NullBackend (host memory) and the owner
// check (threads). No engine, no ServerModel, no model, no HIP layer: the span order of a tier entry, a slot spill into a
// RAM-only tier, the restore pick (gain rule) and the published counters run on host memory.
// SPDX-License-Identifier: Apache-2.0
#include "cache/ckpt_store.h"
#include "device/null_backend.h"
#include "iso/fake_cache_model.h"
#include "iso/iso_test.h"
#include "server/prefix_store.h"
#include "server/tier_bridge.h"
#include "tier/kv_tier.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace {

namespace sv = whirl::server;
namespace dv = whirl::device;
namespace t = whirl::tier;

constexpr std::uint32_t KP = whirl::qwen35::kv_page;
constexpr std::uint64_t kConv = 100, kSsm = 300, kRow = 16, kPoolPages = 16;

// 4 layers (DeltaNet state in layers 1 and 3), n_embd 8, n_vocab 10; two KV pool arrays (K, V) of kPoolPages pages of KP
// (qwen35::kv_page = 256) tokens; a RAM-only tier (no SSD directory). Teardown in the product's order (TbRig in tests/server/server_tests.cpp): the tier's
// IO thread stops, then the bridge / prefix store / checkpoint store, the tier last.
struct Rig {
    dv::NullBackend nb;
    iso::FakeCacheModel m{nb, {false, true, false, true}, kConv, kSsm, 8, 10};
    std::vector<sv::KvArr> kv;
    whirl::chat::ChatTokens ct;
    std::unique_ptr<sv::PrefixStore> ps;
    std::unique_ptr<sv::CkptStore> cs;
    std::unique_ptr<t::Tier> tier;
    std::unique_ptr<sv::TierBridge> tb;
    explicit Rig(bool with_tier, const sv::TierBridge::Config& tc = {}) {
        for (int a = 0; a < 2; ++a) kv.push_back({nb.malloc(kPoolPages * KP * kRow), kRow});
        ps = std::make_unique<sv::PrefixStore>(ct, sv::PrefixStore::Config{});
        cs = std::make_unique<sv::CkptStore>(nb, m, sv::CkptStore::Layout{kConv, kSsm, false});
        tb = std::make_unique<sv::TierBridge>(nb, m, ct, *ps, kv, tc);
        if (!with_tier) return;
        tier = t::Tier::create(nb, 64ull << 20, 42, 0);
        t::Config c;
        c.ram_bytes = 64ull << 20;
        c.min_tokens = 64;
        c.ssd_delay_ms = 0;
        t::Layout lay;
        lay.ck_bytes = 2 * (kConv + kSsm) + 8 * 4 + 10 * 4;
        lay.page_bytes = 2 * kRow * KP;
        lay.page_tokens = KP;
        lay.tok_cap = 16384;
        tb->attach(tier.get());
        ps->attachTier(tier.get());
        tier->start(c, lay, 7);
    }
    ~Rig() {
        if (tier) tier->stopIo();
        tb.reset();
        ps.reset();
        cs.reset();
        tier.reset();
        for (const sv::KvArr& a : kv) nb.free(a.base);
    }
    // a slot holding toks, checkpoints at ck_pos (seq 1, 2, ...), pages first..
    struct Slot {
        sv::SlotPrefix sp;
        sv::SlotTier st;
    };
    std::unique_ptr<Slot> slot(const std::vector<std::uint32_t>& toks, const std::vector<std::uint32_t>& ck_pos, std::int32_t first) {
        auto s = std::make_unique<Slot>();
        s->sp.cache_tokens = toks;
        s->sp.ckpts.resize(ck_pos.size());
        for (std::size_t j = 0; j < ck_pos.size(); ++j) {
            sv::Ckpt& c = s->sp.ckpts[j];
            cs->init(c, true);
            c.valid = true;
            c.pos = ck_pos[j];
            c.seq = j + 1;
            c.kind = "prompt-end";
        }
        for (std::size_t p = 0; p < (toks.size() + KP - 1) / KP; ++p) s->sp.pages.push_back(first + static_cast<std::int32_t>(p));
        s->sp.ckpt_seq = 10;
        return s;
    }
};

std::vector<std::uint32_t> tokens(std::size_t n) {
    std::vector<std::uint32_t> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = static_cast<std::uint32_t>(1000 + (i * 7 + 1) % 5000);
    return v;
}

void testSpanOrder() {
    Rig r(false);
    sv::Ckpt c;
    r.cs->init(c, false);
    r.tb->spans().clear();
    r.tb->ckSpans(c, 500, 3);
    // per DeltaNet layer conv then ssm, then the hidden row (n_embd * 4) and the logits row (n_vocab * 4)
    const auto& s = r.tb->spans();
    ISO_CHECK(s.size() == 6);
    ISO_CHECK(s[0].dev == c.conv[1] && s[0].off == 500 && s[0].len == kConv && s[0].ck == 3);
    ISO_CHECK(s[1].dev == c.ssm[1] && s[1].off == 600 && s[1].len == kSsm);
    ISO_CHECK(s[2].dev == c.conv[3] && s[2].off == 900 && s[3].dev == c.ssm[3] && s[3].off == 1000);
    ISO_CHECK(s[4].dev == c.hid && s[4].off == 1300 && s[4].len == 32);
    ISO_CHECK(s[5].dev == c.logits && s[5].off == 1332 && s[5].len == 40);
    // one KV page: its rows of every pool array in order, untagged
    r.tb->spans().clear();
    r.tb->pageSpans(3, 1000);
    ISO_CHECK(s.size() == 2);
    ISO_CHECK(s[0].dev == r.kv[0].base + 3 * KP * kRow && s[0].off == 1000 && s[0].len == KP * kRow && s[0].ck == 0xff);
    ISO_CHECK(s[1].dev == r.kv[1].base + 3 * KP * kRow && s[1].off == 1000 + KP * kRow);
    r.cs->release(c, {});
}

void testNoTier() {
    Rig r(false);
    auto s = r.slot(tokens(3 * KP - 20), {KP - 50, 3 * KP - 20}, 5);
    r.tb->spillSlot(0, true, s->sp, s->st);
    ISO_CHECK(s->st.tier_id == 0 && r.tb->tier() == nullptr);
    ISO_CHECK(!r.tb->restoreMatch(s->sp, tokens(4 * KP)).has_value());
    ISO_CHECK(!r.tb->tick() && !r.tb->flushOnStop([] { return true; }) && !r.tb->idleDue().has_value());
    ISO_CHECK(r.tb->queuedReuse(s->st, tokens(4 * KP)) == 0);
}

void testSpillSlot() {
    Rig r(true);
    const auto toks = tokens(3 * KP - 20);
    auto s = r.slot(toks, {KP - 50, 3 * KP - 20, 2 * KP}, 5);
    // not idle: nothing
    r.tb->spillSlot(0, false, s->sp, s->st);
    ISO_CHECK(r.tier->stats.spills == 0 && s->st.tier_id == 0);
    // idle: one new RAM entry holding the slot's tokens up to its last checkpoint
    r.tb->spillSlot(0, true, s->sp, s->st);
    ISO_CHECK(r.tier->stats.spills == 1 && s->st.tier_id != 0 && r.tier->countRam() == 1);
    const t::Entry* x = r.tier->find(s->st.tier_id);
    ISO_CHECK(x != nullptr && x->n_tok == 3 * KP - 20 && x->pages == 3 && x->nck == 3);
    ISO_CHECK(static_cast<bool>(s->st.spill_ev) && s->st.spill_busy);
    r.tb->waitSpill(s->st);
    ISO_CHECK(!s->st.spill_busy);
    // a slot below the tier's minimum (64 tokens at its last checkpoint) is not spilled
    auto small = r.slot(tokens(40), {40}, 9);
    r.tb->spillSlot(1, true, small->sp, small->st);
    ISO_CHECK(r.tier->stats.spills == 1 && small->st.tier_id == 0);
}

void testRestoreMatchGain() {
    sv::TierBridge::Config tc;
    tc.min_gain = 64;
    Rig r(true, tc);
    const auto toks = tokens(3 * KP - 20);
    auto s = r.slot(toks, {KP - 50, 3 * KP - 20}, 5);
    r.tb->spillSlot(0, true, s->sp, s->st);
    ISO_CHECK(r.tier->stats.spills == 1);
    r.tb->waitSpill(s->st);
    // the same conversation one turn later, on an empty slot: the entry brings 3 KP - 20 tokens
    auto req = tokens(4 * KP);
    sv::SlotPrefix empty;
    const auto mt = r.tb->restoreMatch(empty, req);
    ISO_CHECK(mt.has_value() && mt->reuse == 3 * KP - 20 && mt->e->id == s->st.tier_id);
    // the slot that holds it in VRAM already: no gain, no restore
    ISO_CHECK(!r.tb->restoreMatch(s->sp, req).has_value());
    // another conversation: nothing
    std::vector<std::uint32_t> other(4 * KP, 7);
    ISO_CHECK(!r.tb->restoreMatch(empty, other).has_value());
}

void testRestoreMatchMinGainDefault() {
    Rig r(true);  // min_gain 512 (TIER_MIN_GAIN default)
    auto s = r.slot(tokens(KP + 100), {KP - 50, KP + 100}, 5);
    r.tb->spillSlot(0, true, s->sp, s->st);
    r.tb->waitSpill(s->st);
    ISO_CHECK(r.tier->stats.spills == 1);
    sv::SlotPrefix empty;
    ISO_CHECK(!r.tb->restoreMatch(empty, tokens(4 * KP)).has_value());  // KP + 100 tokens < 512: not worth a restore
    // a longer entry of the same conversation (another slot): 3 KP - 20 >= 512 and >= reuse / 40
    auto b = r.slot(tokens(3 * KP - 20), {KP - 50, 3 * KP - 20}, 8);
    r.tb->spillSlot(1, true, b->sp, b->st);
    r.tb->waitSpill(b->st);
    const auto mt = r.tb->restoreMatch(empty, tokens(4 * KP));
    ISO_CHECK(mt.has_value() && mt->reuse == 3 * KP - 20 && mt->e->id == b->st.tier_id);
    r.tb->tick();
    const sv::TierBridge::Published p = r.tb->published();
    ISO_CHECK(p.ssd_written == 0 && p.ssd_entries == 0 && p.ssd_used == 0);
    r.tb->logStats(0, 0);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("tier_bridge",
                    {{"span_order", testSpanOrder},
                     {"no_tier", testNoTier},
                     {"spill_slot", testSpillSlot},
                     {"restore_match_gain", testRestoreMatchGain},
                     {"restore_min_gain_default", testRestoreMatchMinGainDefault}},
                    argc, argv);
}
