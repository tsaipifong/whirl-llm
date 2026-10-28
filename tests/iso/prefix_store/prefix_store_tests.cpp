// whirl-iso-prefix-store-tests: src/server/prefix_store.cpp built alone (ARCH-1 8.5, arch025-lane2 leg 4) with its declared
// dependencies: the chat-token facts (tokens.cpp + the tokenizer), the prefix cache decisions (prefix_cache.cpp), the tier
// view (tier_adapter.cpp over tier.kv), the checkpoint buffers (ckpt_store through cache/cache_model.h), the device contract
// with the NullBackend and the owner check (threads). No engine, no tier bridge, no model, no HIP layer: the chunk schedule,
// the COLD-RESUME rule, the slot / shared-checkpoint lookup and the parked shared-checkpoint table run on host memory.
// SPDX-License-Identifier: Apache-2.0
#include "cache/ckpt_store.h"
#include "device/null_backend.h"
#include "iso/fake_cache_model.h"
#include "iso/iso_test.h"
#include "server/prefix_store.h"
#include "tier/kv_tier.h"

#include <cstdint>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace {

namespace sv = whirl::server;
namespace dv = whirl::device;

// chat ids of the test vocabulary: <|im_start|> 1, system 2, <|im_end|> 3, "\n" 4, <think> 5; text ids from 10
sv::ChatTokens chat() {
    sv::ChatTokens ct;
    ct.im_start = 1;
    ct.system = 2;
    ct.im_end = 3;
    ct.nl = 4;
    ct.think = 5;
    return ct;
}

// 3000 tokens: a system message that ends at 2002 ("<|im_end|>" "\n" at 2000 / 2001, then "<|im_start|>")
std::vector<std::uint32_t> prompt() {
    std::vector<std::uint32_t> t(3000);
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = 10 + static_cast<std::uint32_t>(i % 97);
    t[0] = 1;
    t[1] = 2;
    t[2000] = 3;
    t[2001] = 4;
    t[2002] = 1;
    return t;
}

sv::PrefixStore::Config config(bool cold) {
    sv::PrefixStore::Config c;
    c.sys_min = 4;
    c.cold_resume = cold;
    return c;
}

void testCkKindCodes() {
    const char* kinds[] = {"prompt-end", "think-open", "gen-end", "system", "prefix"};
    for (std::uint8_t c = 1; c <= 5; ++c) {
        ISO_CHECK(sv::ckKindCode(kinds[c - 1]) == c);
        ISO_CHECK(std::strcmp(sv::ckKindStr(c), kinds[c - 1]) == 0);
    }
    ISO_CHECK(sv::ckKindCode("restored") == 0 && sv::ckKindCode("") == 0);
    ISO_CHECK(std::strcmp(sv::ckKindStr(0), "restored") == 0 && std::strcmp(sv::ckKindStr(9), "restored") == 0);
}

void testScheduleAndColdResume() {
    const sv::ChatTokens ct = chat();
    const std::vector<std::uint32_t> t = prompt();
    sv::PrefixStore ps(ct, config(false));
    // schedule: 0, 1024 (one chunk), 2002 (the system split; 978 rows <= chunk + merge), 3000
    ISO_CHECK(ps.sysSplit(t) == 2002);
    ISO_CHECK(ps.onSchedule(t, 0) && ps.onSchedule(t, 1024) && ps.onSchedule(t, 2002) && ps.onSchedule(t, 3000));
    ISO_CHECK(!ps.onSchedule(t, 1500) && !ps.onSchedule(t, 2050));
    ISO_CHECK(ps.schedFloor(t, 2500) == 2002 && ps.schedFloor(t, 1023) == 0);
    ISO_CHECK(ps.imgSplits(t).empty());  // no vision attached
    // the sys split is off below sys_min
    sv::PrefixStore::Config big = config(false);
    big.sys_min = 2048;
    sv::PrefixStore off(ct, big);
    ISO_CHECK(off.sysSplit(t) == 0 && off.onSchedule(t, 2048) && !off.onSchedule(t, 2002));
    // COLD-RESUME off: every checkpoint may be resumed
    ISO_CHECK(ps.resumeOk(t, 1500, sv::ckKindCode("prefix")) && ps.resumeOk(t, 2002, sv::ckKindCode("gen-end")));
    // COLD-RESUME on: only schedule points, never gen-end
    sv::PrefixStore cold(ct, config(true));
    ISO_CHECK(cold.resumeOk(t, 1024, sv::ckKindCode("prompt-end")) && cold.resumeOk(t, 2002, sv::ckKindCode("system")));
    ISO_CHECK(!cold.resumeOk(t, 1500, sv::ckKindCode("prefix")));
    ISO_CHECK(!cold.resumeOk(t, 2002, sv::ckKindCode("gen-end")));
    // think-open: a prompt ending "<think>" "\n" splits before the last "\n"
    std::vector<std::uint32_t> th(t.begin(), t.begin() + 2100);
    th.push_back(5);
    th.push_back(4);
    ISO_CHECK(ct.thinkOpenSplit(th) == th.size() - 1);
    ISO_CHECK(ps.onSchedule(th, th.size() - 1));
}

void addCk(sv::SlotPrefix& sl, std::uint32_t pos, const char* kind) {
    sl.ckpts.emplace_back();
    sl.ckpts.back().valid = true;
    sl.ckpts.back().pos = pos;
    sl.ckpts.back().kind = kind;
}

void testCacheMatch() {
    const sv::ChatTokens ct = chat();
    const std::vector<std::uint32_t> t = prompt();
    sv::SlotPrefix sl;
    sl.cache_tokens.assign(t.begin(), t.begin() + 2100);
    addCk(sl, 1024, "prompt-end");
    addCk(sl, 2050, "prefix");  // not a schedule point of t
    sv::PrefixStore ps(ct, config(false));
    const sv::PrefixStore::Match m = ps.cacheMatch(sl, t);
    ISO_CHECK(m.lcp == 2100 && m.reuse == 2050 && m.ck == &sl.ckpts[1]);
    // COLD-RESUME: the off-schedule checkpoint is skipped, the earlier one is the resume point
    sv::PrefixStore cold(ct, config(true));
    const sv::PrefixStore::Match c = cold.cacheMatch(sl, t);
    ISO_CHECK(c.lcp == 2100 && c.reuse == 1024 && c.ck == &sl.ckpts[0]);
    ISO_CHECK(sl.ckpts[1].valid);  // the slot's own checkpoints are not changed
    // another conversation: no common prefix, nothing to resume
    std::vector<std::uint32_t> other(t.size(), 7);
    const sv::PrefixStore::Match o = ps.cacheMatch(sl, other);
    ISO_CHECK(o.lcp == 0 && o.reuse == 0 && o.ck == nullptr);
}

void testSpeMatchAndPickSlot() {
    const sv::ChatTokens ct = chat();
    const std::vector<std::uint32_t> t = prompt();
    sv::PrefixStore ps(ct, config(false));
    ISO_CHECK(ps.speMatch(t) == nullptr);
    ps.spes().emplace_back();
    sv::Spe& s = ps.spes().back();
    s.valid = true;
    s.n = 1024;
    s.kind = "prefix";
    s.tokens.assign(t.begin(), t.begin() + 1024);
    ISO_CHECK(ps.speMatch(t) == &ps.spes()[0]);
    s.n = 1000;  // not a schedule point: no shared resume
    s.tokens.resize(1000);
    ISO_CHECK(ps.speMatch(t) == nullptr);
    s.n = 1024;
    s.tokens.assign(t.begin(), t.begin() + 1024);
    // slots: 0 and 2 hold other text, 1 holds this conversation with a checkpoint at the system split
    std::vector<sv::SlotPrefix> slots(3);
    slots[0].cache_tokens.assign(50, 8);
    slots[0].last_used = 5;
    slots[2].cache_tokens.assign(50, 9);
    slots[2].last_used = 3;
    slots[1].cache_tokens.assign(t.begin(), t.begin() + 2100);
    addCk(slots[1], 2002, "system");
    slots[1].last_used = 9;
    std::span<sv::SlotPrefix> sp(slots);
    ISO_CHECK(ps.pickSlot(sp, t, [](const sv::SlotPrefix&) { return true; }) == &slots[1]);
    // slot 1 busy: the others resume only the shared checkpoint (1024 both): the least recently used
    ISO_CHECK(ps.pickSlot(sp, t, [&](const sv::SlotPrefix& x) { return &x != &slots[1]; }) == &slots[2]);
    ISO_CHECK(ps.pickSlot(sp, t, [](const sv::SlotPrefix&) { return false; }) == nullptr);
}

void testTierViewAndParkedSpes() {
    dv::NullBackend ops;
    const sv::ChatTokens ct = chat();
    sv::PrefixStore ps(ct, config(false));
    ISO_CHECK(ps.tierView() == nullptr);
    auto tier = whirl::tier::Tier::create(ops, 8 * whirl::tier::block_bytes, 42, 0);
    ISO_CHECK(tier != nullptr);
    ps.attachTier(tier.get());
    ISO_CHECK(ps.tierView() != nullptr && ps.tierView()->entryCount() == 0);
    ps.attachTier(nullptr);
    ISO_CHECK(ps.tierView() == nullptr);
    // memory governor: the shared checkpoint table parked and back (CkptStore on host memory)
    iso::FakeCacheModel m(ops, {false, true, false, true}, 100, 300, 8, 10);
    sv::CkptStore cs(ops, m, {100, 300, false});
    const std::uint64_t one = cs.bytes(false);
    ISO_CHECK(one > 0);
    ISO_CHECK(ps.unparkSpes(cs, 3) == 3 * one);
    ISO_CHECK(ps.spes().size() == 3 && ps.spes()[2].ck.dev.get() != 0 && !ps.spes()[2].valid);
    ps.waitSpeFences();
    ISO_CHECK(ps.parkSpes(cs) == 3 * one);
    ISO_CHECK(ps.spes().empty());
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("prefix_store",
                    {{"ck_kind_codes", testCkKindCodes},
                     {"schedule_cold_resume", testScheduleAndColdResume},
                     {"cache_match", testCacheMatch},
                     {"spe_match_pick_slot", testSpeMatchAndPickSlot},
                     {"tier_view_parked_spes", testTierViewAndParkedSpes}},
                    argc, argv);
}
