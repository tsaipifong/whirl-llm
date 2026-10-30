// whirl-iso-forward-state-tests: the module forward.state (class qwen35::Context: src/model/qwen35/context.h, the members
// defined in src/model/qwen35/context.cpp) built WITHOUT any forward source (no src/model/qwen35/forward/*.cpp: a call from
// context.cpp into a forward member is LNK2019 here) and run (ARCH-1 8.5 isolation, arch025-n8b2, PLAN N8 batch 2). The first
// forward.* module with its own manifest: the bottom of the forward call graph (progress/arch025-n8b2.PRIOR_ART.md, decision 3).
// Cases: tests/iso/forward/forward_state_cases.inc (moved verbatim out of forward_tests.cpp, which still runs them) and the
// context.cpp members below (device-memory ledger, recurrent-state sizes and snapshot sets, replay offsets, KV arguments,
// sequence selection, reset, layerMats), on the HIP layer's link double tests/iso/fake_hip.cpp (its allocation recorder).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../forward/forward_state_cases.inc"

// Link doubles for this test program only (as tests/iso/forward/forward_tests.cpp): Context's TuneTable member defaults
// (src/model/qwen35/tune_table.h; defined in src/model/config.cpp, model.core, which forward.state does not link). The
// Context constructor calls them; no value is read here.
namespace whirl::qwen35 {
GemvR defaultGemvR() { return {}; }
GemvW defaultGemvW() { return {}; }
GemvR defaultGemvWHead() { return {}; }
}  // namespace whirl::qwen35

// The allocation recorder of the HIP link double (tests/iso/fake_hip.cpp).
namespace whirl::hip::isofake {
void recordAllocs(bool on);
std::vector<std::uint64_t> recordedSizes();
}  // namespace whirl::hip::isofake

namespace {

namespace fake = whirl::hip::isofake;
using DevPtr = whirl::hip::DevPtr;  // the device address type of the Context members

struct StateProbe : q::Context {
    using q::Context::kvArgs;
    using q::Context::pendPtr;
    using q::Context::pendSsmOff;
    using q::Context::KvLayer;
    using q::Context::mtp_h;
};

std::unique_ptr<StateProbe> smallModel() {
    auto c = std::make_unique<StateProbe>();
    c->cfg.n_layer = 3;
    c->cfg.d_conv = 4;
    c->cfg.n_k_heads = 2;
    c->cfg.d_state = 8;
    c->cfg.d_inner = 32;
    c->cfg.n_v_heads = 4;
    return c;  // convCh = 2*2*8 + 32 = 64, headV = 32 / 4 = 8
}

void testStateSizes() {
    const auto c = smallModel();
    ISO_CHECK(c->cfg.convCh() == 64 && c->cfg.headV() == 8);
    ISO_CHECK(c->convBytes() == 3ull * 64 * 4);   // (d_conv - 1) * convCh f32
    ISO_CHECK(c->ssmBytes() == 4ull * 8 * 8 * 4);  // n_v_heads * d_state * headV f32
    ISO_CHECK(c->pendSsmOff() == static_cast<std::uint64_t>(q::max_small_batch) * 64 * 4);
    ISO_CHECK(c->pendLayerBytes() == c->pendSsmOff() + static_cast<std::uint64_t>(q::max_small_batch) * 4 * (8 + 8 + 2) * 4);
}

void testPendPtr() {
    auto c = smallModel();
    c->gdn_ord = {0, 2, 1};  // layer i -> its slot in the replay buffer
    whirl::qwen35::Seq sq;
    sq.pend = 0x100000;
    const std::uint64_t L = c->pendLayerBytes();
    ISO_CHECK(c->pendPtr(sq, 0, true) == 0x100000);
    ISO_CHECK(c->pendPtr(sq, 1, true) == 0x100000 + 2 * L);
    ISO_CHECK(c->pendPtr(sq, 2, true) == 0x100000 + L);
    ISO_CHECK(c->pendPtr(sq, 1, false) == 0x100000 + 2 * L + c->pendSsmOff());  // the SSM inputs follow the conv inputs
}

void testKvArgs() {
    auto c = smallModel();
    c->ptab = 0x7700;
    const StateProbe::KvLayer kv{0x10, 0x20, 0x30, 0x40};
    const auto a = c->kvArgs(kv, 0x900, 5);
    ISO_CHECK(a.k == 0x10 && a.v == 0x20 && a.ks == 0x30 && a.vs == 0x40);
    ISO_CHECK(a.ptab == 0x7700 && a.kvbase == 0x900 && a.tab0 == 5 && a.win == 0);  // the window is the caller's (attn.cpp)
}

void testSelectSeq() {
    auto c = smallModel();
    std::vector<DevPtr> conv0(3, 1), ssm0(3, 2), conv1(3, 3), ssm1(3, 4);
    c->seqs.resize(2);
    c->seqs[0].conv = conv0;
    c->seqs[0].ssm = ssm0;
    c->seqs[1].conv = conv1;
    c->seqs[1].ssm = ssm1;
    c->seqs[1].hid = 0xabc;
    c->seqs[1].kv_base = 640;
    c->seqs[1].npend = 9;
    c->selectSeq(1);
    ISO_CHECK(c->cur_seq == 1 && c->conv_state.data() == conv1.data() && c->ssm_state.data() == ssm1.data());
    ISO_CHECK(c->mtp_h == 0xabc && c->kv_off == 640 && c->ctl_off == q::ctl_words);
    c->selectSeq(0);
    ISO_CHECK(c->cur_seq == 0 && c->conv_state.data() == conv0.data() && c->kv_off == 0 && c->ctl_off == 0);
    c->dropPending(1);
    ISO_CHECK(c->seqs[1].npend == 0);
    c->dropPending(7);  // no such sequence: nothing
    ISO_CHECK(c->seqs.size() == 2);
}

void testAllocLedger() {
    auto c = smallModel();
    fake::recordAllocs(true);
    const DevPtr a = c->alloc(100);
    const DevPtr b = c->allocZero(300);  // alloc + memset of the same address
    ISO_CHECK(a != 0 && b != 0 && a != b && c->allocations.size() == 2 && c->alloc_total == 400);
    c->freeAlloc(0x5);  // not one of ours: nothing
    ISO_CHECK(c->allocations.size() == 2 && c->alloc_total == 400);
    c->freeAlloc(a);
    ISO_CHECK(c->allocations.size() == 1 && c->alloc_total == 300 && c->allocations[0].get() == b);
    c->freeAlloc(b);
    ISO_CHECK(c->allocations.empty() && c->alloc_total == 0);
    ISO_CHECK((fake::recordedSizes() == std::vector<std::uint64_t>{100, 300}));
    fake::recordAllocs(false);
    bool refused = false;
    try {
        c->alloc(8);  // recorder off: the device runtime, which the double refuses
    } catch (const std::logic_error&) {
        refused = true;
    }
    ISO_CHECK(refused && c->allocations.empty() && c->alloc_total == 0);
}

void testEnsureSnapshots() {
    auto c = smallModel();
    std::vector<DevPtr> ssm(3, 0);
    ssm[1] = 0x42;  // only layer 1 is DeltaNet (attention layers have no recurrent state)
    c->ssm_state = ssm;
    fake::recordAllocs(true);
    c->ensureSnapshots(2);
    ISO_CHECK(c->snap_sets == 2 && c->snap_conv[0].size() == 3 && c->snap_ssm[1].size() == 3);
    ISO_CHECK(c->snap_conv[0][0] == 0 && c->snap_conv[0][1] != 0 && c->snap_ssm[1][2] == 0 && c->snap_ssm[1][1] != 0);
    ISO_CHECK((fake::recordedSizes() == std::vector<std::uint64_t>{c->convBytes(), c->ssmBytes(), c->convBytes(), c->ssmBytes()}));
    c->ensureSnapshots(1);  // never shrinks, allocates nothing
    ISO_CHECK(c->snap_sets == 2 && fake::recordedSizes().size() == 4);
    c->ensureSnapshots(1000);  // capped at gdn_max_snap sets
    ISO_CHECK(c->snap_sets == q::gdn_max_snap && fake::recordedSizes().size() == 2 * q::gdn_max_snap);
    fake::recordAllocs(false);
}

void testReset() {
    auto c = smallModel();
    fake::recordAllocs(true);
    const DevPtr conv = c->alloc(c->convBytes()), ssm = c->alloc(c->ssmBytes());
    fake::recordAllocs(false);
    std::vector<DevPtr> cv(3, 0), sv(3, 0);
    cv[2] = conv;
    sv[2] = ssm;
    c->seqs.resize(1);
    c->seqs[0].conv = cv;
    c->seqs[0].ssm = sv;
    c->seqs[0].npend = 4;
    c->selectSeq(0);
    c->reset();  // clears the selected sequence's pending replay and zeroes the DeltaNet layers' state (memset of our addresses)
    ISO_CHECK(c->seqs[0].npend == 0);
    cv[1] = 0x999;  // an address the device runtime never handed out: reset reaches the runtime for it
    bool refused = false;
    try {
        c->reset();
    } catch (const std::logic_error&) {
        refused = true;
    }
    ISO_CHECK(refused);
}

void testLayerMats() {
    const auto c = smallModel();
    whirl::qwen35::Layer a;
    a.kind = whirl::qwen35::LayerKind::attn;
    a.attn.q.ptr = 1, a.attn.k.ptr = 2, a.attn.v.ptr = 3, a.attn.o.ptr = 4;
    a.ffn_gate.ptr = 5, a.ffn_up.ptr = 6, a.ffn_down.ptr = 7;
    const whirl::qwen35::MatList ma = c->layerMats(a);
    ISO_CHECK(ma.len == 7);
    for (std::size_t i = 0; i < ma.len; ++i) ISO_CHECK(ma.items[i].ptr == i + 1);
    whirl::qwen35::Layer g;  // kind gdn (the default)
    g.gdn.qkv.ptr = 1, g.gdn.gate.ptr = 2, g.gdn.beta.ptr = 3, g.gdn.alpha.ptr = 4, g.gdn.out.ptr = 5;
    g.ffn_gate.ptr = 6, g.ffn_up.ptr = 7, g.ffn_down.ptr = 8;
    const whirl::qwen35::MatList mg = c->layerMats(g);
    ISO_CHECK(mg.len == 8);
    for (std::size_t i = 0; i < mg.len; ++i) ISO_CHECK(mg.items[i].ptr == i + 1);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.state",
                    {{"tune_bucket_of_each_size", testTuneBucketOfEachSize},
                     {"tune_bucket_edges", testTuneBucketEdges},
                     {"op_class_names", testOpClassNames},
                     {"profile_host_only", testProfileHostOnly},
                     {"draft_window_arg", testDraftWindowArg},
                     {"state_sizes", testStateSizes},
                     {"pend_ptr", testPendPtr},
                     {"kv_args", testKvArgs},
                     {"select_seq", testSelectSeq},
                     {"alloc_ledger", testAllocLedger},
                     {"ensure_snapshots", testEnsureSnapshots},
                     {"reset", testReset},
                     {"layer_mats", testLayerMats}},
                    argc, argv);
}
