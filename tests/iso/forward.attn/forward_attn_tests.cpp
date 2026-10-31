// whirl-iso-forward-attn-tests: the module forward.attn (src/model/qwen35/forward/attn.cpp: the MTP draft attention draftAttn,
// the gated full-attention sublayer attnBlock, the WMMA prefill attention prefillAttn) built over forward.dispatch (dispatch.cpp,
// the matmul / norm helpers it calls) and forward.state (context.cpp), WITHOUT any other forward source (ARCH-1 8.5 isolation,
// arch025-n8b3, PLAN N8 batch 3): a call from attn.cpp into run / mtp_verify / prefill / ffn / gdn / vision_rows, class Model,
// the engine or the server log is LNK2019 here. attn.cpp only launches: the cases read what it launches through the launch
// recorder of tests/iso/fake_hip.cpp (kernel handle, grid, block; never a GPU result) and the attention test hook of Context.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

// Link doubles for this test program only (as tests/iso/forward.state): Context's TuneTable member defaults (model.core).
namespace whirl::qwen35 {
GemvR defaultGemvR() { return {}; }
GemvW defaultGemvW() { return {}; }
GemvR defaultGemvWHead() { return {}; }
}  // namespace whirl::qwen35

// The launch recorder of the HIP link double (tests/iso/fake_hip.cpp).
namespace whirl::hip::isofake {
struct Launch {
    Function f;
    Dim3 grid, block;
};
void recordLaunches(bool on);
std::vector<Launch> recordedLaunches();
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;

struct AttnProbe : q::Context {
    using q::Context::attnBlock;
    using q::Context::KvLayer;
    using q::Context::mtpKv;
};

whirl::hip::Function fn(std::uintptr_t i) { return reinterpret_cast<whirl::hip::Function>(i); }

std::vector<std::int32_t> g_wins;
void hook(void*, std::int32_t win) { g_wins.push_back(win); }

void testDraftAttnWindow() {
    auto c = std::make_unique<AttnProbe>();
    c->num_plan.flags.draft_window = 200;
    c->num_plan.flags.draft_window_min = 8192;
    c->attn_win = 7;
    c->attn_test_hook = hook;
    g_wins.clear();
    q::AttnW a;
    bool refused = false;
    try {
        c->draftAttn(a, 1);  // the q / k / v projection reaches the link double (recorder off)
    } catch (const std::logic_error&) {
        refused = true;
    }
    ISO_CHECK(refused);
    ISO_CHECK(g_wins.size() == 1 && g_wins[0] == q::draftWindowArg(200, 8192));  // attention ran with the draft window
    ISO_CHECK(c->attn_win == 7);  // restored on every exit, also when attention throws (OVH-C1)
    g_wins.clear();
    try {
        c->attnBlock(a, c->mtpKv(), 1);
    } catch (const std::logic_error&) {
    }
    ISO_CHECK(g_wins.size() == 1 && g_wins[0] == 7);  // a plain attention block: the window as it is
}

void testAttnPlainPrepLaunches() {
    // no fused / M-RoPE prep kernel: q / k norms, two RoPE launches (q heads, then kv heads) and the KV store
    auto c = std::make_unique<AttnProbe>();
    c->cfg.head_dim = 128;
    c->cfg.n_head = 4;
    c->cfg.n_head_kv = 2;
    c->cfg.n_rot = 64;
    c->cfg.n_embd = 256;
    c->cfg.rope_base = 10000.0f;
    c->k.rope_neox = fn(0x10);
    c->k.kv_store = fn(0x20);
    fake::recordLaunches(true);
    q::AttnW a;
    try {
        c->attnBlock(a, c->mtpKv(), 3);
    } catch (const std::exception&) {
        // whatever comes after the prep (attention core, output projection) may refuse; the prep launches are recorded
    }
    const auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    std::vector<fake::Launch> rope, store;
    for (const auto& x : l) {
        if (x.f == fn(0x10)) rope.push_back(x);
        if (x.f == fn(0x20)) store.push_back(x);
    }
    ISO_CHECK(rope.size() == 2 && store.size() == 1);
    if (rope.size() == 2) {
        ISO_CHECK(rope[0].grid.x == 4 && rope[0].grid.y == 3 && rope[0].block.x == 64);  // q heads x rows
        ISO_CHECK(rope[1].grid.x == 2 && rope[1].grid.y == 3 && rope[1].block.x == 64);  // kv heads x rows
    }
    if (store.size() == 1) ISO_CHECK(store[0].grid.x == 3 && store[0].block.x == 256);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.attn", {{"draft_attn_window", testDraftAttnWindow}, {"attn_plain_prep_launches", testAttnPlainPrepLaunches}},
                    argc, argv);
}
