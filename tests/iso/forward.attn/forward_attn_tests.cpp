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

// v0252-e1grp (PLAN V1-93): an E1 verify of one slot (Context::row_seq1 = n, positions only on the device) runs its attention as
// ONE 2-row group when the interval of row 0's position (Context::row0_lo / row0_hi, from VSeg::pos_n) keeps the rows in one
// split range at every position of it; one-row groups when a position of the interval crosses a range, when no interval is
// given (row0_lo < 0), or when the rows are not one sequence; rows the host knows: their own grouping (unchanged)
void testE1VerifyGroups() {
    struct Got {
        whirl::hip::Function f = nullptr;
        unsigned z = 0;
        std::size_t n = 0;
    };
    auto launch = [](std::int32_t lo, std::int32_t hi, std::uint32_t seq1, std::uint32_t known) {
        auto c = std::make_unique<AttnProbe>();
        c->cfg.head_dim = 256;
        c->cfg.n_head = 16;
        c->cfg.n_head_kv = 2;
        c->cfg.n_rot = 64;
        c->cfg.n_embd = 2048;
        c->cfg.rope_base = 10000000.0f;
        c->k.rope_neox = fn(0x10);
        c->k.kv_store = fn(0x20);
        c->k.attn_ws4 = fn(0x30);
        for (int i = 0; i < 3; ++i) c->k.attn_ws4g[i] = fn(0x31 + static_cast<std::uintptr_t>(i));
        c->k.attn_wsplit1 = fn(0x34);
        c->k.attn_combine = fn(0x35);
        c->max_ctx = 65536;
        c->row_n = known;
        c->row_seq1 = seq1;
        c->row0_lo = lo;
        c->row0_hi = hi;
        for (std::uint32_t r = 0; r < 2; ++r) {
            c->row_pos[r] = (lo < 0 ? 40000 : lo) + static_cast<std::int32_t>(r);
            c->row_base[r] = 0;
        }
        fake::recordLaunches(true);
        q::AttnW a;
        try {
            c->attnBlock(a, c->mtpKv(), 2);
        } catch (const std::exception&) {
            // what comes after the attention launch (combine, o-projection) may refuse; the launch is recorded
        }
        const auto l = fake::recordedLaunches();
        fake::recordLaunches(false);
        Got g;
        for (const auto& x : l)
            if (x.f == fn(0x30) || x.f == fn(0x31) || x.f == fn(0x32) || x.f == fn(0x33)) {
                g.f = x.f;
                g.z = x.grid.z;
                g.n += 1;
            }
        return g;
    };
    const Got grp = launch(40000, 40001, 2, 0);  // away from a range boundary: one group of 2 rows (attn_ws4g, CG 1)
    ISO_CHECK(grp.n == 1 && grp.f == fn(0x31) && grp.z == 1);
    const Got crs = launch(32766, 32767, 2, 0);  // row 1 at n_pos 32769 for row 0 at 32767: one-row groups
    ISO_CHECK(crs.n == 1 && crs.f == fn(0x30) && crs.z == 2);
    const Got none = launch(-1, -1, 2, 0);  // no interval: today's E1 plan
    ISO_CHECK(none.n == 1 && none.f == fn(0x30) && none.z == 2);
    const Got two = launch(40000, 40001, 0, 0);  // unknown rows that are not one sequence: one-row groups
    ISO_CHECK(two.n == 1 && two.f == fn(0x30) && two.z == 2);
    const Got kn = launch(40000, 40001, 0, 2);  // rows the host knows (E1 off): their own grouping, as before
    ISO_CHECK(kn.n == 1 && kn.f == fn(0x31) && kn.z == 1);
    const Got knx = launch(32767, 32767, 0, 2);  // known rows across a boundary: two groups
    ISO_CHECK(knx.n == 1 && knx.f == fn(0x30) && knx.z == 2);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.attn", {{"draft_attn_window", testDraftAttnWindow}, {"attn_plain_prep_launches", testAttnPlainPrepLaunches},
                     {"e1_verify_groups", testE1VerifyGroups}},
                    argc, argv);
}
