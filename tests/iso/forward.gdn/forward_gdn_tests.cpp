// whirl-iso-forward-gdn-tests: the module forward.gdn (src/model/qwen35/forward/gdn.cpp: the Gated-DeltaNet layer gdnLayer, the
// recurrent-state segment tables gdnSegsBatch / gdnSegsSingle, gdnChunked / gdnScan) built over forward.dispatch (dispatch.cpp)
// and forward.state (context.cpp), WITHOUT any other forward source (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8 batch 3): a
// call from gdn.cpp into run / mtp_verify / prefill / attn / ffn / vision_rows, class Model, the engine or the server log is
// LNK2019 here. Host-only cases: the segment tables the DeltaNet kernels read (kernels_abi.h GdnSegs).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <cstdint>
#include <memory>
#include <vector>

// Link doubles for this test program only (as tests/iso/forward.state): Context's TuneTable member defaults (model.core).
namespace whirl::qwen35 {
GemvR defaultGemvR() { return {}; }
GemvW defaultGemvW() { return {}; }
GemvR defaultGemvWHead() { return {}; }
}  // namespace whirl::qwen35

namespace {

namespace q = whirl::qwen35;
using DevPtr = whirl::hip::DevPtr;

struct GdnProbe : q::Context {
    using q::Context::gdnSegsBatch;
    using q::Context::gdnSegsSingle;
    using q::Context::pendPtr;
};

// two layers (1 = DeltaNet), two sequences with their own state views
struct World {
    std::unique_ptr<GdnProbe> c = std::make_unique<GdnProbe>();
    std::vector<DevPtr> conv0{0, 0x1000}, ssm0{0, 0x2000}, conv1{0, 0x3000}, ssm1{0, 0x4000};
    std::vector<DevPtr> cs{0, 0x5000}, ss{0, 0x6000};
    World() {
        c->cfg.n_layer = 2;
        c->cfg.d_conv = 4;
        c->cfg.n_k_heads = 2;
        c->cfg.d_state = 8;
        c->cfg.d_inner = 32;
        c->cfg.n_v_heads = 4;
        c->seqs.resize(2);
        c->seqs[0].conv = conv0, c->seqs[0].ssm = ssm0;
        c->seqs[1].conv = conv1, c->seqs[1].ssm = ssm1;
        c->conv_state = cs;
        c->ssm_state = ss;
        for (std::size_t s = 0; s < q::gdn_max_snap; ++s) {
            c->snap_conv[s] = {0, 0x10000 + 0x100 * s};
            c->snap_ssm[s] = {0, 0x20000 + 0x100 * s};
        }
    }
};

void testSegsSingle() {
    World w;
    w.c->snap_rows = 3;
    const whirl::kernels::GdnSegs sc = w.c->gdnSegsSingle(1, true, 5);
    ISO_CHECK(sc.s[0].state == 0x5000 && sc.s[0].row0 == 0 && sc.s[0].nrows == 5);
    for (std::size_t k = 0; k < 3; ++k) ISO_CHECK(sc.s[0].snap[k] == 0x10000 + 0x100 * k);
    ISO_CHECK(sc.s[0].snap[3] == 0);  // only snap_rows snapshot sets
    const whirl::kernels::GdnSegs ss = w.c->gdnSegsSingle(1, false, 2);
    ISO_CHECK(ss.s[0].state == 0x6000 && ss.s[0].nrows == 2 && ss.s[0].snap[0] == 0x20000);
    ISO_CHECK(sc.s[1].nrows == 0);  // one segment only
}

void testSegsBatchSnapshots() {
    World w;
    const q::VSeg segs[2] = {{0, 0, 2, 0, {}}, {1, 0, 1, 0, {}}};
    q::BatchPlan bp;
    bp.segs = segs;
    bp.snap_base = {0, 2};
    const whirl::kernels::GdnSegs sc = w.c->gdnSegsBatch(bp, 1, true);
    ISO_CHECK(sc.s[0].state == 0x1000 && sc.s[0].row0 == 0 && sc.s[0].nrows == 3);
    ISO_CHECK(sc.s[1].state == 0x3000 && sc.s[1].row0 == 3 && sc.s[1].nrows == 2);  // rows follow the first segment's
    ISO_CHECK(sc.s[0].snap[0] == 0x10000 && sc.s[0].snap[1] == 0x10100 && sc.s[0].snap[2] == 0);
    ISO_CHECK(sc.s[1].snap[0] == 0x10200 && sc.s[1].snap[1] == 0);  // its snapshot sets start at snap_base
    ISO_CHECK(sc.s[0].pend == 0 && sc.s[0].npend == 0);
    const whirl::kernels::GdnSegs ss = w.c->gdnSegsBatch(bp, 1, false);
    ISO_CHECK(ss.s[0].state == 0x2000 && ss.s[1].state == 0x4000 && ss.s[1].snap[0] == 0x20200);
}

void testSegsBatchReplay() {
    World w;
    w.c->gdn_replay = true;
    w.c->gdn_ord = {0, 0};
    w.c->seqs[0].pend = 0x80000;
    w.c->seqs[0].npend = 2;
    w.c->seqs[1].pend = 0x90000;
    w.c->seqs[1].npend = 1;
    const q::VSeg segs[2] = {{0, 0, 2, 0, {}}, {1, 0, 1, 0, {}}};
    q::BatchPlan bp;
    bp.segs = segs;
    const whirl::kernels::GdnSegs sc = w.c->gdnSegsBatch(bp, 1, true);
    ISO_CHECK(sc.s[0].row0 == 0 && sc.s[0].nrows == 3 && sc.s[1].row0 == 3 && sc.s[1].nrows == 2);
    ISO_CHECK(sc.s[0].pend == w.c->pendPtr(w.c->seqs[0], 1, true) && sc.s[0].npend == 2);
    ISO_CHECK(sc.s[1].pend == w.c->pendPtr(w.c->seqs[1], 1, true) && sc.s[1].npend == 1);
    ISO_CHECK(sc.s[0].snap[0] == 0);  // replay mode: no snapshot sets
    const whirl::kernels::GdnSegs ss = w.c->gdnSegsBatch(bp, 1, false);
    ISO_CHECK(ss.s[0].pend == w.c->pendPtr(w.c->seqs[0], 1, false) && ss.s[0].pend != sc.s[0].pend);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.gdn",
                    {{"segs_single", testSegsSingle}, {"segs_batch_snapshots", testSegsBatchSnapshots}, {"segs_batch_replay", testSegsBatchReplay}},
                    argc, argv);
}
