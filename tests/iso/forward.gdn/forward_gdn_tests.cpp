// whirl-iso-forward-gdn-tests: the module forward.gdn (src/model/qwen35/forward/gdn.cpp: the Gated-DeltaNet layer gdnLayer, the
// recurrent-state segment tables gdnSegsBatch / gdnSegsSingle, gdnChunked / gdnScan) built over forward.dispatch (dispatch.cpp)
// and forward.state (context.cpp), WITHOUT any other forward source (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8 batch 3): a
// call from gdn.cpp into run / mtp_verify / prefill / attn / ffn / vision_rows, class Model, the engine or the server log is
// LNK2019 here. Host-only cases: the segment tables the DeltaNet kernels read (kernels_abi.h GdnSegs).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"
#include "whirl/ops/gdn.h"  // v0252-e1s8 round 2: ops::GdnDecodePlan (Context::gdnDecodeFns)

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
    using q::Context::out_tok;  // v0252-e1s8: the control area base (ctlWord)
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

// v0252-e1s8 (E1 in snapshot mode, kernels/gdn_select.h): a sequence marked by setPending(s, npend_dev) in snapshot mode gets
// a SELECT segment - candidates = the snapshot sets of its last verify then its live state, the final state into the spare
// set, pend = its NPEND control word, npend = the candidate count m. A sequence without the mark keeps the table of before.
void testSegsBatchSelect() {
    World w;
    w.c->out_tok = 0x700000;
    w.c->e1_conv = {0, 0x7000};
    w.c->e1_ssm = {0, 0x8000};
    w.c->seqs[0].npend = q::npend_dev;
    w.c->seqs[0].sel_m = 2;
    const q::VSeg segs[1] = {{0, 0, 2, 0, {}}};
    q::BatchPlan bp;
    bp.segs = segs;
    const whirl::kernels::GdnSegs sc = w.c->gdnSegsBatch(bp, 1, true);
    ISO_CHECK(sc.s[0].state == 0x7000 && sc.s[0].row0 == 0 && sc.s[0].nrows == 3);  // the final state -> the spare set
    ISO_CHECK(sc.s[0].snap[0] == 0x10000 && sc.s[0].snap[1] == 0x10100);           // candidates 0 .. m - 1: the sets
    ISO_CHECK(sc.s[0].snap[2] == 0x1000 && sc.s[0].snap[3] == 0);                   // candidate m: the live state
    ISO_CHECK(sc.s[0].pend == w.c->ctlWord(0, q::ctl_npend) && sc.s[0].pend == 0x700000 + 4 * q::ctl_npend);
    ISO_CHECK(sc.s[0].npend == 2);
    const whirl::kernels::GdnSegs ss = w.c->gdnSegsBatch(bp, 1, false);
    ISO_CHECK(ss.s[0].state == 0x8000 && ss.s[0].snap[0] == 0x20000 && ss.s[0].snap[1] == 0x20100 && ss.s[0].snap[2] == 0x2000);
    ISO_CHECK(ss.s[0].pend == sc.s[0].pend && ss.s[0].npend == 2);
    // without the mark: in place, as testSegsBatchSnapshots
    w.c->seqs[0].npend = 0;
    const whirl::kernels::GdnSegs pc = w.c->gdnSegsBatch(bp, 1, true);
    ISO_CHECK(pc.s[0].state == 0x1000 && pc.s[0].snap[0] == 0x10000 && pc.s[0].snap[1] == 0x10100 && pc.s[0].snap[2] == 0);
    ISO_CHECK(pc.s[0].pend == 0 && pc.s[0].npend == 0);
    // replay mode ignores the select fields (npend_dev is the replay count word there)
    w.c->gdn_replay = true;
    w.c->gdn_ord = {0, 0};
    w.c->seqs[0].pend = 0x80000;
    w.c->seqs[0].npend = q::npend_dev;
    const whirl::kernels::GdnSegs rc = w.c->gdnSegsBatch(bp, 1, true);
    ISO_CHECK(rc.s[0].state == 0x1000 && rc.s[0].npend == -1 && rc.s[0].pend == w.c->pendPtr(w.c->seqs[0], 1, true));
}

// v0252-e1s8: Context::settleSelect renames the tables once the host knows the accepted count of the verify before the select
// verify (kernels/gdn_select.h gdn_sel_settle); dropPending then gives the state of that cycle back
void testSettleSelect() {
    for (const std::uint32_t acc : {0u, 1u, 2u}) {
        World w;
        w.c->e1_conv = {0, 0x7000};
        w.c->e1_ssm = {0, 0x8000};
        w.c->snap_sets = 2;
        w.c->seqs[0].sel_cm = 2;  // the last verify selected among sets 0, 1 and live
        w.c->seqs[0].sel_m = 2;
        w.c->settleSelect(0, acc);
        const DevPtr cand_c[3] = {0x10000, 0x10100, 0x1000}, cand_s[3] = {0x20000, 0x20100, 0x2000};
        // sets r <- candidates other than acc, in order; live <- spare; spare <- candidate acc
        std::uint32_t r = 0;
        for (std::uint32_t j = 0; j < 3; ++j) {
            if (j == acc) continue;
            if (r < 2) ISO_CHECK(w.c->snap_conv[r][1] == cand_c[j] && w.c->snap_ssm[r][1] == cand_s[j]);
            ++r;
        }
        ISO_CHECK(w.c->seqs[0].conv[1] == 0x7000 && w.c->seqs[0].ssm[1] == 0x8000);
        ISO_CHECK(w.c->e1_conv[1] == cand_c[acc] && w.c->e1_ssm[1] == cand_s[acc]);
        ISO_CHECK(w.c->seqs[0].held && w.c->seqs[0].sel_cm == -1);
        ISO_CHECK(w.c->seqs[0].conv[0] == 0 && w.c->snap_conv[0][0] == 0);  // attention layers untouched
        w.c->dropPending(0);  // the select verify in flight is dropped: the state of the read cycle is live again
        ISO_CHECK(w.c->seqs[0].conv[1] == cand_c[acc] && w.c->seqs[0].ssm[1] == cand_s[acc] && !w.c->seqs[0].held);
        w.c->dropPending(0);  // once only
        ISO_CHECK(w.c->seqs[0].conv[1] == cand_c[acc]);
    }
    // nothing to settle (the last verify did not select): a bookkeeping error
    World w;
    w.c->e1_conv = {0, 0x7000};
    w.c->e1_ssm = {0, 0x8000};
    bool threw = false;
    try {
        w.c->settleSelect(0, 0);
    } catch (const q::ModelError&) {
        threw = true;
    }
    ISO_CHECK(threw && w.c->seqs[0].conv[1] == 0x1000);
}

// v0252-e1s8 round 2 (A): the select verify launches its OWN entries (gdn_conv_l2_sel / gdn_abconv_sel_<T> / gdn_step_norm_sel);
// every other segment table launches exactly the shipping kernels of the plan (the launch table of the integration build).
// isSelect: a snapshot-mode segment table with a select segment (pend != 0); replay tables carry pend too and never select.
void testDecodeFns() {
    using F = whirl::hip::Function;
    const auto fn = [](std::uintptr_t i) { return reinterpret_cast<F>(i); };
    World w;
    auto& k = w.c->k;
    k.gdn_conv_l2 = fn(0x10);
    k.gdn_step_norm = fn(0x20);
    k.gdn_conv_l2_sel = fn(0x11);
    k.gdn_step_norm_sel = fn(0x21);
    const auto ti = [](whirl::gguf::GgmlType t) { return static_cast<std::size_t>(t); };
    const whirl::gguf::GgmlType bt = whirl::gguf::GgmlType::mxfp4;
    k.gdn_abconv[ti(bt)] = fn(0x30);
    k.gdn_abconv_sel[ti(bt)] = fn(0x31);
    k.gdn_ab[ti(bt)] = fn(0x40);
    whirl::ops::GdnDecodePlan pb, pab, pac;
    pb.gates = whirl::ops::GdnDecodeGates::precise_ba;
    pab.gates = whirl::ops::GdnDecodeGates::ab;
    pab.f = fn(0x40);
    pac.gates = whirl::ops::GdnDecodeGates::abconv;
    pac.f = fn(0x30);
    // not selecting: the shipping kernels (conv l2 for precise / ab, the plan's abconv entry; gdn_step_norm)
    for (const auto* p : {&pb, &pab}) {
        const auto f = w.c->gdnDecodeFns(*p, bt, false);
        ISO_CHECK(f.conv == fn(0x10) && f.step == fn(0x20));
    }
    const auto fa = w.c->gdnDecodeFns(pac, bt, false);
    ISO_CHECK(fa.conv == fn(0x30) && fa.step == fn(0x20));
    // selecting: the twins
    for (const auto* p : {&pb, &pab}) {
        const auto f = w.c->gdnDecodeFns(*p, bt, true);
        ISO_CHECK(f.conv == fn(0x11) && f.step == fn(0x21));
    }
    const auto fs = w.c->gdnDecodeFns(pac, bt, true);
    ISO_CHECK(fs.conv == fn(0x31) && fs.step == fn(0x21));
    // a select twin the code object lacks: null (the verify refuses before: Context::selectOk)
    k.gdn_abconv_sel[ti(bt)] = nullptr;
    ISO_CHECK(w.c->gdnDecodeFns(pac, bt, true).conv == nullptr);
    // which tables select: snapshot mode + a select segment only
    const q::VSeg segs[1] = {{0, 0, 1, 0, {}}};
    q::BatchPlan bp;
    bp.segs = segs;
    w.c->out_tok = 0x700000;
    w.c->e1_conv = {0, 0x7000};
    w.c->e1_ssm = {0, 0x8000};
    w.c->seqs[0].sel_m = 1;
    ISO_CHECK(!w.c->isSelect(w.c->gdnSegsBatch(bp, 1, true)));  // in place
    w.c->seqs[0].npend = q::npend_dev;
    ISO_CHECK(w.c->isSelect(w.c->gdnSegsBatch(bp, 1, true)) && w.c->isSelect(w.c->gdnSegsBatch(bp, 1, false)));
    w.c->gdn_replay = true;  // replay: pend is the replay buffer, never a select table
    w.c->gdn_ord = {0, 0};
    w.c->seqs[0].pend = 0x80000;
    ISO_CHECK(!w.c->isSelect(w.c->gdnSegsBatch(bp, 1, true)));
}

// v0252-e1s8 round 2 (A): the model can select only with every select twin in its code object (marker + entries)
void testSelectOkNeedsTwins() {
    using F = whirl::hip::Function;
    const auto fn = [](std::uintptr_t i) { return reinterpret_cast<F>(i); };
    World w;
    w.c->k.caps.gdn_select = true;
    w.c->k.gdn_conv_l2_sel = fn(0x11);
    w.c->k.gdn_step_norm_sel = fn(0x21);
    ISO_CHECK(w.c->selectTwinsOk());
    w.c->k.gdn_step_norm_sel = nullptr;
    ISO_CHECK(!w.c->selectTwinsOk());
    w.c->k.gdn_step_norm_sel = fn(0x21);
    w.c->k.caps.gdn_select = false;
    ISO_CHECK(!w.c->selectTwinsOk());
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.gdn",
                    {{"segs_single", testSegsSingle},
                     {"segs_batch_snapshots", testSegsBatchSnapshots},
                     {"segs_batch_replay", testSegsBatchReplay},
                     {"segs_batch_select", testSegsBatchSelect},
                     {"settle_select", testSettleSelect},
                     {"decode_fns", testDecodeFns},
                     {"select_ok_needs_twins", testSelectOkNeedsTwins}},
                    argc, argv);
}
