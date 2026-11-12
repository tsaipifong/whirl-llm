// whirl-iso-forward-mtp-verify-tests: the module forward.mtp_verify (src/model/qwen35/forward/mtp_verify.cpp: token upload, the
// MTP block, DFlash2 capture, batched and single verify, the replay commit commitSeq, snapshot restore, draft-head logits,
// argmax / cycle readback) built over the forward files it calls - run (which calls mtp_verify.cpp back: the cycle is declared
// with the REAL file, progress/arch025-n8b3.PRIOR_ART.md design 2), attn, ffn, gdn, dispatch, vision_rows - and forward.state,
// WITHOUT prefill.cpp (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8 batch 3): a call into prefill.cpp, class Model, the engine or
// the server log is LNK2019 here. Host-only cases on tests/iso/fake_hip.cpp (its launch recorder).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
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
// v0252-f1arg3: the stream of each recorded launch and one registered argument word; the allocation recorder
struct LaunchCtx {
    Stream s;
    std::uint64_t word;
};
void recordArgWord(Function f, int index);
std::vector<LaunchCtx> recordedLaunchCtx();
void recordAllocs(bool on);
std::vector<std::uint64_t> recordedSizes();
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;
using DevPtr = whirl::hip::DevPtr;

struct MtpProbe : q::Context {
    using q::Context::argmaxRowsEnqueue;  // v0252-f1arg3
    using q::Context::draftHeadProblem;
    using q::Context::out_tok;  // Scratch (a protected base of Context)
    using q::Context::xq_src;
};

whirl::hip::Function fn(std::uintptr_t i) { return reinterpret_cast<whirl::hip::Function>(i); }

std::string codeOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const q::ModelError& e) {
        return e.code();
    } catch (const std::logic_error&) {
        return "device";
    }
    return "";
}

void testCtlSlot() {
    auto c = std::make_unique<MtpProbe>();
    c->out_tok = 0x1000;
    c->ctl_off = 3;
    ISO_CHECK(c->ctlSlot(0) == 0x1000 + 4 * 3);
    ISO_CHECK(c->ctlSlot(2) == 0x1000 + 4 * 5);  // word w of the current sequence's control block, 4 bytes a word
}

void testRefusals() {
    auto c = std::make_unique<MtpProbe>();
    const std::uint32_t t[1] = {1};
    ISO_CHECK(codeOf([&] { c->mtpEnqueue(0, t, 0, 0, std::nullopt); }) == "NoMtp");
    ISO_CHECK(codeOf([&] { (void)c->verifyBatchEnqueue({}); }) == "BadBatch");
    std::vector<q::VSeg> many(q::gdn_max_seg + 1);
    ISO_CHECK(codeOf([&] { (void)c->verifyBatchEnqueue(many); }) == "BadBatch");
}

void testCommitSeq() {
    auto c = std::make_unique<MtpProbe>();
    c->seqs.resize(2);
    c->seqs[1].npend = 2;
    c->commitSeq(1);  // not in replay mode: nothing to apply
    ISO_CHECK(c->seqs[1].npend == 2);
    c->gdn_replay = true;
    c->commitSeq(5);  // no such sequence
    ISO_CHECK(c->seqs[1].npend == 2);
    // two layers, one DeltaNet: one conv + one recurrent-step launch for it
    c->cfg.d_conv = 4, c->cfg.n_k_heads = 2, c->cfg.d_state = 8, c->cfg.d_inner = 256, c->cfg.n_v_heads = 4;
    c->layers.resize(2);
    c->layers[0].kind = q::LayerKind::attn;
    c->layers[1].kind = q::LayerKind::gdn;
    std::vector<DevPtr> conv{0, 0x100}, ssm{0, 0x200};
    c->seqs[1].conv = conv;
    c->seqs[1].ssm = ssm;
    c->gdn_ord = {0, 0};
    c->k.gdn_conv_l2 = fn(0x10);
    c->k.gdn_step_norm = fn(0x20);
    c->xq_src = 7;
    fake::recordLaunches(true);
    c->commitSeq(1);
    const auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l.size() == 2);
    if (l.size() == 2) {
        ISO_CHECK(l[0].f == fn(0x10) && l[0].grid.x == c->cfg.convCh() / 128 && l[0].block.x == 128);
        ISO_CHECK(l[1].f == fn(0x20) && l[1].grid.x == 4 && l[1].block.x == 512);
    }
    ISO_CHECK(c->seqs[1].npend == 0 && c->xq_src == 0);  // applied once: the pending rows are consumed
    fake::recordLaunches(true);
    c->commitSeq(1);
    ISO_CHECK(fake::recordedLaunches().empty());  // nothing pending: no launch
    fake::recordLaunches(false);
}

void testDraftHeadProblem() {
    auto c = std::make_unique<MtpProbe>();
    q::Mat head;
    head.nrows = 1000;
    c->draft_rows = 300;
    c->draft_vocab = 400;
    const auto p = c->draftHeadProblem(head);
    ISO_CHECK(p.head_rows == 1000 && p.draft_rows == 300 && p.draft_vocab == 400 && !p.d2 && !p.map);
}

// v0252-f1 (relaxacc one-pass check): the per-row draft source of argmax_rank_rows_to, the one-shot request and its launch
void testRelaxRank() {
    auto c = std::make_unique<MtpProbe>();
    const std::uint32_t lit[2] = {5, 6};
    std::vector<q::VSeg> segs(3);
    segs[0].seq = 2, segs[0].nd = 3;                                         // MTP drafts: rows 0..3
    segs[1].seq = 0, segs[1].nd = 2, segs[1].drafts = std::span(lit, 2);    // n-gram (host) drafts: rows 4..6
    segs[2].seq = 1, segs[2].nd = 2;                                         // MTP drafts: rows 7..9
    const auto all = q::Context::relaxRankSrc(segs, 0x7);
    const int cw = static_cast<int>(q::ctl_words), cd = static_cast<int>(q::ctl_drafts);
    const int want[10] = {2 * cw + cd, 2 * cw + cd + 1, 2 * cw + cd + 2, -1, -1, -1, -1, cw + cd, cw + cd + 1, -1};
    bool ok = true;
    for (int r = 0; r < 10; ++r) ok = ok && all.v[r] == want[r];
    for (int r = 10; r < 32; ++r) ok = ok && all.v[r] == -1;  // rows past the batch
    ISO_CHECK(ok);
    const auto first = q::Context::relaxRankSrc(segs, 0x1);  // segment 2 not in the mask
    ISO_CHECK(first.v[0] == 2 * cw + cd && first.v[2] == 2 * cw + cd + 2 && first.v[7] == -1 && first.v[8] == -1);
    const auto none = q::Context::relaxRankSrc(segs, 0x2);  // only the n-gram segment: no row
    bool n0 = true;
    for (int r = 0; r < 32; ++r) n0 = n0 && none.v[r] == -1;
    ISO_CHECK(n0);
    // the request: refused without the kernel, without a buffer or without a segment; one-shot
    ISO_CHECK(!c->setRelaxRank(0x5000, 1, 32));  // no argmax_rank_rows_to in the table
    ISO_CHECK(c->takeRelaxRank().stats == 0);
    c->k.argmax_rank_rows_to = fn(0x30);
    c->k.argmax_rows_to = fn(0x31);
    ISO_CHECK(!c->setRelaxRank(0, 1, 32) && !c->setRelaxRank(0x5000, 0, 32));
    ISO_CHECK(c->setRelaxRank(0x5000, 0x5, 32));
    const q::Context::RelaxRank rr = c->takeRelaxRank();
    ISO_CHECK(rr.stats == 0x5000 && rr.mask == 0x5 && rr.stride == 32);
    ISO_CHECK(c->takeRelaxRank().stats == 0);  // taken: cleared
    // a verify that throws still consumes the request (no stale request for the next verify)
    ISO_CHECK(c->setRelaxRank(0x5000, 1, 32));
    ISO_CHECK(codeOf([&] { (void)c->verifyBatchEnqueue({}); }) == "BadBatch");
    ISO_CHECK(c->takeRelaxRank().stats == 0);
    ISO_CHECK(c->setRelaxRank(0x5000, 1, 32));
    c->max_ctx = 4;
    ISO_CHECK(codeOf([&] { c->verifyEnqueueEx(1, 4, std::nullopt, 2); }) == "ContextTooLong");
    ISO_CHECK(c->takeRelaxRank().stats == 0);
    // the launch: only with a request and a draft row; one block of 1024 per verify row
    c->cfg.n_vocab = 1000;
    q::RowIdx dst, src;
    for (int r = 0; r < 32; ++r) dst.v[r] = r, src.v[r] = -1;
    fake::recordLaunches(true);
    ISO_CHECK(!c->relaxRankEnqueue({}, 4, 0x100, dst, src));
    ISO_CHECK(!c->relaxRankEnqueue(rr, 4, 0x100, dst, src));  // no draft row
    src.v[2] = 7;
    ISO_CHECK(!c->relaxRankEnqueue({}, 4, 0x100, dst, src));  // no request
    ISO_CHECK(!c->relaxRankEnqueue(rr, 2, 0x100, dst, src));  // the draft row is past n
    ISO_CHECK(c->relaxRankEnqueue(rr, 4, 0x100, dst, src));
    const auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l.size() == 1);
    if (l.size() == 1) ISO_CHECK(l[0].f == fn(0x30) && l[0].grid.x == 4 && l[0].block.x == 1024);
    // v0252-f1flip (ADR-0015 Revision 1): the one-pass check is enabled per GPU family (device::Profile::relax_rank_check)
    // and is now on for gfx1151 (the Radeon 8060S) as well as gfx1201: the request is taken and the rank launch happens
    // (before f1flip: refused there, relaxacc kept topk_rows, no rank launch even with a stale request)
    c->arch = whirl::hip::Arch::gfx1151;
    ISO_CHECK(c->setRelaxRank(0x5000, 0x5, 32));
    ISO_CHECK(c->takeRelaxRank().stats == 0x5000);
    fake::recordLaunches(true);
    ISO_CHECK(c->relaxRankEnqueue(rr, 4, 0x100, dst, src));
    const auto l51 = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l51.size() == 1);
    if (l51.size() == 1) ISO_CHECK(l51[0].f == fn(0x30) && l51[0].grid.x == 4 && l51[0].block.x == 1024);
    c->arch = whirl::hip::Arch::gfx1201;
    ISO_CHECK(c->setRelaxRank(0x5000, 0x5, 32));
    ISO_CHECK(c->takeRelaxRank().stats == 0x5000);
}

// v0252-e1p1b (E1, PLAN V1-93 P1b): the replay count from the NPEND control word (setPending npend_dev), the word address of
// any sequence, and the E1 refusals (a device-input segment without the E1 kernels, with literal drafts or in snapshot mode)
void testE1Inputs() {
    auto c = std::make_unique<MtpProbe>();
    c->out_tok = 0x8000;
    ISO_CHECK(c->ctlWord(3, q::ctl_npend) == 0x8000 + 4 * (3 * q::ctl_words + q::ctl_npend));
    ISO_CHECK(q::ctl_acc == 50 && q::ctl_chosen == 51 && q::ctl_npos == 54 && q::ctl_npend == 57 && q::ctl_npend < q::ctl_words);
    c->seqs.resize(4);
    q::GdnSeg g;
    c->seqs[2].npend = 3;
    c->replayCount(g, 2);
    ISO_CHECK(g.npend == 3 && g.snap[0] == 0);  // by value: the snapshot pointers stay unused
    c->seqs[2].npend = q::npend_dev;
    q::GdnSeg gd;
    c->replayCount(gd, 2);
    ISO_CHECK(gd.npend == -1 && gd.snap[0] == c->ctlWord(2, q::ctl_npend));
    // commitSeq with the device count: the same two launches as by value, and the pending marker is consumed
    c->gdn_replay = true;
    c->cfg.d_conv = 4, c->cfg.n_k_heads = 2, c->cfg.d_state = 8, c->cfg.d_inner = 256, c->cfg.n_v_heads = 4;
    c->layers.resize(2);
    c->layers[0].kind = q::LayerKind::attn;
    c->layers[1].kind = q::LayerKind::gdn;
    std::vector<DevPtr> conv{0, 0x100}, ssm{0, 0x200};
    c->seqs[2].conv = conv;
    c->seqs[2].ssm = ssm;
    c->gdn_ord = {0, 0};
    c->k.gdn_conv_l2 = fn(0x10);
    c->k.gdn_step_norm = fn(0x20);
    fake::recordLaunches(true);
    c->commitSeq(2);
    const auto l = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l.size() == 2 && c->seqs[2].npend == 0);
    // refusals: an E1 segment needs the kernels (set_rows_dp / rmsnorm_q8_rows_dv), MTP drafts and replay mode
    std::vector<q::VSeg> vs(1);
    vs[0].seq = 1, vs[0].nd = 1, vs[0].pos = 10, vs[0].dev = true;  // nd 1: the probe has no fused decode (unfusedVerifyOk)
    c->max_ctx = 4096;
    ISO_CHECK(codeOf([&] { (void)c->verifyBatchEnqueue(vs); }) == "BadBatch");  // no set_rows_dp
    c->k.set_rows_dp = fn(0x40);
    const std::uint32_t lit[1] = {5};
    vs[0].drafts = std::span(lit, 1);
    ISO_CHECK(codeOf([&] { (void)c->verifyBatchEnqueue(vs); }) == "BadBatch");  // literal drafts
    std::vector<q::MSeg> ms(1);
    ms[0].seq = 1, ms[0].e1_rows = 3, ms[0].pend_pos = 11;
    c->mtp.emplace();
    ISO_CHECK(codeOf([&] { c->mtpBatchStepEx(ms, 0, true); }) == "BadBatch");  // no rmsnorm_q8_rows_dv
}

// v0252-f1arg3 (PLAN V1-63 / V1-64): the R9700 host switch of the multi-block verify argmax twins argmax_rows_to_mb /
// argmax_rank_rows_to_mb (gfx1201 code object). Rules of progress/v0252-f1arg2.HANDOFF.md section 6 (C2): the switch is
// device::Profile::argmax_mb (the branch: gfx1201 on, gfx1151 off; merged off on both) with the env override WHIRL_ARGMAX_MB (0 off, 1 on, 2 on + the
// tripwire), read once per Context; the 512 B workspace is a member of the Context that owns the launching stream,
// allocated by that Context alone directly before its control words (one allocation, zeroed once); every twin launch
// uses the launching Context's stream and workspace, grid AM_TILES(n_vocab) x rows, AM_THREADS threads; off (or a code
// object without the twins) launches the one-block kernels exactly as before.
void testArgmaxMb() {
    namespace dev = whirl::device;
    using whirl::hip::Arch;
    // the mode: the profile default, then the envFlag rule of the override ("0" off, any other value on), "2" adds the tripwire
    // v0252-b5 round 2c: merged off on both cards (the gfx1201 default comes with the defaults round, PLAN V1-117); the
    // switch itself is exercised below through argmaxMbMode / setArgmaxMb
    ISO_CHECK(!dev::profile(Arch::gfx1201).argmax_mb && !dev::profile(Arch::gfx1151).argmax_mb);
    ISO_CHECK(q::Context::argmaxMbMode(true, std::nullopt) == 1 && q::Context::argmaxMbMode(false, std::nullopt) == 0);
    ISO_CHECK(q::Context::argmaxMbMode(true, std::string("0")) == 0 && q::Context::argmaxMbMode(false, std::string("1")) == 1);
    ISO_CHECK(q::Context::argmaxMbMode(false, std::string("2")) == 2 && q::Context::argmaxMbMode(true, std::string("")) == 1);
    const whirl::hip::Function rows = fn(0x32), rows_to = fn(0x31), rank = fn(0x30), mb = fn(0x40), rank_mb = fn(0x41);
    auto setup = [&](MtpProbe& c, std::uintptr_t s, bool twins) {
        c.arch = Arch::gfx1201;
        c.cfg.n_vocab = 248320;  // Qwen3.x vocabulary: AM_TILES = 61
        c.k.argmax_rows = rows, c.k.argmax_rows_to = rows_to, c.k.argmax_rank_rows_to = rank;
        if (twins) c.k.argmax_rows_to_mb = mb, c.k.argmax_rank_rows_to_mb = rank_mb;
        c.stream = reinterpret_cast<whirl::hip::Stream>(s);
    };
    auto a = std::make_unique<MtpProbe>(), b = std::make_unique<MtpProbe>(), none = std::make_unique<MtpProbe>();
    setup(*a, 0x11, true), setup(*b, 0x22, true), setup(*none, 0x33, false);
    // a code object without the twins (gfx1151, an old gfx1201 object): off whatever is asked
    none->setArgmaxMb(2);
    ISO_CHECK(none->argmax_mb == 0);
    fake::recordAllocs(true);
    a->setArgmaxMb(1);
    b->setArgmaxMb(2);
    none->out_tok = none->allocCtl(256);
    a->out_tok = a->allocCtl(256);
    b->out_tok = b->allocCtl(256);
    const auto sizes = fake::recordedSizes();
    fake::recordAllocs(false);
    ISO_CHECK(a->argmax_mb == 1 && b->argmax_mb == 2 && a->argmaxMbOn() && b->argmaxMbOn() && !none->argmaxMbOn());
    // one workspace per Context (per stream), directly before its control words, in the same allocation
    ISO_CHECK(none->am_ws == 0);
    ISO_CHECK(a->am_ws != 0 && b->am_ws != 0 && a->am_ws != b->am_ws);
    ISO_CHECK(a->out_tok == a->am_ws + q::am_ws_bytes && b->out_tok == b->am_ws + q::am_ws_bytes);
    ISO_CHECK(sizes.size() == 3 && sizes[0] == 256 && sizes[1] == 256 + q::am_ws_bytes && sizes[2] == 256 + q::am_ws_bytes);
    // the launches: every twin launch on the launching Context's stream with its own workspace (argument 4 of
    // argmax_rows_to_mb, 7 of argmax_rank_rows_to_mb), grid (AM_TILES(n_vocab), rows) x 256
    fake::recordArgWord(mb, 4);
    fake::recordArgWord(rank_mb, 7);
    q::RowIdx dst, src;
    for (int r = 0; r < 32; ++r) dst.v[r] = r, src.v[r] = -1;
    src.v[1] = 40;
    const q::Context::RelaxRank rr{0x5000, 1, 32};
    fake::recordLaunches(true);
    a->argmaxRowsEnqueue(4);                                  // verifyEnqueueEx / verify: was argmax_rows
    b->argmaxToEnqueue(3, 0x900, dst);                        // verifyBatchEnqueue: was argmax_rows_to
    ISO_CHECK(a->relaxRankEnqueue(rr, 4, 0x900, dst, src));   // was argmax_rank_rows_to
    b->argmaxRowsEnqueue(1);
    none->argmaxRowsEnqueue(2);                               // no twins: the one-block kernels
    none->argmaxToEnqueue(2, 0x900, dst);
    ISO_CHECK(none->relaxRankEnqueue(rr, 2, 0x900, dst, src));
    const auto l = fake::recordedLaunches();
    const auto x = fake::recordedLaunchCtx();
    fake::recordLaunches(false);
    ISO_CHECK(l.size() == 7 && x.size() == 7);
    if (l.size() == 7 && x.size() == 7) {
        auto twin = [&](std::size_t i, whirl::hip::Function f, unsigned n, const MtpProbe& c) {
            return l[i].f == f && l[i].grid.x == 61 && l[i].grid.y == n && l[i].block.x == 256 && x[i].s == c.stream && x[i].word == c.am_ws;
        };
        ISO_CHECK(twin(0, mb, 4, *a));
        ISO_CHECK(twin(1, mb, 3, *b));
        ISO_CHECK(twin(2, rank_mb, 4, *a));
        ISO_CHECK(twin(3, mb, 1, *b));
        auto old = [&](std::size_t i, whirl::hip::Function f) {
            return l[i].f == f && l[i].grid.x == 2 && l[i].grid.y == 1 && l[i].block.x == 1024 && x[i].s == none->stream;
        };
        ISO_CHECK(old(4, rows) && old(5, rows_to) && old(6, rank));
    }
    // switched off by the override: the same Context launches the one-block kernels again
    a->setArgmaxMb(0);
    ISO_CHECK(!a->argmaxMbOn());
    fake::recordLaunches(true);
    a->argmaxRowsEnqueue(4);
    a->argmaxToEnqueue(4, 0x900, dst);
    ISO_CHECK(a->relaxRankEnqueue(rr, 4, 0x900, dst, src));
    const auto l0 = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l0.size() == 3);
    if (l0.size() == 3) ISO_CHECK(l0[0].f == rows && l0[1].f == rows_to && l0[2].f == rank && l0[0].grid.x == 4 && l0[0].block.x == 1024);
    // more rows than one workspace has: refused, nothing launched
    fake::recordLaunches(true);
    ISO_CHECK(codeOf([&] { b->argmaxToEnqueue(33, 0x900, dst); }) == "BatchTooLarge");
    ISO_CHECK(fake::recordedLaunches().empty());
    fake::recordLaunches(false);
    // the tripwire: a workspace left non-zero after a cycle fails loudly
    std::vector<std::int32_t> ws(q::am_ws_bytes / 4, 0);
    ISO_CHECK(codeOf([&] { q::Context::checkAmWs(ws); }).empty());
    ws[q::am_ws_bytes / 4 - 1] = 1;
    ISO_CHECK(codeOf([&] { q::Context::checkAmWs(ws); }) == "ArgmaxWsDirty");
    // the Radeon 8060S: the profile keeps the one-block kernels (its code object has no twins either)
    auto g = std::make_unique<MtpProbe>();
    setup(*g, 0x44, true);
    g->arch = Arch::gfx1151;
    g->setArgmaxMb(q::Context::argmaxMbMode(dev::profile(Arch::gfx1151).argmax_mb, std::nullopt));
    ISO_CHECK(g->argmax_mb == 0 && !g->argmaxMbOn());
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.mtp_verify",
                    {{"ctl_slot", testCtlSlot}, {"refusals", testRefusals}, {"commit_seq", testCommitSeq}, {"draft_head_problem", testDraftHeadProblem},
                     {"relax_rank", testRelaxRank}, {"e1_inputs", testE1Inputs}, {"argmax_mb", testArgmaxMb}},
                    argc, argv);
}
