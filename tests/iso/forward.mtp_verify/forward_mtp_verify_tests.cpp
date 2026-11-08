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
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;
using DevPtr = whirl::hip::DevPtr;

struct MtpProbe : q::Context {
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
    // v0252-f2def2: the one-pass check is enabled per GPU family (device::Profile::relax_rank_check, gfx1201 only). The
    // gfx1151 code object has argmax_rank_rows_to too, but F1 was never measured on the Radeon 8060S (ADR-0015, no G4a):
    // there the request is refused (relaxacc keeps topk_rows) and no rank launch happens, even with a stale request
    c->arch = whirl::hip::Arch::gfx1151;
    ISO_CHECK(!c->setRelaxRank(0x5000, 0x5, 32));
    ISO_CHECK(c->takeRelaxRank().stats == 0);
    fake::recordLaunches(true);
    ISO_CHECK(!c->relaxRankEnqueue(rr, 4, 0x100, dst, src));
    const auto l51 = fake::recordedLaunches();
    fake::recordLaunches(false);
    ISO_CHECK(l51.empty());
    c->arch = whirl::hip::Arch::gfx1201;
    ISO_CHECK(c->setRelaxRank(0x5000, 0x5, 32));
    ISO_CHECK(c->takeRelaxRank().stats == 0x5000);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.mtp_verify",
                    {{"ctl_slot", testCtlSlot}, {"refusals", testRefusals}, {"commit_seq", testCommitSeq}, {"draft_head_problem", testDraftHeadProblem},
                     {"relax_rank", testRelaxRank}},
                    argc, argv);
}
