// whirl-iso-forward-prefill-tests: the module forward.prefill (src/model/qwen35/forward/prefill.cpp: prefill with MTP, segmented
// multi-sequence prefill, plain prefill, argmax and logits readback; the top of the forward call graph) built over every forward
// file it calls (run, mtp_verify, attn, ffn, gdn, dispatch, vision_rows) and forward.state (ARCH-1 8.5 isolation, arch025-n8b3,
// PLAN N8 batch 3): a call from prefill.cpp into class Model, the engine or the server log is LNK2019 here. Host-only cases
// (argument checks and chunking before any launch) on tests/iso/fake_hip.cpp.
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

// The launch / allocation recorders of the HIP link double (tests/iso/fake_hip.cpp).
namespace whirl::hip::isofake {
struct Launch {
    Function f;
    Dim3 grid, block;
};
void recordLaunches(bool on);
std::vector<Launch> recordedLaunches();
void recordAllocs(bool on);
}  // namespace whirl::hip::isofake

namespace {

namespace q = whirl::qwen35;
namespace fake = whirl::hip::isofake;

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

void testCanSegment() {
    auto c = std::make_unique<q::Context>();
    c->seqs.resize(2);
    c->k.attn_prep = reinterpret_cast<whirl::hip::Function>(0x10);
    c->cfg.head_dim = 256;
    c->cfg.d_conv = 4;
    ISO_CHECK(c->canSegment());
    c->cfg.head_dim = 128;
    ISO_CHECK(!c->canSegment());  // the segmented attention is built for head_dim 256
    c->cfg.head_dim = 256;
    c->cfg.d_conv = 3;
    ISO_CHECK(!c->canSegment());
    c->cfg.d_conv = 4;
    c->num_plan.flags.no_fuse = true;
    ISO_CHECK(!c->canSegment());
    c->num_plan.flags.no_fuse = false;
    c->num_plan.flags.naive_attn = true;
    ISO_CHECK(!c->canSegment());
    c->num_plan.flags.naive_attn = false;
    c->seqs.resize(1);
    ISO_CHECK(!c->canSegment());  // one sequence: nothing to segment
}

void testPrefillSegsBatch() {
    auto c = std::make_unique<q::Context>();
    ISO_CHECK(codeOf([&] { c->prefillSegs({}, false); }) == "BadBatch");
    std::vector<q::PSeg> many(q::max_small_batch + 1);
    ISO_CHECK(codeOf([&] { c->prefillSegs(many, false); }) == "BadBatch");
    // arch025-n8b3 fix, boundary value of the segment-count limit: every segment below is valid in its own length
    // (n > max_small_batch), so the count check is the only possible BadBatch. The empty segments above also fail the
    // per-segment length check, which hid an off-by-one in the count check (mutant n8b3-prefill-segs-limit survived).
    c->max_ctx = q::max_small_batch + 1;
    q::PSeg ok_len;
    ok_len.n = q::max_small_batch + 1;
    // max_small_batch + 1 segments: refused by the count check before any segment is read (a limit of
    // max_small_batch + 1 would read segment 0 and throw ContextTooLong: its pos0 runs past max_ctx)
    q::PSeg past = ok_len;
    past.pos0 = 1;
    const std::vector<q::PSeg> over(q::max_small_batch + 1, past);
    ISO_CHECK(codeOf([&] { c->prefillSegs(over, false); }) == "BadBatch");
    // exactly max_small_batch segments: accepted by the count check; their rows (max_small_batch * (max_small_batch + 1))
    // exceed max_batch, a host-side refusal before any launch (a limit of max_small_batch - 1 would give BadBatch)
    c->max_batch = q::max_small_batch;
    const std::vector<q::PSeg> at(q::max_small_batch, ok_len);
    ISO_CHECK(codeOf([&] { c->prefillSegs(at, false); }) == "BatchTooLarge");
}

void testSegLogitsToFront() {
    auto c = std::make_unique<q::Context>();
    ISO_CHECK(codeOf([&] { c->segLogitsToFront(0); }).empty());  // row 0 is already in front: no copy
    ISO_CHECK(codeOf([&] { c->segLogitsToFront(1); }) == "device");
}

void testPrefillChunks() {
    auto c = std::make_unique<q::Context>();
    c->max_batch = 4;
    c->max_ctx = 100;
    c->host_ids.assign(2 * 64, 0);
    const std::vector<std::uint32_t> none, six(6, 3);
    ISO_CHECK(codeOf([&] { c->prefill(none, 0); }).empty());  // nothing to run
    ISO_CHECK(codeOf([&] { c->prefill(six, 0); }) == "device");  // first chunk = max_batch rows: accepted by forward
    ISO_CHECK(codeOf([&] { c->prefillWithMtp(six); }) == "device");
}

// v0252-mtpkv0 (PLAN V1-51): the MTP block's attention reads its own KV from position 0 (attn.cpp: positions up to
// pos_buf[t]; attn_dq4.h: the draft window's sink [0, AW_SINK)), so the MTP KV row 0 must be written before any MTP row
// reads it. The bug: a cold prefill (position 0, no previous hidden) ran the MTP rows from position 1 only, so row 0
// kept whatever the recycled KV page held and the drafts depended on the page's history (fast mode: run-to-run text).
// The reference (llama.cpp common/speculative.cpp pending_h, ik_llama.cpp speculative.cpp; progress\v0252-mtpkv0.
// PRIOR_ART.md): the MTP row at position 0 pairs a ZERO hidden with token 0. Observed on the HIP link double: launches
// are recorded, every attention block calls the test hook first; the hook records the rows' positions (row_pos) and
// stops the forward there (nothing after the attention entry is needed). The draft window marks an MTP attention
// (draftAttn passes draftWindowArg; a trunk block 0); the context has no trunk layers.
struct StopAtAttn {};
struct AttnCall {
    std::int32_t win = 0;
    std::vector<std::int32_t> pos;
};
std::vector<AttnCall> g_attn;

void attnRowsHook(void* ctx, std::int32_t win) {
    const auto* c = static_cast<const q::Context*>(ctx);
    AttnCall a;
    a.win = win;
    for (std::uint32_t r = 0; r < c->row_n; ++r) a.pos.push_back(c->row_pos[r]);
    g_attn.push_back(a);
    throw StopAtAttn{};
}

struct MtpProbe : q::Context {
    using q::Context::mtp_h;
};

constexpr std::uint32_t kE = 256;
const std::int32_t kWin = q::draftWindowArg(200, 8192);

// fake device buffers (allocation recorder on while the context is set up): mtp_h and every sequence's hidden rows
std::unique_ptr<MtpProbe> mtpContext(std::uint32_t n_seqs) {
    auto c = std::make_unique<MtpProbe>();
    c->cfg.n_embd = kE;
    c->cfg.n_vocab = 1000;
    c->max_batch = 64;
    c->max_ctx = 1024;
    c->host_ids.assign(2 * 64, 0);
    c->mtp.emplace();
    c->num_plan.flags.draft_window = 200;
    c->num_plan.flags.draft_window_min = 8192;
    c->attn_test_hook = attnRowsHook;
    c->attn_test_ctx = static_cast<q::Context*>(c.get());
    fake::recordAllocs(true);
    c->mtp_h = whirl::hip::malloc(static_cast<std::size_t>(q::max_small_batch) * kE * 4);
    c->seqs.resize(n_seqs);
    for (q::Seq& s : c->seqs) s.hid = whirl::hip::malloc(static_cast<std::size_t>(q::max_small_batch) * kE * 4);
    fake::recordAllocs(false);
    return c;
}

// run f with the launch recorder on; true when an attention block was reached (the hook stopped it)
bool runToAttn(const std::function<void()>& f) {
    g_attn.clear();
    fake::recordLaunches(true);
    bool stopped = false;
    try {
        f();
    } catch (const StopAtAttn&) {
        stopped = true;
    } catch (const std::exception&) {
        // a device call of the link double (the trunk forward's uploads): no attention ran
    }
    fake::recordLaunches(false);
    return stopped;
}

void testMtpRow0ColdChunk() {
    // a cold prompt's first chunk (position 0, no previous hidden): the first MTP pass is the row at position 0
    auto c = mtpContext(1);
    const std::vector<std::uint32_t> toks = {11, 12};
    const bool stopped = runToAttn([&] { c->prefillMtpChunk(toks, 0, 0, toks.size(), std::nullopt); });
    ISO_CHECK(stopped);
    ISO_CHECK(!g_attn.empty() && g_attn[0].win == kWin);  // the first attention block is the MTP block's
    // V1-51: the first MTP pass was the rows from position 1 (pos 1 reads the never-written row 0)
    ISO_CHECK(!g_attn.empty() && g_attn[0].pos == std::vector<std::int32_t>{0});
    // prefillWithMtp (the CLI's chunk loop) starts the same way
    auto c2 = mtpContext(1);
    const bool stopped2 = runToAttn([&] { c2->prefillWithMtp(toks); });
    ISO_CHECK(stopped2 && !g_attn.empty() && g_attn[0].win == kWin && g_attn[0].pos == std::vector<std::int32_t>{0});
}

void testMtpRow0SegmentedPrefill() {
    // the segmented prefill (server: several prompts in one forward): a cold segment gets its MTP row 0 with the boundary
    // rows (one row each, before the segments' forward reads anything), in the sequence of that segment
    auto c = mtpContext(2);
    std::vector<std::uint32_t> toks(q::max_small_batch + 1, 7);
    q::PSeg sg;
    sg.seq = 1;
    sg.tokens = toks;
    sg.pos0 = 0;
    sg.off = 0;
    sg.n = toks.size();
    const std::vector<q::PSeg> segs = {sg};
    (void)runToAttn([&] { c->prefillSegs(segs, true); });
    // V1-51: no MTP pass covered position 0 before the forward
    ISO_CHECK(!g_attn.empty() && g_attn[0].win == kWin && g_attn[0].pos == std::vector<std::int32_t>{0});
    ISO_CHECK(c->cur_seq == 1);  // the row went to the segment's own sequence (its KV region and hidden rows)
}

void testMtpLaterChunkBoundaryRow() {
    // not a cold start (unchanged by V1-51): a later chunk's first MTP pass is the boundary row at its first position
    // (the previous chunk's last hidden with tokens[off]); no MTP pass at position 0
    auto c = mtpContext(1);
    const std::vector<std::uint32_t> toks = {11, 12, 13, 14};
    const bool stopped = runToAttn([&] { c->prefillMtpChunk(toks, 0, 2, 2, std::nullopt); });
    ISO_CHECK(stopped && !g_attn.empty() && g_attn[0].win == kWin && g_attn[0].pos == std::vector<std::int32_t>{2});
    // a prefix-cache resume chunk (pos0 > 0): the first MTP pass of its second chunk is at pos0 + off, never 0
    auto c2 = mtpContext(1);
    const std::vector<std::uint32_t> rest = {21, 22, 23, 24};
    const bool stopped2 = runToAttn([&] { c2->prefillMtpChunk(rest, 300, 2, 2, std::nullopt); });
    ISO_CHECK(stopped2 && !g_attn.empty() && g_attn[0].pos == std::vector<std::int32_t>{302});
    // a first chunk at pos0 > 0 without a hidden is no cold start: its first MTP pass is its row pos0 + 1 (the row at
    // position 0 belongs to the sequence's own start only)
    auto c3 = mtpContext(1);
    const bool stopped3 = runToAttn([&] { c3->prefillMtpChunk(rest, 300, 0, 2, std::nullopt); });
    ISO_CHECK(stopped3 && !g_attn.empty() && g_attn[0].pos == std::vector<std::int32_t>{301});
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.prefill",
                    {{"can_segment", testCanSegment}, {"prefill_segs_batch", testPrefillSegsBatch}, {"seg_logits_to_front", testSegLogitsToFront},
                     {"prefill_chunks", testPrefillChunks}, {"mtp_row0_cold_chunk", testMtpRow0ColdChunk},
                     {"mtp_row0_segmented_prefill", testMtpRow0SegmentedPrefill}, {"mtp_later_chunk_boundary_row", testMtpLaterChunkBoundaryRow}},
                    argc, argv);
}
