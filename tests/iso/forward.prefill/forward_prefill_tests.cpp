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

namespace {

namespace q = whirl::qwen35;

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

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.prefill",
                    {{"can_segment", testCanSegment}, {"prefill_segs_batch", testPrefillSegsBatch}, {"seg_logits_to_front", testSegLogitsToFront},
                     {"prefill_chunks", testPrefillChunks}},
                    argc, argv);
}
