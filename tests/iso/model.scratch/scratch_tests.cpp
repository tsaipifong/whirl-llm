// whirl-iso-model-scratch-tests: src/model/qwen35/scratch.cpp built alone (ARCH-1 8.5 isolation, arch025-lane2 N9 S2c):
// bufferBytes, the pre-load estimate of Model::load's prefill / decode buffers, as a pure function of the config. That
// it covers what Model::allocScratch (model_scratch.cpp) allocates stays a load-time self-check (load_note warning).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/scratch.h"
#include "whirl/model_types.h"
#include "whirl/ops/attention.h"

#include <cstdint>

namespace {

namespace q = whirl::qwen35;
using u64 = std::uint64_t;

q::Config dense() {
    q::Config c;
    c.n_layer = 64;
    c.n_embd = 5120;
    c.n_ff = 17408;
    c.n_head = 24;
    c.n_head_kv = 4;
    c.head_dim = 256;
    c.d_state = 128;
    c.n_k_heads = 16;
    c.n_v_heads = 48;
    c.d_inner = 6144;
    c.n_vocab = 248320;
    return c;
}

void testFdSplitsTerm() {
    // fd_splits sizes part_ml and part_acc only: max_verify_rows * n_head * (2 + head_dim) * 4 bytes per split
    const q::Config c = dense();
    const u64 a = q::bufferBytes(c, 512, 17408, 1u << 20, 64), b = q::bufferBytes(c, 512, 17408, 1u << 20, 256);
    ISO_CHECK(b - a == 192ull * q::max_verify_rows * c.n_head * (2 + c.head_dim) * 4);
    ISO_CHECK(q::bufferBytes(c, 512, 17408, 1u << 20) == a);  // default 64
}

void testMaxElemsAndFfsTerms() {
    const q::Config c = dense();
    const u64 base = q::bufferBytes(c, 256, 17408, 1000, 64);
    ISO_CHECK(q::bufferBytes(c, 256, 17408, 1100, 64) - base == 200);         // w16: f16 per element
    ISO_CHECK(q::bufferBytes(c, 256, 17409, 1000, 64) - base == 256ull * 8);  // ffn_g + ffn_u: 2 x f32 per row
}

void testDqPtabTerm() {
    // dq_ptab: identity pages for up to ops::kDqMaxKeys keys (Model::kDqMaxKeys is its alias)
    q::Config c;  // all zero: only the size-independent terms remain
    const u64 n = q::bufferBytes(c, 0, 0, 0, 64);
    ISO_CHECK(n == (static_cast<u64>(whirl::ops::kDqMaxKeys) / 256 + 1) * 4 + q::ctl_words * 4 + 256 + 8 * 4);
}

void testMoeAddsExpertBuffers() {
    q::Config c = dense();
    const u64 d = q::bufferBytes(c, 128, 17408, 0, 64);
    c.moe = true;
    c.n_expert = 256;
    c.n_expert_used = 8;
    c.n_ff_exp = 512;
    ISO_CHECK(q::bufferBytes(c, 128, 17408, 0, 64) > d + 128ull * 256 * 4);  // at least moe_logits
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.scratch",
                    {{"fd_splits_term", testFdSplitsTerm},
                     {"max_elems_and_ffs_terms", testMaxElemsAndFfsTerms},
                     {"dq_ptab_term", testDqPtabTerm},
                     {"moe_adds_expert_buffers", testMoeAddsExpertBuffers}},
                    argc, argv);
}
