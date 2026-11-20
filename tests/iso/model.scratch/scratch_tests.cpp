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
    // v0252-f1arg3: + the argmax workspace (q::am_ws_bytes) that Context::allocCtl places before the control words; the
    // estimate counts it on every GPU (an upper bound where the switch is off), so it is size-independent too
    ISO_CHECK(n == (static_cast<u64>(whirl::ops::kDqMaxKeys) / 256 + 1) * 4 + q::ctl_words * 4 + q::am_ws_bytes + 256 + 8 * 4);
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

// v0252-kvlong2 (O1, PLAN V1-137): the q8 / q8h / q4 prefill dequant copy (ffn_g / ffn_u) covers the configured context.
// Before O1 it held max_batch * ffs * 4 / (n_head_kv * head_dim * 2) rows whatever the context (131072 on Ornith), so a
// 256k prefill split into key ranges and fell back from attn_kgk to attn_kgs (v0252-kvlong:q4-prefill-dq-scratch-cliff).
q::Config ornith() {  // the attention / FFN-scratch shape of Ornith-1.5-35B-A3B (ffs 8192: the q / gate projection rows)
    q::Config c;
    c.n_head = 16;
    c.n_head_kv = 2;
    c.head_dim = 256;
    return c;
}

void testDqCopyCoversContext() {
    const q::Config o = ornith();
    const u64 base = 4096ull * 8192 * 4;  // B * ffs * f32: what ffn_g / ffn_u hold for the FFN
    // no context asked (ctx 0) or a context the FFN scratch already covers: the old rows and sizes, nothing extra
    for (const std::uint32_t ctx : {0u, 4096u, 100000u, 131072u}) {
        const q::DqCopy d = q::dqCopyPlan(o, 4096, 8192, ctx);
        ISO_CHECK(d.rows == 131072 && d.bytes_each == base && d.extra == 0);
    }
    // 200000 / 262144 keys: the copy grows to the context (rounded up to 256 rows), ffn_g and ffn_u both
    {
        const q::DqCopy d = q::dqCopyPlan(o, 4096, 8192, 200000);
        ISO_CHECK(d.rows >= 200000 && d.rows % 256 == 0 && d.rows < 200000 + 256);
        ISO_CHECK(d.bytes_each == static_cast<u64>(d.rows) * 2 * 256 * 2 && d.extra == 2 * (d.bytes_each - base));
    }
    {
        const q::DqCopy d = q::dqCopyPlan(o, 4096, 8192, 262144);
        ISO_CHECK(d.rows == 262144 && d.bytes_each == (256ull << 20) && d.extra == (256ull << 20));  // Ornith 256k: +256 MiB
    }
    // Swift-1.5 shape (n_head_kv 4, ffs 17408): 139264 rows before; 256k needs +480 MiB
    {
        q::Config s = ornith();
        s.n_head = 24;
        s.n_head_kv = 4;
        const q::DqCopy d0 = q::dqCopyPlan(s, 4096, 17408, 0), d = q::dqCopyPlan(s, 4096, 17408, 262144);
        ISO_CHECK(d0.rows == 139264 && d0.extra == 0);
        ISO_CHECK(d.rows == 262144 && d.extra == 2 * (262144ull * 2048 - 4096ull * 17408 * 4));
    }
    // never past the dequant page table (kDqMaxKeys keys)
    ISO_CHECK(q::dqCopyPlan(o, 4096, 8192, whirl::ops::kDqMaxKeys + 1000).rows == whirl::ops::kDqMaxKeys);
    // fit: the extra bytes, what the load still allocates and a 1 GiB margin must all fit in the free VRAM
    const u64 g = 1ull << 30;
    ISO_CHECK(q::dqCopyFits(256ull << 20, 4 * g, 2 * g));
    ISO_CHECK(!q::dqCopyFits(256ull << 20, 3 * g, 2 * g));
    ISO_CHECK(q::dqCopyFits(0, 0, 0));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.scratch",
                    {{"fd_splits_term", testFdSplitsTerm},
                     {"max_elems_and_ffs_terms", testMaxElemsAndFfsTerms},
                     {"dq_ptab_term", testDqPtabTerm},
                     {"moe_adds_expert_buffers", testMoeAddsExpertBuffers},
                     {"dq_copy_covers_context", testDqCopyCoversContext}},
                    argc, argv);
}
