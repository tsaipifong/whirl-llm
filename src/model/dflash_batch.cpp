// DFlash2 batched drafting (feat029-dflashbatch, DFlash T3): one drafter forward over the draft blocks of
// several slots. SPDX-License-Identifier: Apache-2.0
// Same math and kernels as Model::dflashDraft (src/model/dflash.cpp, which stays the one-block path); the
// shape follows llama.cpp common/speculative.cpp draft-dflash (MIT; idea only, no code copied): every block in
// one batch at its own positions, per-sequence attention, per-sequence selector walk. Layout and planning:
// whirl/dflash_batch.h.
//
// MERGE NOTE (feat029-dflashcost changes dflashDraft's attention / top-k kernels): the per-block attention
// below is the same launch as dflashDraft's; when dflashcost lands, take the same kernel choice here
// (DFlashRun::attn_kind: dflash_attn_part + dflash_attn_merge on this block's q / kb / vb / att offsets,
// one launch pair per block; DFlashRun::topk_kind: the split top-k needs its workspace sized for S * P rows,
// or keep dflash_topk here). Both only change proposals, never the verified output.

#include "whirl/model.h"
#include "whirl/dflash_batch.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

inline hip::Dim3 D(u32 x, u32 y = 1, u32 z = 1) { return {x, y, z}; }
inline i32 I(u64 v) { return static_cast<i32>(v); }

::whirl::dflash::batch::MemInput batchMem(const DFlashW& d, u32 vocab) {
    ::whirl::dflash::batch::MemInput in;
    const DFlashConfig& c = d.cfg;
    in.n_embd = c.n_embd;
    in.n_head = c.n_head;
    in.n_head_kv = c.n_head_kv;
    in.head_dim = c.head_dim;
    in.n_ff = c.n_ff;
    in.block = c.block;
    in.conv_k = c.conv_k;
    in.conv_group = c.conv_group;
    in.sel_topk = c.sel_topk;
    in.sel_rank = c.sel_rank;
    in.vocab = vocab;
    return in;
}

}  // namespace

u64 Model::dflashBatchBytes(u32 max_slots) const {
    if (!dflash) return 0;
    return ::whirl::dflash::batch::scratchBytes(batchMem(*dflash, static_cast<u32>(output.nrows)), max_slots);
}

void Model::dflashBatchSetup(u32 max_slots) {
    dfb.reset();
    if (!dflash || !dfr || max_slots < 2) return;
    const DFlashConfig& c = dflash->cfg;
    const u32 cap = ::whirl::dflash::batch::groupCap(dfr->block, max_slots, max_batch);
    if (cap < 2) return;  // matmul rows hold one block only
    auto r = std::make_unique<DFlashBatchRun>();
    r->slots = cap;
    const u64 S = cap, E = c.n_embd, B = dfr->block, P = B - 1, kvw = static_cast<u64>(c.n_head_kv) * c.head_dim;
    const u64 qw = static_cast<u64>(c.n_head) * c.head_dim, ds = dflash->layers[0].attn_conv_proj.nrows, V = output.nrows;
    const u64 K = c.sel_topk, R = c.sel_rank, N = S * B, NP = S * P;
    r->ids = alloc(N * 4);
    r->pos = alloc(N * 4);
    r->x = alloc(N * E * 4);
    r->h = alloc(N * E * 4);
    r->h2 = alloc(N * E * 4);
    r->dyn = alloc(N * ds * 4);
    r->q = alloc(N * qw * 4);
    r->kb = alloc(N * kvw * 4);
    r->vb = alloc(N * kvw * 4);
    r->att = alloc(N * qw * 4);
    r->o = alloc(N * E * 4);
    r->g = alloc(N * c.n_ff * 4);
    r->u = alloc(N * c.n_ff * 4);
    r->e = alloc(N * E * 4);
    r->ed = alloc(NP * E * 4);
    r->logits = alloc(NP * V * 4);
    r->cand = alloc(NP * K * 4);
    r->unary = alloc(NP * K * 4);
    r->gate = alloc(NP * R * 4);
    r->lat = alloc(NP * K * K * 4);
    r->out = alloc(S * B * 4);
    dfb = std::move(r);
}

std::vector<std::vector<u32>> Model::dflashDraftBatch(std::span<const ::whirl::dflash::batch::Req> reqs) {
    namespace db = ::whirl::dflash::batch;
    std::vector<std::vector<u32>> res(reqs.size());
    if (!dfr || reqs.empty()) return res;
    // one request, or no batch scratch: the one-block path (bit-identical to the unbatched engine)
    if (reqs.size() == 1 || !dfb || !db::validSlots(reqs, static_cast<u32>(dfr->slots.size()))) {
        for (std::size_t i = 0; i < reqs.size(); ++i)
            res[i] = dflashDraft(reqs[i].slot, reqs[i].anchor, reqs[i].pos, reqs[i].p_min, reqs[i].n_min);
        return res;
    }
    const DFlashConfig& c = dflash->cfg;
    DFlashRun& r = *dfr;
    DFlashBatchRun& b = *dfb;
    const u32 B = r.block, P = B - 1, E = c.n_embd, hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv;
    const u32 kvw = nkv * hd, qw = nh * hd, K = c.sel_topk, R = c.sel_rank;
    const u64 V = output.nrows;
    const float theta_scale = std::pow(c.rope_base, -2.0f / static_cast<float>(hd));
    const auto& attn = r.ring + B > static_cast<u32>(kernels::kDFlashAttnKeys) ? k.dflash_attn_long : k.dflash_attn;
    const int q8 = dflash->sel_prev.ty == GgmlType::q8_0 ? 1 : 0;
    auto mm = [&](const Mat& w, DevPtr in, DevPtr out, u32 n) {
        xq_src = 0;
        x16_src = 0;
        matmul(w, in, out, n, false);
    };
    auto norm = [&](DevPtr in, DevPtr w, DevPtr out, u32 len, u32 count) {
        hip::launch(k.rmsnorm, D(count), D(256), 0, stream, in, w, out, I(len), I(len), I(len), c.eps);
    };
    for (const db::Group& gr : db::groups(static_cast<u32>(reqs.size()), b.slots)) {
        const std::span<const db::Req> rq = reqs.subspan(gr.begin, gr.count);
        const u32 S = gr.count, N = S * B, NP = S * P;
        // per block: rows i*B .. (byte offset of request i in a [N][width] f32 buffer)
        auto off = [&](u32 i, u64 width) { return db::blockRow(i, 0, B) * width * 4; };
        auto conv = [&](u32 i, DevPtr in, DevPtr dyn, DevPtr base, DevPtr out, u32 ds, int side) {
            hip::launch(k.dflash_conv, D((E + 255) / 256, B), D(256), 0, stream, in + off(i, E), dyn + off(i, ds), base, out + off(i, E), I(B),
                        I(E), I(c.conv_k), I(c.conv_group), side, I(ds));
        };
        const std::vector<i32> blk = db::tokens(rq, B, c.mask_id), ph = db::positions(rq, B);
        hip::streamSync(stream);
        hip::upload(b.ids, blk.data(), blk.size() * 4);
        hip::upload(b.pos, ph.data(), ph.size() * 4);
        hip::launch(k.get_rows[static_cast<std::size_t>(tok_embd.ty) % n_types], D(8, N), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes,
                    b.ids, b.x, I(E));
        for (u32 l = 0; l < c.n_layer; ++l) {
            const DFlashLayerW& w = dflash->layers[l];
            const u32 dsa = w.attn_conv_proj.nrows, dsf = w.ffn_conv_proj.nrows;
            norm(b.x, w.attn_norm, b.h, E, N);
            mm(w.attn_conv_proj, b.h, b.dyn, N);
            for (u32 i = 0; i < S; ++i) conv(i, b.h, b.dyn, w.attn_conv_base, b.h2, dsa, 0);
            mm(w.q, b.h2, b.q, N);
            mm(w.k, b.h2, b.kb, N);
            mm(w.v, b.h2, b.vb, N);
            norm(b.q, w.q_norm, b.q, hd, N * nh);
            norm(b.kb, w.k_norm, b.kb, hd, N * nkv);
            hip::launch(k.rope_neox, D(nh, N), D(64), 0, stream, b.q, b.pos, I(hd), I(qw), I(hd), theta_scale);
            hip::launch(k.rope_neox, D(nkv, N), D(64), 0, stream, b.kb, b.pos, I(hd), I(kvw), I(hd), theta_scale);
            // attention: block i against slot reqs[i]'s own ring and its own block keys (the per-sequence mask)
            for (u32 i = 0; i < S; ++i) {
                const DFlashSlotCtx& sc = r.slots[rq[i].slot];
                hip::launch(attn, D(nh, B), D(hd), 0, stream, b.q + off(i, qw), b.kb + off(i, kvw), b.vb + off(i, kvw), sc.kc[l], sc.vc[l],
                            sc.cpos, I(r.ring), I(B), I(nh), I(nkv), I(hd), I(rq[i].pos), I(c.window), 1.0f / std::sqrt(static_cast<float>(hd)),
                            b.att + off(i, qw));
            }
            mm(w.o, b.att, b.o, N);
            for (u32 i = 0; i < S; ++i) conv(i, b.o, b.dyn, w.attn_conv_base, b.h, dsa, 1);
            hip::launch(k.add_inplace, D((N * E + 255) / 256), D(256), 0, stream, b.x, b.h, I(static_cast<u64>(N) * E));
            norm(b.x, w.ffn_norm, b.h, E, N);
            mm(w.ffn_conv_proj, b.h, b.dyn, N);
            for (u32 i = 0; i < S; ++i) conv(i, b.h, b.dyn, w.ffn_conv_base, b.h2, dsf, 0);
            mm(w.gate, b.h2, b.g, N);
            mm(w.up, b.h2, b.u, N);
            hip::launch(k.silu_mul, D(static_cast<u32>((static_cast<u64>(N) * c.n_ff + 255) / 256)), D(256), 0, stream, b.g, b.u,
                        I(static_cast<u64>(N) * c.n_ff));
            mm(w.down, b.g, b.o, N);
            for (u32 i = 0; i < S; ++i) conv(i, b.o, b.dyn, w.ffn_conv_base, b.h, dsf, 1);
            hip::launch(k.add_inplace, D((N * E + 255) / 256), D(256), 0, stream, b.x, b.h, I(static_cast<u64>(N) * E));
        }
        // final norm; gather each block's rows 1 .. B-1 into [S * P] rows -> target head -> top-K
        norm(b.x, dflash->out_norm, b.e, E, N);
        for (u32 i = 0; i < S; ++i)
            hip::copyAsync(b.ed + db::draftRow(i, 0, B) * E * 4, b.e + db::blockRow(i, 1, B) * E * 4, static_cast<std::size_t>(P) * E * 4, stream);
        mm(output, b.ed, b.logits, NP);
        hip::launch(k.dflash_topk, D(NP), D(1024), 0, stream, b.logits, I(V), I(V), I(K), b.cand, b.unary);
        mm(dflash->sel_hidden, b.ed, b.gate, NP);
        // selector lattice and walk per block (anchor, p-min / n-min are per request)
        for (u32 i = 0; i < S; ++i) {
            const u64 d0 = db::draftRow(i, 0, B);
            hip::launch(k.dflash_lattice, D(P, K), D(32 * K), 0, stream, b.cand + d0 * K * 4, b.unary + d0 * K * 4, b.gate + d0 * R * 4,
                        dflash->sel_prev.ptr, dflash->sel_next.ptr, dflash->sel_prev.row_bytes, q8, I(R), I(K), static_cast<i32>(rq[i].anchor),
                        b.lat + d0 * K * K * 4);
            hip::launch(k.dflash_walk, D(1), D(32), 0, stream, b.lat + d0 * K * K * 4, b.cand + d0 * K * 4, I(P), I(K), rq[i].p_min,
                        static_cast<i32>(rq[i].n_min), b.out + db::outWord(i, B) * 4);
        }
        std::vector<i32> o(static_cast<std::size_t>(S) * B);
        hip::streamSync(stream);
        hip::download(o.data(), b.out, o.size() * 4);
        std::vector<std::vector<u32>> d = db::decodeOut(o, S, B);
        for (u32 i = 0; i < S; ++i) res[gr.begin + i] = std::move(d[i]);
    }
    return res;
}

}  // namespace whirl::qwen35
