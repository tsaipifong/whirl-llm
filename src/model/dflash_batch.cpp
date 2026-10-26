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
//
// feat029-dflashingest: Model::dflashIngestBatch (end of file) is dflashIngest (src/model/dflash.cpp) over the
// stacked rows of several slots; same kernels and matmuls, only the row count changes. dflash.cpp is untouched.

#include "whirl/model.h"
#include "whirl/dflash_batch.h"
#include "whirl/dflash_policy.h"

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
    namespace db = ::whirl::dflash::batch;
    const db::MemInput in = batchMem(*dflash, static_cast<u32>(output.nrows));
    const u64 draft = db::scratchBytes(in, max_slots);
    if (draft == 0) return 0;
    // feat029-dflashingest: plus the batched ingest scratch (one verify block per slot)
    const u32 cap = db::groupCap(dflash->cfg.block, max_slots, max_batch);
    return draft + db::ingestScratchBytes(in, static_cast<u32>(dflash->cfg.target_layers.size()), db::ingestCap(dflash->cfg.block, cap, max_batch));
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
    // feat029-dflashingest: stacked ingest rows (one verify block per slot, at most max_batch)
    const u64 nl = c.target_layers.size();
    r->ing_rows = ::whirl::dflash::batch::ingestCap(static_cast<u32>(B), cap, max_batch);
    const u64 IR = r->ing_rows;
    r->ing_f = alloc(nl * IR * E * 4);
    r->ing_g = alloc(IR * E * 4);
    r->ing_k = alloc(IR * kvw * 4);
    r->ing_v = alloc(IR * kvw * 4);
    r->ing_pos = alloc(IR * 4);
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

void Model::dflashIngestBatch(DevPtr feats, u32 stride, std::span<const ::whirl::dflash::batch::IngestReq> reqs) {
    namespace db = ::whirl::dflash::batch;
    if (!dfr || reqs.empty()) return;
    DFlashRun& r = *dfr;
    auto one = [&](const db::IngestReq& q) { dflashIngest(q.slot, feats, stride, q.cap_row, q.n, q.pos); };
    // one request, no batch scratch, a repeated slot: the one-slot path (bit-identical to the unbatched engine)
    if (reqs.size() == 1 || !dfb || dfb->ing_rows == 0 || !db::validIngest(reqs, static_cast<u32>(r.slots.size()))) {
        for (const db::IngestReq& q : reqs) one(q);
        return;
    }
    const DFlashConfig& c = dflash->cfg;
    DFlashBatchRun& b = *dfb;
    const u64 E = c.n_embd, kvw = static_cast<u64>(c.n_head_kv) * c.head_dim;
    const u32 nl = static_cast<u32>(c.target_layers.size());
    const float theta_scale = std::pow(c.rope_base, -2.0f / static_cast<float>(c.head_dim));
    const u64 col_bytes = dflash->fc.row_bytes / nl;
    // a request longer than the ring keeps only its last ring rows (dflashIngest): such a request is "single"
    const u32 cap = std::min(b.ing_rows, r.ring);
    std::vector<u32> touched;
    for (const db::IngestGroup& gr : db::ingestGroups(reqs, cap)) {
        const std::span<const db::IngestReq> rq = reqs.subspan(gr.begin, gr.count);
        if (gr.single) {
            for (const db::IngestReq& q : rq) one(q);
            continue;
        }
        const std::vector<u64> off = db::ingestOffsets(rq);
        const u32 N = static_cast<u32>(off.back());
        const std::vector<i32> ph = db::ingestPositions(rq);
        // the previous group's kernels still read ing_pos / ing_f
        hip::streamSync(stream);
        hip::upload(b.ing_pos, ph.data(), ph.size() * 4);
        // gather: target layer j of request i -> ing_f rows j * N + off[i] .. (feature row r of layer j at feats + (j * stride + r) * E)
        for (u32 j = 0; j < nl; ++j)
            for (std::size_t i = 0; i < rq.size(); ++i)
                hip::copyAsync(b.ing_f + (static_cast<u64>(j) * N + off[i]) * E * 4, feats + (static_cast<u64>(j) * stride + rq[i].cap_row) * E * 4,
                               static_cast<std::size_t>(rq[i].n) * E * 4, stream);
        // g = rmsnorm(fc . concat(features)) * enc_norm over the N stacked rows
        for (u32 j = 0; j < nl; ++j) {
            Mat fj = dflash->fc;
            fj.ptr += j * col_bytes;
            fj.ncols = c.n_embd;
            xq_src = 0;
            x16_src = 0;
            matmul(fj, b.ing_f + static_cast<u64>(j) * N * E * 4, b.ing_g, N, j > 0);
        }
        hip::launch(k.rmsnorm, D(N), D(256), 0, stream, b.ing_g, dflash->enc_norm, b.ing_g, I(E), I(E), I(E), c.eps);
        for (u32 l = 0; l < c.n_layer; ++l) {
            const DFlashLayerW& w = dflash->layers[l];
            xq_src = 0;
            x16_src = 0;
            matmul(w.k, b.ing_g, b.ing_k, N, false);
            xq_src = 0;
            x16_src = 0;
            matmul(w.v, b.ing_g, b.ing_v, N, false);
            hip::launch(k.rmsnorm, D(N * c.n_head_kv), D(256), 0, stream, b.ing_k, w.k_norm, b.ing_k, I(c.head_dim), I(c.head_dim),
                        I(c.head_dim), c.eps);
            hip::launch(k.rope_neox, D(c.n_head_kv, N), D(64), 0, stream, b.ing_k, b.ing_pos, I(c.head_dim), I(kvw), I(c.head_dim), theta_scale);
            // per slot: its rows go to its own ring at position % ring (at most two contiguous pieces)
            for (std::size_t i = 0; i < rq.size(); ++i) {
                DFlashSlotCtx& sc = r.slots[rq[i].slot];
                const DevPtr ks = b.ing_k + off[i] * kvw * 4, vs = b.ing_v + off[i] * kvw * 4;
                const ::whirl::dflash::RingPieces rp = ::whirl::dflash::ringPieces(rq[i].pos, rq[i].n, r.ring);
                hip::copyAsync(sc.kc[l] + rp.start * kvw * 4, ks, rp.first * kvw * 4, stream);
                hip::copyAsync(sc.vc[l] + rp.start * kvw * 4, vs, rp.first * kvw * 4, stream);
                if (rp.second > 0) {
                    hip::copyAsync(sc.kc[l], ks + rp.first * kvw * 4, rp.second * kvw * 4, stream);
                    hip::copyAsync(sc.vc[l], vs + rp.first * kvw * 4, rp.second * kvw * 4, stream);
                }
            }
        }
        for (const db::IngestReq& q : rq) {
            DFlashSlotCtx& sc = r.slots[q.slot];
            for (u32 t = 0; t < q.n; ++t) sc.cpos_h[(q.pos + t) % r.ring] = static_cast<i32>(q.pos + t);
            touched.push_back(q.slot);
        }
    }
    if (touched.empty()) return;
    hip::streamSync(stream);
    for (const u32 s : touched) hip::upload(r.slots[s].cpos, r.slots[s].cpos_h.data(), r.slots[s].cpos_h.size() * 4);
}

}  // namespace whirl::qwen35
