// DFlash2 block-diffusion drafter: context ingest and one draft block on the GPU.
// SPDX-License-Identifier: Apache-2.0
// The math follows llama.cpp's src/models/dflash.cpp and z-lab/dflash (both MIT;
// written from the formulas in whirl-cloud results/C-23.md and tools/dflash_ref,
// no code copied). The speculative loops (`--draft dflash`) call the Model::dflash* functions
// (whirl chat / bench) and ServerModel (whirl-server).

#include "whirl/model.h"
#include "whirl/dflash_policy.h"
#include "config/env_snapshot.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

inline hip::Dim3 D(u32 x, u32 y = 1, u32 z = 1) { return {x, y, z}; }
inline i32 I(u64 v) { return static_cast<i32>(v); }

// WHIRL_DFLASH_PROF=1: per-stage wall time of the drafter (each stage ends with a stream sync, so the total
// is a little above the unprofiled time). Printed to stderr at exit. Off by default; the outputs do not change.
struct DflProf {
    bool on = false;
    std::map<std::string, std::pair<double, unsigned>> t;
    std::chrono::steady_clock::time_point last;
    unsigned draws = 0, ingests = 0;
    DflProf() {
        // arch025-N1: read at construction (g_prof: static initialisation), as before
        const whirl::config::EnvSnapshot env = whirl::config::EnvSnapshot::capture();
        on = env.dflash_prof && !env.dflash_prof->empty() && env.dflash_prof->front() == '1';
    }
    ~DflProf() {
        if (!on || (draws == 0 && ingests == 0)) return;
        double tot = 0;
        for (const auto& kv : t) tot += kv.second.first;
        std::fprintf(stderr, "DFlash2 drafter profile: %u draft blocks, %u ingests, %.2f ms per draft + its ingest\n", draws, ingests,
                     draws ? tot / draws : 0.0);
        for (const auto& kv : t)
            std::fprintf(stderr, "  %-22s %8.3f ms/draft  %5.1f %%  (%u calls)\n", kv.first.c_str(), draws ? kv.second.first / draws : 0.0,
                         tot > 0 ? 100.0 * kv.second.first / tot : 0.0, kv.second.second);
    }
    void start(hip::Stream s) {
        if (!on) return;
        hip::streamSync(s);
        last = std::chrono::steady_clock::now();
    }
    void mark(hip::Stream s, const char* name) {
        if (!on) return;
        hip::streamSync(s);
        const auto now = std::chrono::steady_clock::now();
        auto& e = t[name];
        e.first += std::chrono::duration<double, std::milli>(now - last).count();
        e.second++;
        last = now;
    }
};
DflProf g_prof;

}  // namespace

std::string Model::dflashMissingKernel() const {
    const std::pair<const char*, hip::Function> need[] = {{"dflash_conv", k.dflash_conv},   {"dflash_topk", k.dflash_topk},
                                                          {"dflash_lattice", k.dflash_lattice}, {"dflash_walk", k.dflash_walk},
                                                          {"dflash_attn", k.dflash_attn},   {"rope_neox", k.rope_neox},
                                                          {"add_inplace", k.add_inplace},   {"silu_mul", k.silu_mul},
                                                          {"rmsnorm", k.rmsnorm}};
    for (const auto& [name, f] : need)
        if (f == nullptr) return name;
    if (k.get_rows[static_cast<std::size_t>(tok_embd.ty) % n_types] == nullptr) return "get_rows";
    return {};
}

void Model::dflashSetup(u32 slots, u32 ingest_rows, bool debug) {
    if (!dflash) throw ModelError("NoDraftModel", "dflashSetup before loadDrafter");
    if (const std::string miss = dflashMissingKernel(); !miss.empty()) throw ModelError("MissingKernel", miss + " (DFlash2 drafter kernels)");
    const DFlashConfig& c = dflash->cfg;
    auto r = std::make_unique<DFlashRun>();
    r->ring = ::whirl::dflash::ringRows(c.window);
    r->block = c.block;
    // keys of the drafter attention: dflash_attn holds up to kDFlashAttnKeys, longer rings (e.g. Ornith's
    // 4096-row window) need dflash_attn_long (validate() already refused anything past kDFlashMaxKeys)
    if (r->ring + r->block > static_cast<u32>(kernels::kDFlashAttnKeys) && !k.dflash_attn_long)
        throw ModelError("MissingKernel", "dflash_attn_long (DFlash2 window " + std::to_string(r->ring) + " + block " +
                                              std::to_string(r->block) + " > " + std::to_string(kernels::kDFlashAttnKeys) + " keys)");
    r->ingest_rows = std::max<u32>(1, std::min(ingest_rows, r->ring));
    const u64 E = c.n_embd, B = c.block, P = B - 1, kvw = static_cast<u64>(c.n_head_kv) * c.head_dim;
    const u64 qw = static_cast<u64>(c.n_head) * c.head_dim, ds = dflash->layers[0].attn_conv_proj.nrows, V = output.nrows;
    const u64 K = c.sel_topk, R = c.sel_rank, IR = r->ingest_rows;
    r->ids = alloc(B * 4);
    r->pos = alloc(B * 4);
    r->x = alloc(B * E * 4);
    r->h = alloc(B * E * 4);
    r->h2 = alloc(B * E * 4);
    r->dyn = alloc(B * ds * 4);
    r->q = alloc(B * qw * 4);
    r->kb = alloc(B * kvw * 4);
    r->vb = alloc(B * kvw * 4);
    r->att = alloc(B * qw * 4);
    r->o = alloc(B * E * 4);
    r->g = alloc(B * c.n_ff * 4);
    r->u = alloc(B * c.n_ff * 4);
    r->e = alloc(B * E * 4);
    r->logits = alloc(P * V * 4);
    r->cand = alloc(P * K * 4);
    r->unary = alloc(P * K * 4);
    r->gate = alloc(P * R * 4);
    r->lat = alloc(P * K * K * 4);
    r->out = alloc((1 + P) * 4);
    r->ing_g = alloc(IR * E * 4);
    r->ing_k = alloc(IR * kvw * 4);
    r->ing_v = alloc(IR * kvw * 4);
    r->ing_pos = alloc(IR * 4);
    if (debug) r->dbg = allocZero(static_cast<u64>(c.n_layer) * B * E * 4);
    r->slots.resize(slots);
    for (DFlashSlotCtx& s : r->slots) {
        for (u32 l = 0; l < c.n_layer; ++l) {
            s.kc.push_back(allocZero(r->ring * kvw * 4));
            s.vc.push_back(allocZero(r->ring * kvw * 4));
        }
        s.cpos_h.assign(r->ring, -1);
        s.cpos = alloc(static_cast<u64>(r->ring) * 4);
        hip::upload(s.cpos, s.cpos_h.data(), s.cpos_h.size() * 4);
    }
    dfr = std::move(r);
}

u64 Model::dflashSlotBytes() const {
    if (!dflash) return 0;
    const DFlashConfig& c = dflash->cfg;
    const u64 ring = ::whirl::dflash::ringRows(c.window), kvw = static_cast<u64>(c.n_head_kv) * c.head_dim;
    return static_cast<u64>(c.n_layer) * 2 * ring * kvw * 4 + ring * 4;
}

void Model::dflashIngest(u32 slot, DevPtr feats, u32 stride, u32 row0, u32 n, u32 pos0) {
    if (!dfr || slot >= dfr->slots.size() || n == 0) return;
    const DFlashConfig& c = dflash->cfg;
    DFlashRun& r = *dfr;
    DFlashSlotCtx& sc = r.slots[slot];
    // only the last ring rows can ever be seen; larger batches go in pieces
    if (n > r.ring) {
        row0 += n - r.ring;
        pos0 += n - r.ring;
        n = r.ring;
    }
    const u64 E = c.n_embd, kvw = static_cast<u64>(c.n_head_kv) * c.head_dim;
    const u32 nl = static_cast<u32>(c.target_layers.size());
    const float theta_scale = std::pow(c.rope_base, -2.0f / static_cast<float>(c.head_dim));
    g_prof.ingests++;
    for (u32 done = 0; done < n;) {
        g_prof.start(stream);
        const u32 m = std::min(n - done, r.ingest_rows);
        const u32 p0 = pos0 + done;
        // g = rmsnorm(fc . concat(features)) * enc_norm; fc as nl column blocks of n_embd
        const u64 col_bytes = dflash->fc.row_bytes / nl;
        for (u32 j = 0; j < nl; ++j) {
            Mat fj = dflash->fc;
            fj.ptr += j * col_bytes;
            fj.ncols = c.n_embd;
            xq_src = 0;
            x16_src = 0;
            matmul(fj, feats + (static_cast<u64>(j) * stride + row0 + done) * E * 4, r.ing_g, m, j > 0);
        }
        hip::launch(k.rmsnorm, D(m), D(256), 0, stream, r.ing_g, dflash->enc_norm, r.ing_g, I(E), I(E), I(E), c.eps);
        std::vector<i32> ph(m);
        for (u32 i = 0; i < m; ++i) ph[i] = static_cast<i32>(p0 + i);
        hip::streamSync(stream);
        g_prof.mark(stream, "i1 fc+norm");
        hip::upload(r.ing_pos, ph.data(), ph.size() * 4);
        for (u32 l = 0; l < c.n_layer; ++l) {
            const DFlashLayerW& w = dflash->layers[l];
            xq_src = 0;
            x16_src = 0;
            matmul(w.k, r.ing_g, r.ing_k, m, false);
            xq_src = 0;
            x16_src = 0;
            matmul(w.v, r.ing_g, r.ing_v, m, false);
            hip::launch(k.rmsnorm, D(m * c.n_head_kv), D(256), 0, stream, r.ing_k, w.k_norm, r.ing_k, I(c.head_dim), I(c.head_dim),
                        I(c.head_dim), c.eps);
            hip::launch(k.rope_neox, D(c.n_head_kv, m), D(64), 0, stream, r.ing_k, r.ing_pos, I(c.head_dim), I(kvw), I(c.head_dim),
                        theta_scale);
            // rows go to ring index position % ring (at most two contiguous pieces)
            const ::whirl::dflash::RingPieces rp = ::whirl::dflash::ringPieces(p0, m, r.ring);
            hip::copyAsync(sc.kc[l] + rp.start * kvw * 4, r.ing_k, rp.first * kvw * 4, stream);
            hip::copyAsync(sc.vc[l] + rp.start * kvw * 4, r.ing_v, rp.first * kvw * 4, stream);
            if (rp.second > 0) {
                hip::copyAsync(sc.kc[l], r.ing_k + rp.first * kvw * 4, rp.second * kvw * 4, stream);
                hip::copyAsync(sc.vc[l], r.ing_v + rp.first * kvw * 4, rp.second * kvw * 4, stream);
            }
        }
        g_prof.mark(stream, "i2 k/v proj+rope+ring");
        for (u32 i = 0; i < m; ++i) sc.cpos_h[(p0 + i) % r.ring] = static_cast<i32>(p0 + i);
        done += m;
    }
    hip::streamSync(stream);
    hip::upload(sc.cpos, sc.cpos_h.data(), sc.cpos_h.size() * 4);
}

void Model::dflashTrim(u32 slot, u32 keep) {
    if (!dfr || slot >= dfr->slots.size()) return;
    DFlashSlotCtx& sc = dfr->slots[slot];
    bool changed = false;
    for (i32& p : sc.cpos_h)
        if (p >= static_cast<i32>(keep)) {
            p = -1;
            changed = true;
        }
    if (!changed) return;
    hip::streamSync(stream);
    hip::upload(sc.cpos, sc.cpos_h.data(), sc.cpos_h.size() * 4);
}

std::vector<u32> Model::dflashDraft(u32 slot, u32 anchor, u32 pos, float p_min, u32 n_min) {
    if (!dfr || slot >= dfr->slots.size()) return {};
    const DFlashConfig& c = dflash->cfg;
    DFlashRun& r = *dfr;
    const DFlashSlotCtx& sc = r.slots[slot];
    const u32 B = r.block, P = B - 1, E = c.n_embd, hd = c.head_dim, nh = c.n_head, nkv = c.n_head_kv;
    const u32 kvw = nkv * hd, qw = nh * hd, K = c.sel_topk, R = c.sel_rank;
    const float theta_scale = std::pow(c.rope_base, -2.0f / static_cast<float>(hd));
    auto mm = [&](const Mat& w, DevPtr in, DevPtr out, u32 n, bool acc) {
        xq_src = 0;
        x16_src = 0;
        matmul(w, in, out, n, acc);
    };
    auto norm = [&](DevPtr in, DevPtr w, DevPtr out, u32 len, u32 count) {
        hip::launch(k.rmsnorm, D(count), D(256), 0, stream, in, w, out, I(len), I(len), I(len), c.eps);
    };
    auto conv = [&](DevPtr in, DevPtr dyn, DevPtr base, DevPtr out, u32 ds, int side) {
        hip::launch(k.dflash_conv, D((E + 255) / 256, B), D(256), 0, stream, in, dyn, base, out, I(B), I(E), I(c.conv_k), I(c.conv_group),
                    side, I(ds));
    };
    // block tokens and positions; the mask token's embedding is the target's row mask_id
    std::vector<i32> blk(B, static_cast<i32>(c.mask_id)), ph(B);
    blk[0] = static_cast<i32>(anchor);
    for (u32 t = 0; t < B; ++t) ph[t] = static_cast<i32>(pos + t);
    g_prof.draws++;
    hip::streamSync(stream);
    g_prof.start(stream);
    hip::upload(r.ids, blk.data(), blk.size() * 4);
    hip::upload(r.pos, ph.data(), ph.size() * 4);
    hip::launch(k.get_rows[static_cast<std::size_t>(tok_embd.ty) % n_types], D(8, B), D(128), 0, stream, tok_embd.ptr, tok_embd.row_bytes,
                r.ids, r.x, I(E));
    g_prof.mark(stream, "d00 embed");
    // rings that fit dflash_attn keep it (unchanged); longer ones take the chunked dflash_attn_long
    const auto& attn = r.ring + B > static_cast<u32>(kernels::kDFlashAttnKeys) ? k.dflash_attn_long : k.dflash_attn;
    for (u32 l = 0; l < c.n_layer; ++l) {
        const DFlashLayerW& w = dflash->layers[l];
        const u32 dsa = w.attn_conv_proj.nrows, dsf = w.ffn_conv_proj.nrows;
        // attention half: norm -> conv side 0 -> q/k/v -> norms, RoPE -> attention -> o -> conv side 1 -> residual
        norm(r.x, w.attn_norm, r.h, E, B);
        mm(w.attn_conv_proj, r.h, r.dyn, B, false);
        conv(r.h, r.dyn, w.attn_conv_base, r.h2, dsa, 0);
        g_prof.mark(stream, "d1 attn norm+conv in");
        mm(w.q, r.h2, r.q, B, false);
        mm(w.k, r.h2, r.kb, B, false);
        mm(w.v, r.h2, r.vb, B, false);
        norm(r.q, w.q_norm, r.q, hd, B * nh);
        norm(r.kb, w.k_norm, r.kb, hd, B * nkv);
        hip::launch(k.rope_neox, D(nh, B), D(64), 0, stream, r.q, r.pos, I(hd), I(qw), I(hd), theta_scale);
        hip::launch(k.rope_neox, D(nkv, B), D(64), 0, stream, r.kb, r.pos, I(hd), I(kvw), I(hd), theta_scale);
        g_prof.mark(stream, "d2 q/k/v proj+norm+rope");
        hip::launch(attn, D(nh, B), D(hd), 0, stream, r.q, r.kb, r.vb, sc.kc[l], sc.vc[l], sc.cpos, I(r.ring), I(B), I(nh), I(nkv),
                    I(hd), I(pos), I(c.window), 1.0f / std::sqrt(static_cast<float>(hd)), r.att);
        g_prof.mark(stream, "d3 attention kernel");
        mm(w.o, r.att, r.o, B, false);
        conv(r.o, r.dyn, w.attn_conv_base, r.h, dsa, 1);
        hip::launch(k.add_inplace, D((B * E + 255) / 256), D(256), 0, stream, r.x, r.h, I(B * E));
        // FFN half: norm -> conv side 0 -> SwiGLU -> conv side 1 -> residual
        g_prof.mark(stream, "d4 o proj+conv out+add");
        norm(r.x, w.ffn_norm, r.h, E, B);
        mm(w.ffn_conv_proj, r.h, r.dyn, B, false);
        conv(r.h, r.dyn, w.ffn_conv_base, r.h2, dsf, 0);
        g_prof.mark(stream, "d5 ffn norm+conv in");
        mm(w.gate, r.h2, r.g, B, false);
        mm(w.up, r.h2, r.u, B, false);
        hip::launch(k.silu_mul, D((B * c.n_ff + 255) / 256), D(256), 0, stream, r.g, r.u, I(B * c.n_ff));
        mm(w.down, r.g, r.o, B, false);
        g_prof.mark(stream, "d6 ffn gate/up/down");
        conv(r.o, r.dyn, w.ffn_conv_base, r.h, dsf, 1);
        hip::launch(k.add_inplace, D((B * E + 255) / 256), D(256), 0, stream, r.x, r.h, I(B * E));
        g_prof.mark(stream, "d7 ffn conv out+add");
        if (r.dbg) hip::copyAsync(r.dbg + static_cast<u64>(l) * B * E * 4, r.x, static_cast<std::size_t>(B) * E * 4, stream);
    }
    // final norm; block rows 1 .. B-1 -> target head -> top-K -> selector -> walk
    norm(r.x, dflash->out_norm, r.e, E, B);
    const DevPtr e1 = r.e + static_cast<u64>(E) * 4;
    g_prof.mark(stream, "d8 final norm");
    mm(output, e1, r.logits, P, false);
    g_prof.mark(stream, "d9 lm_head (target)");
    hip::launch(k.dflash_topk, D(P), D(1024), 0, stream, r.logits, I(output.nrows), I(output.nrows), I(K), r.cand, r.unary);
    g_prof.mark(stream, "da top-k");
    mm(dflash->sel_hidden, e1, r.gate, P, false);
    const int q8 = dflash->sel_prev.ty == GgmlType::q8_0 ? 1 : 0;
    hip::launch(k.dflash_lattice, D(P, K), D(32 * K), 0, stream, r.cand, r.unary, r.gate, dflash->sel_prev.ptr, dflash->sel_next.ptr,
                dflash->sel_prev.row_bytes, q8, I(R), I(K), static_cast<i32>(anchor), r.lat);
    hip::launch(k.dflash_walk, D(1), D(32), 0, stream, r.lat, r.cand, I(P), I(K), p_min, static_cast<i32>(n_min), r.out);
    g_prof.mark(stream, "db selector+lattice+walk");
    std::vector<i32> o(1 + P);
    hip::streamSync(stream);
    hip::download(o.data(), r.out, o.size() * 4);
    std::vector<u32> drafts;
    for (i32 i = 0; i < o[0]; ++i) drafts.push_back(static_cast<u32>(o[1 + i]));
    return drafts;
}

}  // namespace whirl::qwen35
