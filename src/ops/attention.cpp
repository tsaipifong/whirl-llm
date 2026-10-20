// Attention kernel choice (ARCH-1 K-3), see attention.h. Moved from src/model/forward.cpp (Model::attnBlock's
// decode section, Model::prefillAttn, dq4Per / dq4Part / dq4Splits / awPer); same choices and grids, pinned by the
// K-0 launch tables.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/ops/attention.h"

#include "../../kernels/gfx1151/attn_ws4_index.h"  // ws4::block_range (attn_ws4 / attn_ws4g split rule)

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>

namespace whirl::ops {

namespace {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
constexpr Dim3 D(u64 x, u64 y = 1, u64 z = 1) { return Dim3{static_cast<unsigned>(x), static_cast<unsigned>(y), static_cast<unsigned>(z)}; }
// positions per attn_split block (FD_CH in the kernels)
constexpr u32 fd_chunk = 64;

// attn_dq4 (kernels/gfx1151/attn.hip dq_per, DQ_NW 8, DQ_MAXS 128): keys per split for n_keys keys
// under a grid of n_split splits; keep in step with the kernel.
i32 dq4Per(i32 n_keys, u32 n_split) {
    const i32 ns = static_cast<i32>(std::min<u32>(n_split, 128));
    return std::max<i32>(8 * 32, (n_keys + ns * 64 - 1) / (ns * 64) * 64);
}
// a row's attn_dq4 key partition under n_split: {window branch, per, splits carrying keys}
std::array<i32, 3> dq4Part(i32 n_pos, i32 win, u32 n_split) {
    if (win != 0 && n_split > 1) {
        const i32 w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
        const i32 w0 = std::max<i32>(256, (n_pos - w) & ~63);
        if (n_pos >= min_pos && w0 > 256) {
            const i32 per = dq4Per(n_pos - w0, n_split - 1);
            return {1, per, 1 + (n_pos - w0 + per - 1) / per};
        }
    }
    const i32 per = dq4Per(n_pos, n_split);
    return {0, per, (n_pos + per - 1) / per};
}
}  // namespace

// smallest split count (8, 16, ... < full) with every row's partition equal to full's; else full
u32 dq4Splits(std::span<const i32> row_pos, i32 win, u32 full) {
    for (u32 s = 8; s < full; s *= 2) {
        bool same = true;
        for (const i32 p : row_pos) {
            const auto a = dq4Part(p + 1, win, s), b = dq4Part(p + 1, win, full);
            if (a[0] != b[0] || a[1] != b[1] || a[2] > static_cast<i32>(s)) {
                same = false;
                break;
            }
        }
        if (same) return s;
    }
    return full;
}

// attn_split / attn_wsplit split size for n_pos positions (whole FD_CH chunks).
i32 awPer(i32 n_pos, u32 n_split) {
    const i32 span = static_cast<i32>(n_split * fd_chunk);
    const i32 q = n_pos + span - 1;
    // floor division (n_pos >= 1 here, so q >= 0)
    return (q >= 0 ? q / span : -((-q + span - 1) / span)) * static_cast<i32>(fd_chunk);
}

AttnDecodePlan chooseAttnDecode(const AttnDecodeProblem& p, const AttnDecodeFlags& f, const kernels::KernelTable& k, const device::Profile& prof) {
    const u32 n = p.n, hd = p.head_dim;
    const u32 grp_q = p.n_head / p.n_head_kv;
    AttnDecodePlan pl;
    // gfx1151, f16 / int8 KV: attn_ws4 (groups of one row) and attn_ws4g (multi-row verify
    // groups; DES-ATTN-8S T2 / T3). Every row's partials are bitwise those of an attn_wsplit1
    // one-query group whenever a group's rows share the split ranges, so the grouping that
    // caps.attn_group1 forbids for attn_wsplit1 is safe here. The split rule is the constant
    // ws4::kNs (64): the ranges depend on n_pos (and the draft window) only, WHIRL_FD_SPLITS
    // does not apply. WHIRL_ATTN_WS4=0: the old path below.
    const bool ws4_path = f.attn_ws4 && k.attn_ws4 != nullptr && k.attn_ws4g[0] != nullptr && k.attn_ws4g[1] != nullptr && k.attn_ws4g[2] != nullptr &&
                          k.attn_wsplit1 != nullptr && !p.kv_q4 && hd == ws4::kD && grp_q <= 16 && (f.dbg_flags & 2) == 0;
    u32 n_split = ws4_path ? std::min<u32>(ws4::kNs, (p.max_ctx + fd_chunk - 1) / fd_chunk) : std::min(f.fd_splits, (p.max_ctx + fd_chunk - 1) / fd_chunk);
    // FAST-8S-1 (gfx1151, q4 KV through attn_dq4): the smallest split count whose key partition (dq_per) is the
    // same as fd_splits' for every row: same partials, same combine order, so bitwise the same output. The
    // profile row says where this was measured and output-checked (the 8060S); the R9700 keeps the fixed grid.
    // (the same conditions as the wsplit branch below, REVIEW-022 F10)
    if (prof.attn_dq4_dynamic_splits && !ws4_path && p.kv_q4 && f.attn_dq4 && f.dq4_dyn && k.attn_dq4 != nullptr && k.caps.attn_group1 && grp_q <= 8 &&
        p.rows_known && k.attn_wsplit1 != nullptr && hd == 256 && (f.dbg_flags & 2) == 0)
        n_split = dq4Splits(p.row_pos.first(n), p.win, n_split);
    pl.n_split = n_split;
    if (ws4_path) {
        // groups: consecutive rows of one sequence at consecutive positions whose split ranges
        // (per and draft-window ranges, ws4::block_range of their n_pos) are the same
        const i32 win = p.win;
        auto key = [&](i32 pp) {
            const i32 np = pp + 1;
            const ws4::Range r0 = ws4::block_range(np, 0, ws4::kNs, win);
            return std::array<i32, 3>{r0.b1 == ws4::kSink ? r0.b0 - 1 : 0, ws4::block_range(np, 1, ws4::kNs, win).b0,
                                      ws4::block_range(np, 2, ws4::kNs, win).b0};
        };
        const bool known = p.rows_known && n > 1 && (f.dbg_flags & 4) == 0;
        u32 ng = 0, r = 0, widest = 1;
        while (r < n) {
            u32 c = 1;
            if (known) {
                const auto k0 = key(p.row_pos[r]);
                while (r + c < n && c < static_cast<u32>(kernels::kMaxVerifyRows) && p.row_base[r + c] == p.row_base[r] &&
                       p.row_pos[r + c] == p.row_pos[r] + static_cast<i32>(c) && key(p.row_pos[r + c]) == k0)
                    c += 1;
            }
            pl.groups.first[ng] = static_cast<i32>(r);
            pl.groups.count[ng] = static_cast<i32>(c);
            widest = std::max(widest, c);
            ng += 1;
            r += c;
        }
        if (widest == 1) {
            pl.kind = AttnDecodeKind::ws4;
            pl.f = k.attn_ws4;
            pl.grid = D(p.n_head_kv, n_split, ng);
            pl.block = D(128);
        } else {
            // CG 16-column tiles per block (<= 3: 8 rows x 6 heads), more rows in column-group blocks on grid.x
            const u32 cg = std::min<u32>(3, (widest * grp_q + 15) / 16);
            const u32 rpb = static_cast<u32>(ws4::rows_per_block(static_cast<int>(cg), static_cast<int>(grp_q)));
            pl.kind = AttnDecodeKind::ws4g;
            pl.f = k.attn_ws4g[cg - 1];
            pl.grid = D(p.n_head_kv * ((widest + rpb - 1) / rpb), n_split, ng);
            pl.block = D(256);
        }
    } else if (k.attn_wsplit1 != nullptr && hd == 256 && grp_q <= 16 && (f.dbg_flags & 2) == 0) {
        // query groups: consecutive rows of one sequence whose split size `per` (a function of the position)
        // matches, <= 16 columns each (attn_wsplit1) or <= 32 (attn_wsplit2: one K/V pass for up to 32 / grp_q rows).
        // caps.attn_group1 (gfx1151): one query per group (its WMMA P.V is not exact when a masked key carries a
        // real V row). q4 KV with attn_dq4 (FC-1c, gfx1201 too): every row its own group, so decode, verify and
        // MTP rows all take attn_dq4 (a row's partials never depend on its batch)
        const bool dq4_rows = p.kv_q4 && f.attn_dq4 && k.attn_dq4 != nullptr && grp_q <= 8;
        const bool one_q = (f.dbg_flags & 4) != 0 || k.caps.attn_group1 || dq4_rows;
        const bool known = p.rows_known && n > 1;
        auto group = [&](u32 max_q, kernels::AwGroups& groups) {
            u32 ng = 0, r = 0, widest = 0;
            while (r < n) {
                u32 c = 1;
                if (known) {
                    const i32 per0 = awPer(p.row_pos[r] + 1, n_split);
                    while (r + c < n && c < max_q && p.row_base[r + c] == p.row_base[r] && p.row_pos[r + c] == p.row_pos[r] + static_cast<i32>(c) &&
                           awPer(p.row_pos[r + c] + 1, n_split) == per0)
                        c += 1;
                }
                groups.first[ng] = static_cast<i32>(r);
                groups.count[ng] = static_cast<i32>(c);
                widest = std::max(widest, c);
                ng += 1;
                r += c;
            }
            return std::pair<u32, u32>{ng, widest};
        };
        const u32 max_q1 = one_q ? 1 : 16 / grp_q;
        Function aw = k.attn_wsplit1;
        u32 ng = 0;
        bool wide = false;
        if (!one_q && f.attn_wide && k.attn_wsplit2 != nullptr && known && 32 / grp_q > max_q1) {
            const auto [ng2, widest2] = group(32 / grp_q, pl.groups);
            if (widest2 > max_q1) {
                aw = k.attn_wsplit2;
                ng = ng2;
                wide = true;
            }
        }
        if (!wide) ng = group(max_q1, pl.groups).first;
        // q4 KV: attn_dq4 (one query per group, GQA group <= 8; lossy fast-mode path)
        const bool dq4 = !wide && max_q1 == 1 && f.attn_dq4 && k.attn_dq4 != nullptr && grp_q <= 8;
        pl.kind = dq4 ? AttnDecodeKind::dq4 : AttnDecodeKind::wsplit;
        pl.f = dq4 ? k.attn_dq4 : aw;
        pl.grid = D(p.n_head_kv, n_split, ng);
        pl.block = D(dq4 ? 256 : 128);
    } else {
        pl.kind = AttnDecodeKind::split;
        pl.f = k.attn_split;
        pl.grid = D(p.n_head_kv, n_split, n);
        pl.block = D(256);
    }
    pl.combine_q8 = k.attn_combine_q8 != nullptr && p.o_on_gemv && !f.prec_dec && hd == 256;
    pl.combine_grid = D(p.n_head, n);
    return pl;
}

// WMMA prefill attention of n queries at positions pos0 .. (keys 0 .. pos0 + n - 1), split over head ranges so
// one launch does at most ~4096 x 128k query-key pairs x all heads. Same results as one launch.
// f16 / q8 / q8h / q8v KV: attn_kg (NP query heads of one KV head per block; bit-identical to attn_kx) when NP
// divides the GQA group, the head ranges stay whole groups of NP and the softmax scale is a power of two.
AttnPrefillPlan chooseAttnPrefill(const AttnPrefillProblem& p, const AttnPrefillFlags& f, const kernels::KernelTable& k) {
    AttnPrefillPlan pl;
    auto& st = pl.steps;
    const u32 n = p.n, n_head = p.n_head;
    // head ranges per launch: at most ~4096 x 128k query-key pairs x all heads (Windows TDR)
    auto headParts = [&](u64 keys) {
        const u64 work = static_cast<u64>(n) * keys;
        const u64 budget = 4096ull * 131072ull;
        u32 parts = static_cast<u32>(std::min<u64>(n_head, (work + budget - 1) / budget));
        while (parts > 1 && n_head % parts != 0) parts += 1;  // whole head ranges
        return parts;
    };
    auto attn = [&](Function fn, Dim3 g, Dim3 b, u32 head0, bool copy, bool ranged, u32 r0, u32 r1) {
        AttnStep s;
        s.op = AttnStep::Op::attn;
        s.f = fn;
        s.grid = g;
        s.block = b;
        s.head0 = head0;
        s.from_copy = copy;
        s.ranged = ranged;
        s.r0 = r0;
        s.r1 = r1;
        st.push_back(s);
    };
    const u32 parts = headParts(static_cast<u64>(p.pos0) + n);
    const u32 hp = n_head / parts;
    const u32 grp = n_head / p.n_head_kv;
    Function kg = nullptr;
    u32 np = 0;
    int sexp = 0;
    const bool pow2 = std::frexp(p.scale, &sexp) == 0.5f;
    if (f.attn_kg_on && p.head_dim == 256 && pow2) {
        for (const auto& [fn, c] : {std::pair{k.attn_kg6, 6u}, std::pair{k.attn_kg4, 4u}, std::pair{k.attn_kg2, 2u}})
            if (fn != nullptr && grp % c == 0 && hp % c == 0) {
                kg = fn;
                np = c;
                break;
            }
    }
    const bool kx = f.attn_kx_on && k.attn_kx != nullptr && p.head_dim == 256;
    // q8 / q8h KV: keys 0 .. pos0 + n - 1 dequantized once into f16 rows (ffn_g / ffn_u, free during attention),
    // then the f16 attn_kg: the same f16 K / V values as attn_kg_q8 converts in its waves (bit-identical)
    const u32 rows = p.pos0 + n, row_el = p.n_head_kv * p.head_dim;
    if (kg != nullptr && f.attn_dqf_on && k.kv_dq_rows != nullptr && p.dq_ptab && rows <= p.dq_rows) {
        const Function kf = np == 6 ? k.attn_kgf6 : np == 4 ? k.attn_kgf4 : k.attn_kgf2;
        if (kf != nullptr) {
            AttnStep dq;
            dq.op = AttnStep::Op::dq_rows;
            dq.f = k.kv_dq_rows;
            dq.grid = D(static_cast<u32>((static_cast<u64>(rows) * row_el + 2047) / 2048));
            dq.nrows = rows;
            dq.row_el = row_el;
            st.push_back(dq);
            for (u32 q = 0; q < parts; ++q) attn(kf, D((n + 15) / 16, hp / np), D(64 * np), q * hp, true, false, 0, 0);
            return pl;
        }
    }
    auto dqRange = [&](u32 r0, u32 nr) {
        AttnStep dq;
        dq.op = AttnStep::Op::dq_rows_r;
        dq.f = k.kv_dq_rows_r;
        dq.grid = D(static_cast<u32>((static_cast<u64>(nr) * row_el + 2047) / 2048));
        dq.r0 = r0;
        dq.nrows = nr;
        dq.row_el = row_el;
        st.push_back(dq);
    };
    // KG-3 (gfx1151, no attn_kg): q8 / q8h / q4 keys dequantized into ffn_g / ffn_u (kv_dq_rows_r) and the f16
    // attn_prefill_wmma on them - one launch series when they fit, else key ranges of <= dq_rows walked by
    // attn_prefill_wmma_s with the softmax state carried. Bit-identical to attn_prefill_wmma_q8 / _q4.
    if (kg == nullptr && f.attn_dqf_on && k.attn_pwf != nullptr && k.attn_pws != nullptr && k.kv_dq_rows_r != nullptr && p.dq_ptab && p.dq_st_ml &&
        p.dq_rows >= 4096 && rows <= kDqMaxKeys && p.head_dim == 256) {
        const u32 seg = f.attn_seg >= 256 ? std::min(f.attn_seg, p.dq_rows / 256 * 256) : p.dq_rows / 256 * 256;
        for (u32 r0 = 0; r0 < rows; r0 += seg) {
            const u32 r1 = std::min(rows, r0 + seg), nr = r1 - r0;
            dqRange(r0, nr);
            if (r0 == 0 && r1 == rows) {  // all keys in one range: the plain f16 kernel
                for (u32 q = 0; q < parts; ++q) attn(k.attn_pwf, D((n + 127) / 128, hp), D(256), q * hp, true, false, r0, r1);
                break;
            }
            const u32 sp = headParts(nr), shp = n_head / sp;
            for (u32 q = 0; q < sp; ++q) attn(k.attn_pws, D((n + 127) / 128, shp), D(256), q * shp, true, true, r0, r1);
        }
        return pl;
    }
    // KG-3 (gfx1151), WHIRL_ATTN_SEG: f16 KV walked in key ranges straight from the cache (same series)
    if (kg == nullptr && !p.kv_q8 && f.attn_seg >= 256 && k.attn_pws != nullptr && p.dq_st_ml && p.head_dim == 256 && !kx && rows > f.attn_seg) {
        for (u32 r0 = 0; r0 < rows; r0 += f.attn_seg) {
            const u32 r1 = std::min(rows, r0 + f.attn_seg), nr = r1 - r0;
            const u32 sp = headParts(nr), shp = n_head / sp;
            for (u32 q = 0; q < sp; ++q) attn(k.attn_pws, D((n + 127) / 128, shp), D(256), q * shp, false, true, r0, r1);
        }
        return pl;
    }
    // FC-1c: q8 / q8h past the scratch (rows > dq_rows) and q4 (no native attn_kg): the keys in ranges of
    // <= dq_rows, each dequantized into ffn_g / ffn_u and walked by attn_kgs with the softmax state carried
    if (f.attn_dqf_on && k.kv_dq_rows_r != nullptr && p.dq_ptab && p.dq_st_ml && p.dq_rows >= 4096 && rows <= kDqMaxKeys && p.head_dim == 256 && pow2 &&
        f.attn_kg_on) {
        Function ks = nullptr;
        u32 nps = 0;
        for (const auto& [fn, c] : {std::pair{k.attn_kgs6, 6u}, std::pair{k.attn_kgs4, 4u}, std::pair{k.attn_kgs2, 2u}})
            if (fn != nullptr && grp % c == 0 && n_head % (c * 2) == 0) {
                ks = fn;
                nps = c;
                break;
            }
        if (ks != nullptr) {
            const u32 seg = p.dq_rows / 256 * 256;
            for (u32 r0 = 0; r0 < rows; r0 += seg) {
                const u32 r1 = std::min(rows, r0 + seg), nr = r1 - r0;
                dqRange(r0, nr);
                u32 sp = headParts(nr);
                while (sp > 1 && (n_head / sp) % nps != 0) sp += 1;
                if (n_head % sp != 0 || (n_head / sp) % nps != 0) sp = 1;
                const u32 shp = n_head / sp;
                for (u32 q = 0; q < sp; ++q) attn(ks, D((n + 15) / 16, shp / nps), D(64 * nps), q * shp, true, true, r0, r1);
            }
            return pl;
        }
    }
    for (u32 q = 0; q < parts; ++q) {
        if (kg != nullptr)
            attn(kg, D((n + 15) / 16, hp / np), D(64 * np), q * hp, false, false, 0, 0);
        else
            attn(kx ? k.attn_kx : k.attn_prefill_wmma, D((n + 127) / 128, hp), D(kx ? 512 : 256), q * hp, false, false, 0, 0);
    }
    return pl;
}

}  // namespace whirl::ops
