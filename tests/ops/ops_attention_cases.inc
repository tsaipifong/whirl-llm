// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.attention/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/device_profile.h"
#include "whirl/ops/attention.h"
#include "../test_check.h"
#include "ops_fakes.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <format>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ops = whirl::ops;

// ---- attention (K-3)
kn::KernelTable attnTable(bool ws4, bool group1, bool dq4, bool kg) {
    kn::KernelTable k;
    k.attn_split = tag("attn_split");
    k.attn_wsplit1 = tag("attn_wsplit1");
    k.attn_wsplit2 = tag("attn_wsplit2");
    k.attn_combine = tag("attn_combine");
    k.attn_combine_q8 = tag("attn_combine_q8");
    if (ws4) {
        k.attn_ws4 = tag("attn_ws4");
        for (int i = 0; i < 3; ++i) k.attn_ws4g[i] = tag("attn_ws4g" + std::to_string(i + 1));
    }
    if (dq4) k.attn_dq4 = tag("attn_dq4");
    k.caps.attn_group1 = group1;
    k.attn_prefill_wmma = tag("attn_prefill_wmma");
    k.attn_kx = tag("attn_kx");
    if (kg) {
        k.attn_kg2 = tag("attn_kg2");
        k.attn_kg4 = tag("attn_kg4");
        k.attn_kgf4 = tag("attn_kgf4");
        k.attn_kgs4 = tag("attn_kgs4");
        k.kv_dq_rows = tag("kv_dq_rows");
        k.kv_dq_rows_r = tag("kv_dq_rows_r");
    } else {
        k.attn_pwf = tag("attn_pwf");
        k.attn_pws = tag("attn_pws");
        k.kv_dq_rows_r = tag("kv_dq_rows_r");
    }
    return k;
}

void testAttnDecode() {
    const auto& p97 = whirl::device::k_profile_gfx1201;
    const auto& p8s = whirl::device::k_profile_gfx1151;
    ops::AttnDecodeProblem p;
    p.n = 1;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;  // GQA group 6
    p.max_ctx = 32768;
    ops::AttnDecodeFlags f;
    // gfx1201-like set (no ws4, groups allowed): one decode row on attn_wsplit1, 64 splits, combine_q8 when the
    // o-projection is on the int8 GEMV
    const kn::KernelTable k97 = attnTable(false, false, true, true);
    p.o_on_gemv = true;
    auto pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_wsplit1" && pl.kind == ops::AttnDecodeKind::wsplit && pl.grid.x == 4 && pl.grid.y == 64 && pl.grid.z == 1);
    CHECK(pl.block.x == 128 && pl.combine_q8 && pl.combine_grid.x == 24 && pl.combine_grid.y == 1);
    // precise never takes the int8 combine
    ops::AttnDecodeFlags fp = f;
    fp.prec_dec = true;
    CHECK(!ops::chooseAttnDecode(p, fp, k97, k97.caps, p97).combine_q8);
    // a 4-row verify at consecutive positions of one sequence: one group of 2 (16 / 6 columns), then 2
    const std::int32_t pos[4] = {1000, 1001, 1002, 1003}, base[4] = {0, 0, 0, 0};
    p.n = 4;
    p.rows_known = true;
    p.row_pos = pos;
    p.row_base = base;
    f.attn_wide = false;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_wsplit1" && pl.grid.z == 2 && pl.groups.first[1] == 2 && pl.groups.count[0] == 2);
    // with attn_wide: attn_wsplit2 takes 5 rows per group (32 / 6): one group of 4
    f.attn_wide = true;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_wsplit2" && pl.grid.z == 1 && pl.groups.count[0] == 4);
    // q4 KV with attn_dq4: one query per group on attn_dq4, 256 threads. v0252-n2 (FAST-8S-1 item 1 on gfx1201): the R9700
    // (no attn_group1 marker; dq4 rows are one query per group anyway) takes the dynamic split count for host-known rows
    // too: the smallest grid with every row's key partition equal to the 256 grid's (same partials, same combine order)
    p.kv_q4 = true;
    f.fd_splits = 256;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.kind == ops::AttnDecodeKind::dq4 && pl.grid.z == 4 && pl.block.x == 256);
    CHECK(pl.n_split < 256 && pl.n_split == ops::dq4Splits(std::span<const std::int32_t>(pos, 4), 0, 256) && pl.grid.y == pl.n_split);
    // ... WHIRL_DQ4_DYN=0: the fixed grid
    {
        ops::AttnDecodeFlags f0 = f;
        f0.dq4_dyn = false;
        CHECK(ops::chooseAttnDecode(p, f0, k97, k97.caps, p97).n_split == 256);
    }
    // ... rows the host does not know (graph-captured plain decode): the fixed grid
    {
        ops::AttnDecodeProblem p1 = p;
        p1.n = 1;
        p1.rows_known = false;
        p1.row_pos = {};
        p1.row_base = {};
        CHECK(ops::chooseAttnDecode(p1, f, k97, k97.caps, p97).n_split == 256);
    }
    // ... MTP draft window (W 16384 from 64k): the rule follows the window branch, still the 256 grid's partition
    {
        const std::int32_t lpos[4] = {70000, 70001, 70002, 70003};
        ops::AttnDecodeProblem pw = p;
        pw.max_ctx = 131072;
        pw.row_pos = lpos;
        pw.win = (16384 / 64) | ((65536 / 1024) << 16);
        const std::uint32_t sw = ops::dq4Splits(std::span<const std::int32_t>(lpos, 4), pw.win, 256);
        CHECK(sw < 256 && ops::chooseAttnDecode(pw, f, k97, k97.caps, p97).n_split == sw);
    }
    // the 8060S (attn_group1 set): the dynamic split count for short rows (same partition), FAST-8S-1 - unchanged
    const kn::KernelTable k8 = attnTable(true, true, true, false);
    pl = ops::chooseAttnDecode(p, f, k8, k8.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.n_split < 256 && pl.n_split == ops::dq4Splits(std::span<const std::int32_t>(pos, 4), 0, 256));
    // ... a profile row decides it, not the kernel set: a row without attn_dq4_dynamic_splits keeps the fixed grid
    // under either kernel set
    {
        whirl::device::Profile poff = p97;
        poff.attn_dq4_dynamic_splits = false;
        CHECK(ops::chooseAttnDecode(p, f, k8, k8.caps, poff).n_split == 256);
        CHECK(ops::chooseAttnDecode(p, f, k97, k97.caps, poff).n_split == 256);
    }
    // f16 KV on the 8060S (an f16-KV kernel table has no attn_dq4): attn_ws4g for a multi-row verify group (64 splits)
    const kn::KernelTable k8f = attnTable(true, true, false, false);
    p.kv_q4 = false;
    f.fd_splits = 64;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(pl.kind == ops::AttnDecodeKind::ws4g && nameOf(pl.f).rfind("attn_ws4g", 0) == 0 && pl.grid.y == 64 && pl.grid.z == 1 && pl.block.x == 256);
    // rows of two sequences never share a group
    const std::int32_t base2[4] = {0, 0, 4096, 4096};
    p.row_base = base2;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(pl.grid.z == 2);
    // WHIRL_ATTN_WS4=0 on the 8060S: attn_wsplit1, one query per group (attn_group1)
    f.attn_ws4 = false;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_wsplit1" && pl.grid.z == 4);
    // dbg_flags bit 2 forces attn_split (grid.z = rows, 256 threads)
    f.dbg_flags = 2;
    pl = ops::chooseAttnDecode(p, f, k8f, k8f.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_split" && pl.kind == ops::AttnDecodeKind::split && pl.grid.z == 4 && pl.block.x == 256);
    CHECK(ops::awPer(1, 64) == 64 && ops::awPer(4096, 64) == 64 && ops::awPer(4097, 64) == 128);
}

// v0252-n2 (taken over from feat027-dq4r9700 764af36, tests/ops/ops_tests.cpp, where it was written and mutation-checked
// against the pre-0.2.5 tree; never merged): host model of attn_dq4's split ranges, written from the kernel source
// (kernels/shared/attn_dq4.h + attn_dq_nw.h, one copy for both families: dq_per, DQ_NW 8, DQ_MAXS 128, the MTP
// draft-window branch), not from ops::dq4Part. A dynamic grid of s splits is bitwise the fixed grid of `full` splits
// when every split i < s has the same [b0, b1) under both grids (same partial) and every split i >= s of the full grid
// is empty (b0 >= n_pos: it writes m = -inf, which attn_combine / attn_combine_q8 skip, and fmaxf(M, -inf) == M): the
// combine then sees the same partials in the same order.
struct KRange {
    std::int32_t b0, b1;
};
std::int32_t kDqPer(std::int32_t n_keys, std::int32_t n_split) {
    const std::int32_t ns = std::min<std::int32_t>(n_split, 128);
    return std::max<std::int32_t>(8 * 32, (n_keys + ns * 64 - 1) / (ns * 64) * 64);
}
KRange kRange(std::int32_t n_pos, std::int32_t split, std::int32_t n_split, std::int32_t win) {
    std::int32_t per = kDqPer(n_pos, n_split);
    std::int32_t b0 = split * per, b1 = std::min(n_pos, b0 + per);
    if (win != 0 && n_split > 1) {
        const std::int32_t w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
        const std::int32_t w0 = std::max<std::int32_t>(256, (n_pos - w) & ~63);
        if (n_pos >= min_pos && w0 > 256) {
            if (split == 0) {
                b0 = 0;
                b1 = 256;
            } else {
                per = kDqPer(n_pos - w0, n_split - 1);
                b0 = w0 + (split - 1) * per;
                b1 = std::min(n_pos, b0 + per);
            }
        }
    }
    return {b0, b1};
}
// the dynamic grid s gives row n_pos the fixed grid's partials and combine sequence
bool samePartials(std::int32_t n_pos, std::uint32_t s, std::uint32_t full, std::int32_t win) {
    if (s > full) return false;
    for (std::uint32_t i = 0; i < full; ++i) {
        const KRange b = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(full), win);
        if (i < s) {
            const KRange a = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(s), win);
            const bool ea = a.b0 >= n_pos, eb = b.b0 >= n_pos;
            if (ea != eb) return false;
            if (!ea && (a.b0 != b.b0 || a.b1 != b.b1)) return false;
        } else if (b.b0 < n_pos) {
            return false;  // a keyed split the dynamic grid does not launch
        }
    }
    return true;
}
std::int32_t dq4WinArg(std::uint32_t w, std::uint32_t min_ctx) {  // qwen35::draftWindowArg
    if (w == 0) return 0;
    const std::uint32_t lo = std::min<std::uint32_t>((w + 63) / 64, 0xffffu), hi = std::min<std::uint32_t>(min_ctx / 1024, 0x7fffu);
    return static_cast<std::int32_t>(lo | hi << 16);
}

void testDq4Partition() {
    // windows: off, the default (16384 keys from 64k), a short one from 0, one from 16k
    const std::int32_t wins[4] = {0, dq4WinArg(16384, 65536), dq4WinArg(4096, 0), dq4WinArg(1024, 16384)};
    // every n_pos 1 .. 256k for the q4 default grid (256) and a non-power-of-two WHIRL_FD_SPLITS (100); step 7 for
    // the smaller grids (max_ctx / 64 caps the grid at short contexts)
    const std::uint32_t fulls[6] = {256, 100, 128, 64, 32, 16};
    long long rows = 0, bad = 0, smaller = 0, notmin = 0;
    for (const std::uint32_t full : fulls) {
        const std::int32_t step = (full == 256 || full == 100) ? 1 : 7;
        for (const std::int32_t win : wins) {
            for (std::int32_t n_pos = 1; n_pos <= 262144; n_pos += step) {
                const std::int32_t pos = n_pos - 1;
                const std::uint32_t s = ops::dq4Splits(std::span<const std::int32_t>(&pos, 1), win, full);
                rows += 1;
                if (!samePartials(n_pos, s, full, win)) bad += 1;
                if (s < full) smaller += 1;
                // minimal: every smaller candidate (8, 16, ... < s) changes some partial (kernel model), so the
                // choice is the cheapest bitwise grid (a mutant that keeps the full grid or a larger count fails)
                for (std::uint32_t c = 8; c < s; c *= 2)
                    if (samePartials(n_pos, c, full, win)) {
                        notmin += 1;
                        break;
                    }
            }
        }
    }
    CHECK(bad == 0);
    CHECK(smaller > 0);
    CHECK(notmin == 0);
    // dq4Splits' third test (keyed splits a[2] <= s) never decides (feat027-dq4r9700 mutcheck: the mutant that drops it
    // survives): with the same branch and per, per = dq_per(n, min(s, 128)) >= ceil(n / s), so ceil(n / per) <= s; in
    // the window branch per = dq_per(n - w0, s - 1) gives 1 + ceil((n - w0) / per) <= s. Shown here for every n_pos
    // 1 .. 256k, s = 8 .. 128 and the four windows. The check stays in dq4Splits as a guard.
    long long redundant_hits = 0;
    for (const std::int32_t win : wins)
        for (std::int32_t n_pos = 1; n_pos <= 262144; ++n_pos)
            for (std::uint32_t c = 8; c <= 128; c *= 2) {
                const KRange a0 = kRange(n_pos, 0, static_cast<std::int32_t>(c), win), b0 = kRange(n_pos, 0, 256, win);
                const KRange a1 = kRange(n_pos, 1, static_cast<std::int32_t>(c), win), b1 = kRange(n_pos, 1, 256, win);
                if (a0.b0 != b0.b0 || a0.b1 != b0.b1 || a1.b0 != b1.b0 || a1.b1 != b1.b1) continue;
                if (kRange(n_pos, static_cast<std::int32_t>(c), 256, win).b0 < n_pos) redundant_hits += 1;
            }
    CHECK(redundant_hits == 0);
    // verify / MTP row sets: k consecutive rows of one sequence (k = 2, 9, 16) and two sequences far apart; every
    // row of the set must keep its partials under the set's one split count
    long long sets = 0, sbad = 0;
    for (const std::int32_t win : wins) {
        for (std::int32_t p0 = 0; p0 < 262144; p0 += 61) {
            std::int32_t rp[16];
            for (std::int32_t k = 2; k <= 16; k += 7) {
                for (std::int32_t r = 0; r < k; ++r) rp[r] = std::min(p0 + r, 262143);
                const std::uint32_t s = ops::dq4Splits(std::span<const std::int32_t>(rp, static_cast<std::size_t>(k)), win, 256);
                sets += 1;
                for (std::int32_t r = 0; r < k; ++r)
                    if (!samePartials(rp[r] + 1, s, 256, win)) sbad += 1;
            }
            const std::int32_t two[2] = {p0, (p0 * 7 + 4099) % 262144};
            const std::uint32_t s2 = ops::dq4Splits(std::span<const std::int32_t>(two, 2), win, 256);
            sets += 1;
            for (const std::int32_t q : two)
                if (!samePartials(q + 1, s2, 256, win)) sbad += 1;
        }
    }
    CHECK(sbad == 0);
    // n_pos 3000: grid 8 has per 384 (the 256 grid: 256), so its partials differ; grid 16 has per 256 = the 256 grid's
    CHECK(!samePartials(3000, 8, 256, 0) && samePartials(3000, 16, 256, 0));
    {
        const std::int32_t pos = 2999;
        CHECK(ops::dq4Splits(std::span<const std::int32_t>(&pos, 1), 0, 256) == 16);
    }
    // launch-table snapshot (gfx1201 / gfx1151 profile rows, q4 KV, fd_splits 256, one decode / verify row at n_pos keys):
    // the attn_dq4 grid.y and the combine's n_split, both families. Values from an independent Python model
    // (progress/feat027-dq4r9700.HANDOFF.md); window off and the default window give the same column.
    struct Snap {
        std::int32_t n_pos;
        std::uint32_t s;
    };
    const Snap snap[] = {{1, 8},       {256, 8},     {257, 8},     {2048, 8},    {2049, 16},    {4096, 16},    {4097, 32},     {8192, 32},    {8193, 64},
                         {16384, 64},  {16385, 128}, {32768, 128}, {65536, 128}, {65537, 128}, {131072, 128}, {131073, 128}, {262144, 128}};
    ops::AttnDecodeProblem p;
    p.n = 1;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.max_ctx = 262144;
    p.kv_q4 = true;
    p.rows_known = true;
    const std::int32_t base[1] = {0};
    p.row_base = base;
    ops::AttnDecodeFlags f;
    f.fd_splits = 256;
    const kn::KernelTable k97 = attnTable(false, false, true, true);
    const kn::KernelTable k8 = attnTable(true, true, true, false);
    long long snap_bad = 0;
    for (const Snap& e : snap) {
        const std::int32_t pos[1] = {e.n_pos - 1};
        p.row_pos = pos;
        for (const std::int32_t win : {wins[0], wins[1]}) {
            p.win = win;
            const auto a = ops::chooseAttnDecode(p, f, k97, k97.caps, whirl::device::k_profile_gfx1201);
            const auto b = ops::chooseAttnDecode(p, f, k8, k8.caps, whirl::device::k_profile_gfx1151);
            if (nameOf(a.f) != "attn_dq4" || a.n_split != e.s || a.grid.y != e.s || nameOf(b.f) != "attn_dq4" || b.n_split != e.s) {
                std::printf("  snapshot n_pos %d win %d: gfx1201 %u, gfx1151 %u, want %u\n", e.n_pos, win, a.n_split, b.n_split, e.s);
                snap_bad += 1;
            }
        }
    }
    CHECK(snap_bad == 0);
    std::printf("  dq4 partition: %lld rows (%lld on a smaller grid), %lld row sets, mismatches %lld / %lld, not minimal %lld, keyed > s %lld\n",
                rows, smaller, sets, bad, sbad, notmin, redundant_hits);
}

void testAttnPrefill() {
    ops::AttnPrefillProblem p;
    p.n = 512;
    p.pos0 = 0;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.scale = 1.0f / 16.0f;  // 1 / sqrt(256): a power of two
    ops::AttnPrefillFlags f;
    // attn_kg set, f16 KV: attn_kg4 (group 6 is not a multiple of 4 -> kg2), one launch, from the cache
    kn::KernelTable k = attnTable(false, false, false, true);
    auto pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == "attn_kg2" && !pl.steps[0].from_copy && pl.steps[0].grid.x == 32 && pl.steps[0].grid.y == 12 &&
          pl.steps[0].block.x == 128);
    // q8 KV with the dequant scratch: kv_dq_rows once, then attn_kgf (only kgf4 exists here -> kg2's kgf2 is null -> falls through)
    p.kv_q8 = true;
    p.dq_rows = 8192;
    p.dq_ptab = true;
    p.dq_st_ml = true;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(!pl.steps.empty());
    // a non-power-of-two scale: no attn_kg; attn_kx (512 threads)
    p.kv_q8 = false;
    p.scale = 0.1f;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == "attn_kx" && pl.steps[0].block.x == 512 && pl.steps[0].grid.x == 4 && pl.steps[0].grid.y == 24);
    // the 8060S-like set (no attn_kg, pwf / pws): q8 KV in one dequantized range -> kv_dq_rows_r + attn_pwf
    k = attnTable(true, true, true, false);
    p.scale = 1.0f / 16.0f;
    p.kv_q8 = true;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 2 && pl.steps[0].op == ops::AttnStep::Op::dq_rows_r && nameOf(pl.steps[1].f) == "attn_pwf" && pl.steps[1].from_copy &&
          !pl.steps[1].ranged);
    // keys past the scratch: ranges of dq_rows, each dequantized and walked by attn_pws with the state carried
    p.pos0 = 20000;
    pl = ops::chooseAttnPrefill(p, f, k);
    std::size_t dq = 0, ranged = 0;
    for (const auto& s : pl.steps) {
        dq += s.op == ops::AttnStep::Op::dq_rows_r;
        ranged += s.op == ops::AttnStep::Op::attn && s.ranged && nameOf(s.f) == "attn_pws";
    }
    CHECK(dq == 3 && ranged == 3 && pl.steps.back().r1 == 20512);
    // a long prefill splits the heads (Windows TDR budget): 4096 x 128k query-key pairs per launch
    p.kv_q8 = false;
    p.pos0 = 0;
    p.n = 8192;
    p.dq_rows = 0;
    k = attnTable(false, false, false, true);
    p.scale = 0.1f;
    p.n = 4096;
    p.pos0 = 131072;
    pl = ops::chooseAttnPrefill(p, f, k);
    CHECK(pl.steps.size() == 2 && pl.steps[1].head0 == 12 && pl.steps[0].grid.y == 12);
}

// v0252-c1port (C1, feat026-attnkg97 / -97k; DISCOVERY v4 C3-1): the f16 attn_kg variant selector. ops::attnKgVariant is
// exhaustive (np 6 / 4 / 2 written out, any other np -> nullptr: never the kernel of another NP, whose grid / block would
// not match), and chooseAttnPrefill takes the variant only on the plain f16 path and on Q8P (kv_dq_rows + the f16
// attn_kgf), with the shipped kernel's grid / block; q8 / q8v without the scratch, multi-range FC-1c (attn_kgs) and the
// 8060S set keep their kernels (v0252-fasta: one-range FC-1c, the q4 fast path, takes var 3 = attn_kgk, case 6). Fails
// before the port (test-first stub: attnKgVariant returns nullptr, chooseAttnPrefill never asks it).
kn::KernelTable kgTable(bool variants) {
    kn::KernelTable k = attnTable(false, false, false, true);
    for (const char* np : {"6", "4", "2"}) {
        const std::string s = np;
        Fn* kg = s == "6" ? &k.attn_kg6 : s == "4" ? &k.attn_kg4 : &k.attn_kg2;
        Fn* kf = s == "6" ? &k.attn_kgf6 : s == "4" ? &k.attn_kgf4 : &k.attn_kgf2;
        Fn* ks = s == "6" ? &k.attn_kgs6 : s == "4" ? &k.attn_kgs4 : &k.attn_kgs2;
        *kg = tag("attn_kg" + s);
        *kf = tag("attn_kgf" + s);
        *ks = tag("attn_kgs" + s);
        if (variants) {
            *(s == "6" ? &k.attn_kgv6 : s == "4" ? &k.attn_kgv4 : &k.attn_kgv2) = tag("attn_kgv" + s);
            *(s == "6" ? &k.attn_kgk6 : s == "4" ? &k.attn_kgk4 : &k.attn_kgk2) = tag("attn_kgk" + s);
        }
    }
    return k;
}

// every attn_kg-family launch (kg / kgv / kgk / kgf / kgs <NP>) has the block of its own NP: 64 * NP threads
bool kgShapesMatch(const ops::AttnPrefillPlan& pl) {
    for (const auto& s : pl.steps) {
        if (s.op != ops::AttnStep::Op::attn) continue;
        const std::string n = nameOf(s.f);
        if (n.rfind("attn_kg", 0) != 0 || n.empty()) continue;
        const char d = n.back();
        if (d != '6' && d != '4' && d != '2') return false;
        if (s.block.x != 64u * static_cast<unsigned>(d - '0')) return false;
    }
    return true;
}

void testAttnKgAuto();
void testAttnFastQ4Kgk();  // v0252-fasta

void testAttnKgVariant() {
    // v0252-c1asan: a KernelTable is large and under ASan every local gets its own stack slot; main -> testAttnKgVariant ->
    // testAttnKgAuto -> kgTable -> attnTable overflowed the 1 MB main stack of whirl-ops-tests. The tables are built once, in
    // static storage (the mutated copy on the heap), like llama.cpp's test-backend-ops keeps its test cases on the heap.
    static const kn::KernelTable kv = kgTable(true), k0 = kgTable(false);
    // (1) the selector alone: var 1 -> attn_kgv<np>, var 3 -> attn_kgk<np>, np 6 / 4 / 2 only
    for (const std::uint32_t np : {6u, 4u, 2u}) {
        const std::string s = std::to_string(np);
        CHECK(nameOf(ops::attnKgVariant(1, kv, np)) == "attn_kgv" + s);
        CHECK(nameOf(ops::attnKgVariant(3, kv, np)) == "attn_kgk" + s);
        for (const std::uint32_t var : {0u, 2u, 4u, 7u}) CHECK(ops::attnKgVariant(var, kv, np) == nullptr);  // var 2 attn_kgw: not ported
        for (const std::uint32_t var : {1u, 3u}) CHECK(ops::attnKgVariant(var, k0, np) == nullptr);  // not in the code object
    }
    // any other np (C11's NP 8 among them): nullptr, never the NP-2 (or any other) kernel
    for (const std::uint32_t np : {8u, 16u, 12u, 3u, 1u, 0u})
        for (const std::uint32_t var : {0u, 1u, 2u, 3u}) CHECK(ops::attnKgVariant(var, kv, np) == nullptr);
    // (2) chooseAttnPrefill, f16 KV, one launch from the cache (512 queries, 2048 keys): Swift 24 / 4 (group 6 -> NP 6),
    // Ornith 16 / 2 (group 8 -> NP 4), 4 / 2 (group 2 -> NP 2); grid (32, heads / NP), block 64 NP for every var
    struct Shape {
        std::uint32_t heads, kv, np;
    };
    for (const Shape sh : {Shape{24, 4, 6}, Shape{16, 2, 4}, Shape{4, 2, 2}}) {
        const std::string s = std::to_string(sh.np);
        ops::AttnPrefillProblem p;
        p.n = 512;
        p.pos0 = 1536;
        p.head_dim = 256;
        p.n_head = sh.heads;
        p.n_head_kv = sh.kv;
        p.scale = 1.0f / 16.0f;
        auto plan = [&](std::uint32_t var, const kn::KernelTable& k) {
            ops::AttnPrefillFlags f;
            f.attn_kg_var = var;
            return ops::chooseAttnPrefill(p, f, k);
        };
        const std::pair<std::uint32_t, const char*> f16_want[] = {{0, "attn_kg"}, {1, "attn_kgv"}, {3, "attn_kgk"}, {2, "attn_kg"}, {5, "attn_kg"}};
        for (const auto& [var, want] : f16_want) {
            const auto pl = plan(var, kv);
            CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == want + s && !pl.steps[0].from_copy && !pl.steps[0].ranged &&
                  pl.steps[0].grid.x == 32 && pl.steps[0].grid.y == sh.heads / sh.np && pl.steps[0].block.x == 64 * sh.np &&
                  pl.steps[0].head0 == 0 && kgShapesMatch(pl));
            // without the variants in the code object: the shipped kernel, same launch
            const auto pz = plan(var, k0);
            CHECK(pz.steps.size() == 1 && nameOf(pz.steps[0].f) == "attn_kg" + s && pz.steps[0].block.x == 64 * sh.np);
        }
        // (3) Q8P (q8 / q8h KV, keys inside the dequant scratch): kv_dq_rows once, then the variant in place of attn_kgf
        p.kv_q8 = true;
        p.dq_rows = 8192;
        p.dq_ptab = true;
        p.dq_st_ml = true;
        const std::pair<std::uint32_t, const char*> q8p_want[] = {{0, "attn_kgf"}, {1, "attn_kgv"}, {3, "attn_kgk"}, {2, "attn_kgf"}};
        for (const auto& [var, want] : q8p_want) {
            const auto pl = plan(var, kv);
            CHECK(pl.steps.size() == 2 && pl.steps[0].op == ops::AttnStep::Op::dq_rows && pl.steps[0].nrows == 2048 &&
                  nameOf(pl.steps[1].f) == want + s && pl.steps[1].from_copy && !pl.steps[1].ranged && pl.steps[1].grid.x == 32 &&
                  pl.steps[1].grid.y == sh.heads / sh.np && pl.steps[1].block.x == 64 * sh.np && kgShapesMatch(pl));
        }
        // (4) q8 / q8h past the scratch (FC-1c: key ranges + attn_kgs): no variant, every var
        p.pos0 = 20000;
        for (const std::uint32_t var : {0u, 1u, 3u}) {
            const auto pl = plan(var, kv);
            std::size_t kgs = 0, other = 0;
            for (const auto& st : pl.steps)
                if (st.op == ops::AttnStep::Op::attn) (nameOf(st.f) == "attn_kgs" + s && st.ranged ? kgs : other) += 1;
            CHECK(kgs == 3 && other == 0 && pl.steps.back().r1 == 20512 && kgShapesMatch(pl));
        }
        // (5) q8 / q8v from the cache (no dequant scratch): the shipped attn_kg<np> (attn_kg<np>_q8 / _q8v in a real
        // table), every var
        p.pos0 = 1536;
        p.dq_rows = 0;
        p.dq_ptab = false;
        p.dq_st_ml = false;
        for (const std::uint32_t var : {0u, 1u, 3u}) {
            const auto pl = plan(var, kv);
            CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == "attn_kg" + s && !pl.steps[0].from_copy && kgShapesMatch(pl));
        }
        // (6) q4 KV on the gfx1201 set (no attn_kg*_q4: FC-1c over dequantized ranges), all 2048 keys in one range.
        // v0252-fasta (KgPath::dqr1): var 0 and var 1 keep attn_kgs (ranged); var 3 takes attn_kgk<np> (one launch on the
        // copy, not ranged); auto takes attn_kgk for NP 4 / 6 (the M0-A shapes) and attn_kgs for NP 2 (not measured)
        const auto kq4p = std::make_unique<kn::KernelTable>(kv);  // heap, see the note at the top of the function
        kn::KernelTable& kq4 = *kq4p;
        kq4.attn_kg6 = kq4.attn_kg4 = kq4.attn_kg2 = nullptr;
        kq4.kv_dq_rows = nullptr;
        p.dq_rows = 8192;
        p.dq_ptab = true;
        p.dq_st_ml = true;
        for (const std::uint32_t var : {0u, 1u, 3u, ops::kAttnKgVarAuto}) {
            const auto pl = plan(var, kq4);
            const bool kgk = var == 3 || (var == ops::kAttnKgVarAuto && sh.np != 2);
            std::size_t kgs = 0, kgks = 0, other = 0;
            for (const auto& st : pl.steps)
                if (st.op == ops::AttnStep::Op::attn) {
                    if (nameOf(st.f) == "attn_kgs" + s && st.ranged) kgs += 1;
                    else if (nameOf(st.f) == "attn_kgk" + s && !st.ranged && st.from_copy) kgks += 1;
                    else other += 1;
                }
            CHECK(kgs == (kgk ? 0u : 1u) && kgks == (kgk ? 1u : 0u) && other == 0 && kgShapesMatch(pl));
        }
        p.kv_q8 = false;
        p.dq_rows = 0;
        p.dq_ptab = false;
        p.dq_st_ml = false;
        // (7) a long prefill split into head ranges (Windows TDR budget): each range gets the variant, head0 0 / heads / 2
        p.n = 4096;
        p.pos0 = 131072;
        {
            const auto pl = plan(3, kv);
            const std::uint32_t hp = sh.heads / 2;
            CHECK(pl.steps.size() == 2 && nameOf(pl.steps[0].f) == "attn_kgk" + s && nameOf(pl.steps[1].f) == "attn_kgk" + s &&
                  pl.steps[0].head0 == 0 && pl.steps[1].head0 == hp && pl.steps[0].grid.y == hp / sh.np && kgShapesMatch(pl));
        }
    }
    // (8) the 8060S set (no attn_kg; attn_prefill_wmma / pwf / pws): WHIRL_ATTN_KGVAR changes nothing, f16 and q8 KV
    {
        static const kn::KernelTable k8 = attnTable(true, true, true, false);
        ops::AttnPrefillProblem p;
        p.n = 512;
        p.head_dim = 256;
        p.n_head = 24;
        p.n_head_kv = 4;
        p.scale = 1.0f / 16.0f;
        for (const bool q8 : {false, true})
            for (const std::uint32_t pos0 : {0u, 20000u}) {
                p.kv_q8 = q8;
                p.pos0 = pos0;
                p.dq_rows = q8 ? 8192 : 0;
                p.dq_ptab = p.dq_st_ml = q8;
                ops::AttnPrefillFlags f0, f3;
                f3.attn_kg_var = 3;
                const auto a = ops::chooseAttnPrefill(p, f0, k8), b = ops::chooseAttnPrefill(p, f3, k8);
                bool same = a.steps.size() == b.steps.size() && !a.steps.empty();
                for (std::size_t i = 0; same && i < a.steps.size(); ++i) {
                    const auto &x = a.steps[i], &y = b.steps[i];
                    same = x.op == y.op && x.f == y.f && x.grid.x == y.grid.x && x.grid.y == y.grid.y && x.block.x == y.block.x &&
                           x.from_copy == y.from_copy && x.ranged == y.ranged && x.r0 == y.r0 && x.r1 == y.r1 && x.head0 == y.head0;
                }
                CHECK(same);
            }
    }
    testAttnKgAuto();  // v0252-c1acc
}


// v0252-c1acc (REVIEW-v0252-c1 C-1 / C-3): attn_kg_var "auto" (ops::kAttnKgVarAuto, the gfx1201 default) = the explicit
// table ops::kAttnKgAuto: var 3 only on NP 4 plain f16 (Ornith precise / balance) and NP 6 Q8P (Swift balance), var 0 on
// every other (NP, KV path): Swift precise f16 (NP 6, unproven: AO-1 confound), NP 4 Q8P (not measured), NP 2 and any
// other np. Explicit 0 / 1 / 3 bypass the table; the 8060S set and multi-range FC-1c are unchanged (v0252-fasta: the
// dqr1 rows, testAttnFastQ4Kgk). Fails before c1acc (the test-first stub attnKgAutoVar returns 0 for every row).
// v0252-c1np6 (PLAN V1-52): fifth row {6, f16, 3} = Swift precise (plain f16 KV, NP 6) after AO-1 lifted the confound
// (c1acc same-object e2e 131072 +3.59..+3.70 % in 6/6 rounds, attnp kgk6 f16 -11.6 %; progress\v0252-c1np6.HANDOFF.md).
// NP 6 now takes attn_kgk6 on both the f16 and the Q8P path. Fails before c1np6 (table of 4 rows, Swift f16 -> attn_kg6).
void testAttnKgAuto() {
    CHECK(ops::kAttnKgVarAuto == 255);  // == numerics::kAttnKgVarAuto (static_assert in src/model/qwen35/forward/attn.cpp)
    // (1) the table: exactly the measured rows (C1: two; v0252-fasta: dqr1 for NP 4 and NP 6, M0-A; v0252-c1np6: NP 6 f16);
    // every other (np, path) -> 0
    CHECK(std::size(ops::kAttnKgAuto) == 5);
    for (const std::uint32_t np : {0u, 1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u})
        for (const ops::KgPath path : {ops::KgPath::f16, ops::KgPath::q8p, ops::KgPath::dqr1}) {
            const bool on = (np == 4 && path == ops::KgPath::f16) || (np == 6 && path == ops::KgPath::q8p) || (np == 6 && path == ops::KgPath::f16) ||
                            ((np == 4 || np == 6) && path == ops::KgPath::dqr1);
            CHECK(ops::attnKgAutoVar(np, path) == (on ? 3u : 0u));
            CHECK(ops::attnKgVarFor(ops::kAttnKgVarAuto, np, path) == (on ? 3u : 0u));
            for (const std::uint32_t var : {0u, 1u, 3u, 2u}) CHECK(ops::attnKgVarFor(var, np, path) == var);  // explicit: as given
        }
    // (2) chooseAttnPrefill with auto: the kernel per model shape and KV path
    static const kn::KernelTable kv = kgTable(true);  // static: see testAttnKgVariant
    struct Want {
        std::uint32_t heads, kv, np;
        const char *f16, *q8p;
    };
    for (const Want w : {Want{24, 4, 6, "attn_kgk6", "attn_kgk6"}, Want{16, 2, 4, "attn_kgk4", "attn_kgf4"}, Want{4, 2, 2, "attn_kg2", "attn_kgf2"}}) {
        ops::AttnPrefillProblem p;
        p.n = 512;
        p.pos0 = 1536;
        p.head_dim = 256;
        p.n_head = w.heads;
        p.n_head_kv = w.kv;
        p.scale = 1.0f / 16.0f;
        ops::AttnPrefillFlags f;
        f.attn_kg_var = ops::kAttnKgVarAuto;
        {
            const auto pl = ops::chooseAttnPrefill(p, f, kv);
            CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == w.f16 && pl.steps[0].block.x == 64 * w.np && kgShapesMatch(pl));
        }
        // a long prefill split into head ranges: every range takes the same kernel
        p.n = 4096;
        p.pos0 = 131072;
        {
            const auto pl = ops::chooseAttnPrefill(p, f, kv);
            CHECK(pl.steps.size() == 2 && nameOf(pl.steps[0].f) == w.f16 && nameOf(pl.steps[1].f) == w.f16 && kgShapesMatch(pl));
        }
        p.n = 512;
        p.pos0 = 1536;
        p.kv_q8 = true;
        p.dq_rows = 8192;
        p.dq_ptab = true;
        p.dq_st_ml = true;
        {
            const auto pl = ops::chooseAttnPrefill(p, f, kv);
            CHECK(pl.steps.size() == 2 && pl.steps[0].op == ops::AttnStep::Op::dq_rows && nameOf(pl.steps[1].f) == w.q8p && kgShapesMatch(pl));
        }
        // FC-1c (keys past the scratch): attn_kgs, never a variant
        p.pos0 = 20000;
        {
            const auto pl = ops::chooseAttnPrefill(p, f, kv);
            bool kgs_only = true;
            for (const auto& st : pl.steps)
                if (st.op == ops::AttnStep::Op::attn) kgs_only = kgs_only && nameOf(st.f) == "attn_kgs" + std::to_string(w.np);
            CHECK(kgs_only && kgShapesMatch(pl));
        }
        // explicit 0 overrides the table on both paths; explicit 3 takes attn_kgk on both
        p.pos0 = 1536;
        for (const bool q8 : {false, true}) {
            p.kv_q8 = q8;
            p.dq_ptab = q8;  // production allocates dq_ptab only for quantized KV (model_scratch.cpp:64-72); f16 KV never has it
            ops::AttnPrefillFlags f0, f3;
            f0.attn_kg_var = 0;
            f3.attn_kg_var = 3;
            const auto a = ops::chooseAttnPrefill(p, f0, kv), b = ops::chooseAttnPrefill(p, f3, kv);
            const std::string s = std::to_string(w.np);
            CHECK(nameOf(a.steps.back().f) == (q8 ? "attn_kgf" : "attn_kg") + s && nameOf(b.steps.back().f) == "attn_kgk" + s);
        }
    }
    // (2b) v0252-c1np6: the row is per NP, not per model: Swift 24 / 4 past 256k keys (n 4096, 3 head ranges of 8 heads)
    // falls to NP 2, which is not in the table -> the shipped attn_kg2 under auto
    {
        ops::AttnPrefillProblem p;
        p.n = 4096;
        p.pos0 = 3 * 131072 - 4096;
        p.head_dim = 256;
        p.n_head = 24;
        p.n_head_kv = 4;
        p.scale = 1.0f / 16.0f;
        ops::AttnPrefillFlags f;
        f.attn_kg_var = ops::kAttnKgVarAuto;
        const auto pl = ops::chooseAttnPrefill(p, f, kv);
        bool kg2 = pl.steps.size() == 3;
        for (const auto& st : pl.steps) kg2 = kg2 && nameOf(st.f) == "attn_kg2" && st.block.x == 128;
        CHECK(kg2 && kgShapesMatch(pl));
    }
    // (3) the 8060S set: auto changes nothing (no attn_kg there; numerics also keeps 0 on gfx1151)
    {
        static const kn::KernelTable k8 = attnTable(true, true, true, false);
        ops::AttnPrefillProblem p;
        p.n = 512;
        p.head_dim = 256;
        p.n_head = 16;
        p.n_head_kv = 2;
        p.scale = 1.0f / 16.0f;
        for (const bool q8 : {false, true}) {
            p.kv_q8 = q8;
            p.dq_rows = q8 ? 8192 : 0;
            p.dq_ptab = p.dq_st_ml = q8;
            ops::AttnPrefillFlags f0, fa;
            fa.attn_kg_var = ops::kAttnKgVarAuto;
            const auto a = ops::chooseAttnPrefill(p, f0, k8), b = ops::chooseAttnPrefill(p, fa, k8);
            bool same = a.steps.size() == b.steps.size() && !a.steps.empty();
            for (std::size_t i = 0; same && i < a.steps.size(); ++i) same = a.steps[i].f == b.steps[i].f && a.steps[i].grid.y == b.steps[i].grid.y;
            CHECK(same);
        }
    }
    testAttnFastQ4Kgk();  // v0252-fasta
}

// v0252-fasta (PLAN V1-57 candidate A, DISCOVERY v0252-r97fast 2 A): the R9700 fast-mode prefill (q4 KV: no attn_kg*_q4,
// FC-1c) with every key in ONE dequantized range takes attn_kgk<np> (auto, KgPath::dqr1) instead of attn_kgs<np>, at the
// product shapes and lengths (Ornith 16 / 2 heads NP 4, Swift 24 / 4 NP 6; 8k / 32k / 128k): the same kv_dq_rows_r step,
// then the same launches (count, grid, block, head0) with attn_kgk, on the copy, not ranged. Bit-identical by M0-A
// (kt attnkgl Q4P 24/24). More than one range keeps attn_kgs; WHIRL_ATTN_KGVAR=0 keeps attn_kgs; the 8060S set, f16
// (precise) and Q8P (balance) are unchanged. Fails before the v0252-fasta selector (the stub table has no dqr1 rows).
void testAttnFastQ4Kgk() {
    static const kn::KernelTable kv = kgTable(true);  // static: see testAttnKgVariant
    const auto kq4p = std::make_unique<kn::KernelTable>(kv);
    kn::KernelTable& kq4 = *kq4p;  // the gfx1201 q4 set: no attn_kg*_q4, no kv_dq_rows (kv_dq_rows_r_q4 only)
    kq4.attn_kg6 = kq4.attn_kg4 = kq4.attn_kg2 = nullptr;
    kq4.kv_dq_rows = nullptr;
    static const kn::KernelTable k8 = attnTable(true, true, true, false);  // the gfx1151 set
    struct Shape {
        std::uint32_t heads, kv, np;
    };
    std::size_t cells = 0;
    for (const Shape sh : {Shape{16, 2, 4}, Shape{24, 4, 6}}) {
        const std::string s = std::to_string(sh.np);
        for (const std::uint32_t L : {8192u, 32768u, 131072u})
            for (const std::uint32_t n : {2048u, 4096u, 8192u}) {
                ops::AttnPrefillProblem p;
                p.n = n;
                p.pos0 = L - n;  // the last prefill chunk of an L-token prompt (the longest key range)
                p.head_dim = 256;
                p.n_head = sh.heads;
                p.n_head_kv = sh.kv;
                p.scale = 1.0f / 16.0f;
                p.kv_q8 = true;
                p.dq_ptab = p.dq_st_ml = true;
                ops::AttnPrefillFlags fa, f0, f1, f3;
                fa.attn_kg_var = ops::kAttnKgVarAuto;
                f1.attn_kg_var = 1;
                f3.attn_kg_var = 3;
                // (1) one range (scratch >= L): auto and var 3 = the var-0 plan with attn_kgk in place of attn_kgs
                p.dq_rows = 131072;
                const auto base = ops::chooseAttnPrefill(p, f0, kq4);
                bool base_kgs = base.steps.size() >= 2 && base.steps[0].op == ops::AttnStep::Op::dq_rows_r && base.steps[0].r0 == 0 &&
                                base.steps[0].nrows == L;
                for (std::size_t i = 1; base_kgs && i < base.steps.size(); ++i)
                    base_kgs = base.steps[i].op == ops::AttnStep::Op::attn && nameOf(base.steps[i].f) == "attn_kgs" + s && base.steps[i].ranged;
                CHECK(base_kgs);
                for (const ops::AttnPrefillFlags& f : {fa, f3}) {
                    const auto pl = ops::chooseAttnPrefill(p, f, kq4);
                    bool ok = pl.steps.size() == base.steps.size() && kgShapesMatch(pl);
                    for (std::size_t i = 0; ok && i < pl.steps.size(); ++i) {
                        const auto &x = pl.steps[i], &y = base.steps[i];
                        ok = x.op == y.op && x.grid.x == y.grid.x && x.grid.y == y.grid.y && x.block.x == y.block.x && x.head0 == y.head0;
                        if (ok && i == 0) ok = x.f == y.f && x.r0 == y.r0 && x.nrows == y.nrows && x.row_el == y.row_el;
                        if (ok && i > 0) ok = nameOf(x.f) == "attn_kgk" + s && x.from_copy && !x.ranged && x.r0 == 0 && x.r1 == 0;
                    }
                    if (!ok) std::printf("  fastq4 NP %s L %u n %u: not the attn_kgk plan\n", s.c_str(), L, n);
                    CHECK(ok);
                }
                // var 1 (attn_kgv not proven on q4 rows): attn_kgs as var 0
                {
                    const auto pl = ops::chooseAttnPrefill(p, f1, kq4);
                    bool same = pl.steps.size() == base.steps.size();
                    for (std::size_t i = 0; same && i < pl.steps.size(); ++i) same = pl.steps[i].f == base.steps[i].f && pl.steps[i].ranged == base.steps[i].ranged;
                    CHECK(same);
                }
                // (2) more than one range (scratch < L): attn_kgs with the softmax state carried, every var
                if (L > 65536) {
                    p.dq_rows = 65536;
                    for (const ops::AttnPrefillFlags& f : {fa, f0, f1, f3}) {
                        const auto pl = ops::chooseAttnPrefill(p, f, kq4);
                        std::size_t kgs = 0, other = 0, dq = 0;
                        for (const auto& st : pl.steps) {
                            dq += st.op == ops::AttnStep::Op::dq_rows_r;
                            if (st.op == ops::AttnStep::Op::attn) (nameOf(st.f) == "attn_kgs" + s && st.ranged ? kgs : other) += 1;
                        }
                        CHECK(dq == 2 && kgs >= 2 && other == 0 && kgShapesMatch(pl));
                    }
                    p.dq_rows = 131072;
                }
                // (3) the gfx1151 set: auto == 0 (KG-3, attn_pwf on the copy)
                {
                    const auto a = ops::chooseAttnPrefill(p, f0, k8), b = ops::chooseAttnPrefill(p, fa, k8);
                    bool same = a.steps.size() == b.steps.size() && !a.steps.empty();
                    for (std::size_t i = 0; same && i < a.steps.size(); ++i) same = a.steps[i].f == b.steps[i].f && a.steps[i].grid.y == b.steps[i].grid.y;
                    CHECK(same && nameOf(b.steps.back().f) != "attn_kgk" + s);
                }
                cells += 1;
            }
        // (4) precise (f16 KV) and balance (q8h in Q8P) on gfx1201 with auto: C1's kernels, unchanged
        ops::AttnPrefillProblem p;
        p.n = 512;
        p.pos0 = 1536;
        p.head_dim = 256;
        p.n_head = sh.heads;
        p.n_head_kv = sh.kv;
        p.scale = 1.0f / 16.0f;
        ops::AttnPrefillFlags fa;
        fa.attn_kg_var = ops::kAttnKgVarAuto;
        CHECK(nameOf(ops::chooseAttnPrefill(p, fa, kv).steps.back().f) == (sh.np == 4 ? "attn_kgk4" : "attn_kgk6"));  // v0252-c1np6: Swift f16 now kgk6
        p.kv_q8 = true;
        p.dq_rows = 8192;
        p.dq_ptab = p.dq_st_ml = true;
        CHECK(nameOf(ops::chooseAttnPrefill(p, fa, kv).steps.back().f) == (sh.np == 4 ? "attn_kgf4" : "attn_kgk6"));
    }
    std::printf("  fast q4 one-range prefill: %zu shape x length x chunk cells\n", cells);
}

// the attention-prep / attention-core rules (arch025-n8b1: the block of testGdnChoice that calls only src/ops/attention.cpp,
// in a function of its own so ops.attention's isolation program runs it; whirl-ops-tests calls it from testGdnChoice, in
// the same place as before)
void testAttnPrepRules() {
    const kn::KernelTable k = gdnTable(true);
    // attention prep: f16 KV with WHIRL_NO_FUSE / head_dim > 512 / no attn_prep -> split; q8 KV keeps the fused prep
    {
        CHECK(ops::chooseAttnPrep(false, 256, false, true, k) == ops::AttnPrep::split);
        CHECK(ops::chooseAttnPrep(false, 256, true, true, k) == ops::AttnPrep::fused);
        CHECK(ops::chooseAttnPrep(false, 576, true, false, k) == ops::AttnPrep::split);
        CHECK(ops::chooseAttnPrep(true, 576, false, true, k) == ops::AttnPrep::mrope);
        kn::KernelTable k0 = k;
        k0.attn_prep = nullptr;
        CHECK(ops::chooseAttnPrep(false, 256, true, false, k0) == ops::AttnPrep::split);
        CHECK(ops::chooseAttnPrepSegmented(false) == ops::AttnPrep::fused && ops::chooseAttnPrepSegmented(true) == ops::AttnPrep::mrope);
        CHECK(ops::chooseAttnCore(16, 16, 256, false) == ops::AttnCore::decode && ops::chooseAttnCore(17, 16, 256, false) == ops::AttnCore::prefill);
        CHECK(ops::chooseAttnCore(17, 16, 128, false) == ops::AttnCore::naive && ops::chooseAttnCore(512, 16, 256, true) == ops::AttnCore::naive);
        CHECK(ops::chooseAttnCore(32, 32, 256, true) == ops::AttnCore::decode);
    }
}

// v0252-blasst (PLAN V1-70, T0 step 1): ops::attnSkipLimit = (log2 a - log2 keys) / (scale * log2 e), the BLASST limit in the
// gfx1201 kgk units (unscaled q.k); -inf (never skip) for a <= 0 / NaN, no keys, and lambda = a / keys >= 1. Values from the
// plan (research note 3.1: a 1000 at 131072 keys -> -78.01). Fails before the port (test-first stub returns 0).
void testAttnSkipLimit() {
    const float s = 1.0f / 16.0f;
    auto near = [](float x, double want, double tol) { return std::isfinite(x) && std::fabs(static_cast<double>(x) - want) <= tol; };
    CHECK(near(ops::attnSkipLimit(1000.0f, 131072, s), -78.01, 0.01));
    CHECK(near(ops::attnSkipLimit(1000.0f, 65536, s), -66.92, 0.01));
    CHECK(near(ops::attnSkipLimit(300.0f, 131072, s), -97.28, 0.01));
    CHECK(ops::attnSkipLimit(0.0f, 131072, s) == -INFINITY);
    CHECK(ops::attnSkipLimit(-1.0f, 131072, s) == -INFINITY);
    CHECK(ops::attnSkipLimit(std::nanf(""), 131072, s) == -INFINITY);
    CHECK(ops::attnSkipLimit(1000.0f, 0, s) == -INFINITY);
    CHECK(ops::attnSkipLimit(20000.0f, 16384, s) == -INFINITY);  // lambda >= 1 would skip tiles above the running max
    CHECK(ops::attnSkipLimit(16384.0f, 16384, s) == -INFINITY);
    // the gfx1151 kernels compare scaled scores with exp: their limit is lim * scale = ln a - ln keys (PLAN T4 units note)
    CHECK(near(ops::attnSkipLimit(1000.0f, 131072, s) * s, -4.876, 0.001));
    // a larger a skips more (a higher limit), more keys a lower one
    CHECK(ops::attnSkipLimit(3000.0f, 131072, s) > ops::attnSkipLimit(1000.0f, 131072, s));
    CHECK(ops::attnSkipLimit(1000.0f, 131072, s) < ops::attnSkipLimit(1000.0f, 32768, s));
}

// v0252-blasst (PLAN V1-70, T2): the selector's BLASST rows. On the one-range FC-1c path (KgPath::dqr1, the gfx1201 q4 fast
// prefill) a launch with keys (pos0 + n) >= skip_min_keys takes attn_kgk<np>_sk with skip_lim = attnSkipLimit(skip_a, keys,
// scale) when skip_a > 0, or attn_kgk<np>_pr with probe_lim[t] = attnSkipLimit(probe_a[t], keys, scale) when a probe value
// is set (the probe wins); grid / block / head ranges / the dequant step stay those of the attn_kgk plan. Everything else is
// the plan without the fields: below the length gate (and across it chunk by chunk), skip_a 0, no _sk / _pr in the object,
// more than one key range (attn_kgs: Stage A has no kgs skip), a >= keys, the f16 / Q8P paths (precise / balance) and the
// gfx1151 set. Ornith 16 / 2 (NP 4) and Swift 24 / 4 (NP 6). Fails before the port (the selector ignores the fields).
void testAttnSkipSelect() {
    static const kn::KernelTable kv = kgTable(true);
    const auto kp = std::make_unique<kn::KernelTable>(kv);
    kn::KernelTable& kq4 = *kp;  // the gfx1201 q4 set (as testAttnFastQ4Kgk) with the v0252-blasst variants
    kq4.attn_kg6 = kq4.attn_kg4 = kq4.attn_kg2 = nullptr;
    kq4.kv_dq_rows = nullptr;
    kq4.attn_kgk6_sk = tag("attn_kgk6_sk");
    kq4.attn_kgk4_sk = tag("attn_kgk4_sk");
    kq4.attn_kgk6_pr = tag("attn_kgk6_pr");
    kq4.attn_kgk4_pr = tag("attn_kgk4_pr");
    const auto knp = std::make_unique<kn::KernelTable>(kq4);  // an older gfx1201 object: attn_kgk, no _sk / _pr
    knp->attn_kgk6_sk = knp->attn_kgk4_sk = knp->attn_kgk6_pr = knp->attn_kgk4_pr = nullptr;
    static const kn::KernelTable k8 = attnTable(true, true, true, false);  // the gfx1151 set
    const float scale = 1.0f / 16.0f;
    struct Shape {
        std::uint32_t heads, kv, np;
    };
    // the attn launches of a plan: (kernel name, grid.x, grid.y, block.x, head0)
    auto launches = [](const ops::AttnPrefillPlan& pl) {
        std::vector<std::string> v;
        for (const auto& s : pl.steps)
            v.push_back(std::format("{}:{}:{}:{}:{}:{}", s.op == ops::AttnStep::Op::attn ? nameOf(s.f) : std::string("dq"), s.grid.x, s.grid.y, s.block.x,
                                    s.head0, s.nrows));
        return v;
    };
    auto renamed = [](std::vector<std::string> v, const std::string& from, const std::string& to) {
        for (auto& x : v)
            if (x.rfind(from + ":", 0) == 0) x = to + x.substr(from.size());
        return v;
    };
    for (const Shape sh : {Shape{16, 2, 4}, Shape{24, 4, 6}}) {
        const std::string s = std::to_string(sh.np), kgk = "attn_kgk" + s;
        auto prob = [&](std::uint32_t L, std::uint32_t n) {
            ops::AttnPrefillProblem p;
            p.n = n;
            p.pos0 = L - n;
            p.head_dim = 256;
            p.n_head = sh.heads;
            p.n_head_kv = sh.kv;
            p.scale = scale;
            p.kv_q8 = true;
            p.dq_ptab = p.dq_st_ml = true;
            p.dq_rows = 131072;
            return p;
        };
        ops::AttnPrefillFlags f0;
        f0.attn_kg_var = ops::kAttnKgVarAuto;
        ops::AttnPrefillFlags fs = f0;
        fs.skip_a = 1000.0f;
        ops::AttnPrefillFlags fp = f0;
        fp.probe_a = {10.0f, 100.0f, 1000.0f, 10000.0f};
        for (const std::uint32_t n : {4096u, 301u}) {
            // (1) 128k keys, skip_a 1000: attn_kgk<np>_sk in place of attn_kgk<np>, the limit of this launch's keys
            const ops::AttnPrefillProblem p = prob(131072, n);
            const auto base = ops::chooseAttnPrefill(p, f0, kq4), sk = ops::chooseAttnPrefill(p, fs, kq4);
            CHECK(!base.steps.empty() && nameOf(base.steps.back().f) == kgk);
            CHECK(launches(sk) == renamed(launches(base), kgk, kgk + "_sk"));
            bool lim_ok = true;
            for (const auto& st : sk.steps)
                if (st.op == ops::AttnStep::Op::attn)
                    lim_ok = lim_ok && st.skip && !st.probe && st.skip_lim == ops::attnSkipLimit(1000.0f, 131072, scale) && std::isfinite(st.skip_lim);
            CHECK(lim_ok);
            // (6) the probe: attn_kgk<np>_pr with the four limits, the rest as (1)'s base plan; it wins over skip_a
            for (const ops::AttnPrefillFlags& f : {fp, [&] {
                                                       ops::AttnPrefillFlags b = fp;
                                                       b.skip_a = 1000.0f;
                                                       return b;
                                                   }()}) {
                const auto pr = ops::chooseAttnPrefill(p, f, kq4);
                CHECK(launches(pr) == renamed(launches(base), kgk, kgk + "_pr"));
                bool pr_ok = true;
                for (const auto& st : pr.steps)
                    if (st.op == ops::AttnStep::Op::attn) {
                        pr_ok = pr_ok && st.probe && !st.skip;
                        for (std::size_t t = 0; t < 4; ++t) pr_ok = pr_ok && st.probe_lim[t] == ops::attnSkipLimit(fp.probe_a[t], 131072, scale);
                    }
                CHECK(pr_ok);
            }
            // (2) below the length gate (8k keys): the attn_kgk plan, no skip / probe fields
            {
                const ops::AttnPrefillProblem q = prob(8192, std::min(n, 8192u));
                const auto a = ops::chooseAttnPrefill(q, fs, kq4), b = ops::chooseAttnPrefill(q, f0, kq4), c = ops::chooseAttnPrefill(q, fp, kq4);
                CHECK(launches(a) == launches(b) && launches(c) == launches(b) && nameOf(a.steps.back().f) == kgk && !a.steps.back().skip && !c.steps.back().probe);
                // WHIRL_ATTN_SKIP_MIN=0: every length
                ops::AttnPrefillFlags fz = fs;
                fz.skip_min_keys = 0;
                CHECK(nameOf(ops::chooseAttnPrefill(q, fz, kq4).steps.back().f) == kgk + "_sk");
            }
            // (3) skip_a 0 and no probe: unchanged; (4) an object without _sk / _pr: attn_kgk<np>
            CHECK(launches(ops::chooseAttnPrefill(p, f0, kq4)) == launches(base));
            CHECK(launches(ops::chooseAttnPrefill(p, fs, *knp)) == launches(base) && launches(ops::chooseAttnPrefill(p, fp, *knp)) == launches(base));
            // a >= keys (lambda >= 1): no _sk
            {
                ops::AttnPrefillFlags fa = fs;
                fa.skip_a = 200000.0f;
                const auto pa = ops::chooseAttnPrefill(p, fa, kq4);
                CHECK(launches(pa) == launches(base) && !pa.steps.back().skip);
            }
            // (5) more than one key range (scratch 65536 < 131072 keys): attn_kgs<np>, never _sk / _pr
            {
                ops::AttnPrefillProblem q = p;
                q.dq_rows = 65536;
                const auto a = ops::chooseAttnPrefill(q, fs, kq4), b = ops::chooseAttnPrefill(q, f0, kq4), c = ops::chooseAttnPrefill(q, fp, kq4);
                CHECK(launches(a) == launches(b) && launches(c) == launches(b) && nameOf(a.steps.back().f) == "attn_kgs" + s);
            }
        }
        // (2b) PLAN review focus 3: the length gate per chunk of one prompt (4096-row chunks): the chunk ending at 12288 keys
        // keeps attn_kgk<np>, the one ending at 16384 (= skip_min_keys) takes _sk
        CHECK(nameOf(ops::chooseAttnPrefill(prob(12288, 4096), fs, kq4).steps.back().f) == kgk);
        CHECK(nameOf(ops::chooseAttnPrefill(prob(16384, 4096), fs, kq4).steps.back().f) == kgk + "_sk");
        CHECK(nameOf(ops::chooseAttnPrefill(prob(16383, 4095), fs, kq4).steps.back().f) == kgk);
        // (7) precise (f16 KV) and balance (Q8P) on gfx1201: the fields never reach them (kgk on the cache / the Q8P rows)
        {
            ops::AttnPrefillProblem p;
            p.n = 4096;
            p.pos0 = 131072 - 4096;
            p.head_dim = 256;
            p.n_head = sh.heads;
            p.n_head_kv = sh.kv;
            p.scale = scale;
            const auto a = ops::chooseAttnPrefill(p, fs, kq4), b = ops::chooseAttnPrefill(p, f0, kq4);
            CHECK(launches(a) == launches(b));
            ops::AttnPrefillFlags fsp = fp;
            const auto c = ops::chooseAttnPrefill(p, fsp, kv), d = ops::chooseAttnPrefill(p, f0, kv);
            CHECK(launches(c) == launches(d));
            p.kv_q8 = true;
            p.dq_rows = 8192;
            p.dq_ptab = p.dq_st_ml = true;
            p.pos0 = 2048;
            CHECK(launches(ops::chooseAttnPrefill(p, fs, kv)) == launches(ops::chooseAttnPrefill(p, f0, kv)));
        }
        // (8) the gfx1151 set: unchanged (KG-3 attn_pwf; no kgk there)
        {
            const ops::AttnPrefillProblem p = prob(131072, 4096);
            CHECK(launches(ops::chooseAttnPrefill(p, fs, k8)) == launches(ops::chooseAttnPrefill(p, f0, k8)) &&
                  launches(ops::chooseAttnPrefill(p, fp, k8)) == launches(ops::chooseAttnPrefill(p, f0, k8)));
        }
    }
}

}  // namespace
