// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.attention/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/device_profile.h"
#include "whirl/ops/attention.h"
#include "../test_check.h"
#include "ops_fakes.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>

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

}  // namespace
