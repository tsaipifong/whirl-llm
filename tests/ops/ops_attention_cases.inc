// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.attention/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/device_profile.h"
#include "whirl/ops/attention.h"
#include "../test_check.h"
#include "ops_fakes.h"
#include "../../kernels/gfx1151/attn_ws4_index.h"  // v0252-e1grp: ws4::block_range (the reference group key)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace {

namespace ops = whirl::ops;

// ---- attention (K-3)
// Stack note (v0252-c1asan, v0252-opsasan): a kn::KernelTable is about 140 KB, and under MSVC ASan (gate step cpu_asan)
// every local AND every temporary of that type gets its own stack slot in the 1 MB stack. A case that needs more than
// two or three tables keeps them off the stack: static const for a table built once (the return value is built in
// static storage), std::make_unique for a copy the case changes; never a table passed or bound as a temporary.
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

// v0252-ws4r97b (PLAN V1-87 phase 2): the gfx1201 4-wave pipelined decode / verify attention attn_ws4r1 / attn_ws4r2
// (kernels/attn_ws4r.hip; kt ws4r: bitwise == attn_wsplit1 / attn_wsplit2 for any grouping, R9700 q7 9525u 13:32) takes
// the place of attn_wsplit1 / attn_wsplit2 AFTER the wsplit choice: same kind, groups, grid, 128 threads, split count and
// combine - only the kernel changes. Which variant per KV format is a device-profile row (kt verdict q8=KEEP-b f16=KEEP-plain:
// gfx1201 q8 / q8h -> 2 "b", f16 -> 1 "plain"; gfx1151 0 = off, its object has none); WHIRL_ATTN_WS4R (flag attn_ws4r:
// auto / 0 / 1 / 2) overrides both rows. Fails before phase 2 (the stub profile rows are 0 and chooseAttnDecode never
// swaps).
// v0252-opsasan: the base table by reference - a by-value table parameter is one more 140 KB temporary in the caller's frame
kn::KernelTable ws4rTable(const kn::KernelTable& base, bool q8) {
    kn::KernelTable k = base;
    const std::string s = q8 ? "_q8" : "";
    k.attn_wsplit1 = tag("attn_wsplit1" + s);
    k.attn_wsplit2 = tag("attn_wsplit2" + s);
    k.attn_ws4r1[0] = tag("attn_ws4r1" + s);
    k.attn_ws4r1[1] = tag("attn_ws4r1b" + s);
    k.attn_ws4r2[0] = tag("attn_ws4r2" + s);
    k.attn_ws4r2[1] = tag("attn_ws4r2b" + s);
    return k;
}
// everything but the kernel is the plan without the swap
bool samePlanButKernel(const ops::AttnDecodePlan& a, const ops::AttnDecodePlan& b) {
    bool same = a.kind == b.kind && a.grid.x == b.grid.x && a.grid.y == b.grid.y && a.grid.z == b.grid.z && a.block.x == b.block.x &&
                a.block.y == b.block.y && a.n_split == b.n_split && a.combine_q8 == b.combine_q8 && a.combine_grid.x == b.combine_grid.x &&
                a.combine_grid.y == b.combine_grid.y && a.combine_block.x == b.combine_block.x;
    for (std::uint32_t g = 0; g < a.grid.z && g < static_cast<std::uint32_t>(kn::kMaxVerifyRows); ++g)
        same = same && a.groups.first[g] == b.groups.first[g] && a.groups.count[g] == b.groups.count[g];
    return same;
}

void testAttnDecodeWs4r() {
    namespace dv = whirl::device;
    const auto& p97 = dv::k_profile_gfx1201;
    const auto& p8s = dv::k_profile_gfx1151;
    // v0252-opsasan: this case's tables are static (built once, in static storage) or on the heap, see the stack note at
    // attnTable. With 7 table locals and 4 table temporaries its frame was 1.9 MiB under ASan: whirl-ops-tests overflowed
    // the 1 MB stack on entry (cpu_asan, _alloca_probe). kb / kfb: the gfx1201-like sets before the ws4r kernels (dq4 and
    // f16 KV); kq / kf: the same with the ws4r kernels (q8 names / plain names).
    static const kn::KernelTable kb = attnTable(false, false, true, true), kfb = attnTable(false, false, false, true);
    static const kn::KernelTable kq = ws4rTable(kb, true), kf = ws4rTable(kfb, false);
    // (0) the profile rows (kt verdict): gfx1201 q8 / q8h -> b, f16 -> plain; gfx1151 off
    CHECK(p97.attn_ws4r_q8 == 2 && p97.attn_ws4r_f16 == 1);
    CHECK(p8s.attn_ws4r_q8 == 0 && p8s.attn_ws4r_f16 == 0);
    CHECK(ops::kAttnWs4rAuto == 255);
    // (1) attnWs4rFor: auto -> the profile row; 1 plain, 2 b, anything else none; CG from `wide`
    {
        const kn::KernelTable& k = kq;
        CHECK(nameOf(ops::attnWs4rFor(ops::kAttnWs4rAuto, 2, false, k)) == "attn_ws4r1b_q8");
        CHECK(nameOf(ops::attnWs4rFor(ops::kAttnWs4rAuto, 1, true, k)) == "attn_ws4r2_q8");
        CHECK(nameOf(ops::attnWs4rFor(ops::kAttnWs4rAuto, 0, false, k)) == "null");
        CHECK(nameOf(ops::attnWs4rFor(1, 0, false, k)) == "attn_ws4r1_q8" && nameOf(ops::attnWs4rFor(2, 0, true, k)) == "attn_ws4r2b_q8");
        CHECK(nameOf(ops::attnWs4rFor(0, 2, false, k)) == "null" && nameOf(ops::attnWs4rFor(0, 1, true, k)) == "null");
        for (const std::uint32_t v : {3u, 4u, 254u, 256u}) CHECK(nameOf(ops::attnWs4rFor(v, 2, false, k)) == "null");
        // a table without the kernels (the gfx1151 object, a gfx1201 object before phase 1): none
        const kn::KernelTable& k0 = kb;
        CHECK(nameOf(ops::attnWs4rFor(2, 2, false, k0)) == "null" && nameOf(ops::attnWs4rFor(1, 1, true, k0)) == "null");
    }
    ops::AttnDecodeProblem p;
    p.n = 1;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;  // Swift: GQA group 6
    p.max_ctx = 131072;
    p.o_on_gemv = true;
    const std::int32_t pos[4] = {70000, 70001, 70002, 70003}, base[4] = {0, 0, 0, 0};
    auto rows = [&](ops::AttnDecodeProblem q, std::uint32_t n) {
        q.n = n;
        q.rows_known = n > 1;
        q.row_pos = n > 1 ? std::span<const std::int32_t>(pos, n) : std::span<const std::int32_t>{};
        q.row_base = n > 1 ? std::span<const std::int32_t>(base, n) : std::span<const std::int32_t>{};
        return q;
    };
    ops::AttnDecodeFlags fa;  // default: auto
    ops::AttnDecodeFlags f0 = fa, f1 = fa, f2 = fa;
    f0.attn_ws4r = 0;
    f1.attn_ws4r = 1;
    f2.attn_ws4r = 2;
    CHECK(fa.attn_ws4r == ops::kAttnWs4rAuto);
    // (2) Swift balance (q8h KV) on the R9700: decode row -> attn_ws4r1b_q8, a 4-row MTP verify (attn_wsplit2 group of 4) ->
    // attn_ws4r2b_q8; the rest of the plan is the attn_wsplit plan (WHIRL_ATTN_WS4R=0)
    {
        ops::AttnDecodeProblem q = rows(p, 1);
        q.kv_q8 = true;
        const auto a = ops::chooseAttnDecode(q, fa, kq, kq.caps, p97), o = ops::chooseAttnDecode(q, f0, kq, kq.caps, p97);
        CHECK(nameOf(a.f) == "attn_ws4r1b_q8" && nameOf(o.f) == "attn_wsplit1_q8");
        CHECK(a.kind == ops::AttnDecodeKind::wsplit && a.block.x == 128 && a.grid.x == 4 && a.grid.y == 64 && a.grid.z == 1);
        CHECK(samePlanButKernel(a, o));
        q = rows(q, 4);
        const auto av = ops::chooseAttnDecode(q, fa, kq, kq.caps, p97), ov = ops::chooseAttnDecode(q, f0, kq, kq.caps, p97);
        CHECK(nameOf(av.f) == "attn_ws4r2b_q8" && nameOf(ov.f) == "attn_wsplit2_q8" && av.groups.count[0] == 4 && av.grid.z == 1);
        CHECK(samePlanButKernel(av, ov));
        // the override picks the variant on either row: 1 plain, 2 b, 0 off
        CHECK(nameOf(ops::chooseAttnDecode(q, f1, kq, kq.caps, p97).f) == "attn_ws4r2_q8");
        CHECK(nameOf(ops::chooseAttnDecode(q, f2, kq, kq.caps, p97).f) == "attn_ws4r2b_q8");
        // without attn_wide the 4 rows are groups of 2 on attn_wsplit1 -> attn_ws4r1b_q8, same groups
        ops::AttnDecodeFlags fn = fa, fn0 = f0;
        fn.attn_wide = fn0.attn_wide = false;
        const auto an = ops::chooseAttnDecode(q, fn, kq, kq.caps, p97), on = ops::chooseAttnDecode(q, fn0, kq, kq.caps, p97);
        CHECK(nameOf(an.f) == "attn_ws4r1b_q8" && nameOf(on.f) == "attn_wsplit1_q8" && an.grid.z == 2 && samePlanButKernel(an, on));
        // a draft window (MTP at 64k+) keeps the same plan too
        ops::AttnDecodeProblem qw = q;
        qw.win = (16384 / 64) | ((65536 / 1024) << 16);
        CHECK(samePlanButKernel(ops::chooseAttnDecode(qw, fa, kq, kq.caps, p97), ops::chooseAttnDecode(qw, f0, kq, kq.caps, p97)));
        // a profile row decides, not the kernel set: a row with 0 keeps attn_wsplit under the auto flag
        dv::Profile poff = p97;
        poff.attn_ws4r_q8 = 0;
        CHECK(nameOf(ops::chooseAttnDecode(q, fa, kq, kq.caps, poff).f) == "attn_wsplit2_q8");
        // dbg_flags bit 2 (attn_split) is untouched
        ops::AttnDecodeFlags fd = fa;
        fd.dbg_flags = 2;
        CHECK(nameOf(ops::chooseAttnDecode(q, fd, kq, kq.caps, p97).f) == "attn_split");
    }
    // (3) Swift / Ornith precise (f16 KV) on the R9700: plain
    for (const std::uint32_t nkv : {4u, 2u}) {
        ops::AttnDecodeProblem q = rows(p, 1);
        q.n_head = nkv == 4 ? 24 : 16;
        q.n_head_kv = nkv;
        const auto a = ops::chooseAttnDecode(q, fa, kf, kf.caps, p97), o = ops::chooseAttnDecode(q, f0, kf, kf.caps, p97);
        CHECK(nameOf(a.f) == "attn_ws4r1" && nameOf(o.f) == "attn_wsplit1" && samePlanButKernel(a, o));
        q = rows(q, 4);
        const auto av = ops::chooseAttnDecode(q, fa, kf, kf.caps, p97), ov = ops::chooseAttnDecode(q, f0, kf, kf.caps, p97);
        CHECK(nameOf(av.f) == "attn_ws4r2" && nameOf(ov.f) == "attn_wsplit2" && samePlanButKernel(av, ov));
        CHECK(nameOf(ops::chooseAttnDecode(q, f2, kf, kf.caps, p97).f) == "attn_ws4r2b");
    }
    // (4) q4 KV (fast): attn_dq4 stays; with attn_dq4 off the q4 attn_wsplit is never swapped (no q4 ws4r)
    {
        ops::AttnDecodeProblem q = rows(p, 4);
        q.kv_q4 = true;
        ops::AttnDecodeFlags fq = fa;
        fq.fd_splits = 256;
        CHECK(nameOf(ops::chooseAttnDecode(q, fq, kq, kq.caps, p97).f) == "attn_dq4");
        fq.attn_dq4 = false;
        const std::string n4 = nameOf(ops::chooseAttnDecode(q, fq, kq, kq.caps, p97).f);
        CHECK(n4 == "attn_wsplit2_q8" || n4 == "attn_wsplit1_q8");
        fq.attn_ws4r = 2;
        CHECK(nameOf(ops::chooseAttnDecode(q, fq, kq, kq.caps, p97).f).rfind("attn_wsplit", 0) == 0);
    }
    // (5) the 8060S: its plan is the same with or without ws4r kernels in the table, under every flag value (row 0; the
    // gfx1151 object has none) - attn_ws4 / attn_ws4g / attn_wsplit1 as before
    for (const bool q8 : {false, true}) {
        static const kn::KernelTable k8 = attnTable(true, true, false, false);
        static const kn::KernelTable k8rs[2] = {ws4rTable(k8, false), ws4rTable(k8, true)};  // by q8
        const kn::KernelTable& k8r = k8rs[q8 ? 1 : 0];
        const auto k8bp = std::make_unique<kn::KernelTable>(k8);  // heap: changed below
        kn::KernelTable& k8b = *k8bp;
        k8b.attn_wsplit1 = k8r.attn_wsplit1;
        k8b.attn_wsplit2 = k8r.attn_wsplit2;
        for (const std::uint32_t n : {1u, 4u}) {
            ops::AttnDecodeProblem q = rows(p, n);
            q.max_ctx = 32768;
            q.kv_q8 = q8;
            for (const bool ws4 : {true, false}) {
                ops::AttnDecodeFlags g = fa;
                g.attn_ws4 = ws4;
                const auto want = ops::chooseAttnDecode(q, g, k8b, k8b.caps, p8s);
                const auto got = ops::chooseAttnDecode(q, g, k8r, k8r.caps, p8s);
                CHECK(nameOf(got.f) == nameOf(want.f) && samePlanButKernel(got, want));
                CHECK(nameOf(got.f).rfind("attn_ws4r", 0) != 0);
            }
        }
    }
}

// v0252-l2split (PLAN V1-88): the f16 / q8 / q8h decode / verify split count by machine fill (ops::lsplitCount). The numerics
// flag attn_lsplit (precise 0, balance / fast auto = device::Profile::attn_lsplit_blocks; WHIRL_ATTN_LSPLIT) picks the block
// target; split = min(today's, max(2, blocks / n_head_kv)) on the attn_wsplit / attn_ws4r branch only. Pins: (0) the profile
// rows and the rule; (1) flag 0 (precise) = the plan of today, every field; (2) Swift balance q8h decode / 4-row verify on
// the R9700 -> 16 splits, same kernel / groups / combine as today but the split; (3) Ornith f16 -> 32; (4) the split is the
// same for every batch shape (row invariance: one row, a verify group, two sequences); (5) q4 / attn_dq4, attn_split and the
// 8060S attn_ws4 path unchanged under every flag value. Fails to build before v0252-l2split (no lsplitCount / attn_lsplit /
// attn_lsplit_blocks).
bool samePlan(const ops::AttnDecodePlan& a, const ops::AttnDecodePlan& b) { return nameOf(a.f) == nameOf(b.f) && samePlanButKernel(a, b); }

void testAttnDecodeLsplit() {
    namespace dv = whirl::device;
    const auto& p97 = dv::k_profile_gfx1201;
    const auto& p8s = dv::k_profile_gfx1151;
    const std::uint32_t A = ops::kAttnLsplitAuto;
    // (0) profile rows and the rule
    CHECK(p97.attn_lsplit_blocks == 64 && p8s.attn_lsplit_blocks == 0);
    CHECK(A == 0xFFFFFFFFu);
    CHECK(ops::lsplitCount(64, 4, A, 64) == 16 && ops::lsplitCount(64, 2, A, 64) == 32);  // Swift / Ornith on the R9700
    CHECK(ops::lsplitCount(64, 4, 0, 64) == 64 && ops::lsplitCount(64, 4, A, 0) == 64);   // flag 0 / row 0: today's
    CHECK(ops::lsplitCount(64, 4, 128, 0) == 32 && ops::lsplitCount(64, 4, 192, 64) == 48 && ops::lsplitCount(64, 4, 4096, 64) == 64);
    CHECK(ops::lsplitCount(64, 4, 8, 64) == 2 && ops::lsplitCount(64, 4, 1, 64) == 2);    // >= 2 (the draft-window branch)
    CHECK(ops::lsplitCount(8, 4, A, 64) == 8 && ops::lsplitCount(1, 4, A, 64) == 1);      // never above today's
    CHECK(ops::lsplitCount(64, 0, A, 64) == 64);
    ops::AttnDecodeProblem p;
    p.n = 1;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;  // Swift: GQA group 6
    p.max_ctx = 131072;
    p.o_on_gemv = true;
    const std::int32_t pos[4] = {70000, 70001, 70002, 70003}, base[4] = {0, 0, 0, 0}, base2[4] = {0, 0, 4096, 4096};
    const std::int32_t pos2[4] = {70000, 70001, 1500, 1501};
    auto rows = [&](ops::AttnDecodeProblem q, std::uint32_t n, const std::int32_t* rp = nullptr, const std::int32_t* rb = nullptr) {
        q.n = n;
        q.rows_known = n > 1;
        q.row_pos = n > 1 ? std::span<const std::int32_t>(rp ? rp : pos, n) : std::span<const std::int32_t>{};
        q.row_base = n > 1 ? std::span<const std::int32_t>(rb ? rb : base, n) : std::span<const std::int32_t>{};
        return q;
    };
    ops::AttnDecodeFlags f0;  // ops default = precise's flag: today's split
    CHECK(f0.attn_lsplit == 0);
    ops::AttnDecodeFlags fa = f0;
    fa.attn_lsplit = A;
    // v0252-opsasan: static, see the stack note at attnTable (this frame was 0.96 MiB of the 1 MB stack under ASan)
    static const kn::KernelTable kb = attnTable(false, false, true, true), kfb = attnTable(false, false, false, true);
    static const kn::KernelTable kq = ws4rTable(kb, true), kf = ws4rTable(kfb, false);
    // (1) + (2) Swift q8h on the R9700
    {
        ops::AttnDecodeProblem q = rows(p, 1);
        q.kv_q8 = true;
        const auto t = ops::chooseAttnDecode(q, f0, kq, kq.caps, p97), b = ops::chooseAttnDecode(q, fa, kq, kq.caps, p97);
        CHECK(t.n_split == 64 && t.grid.y == 64 && nameOf(t.f) == "attn_ws4r1b_q8");  // precise / A arm: today's grid
        CHECK(b.n_split == 16 && b.grid.y == 16 && b.grid.x == 4 && b.grid.z == 1 && nameOf(b.f) == "attn_ws4r1b_q8");
        CHECK(b.kind == t.kind && b.block.x == t.block.x && b.combine_q8 == t.combine_q8 && b.combine_grid.x == t.combine_grid.x &&
              b.combine_grid.y == t.combine_grid.y);
        ops::AttnDecodePlan bb = b;  // everything but the split is today's plan
        bb.n_split = t.n_split;
        bb.grid.y = t.grid.y;
        CHECK(samePlan(bb, t));
        // a 4-row MTP verify at 70000..70003: one attn_ws4r2b_q8 group of 4, 16 splits; WHIRL_ATTN_LSPLIT=128 -> 32
        q = rows(q, 4);
        const auto tv = ops::chooseAttnDecode(q, f0, kq, kq.caps, p97), bv = ops::chooseAttnDecode(q, fa, kq, kq.caps, p97);
        CHECK(tv.grid.y == 64 && bv.grid.y == 16 && bv.n_split == 16 && nameOf(bv.f) == "attn_ws4r2b_q8" && bv.groups.count[0] == 4 &&
              bv.grid.z == 1 && nameOf(tv.f) == nameOf(bv.f));
        ops::AttnDecodeFlags f128 = f0;
        f128.attn_lsplit = 128;
        CHECK(ops::chooseAttnDecode(q, f128, kq, kq.caps, p97).grid.y == 32);
        // a profile row decides under auto, not the kernel set: row 0 -> today's
        dv::Profile poff = p97;
        poff.attn_lsplit_blocks = 0;
        CHECK(samePlan(ops::chooseAttnDecode(q, fa, kq, kq.caps, poff), tv));
        // the draft window (MTP at 64k+) keeps >= 2 splits and the same rule
        ops::AttnDecodeProblem qw = q;
        qw.win = (16384 / 64) | ((65536 / 1024) << 16);
        CHECK(ops::chooseAttnDecode(qw, fa, kq, kq.caps, p97).grid.y == 16);
        // a small KV (max_ctx 512: today 8 splits) is never raised
        ops::AttnDecodeProblem qs = rows(p, 1);
        qs.kv_q8 = true;
        qs.max_ctx = 512;
        CHECK(ops::chooseAttnDecode(qs, fa, kq, kq.caps, p97).grid.y == 8 && ops::chooseAttnDecode(qs, f0, kq, kq.caps, p97).grid.y == 8);
        // WHIRL_ATTN_WS4R=0 (attn_wsplit kernels): the same split rule
        ops::AttnDecodeFlags fw = fa;
        fw.attn_ws4r = 0;
        const auto w = ops::chooseAttnDecode(q, fw, kq, kq.caps, p97);
        CHECK(nameOf(w.f) == "attn_wsplit2_q8" && w.grid.y == 16);
    }
    // (1) + (3) f16 KV (Swift / Ornith precise and Ornith balance): flag 0 = today, auto = blocks / n_head_kv
    for (const std::uint32_t nkv : {4u, 2u}) {
        for (const std::uint32_t n : {1u, 4u}) {
            ops::AttnDecodeProblem q = rows(p, n);
            q.n_head = nkv == 4 ? 24 : 16;
            q.n_head_kv = nkv;
            const auto t = ops::chooseAttnDecode(q, f0, kf, kf.caps, p97), b = ops::chooseAttnDecode(q, fa, kf, kf.caps, p97);
            CHECK(t.grid.y == 64 && b.grid.y == (nkv == 4 ? 16u : 32u) && b.n_split == b.grid.y && nameOf(b.f) == nameOf(t.f));
            CHECK(b.grid.z == t.grid.z && b.groups.count[0] == t.groups.count[0]);
        }
    }
    // (4) row invariance: one decode row, a 4-row verify group, two sequences of different lengths - one split count, so every
    // row's key partition awPer(pos + 1, split) is a function of its own position only
    {
        ops::AttnDecodeProblem q1 = rows(p, 1), q4 = rows(p, 4), q2 = rows(p, 4, pos2, base2);
        for (ops::AttnDecodeProblem* q : {&q1, &q4, &q2}) q->kv_q8 = true;
        const auto a1 = ops::chooseAttnDecode(q1, fa, kq, kq.caps, p97), a4 = ops::chooseAttnDecode(q4, fa, kq, kq.caps, p97),
                   a2 = ops::chooseAttnDecode(q2, fa, kq, kq.caps, p97);
        CHECK(a1.n_split == 16 && a4.n_split == 16 && a2.n_split == 16 && a2.grid.z == 2);
        CHECK(ops::awPer(70001, a1.n_split) == ops::awPer(70001, a4.n_split) && ops::awPer(1501, a2.n_split) == ops::awPer(1501, a1.n_split));
        // graph-captured plain decode (rows the host does not know): the same split
        ops::AttnDecodeProblem qu = q1;
        qu.rows_known = false;
        CHECK(ops::chooseAttnDecode(qu, fa, kq, kq.caps, p97).n_split == 16);
    }
    // (5) unchanged: q4 KV (attn_dq4 and its dynamic split), dbg attn_split, and the 8060S under every flag value
    for (const std::uint32_t fl : {A, 0u, 64u, 128u}) {
        ops::AttnDecodeFlags g = f0;
        g.attn_lsplit = fl;
        ops::AttnDecodeProblem q = rows(p, 4);
        q.kv_q4 = true;
        ops::AttnDecodeFlags g4 = g, t4 = f0;
        g4.fd_splits = t4.fd_splits = 256;
        CHECK(samePlan(ops::chooseAttnDecode(q, g4, kq, kq.caps, p97), ops::chooseAttnDecode(q, t4, kq, kq.caps, p97)));
        g4.attn_dq4 = t4.attn_dq4 = false;  // the q4 attn_wsplit fallback is never re-split either
        CHECK(samePlan(ops::chooseAttnDecode(q, g4, kq, kq.caps, p97), ops::chooseAttnDecode(q, t4, kq, kq.caps, p97)));
        ops::AttnDecodeProblem qd = rows(p, 4);
        qd.kv_q8 = true;
        ops::AttnDecodeFlags gd = g, td = f0;
        gd.dbg_flags = td.dbg_flags = 2;
        const auto d = ops::chooseAttnDecode(qd, gd, kq, kq.caps, p97);
        CHECK(nameOf(d.f) == "attn_split" && d.grid.y == 64 && samePlan(d, ops::chooseAttnDecode(qd, td, kq, kq.caps, p97)));
        for (const bool q8 : {false, true}) {
            static const kn::KernelTable k8 = attnTable(true, true, false, false);
            for (const std::uint32_t n : {1u, 4u}) {
                ops::AttnDecodeProblem q8p = rows(p, n);
                q8p.max_ctx = 32768;
                q8p.kv_q8 = q8;
                ops::AttnDecodeFlags gw = g, tw = f0;
                CHECK(samePlan(ops::chooseAttnDecode(q8p, gw, k8, k8.caps, p8s), ops::chooseAttnDecode(q8p, tw, k8, k8.caps, p8s)));
                // WHIRL_ATTN_WS4=0 on the 8060S (attn_wsplit1): the auto row is 0 -> today's split
                if (fl == A) {
                    gw.attn_ws4 = tw.attn_ws4 = false;
                    CHECK(samePlan(ops::chooseAttnDecode(q8p, gw, k8, k8.caps, p8s), ops::chooseAttnDecode(q8p, tw, k8, k8.caps, p8s)));
                }
            }
        }
    }
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

// v0252-8sfp (PLAN V1-144): the FAST gfx1151 prefill attention with int4 Q.K^T. With attn_i4qk_on and both kernels in the
// table, the KG-3 copy path launches attn_pwf_i4 (one range) / attn_pws_i4 (key ranges) in place of attn_pwf / attn_pws with
// the same steps, grids, blocks, head ranges and key ranges; flag off, or either kernel missing (the gfx1201 object): the
// f16 kernels exactly as before. The f16-cache paths (attn_kg / attn_kx / attn_prefill_wmma) never take it.
void testAttnI4qk() {
    ops::AttnPrefillProblem p;
    p.n = 512;  // the 8060S-like set of testAttnPrefill (Swift: 24 heads, 4 KV heads, head_dim 256)
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.scale = 1.0f / 16.0f;
    p.kv_q8 = true;
    p.dq_rows = 8192;
    p.dq_ptab = true;
    p.dq_st_ml = true;
    ops::AttnPrefillFlags off, on;
    on.attn_i4qk_on = true;
    kn::KernelTable k = attnTable(true, true, true, false);
    k.attn_pwf_i4 = tag("attn_pwf_i4");
    k.attn_pws_i4 = tag("attn_pws_i4");
    auto same = [](const ops::AttnPrefillPlan& a, const ops::AttnPrefillPlan& b, const char* fa, const char* fb) {
        bool ok = a.steps.size() == b.steps.size();
        for (std::size_t i = 0; ok && i < a.steps.size(); ++i) {
            const ops::AttnStep &x = a.steps[i], &y = b.steps[i];
            ok = x.op == y.op && x.grid.x == y.grid.x && x.grid.y == y.grid.y && x.grid.z == y.grid.z && x.block.x == y.block.x && x.head0 == y.head0 &&
                 x.from_copy == y.from_copy && x.ranged == y.ranged && x.r0 == y.r0 && x.r1 == y.r1;
            if (ok && x.op == ops::AttnStep::Op::attn) ok = nameOf(x.f) == fa && nameOf(y.f) == fb;
        }
        return ok;
    };
    // one dequantized range: kv_dq_rows_r + attn_pwf_i4 (same grid as attn_pwf)
    CHECK(same(ops::chooseAttnPrefill(p, off, k), ops::chooseAttnPrefill(p, on, k), "attn_pwf", "attn_pwf_i4"));
    // keys past the scratch: the attn_pws_i4 series, the same ranges
    p.pos0 = 20000;
    CHECK(same(ops::chooseAttnPrefill(p, off, k), ops::chooseAttnPrefill(p, on, k), "attn_pws", "attn_pws_i4"));
    // a long prefill with head ranges (4096 queries at 32k, one range of 40960 rows)
    p.pos0 = 32768;
    p.n = 4096;
    p.dq_rows = 40960;
    CHECK(same(ops::chooseAttnPrefill(p, off, k), ops::chooseAttnPrefill(p, on, k), "attn_pwf", "attn_pwf_i4"));
    // either kernel missing: the f16 kernels
    for (int miss = 0; miss < 2; ++miss) {
        kn::KernelTable km = k;
        (miss == 0 ? km.attn_pwf_i4 : km.attn_pws_i4) = nullptr;
        CHECK(same(ops::chooseAttnPrefill(p, off, km), ops::chooseAttnPrefill(p, on, km), "attn_pwf", "attn_pwf"));
    }
    // the f16 cache read in place (no copy scratch) and the attn_kg table: untouched by the flag
    p.kv_q8 = false;
    p.dq_ptab = false;
    p.dq_rows = 0;
    p.pos0 = 0;
    p.n = 512;
    for (const kn::KernelTable& kk : {k, attnTable(false, false, false, true)}) {
        const auto a = ops::chooseAttnPrefill(p, off, kk), b = ops::chooseAttnPrefill(p, on, kk);
        CHECK(a.steps.size() == b.steps.size());
        for (std::size_t i = 0; i < a.steps.size() && i < b.steps.size(); ++i) CHECK(a.steps[i].f == b.steps[i].f && nameOf(b.steps[i].f).find("_i4") == std::string::npos);
    }
}

// v0252-8sattn3 (PLAN V1-148): the second int4 form (attn_pwf_i4p / attn_pws_i4p = attn_prefill_wmma_i4p / _s_i4p: int4 Q
// codes in LDS + K / V register prefetch, bit-identical to the int4 kernels). With attn_i4qk_on AND attn_i4pf_on and both
// i4p kernels in the table, the KG-3 copy path launches them in place of attn_pwf_i4 / attn_pws_i4 with the same steps,
// grids, blocks, head ranges and key ranges. attn_i4pf_on alone (int4 off) never picks anything; either i4p kernel missing:
// the int4 kernels exactly as before (the old kernel stays the reference arm); the f16-cache paths never take it.
void testAttnI4pf() {
    ops::AttnPrefillProblem p;
    p.n = 512;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.scale = 1.0f / 16.0f;
    p.kv_q8 = true;
    p.dq_rows = 8192;
    p.dq_ptab = true;
    p.dq_st_ml = true;
    ops::AttnPrefillFlags i4, pf, pf_only;
    i4.attn_i4qk_on = true;
    pf.attn_i4qk_on = true;
    pf.attn_i4pf_on = true;
    pf_only.attn_i4pf_on = true;
    kn::KernelTable k = attnTable(true, true, true, false);
    k.attn_pwf_i4 = tag("attn_pwf_i4");
    k.attn_pws_i4 = tag("attn_pws_i4");
    k.attn_pwf_i4p = tag("attn_pwf_i4p");
    k.attn_pws_i4p = tag("attn_pws_i4p");
    auto same = [](const ops::AttnPrefillPlan& a, const ops::AttnPrefillPlan& b, const char* fa, const char* fb) {
        bool ok = a.steps.size() == b.steps.size();
        for (std::size_t i = 0; ok && i < a.steps.size(); ++i) {
            const ops::AttnStep &x = a.steps[i], &y = b.steps[i];
            ok = x.op == y.op && x.grid.x == y.grid.x && x.grid.y == y.grid.y && x.grid.z == y.grid.z && x.block.x == y.block.x && x.head0 == y.head0 &&
                 x.from_copy == y.from_copy && x.ranged == y.ranged && x.r0 == y.r0 && x.r1 == y.r1;
            if (ok && x.op == ops::AttnStep::Op::attn) ok = nameOf(x.f) == fa && nameOf(y.f) == fb;
        }
        return ok;
    };
    CHECK(same(ops::chooseAttnPrefill(p, i4, k), ops::chooseAttnPrefill(p, pf, k), "attn_pwf_i4", "attn_pwf_i4p"));
    CHECK(same(ops::chooseAttnPrefill(p, i4, k), ops::chooseAttnPrefill(p, {}, k), "attn_pwf_i4", "attn_pwf"));  // sanity: f16 off
    CHECK(same(ops::chooseAttnPrefill(p, {}, k), ops::chooseAttnPrefill(p, pf_only, k), "attn_pwf", "attn_pwf"));
    p.pos0 = 20000;  // keys past the scratch: the series
    CHECK(same(ops::chooseAttnPrefill(p, i4, k), ops::chooseAttnPrefill(p, pf, k), "attn_pws_i4", "attn_pws_i4p"));
    CHECK(same(ops::chooseAttnPrefill(p, {}, k), ops::chooseAttnPrefill(p, pf_only, k), "attn_pws", "attn_pws"));
    p.pos0 = 32768;  // head ranges, one range
    p.n = 4096;
    p.dq_rows = 40960;
    CHECK(same(ops::chooseAttnPrefill(p, i4, k), ops::chooseAttnPrefill(p, pf, k), "attn_pwf_i4", "attn_pwf_i4p"));
    // either i4p kernel missing (an object without them): the int4 kernels
    for (int miss = 0; miss < 2; ++miss) {
        kn::KernelTable km = k;
        (miss == 0 ? km.attn_pwf_i4p : km.attn_pws_i4p) = nullptr;
        CHECK(same(ops::chooseAttnPrefill(p, i4, km), ops::chooseAttnPrefill(p, pf, km), "attn_pwf_i4", "attn_pwf_i4"));
    }
    // the f16 cache read in place and the attn_kg table: untouched
    p.kv_q8 = false;
    p.dq_ptab = false;
    p.dq_rows = 0;
    p.pos0 = 0;
    p.n = 512;
    for (const kn::KernelTable& kk : {k, attnTable(false, false, false, true)}) {
        const auto a = ops::chooseAttnPrefill(p, {}, kk), b = ops::chooseAttnPrefill(p, pf, kk);
        CHECK(a.steps.size() == b.steps.size());
        for (std::size_t i = 0; i < a.steps.size() && i < b.steps.size(); ++i) CHECK(a.steps[i].f == b.steps[i].f && nameOf(b.steps[i].f).find("_i4") == std::string::npos);
    }
}

// v0252-attnlat (PLAN V1-148): attn_pfx_f16r (attn_prefill_wmma with all 13 K / V loads of a key step in one clause; same
// arguments, grid, block and bits; 32-bit byte offsets). (1) attnPfxPick: off without the flag or the kernel; on only for a
// known K / V array below 4 GiB; else guarded. (2) chooseAttnPrefill with attn_pfx_on: the one-range KG-3 copy takes it in
// place of attn_pwf (guard size dq_rows x row bytes) and the f16 cache read in place takes it in place of attn_prefill_wmma
// (guard size AttnPrefillProblem::kv_bytes), every other step / grid / block / head range unchanged; ranged series (attn_pws),
// int4 (attn_i4qk), attn_sx, attn_kx, attn_kg and q8 KV read in place never take it; the guard keeps attn_prefill_wmma and
// sets pfx_guarded. (3) flag off, or no kernel in the object (gfx1201): exactly the plan without the flag, pfx_guarded false.
// Fails to build before the change (no attnPfxPick / attn_pfx_on / AttnKernels::attn_pfx / kv_bytes / pfx_guarded).
void testAttnPfx() {
    static_assert(ops::attnPfxPick(false, true, 1u << 20) == ops::PfxPick::off);
    static_assert(ops::attnPfxPick(true, false, 1u << 20) == ops::PfxPick::off);
    static_assert(ops::attnPfxPick(true, true, 1u << 20) == ops::PfxPick::on);
    static_assert(ops::attnPfxPick(true, true, ops::kAttnPfxMaxBytes - 1) == ops::PfxPick::on);
    static_assert(ops::attnPfxPick(true, true, ops::kAttnPfxMaxBytes) == ops::PfxPick::guarded);
    static_assert(ops::attnPfxPick(true, true, 0) == ops::PfxPick::guarded);
    static_assert(ops::kAttnPfxMaxBytes == 4294967296ull);
    auto same = [](const ops::AttnPrefillPlan& a, const ops::AttnPrefillPlan& b, const char* fa, const char* fb) {
        bool ok = a.steps.size() == b.steps.size() && !a.steps.empty();
        for (std::size_t i = 0; ok && i < a.steps.size(); ++i) {
            const ops::AttnStep &x = a.steps[i], &y = b.steps[i];
            ok = x.op == y.op && x.grid.x == y.grid.x && x.grid.y == y.grid.y && x.grid.z == y.grid.z && x.block.x == y.block.x && x.head0 == y.head0 &&
                 x.from_copy == y.from_copy && x.ranged == y.ranged && x.r0 == y.r0 && x.r1 == y.r1;
            if (ok && x.op == ops::AttnStep::Op::attn) ok = nameOf(x.f) == fa && nameOf(y.f) == fb;
            if (ok && x.op != ops::AttnStep::Op::attn) ok = x.f == y.f;
        }
        return ok;
    };
    // (2) the KG-3 copy (Swift 24 / 4 heads, q8h keys dequantized into the scratch), the 8060S-like table
    ops::AttnPrefillProblem p;
    p.n = 512;
    p.head_dim = 256;
    p.n_head = 24;
    p.n_head_kv = 4;
    p.scale = 1.0f / 16.0f;
    p.kv_q8 = true;
    p.dq_rows = 8192;
    p.dq_ptab = true;
    p.dq_st_ml = true;
    ops::AttnPrefillFlags pf;
    pf.attn_pfx_on = true;
    kn::KernelTable k = attnTable(true, true, true, false);
    k.attn_pfx = tag("attn_pfx_f16r");
    {
        const auto a = ops::chooseAttnPrefill(p, {}, k), b = ops::chooseAttnPrefill(p, pf, k);
        CHECK(same(a, b, "attn_pwf", "attn_pfx_f16r") && !a.pfx_guarded && !b.pfx_guarded);
    }
    p.pos0 = 129166 - 2048;  // the Swift 129k cell: n 2048 at L 129166, one range
    p.n = 2048;
    p.dq_rows = 131072;
    CHECK(same(ops::chooseAttnPrefill(p, {}, k), ops::chooseAttnPrefill(p, pf, k), "attn_pwf", "attn_pfx_f16r"));
    p.pos0 = 200000;  // head ranges (n 4096 x 204096 keys > 4096 x 128k: several launches), still one range
    p.n = 4096;
    p.dq_rows = 204800;
    CHECK(same(ops::chooseAttnPrefill(p, {}, k), ops::chooseAttnPrefill(p, pf, k), "attn_pwf", "attn_pfx_f16r"));
    CHECK(ops::chooseAttnPrefill(p, pf, k).steps.size() >= 3);  // dq + >= 2 head-range launches
    p.pos0 = 32768;
    // the guard on the copy: dq_rows x 4 x 256 x 2 bytes; 2097152 rows = exactly 4 GiB -> attn_pwf kept, said
    p.dq_rows = 2097152;
    {
        const auto a = ops::chooseAttnPrefill(p, {}, k), b = ops::chooseAttnPrefill(p, pf, k);
        CHECK(same(a, b, "attn_pwf", "attn_pwf") && b.pfx_guarded && !a.pfx_guarded);
    }
    p.dq_rows = 2097151;  // one row less: below 4 GiB
    CHECK(same(ops::chooseAttnPrefill(p, {}, k), ops::chooseAttnPrefill(p, pf, k), "attn_pwf", "attn_pfx_f16r"));
    // the ranged series (keys past the scratch): attn_pws, never the one-launch kernel
    p.pos0 = 20000;
    p.n = 512;
    p.dq_rows = 8192;
    {
        const auto a = ops::chooseAttnPrefill(p, {}, k), b = ops::chooseAttnPrefill(p, pf, k);
        CHECK(same(a, b, "attn_pws", "attn_pws") && !b.pfx_guarded);
    }
    p.pos0 = 0;
    // int4 (fast) on the copy keeps the int4 kernels
    {
        kn::KernelTable ki = k;
        ki.attn_pwf_i4 = tag("attn_pwf_i4");
        ki.attn_pws_i4 = tag("attn_pws_i4");
        ops::AttnPrefillFlags i4 = pf, i4o;
        i4.attn_i4qk_on = true;
        i4o.attn_i4qk_on = true;
        CHECK(same(ops::chooseAttnPrefill(p, i4o, ki), ops::chooseAttnPrefill(p, i4, ki), "attn_pwf_i4", "attn_pwf_i4"));
    }
    // attn_sx (balance, where the plan turns it on) keeps attn_sx
    {
        kn::KernelTable ks = k;
        ks.attn_sx2 = tag("attn_sx2");
        ops::AttnPrefillFlags sx = pf, sxo;
        sx.attn_sx_on = true;
        sxo.attn_sx_on = true;
        CHECK(same(ops::chooseAttnPrefill(p, sxo, ks), ops::chooseAttnPrefill(p, sx, ks), "attn_sx2", "attn_sx2"));
    }
    // (3) no kernel in the object: the plan without the flag
    {
        kn::KernelTable kz = k;
        kz.attn_pfx = nullptr;
        const auto b = ops::chooseAttnPrefill(p, pf, kz);
        CHECK(same(ops::chooseAttnPrefill(p, {}, kz), b, "attn_pwf", "attn_pwf") && !b.pfx_guarded);
    }
    // (2) the f16 cache read in place (precise: f16 KV, no scratch), a table without attn_kx
    p.kv_q8 = false;
    p.dq_ptab = false;
    p.dq_st_ml = false;
    p.dq_rows = 0;
    kn::KernelTable kc = k;
    kc.attn_kx = nullptr;
    p.kv_bytes = 512ull << 20;  // one K array of 512 MiB
    {
        const auto a = ops::chooseAttnPrefill(p, {}, kc), b = ops::chooseAttnPrefill(p, pf, kc);
        CHECK(same(a, b, "attn_prefill_wmma", "attn_pfx_f16r") && !b.pfx_guarded);
    }
    for (const std::uint64_t bytes : {0ull, ops::kAttnPfxMaxBytes, ops::kAttnPfxMaxBytes * 3}) {  // unknown / 4 GiB / 12 GiB
        p.kv_bytes = bytes;
        const auto b = ops::chooseAttnPrefill(p, pf, kc);
        CHECK(same(ops::chooseAttnPrefill(p, {}, kc), b, "attn_prefill_wmma", "attn_prefill_wmma") && b.pfx_guarded);
    }
    p.kv_bytes = 512ull << 20;
    // attn_kx in the object (the f16 cache takes attn_kx): unchanged, nothing guarded
    {
        const auto b = ops::chooseAttnPrefill(p, pf, k);
        CHECK(same(ops::chooseAttnPrefill(p, {}, k), b, "attn_kx", "attn_kx") && !b.pfx_guarded);
    }
    // attn_kg (the gfx1201 table, no attn_pfx there anyway): unchanged
    {
        kn::KernelTable kg = attnTable(false, false, false, true);
        kg.attn_pfx = tag("attn_pfx_f16r");
        const auto a = ops::chooseAttnPrefill(p, {}, kg), b = ops::chooseAttnPrefill(p, pf, kg);
        CHECK(a.steps.size() == b.steps.size() && !b.pfx_guarded);
        for (std::size_t i = 0; i < a.steps.size() && i < b.steps.size(); ++i) CHECK(a.steps[i].f == b.steps[i].f);
    }
    // q8 KV read in place (no scratch: the q8 attn_prefill_wmma variant of the table): never the f16 one-clause kernel
    p.kv_q8 = true;
    {
        const auto b = ops::chooseAttnPrefill(p, pf, kc);
        CHECK(same(ops::chooseAttnPrefill(p, {}, kc), b, "attn_prefill_wmma", "attn_prefill_wmma") && !b.pfx_guarded);
    }
}

// v0252-strixc2 (PLAN V1-09): the balance gfx1151 prefill attention attn_sx<NP> (strix-llama.cpp fattn-rdna35, not bitwise).
// (1) attnSxFor: NP = the first of 8 / 4 / 2 dividing the GQA ratio (fattn-rdna35.cu:863-870), only when it also divides
// the heads of the launch, head_dim 256, and the code object has that NP's kernel; exhaustive (never another NP's kernel).
// (2) chooseAttnPrefill with attn_sx_on on the 8060S-like table: the f16 cache takes attn_sx<NP> in place of attn_kx
// (one launch per head range, grid ceil(n / (64 / NP)) x (heads / NP) flattened in x, block 256); q8 / q8h keys in ONE
// dequantized range take it on the copy in place of attn_pwf; multi-range series (attn_pws), WHIRL_ATTN_SEG and a head
// range NP does not divide keep the old kernels. (3) attn_sx_on off, or no attn_sx in the code object (gfx1201): every
// plan is exactly the plan without the flag. Fails before the port (attnSxFor / the flag do not exist).
void testAttnSx() {
    static const kn::KernelTable k0 = attnTable(true, true, true, false);
    static const kn::KernelTable ks = [] {
        kn::KernelTable k = attnTable(true, true, true, false);
        k.attn_sx2 = tag("attn_sx2");
        k.attn_sx4 = tag("attn_sx4");
        k.attn_sx8 = tag("attn_sx8");
        return k;
    }();
    // (1) the selector
    struct Pick {
        std::uint32_t grp, hp, hd;
        const char* want;
        std::uint32_t np;
    };
    const Pick picks[] = {{6, 24, 256, "attn_sx2", 2}, {8, 16, 256, "attn_sx8", 8}, {4, 16, 256, "attn_sx4", 4}, {2, 4, 256, "attn_sx2", 2},
                          {12, 24, 256, "attn_sx4", 4}, {6, 12, 256, "attn_sx2", 2}, {8, 12, 256, "null", 0},     {3, 24, 256, "null", 0},
                          {1, 24, 256, "null", 0},      {6, 24, 128, "null", 0},     {8, 16, 512, "null", 0},     {0, 16, 256, "null", 0}};
    for (const Pick& c : picks) {
        const ops::AttnSxPick p = ops::attnSxFor(ks, c.grp, c.hp, c.hd);
        CHECK(nameOf(p.f) == c.want && p.np == c.np);
        const ops::AttnSxPick z = ops::attnSxFor(k0, c.grp, c.hp, c.hd);  // not in the code object: none
        CHECK(z.f == nullptr && z.np == 0);
    }
    // (2) the plans
    struct Shape {
        std::uint32_t heads, kv, np;
    };
    auto prob = [](const Shape& sh, std::uint32_t n, std::uint32_t pos0) {
        ops::AttnPrefillProblem p;
        p.n = n;
        p.pos0 = pos0;
        p.head_dim = 256;
        p.n_head = sh.heads;
        p.n_head_kv = sh.kv;
        p.scale = 1.0f / 16.0f;
        return p;
    };
    // v0252-strixc2fix: kernels compared by NAME, not by pointer - k0 and ks are two tables whose tag() pointers differ for
    // the same kernel (ops_fakes.h: every tag() call is a new string), so a pointer compare across tables is always false
    // (GREEN 10:38, :444). A mismatch prints both plans.
    auto dump = [](const char* w, const ops::AttnPrefillPlan& a) {
        std::printf("  %s: %zu steps\n", w, a.steps.size());
        for (const auto& x : a.steps)
            std::printf("    op %d %s grid %u x %u block %u head0 %u copy %d ranged %d r %u..%u nrows %u\n", static_cast<int>(x.op), nameOf(x.f).c_str(),
                        x.grid.x, x.grid.y, x.block.x, x.head0, x.from_copy ? 1 : 0, x.ranged ? 1 : 0, x.r0, x.r1, x.nrows);
    };
    auto same = [&dump](const ops::AttnPrefillPlan& a, const ops::AttnPrefillPlan& b) {
        bool eq = a.steps.size() == b.steps.size();
        for (std::size_t i = 0; eq && i < a.steps.size(); ++i) {
            const auto& x = a.steps[i];
            const auto& y = b.steps[i];
            if (x.op != y.op || nameOf(x.f) != nameOf(y.f) || x.grid.x != y.grid.x || x.grid.y != y.grid.y || x.block.x != y.block.x ||
                x.head0 != y.head0 || x.from_copy != y.from_copy || x.ranged != y.ranged || x.r0 != y.r0 || x.r1 != y.r1 || x.nrows != y.nrows)
                eq = false;
        }
        if (!eq) {
            dump("plan a", a);
            dump("plan b", b);
        }
        return eq;
    };
    ops::AttnPrefillFlags off, on;
    on.attn_sx_on = true;
    const Shape shapes[] = {{24, 4, 2}, {16, 2, 8}, {4, 2, 2}};
    for (const Shape& sh : shapes) {
        const std::string s = "attn_sx" + std::to_string(sh.np);
        const std::uint32_t cols = 64 / sh.np;
        // f16 cache, 512 queries at 1536: one launch of attn_sx<NP> in place of attn_kx
        ops::AttnPrefillProblem p = prob(sh, 512, 1536);
        auto pl = ops::chooseAttnPrefill(p, on, ks);
        CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == s && !pl.steps[0].from_copy && !pl.steps[0].ranged && pl.steps[0].head0 == 0 &&
              pl.steps[0].grid.x == (512 + cols - 1) / cols * (sh.heads / sh.np) && pl.steps[0].grid.y == 1 && pl.steps[0].block.x == 256);
        CHECK(nameOf(ops::chooseAttnPrefill(p, off, ks).steps.back().f) == "attn_kx");
        CHECK(same(ops::chooseAttnPrefill(p, off, ks), ops::chooseAttnPrefill(p, off, k0)));
        CHECK(same(ops::chooseAttnPrefill(p, on, k0), ops::chooseAttnPrefill(p, off, k0)));
        // an odd query count: the last tile is partial (ceil)
        p.n = 37;
        pl = ops::chooseAttnPrefill(p, on, ks);
        CHECK(pl.steps.size() == 1 && pl.steps[0].grid.x == (37 + cols - 1) / cols * (sh.heads / sh.np));
        // q8 / q8h keys in one dequantized range (2048 rows <= 8192): kv_dq_rows_r, then attn_sx<NP> on the copy (in place of attn_pwf)
        p = prob(sh, 512, 1536);
        p.kv_q8 = true;
        p.dq_rows = 8192;
        p.dq_ptab = true;
        p.dq_st_ml = true;
        pl = ops::chooseAttnPrefill(p, on, ks);
        CHECK(pl.steps.size() == 2 && pl.steps[0].op == ops::AttnStep::Op::dq_rows_r && pl.steps[0].nrows == 2048 && nameOf(pl.steps[1].f) == s &&
              pl.steps[1].from_copy && !pl.steps[1].ranged && pl.steps[1].r0 == 0 && pl.steps[1].r1 == 2048 && pl.steps[1].block.x == 256);
        CHECK(nameOf(ops::chooseAttnPrefill(p, off, ks).steps.back().f) == "attn_pwf");
        CHECK(same(ops::chooseAttnPrefill(p, on, k0), ops::chooseAttnPrefill(p, off, k0)));
        // keys past the scratch: the attn_pws series with the state carried, no attn_sx (no carried state in strix's kernel)
        p.pos0 = 20000;
        pl = ops::chooseAttnPrefill(p, on, ks);
        CHECK(same(pl, ops::chooseAttnPrefill(p, off, ks)));
        for (const auto& st : pl.steps) CHECK(nameOf(st.f).rfind("attn_sx", 0) != 0);
        // WHIRL_ATTN_SEG on f16 KV past the range: the attn_pws series stays. v0252-strixc2fix: the series exists only with
        // the softmax state buffers (dq_st_ml) and without attn_kx (the gfx1151 object has no attn_kx; selector
        // `... && !kx && rows > attn_seg`); the 10:38 case had neither, so no series was planned at all (seg0 = one attn_kx
        // launch, seg = attn_sx in its place: both one-range plans, :471 compared two different one-range kernels)
        p = prob(sh, 512, 8192);
        p.dq_st_ml = true;
        ops::AttnPrefillFlags seg = on, seg0 = off;
        seg.attn_seg = 4096;
        seg0.attn_seg = 4096;
        seg.attn_kx_on = false;
        seg0.attn_kx_on = false;
        pl = ops::chooseAttnPrefill(p, seg0, ks);
        std::size_t pws = 0;
        for (const auto& st : pl.steps) pws += nameOf(st.f) == "attn_pws" && st.ranged && !st.from_copy;
        CHECK(pws >= 3 && pws == pl.steps.size());  // 8704 keys in ranges of 4096: 3 ranges (x head parts)
        CHECK(same(ops::chooseAttnPrefill(p, seg, ks), pl));
        // with attn_kx in the table (no real code object has attn_kx AND attn_sx) the selector never walked ranges: one
        // attn_kx launch, and attn_sx takes its place like on any one-range f16 plan
        seg.attn_kx_on = seg0.attn_kx_on = true;
        CHECK(nameOf(ops::chooseAttnPrefill(p, seg0, ks).steps.back().f) == "attn_kx");
        pl = ops::chooseAttnPrefill(p, seg, ks);
        CHECK(pl.steps.size() == 1 && nameOf(pl.steps[0].f) == s && !pl.steps[0].ranged);
        // head_dim 128: never
        p = prob(sh, 512, 1536);
        p.head_dim = 128;
        CHECK(same(ops::chooseAttnPrefill(p, on, ks), ops::chooseAttnPrefill(p, off, ks)));
    }
    // long prefill (Windows TDR budget: 4096 x 128k pairs per launch): two head ranges, each its own attn_sx launch
    {
        const Shape sw{24, 4, 2}, orn{16, 2, 8}, g8{24, 3, 8};
        auto pl = ops::chooseAttnPrefill(prob(sw, 4096, 131072), on, ks);
        CHECK(pl.steps.size() == 2 && nameOf(pl.steps[0].f) == "attn_sx2" && pl.steps[0].head0 == 0 && pl.steps[1].head0 == 12 &&
              pl.steps[0].grid.x == 128 * 6 && pl.steps[1].grid.x == 128 * 6);
        pl = ops::chooseAttnPrefill(prob(orn, 4096, 131072), on, ks);
        CHECK(pl.steps.size() == 2 && nameOf(pl.steps[1].f) == "attn_sx8" && pl.steps[1].head0 == 8 && pl.steps[0].grid.x == 512 * 1);
        // GQA 8 with 12-head ranges (24 / 3): NP 8 does not divide the range -> exactly the plan without the flag
        CHECK(same(ops::chooseAttnPrefill(prob(g8, 4096, 131072), on, ks), ops::chooseAttnPrefill(prob(g8, 4096, 131072), off, ks)));
        CHECK(nameOf(ops::chooseAttnPrefill(prob(g8, 512, 1536), on, ks).steps[0].f) == "attn_sx8");
    }
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
    // v0252-kvlong2 (O1, PLAN V1-137): past 131072 keys (a 256k prompt's later chunks, here 200000 keys) the plan is ONE
    // dequant range + attn_kgk and no attn_kgs when the copy holds every key (dq_rows >= rows: the context-sized copy of
    // qwen35::dqCopyPlan); a copy smaller than the keys (a card that cannot afford it) keeps the key ranges + attn_kgs.
    for (const Shape sh : {Shape{16, 2, 4}, Shape{24, 4, 6}}) {
        const std::string s = std::to_string(sh.np);
        ops::AttnPrefillProblem p;
        p.n = 4096;
        p.pos0 = 200000 - 4096;
        p.head_dim = 256;
        p.n_head = sh.heads;
        p.n_head_kv = sh.kv;
        p.scale = 1.0f / 16.0f;
        p.kv_q8 = true;
        p.dq_ptab = p.dq_st_ml = true;
        ops::AttnPrefillFlags fa;
        fa.attn_kg_var = ops::kAttnKgVarAuto;
        for (const std::uint32_t dq : {200192u, 262144u}) {
            p.dq_rows = dq;
            const auto pl = ops::chooseAttnPrefill(p, fa, kq4);
            std::size_t dqs = 0, kgk = 0, other = 0;
            bool one = !pl.steps.empty() && pl.steps[0].op == ops::AttnStep::Op::dq_rows_r && pl.steps[0].r0 == 0 && pl.steps[0].nrows == 200000;
            for (const auto& st : pl.steps) {
                dqs += st.op == ops::AttnStep::Op::dq_rows_r;
                if (st.op == ops::AttnStep::Op::attn) (nameOf(st.f) == "attn_kgk" + s && st.from_copy && !st.ranged ? kgk : other) += 1;
            }
            if (!(one && dqs == 1 && kgk >= 1 && other == 0)) std::printf("  kvlong2 NP %s dq_rows %u: not one range + attn_kgk\n", s.c_str(), dq);
            CHECK(one && dqs == 1 && kgk >= 1 && other == 0 && kgShapesMatch(pl));
        }
        p.dq_rows = 131072;  // the FFN-sized copy (before O1, or no VRAM for the grown one): two ranges, attn_kgs
        const auto pl = ops::chooseAttnPrefill(p, fa, kq4);
        std::size_t dqs = 0, kgs = 0, other = 0;
        for (const auto& st : pl.steps) {
            dqs += st.op == ops::AttnStep::Op::dq_rows_r;
            if (st.op == ops::AttnStep::Op::attn) (nameOf(st.f) == "attn_kgs" + s && st.ranged ? kgs : other) += 1;
        }
        CHECK(dqs == 2 && kgs >= 2 && other == 0 && kgShapesMatch(pl));
    }
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

// v0252-8sq4 (PLAN V1-140 / V1-109, F8; fast = the only mode with q4 KV). Two opt-in switches on the attn_dq4 path:
//  - dq4_order (WHIRL_DQ4_ORDER, profile attn_dq4_row_order): attn_dq4o = attn_dq4 with the gfx1201 block order (the verify
//    row fastest), so the MTP verify rows of one (head, split) read the same q4 K / V tiles back to back (L2) instead of
//    one DRAM pass per row. Same arguments, grid, block and partials (the CPU model below: the order is a bijection).
//  - dq4_min_keys (WHIRL_DQ4_MINKEYS, profile dq4_min_keys): the split count from the longest row and a target of keys per
//    split (ops::dq4MinKeysSplits): power of two >= ceil(n_max / min_keys), in [8, today's]. 0 = today's rule.
// Fails to build before the change (no AttnDecodeFlags::dq4_order / dq4_min_keys, AttnKernels::attn_dq4o,
// dq4MinKeysSplits, Profile fields).
// gfx1201 ADQ_GRID (kernels/attn.hip:559), the order attn_dq4o takes on gfx1151: block (bx, by, bz) -> (g, split, z)
std::array<std::int32_t, 3> kDq4oMap(std::int32_t bx, std::int32_t by, std::int32_t bz, std::int32_t gx, std::int32_t gy, std::int32_t gz) {
    const std::int32_t lin = bx + gx * (by + gy * bz);
    const std::int32_t z = lin % gz, gs = lin / gz;
    return {gs % gx, gs / gx, z};
}
void testAttnDq4F8() {
    const auto& p97 = whirl::device::k_profile_gfx1201;
    const auto& p8s = whirl::device::k_profile_gfx1151;
    // profile rows: v0252-8sflip2: the order is the 8060S default (screens 8sq4-order / -long gained beyond their error,
    // ADR-0019); gfx1201 off (no attn_dq4o in its object: attn_dq4 has the order). min keys: opt-in on both rows
    CHECK(!p97.attn_dq4_row_order && p8s.attn_dq4_row_order);
    CHECK(p97.dq4_min_keys == 0 && p8s.dq4_min_keys == 0);
    // ops defaults: off (a caller that does not pass the plan's flags keeps today's kernel and grid)
    {
        const ops::AttnDecodeFlags d;
        CHECK(!d.dq4_order && d.dq4_min_keys == 0);
    }
    // --- the block order is a bijection of the grid (each (g, split, z) exactly once), for Ornith (2 kv heads), Swift (4),
    // 1..4 verify rows and the split counts the rules give (8 .. 256)
    long long maps = 0, map_bad = 0;
    for (const std::int32_t gx : {2, 4})
        for (const std::int32_t gz : {1, 2, 3, 4})
            for (const std::int32_t gy : {8, 16, 32, 64, 100, 128, 256}) {
                std::vector<unsigned char> seen(static_cast<std::size_t>(gx * gy * gz), 0);
                for (std::int32_t bz = 0; bz < gz; ++bz)
                    for (std::int32_t by = 0; by < gy; ++by)
                        for (std::int32_t bx = 0; bx < gx; ++bx) {
                            const auto m = kDq4oMap(bx, by, bz, gx, gy, gz);
                            maps += 1;
                            if (m[0] < 0 || m[0] >= gx || m[1] < 0 || m[1] >= gy || m[2] < 0 || m[2] >= gz) {
                                map_bad += 1;
                                continue;
                            }
                            seen[static_cast<std::size_t>((m[2] * gy + m[1]) * gx + m[0])] += 1;
                        }
                for (const unsigned char c : seen)
                    if (c != 1) map_bad += 1;
                // consecutive hardware block ids (x fastest) of one (g, split) are its verify rows: the L2 sharing
                if (gz > 1) {
                    const auto a = kDq4oMap(0, 0, 0, gx, gy, gz), b = kDq4oMap(1, 0, 0, gx, gy, gz);
                    if (!(a[0] == b[0] && a[1] == b[1] && b[2] == a[2] + 1)) map_bad += 1;
                }
            }
    CHECK(maps > 0 && map_bad == 0);
    // --- the plan: Ornith-like shape (16 / 2 heads, GQA 8), q4 KV, 2 verify rows at 128k on the 8060S kernel set
    ops::AttnDecodeProblem p;
    p.n = 2;
    p.head_dim = 256;
    p.n_head = 16;
    p.n_head_kv = 2;
    p.max_ctx = 262144;
    p.kv_q4 = true;
    p.rows_known = true;
    const std::int32_t base[4] = {0, 0, 0, 0};
    const std::int32_t pos128[2] = {131071, 131072};
    p.row_pos = pos128;
    p.row_base = base;
    ops::AttnDecodeFlags f;
    f.fd_splits = 256;
    // tables off the stack (stack note at attnTable): statics built once, one heap copy the case changes
    static const kn::KernelTable s8 = attnTable(true, true, true, false), s8f = attnTable(true, true, false, false),
                                 s97 = attnTable(false, false, true, true);
    auto k8 = std::make_unique<kn::KernelTable>(s8);
    auto plan = [&](const ops::AttnDecodeFlags& ff, const whirl::device::Profile& pr) { return ops::chooseAttnDecode(p, ff, *k8, k8->caps, pr); };
    // off: today's kernel and dynamic grid (nothing changes without the switch)
    const ops::AttnDecodePlan off = plan(f, p8s);
    CHECK(nameOf(off.f) == "attn_dq4" && off.n_split == ops::dq4Splits(std::span<const std::int32_t>(pos128, 2), 0, 256));
    // dq4_order without attn_dq4o in the code object (gfx1201 has none: its attn_dq4 has the order already): attn_dq4
    {
        ops::AttnDecodeFlags fo = f;
        fo.dq4_order = true;
        CHECK(nameOf(plan(fo, p8s).f) == "attn_dq4");
    }
    k8->attn_dq4o = tag("attn_dq4o");
    {
        CHECK(nameOf(plan(f, p8s).f) == "attn_dq4");  // the kernel alone does not switch
        ops::AttnDecodeFlags fo = f;
        fo.dq4_order = true;
        const ops::AttnDecodePlan on = plan(fo, p8s);
        // same grid, block, groups, split count and combine: only the kernel entry differs
        CHECK(nameOf(on.f) == "attn_dq4o" && on.kind == ops::AttnDecodeKind::dq4 && on.n_split == off.n_split);
        CHECK(on.grid.x == off.grid.x && on.grid.y == off.grid.y && on.grid.z == off.grid.z && on.block.x == off.block.x);
        CHECK(on.groups.count[0] == off.groups.count[0] && on.groups.first[1] == off.groups.first[1] && on.combine_q8 == off.combine_q8);
        // never outside the attn_dq4 choice: f16 / q8h KV (balance / precise) keep their kernels whatever the flag says
        ops::AttnDecodeProblem pf = p;
        pf.kv_q4 = false;
        const kn::KernelTable* kf = &s8f;
        ops::AttnDecodeFlags f64 = fo;
        f64.fd_splits = 64;
        const std::string nf = nameOf(ops::chooseAttnDecode(pf, f64, *kf, kf->caps, p8s).f);
        CHECK(nf.rfind("attn_ws4", 0) == 0);
        // WHIRL_ATTN_DQ4=0 (attn_dq4 off): no attn_dq4o either
        ops::AttnDecodeFlags fd = fo;
        fd.attn_dq4 = false;
        const std::string nd = nameOf(plan(fd, p8s).f);
        CHECK(nd != "attn_dq4o" && nd != "attn_dq4");
    }
    // --- dq4_min_keys: the grid from the longest row and the keys-per-split target
    // 128k: ceil(131073 / 1024) = 129 -> 256, capped at DQ_MAXS 128 (the kernel's dq_per never uses more): today's grid,
    // so at 128k only the block order changes anything
    CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(pos128, 2), 0, 256, 1024) == 128);
    CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(pos128, 2), 0, 256, 1024) == ops::dq4Splits(std::span<const std::int32_t>(pos128, 2), 0, 256));
    {
        const std::int32_t p32[2] = {32767, 32768}, p64[2] = {65535, 65536}, p2[2] = {2047, 2048}, p1[1] = {100};
        // 32k: ceil(32769 / 1024) = 33 -> 64 splits (power of two at or above), per = dq_per(32769, 64) = 576
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p32, 2), 0, 256, 1024) == 64);
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p32, 1), 0, 256, 1024) == 32);  // 32768 keys: exactly 32
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p64, 1), 0, 256, 1024) == 64);
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p32, 1), 0, 256, 2048) == 16);
        // short rows: never under 8 splits (the dq4Splits floor), never over today's grid
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p2, 2), 0, 256, 1024) == 8);
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p1, 1), 0, 256, 1024) == 8);
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p32, 1), 0, 16, 512) == 16);
        // min_keys 0: today's grid untouched
        CHECK(ops::dq4MinKeysSplits(std::span<const std::int32_t>(p32, 1), 0, 256, 0) == 256);
    }
    // the plan takes it (8060S and R9700 sets alike: a host rule, "the R9700 can take it later" = its profile row)
    {
        const std::int32_t p32[2] = {32767, 32768};
        ops::AttnDecodeProblem q = p;
        q.row_pos = p32;
        ops::AttnDecodeFlags fm = f;
        fm.dq4_min_keys = 1024;
        const ops::AttnDecodePlan a = ops::chooseAttnDecode(q, fm, *k8, k8->caps, p8s);
        CHECK(nameOf(a.f) == "attn_dq4" && a.n_split == 64 && a.grid.y == 64);
        const kn::KernelTable* k97 = &s97;
        CHECK(ops::chooseAttnDecode(q, fm, *k97, k97->caps, p97).n_split == 64);
        // rows the host does not know: today's fixed grid (a graph replays it)
        ops::AttnDecodeProblem qu = q;
        qu.rows_known = false;
        qu.row_pos = {};
        qu.row_base = {};
        CHECK(ops::chooseAttnDecode(qu, fm, *k8, k8->caps, p8s).n_split == 256);
        // with both switches: attn_dq4o on the min-keys grid
        fm.dq4_order = true;
        const ops::AttnDecodePlan b = ops::chooseAttnDecode(q, fm, *k8, k8->caps, p8s);
        CHECK(nameOf(b.f) == "attn_dq4o" && b.n_split == 64);
        // f16 KV: the min-keys rule never touches the ws4 grid
        ops::AttnDecodeProblem qf = q;
        qf.kv_q4 = false;
        const kn::KernelTable* kf = &s8f;
        ops::AttnDecodeFlags f64 = fm;
        f64.fd_splits = 64;
        CHECK(ops::chooseAttnDecode(qf, f64, *kf, kf->caps, p8s).n_split == 64);
    }
    // --- kernel model: under the min-keys grid every key of every row is in exactly one launched split, no keyed split is
    // past the grid, and (window off) every keyed split holds at least min(n_pos, min_keys / 2) keys: n_pos 1 .. 256k,
    // targets 512 / 1024 / 2048, the four windows of testDq4Partition
    const std::int32_t wins[4] = {0, dq4WinArg(16384, 65536), dq4WinArg(4096, 0), dq4WinArg(1024, 16384)};
    long long rows = 0, cover_bad = 0, small_bad = 0, fewer = 0;
    for (const std::uint32_t mk : {512u, 1024u, 2048u})
        for (const std::int32_t win : wins)
            for (std::int32_t n_pos = 1; n_pos <= 262144; n_pos += (win == 0 ? 1 : 13)) {
                const std::int32_t pos = n_pos - 1;
                const std::uint32_t s = ops::dq4MinKeysSplits(std::span<const std::int32_t>(&pos, 1), win, 256, mk);
                rows += 1;
                if (s < ops::dq4Splits(std::span<const std::int32_t>(&pos, 1), win, 256)) fewer += 1;
                std::int32_t covered = 0, prev_end = -1;
                bool ok = s >= 1 && s <= 256;
                std::int32_t min_keyed = 1 << 30;
                for (std::uint32_t i = 0; ok && i < 256; ++i) {
                    if (i >= s) {
                        // a split past the grid must hold no key (it is never launched)
                        const KRange r = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(s), win);
                        if (r.b0 < n_pos && r.b1 > r.b0) ok = false;
                        continue;
                    }
                    const KRange r = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(s), win);
                    if (r.b0 >= n_pos) continue;  // empty split: the kernel writes (-inf, 0), the combine skips it
                    if (r.b0 < prev_end) ok = false;  // overlap
                    covered += r.b1 - r.b0;
                    prev_end = r.b1;
                    if (i > 0) min_keyed = std::min(min_keyed, r.b1 - r.b0);
                }
                if (!ok) {
                    cover_bad += 1;
                    continue;
                }
                // window off: all keys; window on: the sink + the window's keys (kRange's own key set)
                std::int32_t want = n_pos;
                if (win != 0) {
                    const std::int32_t w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
                    const std::int32_t w0 = std::max<std::int32_t>(256, (n_pos - w) & ~63);
                    if (n_pos >= min_pos && w0 > 256) want = 256 + (n_pos - w0);
                }
                if (covered != want) cover_bad += 1;
                // window off, more than one keyed split: the last split may be short, the others hold >= min_keys / 2
                if (win == 0 && s > 8 && min_keyed != (1 << 30)) {
                    std::int32_t keyed = 0, short_ok = 0;
                    for (std::uint32_t i = 0; i + 1 < s; ++i) {
                        const KRange r = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(s), win);
                        const KRange nx = kRange(n_pos, static_cast<std::int32_t>(i + 1), static_cast<std::int32_t>(s), win);
                        if (r.b0 < n_pos && nx.b0 < n_pos) {
                            keyed += 1;
                            if (r.b1 - r.b0 >= static_cast<std::int32_t>(mk / 2)) short_ok += 1;
                        }
                    }
                    if (short_ok != keyed) small_bad += 1;
                }
            }
    CHECK(rows > 0 && cover_bad == 0);
    CHECK(small_bad == 0);
    CHECK(fewer > 0);  // the rule launches fewer splits than today somewhere (32k / 64k)
    std::printf("  testAttnDq4F8: %lld block maps, %lld min-keys rows (%lld with fewer splits than today)\n", maps, rows, fewer);
}

// v0252-8sq5 (PLAN V1-109, fast only: q4 KV): the q4 decode / verify split count from the device profile's measured table
// (ops::dq4TableSplits, flag dq4_splits: 0 = today's rule ops::dq4Splits, kDq4SplitsAuto = the profile table, 1..256 = that
// count). Expected values are the kernel bench's picks (scratch/v0252_8sq5/gpu/20261124-1609/table.txt), written out here,
// not read from the row.
void testAttnDq4Table() {
    const auto& p97 = whirl::device::k_profile_gfx1201;
    const auto& p8s = whirl::device::k_profile_gfx1151;
    CHECK(!p97.dq4_split_table.on && p8s.dq4_split_table.on);
    {
        const ops::AttnDecodeFlags d;
        CHECK(d.dq4_splits == 0u);  // ops default: today's rule (a caller without the plan's flags keeps today's grid)
    }
    const std::int32_t win_def = dq4WinArg(16384, 65536);
    // rows of one sequence ending at keys - 1 (verify rows), or the draft row (win_def)
    auto rowsAt = [](std::int32_t keys, std::int32_t n) {
        std::vector<std::int32_t> v;
        for (std::int32_t r = keys - n; r < keys; ++r) v.push_back(r);
        return v;
    };
    const std::vector<std::int32_t> zeros(8, 0);
    auto tab = [&](const std::vector<std::int32_t>& pos, std::int32_t win, std::uint32_t flag, const whirl::device::Profile& pr) {
        return ops::dq4TableSplits(std::span<const std::int32_t>(pos), std::span<const std::int32_t>(zeros.data(), pos.size()), win, 256, flag,
                                   pr.dq4_split_table);
    };
    auto today = [](const std::vector<std::int32_t>& pos, std::int32_t win) { return ops::dq4Splits(std::span<const std::int32_t>(pos), win, 256); };
    const std::uint32_t A = ops::kDq4SplitsAuto;
    struct Cell {
        std::int32_t keys, rows;  // rows 0 = the MTP draft row (window on)
        std::uint32_t s;
    };
    // every measured (row set, keys) cell -> the bench's pick (ties resolved as in progress/v0252-8sq5.HANDOFF.md round 2)
    const Cell measured[] = {
        {8192, 1, 8},   {16384, 1, 16}, {32768, 1, 16}, {49152, 1, 16}, {65536, 1, 16}, {98304, 1, 16}, {131072, 1, 16},
        {8192, 2, 8},   {16384, 2, 8},  {32768, 2, 8},  {49152, 2, 8},  {65536, 2, 8},  {98304, 2, 64}, {131072, 2, 64},
        {8192, 3, 8},   {16384, 3, 8},  {32768, 3, 8},  {49152, 3, 8},  {65536, 3, 32}, {98304, 3, 32}, {131072, 3, 32},
        {65536, 0, 16}, {98304, 0, 16}, {131072, 0, 16},
        // between and past the breakpoints: the nearest lower-or-equal breakpoint
        {8193, 1, 8},   {16383, 1, 8},  {40000, 1, 16}, {262144, 1, 16}, {65535, 2, 8},  {98303, 2, 8},  {120000, 2, 64},
        {262144, 2, 64}, {65535, 3, 8}, {200000, 3, 32}, {70000, 0, 16}, {262144, 0, 16},
    };
    long long cells = 0;
    ops::AttnDecodeProblem p;
    p.head_dim = 256;
    p.n_head = 16;
    p.n_head_kv = 2;
    p.max_ctx = 262144;
    p.kv_q4 = true;
    p.rows_known = true;
    ops::AttnDecodeFlags f;
    f.fd_splits = 256;
    static const kn::KernelTable s8 = attnTable(true, true, true, false), s97 = attnTable(false, false, true, true);
    auto k8 = std::make_unique<kn::KernelTable>(s8);
    k8->attn_dq4o = tag("attn_dq4o");
    const kn::KernelTable* k97 = &s97;
    for (const Cell& c : measured) {
        const std::int32_t n = c.rows == 0 ? 1 : c.rows, win = c.rows == 0 ? win_def : 0;
        const std::vector<std::int32_t> pos = rowsAt(c.keys, n);
        const std::uint32_t got = tab(pos, win, A, p8s);
        CHECK(got == c.s);
        if (got != c.s) std::printf("    dq4 table keys %d rows %d: %u, want %u\n", c.keys, c.rows, got, c.s);
        // the plan launches it (gfx1151 kernel set, attn_dq4o): grid.y = the table's count
        ops::AttnDecodeProblem q = p;
        q.n = static_cast<std::uint32_t>(n);
        q.row_pos = std::span<const std::int32_t>(pos);
        q.row_base = std::span<const std::int32_t>(zeros.data(), pos.size());
        q.win = win;
        ops::AttnDecodeFlags fa = f;
        fa.dq4_splits = A;
        fa.dq4_order = true;
        const ops::AttnDecodePlan pl = ops::chooseAttnDecode(q, fa, *k8, k8->caps, p8s);
        CHECK(pl.kind == ops::AttnDecodeKind::dq4 && nameOf(pl.f) == "attn_dq4o" && pl.n_split == c.s && pl.grid.y == c.s && pl.grid.z == q.n);
        // gfx1201: the row is off -> today's rule for the same inputs
        CHECK(tab(pos, win, A, p97) == 0u);
        CHECK(ops::chooseAttnDecode(q, fa, *k97, k97->caps, p97).n_split == today(pos, win));
        // switch 0 = today's rule on the 8060S too
        ops::AttnDecodeFlags f0 = fa;
        f0.dq4_splits = 0;
        CHECK(tab(pos, win, 0, p8s) == 0u && ops::chooseAttnDecode(q, f0, *k8, k8->caps, p8s).n_split == today(pos, win));
        // WHIRL_DQ4_MINKEYS set: it wins over the table
        ops::AttnDecodeFlags fm = fa;
        fm.dq4_min_keys = 1024;
        CHECK(ops::chooseAttnDecode(q, fm, *k8, k8->caps, p8s).n_split ==
              ops::dq4MinKeysSplits(std::span<const std::int32_t>(pos), win, 256, 1024));
        cells += 1;
    }
    // forced counts: a valid count is taken as is, an invalid one falls back to today's rule
    {
        const std::vector<std::int32_t> r2 = rowsAt(32768, 2), d1 = rowsAt(131072, 1);
        CHECK(tab(r2, 0, 32, p8s) == 32u && tab(r2, 0, 32, p97) == 32u);  // forced works on any card (the A/B arm)
        CHECK(tab(r2, 0, 1, p8s) == 1u);                                   // one split: valid (no window)
        CHECK(tab(d1, win_def, 4, p8s) == 0u);                             // < 8 with the window: the window branch needs >= 8
        CHECK(tab(r2, 0, 257, p8s) == 0u);                                 // past the grid
        CHECK(ops::dq4TableSplits(std::span<const std::int32_t>(r2), std::span<const std::int32_t>(zeros.data(), 2), 0, 64, 128, p8s.dq4_split_table) == 0u);
        CHECK(ops::dq4GridOk(std::span<const std::int32_t>(r2), 0, 32, 256) && !ops::dq4GridOk(std::span<const std::int32_t>(r2), 0, 300, 256) &&
              !ops::dq4GridOk(std::span<const std::int32_t>(r2), 0, 128, 64) && !ops::dq4GridOk(std::span<const std::int32_t>(d1), win_def, 4, 256));
    }
    // row sets the bench did not time keep today's rule: 4+ rows, rows of different sequences (batched decode), rows that are
    // not consecutive, a window with more than one row, keys below the first breakpoint, unknown rows
    {
        const std::vector<std::int32_t> r4 = rowsAt(32768, 4), r2 = rowsAt(32768, 2), r1 = rowsAt(4096, 1);
        CHECK(tab(r4, 0, A, p8s) == 0u);
        const std::int32_t bases[2] = {0, 1};
        CHECK(ops::dq4TableSplits(std::span<const std::int32_t>(r2), std::span<const std::int32_t>(bases, 2), 0, 256, A, p8s.dq4_split_table) == 0u);
        const std::int32_t gap[2] = {32760, 32767};
        CHECK(ops::dq4TableSplits(std::span<const std::int32_t>(gap, 2), std::span<const std::int32_t>(zeros.data(), 2), 0, 256, A, p8s.dq4_split_table) == 0u);
        CHECK(tab(rowsAt(131072, 2), win_def, A, p8s) == 0u);
        CHECK(tab(r1, 0, A, p8s) == 0u);
        // the draft row below the window threshold launches like a plain row: the rows-1 table
        CHECK(tab(rowsAt(32768, 1), win_def, A, p8s) == 16u);
        ops::AttnDecodeProblem q = p;
        q.n = 2;
        q.rows_known = false;
        ops::AttnDecodeFlags fa = f;
        fa.dq4_splits = A;
        CHECK(ops::chooseAttnDecode(q, fa, *k8, k8->caps, p8s).n_split == 256);  // unknown rows: today's fixed grid
        // f16 KV (balance / precise): the flag never touches the ws4 grid
        q = p;
        q.n = 2;
        q.kv_q4 = false;
        q.row_pos = std::span<const std::int32_t>(r2);
        q.row_base = std::span<const std::int32_t>(zeros.data(), 2);
        static const kn::KernelTable s8f = attnTable(true, true, false, false);
        ops::AttnDecodeFlags f64 = fa;
        f64.fd_splits = 64;
        CHECK(ops::chooseAttnDecode(q, f64, s8f, s8f.caps, p8s).n_split == 64);
    }
    // --- no invalid grid: rows 1..4 x keys 1..262144 x window off / default, auto and forced 8 / 16 / 64: whatever count the
    // plan takes, every key of every row is in exactly one launched split (the kernel model kRange) and none past the grid
    long long swept = 0, bad = 0, from_table = 0;
    for (const std::int32_t win : {0, win_def})
        for (std::int32_t nr = 1; nr <= 4; ++nr)
            for (std::int32_t keys = nr; keys <= 262144; keys += (keys < 1024 ? 1 : 61)) {
                const std::vector<std::int32_t> pos = rowsAt(keys, nr);
                for (const std::uint32_t flag : {A, 8u, 16u, 64u}) {
                    const std::uint32_t t = tab(pos, win, flag, p8s);
                    const std::uint32_t s = t != 0 ? t : today(pos, win);
                    from_table += t != 0;
                    swept += 1;
                    bool ok = s >= 1 && s <= 256 && !(win != 0 && s < 8);
                    for (const std::int32_t pp : pos) {
                        const std::int32_t n_pos = pp + 1;
                        std::int32_t covered = 0, prev_end = -1;
                        for (std::uint32_t i = 0; ok && i < 256; ++i) {
                            const KRange r = kRange(n_pos, static_cast<std::int32_t>(i), static_cast<std::int32_t>(s), win);
                            if (r.b0 >= n_pos || r.b1 <= r.b0) continue;
                            if (i >= s || r.b0 < prev_end) ok = false;
                            covered += r.b1 - r.b0;
                            prev_end = r.b1;
                        }
                        std::int32_t want = n_pos;
                        const std::int32_t w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
                        const std::int32_t w0 = std::max<std::int32_t>(256, (n_pos - w) & ~63);
                        if (win != 0 && s > 1 && n_pos >= min_pos && w0 > 256) want = 256 + (n_pos - w0);
                        if (covered != want) ok = false;
                    }
                    bad += !ok;
                }
            }
    CHECK(swept > 0 && bad == 0 && from_table > 0);
    std::printf("  testAttnDq4Table: %lld measured cells, %lld swept (%lld from the table or forced), %lld bad\n", cells, swept, from_table, bad);
}

// v0252-8sws (PLAN V1-88 on gfx1151): the gfx1151 attn_ws4 / attn_ws4g split count for q8 / q8h KV by machine fill - the
// l2split rule (ops::lsplitCount, numerics flag attn_lsplit: precise 0, balance / fast auto) with the profile's
// attn_ws4_split_blocks (gfx1151 32), returned only where the kernel table has that split constant
// (kernels::kWs4SplitCounts 8 / 16: attn_ws4_q8_sp<S>, attn_ws4g<cg>_q8_sp<S>), else today's 64 and today's kernels.
// Measured (kernel bench ws4s, 8060S 2026-11-24 19:10, q8h): Ornith (2 KV heads) best 16, Swift (4) best 8 in every cell
// but Swift 3 rows at 131072 (64 best there; S 8 1.017). Pins: (0) profile / constants / names; (1) every measured cell
// (model, rows 1 / 2 / 3 / draft row, keys 8192 .. 131072) -> the picked count, the _sp kernel of the CG today's plan takes,
// everything else today's plan; (2) precise (flag 0) and f16 KV = today's plan and kernels; (3) the switch states; (4) the
// R9700 unchanged; (5) row invariance (one count per model: rows 1..9, unknown rows); (6) a sweep of KV heads x GQA x rows x
// keys x window: every launched count has its kernel, never a count the table lacks. Fails to build before v0252-8sws.
// Round 3: (1) / (5) for the measured limit table (device::Profile::ws4_split_limits, ops::ws4SplitForStep).
void testAttnDecodeWs4Split() {
    namespace dv = whirl::device;
    const auto& p97 = dv::k_profile_gfx1201;
    const auto& p8s = dv::k_profile_gfx1151;
    const std::uint32_t A = ops::kAttnLsplitAuto;
    // (0)
    CHECK(p8s.attn_ws4_split_blocks == 32 && p97.attn_ws4_split_blocks == 0);
    CHECK(kn::kWs4SplitCounts[0] == 8 && kn::kWs4SplitCounts[1] == 16);
    CHECK(kn::ws4SplitName(0, 8) == "attn_ws4_q8_sp8" && kn::ws4SplitName(2, 16) == "attn_ws4g2_q8_sp16" && kn::ws4SplitName(3, 8) == "attn_ws4g3_q8_sp8");
    auto spTable = [](kn::KernelTable k, bool with8, bool with16) {
        for (int i = 0; i < 2; ++i) {
            const std::uint32_t s = kn::kWs4SplitCounts[i];
            if ((s == 8 && !with8) || (s == 16 && !with16)) continue;
            k.attn_ws4_sp[i] = tag(kn::ws4SplitName(0, s));
            for (int cg = 1; cg <= 3; ++cg) k.attn_ws4g_sp[i][cg - 1] = tag(kn::ws4SplitName(cg, s));
        }
        return k;
    };
    static const kn::KernelTable k8 = attnTable(true, true, false, false);  // today's gfx1151 table (no _sp entries)
    static const kn::KernelTable ks = spTable(k8, true, true), ks16 = spTable(k8, false, true);
    CHECK(ops::ws4SplitCount(2, true, A, 32, ks) == 16 && ops::ws4SplitCount(4, true, A, 32, ks) == 8);
    CHECK(ops::ws4SplitCount(4, true, A, 32, ks16) == 64);  // no S 8 entries -> 64
    CHECK(ops::ws4SplitCount(2, false, A, 32, ks) == 64 && ops::ws4SplitCount(2, true, 0, 32, ks) == 64 && ops::ws4SplitCount(2, true, A, 0, ks) == 64);
    CHECK(ops::ws4SplitCount(2, true, A, 32, k8) == 64);  // a table without the entries -> 64
    {
        kn::AttnKernels part;  // the S 16 entries without CG 1 / 2: a verify group would have no kernel -> 64
        part.attn_ws4_sp[1] = tag(kn::ws4SplitName(0, 16));
        part.attn_ws4g_sp[1][2] = tag(kn::ws4SplitName(3, 16));
        CHECK(ops::ws4SplitCount(2, true, A, 32, part) == 64);
    }
    struct Shape {
        const char* model;
        std::uint32_t heads, nkv, pick;
    };
    const Shape shapes[] = {{"ornith", 16, 2, 16}, {"swift", 24, 4, 8}};
    const std::int32_t kDraftWin = (16384 / 64) | ((65536 / 1024) << 16);
    std::vector<std::int32_t> pos, base;
    auto problem = [&](const Shape& sh, std::uint32_t L, std::uint32_t R, bool draft, bool q8) {
        pos.clear();
        base.clear();
        for (std::uint32_t r = L - R; r < L; ++r) {
            pos.push_back(static_cast<std::int32_t>(r));
            base.push_back(0);
        }
        ops::AttnDecodeProblem q;
        q.n = R;
        q.head_dim = 256;
        q.n_head = sh.heads;
        q.n_head_kv = sh.nkv;
        q.max_ctx = 262144;
        q.kv_q8 = q8;
        q.o_on_gemv = true;
        q.rows_known = true;
        q.row_pos = std::span<const std::int32_t>(pos);
        q.row_base = std::span<const std::int32_t>(base);
        q.win = draft ? kDraftWin : 0;
        return q;
    };
    ops::AttnDecodeFlags f0, fa;  // f0 = precise (flag 0), fa = balance / fast (auto)
    fa.attn_lsplit = A;
    auto spName = [](const ops::AttnDecodePlan& t, std::uint32_t s) {
        const std::string n = nameOf(t.f);  // today's kernel: attn_ws4 or attn_ws4g<cg> (fake names carry no _q8)
        const int cg = n == "attn_ws4" ? 0 : n[9] - '0';
        return kn::ws4SplitName(cg, s);
    };
    // round 3: the measured limits (kernel bench ws4s TABLE runs, 8060S 2026-11-25 01:24-01:43): the largest key count at which
    // S beat 64 (whole 95 % interval < 1) with every smaller measured key count also faster; index 0 = the MTP draft row,
    // 1..9 = rows. A hand copy of the profile table (the pin): Ornith S 16, Swift S 8
    const std::uint32_t lim16[10] = {262144, 262144, 262144, 262144, 262144, 262144, 262144, 16384, 16384, 16384};
    const std::uint32_t lim8[10] = {262144, 262144, 262144, 98304, 98304, 196608, 49152, 65536, 65536, 16384};
    auto want = [&](const Shape& sh, std::uint32_t R, bool draft, std::uint32_t L) -> std::uint32_t {
        const std::uint32_t* lim = sh.pick == 16 ? lim16 : lim8;
        const std::uint32_t cls = draft ? 0 : R;
        return cls <= 9 && L <= lim[cls] ? sh.pick : 64u;
    };
    long long cells = 0, on_cells = 0;
    // (1) + (2): every measured (model, rows 1..9 / draft row, keys) cell, the keys just above each limit, and 262144 + 1
    for (const Shape& sh : shapes)
        for (const std::uint32_t L0 : {8192u, 16384u, 16385u, 32768u, 49152u, 49153u, 65536u, 65537u, 98304u, 98305u, 131072u, 196608u, 196609u, 262144u, 262145u})
            for (int rs = 0; rs <= 9; ++rs) {  // 0 = the MTP draft row
                if (rs == 0 && L0 < 65536) continue;
                const std::uint32_t R = rs == 0 ? 1 : static_cast<std::uint32_t>(rs);
                ops::AttnDecodeProblem q = problem(sh, L0, R, rs == 0, true);
                q.max_ctx = 524288;
                const auto t = ops::chooseAttnDecode(q, f0, ks, ks.caps, p8s);   // precise: today's
                const auto t0 = ops::chooseAttnDecode(q, fa, k8, k8.caps, p8s);  // today's table: today's plan whatever the flag
                const auto b = ops::chooseAttnDecode(q, fa, ks, ks.caps, p8s);
                CHECK(t.grid.y == 64 && t.n_split == 64 && (nameOf(t.f) == "attn_ws4" || nameOf(t.f).rfind("attn_ws4g", 0) == 0));
                CHECK(samePlan(t, t0));
                const std::uint32_t w = want(sh, R, rs == 0, L0);
                CHECK(ops::ws4SplitForStep(q, sh.pick, p8s.ws4_split_limits) == w);
                if (w == 64) {
                    CHECK(samePlan(b, t));  // above a limit: today's plan and kernels exactly
                } else {
                    CHECK(b.grid.y == w && b.n_split == w && nameOf(b.f) == spName(t, w));
                    CHECK(b.kind == t.kind && b.grid.x == t.grid.x && b.grid.z == t.grid.z && b.block.x == t.block.x && b.combine_q8 == t.combine_q8 &&
                          b.combine_grid.x == t.combine_grid.x && b.combine_grid.y == t.combine_grid.y);
                    for (std::uint32_t g = 0; g < t.grid.z; ++g) CHECK(b.groups.first[g] == t.groups.first[g] && b.groups.count[g] == t.groups.count[g]);
                    on_cells += 1;
                }
                // f16 KV (precise's kernel; balance only with a forced format): today's plan under auto too
                const ops::AttnDecodeProblem qf = problem(sh, L0, R, rs == 0, false);
                CHECK(samePlan(ops::chooseAttnDecode(qf, fa, ks, ks.caps, p8s), ops::chooseAttnDecode(qf, f0, ks, ks.caps, p8s)));
                CHECK(ops::chooseAttnDecode(qf, fa, ks, ks.caps, p8s).grid.y == 64);
                cells += 1;
            }
    CHECK(cells == 2 * (15 * 9 + 9) && on_cells > 100);
    // the measured slower cells of round 2 are all 64 now (Ornith rows 7..9 from 32768; Swift rows 3 / 4 at 131072, 5 at 262144,
    // 6 at 65536, 7 / 8 at 98304, 9 at 32768)
    CHECK(want(shapes[0], 7, false, 32768) == 64 && want(shapes[1], 3, false, 131072) == 64 && want(shapes[1], 4, false, 131072) == 64 &&
          want(shapes[1], 5, false, 262144) == 64 && want(shapes[1], 6, false, 65536) == 64 && want(shapes[1], 7, false, 98304) == 64 &&
          want(shapes[1], 8, false, 98304) == 64 && want(shapes[1], 9, false, 32768) == 64);
    // (3) the switch (WHIRL_ATTN_LSPLIT -> flag attn_lsplit): 0 = 64; N = N blocks (Swift 64 -> 16: an _sp entry; Ornith 64 ->
    // 32: none -> 64; 128 -> 32 / 64: none -> 64); the profile row decides under auto (row 0 -> 64)
    {
        const ops::AttnDecodeProblem qs = problem(shapes[1], 32768, 2, false, true), qo = problem(shapes[0], 32768, 2, false, true);
        auto gy = [&](const ops::AttnDecodeProblem& q, std::uint32_t fl, const dv::Profile& pr) {
            ops::AttnDecodeFlags g;
            g.attn_lsplit = fl;
            return ops::chooseAttnDecode(q, g, ks, ks.caps, pr).grid.y;
        };
        CHECK(gy(qs, 0, p8s) == 64 && gy(qo, 0, p8s) == 64);
        CHECK(gy(qs, A, p8s) == 8 && gy(qo, A, p8s) == 16);
        CHECK(gy(qs, 32, p8s) == 8 && gy(qo, 32, p8s) == 16);
        CHECK(gy(qs, 64, p8s) == 16 && gy(qo, 64, p8s) == 64);
        CHECK(gy(qs, 128, p8s) == 64 && gy(qo, 128, p8s) == 64);
        dv::Profile poff = p8s;
        poff.attn_ws4_split_blocks = 0;
        CHECK(gy(qs, A, poff) == 64 && gy(qo, A, poff) == 64);
        // the attn_wsplit fallback on the 8060S (WHIRL_ATTN_WS4=0) keeps today's split: its profile row attn_lsplit_blocks is 0
        ops::AttnDecodeFlags gw = fa, tw = f0;
        gw.attn_ws4 = tw.attn_ws4 = false;
        CHECK(samePlan(ops::chooseAttnDecode(qs, gw, ks, ks.caps, p8s), ops::chooseAttnDecode(qs, tw, ks, ks.caps, p8s)));
    }
    // (4) the R9700: no attn_ws4 path; the _sp entries change nothing there (the l2split rule of its own branch stays)
    {
        static const kn::KernelTable kb = attnTable(false, false, true, true);
        static const kn::KernelTable kq = ws4rTable(kb, true), kqs = spTable(kq, true, true);
        for (const Shape& sh : shapes)
            for (const std::uint32_t R : {1u, 2u, 4u}) {
                const ops::AttnDecodeProblem q = problem(sh, 70000, R, false, true);
                for (const std::uint32_t fl : {A, 0u})
                    for (const ops::AttnDecodeFlags* fp : {&f0, &fa}) {
                        ops::AttnDecodeFlags g = *fp;
                        g.attn_lsplit = fl;
                        CHECK(samePlan(ops::chooseAttnDecode(q, g, kqs, kqs.caps, p97), ops::chooseAttnDecode(q, g, kq, kq.caps, p97)));
                    }
            }
    }
    // (5) round 3 (the count now depends on the step's rows and largest n_pos; the run stays deterministic): one count per launch
    // (grid.y, the group key and the kernel use the same S); the key count compared with the limit is the step's largest
    // n_pos (max row_pos + 1), or max_ctx when the host does not know the rows; rows 10+ / several sequences / gapped rows /
    // unknown multi-row steps (not measured) -> 64; a forced N (WHIRL_ATTN_LSPLIT=N) skips the limits, flag 0 = 64 always
    for (const Shape& sh : shapes) {
        // the largest n_pos of the step decides (7 rows: Ornith limit 16384, Swift 65536)
        {
            ops::AttnDecodeProblem q7 = problem(sh, 16384, 7, false, true);  // largest n_pos 16384
            CHECK(ops::chooseAttnDecode(q7, fa, ks, ks.caps, p8s).grid.y == sh.pick);
            ops::AttnDecodeProblem q7b = problem(sh, 16385, 7, false, true);  // largest n_pos 16385: Ornith over its limit 16384
            CHECK(ops::chooseAttnDecode(q7b, fa, ks, ks.caps, p8s).grid.y == (sh.pick == 16 ? 64u : sh.pick));
        }
        // rows are grouped by the ranges of the count launched, not of 64: n_pos 1024 / 1025 share every 64-split range
        // (64 keys a split) but not the S ranges (Ornith 16: 64 / 128 keys a split; Swift 8: 128 / 192)
        {
            const auto bg = ops::chooseAttnDecode(problem(sh, 1025, 2, false, true), fa, ks, ks.caps, p8s);
            const auto tg = ops::chooseAttnDecode(problem(sh, 1025, 2, false, true), f0, ks, ks.caps, p8s);
            CHECK(tg.grid.z == 1 && bg.grid.z == 2 && bg.groups.count[0] == 1 && bg.groups.count[1] == 1 && nameOf(bg.f) == kn::ws4SplitName(0, sh.pick));
        }
        // unknown rows: one row -> max_ctx against the rows-1 limit (262144); several rows -> 64
        {
            ops::AttnDecodeProblem qu = problem(sh, 50000, 1, false, true);
            qu.rows_known = false;
            qu.row_pos = {};
            qu.row_base = {};
            qu.max_ctx = 262144;
            CHECK(ops::chooseAttnDecode(qu, fa, ks, ks.caps, p8s).grid.y == sh.pick);
            qu.max_ctx = 262208;
            CHECK(ops::chooseAttnDecode(qu, fa, ks, ks.caps, p8s).grid.y == 64);
            ops::AttnDecodeProblem qu3 = problem(sh, 50000, 3, false, true);
            qu3.rows_known = false;
            CHECK(ops::chooseAttnDecode(qu3, fa, ks, ks.caps, p8s).grid.y == 64);
        }
        // v0252-e1s8 round 2 (C): an E1 verify of ONE slot - rows the host does not know, but one sequence at consecutive
        // positions (AttnDecodeProblem::rows_one_seq, set only on that path) -> the rows class against max_ctx (the upper
        // bound of every n_pos, as the one-row unknown case); without the field (any other unknown multi-row step) 64 as before
        {
            auto e1p = [&](std::uint32_t rows, std::uint32_t max_ctx, bool one_seq, bool q8) {
                ops::AttnDecodeProblem q = problem(sh, 50000, rows, false, q8);
                q.rows_known = false;
                q.row_pos = {};
                q.row_base = {};
                q.rows_one_seq = one_seq;
                q.max_ctx = max_ctx;
                return q;
            };
            // 2 rows (Ornith / Swift with one draft): the rows-2 limit is 262144 for both counts
            CHECK(ops::chooseAttnDecode(e1p(2, 32768, true, true), fa, ks, ks.caps, p8s).grid.y == sh.pick);
            CHECK(ops::chooseAttnDecode(e1p(2, 262144, true, true), fa, ks, ks.caps, p8s).grid.y == sh.pick);
            CHECK(ops::chooseAttnDecode(e1p(2, 262208, true, true), fa, ks, ks.caps, p8s).grid.y == 64);
            CHECK(ops::chooseAttnDecode(e1p(2, 32768, false, true), fa, ks, ks.caps, p8s).grid.y == 64);  // round 1 / not E1
            // 3 rows: rows-3 limit S 8 98304 (Swift), S 16 262144 (Ornith)
            CHECK(ops::chooseAttnDecode(e1p(3, 98304, true, true), fa, ks, ks.caps, p8s).grid.y == sh.pick);
            CHECK(ops::chooseAttnDecode(e1p(3, 131072, true, true), fa, ks, ks.caps, p8s).grid.y == (sh.pick == 8 ? 64u : sh.pick));
            // rows 10+: no class -> 64; precise (flag 0) and f16 KV -> 64 (today's plan)
            CHECK(ops::chooseAttnDecode(e1p(10, 8192, true, true), fa, ks, ks.caps, p8s).grid.y == 64);
            CHECK(ops::chooseAttnDecode(e1p(2, 8192, true, true), f0, ks, ks.caps, p8s).grid.y == 64);
            CHECK(ops::chooseAttnDecode(e1p(2, 8192, true, false), fa, ks, ks.caps, p8s).grid.y == 64);
            // the field never changes a step whose rows the host knows (every non-E1 plan)
            ops::AttnDecodeProblem qk = problem(sh, 16385, 7, false, true);
            const auto before = ops::chooseAttnDecode(qk, fa, ks, ks.caps, p8s);
            qk.rows_one_seq = true;
            CHECK(samePlan(ops::chooseAttnDecode(qk, fa, ks, ks.caps, p8s), before));
            // the function itself: unknown rows of one sequence = class n vs max_ctx
            CHECK(ops::ws4SplitForStep(e1p(2, 65536, true, true), sh.pick, p8s.ws4_split_limits) == sh.pick);
            CHECK(ops::ws4SplitForStep(e1p(2, 65536, false, true), sh.pick, p8s.ws4_split_limits) == 64);
        }
        // rows 10+, two sequences, gapped positions: not measured -> 64
        {
            CHECK(ops::chooseAttnDecode(problem(sh, 8192, 10, false, true), fa, ks, ks.caps, p8s).grid.y == 64);
            ops::AttnDecodeProblem q2 = problem(sh, 8192, 2, false, true);
            base[1] = 4096;  // the second row in another sequence
            CHECK(ops::chooseAttnDecode(q2, fa, ks, ks.caps, p8s).grid.y == 64);
            ops::AttnDecodeProblem qg = problem(sh, 8192, 2, false, true);
            pos[0] -= 5;  // gapped
            CHECK(ops::chooseAttnDecode(qg, fa, ks, ks.caps, p8s).grid.y == 64);
        }
        // the draft row has its own class (window on, one row): Swift draft 262144 vs rows 1 262144 - same value; checked via
        // the function with a table whose draft entry differs
        {
            dv::Ws4SplitLimits l = p8s.ws4_split_limits;
            const int si = sh.pick == 8 ? 0 : 1;
            l.max_keys[si][0] = 8192;  // draft row only up to 8192
            const ops::AttnDecodeProblem qd = problem(sh, 65536, 1, true, true), q1 = problem(sh, 65536, 1, false, true);
            CHECK(ops::ws4SplitForStep(qd, sh.pick, l) == 64 && ops::ws4SplitForStep(q1, sh.pick, l) == sh.pick);
            l.max_keys[si][1] = 8192;  // rows-1 only up to 8192, draft up to 262144
            l.max_keys[si][0] = 262144;
            CHECK(ops::ws4SplitForStep(qd, sh.pick, l) == sh.pick && ops::ws4SplitForStep(q1, sh.pick, l) == 64);
            dv::Ws4SplitLimits off = l;
            off.on = false;
            CHECK(ops::ws4SplitForStep(qd, sh.pick, off) == 64 && ops::ws4SplitForStep(problem(sh, 8192, 2, false, true), 64, l) == 64);
        }
        // a forced N skips the limits but not the kernel table; flag 0 = 64
        {
            ops::AttnDecodeProblem qf = problem(sh, 131072, 9, false, true);  // over every rows-9 limit
            ops::AttnDecodeFlags g = f0;
            g.attn_lsplit = 32;
            CHECK(ops::chooseAttnDecode(qf, g, ks, ks.caps, p8s).grid.y == sh.pick);
            CHECK(ops::chooseAttnDecode(qf, fa, ks, ks.caps, p8s).grid.y == 64);
            g.attn_lsplit = 0;
            CHECK(samePlan(ops::chooseAttnDecode(qf, g, ks, ks.caps, p8s), ops::chooseAttnDecode(qf, f0, k8, k8.caps, p8s)));
        }
        ops::AttnDecodeProblem qsm = problem(sh, 400, 1, false, true);
        qsm.max_ctx = 512;
        const auto bs = ops::chooseAttnDecode(qsm, fa, ks, ks.caps, p8s);
        CHECK(bs.grid.y == 8 && nameOf(bs.f) == kn::ws4SplitName(0, sh.pick));
    }
    // (6) sweep: every plan on the ws4 path launches 64 with today's kernel or S with that S's entry of the same CG
    long long swept = 0, bad = 0, on = 0;
    for (const std::uint32_t nkv : {1u, 2u, 3u, 4u, 6u, 8u})
        for (const std::uint32_t grp : {1u, 2u, 4u, 6u, 8u, 16u})
            for (const kn::KernelTable* kt : {&k8, &ks, &ks16})
                for (const std::uint32_t R : {1u, 2u, 3u, 5u, 9u, 17u})
                    for (const std::uint32_t L : {300u, 4097u, 65536u, 131072u})
                        for (const bool draft : {false, true}) {
                            if (R > L) continue;
                            const Shape sh{"sweep", nkv * grp, nkv, 0};
                            const ops::AttnDecodeProblem q = problem(sh, L, R, draft, true);
                            const auto t = ops::chooseAttnDecode(q, f0, *kt, kt->caps, p8s), b = ops::chooseAttnDecode(q, fa, *kt, kt->caps, p8s);
                            ++swept;
                            if (b.grid.y == t.grid.y) {
                                bad += !samePlan(b, t);
                                continue;
                            }
                            ++on;
                            const std::uint32_t s = b.grid.y;
                            const bool have = (s == 8 && kt->attn_ws4_sp[0] != nullptr) || (s == 16 && kt->attn_ws4_sp[1] != nullptr);
                            bad += !have || nameOf(b.f) != spName(t, s) || ops::lsplitCount(64, nkv, A, 32) != s || b.grid.x != t.grid.x || b.grid.z != t.grid.z;
                        }
    CHECK(swept > 1000 && bad == 0 && on > 0);
    std::printf("  testAttnDecodeWs4Split: %lld measured cells, %lld swept (%lld re-split), %lld bad\n", cells, swept, on, bad);
}


// v0252-e1grp (PLAN V1-93): an E1 verify of ONE slot keeps its attention group on the gfx1151 ws4 path. The host knows only an
// interval [pos0_lo, pos0_hi] of row 0's position (the cycle in flight accepts 0..nd drafts); rows r and r + 1 share a group
// iff their ws4 group keys are equal for EVERY row-0 position of the interval (ops::ws4E1Groups); otherwise one-row groups
// (today's E1 plan). The split count of the step (ws4SplitForStep) and the precise 64 rule are unchanged; with E1 off (rows
// known) nothing changes; the R9700 wsplit / ws4r path is unchanged.
void testAttnDecodeE1Group() {
    namespace dv = whirl::device;
    const auto& p97 = dv::k_profile_gfx1201;
    const auto& p8s = dv::k_profile_gfx1151;
    const std::uint32_t A = ops::kAttnLsplitAuto;
    static const kn::KernelTable k8 = attnTable(true, true, false, false);  // the gfx1151 table without the _sp entries
    static const kn::KernelTable ks = [] {
        kn::KernelTable k = attnTable(true, true, false, false);
        for (int i = 0; i < 2; ++i) {
            const std::uint32_t s = kn::kWs4SplitCounts[i];
            k.attn_ws4_sp[i] = tag(kn::ws4SplitName(0, s));
            for (int cg = 1; cg <= 3; ++cg) k.attn_ws4g_sp[i][cg - 1] = tag(kn::ws4SplitName(cg, s));
        }
        return k;
    }();
    const std::int32_t kDraftWin = (16384 / 64) | ((65536 / 1024) << 16);
    // the reference key (the attn_ws4g / attn_ws4 split rule, kernels/gfx1151/attn_ws4_index.h)
    auto refKey = [](std::int32_t pp, std::uint32_t ns, std::int32_t win) {
        const int np = pp + 1, s = static_cast<int>(ns);
        const ws4::Range r0 = ws4::block_range(np, 0, s, win);
        return std::array<std::int32_t, 3>{r0.b1 == ws4::kSink ? r0.b0 - 1 : 0, ws4::block_range(np, 1, s, win).b0, ws4::block_range(np, 2, s, win).b0};
    };
    long long bad = 0, cases = 0;
    // (1) the key function is the reference split rule
    for (const std::uint32_t ns : {8u, 16u, 64u})
        for (const std::int32_t win : {0, kDraftWin})
            for (std::int32_t pp : {0, 1, 63, 64, 1023, 1024, 1025, 4095, 4096, 8191, 8192, 32767, 32768, 65535, 65536, 70001, 262143}) {
                bad += ops::ws4GroupKey(pp, ns, win) != refKey(pp, ns, win);
                ++cases;
            }
    CHECK(bad == 0);
    // positions around every split-range boundary that matters for the swept counts, plus plain ones
    std::vector<std::int32_t> around;
    for (const std::int32_t b : {64, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304, 131072, 262144 - 64})
        for (std::int32_t d = -12; d <= 4; ++d) around.push_back(b + d);
    for (const std::int32_t x : {5, 300, 5000, 40000, 70001, 123457}) around.push_back(x);
    // (2) a one-position interval is the known-rows grouping of chooseAttnDecode for the same rows; (3) never a group whose rows'
    // keys differ at any position of the interval, and never two neighbouring groups that could have been one (maximal)
    std::vector<std::int32_t> pos, base;
    auto known = [&](std::int32_t p0, std::uint32_t n, std::uint32_t heads, std::uint32_t nkv, std::int32_t win, bool q8) {
        pos.clear();
        base.clear();
        for (std::uint32_t r = 0; r < n; ++r) {
            pos.push_back(p0 + static_cast<std::int32_t>(r));
            base.push_back(0);
        }
        ops::AttnDecodeProblem q;
        q.n = n;
        q.n_head = heads;
        q.n_head_kv = nkv;
        q.max_ctx = 262144;
        q.kv_q8 = q8;
        q.o_on_gemv = true;
        q.rows_known = true;
        q.row_pos = std::span<const std::int32_t>(pos);
        q.row_base = std::span<const std::int32_t>(base);
        q.win = win;
        return q;
    };
    auto e1 = [&](std::int32_t lo, std::int32_t hi, std::uint32_t n, std::uint32_t heads, std::uint32_t nkv, std::int32_t win, bool q8) {
        ops::AttnDecodeProblem q;
        q.n = n;
        q.n_head = heads;
        q.n_head_kv = nkv;
        q.max_ctx = 262144;
        q.kv_q8 = q8;
        q.o_on_gemv = true;
        q.rows_known = false;
        q.rows_one_seq = true;
        q.pos0_lo = lo;
        q.pos0_hi = hi;
        q.win = win;
        return q;
    };
    long long bad2 = 0, bad3 = 0, bad_max = 0;
    for (const std::uint32_t ns : {16u, 64u})
        for (const std::int32_t win : {0, kDraftWin})
            for (std::uint32_t n = 1; n <= 9; ++n)
                for (const std::int32_t p0 : around) {
                    if (p0 < 0) continue;
                    kn::AwGroups g{}, gk{};
                    std::uint32_t wd = 0;
                    const std::uint32_t ng = ops::ws4E1Groups(n, p0, p0, ns, win, kn::kMaxVerifyRows, g, wd);
                    // the known-rows reference: precise (flag 0) on the table without _sp = 64; balance auto on ks for 16
                    const ops::AttnDecodeProblem qk = known(p0, n, 16, 2, win, true);
                    ops::AttnDecodeFlags fl;
                    fl.attn_lsplit = ns == 64 ? 0u : A;
                    const auto pk = ops::chooseAttnDecode(qk, fl, ns == 64 ? k8 : ks, (ns == 64 ? k8 : ks).caps, p8s);
                    if (pk.n_split != ns && pk.grid.y != ns) continue;  // (a step over the measured limit runs 64: not this ns)
                    bad2 += ng != pk.grid.z;
                    for (std::uint32_t i = 0; i < ng && i < pk.grid.z; ++i) bad2 += g.first[i] != pk.groups.first[i] || g.count[i] != pk.groups.count[i];
                    ++cases;
                }
    for (const std::uint32_t ns : {16u, 64u})
        for (const std::int32_t win : {0, kDraftWin})
            for (const std::uint32_t n : {2u, 3u, 5u, 9u})
                for (const std::int32_t span : {1, 2, 3, 4, 11})
                    for (const std::int32_t lo : around) {
                        const std::int32_t hi = lo + span - 1;
                        if (lo < 0) continue;
                        kn::AwGroups g{};
                        std::uint32_t wd = 0;
                        const std::uint32_t ng = ops::ws4E1Groups(n, lo, hi, ns, win, kn::kMaxVerifyRows, g, wd);
                        std::uint32_t tot = 0, wmax = 0;
                        for (std::uint32_t i = 0; i < ng; ++i) {
                            bad3 += g.first[i] != static_cast<std::int32_t>(tot) || g.count[i] < 1;
                            tot += static_cast<std::uint32_t>(g.count[i]);
                            wmax = std::max<std::uint32_t>(wmax, static_cast<std::uint32_t>(g.count[i]));
                            // every row of the group has the group head's key at EVERY row-0 position of the interval
                            for (std::int32_t p0 = lo; p0 <= hi; ++p0)
                                for (std::int32_t r = 1; r < g.count[i]; ++r)
                                    bad3 += refKey(p0 + g.first[i] + r, ns, win) != refKey(p0 + g.first[i], ns, win);
                            // maximal: the next group's head differs from this group's head at SOME position of the interval
                            if (i + 1 < ng) {
                                bool differs = false;
                                for (std::int32_t p0 = lo; p0 <= hi; ++p0)
                                    differs = differs || refKey(p0 + g.first[i + 1], ns, win) != refKey(p0 + g.first[i], ns, win);
                                bad_max += !differs;
                            }
                        }
                        bad3 += tot != n || wmax != wd;
                        ++cases;
                    }
    CHECK(bad2 == 0);
    CHECK(bad3 == 0);
    CHECK(bad_max == 0);
    // (4) the plan of an Ornith E1 verify (16 heads, 2 KV heads, 2 rows; S 16 in balance on q8, 64 in precise / f16)
    {
        ops::AttnDecodeFlags f0, fa;
        fa.attn_lsplit = A;
        // away from a boundary: one 2-row group on attn_ws4g (CG 1), the split count unchanged
        const auto b = ops::chooseAttnDecode(e1(40000, 40001, 2, 16, 2, 0, true), fa, ks, ks.caps, p8s);
        CHECK(b.kind == ops::AttnDecodeKind::ws4g && b.grid.z == 1 && b.groups.first[0] == 0 && b.groups.count[0] == 2 && b.grid.y == 16 &&
              nameOf(b.f) == kn::ws4SplitName(1, 16) && b.grid.x == 2);
        const auto t = ops::chooseAttnDecode(e1(40000, 40001, 2, 16, 2, 0, true), f0, ks, ks.caps, p8s);
        CHECK(t.kind == ops::AttnDecodeKind::ws4g && t.grid.z == 1 && t.groups.count[0] == 2 && t.grid.y == 64 && nameOf(t.f) == "attn_ws4g1");
        const auto tf = ops::chooseAttnDecode(e1(40000, 40001, 2, 16, 2, 0, false), fa, ks, ks.caps, p8s);  // f16 KV: 64
        CHECK(tf.kind == ops::AttnDecodeKind::ws4g && tf.grid.z == 1 && tf.grid.y == 64 && nameOf(tf.f) == "attn_ws4g1");
        // the same plan as the known rows at each position of the interval (bitwise-equal launches for the kernel)
        for (const std::int32_t p0 : {40000, 40001}) {
            const auto k1 = ops::chooseAttnDecode(known(p0, 2, 16, 2, 0, true), fa, ks, ks.caps, p8s);
            CHECK(samePlan(b, k1));
            const auto k0 = ops::chooseAttnDecode(known(p0, 2, 16, 2, 0, true), f0, ks, ks.caps, p8s);
            CHECK(samePlan(t, k0));
        }
        // an interval whose second position crosses a range boundary (row 1 at n_pos 32769 = 32 x 1024 + 1 for S 16 and
        // 8 x 4096 + 1 for 64): one-row groups, the plan of today's E1 verify
        for (const ops::AttnDecodeFlags* fp : {&f0, &fa}) {
            ops::AttnDecodeProblem qx = e1(32766, 32767, 2, 16, 2, 0, true);
            const auto x = ops::chooseAttnDecode(qx, *fp, ks, ks.caps, p8s);
            CHECK(x.kind == ops::AttnDecodeKind::ws4 && x.grid.z == 2 && x.groups.count[0] == 1 && x.groups.count[1] == 1);
            qx.pos0_lo = qx.pos0_hi = -1;
            CHECK(samePlan(x, ops::chooseAttnDecode(qx, *fp, ks, ks.caps, p8s)));
            // the lower position alone would group (the rule must test the WHOLE interval)
            CHECK(ops::chooseAttnDecode(e1(32766, 32766, 2, 16, 2, 0, true), *fp, ks, ks.caps, p8s).grid.z == 1);
            CHECK(ops::chooseAttnDecode(e1(32767, 32767, 2, 16, 2, 0, true), *fp, ks, ks.caps, p8s).grid.z == 2);
        }
    }
    // (4b) the E1 groups key on the step's split count: 1022..1023 crosses an S 16 range (row 1 at n_pos 1025 for row 0 at
    // 1023) but no 64 range -> balance (S 16) one-row groups, precise (64) one 2-row group; as the known rows at 1023
    {
        ops::AttnDecodeFlags f0, fa;
        fa.attn_lsplit = A;
        const auto b16 = ops::chooseAttnDecode(e1(1022, 1023, 2, 16, 2, 0, true), fa, ks, ks.caps, p8s);
        CHECK(b16.grid.y == 16 && b16.kind == ops::AttnDecodeKind::ws4 && b16.grid.z == 2);
        CHECK(ops::chooseAttnDecode(known(1023, 2, 16, 2, 0, true), fa, ks, ks.caps, p8s).grid.z == 2);
        const auto p64 = ops::chooseAttnDecode(e1(1022, 1023, 2, 16, 2, 0, true), f0, ks, ks.caps, p8s);
        CHECK(p64.grid.y == 64 && p64.kind == ops::AttnDecodeKind::ws4g && p64.grid.z == 1 && p64.groups.count[0] == 2);
    }
    // (5) E1 off and the other unknown steps: unchanged
    {
        ops::AttnDecodeFlags fa;
        fa.attn_lsplit = A;
        // rows known (every non-E1 step): the interval fields change nothing
        for (const std::int32_t p0 : {5, 32766, 32767, 40000})
            for (const std::uint32_t n : {1u, 2u, 3u, 9u}) {
                const ops::AttnDecodeProblem qk = known(p0, n, 16, 2, 0, true);
                const auto before = ops::chooseAttnDecode(qk, fa, ks, ks.caps, p8s);
                ops::AttnDecodeProblem q2 = known(p0, n, 16, 2, 0, true);
                q2.rows_one_seq = true;
                q2.pos0_lo = p0 - 7;
                q2.pos0_hi = p0 + 3;
                CHECK(samePlan(ops::chooseAttnDecode(q2, fa, ks, ks.caps, p8s), before));
            }
        // unknown rows without one sequence, without an interval, with an empty / negative / too wide interval, or with the
        // one-query debug flag: one-row groups (today's)
        auto oneRow = [&](const ops::AttnDecodeProblem& q, const ops::AttnDecodeFlags& f) {
            const auto pl = ops::chooseAttnDecode(q, f, ks, ks.caps, p8s);
            return pl.kind == ops::AttnDecodeKind::ws4 && pl.grid.z == q.n;
        };
        ops::AttnDecodeProblem q = e1(40000, 40001, 2, 16, 2, 0, true);
        q.rows_one_seq = false;
        CHECK(oneRow(q, fa));
        CHECK(oneRow(e1(-1, -1, 2, 16, 2, 0, true), fa));
        CHECK(oneRow(e1(40001, 40000, 2, 16, 2, 0, true), fa));
        CHECK(oneRow(e1(-1, 40000, 2, 16, 2, 0, true), fa));
        CHECK(oneRow(e1(40000, 40000 + ops::kE1GroupSpan, 2, 16, 2, 0, true), fa));
        CHECK(!oneRow(e1(40000, 40000 + ops::kE1GroupSpan - 1, 2, 16, 2, 0, true), fa));
        ops::AttnDecodeFlags fd = fa;
        fd.dbg_flags = 4;
        CHECK(oneRow(e1(40000, 40001, 2, 16, 2, 0, true), fd));
        // the split count of an E1 step stays the step's (max_ctx against the rows-2 limit: over it 64)
        ops::AttnDecodeProblem qo = e1(40000, 40001, 2, 16, 2, 0, true);
        qo.max_ctx = 262208;
        const auto po = ops::chooseAttnDecode(qo, fa, ks, ks.caps, p8s);
        CHECK(po.grid.y == 64 && po.grid.z == 1 && nameOf(po.f) == "attn_ws4g1");
    }
    // (6) the R9700 (wsplit / ws4r path, rows_known only): the interval changes nothing there
    {
        static const kn::KernelTable kb = attnTable(false, false, true, true);
        static const kn::KernelTable kq = ws4rTable(kb, true);
        for (const std::uint32_t heads : {16u, 24u})
            for (const std::int32_t lo : {5000, 32766})
                for (const bool q8 : {false, true}) {
                    ops::AttnDecodeProblem q = e1(lo, lo + 1, 2, heads, heads == 16 ? 2u : 4u, 0, q8);
                    ops::AttnDecodeFlags fa;
                    fa.attn_lsplit = A;
                    const auto with = ops::chooseAttnDecode(q, fa, kq, kq.caps, p97);
                    q.pos0_lo = q.pos0_hi = -1;
                    CHECK(samePlan(with, ops::chooseAttnDecode(q, fa, kq, kq.caps, p97)));
                    CHECK(with.grid.z == 2);
                }
    }
    // (7) how often an Ornith E1 verify (2 rows, interval of 2 = one draft in flight) keeps its group, over every row-0 position
    for (const std::uint32_t ns : {16u, 64u}) {
        long long all = 0, grouped = 0;
        long long near[3] = {0, 0, 0};
        const std::int32_t marks[3] = {8192, 32768, 65536};
        for (std::int32_t lo = 1; lo < 262144; ++lo) {
            kn::AwGroups g{};
            std::uint32_t wd = 0;
            const bool one = ops::ws4E1Groups(2, lo, lo + 1, ns, 0, kn::kMaxVerifyRows, g, wd) == 1;
            ++all;
            grouped += one;
            for (int m = 0; m < 3; ++m)
                if (lo >= marks[m] - 2048 && lo < marks[m] + 2048 && !one) ++near[m];
        }
        std::printf("  testAttnDecodeE1Group: S %u: %lld of %lld row-0 intervals keep the 2-row group (%.3f %%); split %lld / %lld / %lld of 4096 "
                    "positions around 8192 / 32768 / 65536\n",
                    ns, grouped, all, 100.0 * static_cast<double>(grouped) / static_cast<double>(all), near[0], near[1], near[2]);
        CHECK(all - grouped == 2 * (262144 / (static_cast<long long>(ns) * 64)) - 1 || all - grouped == 2 * (262144 / (static_cast<long long>(ns) * 64)));
    }
    std::printf("  testAttnDecodeE1Group: %lld cases\n", cases);
}

}  // namespace

// v0252-e1fast (PLAN V1-93 P3): an E1 verify in FAST mode (q4 KV through attn_dq4, rows unknown on the host: positions on the
// device, AttnDecodeProblem::rows_one_seq with row 0's interval [pos0_lo, pos0_hi]) takes the split count the host would take
// for known rows (the v0252-8sq5 table / WHIRL_DQ4_MINKEYS / the dynamic rule) when that count is the same at EVERY position
// of the interval; else today's E1 grid (the fixed fd_splits). Before this item every E1 fast verify ran the fixed grid (256
// splits at 32768 keys where the table's 2-row pick is 8). Checks: (1) a one-position interval = the known-rows plan for the
// same rows (split count, kernel, grid); (2) a two-position interval takes the count only where both positions agree; (3) no
// interval / too wide / not one sequence: the fixed grid as before; (4) f16 / q8 KV (balance / precise) never changes.
void testAttnDq4E1() {
    const auto& p8s = whirl::device::k_profile_gfx1151;
    static const kn::KernelTable s8 = attnTable(true, true, true, false);
    auto k8 = std::make_unique<kn::KernelTable>(s8);
    k8->attn_dq4o = tag("attn_dq4o");
    const std::vector<std::int32_t> zeros(16, 0);
    ops::AttnDecodeProblem p;
    p.head_dim = 256;
    p.n_head = 16;
    p.n_head_kv = 2;
    p.max_ctx = 262144;
    p.kv_q4 = true;
    ops::AttnDecodeFlags f;
    f.fd_splits = 256;
    f.dq4_splits = ops::kDq4SplitsAuto;
    f.dq4_order = true;
    std::vector<std::int32_t> pos;
    auto knownPlan = [&](std::int32_t p0, std::uint32_t n, const ops::AttnDecodeFlags& fl) {
        pos.clear();
        for (std::uint32_t r = 0; r < n; ++r) pos.push_back(p0 + static_cast<std::int32_t>(r));
        ops::AttnDecodeProblem q = p;
        q.n = n;
        q.rows_known = true;
        q.row_pos = std::span<const std::int32_t>(pos);
        q.row_base = std::span<const std::int32_t>(zeros.data(), n);
        return ops::chooseAttnDecode(q, fl, *k8, k8->caps, p8s);
    };
    auto e1Plan = [&](std::int32_t lo, std::int32_t hi, std::uint32_t n, bool one_seq, const ops::AttnDecodeFlags& fl) {
        ops::AttnDecodeProblem q = p;
        q.n = n;
        q.rows_known = false;
        q.rows_one_seq = one_seq;
        q.pos0_lo = lo;
        q.pos0_hi = hi;
        return ops::chooseAttnDecode(q, fl, *k8, k8->caps, p8s);
    };
    ops::AttnDecodeProblem q0 = p;
    q0.n = 2;
    const std::uint32_t fixed = ops::chooseAttnDecode(q0, f, *k8, k8->caps, p8s).n_split;  // today's E1 grid (unknown rows)
    CHECK_EQ(fixed, 256u);
    long long one = 0, two_same = 0, two_diff = 0, bad = 0;
    std::vector<std::int32_t> keys_list;
    for (const std::int32_t b : {8192, 16384, 32768, 49152, 65536, 98304, 131072})
        for (std::int32_t d = -3; d <= 3; ++d) keys_list.push_back(b + d);
    for (const std::int32_t x : {3000, 20000, 40000, 70001, 120000, 200000}) keys_list.push_back(x);
    for (const ops::AttnDecodeFlags& fl : {f, [&] {
                                               ops::AttnDecodeFlags m = f;
                                               m.dq4_min_keys = 1024;
                                               return m;
                                           }()})
        for (const std::uint32_t n : {2u, 3u})
            for (const std::int32_t keys : keys_list) {
                const std::int32_t p0 = keys - static_cast<std::int32_t>(n);  // the last row at keys - 1
                // (1) one position: the known-rows plan
                const ops::AttnDecodePlan k1 = knownPlan(p0, n, fl), e = e1Plan(p0, p0, n, true, fl);
                const bool ok1 = e.n_split == k1.n_split && e.f == k1.f && e.grid.y == k1.grid.y && e.kind == k1.kind;
                bad += !ok1;
                one += 1;
                // (2) two positions (one MTP draft in flight): the known count where both agree, else the fixed grid
                const ops::AttnDecodePlan ka = knownPlan(p0 - 1, n, fl), kb = knownPlan(p0, n, fl), e2 = e1Plan(p0 - 1, p0, n, true, fl);
                const std::uint32_t want = ka.n_split == kb.n_split ? ka.n_split : fixed;
                bad += e2.n_split != want;
                (ka.n_split == kb.n_split ? two_same : two_diff) += 1;
                // (3) not one sequence / no interval / too wide: the fixed grid
                bad += e1Plan(p0, p0, n, false, fl).n_split != fixed;
                bad += e1Plan(-1, -1, n, true, fl).n_split != fixed;
                bad += e1Plan(p0 - ops::kE1GroupSpan, p0, n, true, fl).n_split != fixed;
            }
    // (4) f16 KV (balance / precise, the ws4 path): the interval never touches the dq4 rule
    {
        static const kn::KernelTable s8f = attnTable(true, true, false, false);
        ops::AttnDecodeProblem q = p;
        q.n = 2;
        q.kv_q4 = false;
        q.rows_one_seq = true;
        q.pos0_lo = 32766;
        q.pos0_hi = 32766;
        ops::AttnDecodeFlags f64 = f;
        f64.fd_splits = 64;
        ops::AttnDecodeProblem qn = q;
        qn.pos0_lo = -1;
        qn.pos0_hi = -1;
        CHECK_EQ(ops::chooseAttnDecode(q, f64, s8f, s8f.caps, p8s).n_split, ops::chooseAttnDecode(qn, f64, s8f, s8f.caps, p8s).n_split);
    }
    CHECK_EQ(bad, 0LL);
    CHECK(one > 0 && two_same > 0 && two_diff > 0);
    std::printf("  testAttnDq4E1: %lld one-position intervals, %lld two-position (%lld agree, %lld at a breakpoint), %lld bad\n", one,
                two_same + two_diff, two_same, two_diff, bad);
}
