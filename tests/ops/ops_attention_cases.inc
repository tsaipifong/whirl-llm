// arch025-n8b1: cases moved verbatim from tests/ops/ops_tests.cpp; included by whirl-ops-tests (tests/ops/ops_tests.cpp, same
// cases, same order) and by the module's isolation program tests/iso/ops.attention/ (ARCH-1 8.5).
// SPDX-License-Identifier: Apache-2.0

#include "whirl/device_profile.h"
#include "whirl/ops/attention.h"
#include "../test_check.h"
#include "ops_fakes.h"

#include <cstdint>
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
    // q4 KV with attn_dq4: one query per group on attn_dq4, 256 threads; the R9700 keeps the fixed grid
    p.kv_q4 = true;
    f.fd_splits = 256;
    pl = ops::chooseAttnDecode(p, f, k97, k97.caps, p97);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.kind == ops::AttnDecodeKind::dq4 && pl.grid.z == 4 && pl.block.x == 256 && pl.n_split == 256);
    // the 8060S (attn_group1 set): the dynamic split count for short rows (same partition), FAST-8S-1
    const kn::KernelTable k8 = attnTable(true, true, true, false);
    pl = ops::chooseAttnDecode(p, f, k8, k8.caps, p8s);
    CHECK(nameOf(pl.f) == "attn_dq4" && pl.n_split < 256 && pl.n_split == ops::dq4Splits(std::span<const std::int32_t>(pos, 4), 0, 256));
    // ... a profile row decides it, not the kernel set: the same kernels under the gfx1201 row keep the fixed grid
    CHECK(ops::chooseAttnDecode(p, f, k8, k8.caps, p97).n_split == 256);
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
// attn_kgf), with the shipped kernel's grid / block; q8 / q8v without the scratch, FC-1c (attn_kgs), q4 and the 8060S
// set keep their kernels. Fails before the port (test-first stub: attnKgVariant returns nullptr, chooseAttnPrefill
// never asks it).
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
        if (d != '8' && d != '6' && d != '4' && d != '2') return false;  // v0252-c11: NP 8 (attn_kg8 / attn_kgf8) too
        if (s.block.x != 64u * static_cast<unsigned>(d - '0')) return false;
    }
    return true;
}

void testAttnKgVariant() {
    const kn::KernelTable kv = kgTable(true), k0 = kgTable(false);
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
        // (6) q4 KV on the gfx1201 set (no attn_kg*_q4: FC-1c attn_kgs over dequantized ranges): no variant, every var
        kn::KernelTable kq4 = kv;
        kq4.attn_kg6 = kq4.attn_kg4 = kq4.attn_kg2 = nullptr;
        kq4.kv_dq_rows = nullptr;
        p.dq_rows = 8192;
        p.dq_ptab = true;
        p.dq_st_ml = true;
        for (const std::uint32_t var : {0u, 1u, 3u}) {
            const auto pl = plan(var, kq4);
            std::size_t kgs = 0, other = 0;
            for (const auto& st : pl.steps)
                if (st.op == ops::AttnStep::Op::attn) (nameOf(st.f) == "attn_kgs" + s ? kgs : other) += 1;
            CHECK(kgs == 1 && other == 0 && kgShapesMatch(pl));
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
        const kn::KernelTable k8 = attnTable(true, true, true, false);
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
}

// v0252-c11 (C11; DISCOVERY v4 C3-1): the NP-8 attn_kg for GQA 8 (Ornith 16 / 2) behind attn_kg8 (WHIRL_ATTN_KG8).
// ops::attnKgf is exhaustive (np 8 / 6 / 4 / 2 written out, any other np -> nullptr, never attn_kgf2 under another
// NP's grid / block); chooseAttnPrefill takes attn_kg8 (plain f16 KV from the cache) or attn_kgf8 (Q8P) only with the
// flag on, the kernel in the code object, the GQA group and the launch's head range multiples of 8, grid
// (ceil(n / 16), heads / 8), block 512; before attn_kg_var; never on q8 / q8v without the scratch, FC-1c (attn_kgs), q4
// or the 8060S set; flag off (the default) = the shipped plan in every case. Fails before the kernel commit (stub
// attnKgf = the shipped ternary, no NP-8 branch in chooseAttnPrefill).
void testAttnKg8() {
    kn::KernelTable k8 = kgTable(true);  // gfx1201 with C1's variants and C11's attn_kg8 / attn_kgf8
    k8.attn_kg8 = tag("attn_kg8");
    k8.attn_kgf8 = tag("attn_kgf8");
    const kn::KernelTable k0 = kgTable(true);  // no attn_kg8 in the code object
    // (1) attnKgf alone: every NP written out, any other np nullptr
    for (const std::uint32_t np : {8u, 6u, 4u, 2u}) CHECK(nameOf(ops::attnKgf(k8, np)) == "attn_kgf" + std::to_string(np));
    for (const std::uint32_t np : {0u, 1u, 3u, 5u, 12u, 16u}) CHECK(ops::attnKgf(k8, np) == nullptr);
    CHECK(ops::attnKgf(k0, 8) == nullptr);
    // NP 8 has no attn_kg_var variant (C1's selector stays nullptr for np 8)
    for (const std::uint32_t var : {0u, 1u, 3u}) CHECK(ops::attnKgVariant(var, k8, 8) == nullptr);
    struct Shape {
        std::uint32_t heads, kv, np_off, np_on;  // shipped NP, NP with attn_kg8 on
    };
    // Ornith 16 / 2 (GQA 8), 24 / 3 (GQA 8), 32 / 2 (GQA 16), Swift 24 / 4 (GQA 6), 4 / 2 (GQA 2), 8 / 2 (GQA 4)
    for (const Shape sh : {Shape{16, 2, 4, 8}, Shape{24, 3, 4, 8}, Shape{32, 2, 4, 8}, Shape{24, 4, 6, 6}, Shape{4, 2, 2, 2}, Shape{8, 2, 4, 4}}) {
        ops::AttnPrefillProblem p;
        p.n = 512;
        p.pos0 = 1536;
        p.head_dim = 256;
        p.n_head = sh.heads;
        p.n_head_kv = sh.kv;
        p.scale = 1.0f / 16.0f;
        auto plan = [&](bool on, std::uint32_t var, const kn::KernelTable& k) {
            ops::AttnPrefillFlags f;
            f.attn_kg8 = on;
            f.attn_kg_var = var;
            return ops::chooseAttnPrefill(p, f, k);
        };
        auto one = [&](const ops::AttnPrefillPlan& pl, const std::string& want, std::uint32_t np) {
            return pl.steps.size() == 1 && nameOf(pl.steps[0].f) == want && !pl.steps[0].from_copy && !pl.steps[0].ranged &&
                   pl.steps[0].grid.x == 32 && pl.steps[0].grid.y == sh.heads / np && pl.steps[0].block.x == 64 * np && pl.steps[0].head0 == 0 &&
                   kgShapesMatch(pl);
        };
        const std::string so = std::to_string(sh.np_off), sn = std::to_string(sh.np_on);
        const bool takes8 = sh.np_on == 8;
        // (2) f16 KV from the cache: off = shipped (and C1's variant of the shipped NP); on = attn_kg8 where it fits, before var
        CHECK(one(plan(false, 0, k8), "attn_kg" + so, sh.np_off));
        CHECK(one(plan(false, 3, k8), "attn_kgk" + so, sh.np_off));
        CHECK(one(plan(true, 0, k8), "attn_kg" + sn, sh.np_on));
        CHECK(one(plan(true, 3, k8), (takes8 ? "attn_kg" : "attn_kgk") + sn, sh.np_on));
        CHECK(one(plan(true, 1, k8), (takes8 ? "attn_kg" : "attn_kgv") + sn, sh.np_on));
        // no attn_kg8 in the code object: the flag changes nothing
        CHECK(one(plan(true, 0, k0), "attn_kg" + so, sh.np_off));
        CHECK(one(plan(true, 3, k0), "attn_kgk" + so, sh.np_off));
        // (3) Q8P (keys inside the dequant scratch): kv_dq_rows once, then attn_kgf8 (on) / attn_kgf<np> (off)
        p.kv_q8 = true;
        p.dq_rows = 8192;
        p.dq_ptab = true;
        p.dq_st_ml = true;
        for (const bool on : {false, true}) {
            const auto pl = plan(on, 0, k8);
            const std::uint32_t np = on ? sh.np_on : sh.np_off;
            CHECK(pl.steps.size() == 2 && pl.steps[0].op == ops::AttnStep::Op::dq_rows && pl.steps[0].nrows == 2048 &&
                  nameOf(pl.steps[1].f) == "attn_kgf" + std::to_string(np) && pl.steps[1].from_copy && !pl.steps[1].ranged &&
                  pl.steps[1].grid.x == 32 && pl.steps[1].grid.y == sh.heads / np && pl.steps[1].block.x == 64 * np && kgShapesMatch(pl));
        }
        {  // Q8P with attn_kg8 on and var 3: NP 8 first; without attn_kgf8 in the table: the shipped NP with the variant
            kn::KernelTable kn8 = k8;
            kn8.attn_kgf8 = nullptr;
            const auto pa = plan(true, 3, k8), pb = plan(true, 3, kn8);
            CHECK(pa.steps.size() == 2 && nameOf(pa.steps[1].f) == (takes8 ? "attn_kgf8" : "attn_kgk" + so) && kgShapesMatch(pa));
            CHECK(pb.steps.size() == 2 && nameOf(pb.steps[1].f) == "attn_kgk" + so && pb.steps[1].block.x == 64 * sh.np_off && kgShapesMatch(pb));
        }
        // (4) q8 / q8h past the scratch (FC-1c): attn_kgs<shipped NP> over key ranges, flag on or off (no kgs8)
        p.pos0 = 20000;
        for (const bool on : {false, true}) {
            const auto pl = plan(on, 0, k8);
            std::size_t kgs = 0, other = 0;
            for (const auto& st : pl.steps)
                if (st.op == ops::AttnStep::Op::attn) (st.ranged && nameOf(st.f).rfind("attn_kgs", 0) == 0 && nameOf(st.f).back() != '8' ? kgs : other) += 1;
            CHECK(kgs == 3 && other == 0 && kgShapesMatch(pl));
        }
        // (5) q8 / q8v from the cache (no dequant scratch): the shipped attn_kg<np> (_q8 / _q8v in a real table), flag on or off
        p.pos0 = 1536;
        p.dq_rows = 0;
        p.dq_ptab = false;
        p.dq_st_ml = false;
        for (const bool on : {false, true}) CHECK(one(plan(on, 0, k8), "attn_kg" + so, sh.np_off));
        // (6) q4 KV on the gfx1201 set (no attn_kg*_q4): FC-1c attn_kgs, flag on or off
        {
            kn::KernelTable kq4 = k8;
            kq4.attn_kg6 = kq4.attn_kg4 = kq4.attn_kg2 = nullptr;
            kq4.kv_dq_rows = nullptr;
            p.dq_rows = 8192;
            p.dq_ptab = true;
            p.dq_st_ml = true;
            for (const bool on : {false, true}) {
                const auto pl = plan(on, 0, kq4);
                std::size_t kgs = 0, other = 0;
                for (const auto& st : pl.steps)
                    if (st.op == ops::AttnStep::Op::attn) (nameOf(st.f).rfind("attn_kgs", 0) == 0 && nameOf(st.f).back() != '8' ? kgs : other) += 1;
                CHECK(kgs == 1 && other == 0 && kgShapesMatch(pl));
            }
        }
        p.kv_q8 = false;
        p.dq_rows = 0;
        p.dq_ptab = false;
        p.dq_st_ml = false;
        // (7) head ranges (Windows TDR budget): 2 ranges -> heads / 2 per launch (NP 8 where heads / 2 % 8 == 0);
        // 4 ranges (keys past 3 x 128k) -> heads / 4 (Ornith: 4 heads per launch -> the shipped NP 4 even with the flag)
        p.n = 4096;
        for (const std::uint32_t pos0 : {131072u, 393216u}) {
            p.pos0 = pos0;
            const std::uint32_t parts = pos0 == 131072u ? 2 : 4;
            const std::uint32_t hp = sh.heads / parts;
            if (sh.heads % parts != 0) continue;
            const std::uint32_t npw = (takes8 && hp % 8 == 0) ? 8 : sh.np_off;
            if (hp % sh.np_off != 0) continue;
            const auto pl = plan(true, 0, k8);
            bool ok = pl.steps.size() == parts;
            for (std::uint32_t q = 0; ok && q < pl.steps.size(); ++q)
                ok = nameOf(pl.steps[q].f) == "attn_kg" + std::to_string(npw) && pl.steps[q].head0 == q * hp && pl.steps[q].grid.y == hp / npw &&
                     pl.steps[q].block.x == 64 * npw;
            CHECK(ok && kgShapesMatch(pl));
        }
        p.n = 512;
        p.pos0 = 1536;
    }
    // (8) the 8060S set (no attn_kg / attn_kg8): WHIRL_ATTN_KG8 changes nothing, f16 and q8 KV, Ornith and Swift shapes
    {
        const kn::KernelTable ks = attnTable(true, true, true, false);
        for (const std::uint32_t heads : {16u, 24u}) {
            ops::AttnPrefillProblem p;
            p.n = 512;
            p.head_dim = 256;
            p.n_head = heads;
            p.n_head_kv = heads == 16 ? 2 : 4;
            p.scale = 1.0f / 16.0f;
            for (const bool q8 : {false, true})
                for (const std::uint32_t pos0 : {0u, 20000u}) {
                    p.kv_q8 = q8;
                    p.pos0 = pos0;
                    p.dq_rows = q8 ? 8192 : 0;
                    p.dq_ptab = p.dq_st_ml = q8;
                    ops::AttnPrefillFlags f0, f1;
                    f1.attn_kg8 = true;
                    const auto a = ops::chooseAttnPrefill(p, f0, ks), b = ops::chooseAttnPrefill(p, f1, ks);
                    bool same = a.steps.size() == b.steps.size() && !a.steps.empty();
                    for (std::size_t i = 0; same && i < a.steps.size(); ++i) {
                        const auto &x = a.steps[i], &y = b.steps[i];
                        same = x.op == y.op && x.f == y.f && x.grid.x == y.grid.x && x.grid.y == y.grid.y && x.block.x == y.block.x &&
                               x.from_copy == y.from_copy && x.ranged == y.ranged && x.r0 == y.r0 && x.r1 == y.r1 && x.head0 == y.head0;
                    }
                    CHECK(same);
                }
        }
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

}  // namespace
