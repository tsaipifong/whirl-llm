// whirl-kernel-test family "cascade" (feat026-cascade-r1, review C1): the cascade (shared-prefix) decode / verify
// attention on the GPU against the plain decode path on the same paged KV.
// SPDX-License-Identifier: Apache-2.0
//
// For every case: S sequences x r rows (r > 1: MTP verify rows at consecutive positions), random paged KV whose first
// SP page-table entries are the same physical pages in every sharing sequence, random q / gate.
//   plain   = ops::chooseAttnDecode on the real rows, launched like Model::attnBlock, + attn_combine;
//   cascade = ops::planCascade with the REAL group cap (ops::cascadeGroupCap of this code object) + ops::cascadeRows +
//             whirl::cascade::launchRows / launchDecode (the launcher Model::cascadeBegin / cascadeDecode call) +
//             attn_combine over the suffix plan's splits.
// Checks (balance class: the summation order and the f16 rounding of P change, so not bitwise):
//   * the plan (groups, shared keys) and the prefix kernel chosen (gfx1201 attn_wsplit1 / attn_wsplit2, gfx1151
//     attn_ws4g CG 1..3) are the ones the case expects;
//   * attn_cascade_rows output == suffix positions pos - P, suffix page-table offsets base + P / 256, prefix
//     positions P - 1 (-1 outside a group), exact;
//   * cascade vs plain: |d| <= 2e-3 * scale + 1e-6 (scale = sum p |v| / l * gate) and NMSE <= 5e-4
//     (progress\feat026-cascade-r1.PRIOR_ART.md: FlashInfer 1e-3 / 1e-3, llama.cpp flash-attn NMSE 5e-4);
//   * cascade and plain vs a CPU double reference at rtol 1e-2 (the test_attn.cpp bound for attn_wsplit vs CPU);
//   * gate (the split kernels' skip argument): gate -> 0 is bitwise equal to no gate; gate -> 1 leaves the prefix
//     partials and the suffix splits >= 1 untouched (Model::cascadeDecode never takes the cascade with a gate set);
//   * the gfx1201 MTP limit (review C2): two sequences of 4 rows exceed the 5-row cap, the plan is empty.
// Cases cover f16 / int8 (q8, q8v) KV, GQA 6 (27B) and 8 (Ornith), the n_split 64 / 128 layouts, a prefix past 4096
// keys (split size 128), a prefix cut by a row's own position, a batch with a row outside every group, MTP rows,
// gfx1151 WHIRL_ATTN_WS4=0 (suffix attn_wsplit1, prefix attn_ws4g) and large logits (Q sigma 4: the merge rescales).

#include "kt.h"

#include "whirl/cascade_launch.h"
#include "whirl/device_profile.h"
#include "whirl/ops/attention.h"
#include "whirl/ops/cascade.h"
#include "whirl/ops/cascade_kernels.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace kt {

namespace {

namespace ops = whirl::ops;
constexpr int kHd = 256, kPg = 256;
constexpr int kTabStride = 64;  // page-table entries per sequence (16k keys)

enum class Kv { f16, q8, q8v };
const char* kvName(Kv f) { return f == Kv::f16 ? "f16" : f == Kv::q8 ? "q8" : "q8v"; }
wk::KvFormat kvFormat(Kv f) { return f == Kv::f16 ? wk::KvFormat::f16 : f == Kv::q8 ? wk::KvFormat::q8 : wk::KvFormat::q8v; }
bool kQ(Kv f) { return f == Kv::q8; }
bool vQ(Kv f) { return f != Kv::f16; }

// physical pages with random K / V in the kernels' layout, and the values the kernels read (f16, or f16(q * s))
struct Pool {
    Kv fmt;
    int npages = 0, nkv = 0;
    std::vector<float> kd, vd;  // dequantized values, element ((prow * nkv + h) * hd + d)
    Buf k, v, ks, vs;
    Pool(Ctx& c, Kv f, int np, int n_kv) : fmt(f), npages(np), nkv(n_kv) {
        const std::size_t n = static_cast<std::size_t>(np) * kPg * n_kv * kHd;
        kd.resize(n);
        vd.resize(n);
        auto fill = [&](bool q, std::vector<float>& val, Buf& data, Buf& sc) {
            if (!q) {
                const std::vector<float> x = c.randn(n, 0.7f);
                std::vector<std::uint16_t> h(n);
                for (std::size_t i = 0; i < n; ++i) {
                    h[i] = f2h(x[i]);
                    val[i] = h2f(h[i]);
                }
                data = Buf(h);
                sc = Buf(4);
                return;
            }
            const std::vector<int> qi = c.randi(n, -127, 127);
            const std::vector<float> su = c.randu(n / 32, 0.004f, 0.016f);
            std::vector<std::int8_t> q8(n);
            std::vector<std::uint16_t> s(n / 32);
            for (std::size_t g = 0; g < n / 32; ++g) s[g] = f2h(su[g]);
            for (std::size_t i = 0; i < n; ++i) {
                q8[i] = static_cast<std::int8_t>(qi[i]);
                val[i] = h2f(d2h(static_cast<double>(q8[i]) * h2f(s[i >> 5])));
            }
            data = Buf(q8);
            sc = Buf(s);
        };
        fill(kQ(f), kd, k, ks);
        fill(vQ(f), vd, v, vs);
    }
};

struct Seq {
    int pos0 = 0, rows = 1;
    bool shares = true;  // its first SP entries are the shared pages
};
struct ExpGroup {
    int first, count, keys;
};
struct Case {
    std::string name;
    int n_head = 24, n_kv = 4;
    int sp = 4;  // shared pages
    std::vector<Seq> seqs;
    std::vector<ExpGroup> groups;  // expected plan (empty: the plan must be empty)
    ops::CascadePrefixKind prefix = ops::CascadePrefixKind::none;
    int cg = 0;                    // ws4g: expected CG
    std::uint32_t fd = 64, max_ctx = 4096;
    bool ws4 = true, wide = true;
    float qsig = 1.0f;
    bool gate_checks = false;
};

const char* prefixName(ops::CascadePrefixKind k) {
    switch (k) {
    case ops::CascadePrefixKind::none: return "none";
    case ops::CascadePrefixKind::wsplit1: return "attn_wsplit1";
    case ops::CascadePrefixKind::wsplit2: return "attn_wsplit2";
    case ops::CascadePrefixKind::ws4g: return "attn_ws4g";
    }
    return "?";
}

double nmse(const std::vector<float>& a, const std::vector<float>& b) {
    double e = 0, s = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        e += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
        s += static_cast<double>(b[i]) * b[i];
    }
    return s > 0 ? e / s : e;
}

// device buffers shared by all cases (sized for 32 rows x 256 splits x 24 heads)
struct Scratch {
    Buf ml, acc, cml, cacc, ml2, acc2, rows, out, out2, out3, gz, g1;
    Scratch() {
        const std::size_t m = static_cast<std::size_t>(32) * 256 * 24;
        ml = Buf(m * 2 * 4);
        acc = Buf(m * kHd * 4);
        ml2 = Buf(m * 2 * 4);
        acc2 = Buf(m * kHd * 4);
        cml = Buf(m * 2 * 4);
        cacc = Buf(m * kHd * 4);
        rows = Buf(3 * 32 * 4);
        out = Buf(static_cast<std::size_t>(32) * 24 * kHd * 4);
        out2 = Buf(out.bytes());
        out3 = Buf(out.bytes());
        gz = Buf(std::vector<int>{0});
        g1 = Buf(std::vector<int>{1});
    }
};

void runCase(Ctx& c, Scratch& sc, const wk::KernelTable& kt_, Kv fmt, const Case& cs) {
    const std::string tag = " [" + cs.name + ", kv " + kvName(fmt) + "]";
    const int H = cs.n_head, HKV = cs.n_kv, grp = H / HKV, S = static_cast<int>(cs.seqs.size());
    // page tables: sequence s at base s * kTabStride; shared entries -> the same physical pages
    int n_rows = 0, need_pages = cs.sp;
    for (const Seq& q : cs.seqs) {
        n_rows += q.rows;
        const int np = (q.pos0 + q.rows + kPg - 1) / kPg;
        need_pages += np;
    }
    std::vector<int> phys(static_cast<std::size_t>(need_pages));
    std::iota(phys.begin(), phys.end(), 0);
    std::shuffle(phys.begin(), phys.end(), c.rng);
    std::vector<std::vector<std::int32_t>> tabs(static_cast<std::size_t>(S));
    std::vector<int> ptab_h(static_cast<std::size_t>(S) * kTabStride, 0);
    int next = cs.sp;
    for (int s = 0; s < S; ++s) {
        const Seq& q = cs.seqs[static_cast<std::size_t>(s)];
        const int np = (q.pos0 + q.rows + kPg - 1) / kPg;
        for (int i = 0; i < np; ++i) {
            const int pg = q.shares && i < cs.sp ? phys[static_cast<std::size_t>(i)] : phys[static_cast<std::size_t>(next++)];
            tabs[static_cast<std::size_t>(s)].push_back(pg);
            ptab_h[static_cast<std::size_t>(s) * kTabStride + i] = pg;
        }
    }
    Pool pool(c, fmt, need_pages, HKV);
    Buf ptab(ptab_h);
    std::vector<std::int32_t> rp, rb;
    for (int s = 0; s < S; ++s)
        for (int j = 0; j < cs.seqs[static_cast<std::size_t>(s)].rows; ++j) {
            rp.push_back(cs.seqs[static_cast<std::size_t>(s)].pos0 + j);
            rb.push_back(s * kTabStride);
        }
    const std::uint32_t n = static_cast<std::uint32_t>(n_rows);
    Buf dpos(std::vector<int>(rp.begin(), rp.end())), dbase(std::vector<int>(rb.begin(), rb.end()));
    wk::KvArgs kva;
    kva.k = pool.k.p();
    kva.v = pool.v.p();
    kva.ks = kQ(fmt) ? pool.ks.p() : 0;
    kva.vs = vQ(fmt) ? pool.vs.p() : 0;
    kva.ptab = ptab.p();
    kva.kvbase = dbase.p();
    kva.tab0 = 0;
    kva.win = 0;
    const std::vector<float> q = c.randn(static_cast<std::size_t>(n) * H * 2 * kHd, cs.qsig);
    Buf dq(q);
    const float scale = 1.0f / 16.0f;

    ops::AttnDecodeProblem ap;
    ap.n = n;
    ap.head_dim = kHd;
    ap.n_head = static_cast<std::uint32_t>(H);
    ap.n_head_kv = static_cast<std::uint32_t>(HKV);
    ap.max_ctx = cs.max_ctx;
    ap.kv_q4 = false;
    ap.rows_known = true;
    ap.row_pos = std::span<const std::int32_t>(rp.data(), n);
    ap.row_base = std::span<const std::int32_t>(rb.data(), n);
    ap.win = 0;
    ap.o_on_gemv = false;
    ops::AttnDecodeFlags af;
    af.attn_ws4 = cs.ws4;
    af.attn_wide = cs.wide;
    af.fd_splits = cs.fd;
    af.prec_dec = false;
    const whirl::device::Profile& prof = whirl::device::profile(c.mod.arch());
    const std::size_t on = static_cast<std::size_t>(n) * H * kHd;

    // ---- the plan with the real cap
    ops::CascadePrefixProblem cp;
    cp.head_dim = kHd;
    cp.n_head = ap.n_head;
    cp.n_head_kv = ap.n_head_kv;
    const std::uint32_t cap = ops::cascadeGroupCap(cp, kt_, kt_.caps);
    if (cap < 2 || (cs.prefix == ops::CascadePrefixKind::wsplit2 && kt_.attn_wsplit2 == nullptr)) {
        // the expected plan assumes the full kernel set of the format (gfx1151 q8v: no attn_ws4g variant)
        c.rep.skip("cascade", cs.name + " kv " + kvName(fmt), "no multi-row prefix kernel for this KV format / GQA (cascade off by design)");
        return;
    }
    ops::CascadeConfig cc;
    cc.on = true;
    cc.min_keys = 256;
    const ops::CascadePages pages = [&tabs](std::int32_t b) -> std::span<const std::int32_t> {
        const std::size_t s = static_cast<std::size_t>(b / kTabStride);
        return s < tabs.size() ? std::span<const std::int32_t>(tabs[s]) : std::span<const std::int32_t>{};
    };
    const ops::CascadePlan plan = ops::planCascade(ap.row_pos, ap.row_base, pages, cc, cap);
    {
        bool ok = plan.groups.size() == cs.groups.size();
        for (std::size_t g = 0; ok && g < cs.groups.size(); ++g)
            ok = static_cast<int>(plan.groups[g].first) == cs.groups[g].first && static_cast<int>(plan.groups[g].count) == cs.groups[g].count &&
                 static_cast<int>(plan.groups[g].keys) == cs.groups[g].keys;
        Result r;
        r.name = "plan (cap " + std::to_string(cap) + ", " + std::to_string(plan.groups.size()) + " group(s))" + tag;
        r.kind = Kind::invariant;
        r.n = 1;
        r.mismatches = ok ? 0 : 1;
        r.pass = ok;
        c.rep.add(r);
        if (!ok || !plan.any()) return;  // the empty-plan cases end here (documented limit)
    }

    // ---- plain path (Model::attnBlock's decode branch without the cascade)
    sc.ml.zero();
    sc.acc.zero();
    const ops::AttnDecodePlan pl = ops::chooseAttnDecode(ap, af, kt_, kt_.caps, prof);
    if (pl.kind == ops::AttnDecodeKind::split)
        hip::launch(pl.f, pl.grid, pl.block, 0, c.s, dq.p(), kva, sc.ml.p(), sc.acc.p(), H, HKV, kHd, 2 * kHd, dpos.p(), scale, 1, DevPtr{0});
    else
        hip::launch(pl.f, pl.grid, pl.block, 0, c.s, dq.p(), kva, sc.ml.p(), sc.acc.p(), H, HKV, 2 * kHd, dpos.p(), scale, DevPtr{0}, pl.groups);
    hip::launch(kt_.attn_combine, pl.combine_grid, pl.combine_block, 0, c.s, sc.ml.p(), sc.acc.p(), dq.p(), sc.out.p(), H, kHd, 2 * kHd,
                static_cast<int>(pl.n_split));
    c.sync();
    const std::vector<float> o_plain = sc.out.down<float>(on);

    // ---- cascade path (the production launcher)
    const ops::CascadeRows tab = ops::cascadeRows(plan, ap.row_pos, ap.row_base);
    whirl::cascade::Bufs b;
    b.qf = dq.p();
    b.part_ml = sc.ml2.p();
    b.part_acc = sc.acc2.p();
    b.cas_ml = sc.cml.p();
    b.cas_acc = sc.cacc.p();
    b.cas_rows = sc.rows.p();
    b.gate = 0;
    b.stream = c.s;
    sc.rows.fill(0xff);
    whirl::cascade::launchRows(kt_, tab, n, b);
    c.sync();
    {
        const std::vector<int> got = sc.rows.down<int>(96);
        std::vector<int> want(96, -1);
        for (std::uint32_t r = 0; r < n; ++r) {
            int P = 0;
            for (const ExpGroup& g : cs.groups)
                if (static_cast<int>(r) >= g.first && static_cast<int>(r) < g.first + g.count) P = g.keys;
            want[r] = rp[r] - P;
            want[32 + r] = rb[r] + P / kPg;
            want[64 + r] = P > 0 ? P - 1 : -1;
        }
        // rows >= n are not written (0xff bytes = -1)
        c.rep.add(cmpExact("attn_cascade_rows: suffix pos / page-table offset, prefix pos" + tag, got, want));
    }
    sc.ml2.zero();
    sc.acc2.zero();
    ops::AttnDecodePlan cpl;
    const bool took = whirl::cascade::launchDecode(plan, ap, af, kt_, kt_.caps, prof, kva, scale, b, cpl);
    {
        // the prefix kernel the launcher used (same choice function, same suffix split count)
        ops::CascadePrefixProblem cq = cp;
        cq.n_split = cpl.n_split;
        const ops::CascadePrefixPlan pp = took ? ops::chooseCascadePrefix(plan, cq, kt_, kt_.caps) : ops::CascadePrefixPlan{};
        bool ok = took && pp.kind == cs.prefix;
        if (ok && pp.kind == ops::CascadePrefixKind::ws4g) ok = cs.cg >= 1 && cs.cg <= 3 && pp.f == kt_.attn_ws4g[cs.cg - 1];
        if (ok && pp.kind == ops::CascadePrefixKind::wsplit2) ok = pp.f == kt_.attn_wsplit2;
        if (ok && pp.kind == ops::CascadePrefixKind::wsplit1) ok = pp.f == kt_.attn_wsplit1;
        Result r;
        r.name = std::string("prefix kernel ") + prefixName(cs.prefix) + (cs.cg ? std::to_string(cs.cg) : std::string()) + ", suffix n_split " +
                 std::to_string(cpl.n_split) + tag;
        r.kind = Kind::invariant;
        r.n = 1;
        r.mismatches = ok ? 0 : 1;
        r.pass = ok;
        r.note = std::string("got ") + prefixName(pp.kind) + (took ? "" : " (launcher refused)");
        c.rep.add(r);
        if (!took) return;
    }
    hip::launch(kt_.attn_combine, cpl.combine_grid, cpl.combine_block, 0, c.s, sc.ml2.p(), sc.acc2.p(), dq.p(), sc.out2.p(), H, kHd, 2 * kHd,
                static_cast<int>(cpl.n_split));
    c.sync();
    const std::vector<float> o_cas = sc.out2.down<float>(on);

    // ---- CPU reference (double, the values the kernels read; q not rounded: rtol 1e-2)
    std::vector<double> ref(on), rsc(on);
    for (std::uint32_t r = 0; r < n; ++r) {
        const std::size_t s = static_cast<std::size_t>(rb[r] / kTabStride);
        const int np = rp[r] + 1;
        for (int h = 0; h < H; ++h) {
            const int kvh = h / grp;
            const float* qv = q.data() + (static_cast<std::size_t>(r) * H + h) * 2 * kHd;
            std::vector<double> w(static_cast<std::size_t>(np));
            double m = -INFINITY;
            for (int p = 0; p < np; ++p) {
                const std::size_t prow = static_cast<std::size_t>(tabs[s][static_cast<std::size_t>(p / kPg)]) * kPg + p % kPg;
                const float* kr = pool.kd.data() + (prow * HKV + kvh) * kHd;
                double d = 0;
                for (int i = 0; i < kHd; ++i) d += static_cast<double>(qv[i]) * kr[i];
                w[static_cast<std::size_t>(p)] = d * scale;
                m = std::max(m, w[static_cast<std::size_t>(p)]);
            }
            double l = 0;
            for (double& x : w) {
                x = std::exp(x - m);
                l += x;
            }
            std::vector<double> o(kHd, 0.0), a(kHd, 0.0);
            for (int p = 0; p < np; ++p) {
                const std::size_t prow = static_cast<std::size_t>(tabs[s][static_cast<std::size_t>(p / kPg)]) * kPg + p % kPg;
                const float* vr = pool.vd.data() + (prow * HKV + kvh) * kHd;
                const double wp = w[static_cast<std::size_t>(p)];
                for (int d = 0; d < kHd; ++d) {
                    o[static_cast<std::size_t>(d)] += wp * vr[d];
                    a[static_cast<std::size_t>(d)] += wp * std::fabs(vr[d]);
                }
            }
            for (int d = 0; d < kHd; ++d) {
                const double g = 1.0 / (1.0 + std::exp(-static_cast<double>(qv[kHd + d])));
                const std::size_t oi = (static_cast<std::size_t>(r) * H + h) * kHd + d;
                ref[oi] = o[static_cast<std::size_t>(d)] / l * g;
                rsc[oi] = a[static_cast<std::size_t>(d)] / l * g;
            }
        }
    }
    std::vector<double> plain_d(o_plain.begin(), o_plain.end());
    Result rc = cmpTol("cascade vs plain (2e-3 x scale)" + tag, o_cas, plain_d, rsc, 2e-3, 1e-6);
    const double e = nmse(o_cas, o_plain);
    rc.note += " nmse " + std::to_string(e);
    if (e > 5e-4) rc.pass = false;
    c.rep.add(rc);
    c.rep.add(cmpTol("cascade vs CPU" + tag, o_cas, ref, rsc, 1e-2, 1e-4));
    c.rep.add(cmpTol("plain vs CPU" + tag, o_plain, ref, rsc, 1e-2, 1e-4));

    if (!cs.gate_checks) return;
    // ---- gate -> 0: bitwise the no-gate result
    {
        whirl::cascade::Bufs bz = b;
        bz.gate = sc.gz.p();
        sc.ml2.zero();
        sc.acc2.zero();
        ops::AttnDecodePlan zp;
        const bool t2 = whirl::cascade::launchDecode(plan, ap, af, kt_, kt_.caps, prof, kva, scale, bz, zp);
        hip::launch(kt_.attn_combine, zp.combine_grid, zp.combine_block, 0, c.s, sc.ml2.p(), sc.acc2.p(), dq.p(), sc.out3.p(), H, kHd, 2 * kHd,
                    static_cast<int>(zp.n_split));
        c.sync();
        Result r = cmpExact("gate -> 0 == no gate (bits)" + tag, sc.out3.down<float>(on), o_cas, Kind::invariant);
        if (!t2) r.pass = false;
        c.rep.add(r);
    }
    // ---- gate -> 1: the prefix and suffix kernels skip (prefix partials and suffix splits >= 1 keep their bytes)
    {
        whirl::cascade::Bufs b1 = b;
        b1.gate = sc.g1.p();
        const std::size_t mlw = static_cast<std::size_t>(n) * cpl.n_split * H * 2, accw = static_cast<std::size_t>(n) * cpl.n_split * H * kHd;
        sc.ml2.fill(0x55);
        sc.acc2.fill(0x55);
        sc.cml.fill(0x55);
        sc.cacc.fill(0x55);
        ops::AttnDecodePlan gp;
        const bool t3 = whirl::cascade::launchDecode(plan, ap, af, kt_, kt_.caps, prof, kva, scale, b1, gp);
        c.sync();
        const std::vector<std::uint32_t> cm = sc.cml.down<std::uint32_t>(mlw), ca = sc.cacc.down<std::uint32_t>(accw);
        const std::vector<std::uint32_t> pm = sc.ml2.down<std::uint32_t>(mlw), pa = sc.acc2.down<std::uint32_t>(accw);
        std::size_t bad = 0, cnt = 0;
        for (const std::uint32_t x : cm) bad += x != 0x55555555u ? 1 : 0;
        for (const std::uint32_t x : ca) bad += x != 0x55555555u ? 1 : 0;
        cnt += cm.size() + ca.size();
        // suffix splits >= 1 (the merge only rewrites split 0)
        for (std::uint32_t r = 0; r < n; ++r)
            for (std::uint32_t sp = 1; sp < cpl.n_split; ++sp)
                for (int h = 0; h < H; ++h) {
                    const std::size_t i = (static_cast<std::size_t>(r) * cpl.n_split + sp) * H + h;
                    bad += pm[2 * i] != 0x55555555u || pm[2 * i + 1] != 0x55555555u;
                    bad += pa[i * kHd] != 0x55555555u || pa[i * kHd + kHd - 1] != 0x55555555u;
                    cnt += 4;
                }
        Result r;
        r.name = "gate -> 1: prefix / suffix split kernels write nothing" + tag;
        r.kind = Kind::invariant;
        r.n = cnt;
        r.mismatches = bad + (t3 ? 0 : 1);
        r.pass = r.mismatches == 0;
        c.rep.add(r);
    }
}

}  // namespace

void testCascade(Ctx& c) {
    c.rep.family = "cascade";
    const bool g1151 = c.k.caps.attn_group1;
    if (c.k.attn_cascade_rows == nullptr || c.k.attn_cascade_merge == nullptr) {
        c.rep.skip("cascade", "attn_cascade_rows / attn_cascade_merge", "kernel not present");
        return;
    }
    using K = ops::CascadePrefixKind;
    std::vector<Case> cases;
    auto add = [&](Case cs) { cases.push_back(std::move(cs)); };
    if (!g1151) {
        // gfx1201: prefix attn_wsplit1 (<= 16 columns) / attn_wsplit2 (<= 32), cap 32 / grp rows (27B 5, Ornith 4)
        add({.name = "27B 2x1 P1024 wsplit1", .sp = 4, .seqs = {{1100, 1}, {1500, 1}}, .groups = {{0, 2, 1024}}, .prefix = K::wsplit1,
             .gate_checks = true});
        add({.name = "27B 3x1 P1536 wsplit2", .sp = 6, .seqs = {{1700, 1}, {2000, 1}, {1600, 1}}, .groups = {{0, 3, 1536}}, .prefix = K::wsplit2});
        add({.name = "27B 2x2 MTP nd1 wsplit2", .sp = 4, .seqs = {{1300, 2}, {1400, 2}}, .groups = {{0, 4, 1024}}, .prefix = K::wsplit2});
        add({.name = "27B 2x2 MTP nd1, suffix wsplit1 (ATTN_WIDE=0)", .sp = 4, .seqs = {{1300, 2}, {1400, 2}}, .groups = {{0, 4, 1024}},
             .prefix = K::wsplit2, .wide = false});
        add({.name = "Ornith 2x1 wsplit1", .n_head = 16, .n_kv = 2, .sp = 4, .seqs = {{1100, 1}, {1200, 1}}, .groups = {{0, 2, 1024}},
             .prefix = K::wsplit1});
        add({.name = "Ornith 4x1 wsplit2", .n_head = 16, .n_kv = 2, .sp = 4, .seqs = {{1100, 1}, {1200, 1}, {1300, 1}, {1400, 1}},
             .groups = {{0, 4, 1024}}, .prefix = K::wsplit2});
        add({.name = "27B P cut by position", .sp = 8, .seqs = {{1100, 1}, {2500, 1}}, .groups = {{0, 2, 1024}}, .prefix = K::wsplit1});
        add({.name = "27B 2x1 P4608 (split 128 keys)", .sp = 18, .seqs = {{4700, 1}, {4900, 1}}, .groups = {{0, 2, 4608}}, .prefix = K::wsplit1,
             .max_ctx = 8192});
        add({.name = "27B 3x1 n_split 128", .sp = 6, .seqs = {{1700, 1}, {2000, 1}, {1600, 1}}, .groups = {{0, 3, 1536}}, .prefix = K::wsplit2,
             .fd = 128, .max_ctx = 16384});
        add({.name = "27B 3x1 large logits", .sp = 6, .seqs = {{1700, 1}, {2000, 1}, {1600, 1}}, .groups = {{0, 3, 1536}}, .prefix = K::wsplit2,
             .qsig = 4.0f});
        add({.name = "27B 2 shared + 1 own", .sp = 4, .seqs = {{1100, 1}, {1500, 1}, {1300, 1, false}}, .groups = {{0, 2, 1024}},
             .prefix = K::wsplit1});
        // review C2: MTP nd 3 (4 rows / sequence) never fits the 5-row cap: empty plan
        add({.name = "27B 2x4 MTP nd3 over the cap", .sp = 4, .seqs = {{1300, 4}, {1400, 4}}, .groups = {}});
    } else {
        // gfx1151: prefix attn_ws4g CG = min(3, ceil(rows * grp / 16)), cap 32 rows (column-group blocks)
        add({.name = "27B 2x1 ws4g1", .sp = 4, .seqs = {{1100, 1}, {1500, 1}}, .groups = {{0, 2, 1024}}, .prefix = K::ws4g, .cg = 1,
             .gate_checks = true});
        add({.name = "27B 3x1 ws4g2", .sp = 6, .seqs = {{1700, 1}, {2000, 1}, {1600, 1}}, .groups = {{0, 3, 1536}}, .prefix = K::ws4g, .cg = 2});
        add({.name = "27B 4x2 MTP nd1 ws4g3", .sp = 4, .seqs = {{1100, 2}, {1200, 2}, {1300, 2}, {1400, 2}}, .groups = {{0, 8, 1024}},
             .prefix = K::ws4g, .cg = 3});
        add({.name = "27B 4x4 MTP nd3 ws4g3 x2 column groups", .sp = 4, .seqs = {{1100, 4}, {1200, 4}, {1300, 4}, {1400, 4}},
             .groups = {{0, 16, 1024}}, .prefix = K::ws4g, .cg = 3});
        add({.name = "Ornith 3x2 MTP nd1 ws4g3", .n_head = 16, .n_kv = 2, .sp = 4, .seqs = {{1100, 2}, {1200, 2}, {1300, 2}},
             .groups = {{0, 6, 1024}}, .prefix = K::ws4g, .cg = 3});
        add({.name = "27B 3x1 ATTN_WS4=0 (suffix wsplit1)", .sp = 6, .seqs = {{1700, 1}, {2000, 1}, {1600, 1}}, .groups = {{0, 3, 1536}},
             .prefix = K::ws4g, .cg = 2, .ws4 = false});
        add({.name = "27B 3x1 ATTN_WS4=0 n_split 128", .sp = 6, .seqs = {{1700, 1}, {2000, 1}, {1600, 1}}, .groups = {{0, 3, 1536}},
             .prefix = K::ws4g, .cg = 2, .fd = 128, .max_ctx = 16384, .ws4 = false});
        add({.name = "27B P cut by position", .sp = 8, .seqs = {{1100, 1}, {2500, 1}}, .groups = {{0, 2, 1024}}, .prefix = K::ws4g, .cg = 1});
        add({.name = "27B 2x1 P4608 (split 128 keys)", .sp = 18, .seqs = {{4700, 1}, {4900, 1}}, .groups = {{0, 2, 4608}}, .prefix = K::ws4g,
             .cg = 1, .max_ctx = 8192});
        add({.name = "27B 3x1 large logits", .sp = 6, .seqs = {{1700, 1}, {2000, 1}, {1600, 1}}, .groups = {{0, 3, 1536}}, .prefix = K::ws4g,
             .cg = 2, .qsig = 4.0f});
        add({.name = "27B 2 shared + 1 own", .sp = 4, .seqs = {{1100, 1}, {1500, 1}, {1300, 1, false}}, .groups = {{0, 2, 1024}},
             .prefix = K::ws4g, .cg = 1});
    }
    Scratch sc;
    for (const Kv fmt : {Kv::f16, Kv::q8, Kv::q8v}) {
        if (fmt == Kv::q8v && !c.k.caps.kv_q8v) {
            c.rep.skip("cascade", "kv q8v", "caps.kv_q8v off on this code object");
            continue;
        }
        if (c.quick && fmt == Kv::q8v) continue;
        const wk::KernelTable kt_ = wk::KernelTable::load(c.mod, kvFormat(fmt));
        for (const Case& cs : cases) runCase(c, sc, kt_, fmt, cs);
    }
}

}  // namespace kt
