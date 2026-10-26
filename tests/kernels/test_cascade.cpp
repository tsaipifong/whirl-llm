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
//   * the gfx1201 MTP limit (review C2): without attn_cascade_wide (WHIRL_CASCADE_WIDE=0 table) two sequences of 4
//     rows exceed the 5-row cap and the plan is empty;
//   * feat026-cascade-wide (gfx1201 attn_cascade_wide, up to 128 columns): the same checks with the wide kernel
//     (27B 4 x 4 MTP rows, Ornith 4 x 4 = 128 columns, 6 x 4 past the 21-row cap, P 4608, n_split 128, large
//     logits, P cut by position, a sequence outside the group), and per column the prefix partials of
//     attn_cascade_wide == those of attn_wsplit1 / attn_wsplit2 run per sequence (bits: same per-column code), as
//     well as == those of the prefix kernel the launcher chose.
//   * feat026-cascade-wide-spill: attn_cascade_wide is the no-spill build; every wide case runs again with the first
//     build attn_cascade_wide_r0 (WHIRL_CASCADE_WIDE=r0 table) and both must give attn_wsplit's bits per column.
//   * re-review R3: attn_combine_q8 (the balance decode combine when the o-projection is a GEMV) on the same partials:
//     its f32 output == attn_combine's (bits), its int8 copy == the CPU quantizer of that output (wave_q8, exact), and
//     cascade vs plain on the dequantized int8 within the tolerance + half a quantization step of each;
//   * re-review R2: gfx1201 q4 KV (fast): suffix attn_dq4 rows (n_split 256), prefix attn_wsplit*_q4 (no wide q4
//     kernel: r1 caps), cascade vs plain at the fast-class bound 1e-2 x scale, NMSE 5e-3 (attn_dq4 quantizes q to
//     int8 on its keys, the prefix kernel does not), both vs CPU at rtol 2e-2 (initial bound).
// Cases cover f16 / int8 (q8, q8v) / q4 (gfx1201) KV, GQA 6 (27B) and 8 (Ornith), the n_split 64 / 128 layouts, a prefix past 4096
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

enum class Kv { f16, q8, q8v, q4 };
const char* kvName(Kv f) { return f == Kv::f16 ? "f16" : f == Kv::q8 ? "q8" : f == Kv::q8v ? "q8v" : "q4"; }
whirl::KvFormat kvFormat(Kv f) {
    return f == Kv::f16 ? whirl::KvFormat::f16 : f == Kv::q8 ? whirl::KvFormat::q8 : f == Kv::q8v ? whirl::KvFormat::q8v : whirl::KvFormat::q4;
}
bool kQ(Kv f) { return f == Kv::q8 || f == Kv::q4; }
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
            if (f == Kv::q4) {
                // q4 (kernels/attn.hip kv_dq4): element e = nibble e & 1 of byte e / 2 (low first), value
                // f16((nibble - 8) * scale), one f16 scale per 32 values
                const std::vector<int> qi = c.randi(n, -8, 7);
                const std::vector<float> su = c.randu(n / 32, 0.06f, 0.2f);
                std::vector<std::uint8_t> b(n / 2, 0);
                std::vector<std::uint16_t> s(n / 32);
                for (std::size_t g = 0; g < n / 32; ++g) s[g] = f2h(su[g]);
                for (std::size_t i = 0; i < n; ++i) {
                    b[i >> 1] = static_cast<std::uint8_t>(b[i >> 1] | ((qi[i] + 8) << (4 * (i & 1))));
                    val[i] = h2f(d2h(static_cast<double>(qi[i]) * h2f(s[i >> 5])));
                }
                data = Buf(b);
                sc = Buf(s);
                return;
            }
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
    bool narrow_empty = false;  // gfx1201 without attn_cascade_wide: the plan must be empty (> 32 columns per group)
};

const char* prefixName(ops::CascadePrefixKind k) {
    switch (k) {
    case ops::CascadePrefixKind::none: return "none";
    case ops::CascadePrefixKind::wsplit1: return "attn_wsplit1";
    case ops::CascadePrefixKind::wsplit2: return "attn_wsplit2";
    case ops::CascadePrefixKind::ws4g: return "attn_ws4g";
    case ops::CascadePrefixKind::wide: return "attn_cascade_wide";
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
    Buf ml, acc, cml, cacc, ml2, acc2, rows, out, out2, out3, gz, g1, xq1, xd1, xq2, xd2, oq1, oq2;
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
        xq1 = Buf(static_cast<std::size_t>(32) * 24 * kHd);
        xq2 = Buf(xq1.bytes());
        xd1 = Buf(static_cast<std::size_t>(32) * 24 * kHd / 32 * 4);
        xd2 = Buf(xd1.bytes());
        oq1 = Buf(out.bytes());
        oq2 = Buf(out.bytes());
        gz = Buf(std::vector<int>{0});
        g1 = Buf(std::vector<int>{1});
    }
};

// feat026-cascade-wide: the prefix partials of attn_cascade_wide (the plan's groups) vs attn_wsplit1 / attn_wsplit2 run
// per sequence (each sequence's rows its own group, the same prefix positions P - 1 and split count): bitwise, per
// (row in a group, split, head): (m, l) always, acc where m > -inf (an empty split writes no acc). Also the prefix
// partials the launcher left in cas_ml / cas_acc (its chosen kernel) vs the same reference.
void checkWideBits(Ctx& c, Scratch& sc, const wk::KernelTable& kt_, const ops::CascadePlan& plan, const std::vector<std::int32_t>& rb,
                   const wk::KvArgs& kva, const Buf& dq, int H, int HKV, std::uint32_t n_split, float scale, const std::string& tag) {
    const int grp = H / HKV;
    wk::AwGroups ref{}, wide{};
    int nr = 0, nw = 0, widest_seq = 0;
    for (const ops::CascadeGroup& g : plan.groups) {
        wide.first[nw] = static_cast<std::int32_t>(g.first);
        wide.count[nw] = static_cast<std::int32_t>(g.count);
        ++nw;
        for (std::uint32_t r = g.first; r < g.first + g.count;) {
            std::uint32_t e = r + 1;
            while (e < g.first + g.count && rb[e] == rb[r]) ++e;
            if (nr >= 32) return;
            ref.first[nr] = static_cast<std::int32_t>(r);
            ref.count[nr] = static_cast<std::int32_t>(e - r);
            widest_seq = std::max(widest_seq, static_cast<int>(e - r) * grp);
            ++nr;
            r = e;
        }
    }
    const bool two = widest_seq > 16;
    const hip::Function fr = two ? kt_.attn_wsplit2 : kt_.attn_wsplit1;
    if (fr == nullptr || widest_seq > 32 || kt_.attn_cascade_wide == nullptr) {
        c.rep.skip("cascade", "wide bits" + tag, "no per-sequence attn_wsplit reference for this shape");
        return;
    }
    const DevPtr ppos = sc.rows.p() + 64 * 4;
    sc.ml.fill(0x7f);
    sc.acc.fill(0x7f);
    sc.ml2.fill(0x3f);
    sc.acc2.fill(0x3f);
    hip::launch(fr, hip::Dim3{static_cast<std::uint32_t>(HKV), n_split, static_cast<std::uint32_t>(nr)}, hip::Dim3{128, 1, 1}, 0, c.s, dq.p(), kva,
                sc.ml.p(), sc.acc.p(), H, HKV, 2 * kHd, ppos, scale, DevPtr{0}, ref);
    hip::launch(kt_.attn_cascade_wide, hip::Dim3{static_cast<std::uint32_t>(HKV), n_split, static_cast<std::uint32_t>(nw)}, hip::Dim3{256, 1, 1},
                0, c.s, dq.p(), kva, sc.ml2.p(), sc.acc2.p(), H, HKV, 2 * kHd, ppos, scale, DevPtr{0}, wide);
    c.sync();
    const std::size_t rows = 32, mlw = rows * n_split * H * 2, accw = rows * n_split * H * kHd;
    const std::vector<std::uint32_t> rm = sc.ml.down<std::uint32_t>(mlw), ra = sc.acc.down<std::uint32_t>(accw);
    const std::vector<std::uint32_t> wm = sc.ml2.down<std::uint32_t>(mlw), wa = sc.acc2.down<std::uint32_t>(accw);
    const std::vector<std::uint32_t> cm = sc.cml.down<std::uint32_t>(mlw), ca = sc.cacc.down<std::uint32_t>(accw);
    std::vector<std::uint32_t> gr, gw, gc;
    std::size_t empty = 0;
    for (const ops::CascadeGroup& g : plan.groups)
        for (std::uint32_t r = g.first; r < g.first + g.count; ++r)
            for (std::uint32_t sp = 0; sp < n_split; ++sp)
                for (int h = 0; h < H; ++h) {
                    const std::size_t i = (static_cast<std::size_t>(r) * n_split + sp) * H + h;
                    for (int j = 0; j < 2; ++j) {
                        gr.push_back(rm[2 * i + j]);
                        gw.push_back(wm[2 * i + j]);
                        gc.push_back(cm[2 * i + j]);
                    }
                    float m;
                    std::memcpy(&m, &rm[2 * i], 4);
                    if (!(m > -INFINITY)) {
                        ++empty;
                        continue;
                    }
                    for (int d = 0; d < kHd; ++d) {
                        gr.push_back(ra[i * kHd + d]);
                        gw.push_back(wa[i * kHd + d]);
                        gc.push_back(ca[i * kHd + d]);
                    }
                }
    Result r1 = cmpExact("prefix partials: attn_cascade_wide == attn_wsplit" + std::string(two ? "2" : "1") + " per sequence (bits)" + tag, gw, gr,
                         Kind::invariant);
    r1.note += " empty splits " + std::to_string(empty);
    c.rep.add(r1);
    c.rep.add(cmpExact("prefix partials: launcher's prefix kernel == attn_wsplit per sequence (bits)" + tag, gc, gr, Kind::invariant));
}

void runCase(Ctx& c, Scratch& sc, const wk::KernelTable& kt_, Kv fmt, const Case& cs, bool narrow = false, const char* var = "") {
    const std::string tag = " [" + cs.name + ", kv " + kvName(fmt) + (narrow ? (fmt == Kv::q4 ? ", no attn_cascade_wide" : ", WHIRL_CASCADE_WIDE=0") : "") +
                            var + "]";
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
    ap.kv_q4 = fmt == Kv::q4;
    ap.rows_known = true;
    ap.row_pos = std::span<const std::int32_t>(rp.data(), n);
    ap.row_base = std::span<const std::int32_t>(rb.data(), n);
    ap.win = 0;
    ap.o_on_gemv = false;
    ops::AttnDecodeFlags af;
    af.attn_ws4 = cs.ws4;
    af.attn_wide = cs.wide;
    af.fd_splits = fmt == Kv::q4 ? 256 : cs.fd;  // q4: resolveFdSplits(true)
    af.prec_dec = false;
    const whirl::device::Profile& prof = whirl::device::profile(c.mod.arch());
    const std::size_t on = static_cast<std::size_t>(n) * H * kHd;

    // ---- the plan with the real cap
    ops::CascadePrefixProblem cp;
    cp.head_dim = kHd;
    cp.n_head = ap.n_head;
    cp.n_head_kv = ap.n_head_kv;
    cp.kv_q4 = ap.kv_q4;
    const std::uint32_t cap = ops::cascadeGroupCap(cp, kt_, kt_.caps);
    if (cap < 2 || (cs.prefix == ops::CascadePrefixKind::wsplit2 && kt_.attn_wsplit2 == nullptr) ||
        (!narrow && cs.prefix == ops::CascadePrefixKind::wide && kt_.attn_cascade_wide == nullptr)) {
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
        static const std::vector<ExpGroup> k_none;
        const std::vector<ExpGroup>& want = narrow && cs.narrow_empty ? k_none : cs.groups;
        bool ok = plan.groups.size() == want.size();
        for (std::size_t g = 0; ok && g < want.size(); ++g)
            ok = static_cast<int>(plan.groups[g].first) == want[g].first && static_cast<int>(plan.groups[g].count) == want[g].count &&
                 static_cast<int>(plan.groups[g].keys) == want[g].keys;
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
        if (ok && pp.kind == ops::CascadePrefixKind::wide) ok = pp.f == kt_.attn_cascade_wide && pp.block.x == 256;
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
    // q4 (fast): attn_dq4 (plain, suffix) quantizes q to int8 per 32 values, the prefix kernel uses f16 q: fast-class bound
    const bool q4 = fmt == Kv::q4;
    const double tol = q4 ? 1e-2 : 2e-3, nmse_max = q4 ? 5e-3 : 5e-4;
    Result rc = cmpTol(std::string("cascade vs plain (") + (q4 ? "1e-2" : "2e-3") + " x scale)" + tag, o_cas, plain_d, rsc, tol, 1e-6);
    const double e = nmse(o_cas, o_plain);
    rc.note += " nmse " + std::to_string(e);
    if (e > nmse_max) rc.pass = false;
    c.rep.add(rc);
    // q4: attn_dq4's int8 q (plain and suffix) is part of both paths; 2e-2 is an initial bound (first GPU run reports it)
    const double rtol_cpu = q4 ? 2e-2 : 1e-2;
    c.rep.add(cmpTol("cascade vs CPU" + tag, o_cas, ref, rsc, rtol_cpu, 1e-4));
    c.rep.add(cmpTol("plain vs CPU" + tag, o_plain, ref, rsc, rtol_cpu, 1e-4));
    // ---- re-review R3: attn_combine_q8 on the plain (ml / acc) and cascade (ml2 / acc2) partials
    if (kt_.attn_combine_q8 != nullptr) {
        hip::launch(kt_.attn_combine_q8, pl.combine_grid, hip::Dim3{kHd, 1, 1}, 0, c.s, sc.ml.p(), sc.acc.p(), dq.p(), sc.oq1.p(), H, kHd, 2 * kHd,
                    static_cast<int>(pl.n_split), sc.xq1.p(), sc.xd1.p());
        hip::launch(kt_.attn_combine_q8, cpl.combine_grid, hip::Dim3{kHd, 1, 1}, 0, c.s, sc.ml2.p(), sc.acc2.p(), dq.p(), sc.oq2.p(), H, kHd,
                    2 * kHd, static_cast<int>(cpl.n_split), sc.xq2.p(), sc.xd2.p());
        c.sync();
        const std::vector<float> f1 = sc.oq1.down<float>(on), f2 = sc.oq2.down<float>(on);
        c.rep.add(cmpExact("attn_combine_q8 f32 out == attn_combine (plain, bits)" + tag, f1, o_plain, Kind::invariant));
        c.rep.add(cmpExact("attn_combine_q8 f32 out == attn_combine (cascade, bits)" + tag, f2, o_cas, Kind::invariant));
        const std::vector<std::int8_t> q1 = sc.xq1.down<std::int8_t>(on), q2 = sc.xq2.down<std::int8_t>(on);
        const std::vector<float> d1 = sc.xd1.down<float>(on / 32), d2 = sc.xd2.down<float>(on / 32);
        // wave_q8 (kernels/fused_decode.hip) on the CPU: d = amax / 127, q = rint(v * (1 / d)) (f32, ties to even)
        auto quant = [](const std::vector<float>& v, std::vector<std::int8_t>& q, std::vector<float>& d) {
            q.assign(v.size(), 0);
            d.assign(v.size() / 32, 0.f);
            for (std::size_t b = 0; b < v.size() / 32; ++b) {
                float amax = 0.f;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(v[b * 32 + i]));
                const float dd = amax / 127.f, id = dd > 0.f ? 1.f / dd : 0.f;
                d[b] = dd;
                for (int i = 0; i < 32; ++i) q[b * 32 + i] = static_cast<std::int8_t>(std::nearbyint(v[b * 32 + i] * id));
            }
        };
        std::vector<std::int8_t> rq;
        std::vector<float> rd;
        quant(f2, rq, rd);
        c.rep.add(cmpExact("attn_combine_q8 int8 copy == CPU wave_q8 of its output (cascade)" + tag, q2, rq, Kind::invariant));
        c.rep.add(cmpExact("attn_combine_q8 int8 scales == CPU wave_q8 (cascade)" + tag, d2, rd, Kind::invariant));
        Result rq8;
        rq8.name = std::string("cascade vs plain on attn_combine_q8 int8 (") + (q4 ? "1e-2" : "2e-3") + " x scale + half steps)" + tag;
        rq8.kind = Kind::tol;
        rq8.metric = 0;
        rq8.n = on;
        double worst = 0;
        for (std::size_t i = 0; i < on; ++i) {
            const double qa = static_cast<double>(q2[i]) * d2[i >> 5], qb = static_cast<double>(q1[i]) * d1[i >> 5];
            const double lim = tol * rsc[i] + 0.5 * (static_cast<double>(d1[i >> 5]) + d2[i >> 5]) + 1e-6;
            const double x = std::fabs(qa - qb);
            worst = std::max(worst, x / lim);
            if (x > lim) rq8.mismatches += 1;
        }
        rq8.pass = rq8.mismatches == 0;
        rq8.metric = worst;
        rq8.note = "worst |d| / bound " + std::to_string(worst);
        c.rep.add(rq8);
    }
    // feat026-cascade-wide: per-column identity of the wide prefix kernel (gfx1201 tables that have it)
    if (!c.k.caps.attn_group1 && kt_.attn_cascade_wide != nullptr) checkWideBits(c, sc, kt_, plan, rb, kva, dq, H, HKV, cpl.n_split, scale, tag);

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
        // review C2: MTP nd 3 (4 rows / sequence) never fits the 5-row cap of attn_wsplit2: empty plan without
        // attn_cascade_wide (narrow run); feat026-cascade-wide: one 8-row group with it
        add({.name = "27B 2x4 MTP nd3 (over the narrow cap)", .sp = 4, .seqs = {{1300, 4}, {1400, 4}}, .groups = {{0, 8, 1024}},
             .prefix = K::wide, .narrow_empty = true});
        // feat026-cascade-wide: attn_cascade_wide groups (> 32 columns)
        add({.name = "27B 4x4 MTP nd3 wide", .sp = 4, .seqs = {{1100, 4}, {1200, 4}, {1300, 4}, {1400, 4}}, .groups = {{0, 16, 1024}},
             .prefix = K::wide, .gate_checks = true, .narrow_empty = true});
        add({.name = "27B 4x2 MTP nd1 wide (48 columns)", .sp = 4, .seqs = {{1100, 2}, {1200, 2}, {1300, 2}, {1400, 2}},
             .groups = {{0, 8, 1024}}, .prefix = K::wide, .narrow_empty = true});
        add({.name = "Ornith 4x4 MTP nd3 wide (128 columns)", .n_head = 16, .n_kv = 2, .sp = 4,
             .seqs = {{1100, 4}, {1200, 4}, {1300, 4}, {1400, 4}}, .groups = {{0, 16, 1024}}, .prefix = K::wide, .narrow_empty = true});
        add({.name = "27B 6x4 past the wide cap (21 rows)", .sp = 4, .seqs = {{1100, 4}, {1200, 4}, {1300, 4}, {1400, 4}, {1500, 4}, {1600, 4}},
             .groups = {{0, 20, 1024}}, .prefix = K::wide, .narrow_empty = true});
        add({.name = "27B 4x4 wide P4608 (split 128 keys)", .sp = 18, .seqs = {{4700, 4}, {4800, 4}, {4900, 4}, {5000, 4}},
             .groups = {{0, 16, 4608}}, .prefix = K::wide, .max_ctx = 8192, .narrow_empty = true});
        add({.name = "27B 4x4 wide n_split 128", .sp = 6, .seqs = {{1700, 4}, {2000, 4}, {1600, 4}, {1800, 4}}, .groups = {{0, 16, 1536}},
             .prefix = K::wide, .fd = 128, .max_ctx = 16384, .narrow_empty = true});
        add({.name = "27B 4x4 wide large logits", .sp = 6, .seqs = {{1700, 4}, {2000, 4}, {1600, 4}, {1800, 4}}, .groups = {{0, 16, 1536}},
             .prefix = K::wide, .qsig = 4.0f, .narrow_empty = true});
        add({.name = "27B 4x4 wide P cut by position", .sp = 8, .seqs = {{1100, 4}, {2500, 4}, {1300, 4}, {1400, 4}}, .groups = {{0, 16, 1024}},
             .prefix = K::wide, .narrow_empty = true});
        add({.name = "27B 3x4 shared + 1 own, wide", .sp = 4, .seqs = {{1100, 4}, {1200, 4}, {1300, 4}, {1250, 4, false}},
             .groups = {{0, 12, 1024}}, .prefix = K::wide, .narrow_empty = true});
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
    for (const Kv fmt : {Kv::f16, Kv::q8, Kv::q8v, Kv::q4}) {
        if (fmt == Kv::q8v && !c.k.caps.kv_q8v) {
            c.rep.skip("cascade", "kv q8v", "caps.kv_q8v off on this code object");
            continue;
        }
        if (fmt == Kv::q4 && (g1151 || !c.k.caps.kv_q4)) {
            // gfx1151: cascadeGroupCap is 0 for q4 (no multi-row q4 kernel), the cascade is off there by design
            c.rep.skip("cascade", "kv q4", g1151 ? "gfx1151: no q4 prefix kernel (cascade off by design)" : "caps.kv_q4 off on this code object");
            continue;
        }
        if (c.quick && (fmt == Kv::q8v || fmt == Kv::q4)) continue;
        const wk::KernelTable kt_ = wk::KernelTable::load(c.mod, kvFormat(fmt));
        if (fmt == Kv::q4) {
            // no attn_cascade_wide q4 variant: every case with the r1 caps (wide-only cases expect an empty plan)
            for (const Case& cs : cases) runCase(c, sc, kt_, fmt, cs, true);
            continue;
        }
        for (const Case& cs : cases) runCase(c, sc, kt_, fmt, cs);
        // feat026-cascade-wide: the wide-only cases again with the table WHIRL_CASCADE_WIDE=0 gives (r1 caps)
        if (!g1151) {
            wk::KernelTable kn = kt_;
            kn.attn_cascade_wide = nullptr;
            for (const Case& cs : cases)
                if (cs.narrow_empty) runCase(c, sc, kn, fmt, cs, true);
        }
        // feat026-cascade-wide-spill: the wide cases with the first wide build (WHIRL_CASCADE_WIDE=r0)
        if (!g1151 && kt_.attn_cascade_wide_r0 != nullptr) {
            wk::KernelTable kr = kt_;
            kr.attn_cascade_wide = kt_.attn_cascade_wide_r0;
            for (const Case& cs : cases)
                if (cs.prefix == ops::CascadePrefixKind::wide) runCase(c, sc, kr, fmt, cs, false, ", WHIRL_CASCADE_WIDE=r0");
        }
    }
}

}  // namespace kt
