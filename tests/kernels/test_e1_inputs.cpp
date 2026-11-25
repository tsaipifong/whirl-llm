// whirl-kernel-test family "e1in" (v0252-e1p1b, PLAN V1-93 E1 step P1b): every kernel whose launch can now take a value
// from device memory instead of a kernel argument, launched both ways with the same values - the device-value launch must
// write the same bits as today's by-value launch (and, where there is one, the CPU reference):
//   set_rows_dp          (kernels/e1_inputs.hip, both code objects) == set_rows with pos = dev_src[pdev.v[i]] + tab.pos[i],
//                        rows without a device word == set_rows; ids / pos / kvbase (and the untouched rows past n) vs the
//                        CPU reference; random tables of 1..32 rows (tokens by value and by negative index)
//   rmsnorm_q8_rows_dv   (both code objects) == rmsnorm_q8_rows with src.v[b] + dev[add.v[b]]: out, xq, xd at E 2048 / 5120
//   gdn replay count     (gfx1201 replay-mode kernels, kernels/fused_decode.hip: gdn_conv_l2, gdn_abconv_q8_0 (its conv
//                        blocks), gdn_step_norm, gdn_step_norm_v0) npend = -1 with the count at snap[0] == npend by value:
//                        state, pend, out (xq / xd) after a verify launch (rows + replay) and a commitSeq launch (replay
//                        only), counts 0 / 1 / 4 / 6; gfx1151 has no replay mode (skipped as in the "gdn" family)
//   spec_accept_rows     (both code objects) a.pos[k] = -1 with the position in the NPOS word == a.pos[k] by value: control
//                        words (incl. NPEND = acc + 1) and next-cycle rows, both vs kernels/spec_accept.h on the CPU
// The E1 path of the server uses them only with WHIRL_E1=1 (progress/v0252-e1p1b.HANDOFF.md); the default path launches
// set_rows / rmsnorm_q8_rows / npend >= 0 exactly as before.
// SPDX-License-Identifier: Apache-2.0

#include "kt.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "spec_accept.h"  // kernels/ (feat026-acc; v0252-e1p1b NPEND / device position)
#include "gdn_select.h"   // kernels/ (v0252-e1s8: the select rule)

template <>
struct whirl::hip::LaunchArgPtrWords<SpecAccArgs> : whirl::hip::LaunchArgNoPtrWords {};

namespace kt {

namespace {

constexpr int kRows = wk::kMaxVerifyRows;  // RowTab / RowIdx rows (32)
constexpr int kWords = 16 * WHIRL_SA_CTL_WORDS;  // 16 control areas

Result branchHits(const std::string& name, std::size_t n, bool pass) {
    Result r;
    r.family = "e1in";
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.pass = pass;
    return r;
}

// ---- set_rows_dp == set_rows (and the CPU reference)
void testSetRowsDp(Ctx& c) {
    if (!c.k.set_rows_dp || !c.k.set_rows) {
        c.rep.skip("e1in", "set_rows_dp", "kernel not present");
        return;
    }
    auto rnd = [&](std::uint32_t n) { return static_cast<int>(c.rng() % n); };
    const int rounds = c.quick ? 64 : 400;
    std::vector<int> g_dp, g_bv, want;
    std::size_t dev_rows = 0, neg_tok = 0;
    for (int it = 0; it < rounds; ++it) {
        std::vector<int> src(kWords);
        for (int& v : src) v = rnd(1 << 20);
        const int n = 1 + rnd(kRows);
        wk::RowTab tab_dp, tab_bv;
        wk::RowIdx pdev;
        for (int i = 0; i < kRows; ++i) pdev.v[i] = -1;
        for (int i = 0; i < n; ++i) {
            const int tok = rnd(3) == 0 ? -(rnd(kWords) + 1) : rnd(250000);
            tab_dp.tok[i] = tab_bv.tok[i] = tok;
            neg_tok += tok < 0;
            tab_dp.base[i] = tab_bv.base[i] = rnd(1 << 22);
            if (rnd(4) != 0) {  // a device position base (E1: the NPOS word of the sequence) + the row offset r
                const int w = rnd(kWords);
                pdev.v[i] = w;
                tab_dp.pos[i] = rnd(kRows);
                tab_bv.pos[i] = src[static_cast<std::size_t>(w)] + tab_dp.pos[i];
                dev_rows += 1;
            } else {
                tab_dp.pos[i] = tab_bv.pos[i] = rnd(1 << 20);
            }
        }
        Buf dsrc(src);
        const std::vector<int> fill(3 * kRows, -7);
        Buf o_dp(fill), o_bv(fill);
        const DevPtr ids_dp = o_dp.p(), pos_dp = o_dp.p() + 4 * kRows, kb_dp = o_dp.p() + 8 * kRows;
        const DevPtr ids_bv = o_bv.p(), pos_bv = o_bv.p() + 4 * kRows, kb_bv = o_bv.p() + 8 * kRows;
        hip::launch(c.k.set_rows_dp, {1, 1, 1}, {32, 1, 1}, 0, c.s, ids_dp, pos_dp, kb_dp, dsrc.p(), tab_dp, n, pdev);
        hip::launch(c.k.set_rows, {1, 1, 1}, {32, 1, 1}, 0, c.s, ids_bv, pos_bv, kb_bv, dsrc.p(), tab_bv, n);
        c.sync();
        const std::vector<int> a = o_dp.down<int>(3 * kRows), b = o_bv.down<int>(3 * kRows);
        std::vector<int> ref(fill);
        for (int i = 0; i < n; ++i) {
            const int t = tab_bv.tok[i];
            ref[static_cast<std::size_t>(i)] = t >= 0 ? t : src[static_cast<std::size_t>(-t - 1)];
            ref[static_cast<std::size_t>(kRows + i)] = tab_bv.pos[i];
            ref[static_cast<std::size_t>(2 * kRows + i)] = tab_bv.base[i];
        }
        g_dp.insert(g_dp.end(), a.begin(), a.end());
        g_bv.insert(g_bv.end(), b.begin(), b.end());
        want.insert(want.end(), ref.begin(), ref.end());
    }
    c.rep.add(cmpExact("set_rows_dp (device position bases) == set_rows by value (" + std::to_string(rounds) + " tables)", g_dp, g_bv,
                       Kind::invariant));
    c.rep.add(cmpExact("set_rows_dp vs CPU (ids / pos / kvbase, rows past n untouched)", g_dp, want));
    // every row from the table (pdev all -1) == set_rows exactly
    {
        std::vector<int> src(kWords, 5);
        wk::RowTab tab;
        wk::RowIdx none;
        for (int i = 0; i < kRows; ++i) {
            tab.tok[i] = i % 3 == 0 ? -(i + 1) : 1000 + i;
            tab.pos[i] = 77 + i;
            tab.base[i] = 4096 * (i % 4);
            none.v[i] = -1;
        }
        Buf dsrc(src);
        const std::vector<int> fill(3 * kRows, -7);
        Buf o1(fill), o2(fill);
        hip::launch(c.k.set_rows_dp, {1, 1, 1}, {32, 1, 1}, 0, c.s, o1.p(), o1.p() + 4 * kRows, o1.p() + 8 * kRows, dsrc.p(), tab, kRows, none);
        hip::launch(c.k.set_rows, {1, 1, 1}, {32, 1, 1}, 0, c.s, o2.p(), o2.p() + 4 * kRows, o2.p() + 8 * kRows, dsrc.p(), tab, kRows);
        c.sync();
        c.rep.add(cmpExact("set_rows_dp without device words == set_rows (32 rows)", o1.down<int>(3 * kRows), o2.down<int>(3 * kRows),
                           Kind::invariant));
    }
    c.rep.add(branchHits("set_rows_dp rounds reached device positions " + std::to_string(dev_rows) + " and token indices " +
                             std::to_string(neg_tok),
                         static_cast<std::size_t>(rounds), dev_rows > 0 && neg_tok > 0));
}

// ---- rmsnorm_q8_rows_dv == rmsnorm_q8_rows
void testRmsnormRowsDv(Ctx& c) {
    if (!c.k.rmsnorm_q8_rows_dv || !c.k.rmsnorm_q8_rows) {
        c.rep.skip("e1in", "rmsnorm_q8_rows_dv", "kernel not present");
        return;
    }
    auto rnd = [&](std::uint32_t n) { return static_cast<int>(c.rng() % n); };
    for (const int E : {2048, 5120}) {
        const int rounds = c.quick ? 6 : 24;
        const std::vector<float> x = c.randn(static_cast<std::size_t>(kRows) * E), wv = c.randn(static_cast<std::size_t>(E), 0.5f);
        Buf dx(x), dw(wv);
        std::vector<float> o_a, o_b, d_a, d_b;
        std::vector<std::int8_t> q_a, q_b;
        std::size_t dev_segs = 0;
        for (int it = 0; it < rounds; ++it) {
            const int ns = 1 + rnd(16);
            std::vector<int> dev(64);
            for (int& v : dev) v = rnd(16);
            wk::RowIdx src_dv, add, src_bv;
            for (int b = 0; b < kRows; ++b) src_dv.v[b] = add.v[b] = src_bv.v[b] = -1;
            for (int b = 0; b < ns; ++b) {
                const int off = rnd(16);  // the segment's first padded MTP row
                src_dv.v[b] = off;
                if (rnd(5) != 0) {        // + the ACC word of the sequence
                    const int w = rnd(64);
                    add.v[b] = w;
                    src_bv.v[b] = off + dev[static_cast<std::size_t>(w)];
                    dev_segs += 1;
                } else {
                    src_bv.v[b] = off;
                }
            }
            Buf ddev(dev);
            const std::size_t no = static_cast<std::size_t>(ns) * E;
            Buf oa(no * 4), ob(no * 4), qa(no), qb(no), xa(no / 32 * 4), xb(no / 32 * 4);
            oa.fill(0x5a), ob.fill(0x5a), qa.fill(0x5a), qb.fill(0x5a), xa.fill(0x5a), xb.fill(0x5a);
            hip::launch(c.k.rmsnorm_q8_rows_dv, {static_cast<unsigned>(ns), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), dw.p(), oa.p(), qa.p(), xa.p(),
                        E, 1e-6f, src_dv, ddev.p(), add);
            hip::launch(c.k.rmsnorm_q8_rows, {static_cast<unsigned>(ns), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), dw.p(), ob.p(), qb.p(), xb.p(),
                        E, 1e-6f, src_bv);
            c.sync();
            auto app = [](auto& dst, const auto& v) { dst.insert(dst.end(), v.begin(), v.end()); };
            app(o_a, oa.down<float>(no));
            app(o_b, ob.down<float>(no));
            app(q_a, qa.down<std::int8_t>(no));
            app(q_b, qb.down<std::int8_t>(no));
            app(d_a, xa.down<float>(no / 32));
            app(d_b, xb.down<float>(no / 32));
        }
        const std::string e = " (E " + std::to_string(E) + ", " + std::to_string(rounds) + " launches)";
        c.rep.add(cmpExact("rmsnorm_q8_rows_dv == rmsnorm_q8_rows out" + e, o_a, o_b, Kind::invariant));
        c.rep.add(cmpExact("rmsnorm_q8_rows_dv == rmsnorm_q8_rows xq" + e, q_a, q_b, Kind::invariant));
        c.rep.add(cmpExact("rmsnorm_q8_rows_dv == rmsnorm_q8_rows xd" + e, d_a, d_b, Kind::invariant));
        c.rep.add(branchHits("rmsnorm_q8_rows_dv device row offsets used " + std::to_string(dev_segs) + e, static_cast<std::size_t>(rounds),
                             dev_segs > 0));
    }
}

// ---- the DeltaNet replay count from a device word (gfx1201 replay mode)
constexpr int kNk = 16, kNv = 48, kDk = 128, kDv = 128;
constexpr int kCh = 2 * kNk * kDk + kNv * kDv;
constexpr int kDInner = kNv * kDv;
constexpr int kPW = kDk + kDv + 2;  // pend row width per head (delta rule)

void testGdnReplayCount(Ctx& c) {
    if (!c.k.caps.gdn_replay) {
        c.rep.skip("e1in", "gdn replay count from a device word", "replay-mode fused DeltaNet kernels not in this code object");
        return;
    }
    const float eps = 1e-6f, scale = 1.0f / std::sqrt(128.0f);
    const int rec = 6;                       // rows the previous verify recorded into pend
    const int n2 = 3;                        // rows of the next verify
    const std::vector<float> convw = c.randn(static_cast<std::size_t>(kCh) * 4, 0.5f);
    const std::vector<float> xin = c.randn(static_cast<std::size_t>(rec + n2) * kCh, 0.5f);
    const std::vector<float> cst0 = c.randn(3 * kCh);
    const std::vector<float> st0 = c.randn(static_cast<std::size_t>(kNv) * kDk * kDv, 0.1f);
    const std::vector<float> qkv = c.randn(static_cast<std::size_t>(rec + n2) * kCh, 0.1f);
    std::vector<float> g = c.randn(static_cast<std::size_t>(rec + n2) * kNv, 0.1f);
    for (float& v : g) v = -std::fabs(v);
    const std::vector<float> beta = c.randu(static_cast<std::size_t>(rec + n2) * kNv, 0.f, 1.f);
    const std::vector<float> z = c.randn(static_cast<std::size_t>(rec + n2) * kDInner);
    const std::vector<float> wn = c.randn(kDv, 0.5f);
    Buf dconvw(convw), dxin(xin), dq(qkv), dg(g), db(beta), dz(z), dwn(wn);
    const hip::Function fab = c.fnOpt("gdn_abconv_q8_0");
    const hip::Function fv0 = c.fnOpt("gdn_step_norm_v0");
    std::size_t launches = 0;
    for (const int cnt : {0, 1, 4, 6}) {
        for (const int nrows : {n2, 0}) {  // a verify (replay + new rows) and a commitSeq (replay only)
            const std::string tag = " (count " + std::to_string(cnt) + (nrows > 0 ? ", verify of 3 rows)" : ", commitSeq form)");
            const std::vector<int> cw{cnt, -3, -3, -3};
            Buf dcnt(cw);
            // conv: record rows 0..rec-1 into pend once, then the replay launch both ways from identical buffers
            auto convRun = [&](hip::Function f, bool ab, bool dev, std::vector<float>& state, std::vector<float>& pend, std::vector<float>& out) {
                Buf cs(cst0), pd(static_cast<std::size_t>(16) * kCh * 4), ot(static_cast<std::size_t>(rec + n2) * kCh * 4);
                pd.zero();
                ot.zero();
                wk::GdnSegs s;
                s.s[0].state = cs.p();
                s.s[0].nrows = rec;
                s.s[0].pend = pd.p();
                auto go = [&](const wk::GdnSegs& sg, DevPtr x) {
                    if (ab)
                        hip::launch(f, {kCh / 128, 1, 1}, {256, 1, 1}, 0, c.s, DevPtr{0}, DevPtr{0}, std::uint64_t{0}, DevPtr{0}, DevPtr{0}, DevPtr{0},
                                    DevPtr{0}, DevPtr{0}, DevPtr{0}, 0, 4, 0, x, dconvw.p(), ot.p(), kCh, 2 * kNk, eps, sg);
                    else
                        hip::launch(f, {kCh / 128, 1, 1}, {128, 1, 1}, 0, c.s, x, dconvw.p(), ot.p(), kCh, 2 * kNk, eps, sg);
                };
                go(s, dxin.p());  // the previous verify (records pend, state untouched)
                s.s[0].nrows = nrows;
                if (dev) {
                    s.s[0].npend = -1;
                    s.s[0].snap[0] = dcnt.p();
                } else {
                    s.s[0].npend = cnt;
                }
                go(s, dxin.p() + static_cast<std::uint64_t>(rec) * kCh * 4);
                c.sync();
                state = cs.down<float>(3 * kCh);
                pend = pd.down<float>(static_cast<std::size_t>(16) * kCh);
                out = ot.down<float>(static_cast<std::size_t>(rec + n2) * kCh);
                launches += 2;
            };
            for (const bool ab : {false, true}) {
                const hip::Function f = ab ? fab : c.k.gdn_conv_l2;
                const std::string kn = ab ? "gdn_abconv_q8_0 (conv blocks)" : "gdn_conv_l2";
                if (!f) {
                    c.rep.skip("e1in", kn + " replay count", "kernel not present");
                    continue;
                }
                std::vector<float> s_v, p_v, o_v, s_d, p_d, o_d;
                convRun(f, ab, false, s_v, p_v, o_v);
                convRun(f, ab, true, s_d, p_d, o_d);
                c.rep.add(cmpExact(kn + " device count == by value: state" + tag, s_d, s_v, Kind::invariant));
                c.rep.add(cmpExact(kn + " device count == by value: pend" + tag, p_d, p_v, Kind::invariant));
                c.rep.add(cmpExact(kn + " device count == by value: out" + tag, o_d, o_v, Kind::invariant));
            }
            // delta rule + gated norm + q8 (gdn_step_norm, and the per-row version gdn_step_norm_v0)
            auto stepRun = [&](hip::Function f, bool dev, std::vector<float>& state, std::vector<float>& pend, std::vector<float>& out,
                               std::vector<std::int8_t>& xq, std::vector<float>& xd) {
                const std::size_t no = static_cast<std::size_t>(rec + n2) * kDInner;
                Buf st(st0), pd(static_cast<std::size_t>(16) * kNv * kPW * 4), y(no * 4), q(no), d(no / 32 * 4);
                pd.zero();
                y.zero();
                q.zero();
                d.zero();
                wk::GdnSegs s;
                s.s[0].state = st.p();
                s.s[0].nrows = rec;
                s.s[0].pend = pd.p();
                auto go = [&](const wk::GdnSegs& sg, std::uint64_t r0) {
                    hip::launch(f, {kNv, 1, 1}, {512, 1, 1}, 0, c.s, dq.p() + r0 * kCh * 4, dg.p() + r0 * kNv * 4, db.p() + r0 * kNv * 4,
                                dz.p() + r0 * kDInner * 4, dwn.p(), y.p() + r0 * kDInner * 4, q.p() + r0 * kDInner, d.p() + r0 * (kDInner / 32) * 4,
                                kNk, scale, eps, kNv, kCh, sg);
                };
                go(s, 0);
                s.s[0].nrows = nrows;
                if (dev) {
                    s.s[0].npend = -1;
                    s.s[0].snap[0] = dcnt.p();
                } else {
                    s.s[0].npend = cnt;
                }
                go(s, static_cast<std::uint64_t>(rec));
                c.sync();
                state = st.down<float>(st0.size());
                pend = pd.down<float>(static_cast<std::size_t>(16) * kNv * kPW);
                out = y.down<float>(no);
                xq = q.down<std::int8_t>(no);
                xd = d.down<float>(no / 32);
                launches += 2;
            };
            for (const bool v0 : {false, true}) {
                const hip::Function f = v0 ? fv0 : c.k.gdn_step_norm;
                const std::string kn = v0 ? "gdn_step_norm_v0" : "gdn_step_norm";
                if (!f) {
                    c.rep.skip("e1in", kn + " replay count", "kernel not in this code object");
                    continue;
                }
                std::vector<float> s_v, p_v, o_v, d_v, s_d, p_d, o_d, d_d;
                std::vector<std::int8_t> q_v, q_d;
                stepRun(f, false, s_v, p_v, o_v, q_v, d_v);
                stepRun(f, true, s_d, p_d, o_d, q_d, d_d);
                c.rep.add(cmpExact(kn + " device count == by value: state" + tag, s_d, s_v, Kind::invariant));
                c.rep.add(cmpExact(kn + " device count == by value: pend" + tag, p_d, p_v, Kind::invariant));
                c.rep.add(cmpExact(kn + " device count == by value: out" + tag, o_d, o_v, Kind::invariant));
                c.rep.add(cmpExact(kn + " device count == by value: xq" + tag, q_d, q_v, Kind::invariant));
                c.rep.add(cmpExact(kn + " device count == by value: xd" + tag, d_d, d_v, Kind::invariant));
            }
        }
    }
    c.rep.add(branchHits("gdn replay-count launches run " + std::to_string(launches), launches, launches > 0));
}

// ---- v0252-e1s8: the DeltaNet SELECT mode (gfx1151 snapshot mode, kernels/gdn_select.h). Round 2: the select entries are
// their own kernels (kernels/gfx1151/gdn_select.hip: gdn_conv_l2_sel, gdn_abconv_sel_<T>, gdn_step_norm_sel); the shipping
// entries are untouched. A select launch (pend = an NPEND word c + 1, npend = m, snap[0 .. m] = candidates, state = spare)
// against the SHIPPING in-place launch from a copy of candidate c: the spare == the in-place final state, each output
// snapshot == the in-place snapshot, out / xq / xd equal, the candidate read and the candidates not written untouched -
// bitwise, for m 1 / 3, nd 1 .. m, every c in 0 .. m, in two head shapes: Swift-like (16 k-heads / 48 v-heads) and Ornith
// (16 k-heads / 32 v-heads).
void testGdnSelect(Ctx& c) {
    if (!c.k.caps.gdn_select || !c.k.gdn_conv_l2_sel || !c.k.gdn_step_norm_sel) {
        c.rep.skip("e1in", "gdn select mode", "select-mode fused DeltaNet kernels not in this code object (gfx1201: replay mode)");
        return;
    }
    const float eps = 1e-6f, scale = 1.0f / std::sqrt(128.0f);
    const int maxr = 4;
    std::size_t launches = 0;
    struct Shape {
        const char* name;
        int nk, nv;
    };
    for (const Shape sh : {Shape{"swift-like 16/48", 16, 48}, Shape{"ornith 16/32", 16, 32}}) {
        const int nk = sh.nk, nv = sh.nv;
        const int ch = 2 * nk * kDk + nv * kDv, d_inner = nv * kDv;
        const std::vector<float> convw = c.randn(static_cast<std::size_t>(ch) * 4, 0.5f);
        const std::vector<float> xin = c.randn(static_cast<std::size_t>(maxr) * ch, 0.5f);
        const std::vector<float> qkv = c.randn(static_cast<std::size_t>(maxr) * ch, 0.1f);
        std::vector<float> g = c.randn(static_cast<std::size_t>(maxr) * nv, 0.1f);
        for (float& v : g) v = -std::fabs(v);
        const std::vector<float> beta = c.randu(static_cast<std::size_t>(maxr) * nv, 0.f, 1.f);
        const std::vector<float> z = c.randn(static_cast<std::size_t>(maxr) * d_inner);
        const std::vector<float> wn = c.randn(kDv, 0.5f);
        Buf dconvw(convw), dxin(xin), dq(qkv), dg(g), db(beta), dz(z), dwn(wn);
        for (const int m : {1, 3}) {
            for (int nd = 1; nd <= m; ++nd) {
                for (int cc = 0; cc <= m; ++cc) {
                    const std::string tag = std::string(" (") + sh.name + ", m " + std::to_string(m) + ", nd " + std::to_string(nd) + ", c " +
                                            std::to_string(cc) + ")";
                    const int nrows = nd + 1;
                    const std::vector<int> word{cc + 1};
                    Buf dword(word);
                    // ---- conv: gdn_conv_l2_sel vs gdn_conv_l2, gdn_abconv_sel_<T> vs gdn_abconv_<T> (conv blocks); state 3 * ch
                    struct ConvPair {
                        std::string name;
                        hip::Function ship, sel;
                        bool ab;
                    };
                    const ConvPair pairs[] = {{"gdn_conv_l2", c.k.gdn_conv_l2, c.k.gdn_conv_l2_sel, false},
                                              {"gdn_abconv_q8_0 (conv blocks)", c.fnOpt("gdn_abconv_q8_0"), c.fnOpt("gdn_abconv_sel_q8_0"), true},
                                              {"gdn_abconv_mxfp4 (conv blocks)", c.fnOpt("gdn_abconv_mxfp4"), c.fnOpt("gdn_abconv_sel_mxfp4"), true}};
                    for (const ConvPair& cp : pairs) {
                        if (!cp.ship || !cp.sel) {
                            c.rep.skip("e1in", cp.name + " select", "kernel not present");
                            continue;
                        }
                        const std::size_t sn = 3 * static_cast<std::size_t>(ch);
                        std::vector<std::vector<float>> cand;
                        for (int j = 0; j <= m; ++j) cand.push_back(c.randn(sn));
                        auto go = [&](hip::Function f, const wk::GdnSegs& sgs, DevPtr out) {
                            if (cp.ab)
                                hip::launch(f, {static_cast<unsigned>(ch / 128), 1, 1}, {256, 1, 1}, 0, c.s, DevPtr{0}, DevPtr{0}, std::uint64_t{0}, DevPtr{0},
                                            DevPtr{0}, DevPtr{0}, DevPtr{0}, DevPtr{0}, DevPtr{0}, 0, 4, 0, dxin.p(), dconvw.p(), out, ch, 2 * nk, eps, sgs);
                            else
                                hip::launch(f, {static_cast<unsigned>(ch / 128), 1, 1}, {128, 1, 1}, 0, c.s, dxin.p(), dconvw.p(), out, ch, 2 * nk, eps, sgs);
                        };
                        Buf live(cand[static_cast<std::size_t>(cc)]), o_p(static_cast<std::size_t>(maxr) * ch * 4);
                        std::vector<Buf> sp;
                        for (int r = 0; r < nd; ++r) sp.emplace_back(sn * 4);
                        o_p.zero();
                        wk::GdnSegs sa;
                        sa.s[0].state = live.p();
                        sa.s[0].nrows = nrows;
                        for (int r = 0; r < nd; ++r) sp[static_cast<std::size_t>(r)].zero(), sa.s[0].snap[r] = sp[static_cast<std::size_t>(r)].p();
                        go(cp.ship, sa, o_p.p());
                        std::vector<Buf> P;
                        for (int j = 0; j <= m; ++j) P.emplace_back(cand[static_cast<std::size_t>(j)]);
                        Buf spare(sn * 4), o_s(static_cast<std::size_t>(maxr) * ch * 4);
                        spare.fill(0x5a);
                        o_s.zero();
                        wk::GdnSegs ss;
                        ss.s[0].state = spare.p();
                        ss.s[0].nrows = nrows;
                        for (int j = 0; j <= m; ++j) ss.s[0].snap[j] = P[static_cast<std::size_t>(j)].p();
                        ss.s[0].pend = dword.p();
                        ss.s[0].npend = m;
                        go(cp.sel, ss, o_s.p());
                        c.sync();
                        launches += 2;
                        const std::string kn = cp.name;
                        c.rep.add(cmpExact(kn + " select: spare == in-place final" + tag, spare.down<float>(sn), live.down<float>(sn), Kind::invariant));
                        c.rep.add(cmpExact(kn + " select: out == in place" + tag, o_s.down<float>(static_cast<std::size_t>(nrows) * ch),
                                           o_p.down<float>(static_cast<std::size_t>(nrows) * ch), Kind::invariant));
                        std::vector<float> got_snap, want_snap, got_keep, want_keep;
                        for (int r = 0; r < nd; ++r) {
                            const std::vector<float> x = P[static_cast<std::size_t>(gdn_sel_out(r, cc))].down<float>(sn), y = sp[static_cast<std::size_t>(r)].down<float>(sn);
                            got_snap.insert(got_snap.end(), x.begin(), x.end());
                            want_snap.insert(want_snap.end(), y.begin(), y.end());
                        }
                        for (int j = 0; j <= m; ++j) {
                            bool written = false;
                            for (int r = 0; r < nd; ++r) written = written || gdn_sel_out(r, cc) == j;
                            if (written) continue;
                            const std::vector<float> x = P[static_cast<std::size_t>(j)].down<float>(sn);
                            got_keep.insert(got_keep.end(), x.begin(), x.end());
                            want_keep.insert(want_keep.end(), cand[static_cast<std::size_t>(j)].begin(), cand[static_cast<std::size_t>(j)].end());
                        }
                        c.rep.add(cmpExact(kn + " select: snapshots == in place" + tag, got_snap, want_snap, Kind::invariant));
                        c.rep.add(cmpExact(kn + " select: the candidate read (and the unused ones) untouched" + tag, got_keep, want_keep, Kind::invariant));
                    }
                    // ---- delta rule: gdn_step_norm_sel vs gdn_step_norm; state nv * 128 * 128
                    {
                        const std::size_t sn = static_cast<std::size_t>(nv) * kDk * kDv;
                        std::vector<std::vector<float>> cand;
                        for (int j = 0; j <= m; ++j) cand.push_back(c.randn(sn, 0.1f));
                        const std::size_t no = static_cast<std::size_t>(nrows) * d_inner;
                        auto go = [&](hip::Function f, const wk::GdnSegs& sgs, Buf& y, Buf& q, Buf& d) {
                            hip::launch(f, {static_cast<unsigned>(nv), 1, 1}, {512, 1, 1}, 0, c.s, dq.p(), dg.p(), db.p(), dz.p(), dwn.p(), y.p(), q.p(), d.p(), nk,
                                        scale, eps, nv, ch, sgs);
                        };
                        Buf live(cand[static_cast<std::size_t>(cc)]), y_p(no * 4), q_p(no), d_p(no / 32 * 4);
                        y_p.zero(), q_p.zero(), d_p.zero();
                        std::vector<Buf> sp;
                        for (int r = 0; r < nd; ++r) sp.emplace_back(sn * 4);
                        wk::GdnSegs sa;
                        sa.s[0].state = live.p();
                        sa.s[0].nrows = nrows;
                        for (int r = 0; r < nd; ++r) sp[static_cast<std::size_t>(r)].zero(), sa.s[0].snap[r] = sp[static_cast<std::size_t>(r)].p();
                        go(c.k.gdn_step_norm, sa, y_p, q_p, d_p);
                        std::vector<Buf> P;
                        for (int j = 0; j <= m; ++j) P.emplace_back(cand[static_cast<std::size_t>(j)]);
                        Buf spare(sn * 4), y_s(no * 4), q_s(no), d_s(no / 32 * 4);
                        spare.fill(0x5a), y_s.zero(), q_s.zero(), d_s.zero();
                        wk::GdnSegs ss;
                        ss.s[0].state = spare.p();
                        ss.s[0].nrows = nrows;
                        for (int j = 0; j <= m; ++j) ss.s[0].snap[j] = P[static_cast<std::size_t>(j)].p();
                        ss.s[0].pend = dword.p();
                        ss.s[0].npend = m;
                        go(c.k.gdn_step_norm_sel, ss, y_s, q_s, d_s);
                        c.sync();
                        launches += 2;
                        const std::string kn = "gdn_step_norm";
                        c.rep.add(cmpExact(kn + " select: spare == in-place final" + tag, spare.down<float>(sn), live.down<float>(sn), Kind::invariant));
                        c.rep.add(cmpExact(kn + " select: out == in place" + tag, y_s.down<float>(no), y_p.down<float>(no), Kind::invariant));
                        c.rep.add(cmpExact(kn + " select: xq == in place" + tag, q_s.down<std::int8_t>(no), q_p.down<std::int8_t>(no), Kind::invariant));
                        c.rep.add(cmpExact(kn + " select: xd == in place" + tag, d_s.down<float>(no / 32), d_p.down<float>(no / 32), Kind::invariant));
                        std::vector<float> got_snap, want_snap;
                        for (int r = 0; r < nd; ++r) {
                            const std::vector<float> x = P[static_cast<std::size_t>(gdn_sel_out(r, cc))].down<float>(sn), y = sp[static_cast<std::size_t>(r)].down<float>(sn);
                            got_snap.insert(got_snap.end(), x.begin(), x.end());
                            want_snap.insert(want_snap.end(), y.begin(), y.end());
                        }
                        c.rep.add(cmpExact(kn + " select: snapshots == in place" + tag, got_snap, want_snap, Kind::invariant));
                        c.rep.add(cmpExact(kn + " select: the candidate read untouched" + tag, P[static_cast<std::size_t>(cc)].down<float>(sn),
                                           cand[static_cast<std::size_t>(cc)], Kind::invariant));
                    }
                }
            }
        }
    }
    c.rep.add(branchHits("gdn select-mode launches run " + std::to_string(launches), launches, launches > 0));
}

// ---- spec_accept_rows: the position from the NPOS word == by value
void testSpecAccDevPos(Ctx& c) {
    if (!c.k.spec_accept_rows) {
        c.rep.skip("e1in", "spec_accept_rows device position", "kernel not present");
        return;
    }
    auto rnd = [&](std::uint32_t n) { return static_cast<int>(c.rng() % n); };
    const int rounds = c.quick ? 64 : 400;
    std::vector<int> g_dp, g_bv, want, t_dp, t_bv, t_want;
    std::size_t dev_pos = 0, npend_ok = 0, seqs = 0;
    for (int it = 0; it < rounds; ++it) {
        std::vector<int> ctl(kWords);
        for (int& v : ctl) v = rnd(1000);
        SpecAccArgs a_dp{}, a_bv{};
        const int A = 1 + rnd(16);
        int ids[16];
        for (int i = 0; i < 16; ++i) ids[i] = i;
        for (int i = 15; i > 0; --i) std::swap(ids[i], ids[rnd(static_cast<std::uint32_t>(i) + 1)]);
        a_dp.n = a_bv.n = A;
        a_dp.n_vocab = a_bv.n_vocab = 1000;
        a_dp.stop0 = a_bv.stop0 = 998;
        a_dp.stop1 = a_bv.stop1 = 999;
        int row = 0;
        std::vector<int> ctl_bv = ctl;
        for (int k = 0; k < A; ++k) {
            const int sq = ids[k];
            int* cw = ctl.data() + sq * WHIRL_SA_CTL_WORDS;
            const int nd = std::min(rnd(WHIRL_SA_MAX_ND + 1), WHIRL_SA_MAX_ROWS - row - (A - k));
            const int match = rnd(static_cast<std::uint32_t>(std::max(nd, 0)) + 2);
            cw[WHIRL_SA_CTL_ND] = rnd(17) - 1;
            for (int r = 0; r < std::max(nd, 0); ++r) {
                const int d = rnd(8) == 0 ? 998 + rnd(2) : rnd(1000);
                cw[WHIRL_SA_CTL_DRAFTS + r] = d;
                if (r < match) cw[WHIRL_SA_CTL_ROWS + r] = d;
            }
            const int P = rnd(100000);
            const bool dev = rnd(3) != 0;
            for (SpecAccArgs* a : {&a_dp, &a_bv}) {
                a->seq[k] = sq;
                a->nd[k] = std::max(nd, 0);
                a->dev[k] = rnd(6) == 0 ? 0 : 1;
                a->eos[k] = 1;
                a->budget[k] = 100;
                a->row0[k] = row;
                a->kvbase[k] = 64 * sq;
            }
            a_bv.dev[k] = a_dp.dev[k];
            a_dp.pos[k] = dev ? -1 : P;
            a_bv.pos[k] = P;
            if (dev) cw[WHIRL_SA_CTL_NPOS] = P;  // the device position (the last accept's NPOS)
            dev_pos += dev;
            row += std::max(nd, 0) + 1;
        }
        // the by-value launch's control words: the same except the NPOS word (overwritten by the kernel anyway)
        for (int k = 0; k < A; ++k) {
            const int sq = ids[k];
            for (int i = 0; i < WHIRL_SA_CTL_WORDS; ++i) ctl_bv[sq * WHIRL_SA_CTL_WORDS + i] = ctl[sq * WHIRL_SA_CTL_WORDS + i];
            ctl_bv[sq * WHIRL_SA_CTL_WORDS + WHIRL_SA_CTL_NPOS] = -5;
        }
        // CPU reference (the device-position arguments; spec_pos reads the NPOS word first)
        std::vector<int> ref = ctl;
        SpecAccResult res[WHIRL_SA_MAX_SEG];
        for (int k = 0; k < A; ++k) {
            int* cw = ref.data() + a_dp.seq[k] * WHIRL_SA_CTL_WORDS;
            const SpecAccResult r = a_dp.dev[k] ? spec_accept_one(a_dp, k, cw) : spec_copy_one(a_dp, k, cw);
            spec_store(a_dp, k, r, cw);
            res[k] = r;
            npend_ok += cw[WHIRL_SA_CTL_NPEND] == r.acc + 1 && cw[WHIRL_SA_CTL_NPOS] == a_bv.pos[k] + r.acc + 1;
            seqs += 1;
        }
        SpecNextTab tw{};
        spec_next_rows(a_dp, ref.data(), res, &tw);
        Buf c_dp(ctl), c_bv(ctl_bv), tb_dp(sizeof(SpecNextTab)), tb_bv(sizeof(SpecNextTab));
        tb_dp.zero();
        tb_bv.zero();
        hip::launch(c.k.spec_accept_rows, {1, 1, 1}, {64, 1, 1}, 0, c.s, c_dp.p(), a_dp, DevPtr{0}, tb_dp.p());
        hip::launch(c.k.spec_accept_rows, {1, 1, 1}, {64, 1, 1}, 0, c.s, c_bv.p(), a_bv, DevPtr{0}, tb_bv.p());
        c.sync();
        const std::vector<int> gd = c_dp.down<int>(kWords), gb = c_bv.down<int>(kWords);
        const std::vector<int> td = tb_dp.down<int>(sizeof(SpecNextTab) / 4), tbv = tb_bv.down<int>(sizeof(SpecNextTab) / 4);
        std::vector<int> twv(sizeof(SpecNextTab) / 4);
        std::memcpy(twv.data(), &tw, sizeof(SpecNextTab));
        g_dp.insert(g_dp.end(), gd.begin(), gd.end());
        g_bv.insert(g_bv.end(), gb.begin(), gb.end());
        want.insert(want.end(), ref.begin(), ref.end());
        t_dp.insert(t_dp.end(), td.begin(), td.end());
        t_bv.insert(t_bv.end(), tbv.begin(), tbv.end());
        t_want.insert(t_want.end(), twv.begin(), twv.end());
    }
    const std::string rs = " (" + std::to_string(rounds) + " cycles)";
    c.rep.add(cmpExact("spec_accept_rows device position == by value: control words" + rs, g_dp, g_bv, Kind::invariant));
    c.rep.add(cmpExact("spec_accept_rows device position vs CPU: control words incl. NPEND" + rs, g_dp, want));
    c.rep.add(cmpExact("spec_accept_rows device position == by value: next-cycle rows" + rs, t_dp, t_bv, Kind::invariant));
    c.rep.add(cmpExact("spec_accept_rows device position vs CPU: next-cycle rows" + rs, t_dp, t_want));
    c.rep.add(branchHits("spec_accept_rows device positions " + std::to_string(dev_pos) + ", reference NPEND / NPOS words right " +
                             std::to_string(npend_ok) + " / " + std::to_string(seqs),
                         seqs, dev_pos > 0 && npend_ok == seqs));
}

}  // namespace

void testE1Inputs(Ctx& c) {
    testSetRowsDp(c);
    testRmsnormRowsDv(c);
    testGdnReplayCount(c);
    testGdnSelect(c);  // v0252-e1s8
    testSpecAccDevPos(c);
}

}  // namespace kt
