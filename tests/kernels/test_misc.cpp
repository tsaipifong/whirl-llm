// whirl-kernel-test: norms / RoPE / elementwise, token picks, requantizers,
// the prefill activation fusions and the MXFP4 x fp8 GEMMs.
//   * rmsnorm, rope_neox, silu_mul, add_inplace vs CPU; rmsnorm_q8 /
//     silu_mul_q8 int8 copies == quantize_q8 of their f32 output (exact);
//     rmsnorm_q8_rows == rmsnorm_q8 on the gathered rows (bitwise);
//   * argmax / argmax_rows / argmax_rows_to / argmax_prob / draft_pick(_rows)
//     tokens and control words exact (first maximum wins), probabilities vs
//     CPU; set_tokens / set_rows exact; topk_rows candidate sets exact;
//     (gfx1201) argmax_rows_to_mb / argmax_rank_rows_to_mb == argmax_rows_to /
//     argmax_rank_rows_to bitwise and vs the CPU scan (v0252-f1arg);
//   * requant_q6k_q4k / requant_q80_q4k exact (CPU emulation); requant_q6k_d2 + gemv_d2_nt*
//     vs CPU on the decoded 2-bit head and multi-token == 1-token (bitwise);
//   * activation fusions (rmsnorm_x8/x16, silu_mul_x8/x16, gated_norm_x8/x16
//     and the f16-input / fragment-tiled variants) == the unfused kernel
//     followed by qact_fp8 / f32_to_f16 (bitwise); qact_fp8 exact vs CPU e4m3;
//   * gemm8_c* vs CPU, gemm8t_c* == gemm8_c0 for every configuration
//     (bitwise), f16-output twins vs the f32 result.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <limits>
#include <numeric>
#include <set>

#include "argmax_key.h"  // AM_WS_BYTES (v0252-f1arg multi-block argmax workspace)
#include "cpu_ref.h"
#include "whirl/spec_sample.h"

namespace kt {

namespace {

int firstMax(const float* x, int n) {
    int b = 0;
    for (int i = 1; i < n; ++i)
        if (x[i] > x[b]) b = i;
    return b;
}

double softmaxTop(const float* x, int n) {
    const int b = firstMax(x, n);
    double s = 0;
    for (int i = 0; i < n; ++i) s += std::exp(static_cast<double>(x[i]) - x[b]);
    return 1.0 / s;
}

Result inv(const std::string& name, bool same, std::size_t n) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = same ? 0 : 1;
    r.pass = same;
    return r;
}

}  // namespace

void testMisc(Ctx& c) {
    c.rep.family = "misc";
    const float eps = 1e-6f;
    // ---------------- norms / elementwise
    {
        const int n = 5120, rows = 6;
        const std::vector<float> x = c.randn(static_cast<std::size_t>(rows) * n, 3.f), w = c.randu(n, 0.2f, 2.f);
        Buf dx(x), dw(w), out(x.size() * 4);
        hip::launch(c.k.rmsnorm, {static_cast<unsigned>(rows), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dw.p(), out.p(), n, n, n, eps);
        c.sync();
        std::vector<double> ref(x.size());
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int i = 0; i < n; ++i) ss += static_cast<double>(x[static_cast<std::size_t>(r) * n + i]) * x[static_cast<std::size_t>(r) * n + i];
            const double s = 1.0 / std::sqrt(ss / n + eps);
            for (int i = 0; i < n; ++i)
                ref[static_cast<std::size_t>(r) * n + i] = x[static_cast<std::size_t>(r) * n + i] * s * w[static_cast<std::size_t>(i)];
        }
        const auto rn = out.down<float>(x.size());
        c.rep.add(cmpTolRel("rmsnorm vs CPU", rn, ref, 1e-5, 1e-7));
        // rmsnorm_q8 (1024 threads; own reduction order) and its int8 copy
        Buf o8(x.size() * 4), xq(x.size()), xd(x.size() / 32 * 4);
        hip::launch(c.k.rmsnorm_q8, {static_cast<unsigned>(rows), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), dw.p(), o8.p(), xq.p(),
                    xd.p(), n, eps);
        c.sync();
        const auto o8h = o8.down<float>(x.size());
        c.rep.add(cmpTolRel("rmsnorm_q8 out vs CPU", o8h, ref, 1e-5, 1e-7));
        std::vector<std::int8_t> rq(x.size());
        std::vector<float> rd(x.size() / 32);
        ref::quantizeQ8(o8h.data(), static_cast<int>(x.size()), rq.data(), rd.data());
        c.rep.add(cmpExact("rmsnorm_q8 xq == quantize_q8(out)", xq.down<std::int8_t>(rq.size()), rq));
        c.rep.add(cmpExact("rmsnorm_q8 xd == quantize_q8(out)", xd.down<float>(rd.size()), rd));
        // gathered rows
        wk::RowIdx src;
        const int pick[4] = {4, 0, 5, 2};
        for (int i = 0; i < 4; ++i) src.v[i] = pick[i];
        Buf o9(4 * static_cast<std::size_t>(n) * 4), xq9(4 * static_cast<std::size_t>(n)), xd9(4 * static_cast<std::size_t>(n) / 32 * 4);
        hip::launch(c.k.rmsnorm_q8_rows, {4, 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), dw.p(), o9.p(), xq9.p(), xd9.p(), n, eps, src);
        c.sync();
        std::vector<float> want;
        for (int i = 0; i < 4; ++i)
            want.insert(want.end(), o8h.begin() + static_cast<std::ptrdiff_t>(pick[i]) * n, o8h.begin() + static_cast<std::ptrdiff_t>(pick[i] + 1) * n);
        c.rep.add(cmpExact("rmsnorm_q8_rows == rmsnorm_q8 of the gathered rows", o9.down<float>(want.size()), want, Kind::invariant));

        // silu_mul, silu_mul_q8, add_inplace
        const int m = 17408 * 2;
        const std::vector<float> g = c.randn(m, 2.f), u = c.randn(m);
        Buf dg(g), du(u);
        hip::launch(c.k.silu_mul, {cdiv(m, 256), 1, 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), m);
        c.sync();
        std::vector<double> rs(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i)
            rs[static_cast<std::size_t>(i)] = g[static_cast<std::size_t>(i)] / (1.0 + std::exp(-static_cast<double>(g[static_cast<std::size_t>(i)]))) * u[static_cast<std::size_t>(i)];
        const auto sm = dg.down<float>(static_cast<std::size_t>(m));
        c.rep.add(cmpTolRel("silu_mul vs CPU", sm, rs, 1e-5, 1e-7));
        Buf dg2(g), q2(static_cast<std::size_t>(m)), d2(static_cast<std::size_t>(m) / 32 * 4);
        hip::launch(c.k.silu_mul_q8, {cdiv(m, 256), 1, 1}, {256, 1, 1}, 0, c.s, dg2.p(), du.p(), q2.p(), d2.p(), m);
        c.sync();
        const auto sm8 = dg2.down<float>(static_cast<std::size_t>(m));
        c.rep.add(cmpExact("silu_mul_q8 g == silu_mul g", sm8, sm, Kind::invariant));
        std::vector<std::int8_t> rq2(static_cast<std::size_t>(m));
        std::vector<float> rd2(static_cast<std::size_t>(m) / 32);
        ref::quantizeQ8(sm8.data(), m, rq2.data(), rd2.data());
        c.rep.add(cmpExact("silu_mul_q8 xq == quantize_q8(g)", q2.down<std::int8_t>(rq2.size()), rq2));
        c.rep.add(cmpExact("silu_mul_q8 xd == quantize_q8(g)", d2.down<float>(rd2.size()), rd2));
        Buf da(g);
        hip::launch(c.k.add_inplace, {cdiv(m, 256), 1, 1}, {256, 1, 1}, 0, c.s, da.p(), du.p(), m);
        c.sync();
        std::vector<float> ra(static_cast<std::size_t>(m));
        for (int i = 0; i < m; ++i) ra[static_cast<std::size_t>(i)] = g[static_cast<std::size_t>(i)] + u[static_cast<std::size_t>(i)];
        c.rep.add(cmpExact("add_inplace", da.down<float>(ra.size()), ra));

        // rope_neox: heads of 256 (q stride 512 like the q/gate layout), 64 rotated dims
        const int heads = 8, hd = 256, stride = 512, n_rot = 64, nt = 4;
        const float ts = std::pow(1.0e7f, -2.0f / n_rot);
        const std::vector<float> q = c.randn(static_cast<std::size_t>(nt) * heads * stride);
        const std::vector<int> pos = {0, 1, 77, 300};
        Buf dq(q), dpos(pos);
        hip::launch(c.k.rope_neox, {static_cast<unsigned>(heads), static_cast<unsigned>(nt), 1}, {64, 1, 1}, 0, c.s, dq.p(), dpos.p(),
                    stride, stride * heads, n_rot, ts);
        c.sync();
        std::vector<double> rr(q.begin(), q.end()), sc(q.size());
        for (int t = 0; t < nt; ++t)
            for (int h = 0; h < heads; ++h) {
                const std::size_t b = (static_cast<std::size_t>(t) * heads + h) * stride;
                for (int i = 0; i < hd; ++i) sc[b + i] = std::fabs(q[b + i]);
                for (int i = 0; i < n_rot / 2; ++i) {
                    const double th = pos[static_cast<std::size_t>(t)] * std::pow(static_cast<double>(ts), i);
                    const double x0 = q[b + i], x1 = q[b + i + n_rot / 2];
                    rr[b + i] = x0 * std::cos(th) - x1 * std::sin(th);
                    rr[b + i + n_rot / 2] = x0 * std::sin(th) + x1 * std::cos(th);
                    sc[b + i] = sc[b + i + n_rot / 2] = std::fabs(x0) + std::fabs(x1);
                }
            }
        c.rep.add(cmpTol("rope_neox vs CPU (positions up to 300)", dq.down<float>(q.size()), rr, sc, 1e-4, 1e-7));
    }

    // ---------------- token picks
    {
        const int n = 248320, rows = 5;
        std::vector<float> x = c.randn(static_cast<std::size_t>(rows) * n, 3.f);
        x[static_cast<std::size_t>(1) * n + 100] = 50.f;  // tie: first index must win
        x[static_cast<std::size_t>(1) * n + 9000] = 50.f;
        x[static_cast<std::size_t>(2) * n + n - 1] = 60.f;  // last element
        Buf dx(x), out(64 * 4), ids(4), pos(std::vector<int>{41});
        std::vector<int> want(static_cast<std::size_t>(rows));
        for (int r = 0; r < rows; ++r) want[static_cast<std::size_t>(r)] = firstMax(x.data() + static_cast<std::size_t>(r) * n, n);
        hip::launch(c.k.argmax, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dx.p() + static_cast<std::uint64_t>(n) * 4, n, out.p(), ids.p(),
                    pos.p(), 1);
        c.sync();
        const bool am = out.down<int>(1)[0] == want[1] && ids.down<int>(1)[0] == want[1] && pos.down<int>(1)[0] == 42;
        Result ra;
        ra.name = "argmax (tie -> first index; ids / pos advance)";
        ra.n = 3;
        ra.mismatches = am ? 0 : 1;
        ra.pass = am;
        c.rep.add(ra);
        hip::launch(c.k.argmax_rows, {static_cast<unsigned>(rows), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), n, out.p());
        c.sync();
        c.rep.add(cmpExact("argmax_rows", out.down<int>(static_cast<std::size_t>(rows)), want));
        wk::RowIdx dst;
        for (int r = 0; r < rows; ++r) dst.v[r] = 10 + 3 * r;
        out.fill(0xff);
        hip::launch(c.k.argmax_rows_to, {static_cast<unsigned>(rows), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), n, out.p(), dst);
        c.sync();
        const auto o2 = out.down<int>(64);
        std::size_t bad = 0;
        for (int r = 0; r < rows; ++r) bad += o2[static_cast<std::size_t>(10 + 3 * r)] != want[static_cast<std::size_t>(r)];
        Result rt;
        rt.name = "argmax_rows_to";
        rt.n = static_cast<std::size_t>(rows);
        rt.mismatches = bad;
        rt.pass = bad == 0;
        c.rep.add(rt);
        // v0252-f1 relaxacc one-pass check: argmax_rank_rows_to = argmax_rows_to + per draft row the draft's rank in the
        // WHOLE row (entries with a greater logit, or an equal logit and a lower id), the row max and the draft's logit
        {
            const hip::Function f = c.fnOpt("argmax_rank_rows_to");
            Result pr;
            pr.kind = Kind::invariant;
            pr.n = 1;
            pr.name = "argmax_rank_rows_to in the code object and the host table (v0252-f1)";
            pr.pass = f != nullptr && c.k.argmax_rank_rows_to == f;
            if (!pr.pass) pr.note = f == nullptr ? "missing from the code object" : "not in MiscKernels (loadMisc)";
            pr.mismatches = pr.pass ? 0 : 1;
            c.rep.add(pr);
            if (pr.pass) {
                const int R = 8, stride = 16;
                std::vector<float> y = c.randn(static_cast<std::size_t>(R) * n, 3.f);
                auto Y = [&](int r, int i) -> float& { return y[static_cast<std::size_t>(r) * n + i]; };
                std::vector<int> dr(R, -1);
                // row 0: the draft is the argmax; row 1: the 3rd largest
                dr[0] = firstMax(&Y(0, 0), n);
                {
                    std::vector<int> o(n);
                    std::iota(o.begin(), o.end(), 0);
                    std::partial_sort(o.begin(), o.begin() + 3, o.end(), [&](int a, int b) { return Y(1, a) != Y(1, b) ? Y(1, a) > Y(1, b) : a < b; });
                    dr[1] = o[2];
                }
                // row 2: a tied maximum (ids 100 / 9000): argmax 100, draft 9000 ranks 1
                Y(2, 100) = 50.f, Y(2, 9000) = 50.f, dr[2] = 9000;
                // row 3: the K-th tie overflow class: max at 500, ids 100 200 300 700 900 tied at 30 -> draft 700 ranks 4
                Y(3, 500) = 40.f;
                for (int i : {900, 100, 700, 300, 200}) Y(3, i) = 30.f;
                dr[3] = 700;
                // row 4: a draft id out of the row (no rank); row 5: no draft (src -1)
                dr[4] = n + 5;
                // row 6: the draft is the last element, tied with the maximum at a lower id
                Y(6, 77) = 60.f, Y(6, n - 1) = 60.f, dr[6] = n - 1;
                // row 7: +0.0 draft between -0.0 neighbours (equal as floats: the lower id is ahead)
                Y(7, 5000) = -0.0f, Y(7, 5001) = 0.0f, Y(7, 5002) = -0.0f, dr[7] = 5001;
                std::vector<int> want_tok(R), want_rank(R);
                std::vector<float> want_m(R), want_ld(R);
                for (int r = 0; r < R; ++r) {
                    const float* row = &Y(r, 0);
                    want_tok[r] = firstMax(row, n);
                    want_m[r] = row[want_tok[r]];
                    const int d = dr[r];
                    if (d < 0 || d >= n) {
                        want_rank[r] = std::numeric_limits<int>::max();
                        want_ld[r] = -std::numeric_limits<float>::infinity();
                        continue;
                    }
                    int k = 0;
                    for (int i = 0; i < n; ++i) k += (row[i] > row[d] || (row[i] == row[d] && i < d)) ? 1 : 0;
                    want_rank[r] = k;
                    want_ld[r] = row[d];
                }
                Buf dy(y), ctl(64 * 4), st(static_cast<std::size_t>(3 * stride) * 4);
                std::vector<int> ctl0(64, -7);
                for (int r = 0; r < R; ++r) ctl0[static_cast<std::size_t>(40 + r)] = dr[r];
                wk::RowIdx d2, s2;
                for (int r = 0; r < 32; ++r) d2.v[r] = -1, s2.v[r] = -1;
                for (int r = 0; r < R; ++r) d2.v[r] = 10 + r, s2.v[r] = dr[r] >= 0 ? 40 + r : -1;
                std::vector<int> first;
                std::size_t diff_launch = 0;
                std::vector<int> g_tok(R), g_rank(R), g_mb(R), g_ldb(R);
                for (int it = 0; it < 20; ++it) {
                    ctl.up(ctl0);
                    st.fill(0x55);
                    hip::launch(c.k.argmax_rank_rows_to, {static_cast<unsigned>(R), 1, 1}, {1024, 1, 1}, 0, c.s, dy.p(), n, ctl.p(), d2, s2, st.p(), stride);
                    c.sync();
                    const auto o = ctl.down<int>(64);
                    const auto sv = st.down<int>(static_cast<std::size_t>(3 * stride));
                    std::vector<int> all(o.begin(), o.end());
                    all.insert(all.end(), sv.begin(), sv.end());
                    if (it == 0) {
                        first = all;
                        for (int r = 0; r < R; ++r) {
                            g_tok[r] = o[static_cast<std::size_t>(10 + r)];
                            g_rank[r] = sv[static_cast<std::size_t>(r)];
                            g_mb[r] = sv[static_cast<std::size_t>(stride + r)];
                            g_ldb[r] = sv[static_cast<std::size_t>(2 * stride + r)];
                        }
                    } else if (all != first) {
                        ++diff_launch;
                    }
                }
                c.rep.add(cmpExact("argmax_rank_rows_to tokens == argmax_rows_to (first maximum wins)", g_tok, want_tok));
                std::vector<int> got_all, want_all;
                for (int r = 0; r < R; ++r) {
                    int mb, lb;
                    std::memcpy(&mb, &want_m[r], 4);
                    std::memcpy(&lb, &want_ld[r], 4);
                    got_all.insert(got_all.end(), {g_rank[r], g_mb[r], g_ldb[r]});
                    want_all.insert(want_all.end(), {want_rank[r], mb, lb});
                }
                c.rep.add(cmpExact("argmax_rank_rows_to rank / max / l_d vs CPU (ties: lower id first; no draft: INT_MAX, -inf)", got_all, want_all));
                std::vector<int> gt, wt;
                for (int r : {2, 3, 6, 7}) gt.push_back(g_rank[r]), wt.push_back(want_rank[r]);
                c.rep.add(cmpExact("argmax_rank_rows_to tie rows (tied max, K-th tie overflow class, +-0) ranks", gt, wt));
                Result rl;
                rl.kind = Kind::invariant;
                rl.name = "argmax_rank_rows_to launches equal (20 launches: tokens, ranks, max, l_d bitwise)";
                rl.n = 19;
                rl.mismatches = diff_launch;
                rl.pass = diff_launch == 0;
                c.rep.add(rl);
            }
        }
        // v0252-f1arg (gfx1201 only): the multi-block twins argmax_rows_to_mb / argmax_rank_rows_to_mb must give bitwise
        // what argmax_rows_to / argmax_rank_rows_to give (control words, rank, row max bits, l_d bits) and what the legacy
        // CPU scan gives, on the verify shapes (n 248320 x 1 / 2 / 3 / 4 / 8 / 32 rows), on n = 3 * 4096 + 5 (scalar tail
        // path, rows not 16-byte aligned) and n = 7 (one partial tile), with edge rows (tied maxima, also across the 4095 /
        // 4096 tile border, the maximum in the last element, all -inf / NaN or one finite value late in such a row, a NaN
        // at index 0, +-0 and denormal maxima, drafts at tile borders, a draft id n / n - 1 / negative, no draft); every
        // twin launch must leave the workspace zeroed; 10 launches equal. v0252-f1arg2 (REVIEW-v0252-f1arg B2 / C1): 10
        // back-to-back launches with no host sync between them give the same words and leave the workspace zeroed, and a
        // grid of 1, tiles - 1 or tiles + 2 blocks per row (not AM_TILES(n)) gives the same bits. Grid and block size come
        // from kernels/argmax_key.h (AM_TILES, AM_THREADS: the kernel's own constants).
        static_assert(AM_MAX_ROWS == wk::kMaxVerifyRows, "AmWs rows (kernels/argmax_key.h) == the verify rows a launch may have");
        {
            const hip::Function fm = c.fnOpt("argmax_rows_to_mb"), fr = c.fnOpt("argmax_rank_rows_to_mb");
            if (fm == nullptr || fr == nullptr) {
                c.rep.skip("misc", "argmax_mb multi-block twins (v0252-f1arg)", "kernel not present in this code object");
            } else {
                const bool tab = c.k.argmax_rows_to_mb == fm && c.k.argmax_rank_rows_to_mb == fr;
                c.rep.add(inv("argmax_mb twins in the host table (loadMisc, v0252-f1arg)", tab, 2));
                const float inf = std::numeric_limits<float>::infinity(), qnan = std::numeric_limits<float>::quiet_NaN();
                const float den = std::numeric_limits<float>::denorm_min();
                // legacy argmax scan (kernels/shared/block_argmax.h order): first index of the maximum over values above
                // -inf, 0 when none; m = x[index] (bit copy) or -inf
                auto legacy = [&](const float* row, int nn, float& m) {
                    float best = -inf;
                    int idx = 0;
                    for (int i = 0; i < nn; ++i)
                        if (row[i] > best) {
                            best = row[i];
                            idx = i;
                        }
                    m = best;
                    return idx;
                };
                struct Shape {
                    int n, rows;
                };
                const Shape shapes[] = {{248320, 1}, {248320, 2}, {248320, 3}, {248320, 4}, {248320, 8}, {248320, 32}, {3 * 4096 + 5, 10}, {7, 3}};
                const int stride = 32, words = 128;
                Buf ws(AM_WS_BYTES);
                ws.zero();
                std::size_t n_rows = 0, bad_plain = 0, bad_rank = 0, bad_cpu = 0, bad_rep = 0, bad_ws = 0, n_ws = 0;
                std::size_t bad_burst = 0, n_burst = 0, bad_grid = 0, n_grid = 0, n_grid_expose = 0;
                auto wsZero = [&]() {
                    const auto w = ws.down<std::uint8_t>(AM_WS_BYTES);
                    ++n_ws;
                    for (const std::uint8_t b : w)
                        if (b != 0) return false;
                    return true;
                };
                for (const Shape& sh : shapes) {
                    const int nn = sh.n, R = sh.rows;
                    std::vector<float> y = c.randn(static_cast<std::size_t>(R) * nn, 3.f);
                    auto Y = [&](int r, int i) -> float& { return y[static_cast<std::size_t>(r) * nn + static_cast<std::size_t>(i % nn)]; };
                    std::vector<int> dr(static_cast<std::size_t>(R), -1);  // draft id per row (-1: no draft word)
                    float mtmp = 0.f;
                    for (int r = 0; r < R; ++r) {
                        float* row = &Y(r, 0);
                        switch (r % 10) {
                            case 0:  // random row, the draft is the argmax (rank 0)
                                dr[r] = legacy(row, nn, mtmp);
                                break;
                            case 1:  // tied maximum (first index wins), the draft is the later copy (rank 1)
                                Y(r, 100) = 50.f, Y(r, 9000) = 50.f, dr[r] = 9000 % nn;
                                break;
                            case 2:  // the maximum in the last element, draft id 0
                                Y(r, nn - 1) = 60.f, dr[r] = 0;
                                break;
                            case 3:  // no value above -inf (NaN / -inf only): argmax 0, max -inf; in odd decades one finite
                                     // value late in such a row (TRT-LLM test_greedy_argmax.py:84-86)
                                for (int i = 0; i < nn; ++i) row[i] = i % 3 == 0 ? qnan : -inf;
                                if ((r / 10) % 2 == 1) Y(r, nn - 3) = -1e30f;
                                dr[r] = 5 % nn;
                                break;
                            case 4:  // a NaN at index 0, the maximum mid-row; the draft at a tile border
                                Y(r, 0) = qnan, Y(r, nn / 2) = 70.f, dr[r] = 4096 % nn;
                                break;
                            case 5:  // +-0 maximum: -0 first (its index wins, the row max keeps its sign), draft +0
                                for (int i = 0; i < nn; ++i) row[i] = -std::fabs(row[i]) - 1.f;
                                Y(r, 5000) = -0.0f, Y(r, 5001) = 0.0f, Y(r, 5002) = -0.0f, dr[r] = 5001 % nn;
                                break;
                            case 6:  // denormal maximum above +0; draft the +0 (two denormals ahead of it)
                                for (int i = 0; i < nn; ++i) row[i] = -std::fabs(row[i]) - 1.f;
                                Y(r, 3) = 0.0f, Y(r, 7) = den, Y(r, 4097) = den, dr[r] = 3 % nn;
                                break;
                            case 7:  // a draft id at the row's end (REVIEW B2): even decades n = just outside the row (no
                                     // rank: INT_MAX, l_d -inf; a kernel testing d <= n would read the next row's first
                                     // logit and report a count), odd decades n - 1 = the last element (a real rank)
                                dr[r] = (r / 10) % 2 == 0 ? nn : nn - 1;
                                break;
                            case 8:  // a negative draft id in the draft word: no rank
                                dr[r] = -3;
                                break;
                            default:  // no draft word (src -1); a tied maximum across the tile border 4095 / 4096 (blocks
                                      // 0 and 1: the lower index 4095 must win; TRT-LLM test_greedy_argmax.py:76-77)
                                dr[r] = -1;
                                Y(r, 4095) = 80.f, Y(r, 4096) = 80.f;
                                break;
                        }
                    }
                    Buf dy(y), ctl(static_cast<std::size_t>(words) * 4), st(static_cast<std::size_t>(3 * stride) * 4);
                    std::vector<int> ctl0(static_cast<std::size_t>(words), -7);
                    wk::RowIdx mdst, msrc;
                    for (int r = 0; r < 32; ++r) mdst.v[r] = -1, msrc.v[r] = -1;
                    for (int r = 0; r < R; ++r) {
                        mdst.v[r] = r;  // argmax words [0, 32), draft words [64, 96): never overlapping (the kernels' contract)
                        if (r % 10 != 9) {
                            msrc.v[r] = 64 + r;
                            ctl0[static_cast<std::size_t>(64 + r)] = dr[r];
                        }
                    }
                    const unsigned tiles = static_cast<unsigned>(AM_TILES(nn));  // the launchers' grid (argmax_key.h)
                    // one twin launch with gx blocks per row (no sync): 1 = argmax_rows_to_mb, 3 = argmax_rank_rows_to_mb
                    auto launchMb = [&](int which, unsigned gx) {
                        if (which == 1)
                            hip::launch(fm, {gx, static_cast<unsigned>(R), 1}, {AM_THREADS, 1, 1}, 0, c.s, dy.p(), nn, ctl.p(), mdst, ws.p());
                        else
                            hip::launch(fr, {gx, static_cast<unsigned>(R), 1}, {AM_THREADS, 1, 1}, 0, c.s, dy.p(), nn, ctl.p(), mdst, msrc, st.p(), stride,
                                        ws.p());
                    };
                    auto words_now = [&]() {  // control words + stats after a sync
                        c.sync();
                        std::vector<int> all = ctl.down<int>(static_cast<std::size_t>(words));
                        const auto sv = st.down<int>(static_cast<std::size_t>(3 * stride));
                        all.insert(all.end(), sv.begin(), sv.end());
                        return all;
                    };
                    // 0 old plain, 1 new plain, 2 old rank, 3 new rank (new: gx blocks per row) -> control words + stats
                    auto run = [&](int which, unsigned gx) {
                        ctl.up(ctl0);
                        st.fill(0x55);
                        if (which == 0)
                            hip::launch(c.k.argmax_rows_to, {static_cast<unsigned>(R), 1, 1}, {1024, 1, 1}, 0, c.s, dy.p(), nn, ctl.p(), mdst);
                        else if (which == 2)
                            hip::launch(c.k.argmax_rank_rows_to, {static_cast<unsigned>(R), 1, 1}, {1024, 1, 1}, 0, c.s, dy.p(), nn, ctl.p(), mdst, msrc,
                                        st.p(), stride);
                        else
                            launchMb(which, gx);
                        return words_now();
                    };
                    const std::vector<int> a0 = run(0, tiles), a1 = run(1, tiles);
                    bad_ws += wsZero() ? 0 : 1;
                    const std::vector<int> a2 = run(2, tiles), a3 = run(3, tiles);
                    bad_ws += wsZero() ? 0 : 1;
                    n_rows += static_cast<std::size_t>(R);
                    for (std::size_t i = 0; i < a0.size(); ++i) bad_plain += a0[i] != a1[i];
                    for (std::size_t i = 0; i < a2.size(); ++i) bad_rank += a2[i] != a3[i];
                    for (int it = 1; it < 10; ++it) {
                        bad_rep += run(it % 2 == 0 ? 1 : 3, tiles) != (it % 2 == 0 ? a1 : a3);
                        bad_ws += wsZero() ? 0 : 1;
                    }
                    // REVIEW B2: 10 back-to-back launches, plain and rank twins alternating on one workspace with no host
                    // sync between them (the verify loop once V1-61 drops the per-cycle sync); the last (rank) launch
                    // leaves a3's words, and the workspace is zeroed after the burst
                    ctl.up(ctl0);
                    st.fill(0x55);
                    for (int it = 0; it < 10; ++it) launchMb(it % 2 == 0 ? 1 : 3, tiles);
                    bad_burst += words_now() != a3;
                    bad_burst += wsZero() ? 0 : 1;
                    ++n_burst;
                    // REVIEW C1: the twins must not depend on the launcher's grid: 1 block per row (every tile after the
                    // first in the kernel's slow loop), tiles - 1 (one block takes two tiles) and tiles + 2 (empty blocks)
                    for (const unsigned gx : {1u, tiles - 1u, tiles + 2u}) {
                        if (gx < 1u || gx == tiles) continue;
                        bad_grid += run(1, gx) != a1;
                        bad_ws += wsZero() ? 0 : 1;
                        bad_grid += run(3, gx) != a3;
                        bad_ws += wsZero() ? 0 : 1;
                        n_grid += 2;
                    }
                    // the twins vs the legacy CPU scan and the F1 rank rule
                    for (int r = 0; r < R; ++r) {
                        const float* row = &Y(r, 0);
                        float m = 0.f;
                        const int tok = legacy(row, nn, m);
                        const int d = r % 10 != 9 ? dr[r] : -1;
                        const bool has = d >= 0 && d < nn;
                        int rank = 0;  // the F1 rule written out (sample_pick_set_tokens.h:80), not argmax_key.h's helper
                        if (has)
                            for (int i = 0; i < nn; ++i) rank += (row[i] > row[d] || (row[i] == row[d] && i < d)) ? 1 : 0;
                        const float ld = has ? row[d] : -inf;
                        int mb, lb;
                        std::memcpy(&mb, &m, 4);
                        std::memcpy(&lb, &ld, 4);
                        const std::size_t sb = static_cast<std::size_t>(words);
                        bad_cpu += a1[static_cast<std::size_t>(r)] != tok;
                        bad_cpu += a3[static_cast<std::size_t>(r)] != tok;
                        bad_cpu += a3[sb + static_cast<std::size_t>(r)] != (has ? rank : std::numeric_limits<int>::max());
                        bad_cpu += a3[sb + static_cast<std::size_t>(stride + r)] != mb;
                        bad_cpu += a3[sb + static_cast<std::size_t>(2 * stride + r)] != lb;
                        // the grid check bites only on rows whose maximum lies outside tile 0 (what one block per row
                        // reads without the kernel's remaining-tile loop)
                        if (tiles > 1u && tok >= AM_TILE) ++n_grid_expose;
                    }
                }
                Result r1, r2, r3, r4, r5, r6, r7;
                r1.name = "argmax_rows_to_mb == argmax_rows_to (control words bitwise; 8 shapes incl. 248320 x 1..32, edge rows)";
                r2.name = "argmax_rank_rows_to_mb == argmax_rank_rows_to (control words, rank, max bits, l_d bits)";
                r3.name = "argmax_mb twins vs the legacy CPU scan (first maximum, -inf -> 0) and the F1 rank rule";
                r4.name = "argmax_mb twins: 10 launches on one workspace equal";
                r5.name = "argmax_mb twins leave the workspace zeroed (after every launch)";
                r6.name = "argmax_mb twins: 10 back-to-back launches without a host sync equal (workspace zeroed after)";
                r7.name = "argmax_mb twins: grids of 1, tiles - 1 and tiles + 2 blocks per row give the same bits";
                r1.kind = r2.kind = r4.kind = r5.kind = r6.kind = r7.kind = Kind::invariant;
                r1.n = r2.n = r3.n = n_rows;
                r4.n = 9 * (sizeof(shapes) / sizeof(shapes[0]));
                r5.n = n_ws;
                r6.n = n_burst;
                r7.n = n_grid;
                r1.mismatches = bad_plain;
                r2.mismatches = bad_rank;
                r3.mismatches = bad_cpu;
                r4.mismatches = bad_rep;
                r5.mismatches = bad_ws;
                r6.mismatches = bad_burst;
                r7.mismatches = bad_grid + (n_grid_expose == 0 ? 1u : 0u);  // no exposing row = the check proves nothing
                r7.note = "rows with the maximum outside tile 0: " + std::to_string(n_grid_expose);
                for (Result* q : {&r1, &r2, &r3, &r4, &r5, &r6, &r7}) {
                    q->pass = q->mismatches == 0;
                    c.rep.add(*q);
                }
            }
        }
        // argmax_prob
        Buf tok(4), prob(4);
        hip::launch(c.k.argmax_prob, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), n, tok.p(), prob.p());
        c.sync();
        c.rep.add(cmpExact("argmax_prob token", tok.down<int>(1), std::vector<int>{want[0]}));
        c.rep.add(cmpTolRel("argmax_prob probability vs CPU", prob.down<float>(1), {softmaxTop(x.data(), n)}, 1e-4, 1e-9));
        // draft_pick chain: p_min cutoff after n_min drafts
        {
            std::vector<float> lg = c.randn(static_cast<std::size_t>(3) * n);
            lg[5] = 30.f;                                       // r = 0: confident
            lg[static_cast<std::size_t>(n) + 7] = 4.f;          // r = 1: unconfident -> stops (n_min = 1)
            lg[static_cast<std::size_t>(2) * n + 9] = 30.f;     // r = 2: skipped (chain stopped)
            Buf dl(lg), dt(16 * 4), dp(16 * 4), ctl(8);
            dt.fill(0x7f);
            for (int r = 0; r < 3; ++r)
                hip::launch(c.k.draft_pick, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dl.p() + static_cast<std::uint64_t>(r) * n * 4, n, dt.p(),
                            dp.p(), ctl.p(), r, 1, 0.5f, hip::DevPtr{0});
            c.sync();
            const auto ht = dt.down<int>(3);
            const auto hc = ctl.down<int>(2);
            const bool ok = ht[0] == 5 && ht[1] == firstMax(lg.data() + n, n) && ht[2] == 0x7f7f7f7f && hc[0] == 1 && hc[1] == 1;
            Result rd;
            rd.name = "draft_pick chain (tokens, p-min cutoff ctl words, stopped chain skips)";
            rd.n = 5;
            rd.mismatches = ok ? 0 : 1;
            rd.pass = ok;
            if (!ok)
                rd.note = "tok " + std::to_string(ht[0]) + "," + std::to_string(ht[1]) + "," + std::to_string(ht[2]) + " ctl " +
                          std::to_string(hc[0]) + "," + std::to_string(hc[1]);
            c.rep.add(rd);
            c.rep.add(cmpTolRel("draft_pick probability r=1 vs CPU", dp.down<float>(2), {softmaxTop(lg.data(), n), softmaxTop(lg.data() + n, n)},
                                1e-4, 1e-9));
            // draft_pick_rows: two sequences' control areas, same logits rows as r = 0 / 1 above
            Buf ctl_all(2 * wk::kCtlWords * 4);
            ctl_all.zero();
            wk::RowIdx area;
            area.v[0] = 0;
            area.v[1] = wk::kCtlWords;
            hip::launch(c.k.draft_pick_rows, {2, 1, 1}, {1024, 1, 1}, 0, c.s, dl.p(), n, ctl_all.p(), area, 0, 1, 0.5f, hip::DevPtr{0});
            c.sync();
            const auto ca = ctl_all.down<int>(2 * wk::kCtlWords);
            const bool okr = ca[wk::kCtlDrafts] == ht[0] && ca[wk::kCtlWords + wk::kCtlDrafts] == ht[1] && ca[wk::kCtlNd] == 1 &&
                             ca[wk::kCtlWords + wk::kCtlNd] == 1;
            Result rr;
            rr.name = "draft_pick_rows (per-sequence control areas)";
            rr.n = 4;
            rr.mismatches = okr ? 0 : 1;
            rr.pass = okr;
            c.rep.add(rr);
            // draft-head vocabulary subset: an identity map gives the same tokens as no map; any map
            // returns map[top] (draft_pick and draft_pick_rows)
            {
                std::vector<int> ident(static_cast<std::size_t>(n)), perm(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) {
                    ident[static_cast<std::size_t>(i)] = i;
                    perm[static_cast<std::size_t>(i)] = 100000 + 3 * i;
                }
                Buf di(ident), dm(perm), t1(16 * 4), p1(16 * 4), c1(8), t2(16 * 4), p2(16 * 4), c2(8);
                for (int r = 0; r < 2; ++r) {
                    const hip::DevPtr row = dl.p() + static_cast<std::uint64_t>(r) * n * 4;
                    hip::launch(c.k.draft_pick, {1, 1, 1}, {1024, 1, 1}, 0, c.s, row, n, t1.p(), p1.p(), c1.p(), r, 0, 0.f, di.p());
                    hip::launch(c.k.draft_pick, {1, 1, 1}, {1024, 1, 1}, 0, c.s, row, n, t2.p(), p2.p(), c2.p(), r, 0, 0.f, dm.p());
                }
                Buf cm(2 * wk::kCtlWords * 4);
                cm.zero();
                hip::launch(c.k.draft_pick_rows, {2, 1, 1}, {1024, 1, 1}, 0, c.s, dl.p(), n, cm.p(), area, 0, 0, 0.f, dm.p());
                c.sync();
                const auto a1 = t1.down<int>(2), a2 = t2.down<int>(2);
                const auto cmh = cm.down<int>(2 * wk::kCtlWords);
                const int top0 = ht[0], top1 = firstMax(lg.data() + n, n);
                const bool okm = a1[0] == top0 && a1[1] == top1 && a2[0] == perm[static_cast<std::size_t>(top0)] &&
                                 a2[1] == perm[static_cast<std::size_t>(top1)] &&
                                 cmh[wk::kCtlDrafts] == perm[static_cast<std::size_t>(top0)] &&
                                 cmh[wk::kCtlWords + wk::kCtlDrafts] == perm[static_cast<std::size_t>(top1)];
                Result rm;
                rm.name = "draft_pick(_rows) with a vocabulary map (identity == no map, map[top])";
                rm.n = 6;
                rm.mismatches = okm ? 0 : 1;
                rm.pass = okm;
                c.rep.add(rm);
            }
            // copy_rows_map: dst row i = src row map[i] (2-bit draft-head rows, 1600 bytes for 5120 columns)
            if (c.k.copy_rows_map != nullptr) {
                const std::uint64_t rb = 1600;
                const int nsrc = 64;
                std::vector<std::uint8_t> src(static_cast<std::size_t>(nsrc) * rb);
                for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>(i * 131 + i / rb * 7);
                const std::vector<int> mp = {5, 0, 63, 17, 18};
                Buf ds(src), dmp(mp), dd(mp.size() * rb);
                hip::launch(c.k.copy_rows_map, {static_cast<unsigned>(mp.size()), 1, 1}, {256, 1, 1}, 0, c.s, ds.p(), rb, dd.p(), dmp.p());
                c.sync();
                const auto got = dd.down<std::uint8_t>(mp.size() * rb);
                std::vector<std::uint8_t> want_rows;
                for (int m : mp) want_rows.insert(want_rows.end(), src.begin() + static_cast<std::ptrdiff_t>(m * rb), src.begin() + static_cast<std::ptrdiff_t>((m + 1) * rb));
                c.rep.add(cmpExact("copy_rows_map (vocabulary subset rows)", got, want_rows));
            } else {
                Result r;
                r.name = "copy_rows_map present";
                r.pass = false;
                c.rep.add(r);
            }
        }
        // set_tokens / set_rows
        {
            Buf ids2(16 * 4), pos2(16 * 4), kvb(16 * 4), devsrc(std::vector<int>{1000, 1001, 1002, 1003});
            wk::Tok16 tk;
            for (int i = 0; i < 16; ++i) tk.t[i] = 500 + i;
            hip::launch(c.k.set_tokens, {1, 1, 1}, {16, 1, 1}, 0, c.s, ids2.p(), pos2.p(), devsrc.p(), tk, 6, 4, 70);
            c.sync();
            const auto hi = ids2.down<int>(6), hp = pos2.down<int>(6);
            const std::vector<int> wi = {500, 501, 502, 503, 1000, 1001}, wp = {70, 71, 72, 73, 74, 75};
            c.rep.add(cmpExact("set_tokens ids", hi, wi));
            c.rep.add(cmpExact("set_tokens pos", hp, wp));
            wk::RowTab tab;
            for (int i = 0; i < 5; ++i) {
                tab.tok[i] = i % 2 ? -(i / 2) - 1 : 600 + i;
                tab.pos[i] = 10 * i;
                tab.base[i] = 100 + i;
            }
            hip::launch(c.k.set_rows, {1, 1, 1}, {16, 1, 1}, 0, c.s, ids2.p(), pos2.p(), kvb.p(), devsrc.p(), tab, 5);
            c.sync();
            c.rep.add(cmpExact("set_rows ids", ids2.down<int>(5), std::vector<int>{600, 1000, 602, 1001, 604}));
            c.rep.add(cmpExact("set_rows pos / kvbase", pos2.down<int>(5), std::vector<int>{0, 10, 20, 30, 40}));
            c.rep.add(cmpExact("set_rows kvbase", kvb.down<int>(5), std::vector<int>{100, 101, 102, 103, 104}));
        }
        // topk_rows: the K largest (unordered) and the softmax normalizer
        if (c.k.topk_rows) {
            const int K = 40, tr = 3;
            const float inv_t = 1.0f / 0.7f;
            Buf tid(static_cast<std::size_t>(tr) * K * 4), tval(static_cast<std::size_t>(tr) * K * 4), stats(static_cast<std::size_t>(tr) * 2 * 4);
            hip::launch(c.k.topk_rows, {static_cast<unsigned>(tr), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), n, K, inv_t, tid.p(), tval.p(),
                        stats.p());
            c.sync();
            const auto hid = tid.down<int>(static_cast<std::size_t>(tr) * K);
            const auto hv = tval.down<float>(hid.size());
            const auto hs = stats.down<float>(static_cast<std::size_t>(tr) * 2);
            std::size_t badk = 0;
            std::vector<double> rsum;
            std::vector<float> gsum;
            for (int r = 0; r < tr; ++r) {
                const float* xr = x.data() + static_cast<std::size_t>(r) * n;
                std::vector<int> idx(static_cast<std::size_t>(n));
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return xr[a] > xr[b]; });
                std::multiset<float> want_v, got_v;
                for (int k = 0; k < K; ++k) {
                    want_v.insert(xr[idx[static_cast<std::size_t>(k)]]);
                    got_v.insert(hv[static_cast<std::size_t>(r) * K + k]);
                    badk += xr[hid[static_cast<std::size_t>(r) * K + k]] != hv[static_cast<std::size_t>(r) * K + k];
                }
                badk += want_v != got_v;
                badk += hs[static_cast<std::size_t>(r) * 2] != xr[firstMax(xr, n)];
                double s = 0;
                for (int i = 0; i < n; ++i) s += std::exp((static_cast<double>(xr[i]) - xr[firstMax(xr, n)]) * inv_t);
                rsum.push_back(s);
                gsum.push_back(hs[static_cast<std::size_t>(r) * 2 + 1]);
            }
            Result rk;
            rk.name = "topk_rows candidate sets / ids / max";
            rk.n = static_cast<std::size_t>(tr) * K;
            rk.mismatches = badk;
            rk.pass = badk == 0;
            c.rep.add(rk);
            c.rep.add(cmpTolRel("topk_rows softmax normalizer vs CPU", gsum, rsum, 1e-4, 1e-9));
        }
        // draft_sample_rows: draft distribution q (ids exact, probabilities vs whirl::spec::draftDist)
        // and the drawn token = the race draw from the written q (whirl::spec::sampleRace)
        if (c.k.draft_sample_rows) {
            namespace sp = whirl::spec;
            const int tr = 3, r = 1;
            Buf ctl(static_cast<std::size_t>(tr) * 64 * 4), qb(static_cast<std::size_t>(tr) * sp::q_words * 4);
            ctl.zero();
            wk::RowIdx area;
            wk::DSampArgs da;
            const int tks[tr] = {20, 0, 5};
            const float tps[tr] = {0.95f, 0.8f, 1.f}, mps[tr] = {0.f, 0.f, 0.05f}, its[tr] = {1.f / 0.7f, 1.f, 1.f / 0.6f};
            const std::uint64_t keys[tr] = {sp::raceBase(1234, 7), sp::raceBase(42, 0), sp::raceBase(987654321ull, 100000)};
            for (int b = 0; b < tr; ++b) {
                area.v[b] = b * 64;
                da.on[b] = 1;
                da.top_k[b] = tks[b];
                da.qoff[b] = b * static_cast<int>(sp::q_words);
                da.inv_t[b] = its[b];
                da.top_p[b] = tps[b];
                da.min_p[b] = mps[b];
                da.key_lo[b] = static_cast<std::uint32_t>(keys[b]);
                da.key_hi[b] = static_cast<std::uint32_t>(keys[b] >> 32);
            }
            hip::launch(c.k.draft_sample_rows, {static_cast<unsigned>(tr), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), n, ctl.p(), area, r,
                        DevPtr{0}, da, qb.p());
            c.sync();
            const auto hc = ctl.down<int>(static_cast<std::size_t>(tr) * 64);
            const auto hq = qb.down<std::int32_t>(static_cast<std::size_t>(tr) * sp::q_words);
            std::size_t bad = 0;
            std::vector<float> gp;
            std::vector<double> rp;
            for (int b = 0; b < tr; ++b) {
                const float* xr = x.data() + static_cast<std::size_t>(b) * n;
                const int K = static_cast<int>(sp::draftCands(static_cast<std::uint32_t>(tks[b])));
                std::vector<int> idx(static_cast<std::size_t>(n));
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int bb) { return xr[a] != xr[bb] ? xr[a] > xr[bb] : a < bb; });
                const float m = xr[firstMax(xr, n)];
                double s = 0;
                for (int i = 0; i < n; ++i) s += std::exp((static_cast<double>(xr[i]) - m) * its[b]);
                std::vector<std::uint32_t> ci(static_cast<std::size_t>(K));
                std::vector<float> cl(static_cast<std::size_t>(K));
                for (int k = 0; k < K; ++k) {
                    ci[static_cast<std::size_t>(k)] = static_cast<std::uint32_t>(idx[static_cast<std::size_t>(k)]);
                    cl[static_cast<std::size_t>(k)] = xr[idx[static_cast<std::size_t>(k)]];
                }
                const sp::QDist want = sp::draftDist(ci, cl, m, static_cast<float>(s), its[b], static_cast<std::uint32_t>(tks[b]), tps[b], mps[b]);
                const sp::QDist got = sp::qFromWords(hq.data() + static_cast<std::size_t>(b) * sp::q_words);
                bad += got.n != want.n;
                for (std::uint32_t i = 0; i < std::min(got.n, want.n); ++i) {
                    bad += got.id[i] != want.id[i];
                    gp.push_back(got.p[i]);
                    rp.push_back(want.p[i]);
                }
                bad += static_cast<std::uint32_t>(hc[static_cast<std::size_t>(b) * 64 + 16 + r]) != sp::sampleRace(got, keys[b]);
            }
            Result rk;
            rk.name = "draft_sample_rows q support / drawn token";
            rk.n = static_cast<std::size_t>(tr);
            rk.mismatches = bad;
            rk.pass = bad == 0;
            c.rep.add(rk);
            c.rep.add(cmpTolRel("draft_sample_rows q probabilities vs CPU", gp, rp, 1e-4, 1e-7));
            // FIX-K97 (FR-K97 P3-2): NaN logits (the row's argmax and two more set to NaN) are never candidates and do
            // not poison the normalizer: q == the CPU distribution with those logits at -inf, finite, drawn token not NaN
            {
                std::vector<float> xn(x.begin(), x.begin() + n);
                const int amax = firstMax(xn.data(), n);
                const int nan_at[3] = {amax, (amax + 17) % n, (amax + 1001) % n};
                for (int i : nan_at) xn[static_cast<std::size_t>(i)] = std::numeric_limits<float>::quiet_NaN();
                std::vector<float> xc = xn;
                for (int i : nan_at) xc[static_cast<std::size_t>(i)] = -INFINITY;
                Buf dxn(xn), ctl1(64 * 4), qb1(sp::q_words * 4);
                ctl1.zero();
                wk::RowIdx a1;
                wk::DSampArgs d1;
                a1.v[0] = 0;
                d1.on[0] = 1;
                d1.top_k[0] = 20;
                d1.qoff[0] = 0;
                d1.inv_t[0] = 1.f / 0.7f;
                d1.top_p[0] = 0.95f;
                d1.min_p[0] = 0.f;
                const std::uint64_t key = sp::raceBase(77, 3);
                d1.key_lo[0] = static_cast<std::uint32_t>(key);
                d1.key_hi[0] = static_cast<std::uint32_t>(key >> 32);
                hip::launch(c.k.draft_sample_rows, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dxn.p(), n, ctl1.p(), a1, r, DevPtr{0}, d1, qb1.p());
                c.sync();
                const auto hc1 = ctl1.down<int>(64);
                const auto hq1 = qb1.down<std::int32_t>(sp::q_words);
                const int K = static_cast<int>(sp::draftCands(20));
                std::vector<int> idx(static_cast<std::size_t>(n));
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int bb) { return xc[a] != xc[bb] ? xc[a] > xc[bb] : a < bb; });
                const float m = xc[firstMax(xc.data(), n)];
                double s = 0;
                for (int i = 0; i < n; ++i) s += std::exp((static_cast<double>(xc[i]) - m) / 0.7);
                std::vector<std::uint32_t> ci(static_cast<std::size_t>(K));
                std::vector<float> cl(static_cast<std::size_t>(K));
                for (int k = 0; k < K; ++k) {
                    ci[static_cast<std::size_t>(k)] = static_cast<std::uint32_t>(idx[static_cast<std::size_t>(k)]);
                    cl[static_cast<std::size_t>(k)] = xc[idx[static_cast<std::size_t>(k)]];
                }
                const sp::QDist want = sp::draftDist(ci, cl, m, static_cast<float>(s), 1.f / 0.7f, 20, 0.95f, 0.f);
                const sp::QDist got = sp::qFromWords(hq1.data());
                std::size_t nb = got.n != want.n;
                std::vector<float> gp1;
                std::vector<double> rp1;
                for (std::uint32_t i = 0; i < std::min(got.n, want.n); ++i) {
                    nb += got.id[i] != want.id[i];
                    nb += !std::isfinite(got.p[i]);
                    for (int j : nan_at) nb += got.id[i] == static_cast<std::uint32_t>(j);
                    gp1.push_back(got.p[i]);
                    rp1.push_back(want.p[i]);
                }
                const int drawn = hc1[16 + r];
                for (int j : nan_at) nb += drawn == j;
                nb += static_cast<std::uint32_t>(drawn) != sp::sampleRace(got, key);
                Result rn;
                rn.name = "draft_sample_rows NaN logits: never candidates, q == CPU with NaN as -inf, drawn token valid";
                rn.n = got.n + 1;
                rn.mismatches = nb;
                rn.pass = nb == 0;
                c.rep.add(rn);
                c.rep.add(cmpTolRel("draft_sample_rows NaN logits: q probabilities vs CPU", gp1, rp1, 1e-4, 1e-7));
            }
        }
    }

    // ---------------- requantizers and the 2-bit draft head
    if (const whirl::gguf::File* f = c.open(c.models.q4)) {
        HostMat head = loadMat(*f, "output.weight", 0, c.quick ? 256 : 2048);  // Q6_K 5120 x 248320 (rows subset)
        if (head.type == QType::q6_k) {
            const int nr = head.nrows, ncols = head.ncols, nsb = ncols / 256;
            Buf src(head.data);
            // Q6_K -> Q4_K
            const std::uint64_t rb4 = wk::rowBytes(QType::q4_k, ncols);
            Buf dst(static_cast<std::size_t>(nr) * rb4);
            hip::launch(c.k.requant_q6k_q4k, {static_cast<unsigned>(nr), static_cast<unsigned>(nsb), 1}, {256, 1, 1}, 0, c.s, src.p(),
                        head.row_bytes, dst.p(), rb4);
            c.sync();
            // CPU emulation of the Q4_K requantizer (same fit, same rounding) for one row
            auto q4kRow = [](const float* v, int nc, std::uint8_t* orow) {
                const int nsbl = nc / 256;
                for (int sb = 0; sb < nsbl; ++sb) {
                    const float* x = v + 256 * sb;
                    float scl[8], mnl[8];
                    for (int j = 0; j < 8; ++j) {
                        float mn = x[32 * j], mx = x[32 * j];
                        for (int i = 0; i < 32; ++i) {
                            mn = std::min(mn, x[32 * j + i]);
                            mx = std::max(mx, x[32 * j + i]);
                        }
                        mn = std::min(0.f, mn);
                        scl[j] = (mx - mn) / 15.f;
                        mnl[j] = -mn;
                    }
                    float ms = 0, mm = 0;
                    for (int j = 0; j < 8; ++j) {
                        ms = std::max(ms, scl[j]);
                        mm = std::max(mm, mnl[j]);
                    }
                    const std::uint16_t dh = f2h(ms / 63.f), dmh = f2h(mm / 63.f);
                    const float df = h2f(dh), dmf = h2f(dmh);
                    int scq[8], mq[8];
                    std::uint8_t qv[256];
                    for (int j = 0; j < 8; ++j) {
                        scq[j] = df > 0.f ? std::min(63, static_cast<int>(std::ceil(scl[j] / df - 1e-4f))) : 0;
                        mq[j] = dmf > 0.f ? std::min(63, static_cast<int>(std::ceil(mnl[j] / dmf - 1e-4f))) : 0;
                        const float step = df * scq[j], off = dmf * mq[j];
                        for (int i = 0; i < 32; ++i) {
                            const int q = step > 0.f ? static_cast<int>(std::nearbyint((x[32 * j + i] + off) / step)) : 0;
                            qv[32 * j + i] = static_cast<std::uint8_t>(std::max(0, std::min(15, q)));
                        }
                    }
                    std::uint8_t* o = orow + static_cast<std::size_t>(sb) * 144;
                    std::memcpy(o, &dh, 2);
                    std::memcpy(o + 2, &dmh, 2);
                    for (int i = 0; i < 4; ++i) {
                        o[4 + i] = static_cast<std::uint8_t>(scq[i] | ((scq[i + 4] >> 4) << 6));
                        o[8 + i] = static_cast<std::uint8_t>(mq[i] | ((mq[i + 4] >> 4) << 6));
                    }
                    for (int i = 4; i < 8; ++i) o[8 + i] = static_cast<std::uint8_t>((scq[i] & 0xF) | ((mq[i] & 0xF) << 4));
                    for (int t = 0; t < 128; ++t) {
                        const int j64 = t >> 5, l = t & 31;
                        o[16 + t] = static_cast<std::uint8_t>(qv[64 * j64 + l] | (qv[64 * j64 + 32 + l] << 4));
                    }
                }
            };
            std::vector<std::uint8_t> want(static_cast<std::size_t>(nr) * rb4);
            for (int r = 0; r < nr; ++r) {
                std::vector<float> v(static_cast<std::size_t>(ncols));
                ref::dequantRow(QType::q6_k, head.data.data() + static_cast<std::size_t>(r) * head.row_bytes, ncols, v.data());
                q4kRow(v.data(), ncols, want.data() + static_cast<std::size_t>(r) * rb4);
            }
            c.rep.add(cmpExact("requant_q6k_q4k (CPU emulation, " + std::to_string(nr) + " head rows)", dst.down<std::uint8_t>(want.size()), want));
            // Q8_0 -> Q4_K (MTP blocks stored as Q8_0): the same fit on the Q8_0 values
            if (c.k.requant_q80_q4k) {
                HostMat q8 = loadMat(*f, "blk.64.attn_k.weight", 0, c.quick ? 128 : 1024);  // Q8_0 5120 x 1024
                if (q8.type == QType::q8_0 && q8.ncols % 256 == 0) {
                    const int nr8 = q8.nrows, nc8 = q8.ncols;
                    const std::uint64_t rb8 = wk::rowBytes(QType::q4_k, nc8);
                    Buf s8(q8.data), d8(static_cast<std::size_t>(nr8) * rb8);
                    hip::launch(c.k.requant_q80_q4k, {static_cast<unsigned>(nr8), static_cast<unsigned>(nc8 / 256), 1}, {256, 1, 1}, 0, c.s,
                                s8.p(), q8.row_bytes, d8.p(), rb8);
                    c.sync();
                    std::vector<std::uint8_t> w8(static_cast<std::size_t>(nr8) * rb8);
                    for (int r = 0; r < nr8; ++r) {
                        std::vector<float> v(static_cast<std::size_t>(nc8));
                        ref::dequantRow(QType::q8_0, q8.data.data() + static_cast<std::size_t>(r) * q8.row_bytes, nc8, v.data());
                        q4kRow(v.data(), nc8, w8.data() + static_cast<std::size_t>(r) * rb8);
                    }
                    c.rep.add(cmpExact("requant_q80_q4k (CPU emulation, " + std::to_string(nr8) + " rows)", d8.down<std::uint8_t>(w8.size()), w8));
                }
            } else {
                Result r;
                r.name = "requant_q80_q4k present";
                r.pass = false;
                c.rep.add(r);
            }

            // Q6_K -> D2 and the D2 GEMV
            if (c.k.requant_q6k_d2) {
                const std::size_t rbd = static_cast<std::size_t>(ncols / 4 + ncols / 16);
                Buf d2(static_cast<std::size_t>(nr) * rbd);
                hip::launch(c.k.requant_q6k_d2, {static_cast<unsigned>(nr), static_cast<unsigned>(nsb), 1}, {256, 1, 1}, 0, c.s, src.p(),
                            head.row_bytes, d2.p(), ncols);
                const int NT = wk::kMaxSmallBatch;
                const std::vector<float> x = c.randn(static_cast<std::size_t>(NT) * ncols);
                Buf dx(x), xq(static_cast<std::size_t>(NT) * ncols), xd(static_cast<std::size_t>(NT) * ncols / 32 * 4);
                hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(NT) * ncols, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), xq.p(),
                            xd.p(), NT * ncols);
                Buf y1(static_cast<std::size_t>(NT) * nr * 4);
                for (int t = 0; t < NT; ++t)
                    hip::launch(c.k.gemv_d2[0], {cdiv(nr, 8), 1, 1}, {256, 1, 1}, 0, c.s, d2.p(),
                                xq.p() + static_cast<std::uint64_t>(t) * ncols, xd.p() + static_cast<std::uint64_t>(t) * (ncols / 32) * 4,
                                y1.p() + static_cast<std::uint64_t>(t) * nr * 4, ncols, nr, DevPtr{0});
                c.sync();
                const auto ref1 = y1.down<float>(static_cast<std::size_t>(NT) * nr);
                std::size_t vb = 0, nv = 0;
                Buf yn(static_cast<std::size_t>(NT) * nr * 4);
                for (int nt = 2; nt <= NT; ++nt) {
                    if (!c.k.gemv_d2[static_cast<std::size_t>(nt - 1)]) continue;
                    hip::launch(c.k.gemv_d2[static_cast<std::size_t>(nt - 1)], {cdiv(nr, 8), 1, 1}, {256, 1, 1}, 0, c.s, d2.p(), xq.p(),
                                xd.p(), yn.p(), ncols, nr, DevPtr{0});
                    c.sync();
                    ++nv;
                    vb += std::memcmp(yn.down<float>(static_cast<std::size_t>(nt) * nr).data(), ref1.data(),
                                      static_cast<std::size_t>(nt) * nr * 4) != 0;
                }
                Result rv;
                rv.name = "gemv_d2_nt{2..16} == gemv_d2_nt1 per token";
                rv.kind = Kind::invariant;
                rv.n = nv;
                rv.mismatches = vb;
                rv.pass = vb == 0;
                c.rep.add(rv);
                // CPU: decode the D2 rows (documented layout) and dot with the int8 x
                const auto hd2 = d2.down<std::uint8_t>(static_cast<std::size_t>(nr) * rbd);
                const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(NT) * ncols);
                const auto hdd = xd.down<float>(hq.size() / 32);
                const int cr = std::min(nr, 256);
                std::vector<float> wd(static_cast<std::size_t>(cr) * ncols);
                double err2 = 0, sig2 = 0;
                for (int r = 0; r < cr; ++r) {
                    const std::uint8_t* row = hd2.data() + static_cast<std::size_t>(r) * rbd;
                    std::vector<float> orig(static_cast<std::size_t>(ncols));
                    ref::dequantRow(QType::q6_k, head.data.data() + static_cast<std::size_t>(r) * head.row_bytes, ncols, orig.data());
                    for (int i = 0; i < ncols; ++i) {
                        const int u = i / 64, p = i % 64, k = p >> 4, kk = (p & 15) >> 2, j = p & 3;
                        std::uint32_t word;
                        std::memcpy(&word, row + 16 * u + 4 * k, 4);
                        const int code = static_cast<int>((word >> (8 * j + 2 * kk)) & 3);
                        std::uint16_t sh;
                        std::memcpy(&sh, row + ncols / 4 + 2 * (i / 32), 2);
                        const float v = h2f(sh) * static_cast<float>(2 * code - 3);
                        wd[static_cast<std::size_t>(r) * ncols + i] = v;
                        err2 += (v - orig[static_cast<std::size_t>(i)]) * static_cast<double>(v - orig[static_cast<std::size_t>(i)]);
                        sig2 += static_cast<double>(orig[static_cast<std::size_t>(i)]) * orig[static_cast<std::size_t>(i)];
                    }
                }
                std::vector<double> ry, rs;
                ref::gemvQ8F64(wd, cr, ncols, hq.data(), hdd.data(), 2, ry, rs);
                std::vector<float> got;
                for (int t = 0; t < 2; ++t)
                    got.insert(got.end(), ref1.begin() + static_cast<std::ptrdiff_t>(t) * nr, ref1.begin() + static_cast<std::ptrdiff_t>(t) * nr + cr);
                c.rep.add(cmpTol("gemv_d2_nt1 vs CPU on the decoded 2-bit rows", got, ry, rs, 1e-4, 1e-6));
                const double rel = std::sqrt(err2 / sig2);
                Result rq;
                rq.name = "requant_q6k_d2 relative RMS error vs Q6_K <= 0.6 (got " + std::to_string(rel) + ")";
                rq.kind = Kind::tol;
                rq.n = static_cast<std::size_t>(cr) * ncols;
                rq.metric = rel / 0.6;
                rq.pass = rel <= 0.6;
                c.rep.add(rq);
            }
        }
    } else {
        c.rep.skip("misc", "requant", "Q4_K_M model not available");
    }

    // ---------------- activation fusions == unfused kernel + conversion
    c.rep.family = "fp8";
    {
        const int n = 7, E = 5120, F = 17408, NV = 48;
        const float* nul = nullptr;
        (void)nul;
        const std::vector<float> x = c.randn(static_cast<std::size_t>(n) * E, 2.f), w = c.randu(E, 0.3f, 1.7f);
        Buf dx(x), dw(w);
        auto conv8 = [&](DevPtr src, int cols, Buf& q, Buf& sx) {
            if (!c.k.qact_fp8) return;  // no fp8 path in this code object (gfx1151)
            hip::launch(c.k.qact_fp8, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, src, q.p(), sx.p(), cols);
        };
        auto conv16 = [&](DevPtr src, int cols, Buf& q) {
            hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(n) * cols / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, src, q.p(),
                        n * cols);
        };
        auto tiled = [&](const std::vector<std::uint8_t>& rowmajor, int cols) {
            std::vector<std::uint8_t> t(rowmajor.size() + static_cast<std::size_t>(16) * cols, 0);
            for (int r = 0; r < n; ++r)
                for (int k = 0; k < cols; ++k) t[ref::x8tOff(r, k, cols)] = rowmajor[static_cast<std::size_t>(r) * cols + k];
            return t;
        };
        struct Case {
            std::string name;
            std::vector<std::uint8_t> got, want;
        };
        auto addSame = [&](const std::string& name, const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b) {
            c.rep.add(cmpExact(name, a, b, Kind::invariant));
        };
        // unfused references
        Buf rn(x.size() * 4);
        hip::launch(c.k.rmsnorm, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dw.p(), rn.p(), E, E, E, eps);
        const std::vector<float> g = c.randn(static_cast<std::size_t>(n) * F, 2.f), u = c.randn(static_cast<std::size_t>(n) * F);
        Buf dg(g), du(u), sm(g);
        hip::launch(c.k.silu_mul, {cdiv(static_cast<std::uint64_t>(n) * F, 256), 1, 1}, {256, 1, 1}, 0, c.s, sm.p(), du.p(), n * F);
        const std::vector<float> o = c.randn(static_cast<std::size_t>(n) * NV * 128), z = c.randn(o.size()), wn = c.randu(128, 0.5f, 1.5f);
        Buf dO(o), dz(z), dwn(wn), gn(o);
        hip::launch(c.k.gdn_gated_norm, {static_cast<unsigned>(n * NV), 1, 1}, {128, 1, 1}, 0, c.s, gn.p(), dz.p(), dwn.p(), 128, eps);
        // f16 copies of z / g / u for the *h variants
        Buf z16(o.size() * 2), g16(g.size() * 2), u16(u.size() * 2);
        hip::launch(c.k.f32_to_f16, {cdiv(o.size() / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dz.p(), z16.p(), static_cast<int>(o.size()));
        hip::launch(c.k.f32_to_f16, {cdiv(g.size() / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dg.p(), g16.p(), static_cast<int>(g.size()));
        hip::launch(c.k.f32_to_f16, {cdiv(u.size() / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, du.p(), u16.p(), static_cast<int>(u.size()));
        c.sync();
        // f32 versions of the f16-rounded inputs (what the *h kernels see)
        std::vector<float> zf, gf, uf;
        for (auto v : z16.down<std::uint16_t>(o.size())) zf.push_back(h2f(v));
        for (auto v : g16.down<std::uint16_t>(g.size())) gf.push_back(h2f(v));
        for (auto v : u16.down<std::uint16_t>(u.size())) uf.push_back(h2f(v));
        Buf dzf(zf), gnh(o), dgf(gf), duf(uf), smh(gf);
        hip::launch(c.k.gdn_gated_norm, {static_cast<unsigned>(n * NV), 1, 1}, {128, 1, 1}, 0, c.s, gnh.p(), dzf.p(), dwn.p(), 128, eps);
        hip::launch(c.k.silu_mul, {cdiv(static_cast<std::uint64_t>(n) * F, 256), 1, 1}, {256, 1, 1}, 0, c.s, smh.p(), duf.p(), n * F);

        struct Ref {
            std::vector<std::uint8_t> q8, q16;
            std::vector<float> sx;
        };
        auto makeRef = [&](DevPtr src, int cols) {
            Ref r;
            Buf q8(static_cast<std::size_t>(n) * cols), sx(static_cast<std::size_t>(n) * 4), q16(static_cast<std::size_t>(n) * cols * 2);
            conv8(src, cols, q8, sx);
            conv16(src, cols, q16);
            c.sync();
            r.q8 = q8.down<std::uint8_t>(static_cast<std::size_t>(n) * cols);
            r.q16 = q16.down<std::uint8_t>(static_cast<std::size_t>(n) * cols * 2);
            r.sx = sx.down<float>(static_cast<std::size_t>(n));
            return r;
        };
        const Ref r_rn = makeRef(rn.p(), E), r_sm = makeRef(sm.p(), F), r_gn = makeRef(gn.p(), NV * 128),
                  r_smh = makeRef(smh.p(), F), r_gnh = makeRef(gnh.p(), NV * 128);

        // qact_fp8 itself vs CPU e4m3
        if (!c.k.qact_fp8) {
            c.rep.skip("fp8", "qact_fp8", "kernel not present");
        } else {
            const auto rnh = rn.down<float>(x.size());
            std::vector<std::uint8_t> q(x.size());
            std::vector<float> sx(static_cast<std::size_t>(n));
            for (int t = 0; t < n; ++t) ref::qactFp8Row(rnh.data() + static_cast<std::size_t>(t) * E, E, q.data() + static_cast<std::size_t>(t) * E, sx[static_cast<std::size_t>(t)]);
            c.rep.add(cmpExact("qact_fp8 bytes vs CPU e4m3 (RNE)", r_rn.q8, q));
            c.rep.add(cmpExact("qact_fp8 scales vs CPU", r_rn.sx, sx));
        }
        auto runX = [&](hip::Function f, std::initializer_list<DevPtr> in, int cols, bool fp8, bool tl, auto... extra) {
            const std::size_t bytes = static_cast<std::size_t>(n) * cols * (fp8 ? 1 : 2) + (tl ? static_cast<std::size_t>(16) * cols : 0);
            Buf q(bytes), sx(static_cast<std::size_t>(n) * 4);
            q.zero();
            const DevPtr* a = in.begin();
            if (in.size() == 2)
                hip::launch(f, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, a[0], a[1], q.p(), sx.p(), extra...);
            else
                hip::launch(f, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, a[0], a[1], a[2], q.p(), sx.p(), extra...);
            c.sync();
            return std::make_pair(q.down<std::uint8_t>(bytes), sx.down<float>(static_cast<std::size_t>(n)));
        };
        auto checkFused = [&](const std::string& name, hip::Function f, std::initializer_list<DevPtr> in, int cols, bool fp8, bool tl,
                              const Ref& r, auto... extra) {
            if (!f) {
                c.rep.skip("fp8", name, "kernel not present");
                return;
            }
            auto [q, sx] = runX(f, in, cols, fp8, tl, extra...);
            if (!fp8 && name.find("relaxed") != std::string::npos) {
                // not bitwise by design (Q4_K_M relaxed mode): the f16 store rounds the exact
                // product once instead of rounding the f32 product -> within one f16 ulp
                std::vector<float> got(q.size() / 2);
                std::vector<double> ref(got.size());
                for (std::size_t i = 0; i < got.size(); ++i) {
                    std::uint16_t a, b;
                    std::memcpy(&a, &q[2 * i], 2);
                    std::memcpy(&b, &r.q16[2 * i], 2);
                    got[i] = h2f(a);
                    ref[i] = h2f(b);
                }
                c.rep.add(cmpTolRel(name + " vs unfused + f32_to_f16 (1 f16 ulp)", got, ref, 1.0 / 1024, 1e-7));
                return;
            }
            if (!fp8) {
                addSame(name + " == unfused + f32_to_f16", q, r.q16);
                return;
            }
            const std::vector<std::uint8_t> want = tl ? tiled(r.q8, cols) : r.q8;
            addSame(name + " == unfused + qact_fp8" + (tl ? " (tiled layout)" : ""), q, want);
            c.rep.add(cmpExact(name + " scales", sx, r.sx, Kind::invariant));
        };
        checkFused("rmsnorm_x8", c.k.rmsnorm_x8, {dx.p(), dw.p()}, E, true, false, r_rn, E, eps);
        checkFused("rmsnorm_x16", c.k.rmsnorm_x16, {dx.p(), dw.p()}, E, false, false, r_rn, E, eps);
        checkFused("rmsnorm_x8t", c.k.rmsnorm_x8t, {dx.p(), dw.p()}, E, true, true, r_rn, E, eps);
        checkFused("silu_mul_x8", c.k.silu_mul_x8, {dg.p(), du.p()}, F, true, false, r_sm, F);
        checkFused("silu_mul_x16", c.k.silu_mul_x16, {dg.p(), du.p()}, F, false, false, r_sm, F);
        checkFused("silu_mul_x8t", c.k.silu_mul_x8t, {dg.p(), du.p()}, F, true, true, r_sm, F);
        checkFused("silu_mul_x8h (f16 g, u)", c.k.silu_mul_x8h, {g16.p(), u16.p()}, F, true, false, r_smh, F);
        checkFused("silu_mul_x16h (f16 g, u; relaxed)", c.k.silu_mul_x16h, {g16.p(), u16.p()}, F, false, false, r_smh, F);
        checkFused("silu_mul_x8ht (f16 g, u)", c.k.silu_mul_x8ht, {g16.p(), u16.p()}, F, true, true, r_smh, F);
        checkFused("gated_norm_x8", c.k.gated_norm_x8, {dO.p(), dz.p(), dwn.p()}, NV * 128, true, false, r_gn, NV, eps);
        checkFused("gated_norm_x16", c.k.gated_norm_x16, {dO.p(), dz.p(), dwn.p()}, NV * 128, false, false, r_gn, NV, eps);
        checkFused("gated_norm_x8t", c.k.gated_norm_x8t, {dO.p(), dz.p(), dwn.p()}, NV * 128, true, true, r_gn, NV, eps);
        checkFused("gated_norm_x8h (f16 z)", c.k.gated_norm_x8h, {dO.p(), z16.p(), dwn.p()}, NV * 128, true, false, r_gnh, NV, eps);
        checkFused("gated_norm_x16h (f16 z)", c.k.gated_norm_x16h, {dO.p(), z16.p(), dwn.p()}, NV * 128, false, false, r_gnh, NV, eps);
        checkFused("gated_norm_x8ht (f16 z)", c.k.gated_norm_x8ht, {dO.p(), z16.p(), dwn.p()}, NV * 128, true, true, r_gnh, NV, eps);
        if (c.k.qact_fp8t) {
            Buf q(static_cast<std::size_t>(n) * E + 16 * static_cast<std::size_t>(E)), sx(static_cast<std::size_t>(n) * 4);
            q.zero();
            hip::launch(c.k.qact_fp8t, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, rn.p(), q.p(), sx.p(), E);
            c.sync();
            addSame("qact_fp8t == qact_fp8 (tiled layout)", q.down<std::uint8_t>(q.bytes()), tiled(r_rn.q8, E));
        }
    }

    // ---------------- rmsnorm_x8h16(t): one norm -> fp8 (+sx) and f16, each bitwise
    // equal to the unfused chain (rmsnorm -> qact_fp8 / qact_fp8t, rmsnorm -> f32_to_f16)
    if (c.k.rmsnorm_x8h16 == nullptr || c.k.rmsnorm_x8h16t == nullptr || c.k.qact_fp8 == nullptr || c.k.qact_fp8t == nullptr) {
        c.rep.skip("fp8", "rmsnorm_x8h16", "kernel not present");
    } else {
        for (const int E : {5120, 2048, 2064}) {
            const std::vector<float> w = c.randu(E, 0.3f, 1.7f);
            Buf dw(w);
            for (const int n : {1, 17, 4096}) {
                const std::size_t ne = static_cast<std::size_t>(n) * E, tl = ne + static_cast<std::size_t>(16) * E;
                const std::vector<float> x = c.randn(ne, 2.f);
                Buf dx(x), rn(ne * 4);
                Buf r8(ne), r8t(tl), rsx(static_cast<std::size_t>(n) * 4), rsxt(static_cast<std::size_t>(n) * 4), r16(ne * 2);
                Buf q8(ne), q8t(tl), sx(static_cast<std::size_t>(n) * 4), sxt(static_cast<std::size_t>(n) * 4), q16(ne * 2), q16t(ne * 2);
                r8t.zero();
                q8t.zero();
                const unsigned nb = static_cast<unsigned>(n);
                hip::launch(c.k.rmsnorm, {nb, 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dw.p(), rn.p(), E, E, E, eps);
                hip::launch(c.k.qact_fp8, {nb, 1, 1}, {256, 1, 1}, 0, c.s, rn.p(), r8.p(), rsx.p(), E);
                hip::launch(c.k.qact_fp8t, {nb, 1, 1}, {256, 1, 1}, 0, c.s, rn.p(), r8t.p(), rsxt.p(), E);
                hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(ne) / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, rn.p(), r16.p(),
                            static_cast<int>(ne));
                hip::launch(c.k.rmsnorm_x8h16, {nb, 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dw.p(), q8.p(), sx.p(), q16.p(), E, eps);
                hip::launch(c.k.rmsnorm_x8h16t, {nb, 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dw.p(), q8t.p(), sxt.p(), q16t.p(), E, eps);
                c.sync();
                const std::string tag = " (E=" + std::to_string(E) + ", n=" + std::to_string(n) + ")";
                const auto w16 = r16.down<std::uint8_t>(ne * 2);
                const auto wsx = rsx.down<float>(static_cast<std::size_t>(n));
                c.rep.add(cmpExact("rmsnorm_x8h16 q8 == rmsnorm + qact_fp8" + tag, q8.down<std::uint8_t>(ne), r8.down<std::uint8_t>(ne),
                                   Kind::invariant));
                c.rep.add(cmpExact("rmsnorm_x8h16 sx == qact_fp8" + tag, sx.down<float>(static_cast<std::size_t>(n)), wsx, Kind::invariant));
                c.rep.add(cmpExact("rmsnorm_x8h16 q16 == rmsnorm + f32_to_f16" + tag, q16.down<std::uint8_t>(ne * 2), w16, Kind::invariant));
                c.rep.add(cmpExact("rmsnorm_x8h16t q8 == rmsnorm + qact_fp8t" + tag, q8t.down<std::uint8_t>(tl), r8t.down<std::uint8_t>(tl),
                                   Kind::invariant));
                c.rep.add(cmpExact("rmsnorm_x8h16t sx == qact_fp8t" + tag, sxt.down<float>(static_cast<std::size_t>(n)),
                                   rsxt.down<float>(static_cast<std::size_t>(n)), Kind::invariant));
                c.rep.add(cmpExact("rmsnorm_x8h16t q16 == rmsnorm + f32_to_f16" + tag, q16t.down<std::uint8_t>(ne * 2), w16, Kind::invariant));
            }
        }
    }

    // ---------------- MXFP4 x fp8 GEMMs
    {
        const std::vector<HostMat> mats = sampleMats(c, c.quick ? 512 : 2048);
        const HostMat* mx = nullptr;
        for (const auto& m : mats)
            if (m.type == QType::mxfp4) mx = &m;
        if (mx && c.k.gemm8[0] && c.k.qact_fp8) {
            const HostMat& m = *mx;
            const int n = 200, ncols = m.ncols, nrows = m.nrows;
            std::uint64_t lossy = 0;
            if (const auto* f = c.open(c.models.mx)) {
                const auto* t = f->tensor("blk.0.ffn_down.weight");
                const auto raw = f->tensorData(*t);
                for (int r = 0; r < nrows; ++r) {
                    std::vector<std::uint8_t> tmp(m.row_bytes);
                    wk::repackMxfp4Row(raw.data() + static_cast<std::size_t>(r) * m.row_bytes, tmp.data(), ncols, &lossy);
                }
            }
            Buf w(m.data), wref(m.ref);
            const std::vector<float> x = c.randn(static_cast<std::size_t>(n) * ncols);
            Buf dx(x), x8(static_cast<std::size_t>(n) * ncols), sx(static_cast<std::size_t>(n) * 4);
            Buf x8t(static_cast<std::size_t>(n + 16) * ncols);
            x8t.zero();
            hip::launch(c.k.qact_fp8, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x8.p(), sx.p(), ncols);
            if (c.k.qact_fp8t)
                hip::launch(c.k.qact_fp8t, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x8t.p(), sx.p(), ncols);
            auto run = [&](hip::Function f, const wk::GemmCfg& cf, DevPtr X, std::size_t eb) {
                Buf y(static_cast<std::size_t>(n) * nrows * eb);
                y.fill(0xff);
                hip::launch(f, {cdiv(n, cf.bn), cdiv(nrows, cf.bm), 1}, {static_cast<unsigned>(cf.nth), 1, 1}, 0, c.s, w.p(), m.row_bytes,
                            wref.p(), X, sx.p(), y.p(), ncols, nrows, n, 0);
                c.sync();
                return y;
            };
            const auto y0 = run(c.k.gemm8[0], wk::kGemm8Cfgs[0], x8.p(), 4).down<float>(static_cast<std::size_t>(n) * nrows);
            // CPU: exact weights, the GPU's fp8 activations
            {
                const auto hq = x8.down<std::uint8_t>(static_cast<std::size_t>(n) * ncols);
                const auto hs = sx.down<float>(static_cast<std::size_t>(n));
                const int cr = std::min(nrows, 256), ct = 16;
                std::vector<float> xf(static_cast<std::size_t>(ct) * ncols);
                for (int t = 0; t < ct; ++t)
                    for (int k = 0; k < ncols; ++k)
                        xf[static_cast<std::size_t>(t) * ncols + k] = ref::e4m3ToF(hq[static_cast<std::size_t>(t) * ncols + k]) * hs[static_cast<std::size_t>(t)];
                const std::vector<float> wf = ref::dequantRows(m, 0, cr);
                std::vector<double> ry, rs;
                ref::gemvF64(wf, cr, ncols, xf.data(), ct, ncols, ry, rs);
                std::vector<float> got;
                for (int t = 0; t < ct; ++t)
                    got.insert(got.end(), y0.begin() + static_cast<std::ptrdiff_t>(t) * nrows, y0.begin() + static_cast<std::ptrdiff_t>(t) * nrows + cr);
                Result r = cmpTol("gemm8_c0 vs CPU (exact MXFP4 weights, GPU fp8 activations)", got, ry, rs, 1e-4, 1e-6);
                r.note += (r.note.empty() ? "" : "; ") + std::string("blocks beyond the exponent fold in this tensor: ") + std::to_string(lossy);
                c.rep.add(r);
            }
            std::size_t nv = 0, bad = 0;
            std::string fb;
            for (int ci = 0; ci < static_cast<int>(wk::kGemm8Cfgs.size()); ++ci) {
                if (ci > 0 && c.k.gemm8[ci]) {
                    ++nv;
                    if (run(c.k.gemm8[ci], wk::kGemm8Cfgs[static_cast<std::size_t>(ci)], x8.p(), 4).down<float>(y0.size()) != y0) {
                        ++bad;
                        if (fb.empty()) fb = "gemm8_c" + std::to_string(ci);
                    }
                }
            }
            for (int ci = 0; ci < static_cast<int>(wk::kGemm8tCfgs.size()); ++ci) {
                if (!c.k.gemm8t[ci]) continue;
                ++nv;
                if (run(c.k.gemm8t[ci], wk::kGemm8tCfgs[static_cast<std::size_t>(ci)], x8t.p(), 4).down<float>(y0.size()) != y0) {
                    ++bad;
                    if (fb.empty()) fb = "gemm8t_c" + std::to_string(ci);
                }
            }
            Result r;
            r.name = "gemm8_c* / gemm8t_c* == gemm8_c0 (" + std::to_string(nv) + " variants)";
            r.kind = Kind::invariant;
            r.n = nv;
            r.mismatches = bad;
            r.pass = bad == 0;
            if (!fb.empty()) r.note = "first " + fb;
            c.rep.add(r);
            // f16-output twins: within one f16 rounding of the f32 result
            std::vector<double> yd(y0.begin(), y0.end());
            for (int ci = 0; ci < static_cast<int>(wk::kGemm8Cfgs.size()); ++ci) {
                for (int tl = 0; tl < 2; ++tl) {
                    hip::Function f = tl ? c.k.gemm8th[ci] : c.k.gemm8h[ci];
                    if (!f || (tl && ci >= static_cast<int>(wk::kGemm8tCfgs.size()))) continue;
                    const auto& cf = tl ? wk::kGemm8tCfgs[static_cast<std::size_t>(ci)] : wk::kGemm8Cfgs[static_cast<std::size_t>(ci)];
                    const auto yh = run(f, cf, tl ? x8t.p() : x8.p(), 2).down<std::uint16_t>(y0.size());
                    std::vector<float> yf(yh.size());
                    for (std::size_t i = 0; i < yh.size(); ++i) yf[i] = h2f(yh[i]);
                    c.rep.add(cmpTolRel(std::string(tl ? "gemm8th_c" : "gemm8h_c") + std::to_string(ci) + " (f16 out) vs f32 result", yf, yd,
                                        1.0 / 2048, 1e-7));
                }
            }
        } else {
            c.rep.skip("fp8", "gemm8", "MXFP4 matrix or kernels not available");
        }
    }
}

}  // namespace kt
