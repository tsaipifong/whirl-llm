// whirl-kernel-test: mixture of experts (qwen35moe) and the GDN beta/alpha
// projections.
//   * moe_logits_f32 vs CPU; moe_topk ids exact (CPU top-k of the same
//     logits, ties -> lower index), weights / shared gate vs CPU;
//   * decode experts moe_gu_<T> / moe_down_<T> (int8 activations) vs CPU;
//   * prefill: moe_route structure (tiles exact, perm / inv consistent),
//     moe_gather_f16 exact, gemm_moe32 == gemm_moe per (token, slot) row and
//     a 40-token batch == the same rows of the 200-token batch (bitwise, the
//     prototype's checkPrefillInvariance MoE part), grouped GEMM vs CPU,
//     moe_act_f16 / moe_combine vs CPU;
//   * gdn_ab_<T> vs CPU and gdn_abconv_<T> == gdn_ab_<T> + gdn_conv_l2 (bitwise).
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>

#include "cpu_ref.h"

namespace kt {

namespace {

struct MoeW {
    HostMat gate, up, down;  // [R * F][E], [R * F][E], [R * E][F]
    std::vector<float> router, shw;  // [R][E], [E]
    int E = 2048, R = 256, K = 8, F = 512;
};

MoeW loadMoe(Ctx& c) {
    MoeW m;
    const whirl::gguf::File* f = c.open(c.models.moe);
    if (f && f->tensor("blk.0.ffn_gate_exps.weight")) {
        m.gate = loadMat(*f, "blk.0.ffn_gate_exps.weight");
        m.up = loadMat(*f, "blk.0.ffn_up_exps.weight");
        m.down = loadMat(*f, "blk.0.ffn_down_exps.weight");
        HostMat r = loadMat(*f, "blk.0.ffn_gate_inp.weight");
        m.router.resize(r.data.size() / 4);
        std::memcpy(m.router.data(), r.data.data(), r.data.size());
        HostMat s = loadMat(*f, "blk.0.ffn_gate_inp_shexp.weight");
        m.shw.resize(s.data.size() / 4);
        std::memcpy(m.shw.data(), s.data.data(), s.data.size());
        m.E = m.gate.ncols;
        m.F = m.down.ncols;
        m.R = static_cast<int>(m.router.size() / static_cast<std::size_t>(m.E));
        m.K = static_cast<int>(f->getUintOr("qwen35moe.expert_used_count", 8));
    } else {
        c.rep.skip("moe", "weights", "MoE model not available, synthetic experts used");
        m.R = 64;
        m.gate = randomMat(c, QType::q4_k, m.R * m.F, m.E);
        m.up = randomMat(c, QType::q4_k, m.R * m.F, m.E);
        m.down = randomMat(c, QType::q6_k, m.R * m.E, m.F);
        m.router = c.randn(static_cast<std::size_t>(m.R) * m.E, 0.05f);
        m.shw = c.randn(static_cast<std::size_t>(m.E), 0.05f);
    }
    return m;
}

// x[t] dequantized from int8 (exact in double)
std::vector<double> deq8(const std::vector<std::int8_t>& q, const std::vector<float>& d, std::size_t off, int n) {
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = static_cast<double>(q[off + i]) * ref::xdScale(d[(off + i) / 32]);
    return x;
}

void dotRow(const HostMat& m, std::size_t row, const std::vector<double>& x, double& y, double& a) {
    std::vector<float> w(static_cast<std::size_t>(m.ncols));
    ref::dequantRow(m.type, m.data.data() + row * m.row_bytes, m.ncols, w.data());
    y = 0;
    a = 0;
    for (int i = 0; i < m.ncols; ++i) {
        const double p = w[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
        y += p;
        a += std::fabs(p);
    }
}

}  // namespace

void testMoe(Ctx& c) {
    c.rep.family = "moe";
    MoeW mw = loadMoe(c);
    const int E = mw.E, R = mw.R, K = mw.K, F = mw.F;
    Buf dgate(mw.gate.data), dup(mw.up.data), ddown(mw.down.data), drouter(mw.router), dshw(mw.shw);

    // ---------------- decode (n <= 16)
    {
        const int n = 5;
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        Buf dh(h), logits(static_cast<std::size_t>(n) * R * 4);
        hip::launch(c.k.moe_logits_f32, {cdiv(R, 8), 1, 1}, {256, 1, 1}, 0, c.s, drouter.p(), dh.p(), logits.p(), R, E, n,
                    DevPtr{0});
        c.sync();
        const auto lg = logits.down<float>(static_cast<std::size_t>(n) * R);
        {
            std::vector<double> ry, rs;
            ref::gemvF64(mw.router, R, E, h.data(), n, E, ry, rs);
            c.rep.add(cmpTol("moe_logits_f32 vs CPU", lg, ry, rs, 1e-5, 1e-7));
        }
        Buf ids(static_cast<std::size_t>(n) * K * 4), w(static_cast<std::size_t>(n) * K * 4), sg(static_cast<std::size_t>(n) * 4);
        hip::launch(c.k.moe_topk, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(),
                    ids.p(), w.p(), sg.p(), R, K, E);
        c.sync();
        const auto gids = ids.down<int>(static_cast<std::size_t>(n) * K);
        const auto gw = w.down<float>(gids.size());
        {
            std::vector<int> rid;
            std::vector<double> rw, rsg;
            for (int t = 0; t < n; ++t) {
                std::vector<int> idx(static_cast<std::size_t>(R));
                std::iota(idx.begin(), idx.end(), 0);
                const float* l = lg.data() + static_cast<std::size_t>(t) * R;
                std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
                double s = 0;
                for (int k = 0; k < K; ++k) s += std::exp(static_cast<double>(l[idx[static_cast<std::size_t>(k)]]) - l[idx[0]]);
                for (int k = 0; k < K; ++k) {
                    rid.push_back(idx[static_cast<std::size_t>(k)]);
                    rw.push_back(std::exp(static_cast<double>(l[idx[static_cast<std::size_t>(k)]]) - l[idx[0]]) / s);
                }
                double d = 0;
                for (int e = 0; e < E; ++e) d += static_cast<double>(h[static_cast<std::size_t>(t) * E + e]) * mw.shw[static_cast<std::size_t>(e)];
                rsg.push_back(1.0 / (1.0 + std::exp(-d)));
            }
            c.rep.add(cmpExact("moe_topk ids == CPU top-k", gids, rid));
            c.rep.add(cmpTolRel("moe_topk weights vs CPU", gw, rw, 1e-5, 1e-7));
            c.rep.add(cmpTolRel("moe_topk shared gate vs CPU", sg.down<float>(rsg.size()), rsg, 1e-5, 1e-7));
        }
        // int8 h
        Buf xq(static_cast<std::size_t>(n) * E), xd(static_cast<std::size_t>(n) * E / 32 * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(n) * E, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(),
                    xq.p(), xd.p(), n * E);
        const int ti_g = static_cast<int>(mw.gate.type), ti_d = static_cast<int>(mw.down.type);
        Buf yg(static_cast<std::size_t>(n) * K * F * 4), yu(static_cast<std::size_t>(n) * K * F * 4);
        if (mw.gate.type == mw.up.type && c.k.moe_gu[ti_g]) {
            hip::launch(c.k.moe_gu[ti_g], {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dgate.p(),
                        dup.p(), mw.gate.row_bytes, F, 2 * F, xq.p(), xd.p(), ids.p(), yg.p(), yu.p(), E, K, DevPtr{0});
            c.sync();
            const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(n) * E);
            const auto hd = xd.down<float>(static_cast<std::size_t>(n) * E / 32);
            const auto g1 = yg.down<float>(static_cast<std::size_t>(n) * K * F), u1 = yu.down<float>(g1.size());
            std::vector<float> got;
            std::vector<double> ref, sc;
            for (int p = 0; p < n * K; p += 3) {  // a sample of pairs, every row
                const int t = p / K, e = gids[static_cast<std::size_t>(p)];
                const auto x = deq8(hq, hd, static_cast<std::size_t>(t) * E, E);
                for (int r = 0; r < F; r += 7) {
                    double y, a;
                    dotRow(mw.gate, static_cast<std::size_t>(e) * F + r, x, y, a);
                    got.push_back(g1[static_cast<std::size_t>(p) * F + r]);
                    ref.push_back(y);
                    sc.push_back(a);
                    dotRow(mw.up, static_cast<std::size_t>(e) * F + r, x, y, a);
                    got.push_back(u1[static_cast<std::size_t>(p) * F + r]);
                    ref.push_back(y);
                    sc.push_back(a);
                }
            }
            c.rep.add(cmpTol(std::string("moe_gu_") + wk::typeSuffix(mw.gate.type) + " vs CPU", got, ref, sc, 1e-4, 1e-6));
        }
        if (c.k.moe_down[ti_d]) {
            // a = silu(g) * u as int8 (silu_mul_q8), then the down projection + combine into x
            Buf aq(static_cast<std::size_t>(n) * K * F), ad(static_cast<std::size_t>(n) * K * F / 32 * 4);
            hip::launch(c.k.silu_mul_q8, {cdiv(static_cast<std::uint64_t>(n) * K * F, 256), 1, 1}, {256, 1, 1}, 0, c.s,
                        yg.p(), yu.p(), aq.p(), ad.p(), n * K * F);
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
            Buf dx(x0), dysh(ysh);
            hip::launch(c.k.moe_down[ti_d], {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, ddown.p(),
                        mw.down.row_bytes, E, aq.p(), ad.p(), ids.p(), w.p(), sg.p(), dysh.p(), dx.p(), F, K, DevPtr{0});
            c.sync();
            const auto hq = aq.down<std::int8_t>(static_cast<std::size_t>(n) * K * F);
            const auto hd = ad.down<float>(hq.size() / 32);
            const auto hsg = sg.down<float>(static_cast<std::size_t>(n));
            const auto gx = dx.down<float>(x0.size());
            std::vector<float> got;
            std::vector<double> ref, sc;
            for (int t = 0; t < n; ++t)
                for (int r = 0; r < E; r += 5) {
                    double acc = 0, a = 0;
                    for (int k = 0; k < K; ++k) {
                        const int p = t * K + k;
                        const auto x = deq8(hq, hd, static_cast<std::size_t>(p) * F, F);
                        double y, aa;
                        dotRow(mw.down, static_cast<std::size_t>(gids[static_cast<std::size_t>(p)]) * E + r, x, y, aa);
                        acc += gw[static_cast<std::size_t>(p)] * y;
                        a += gw[static_cast<std::size_t>(p)] * aa;
                    }
                    const double shv = static_cast<double>(hsg[static_cast<std::size_t>(t)]) * ysh[static_cast<std::size_t>(t) * E + r];
                    got.push_back(gx[static_cast<std::size_t>(t) * E + r]);
                    ref.push_back(x0[static_cast<std::size_t>(t) * E + r] + acc + shv);
                    sc.push_back(std::fabs(x0[static_cast<std::size_t>(t) * E + r]) + a + std::fabs(shv));
                }
            c.rep.add(cmpTol(std::string("moe_down_") + wk::typeSuffix(mw.down.type) + " (+ shared, residual) vs CPU", got, ref,
                             sc, 1e-4, 1e-6));
        }
    }

    // ---------------- prefill
    {
        const int n = 200, pairs = n * K;
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        Buf dh(h), logits(static_cast<std::size_t>(n) * R * 4);
        // logits for 200 tokens: route through the router weights on the CPU-free path (gemv f32 kernel)
        hip::launch(c.k.gemv1[0], {cdiv(R, 8), 1, 1}, {256, 1, 1}, 0, c.s, drouter.p(), static_cast<std::uint64_t>(E) * 4,
                    dh.p(), logits.p(), E, R, E, R, 0);
        // gemv_f32_1 handles one token; use gemm_f32 for the batch
        hip::launch(c.fn("gemm_f32"), {cdiv(R, 16), cdiv(n, 32), 1}, {256, 1, 1}, 0, c.s, drouter.p(),
                    static_cast<std::uint64_t>(E) * 4, dh.p(), logits.p(), E, R, n, 0);
        Buf ids(static_cast<std::size_t>(pairs) * 4), w(static_cast<std::size_t>(pairs) * 4), sg(static_cast<std::size_t>(n) * 4);
        hip::launch(c.k.moe_topk, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(),
                    ids.p(), w.p(), sg.p(), R, K, E);
        c.sync();
        const auto hids = ids.down<int>(static_cast<std::size_t>(pairs));

        struct Run {
            std::vector<int> perm, inv;
            std::vector<wk::Int4> tiles;
            int ntiles = 0;
            std::vector<std::uint16_t> xg;
            std::vector<float> yg;
        };
        auto route = [&](int np, int bn, Run& out, bool check) {
            Buf perm(static_cast<std::size_t>(np) * 4), inv(static_cast<std::size_t>(np) * 4);
            const int max_tiles = (np + bn - 1) / bn + R;
            Buf tiles(static_cast<std::size_t>(max_tiles) * 16), nt(4);
            hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, ids.p(), np, R, bn, perm.p(), inv.p(), tiles.p(),
                        nt.p());
            Buf xg(static_cast<std::size_t>(np) * E * 2);
            hip::launch(c.k.moe_gather_f16, {static_cast<unsigned>(np), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), xg.p(),
                        E, K);
            Buf yg(static_cast<std::size_t>(np) * F * 4);
            yg.fill(0xff);
            auto gm = bn == 32 ? c.k.gemm_moe32 : c.k.gemm_moe;
            hip::launch(gm[static_cast<int>(mw.gate.type)], {static_cast<unsigned>(max_tiles), cdiv(F, wk::kMoeBm), 1},
                        {256, 1, 1}, 0, c.s, dgate.p(), mw.gate.row_bytes, F, xg.p(), yg.p(), E, tiles.p(), nt.p());
            c.sync();
            out.perm = perm.down<int>(static_cast<std::size_t>(np));
            out.inv = inv.down<int>(static_cast<std::size_t>(np));
            out.ntiles = nt.down<int>(1)[0];
            out.tiles = tiles.down<wk::Int4>(static_cast<std::size_t>(max_tiles));
            out.xg = xg.down<std::uint16_t>(static_cast<std::size_t>(np) * E);
            out.yg = yg.down<float>(static_cast<std::size_t>(np) * F);
            if (!check) return;
            // expected tiles from the expert counts
            std::vector<int> cnt(static_cast<std::size_t>(R), 0);
            for (int p = 0; p < np; ++p) ++cnt[static_cast<std::size_t>(hids[static_cast<std::size_t>(p)])];
            std::vector<wk::Int4> want;
            int o = 0;
            for (int r = 0; r < R; ++r) {
                for (int s = 0; s < cnt[static_cast<std::size_t>(r)]; s += bn) want.push_back({r, o + s, std::min(bn, cnt[static_cast<std::size_t>(r)] - s), 0});
                o += cnt[static_cast<std::size_t>(r)];
            }
            bool ok = out.ntiles == static_cast<int>(want.size());
            for (std::size_t i = 0; ok && i < want.size(); ++i)
                ok = std::memcmp(&want[i], &out.tiles[i], 16) == 0;
            std::size_t bad = 0;
            for (int pos = 0; pos < np; ++pos) {
                const int p = out.perm[static_cast<std::size_t>(pos)];
                if (p < 0 || p >= np || out.inv[static_cast<std::size_t>(p)] != pos) ++bad;
                if (pos > 0 && hids[static_cast<std::size_t>(out.perm[static_cast<std::size_t>(pos - 1)])] > hids[static_cast<std::size_t>(p)]) ++bad;
            }
            Result r;
            r.name = "moe_route bn=" + std::to_string(bn) + " (tiles, perm sorted by expert, inv = perm^-1)";
            r.n = static_cast<std::size_t>(np);
            r.mismatches = bad + (ok ? 0 : 1);
            r.pass = r.mismatches == 0;
            c.rep.add(r);
            // gather exact
            std::size_t gbad = 0;
            for (int pos = 0; pos < np; ++pos) {
                const int t = out.perm[static_cast<std::size_t>(pos)] / K;
                for (int e = 0; e < E; ++e) gbad += out.xg[static_cast<std::size_t>(pos) * E + e] != f2h(h[static_cast<std::size_t>(t) * E + e]);
            }
            Result g;
            g.name = "moe_gather_f16 bn=" + std::to_string(bn);
            g.n = static_cast<std::size_t>(np) * E;
            g.mismatches = gbad;
            g.pass = gbad == 0;
            c.rep.add(g);
        };
        Run r64, r32, r40;
        route(pairs, 64, r64, true);
        route(pairs, 32, r32, true);
        // same (token, slot) row through tile 32 vs 64
        {
            std::size_t bad = 0;
            for (int p = 0; p < pairs; ++p)
                bad += std::memcmp(&r64.yg[static_cast<std::size_t>(r64.inv[static_cast<std::size_t>(p)]) * F],
                                   &r32.yg[static_cast<std::size_t>(r32.inv[static_cast<std::size_t>(p)]) * F], static_cast<std::size_t>(F) * 4) != 0;
            Result r;
            r.name = std::string("gemm_moe32_") + wk::typeSuffix(mw.gate.type) + " == gemm_moe_ per (token, slot) row";
            r.kind = Kind::invariant;
            r.n = static_cast<std::size_t>(pairs);
            r.mismatches = bad;
            r.pass = bad == 0;
            c.rep.add(r);
        }
        // the first 40 tokens alone
        {
            const int np40 = 40 * K;
            route(np40, 32, r40, false);
            std::size_t bad = 0;
            for (int p = 0; p < np40; ++p)
                bad += std::memcmp(&r40.yg[static_cast<std::size_t>(r40.inv[static_cast<std::size_t>(p)]) * F],
                                   &r32.yg[static_cast<std::size_t>(r32.inv[static_cast<std::size_t>(p)]) * F], static_cast<std::size_t>(F) * 4) != 0;
            Result r;
            r.name = "gemm_moe32: 40-token batch rows == 200-token batch rows";
            r.kind = Kind::invariant;
            r.n = static_cast<std::size_t>(np40);
            r.mismatches = bad;
            r.pass = bad == 0;
            c.rep.add(r);
        }
        // down projection (another weight type) through tile 32 vs 64, rows matched by pair
        {
            const std::vector<float> ap = c.randn(static_cast<std::size_t>(pairs) * F);
            std::vector<std::uint16_t> a16(ap.size());
            for (std::size_t i = 0; i < ap.size(); ++i) a16[i] = f2h(ap[i]);
            std::vector<std::vector<float>> ys;
            for (const Run* rr : {&r64, &r32}) {
                const int bn = rr == &r64 ? 64 : 32;
                std::vector<std::uint16_t> xp(a16.size());
                for (int pos = 0; pos < pairs; ++pos)
                    std::memcpy(&xp[static_cast<std::size_t>(pos) * F], &a16[static_cast<std::size_t>(rr->perm[static_cast<std::size_t>(pos)]) * F],
                                static_cast<std::size_t>(F) * 2);
                Buf dxp(xp), tiles(rr->tiles), nt(std::vector<int>{rr->ntiles}), y(static_cast<std::size_t>(pairs) * E * 4);
                y.fill(0xff);
                auto gm = bn == 32 ? c.k.gemm_moe32 : c.k.gemm_moe;
                hip::launch(gm[static_cast<int>(mw.down.type)], {static_cast<unsigned>(rr->tiles.size()), cdiv(E, wk::kMoeBm), 1}, {256, 1, 1},
                            0, c.s, ddown.p(), mw.down.row_bytes, E, dxp.p(), y.p(), F, tiles.p(), nt.p());
                c.sync();
                const auto yy = y.down<float>(static_cast<std::size_t>(pairs) * E);
                std::vector<float> by_pair(yy.size());
                for (int p = 0; p < pairs; ++p)
                    std::memcpy(&by_pair[static_cast<std::size_t>(p) * E], &yy[static_cast<std::size_t>(rr->inv[static_cast<std::size_t>(p)]) * E],
                                static_cast<std::size_t>(E) * 4);
                ys.push_back(std::move(by_pair));
            }
            c.rep.add(cmpExact(std::string("gemm_moe32_") + wk::typeSuffix(mw.down.type) + " == gemm_moe_ per pair (down projection)",
                               ys[1], ys[0], Kind::invariant));
        }
        // grouped GEMM vs CPU (f16 weights as the fused dequant makes them, f16 x)
        {
            std::vector<float> got;
            std::vector<double> ref, sc;
            std::vector<std::uint16_t> wh(static_cast<std::size_t>(E));
            for (int p = 0; p < pairs; p += 97) {
                const int pos = r64.inv[static_cast<std::size_t>(p)], e = hids[static_cast<std::size_t>(p)];
                for (int rr = 0; rr < F; rr += 11) {
                    ref::dequantRowF16(mw.gate.type,
                                       mw.gate.data.data() + (static_cast<std::size_t>(e) * F + rr) * mw.gate.row_bytes, E, wh.data());
                    double y = 0, a = 0;
                    for (int k = 0; k < E; ++k) {
                        const double pr = static_cast<double>(h2f(wh[static_cast<std::size_t>(k)])) *
                                          h2f(r64.xg[static_cast<std::size_t>(pos) * E + k]);
                        y += pr;
                        a += std::fabs(pr);
                    }
                    got.push_back(r64.yg[static_cast<std::size_t>(pos) * F + rr]);
                    ref.push_back(y);
                    sc.push_back(a);
                }
            }
            c.rep.add(cmpTol(std::string("gemm_moe_") + wk::typeSuffix(mw.gate.type) + " vs CPU", got, ref, sc, 1e-4, 1e-6));
        }
        // act + combine
        {
            const std::size_t na = static_cast<std::size_t>(pairs) * F;
            const std::vector<float> g = c.randn(na, 2.f), u = c.randn(na);
            Buf dg(g), du(u), a(na * 2);
            hip::launch(c.k.moe_act_f16, {cdiv(na, 256), 1, 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), a.p(), static_cast<int>(na));
            c.sync();
            const auto ha = a.down<std::uint16_t>(na);
            std::vector<float> got(na);
            std::vector<double> ref(na);
            for (std::size_t i = 0; i < na; ++i) {
                got[i] = h2f(ha[i]);
                ref[i] = g[i] / (1.0 + std::exp(-static_cast<double>(g[i]))) * u[i];
            }
            c.rep.add(cmpTolRel("moe_act_f16 vs CPU", got, ref, 1e-3, 1e-5));  // f16 output: 2^-11 relative

            const std::vector<float> yd = c.randn(static_cast<std::size_t>(pairs) * E), x0 = c.randn(static_cast<std::size_t>(n) * E),
                                     ysh = c.randn(static_cast<std::size_t>(n) * E);
            Buf dyd(yd), dx(x0), dysh(ysh), dinv(r64.inv);
            hip::launch(c.k.moe_combine, {cdiv(E, 256), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dx.p(), dysh.p(),
                        sg.p(), dyd.p(), dinv.p(), w.p(), E, K);
            c.sync();
            const auto hw = w.down<float>(static_cast<std::size_t>(pairs));
            const auto hsg = sg.down<float>(static_cast<std::size_t>(n));
            std::vector<double> rc(x0.size()), sc(x0.size());
            for (int t = 0; t < n; ++t)
                for (int e = 0; e < E; ++e) {
                    double acc = 0, aa = 0;
                    for (int k = 0; k < K; ++k) {
                        const double v = static_cast<double>(hw[static_cast<std::size_t>(t) * K + k]) *
                                         yd[static_cast<std::size_t>(r64.inv[static_cast<std::size_t>(t) * K + k]) * E + e];
                        acc += v;
                        aa += std::fabs(v);
                    }
                    const double s = static_cast<double>(hsg[static_cast<std::size_t>(t)]) * ysh[static_cast<std::size_t>(t) * E + e];
                    rc[static_cast<std::size_t>(t) * E + e] = x0[static_cast<std::size_t>(t) * E + e] + acc + s;
                    sc[static_cast<std::size_t>(t) * E + e] = std::fabs(x0[static_cast<std::size_t>(t) * E + e]) + aa + std::fabs(s);
                }
            c.rep.add(cmpTol("moe_combine vs CPU", dx.down<float>(rc.size()), rc, sc, 1e-6, 1e-7));
        }
    }

    // ---------------- GDN beta / alpha projections (gdn_ab_<T>, gdn_abconv_<T>)
    {
        const std::vector<HostMat> mats = sampleMats(c, 96);  // 2 * nh rows needed
        const int nh = 48, n = 3, ch = 10240;
        for (const HostMat& m : mats) {
            const int ti = static_cast<int>(m.type);
            if (!c.k.gdn_ab[ti] || m.ncols % 256 != 0) continue;
            if (m.nrows < nh) continue;
            const bool two = m.nrows >= 2 * nh;  // beta rows [0, nh), alpha rows [nh, 2nh); else both read rows [0, nh)
            const std::string sfx = wk::typeSuffix(m.type);
            Buf W(m.data);
            const DevPtr Wb = W.p(), Wa = two ? W.p() + static_cast<std::uint64_t>(nh) * m.row_bytes : W.p();
            const std::vector<float> x = c.randn(static_cast<std::size_t>(n) * m.ncols);
            Buf dx(x), xq(static_cast<std::size_t>(n) * m.ncols), xd(static_cast<std::size_t>(n) * m.ncols / 32 * 4);
            hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(n) * m.ncols, 256), 1, 1}, {256, 1, 1}, 0, c.s,
                        dx.p(), xq.p(), xd.p(), n * m.ncols);
            const std::vector<float> dt = c.randn(nh), A = c.randu(nh, -2.f, -0.1f);
            Buf ddt(dt), dA(A), beta(static_cast<std::size_t>(n) * nh * 4), g(static_cast<std::size_t>(n) * nh * 4);
            hip::launch(c.k.gdn_ab[ti], {cdiv(2 * nh, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, Wb, Wa,
                        m.row_bytes, xq.p(), xd.p(), beta.p(), g.p(), ddt.p(), dA.p(), m.ncols, nh);
            c.sync();
            const auto hb = beta.down<float>(static_cast<std::size_t>(n) * nh), hg = g.down<float>(hb.size());
            {
                const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(n) * m.ncols);
                const auto hd = xd.down<float>(hq.size() / 32);
                std::vector<float> wf = ref::dequantRows(m, 0, nh);
                if (two) {
                    const std::vector<float> wa = ref::dequantRows(m, nh, nh);
                    wf.insert(wf.end(), wa.begin(), wa.end());
                } else {
                    const std::vector<float> wb = wf;
                    wf.insert(wf.end(), wb.begin(), wb.end());
                }
                std::vector<double> ry, rs;
                ref::gemvQ8F64(wf, 2 * nh, m.ncols, hq.data(), hd.data(), n, ry, rs);
                std::vector<double> rb, rg;
                for (int t = 0; t < n; ++t)
                    for (int hh = 0; hh < nh; ++hh) {
                        rb.push_back(1.0 / (1.0 + std::exp(-ry[static_cast<std::size_t>(t) * 2 * nh + hh])));
                        const double z = ry[static_cast<std::size_t>(t) * 2 * nh + nh + hh] + dt[static_cast<std::size_t>(hh)];
                        rg.push_back((z > 20 ? z : std::log1p(std::exp(z))) * A[static_cast<std::size_t>(hh)]);
                    }
                c.rep.add(cmpTolRel("gdn_ab_" + sfx + " beta vs CPU", hb, rb, 1e-4, 1e-6));
                c.rep.add(cmpTolRel("gdn_ab_" + sfx + " g vs CPU", hg, rg, 1e-4, 1e-5));
            }
            if (c.k.gdn_abconv[ti]) {
                // fused launch vs gdn_ab + gdn_conv_l2 on the same inputs (one segment of n rows)
                const std::vector<float> xin = c.randn(static_cast<std::size_t>(n) * ch), cw = c.randn(static_cast<std::size_t>(ch) * 4, 0.5f),
                                         st0 = c.randn(3 * static_cast<std::size_t>(ch));
                Buf dxin(xin), dcw(cw), sA(st0), sB(st0), oA(static_cast<std::size_t>(n) * ch * 4), oB(static_cast<std::size_t>(n) * ch * 4);
                Buf b2(hb.size() * 4), g2(hb.size() * 4);
                wk::GdnSegs segA, segB;
                segA.s[0].state = sA.p();
                segA.s[0].nrows = n;
                segB.s[0].state = sB.p();
                segB.s[0].nrows = n;
                const unsigned ab_blocks = cdiv(2 * nh, 8);
                hip::launch(c.k.gdn_abconv[ti], {ab_blocks * n + ch / 128, 1, 1}, {256, 1, 1}, 0, c.s, Wb, Wa, m.row_bytes,
                            xq.p(), xd.p(), b2.p(), g2.p(), ddt.p(), dA.p(), m.ncols, nh, n, dxin.p(), dcw.p(), oA.p(), ch, 32,
                            1e-6f, segA);
                hip::launch(c.k.gdn_conv_l2, {ch / 128, 1, 1}, {128, 1, 1}, 0, c.s, dxin.p(), dcw.p(), oB.p(), ch, 32, 1e-6f,
                            segB);
                c.sync();
                const bool same = b2.down<float>(hb.size()) == hb && g2.down<float>(hb.size()) == hg &&
                                  oA.down<float>(xin.size()) == oB.down<float>(xin.size()) &&
                                  sA.down<float>(st0.size()) == sB.down<float>(st0.size());
                Result r;
                r.name = "gdn_abconv_" + sfx + " == gdn_ab_" + sfx + " + gdn_conv_l2";
                r.kind = Kind::invariant;
                r.n = hb.size() * 2 + xin.size();
                r.mismatches = same ? 0 : 1;
                r.pass = same;
                c.rep.add(r);
            }
        }
    }
}

}  // namespace kt
