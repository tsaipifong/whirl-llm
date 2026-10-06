// whirl-kernel-test: MXFP4 routed experts (Ornith-1.5-35B-A3B MXFP4, kernels/moe_mxfp4.hip).
//   * decode: moe_gu_mxfp4 / moe_gu_mxfp4w and moe_down_mxfp4 / moe_down_mxfp4w (int8
//     activations) vs CPU; the whole-block kernels: a 5-token launch == five 1-token
//     launches (bitwise, the MTP == plain property);
//   * prefill: moe_gather_fp8 == CPU qact_fp8 of the routed token (bytes + scale exact);
//     gemm8_moe (f32 out) vs CPU (exact MXFP4 weights x the GPU's fp8 activations);
//     gemm8_moe32 == gemm8_moe per (token, slot) row and a 40-token batch == the same rows
//     of the 200-token batch (bitwise); gemm8_moeh / gemm8_moe32h fused gate + up within
//     one f16 rounding of the f32 results; down projection (silu_mul_x8h activations) vs
//     CPU and tile 32 == 64; the generic f16 gemm_moe_mxfp4 vs CPU and tile 32 == 64.
//   Code objects without the whole-block / fp8 kernels (gfx1151) run the generic parts:
//   the 1-token == 5-token checks then cover moe_gu_mxfp4 / moe_down_mxfp4.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>

#include "cpu_ref.h"

namespace kt {

namespace {

struct MxMoe {
    HostMat gate, up, down;  // [R * F][E], [R * F][E], [R * E][F], kernel layout + row refs
    int E = 2048, R = 256, K = 8, F = 512;
};

MxMoe loadMxMoe(Ctx& c) {
    MxMoe m;
    const whirl::gguf::File* f = c.open(c.models.moemx);
    const auto* tg = f ? f->tensor("blk.0.ffn_gate_exps.weight") : nullptr;
    if (tg && tg->type == whirl::gguf::GgmlType::mxfp4) {
        m.gate = loadMat(*f, "blk.0.ffn_gate_exps.weight");
        m.up = loadMat(*f, "blk.0.ffn_up_exps.weight");
        m.down = loadMat(*f, "blk.0.ffn_down_exps.weight");
        m.E = m.gate.ncols;
        m.F = m.down.ncols;
        m.R = m.gate.nrows / m.F;
        m.K = static_cast<int>(f->getUintOr("qwen35moe.expert_used_count", 8));
    } else {
        c.rep.skip("moemx", "weights", "MXFP4 MoE model not available, synthetic MXFP4 experts used");
        m.R = 64;
        m.gate = randomMat(c, QType::mxfp4, m.R * m.F, m.E);
        m.up = randomMat(c, QType::mxfp4, m.R * m.F, m.E);
        m.down = randomMat(c, QType::mxfp4, m.R * m.E, m.F);
    }
    return m;
}

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

// K distinct experts per token (mildly skewed to low ids), renormalized positive weights
void randomRouting(Ctx& c, int n, int R, int K, std::vector<int>& ids, std::vector<float>& w) {
    ids.assign(static_cast<std::size_t>(n) * K, 0);
    w.assign(ids.size(), 0.f);
    std::uniform_int_distribution<int> ur(0, R - 1), q4(0, 3);
    std::uniform_real_distribution<float> uw(0.05f, 1.f);
    for (int t = 0; t < n; ++t) {
        float s = 0;
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                const int a = ur(c.rng), b = ur(c.rng);
                e = q4(c.rng) == 0 ? std::min(a, b) : a;
                dup = false;
                for (int j = 0; j < k; ++j) dup = dup || ids[static_cast<std::size_t>(t) * K + j] == e;
            } while (dup);
            ids[static_cast<std::size_t>(t) * K + k] = e;
            const float v = uw(c.rng);
            w[static_cast<std::size_t>(t) * K + k] = v;
            s += v;
        }
        for (int k = 0; k < K; ++k) w[static_cast<std::size_t>(t) * K + k] /= s;
    }
}

Result invariant(const std::string& name, std::size_t n, std::size_t bad) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    return r;
}

}  // namespace

void testMoeMx(Ctx& c) {
    c.rep.family = "moemx";
    const int ti = static_cast<int>(QType::mxfp4);
    if (!c.k.moe_gu[ti] || !c.k.moe_down[ti]) {
        c.rep.skip("moemx", "kernels", "MXFP4 MoE kernels not in this code object");
        return;
    }
    // whole-block decode experts and the fp8 grouped GEMM (gfx1201 only)
    const bool mxw = c.k.moe_gu_mxw && c.k.moe_down_mxw;
    const bool fp8 = c.k.gemm8_moe && c.k.gemm8_moe32 && c.k.gemm8_moeh && c.k.gemm8_moe32h && c.k.moe_gather_fp8;
    if (!mxw) c.rep.skip("moemx", "moe_gu_mxfp4w / moe_down_mxfp4w", "kernel not in this code object");
    if (!fp8) c.rep.skip("moemx", "gemm8_moe* / moe_gather_fp8 (fp8 prefill experts)", "kernel not in this code object");
    MxMoe mw = loadMxMoe(c);
    const int E = mw.E, R = mw.R, K = mw.K, F = mw.F;
    Buf dgate(mw.gate.data), dup(mw.up.data), ddown(mw.down.data);
    Buf rgate(mw.gate.ref), rup(mw.up.ref), rdown(mw.down.ref);

    // ---------------- decode (n <= 16): generic and whole-block kernels
    {
        const int n = 5;
        std::vector<int> hids;
        std::vector<float> hw;
        randomRouting(c, n, R, K, hids, hw);
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        const std::vector<float> hsg = c.randu(static_cast<std::size_t>(n), 0.1f, 0.9f);
        Buf dh(h), ids(hids), w(hw), sg(hsg);
        Buf xq(static_cast<std::size_t>(n) * E), xd(static_cast<std::size_t>(n) * E / 32 * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(n) * E, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), xq.p(), xd.p(), n * E);
        const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(n) * E);
        const auto hd = xd.down<float>(static_cast<std::size_t>(n) * E / 32);
        const std::size_t ng = static_cast<std::size_t>(n) * K * F;
        std::vector<float> g_w, u_w;  // decode-kernel results (feed the down test)
        for (const bool whole : {false, true}) {
            if (whole && !mxw) continue;
            const hip::Function f = whole ? c.k.moe_gu_mxw : c.k.moe_gu[ti];
            const std::string nm = whole ? "moe_gu_mxfp4w" : "moe_gu_mxfp4";
            Buf yg(ng * 4), yu(ng * 4);
            hip::launch(f, {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), mw.gate.row_bytes, F, 2 * F,
                        xq.p(), xd.p(), ids.p(), yg.p(), yu.p(), E, K, DevPtr{0});
            c.sync();
            const auto g1 = yg.down<float>(ng), u1 = yu.down<float>(ng);
            std::vector<float> got;
            std::vector<double> ref, sc;
            for (int p = 0; p < n * K; p += 3) {
                const int t = p / K, e = hids[static_cast<std::size_t>(p)];
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
            c.rep.add(cmpTol(nm + " vs CPU", got, ref, sc, 1e-4, 1e-6));
            if (whole == mxw) {  // the kernel decode uses: whole-block if present, else generic
                g_w = g1;
                u_w = u1;
                // one token per launch (the plain decode step) == the 5-token launch (verify)
                std::size_t bad = 0;
                for (int t = 0; t < n; ++t) {
                    std::vector<int> id1(hids.begin() + t * K, hids.begin() + (t + 1) * K);
                    Buf ids1(id1), y1g(static_cast<std::size_t>(K) * F * 4), y1u(static_cast<std::size_t>(K) * F * 4);
                    hip::launch(f, {cdiv(2 * F, 8), static_cast<unsigned>(K), 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), mw.gate.row_bytes, F,
                                2 * F, xq.p() + static_cast<std::uint64_t>(t) * E, xd.p() + static_cast<std::uint64_t>(t) * E / 32 * 4, ids1.p(),
                                y1g.p(), y1u.p(), E, K, DevPtr{0});
                    c.sync();
                    const auto a = y1g.down<float>(static_cast<std::size_t>(K) * F), b = y1u.down<float>(a.size());
                    bad += std::memcmp(a.data(), g1.data() + static_cast<std::size_t>(t) * K * F, a.size() * 4) != 0;
                    bad += std::memcmp(b.data(), u1.data() + static_cast<std::size_t>(t) * K * F, b.size() * 4) != 0;
                }
                c.rep.add(invariant(nm + ": 1-token launches == 5-token launch", 2 * static_cast<std::size_t>(n), bad));
            }
        }
        // down: a = silu(g) * u as int8, down projection + weights + shared expert + residual
        Buf dg(g_w), du(u_w);
        Buf aq(ng), ad(ng / 32 * 4);
        hip::launch(c.k.silu_mul_q8, {cdiv(ng, 256), 1, 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), aq.p(), ad.p(), static_cast<int>(ng));
        const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
        Buf dysh(ysh);
        c.sync();
        const auto aqh = aq.down<std::int8_t>(ng);
        const auto adh = ad.down<float>(ng / 32);
        std::vector<float> xw;
        for (const bool whole : {false, true}) {
            if (whole && !mxw) continue;
            const hip::Function f = whole ? c.k.moe_down_mxw : c.k.moe_down[ti];
            const std::string nm = whole ? "moe_down_mxfp4w" : "moe_down_mxfp4";
            Buf dx(x0);
            hip::launch(f, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, ddown.p(), mw.down.row_bytes, E, aq.p(), ad.p(), ids.p(),
                        w.p(), sg.p(), dysh.p(), dx.p(), F, K, DevPtr{0});
            c.sync();
            const auto gx = dx.down<float>(x0.size());
            std::vector<float> got;
            std::vector<double> ref, sc;
            for (int t = 0; t < n; ++t)
                for (int r = 0; r < E; r += 5) {
                    double acc = 0, a = 0;
                    for (int k = 0; k < K; ++k) {
                        const int p = t * K + k;
                        const auto x = deq8(aqh, adh, static_cast<std::size_t>(p) * F, F);
                        double y, aa;
                        dotRow(mw.down, static_cast<std::size_t>(hids[static_cast<std::size_t>(p)]) * E + r, x, y, aa);
                        acc += hw[static_cast<std::size_t>(p)] * y;
                        a += hw[static_cast<std::size_t>(p)] * aa;
                    }
                    const double shv = static_cast<double>(hsg[static_cast<std::size_t>(t)]) * ysh[static_cast<std::size_t>(t) * E + r];
                    got.push_back(gx[static_cast<std::size_t>(t) * E + r]);
                    ref.push_back(x0[static_cast<std::size_t>(t) * E + r] + acc + shv);
                    sc.push_back(std::fabs(x0[static_cast<std::size_t>(t) * E + r]) + a + std::fabs(shv));
                }
            c.rep.add(cmpTol(nm + " (+ shared, residual) vs CPU", got, ref, sc, 1e-4, 1e-6));
            if (whole == mxw) xw = gx;
        }
        {
            std::size_t bad = 0;
            for (int t = 0; t < n; ++t) {
                std::vector<int> id1(hids.begin() + t * K, hids.begin() + (t + 1) * K);
                std::vector<float> w1(hw.begin() + t * K, hw.begin() + (t + 1) * K);
                std::vector<float> xt(x0.begin() + static_cast<std::ptrdiff_t>(t) * E, x0.begin() + static_cast<std::ptrdiff_t>(t + 1) * E);
                Buf ids1(id1), dw1(w1), dx1(xt);
                hip::launch(mxw ? c.k.moe_down_mxw : c.k.moe_down[ti], {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), mw.down.row_bytes, E,
                            aq.p() + static_cast<std::uint64_t>(t) * K * F, ad.p() + static_cast<std::uint64_t>(t) * K * F / 32 * 4, ids1.p(), dw1.p(),
                            sg.p() + static_cast<std::uint64_t>(t) * 4, dysh.p() + static_cast<std::uint64_t>(t) * E * 4, dx1.p(), F, K, DevPtr{0});
                c.sync();
                const auto a = dx1.down<float>(static_cast<std::size_t>(E));
                bad += std::memcmp(a.data(), xw.data() + static_cast<std::size_t>(t) * E, a.size() * 4) != 0;
            }
            c.rep.add(invariant(std::string(mxw ? "moe_down_mxfp4w" : "moe_down_mxfp4") + ": 1-token launches == 5-token launch",
                                static_cast<std::size_t>(n), bad));
        }
    }

    // ---------------- prefill: fp8 grouped expert GEMM
    {
        const int n = 200, pairs = n * K;
        std::vector<int> hids;
        std::vector<float> hw;
        randomRouting(c, n, R, K, hids, hw);
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        Buf dh(h), ids(hids);
        struct Run {
            std::vector<int> perm, inv;
            std::vector<wk::Int4> tiles;
            int ntiles = 0;
            std::vector<std::uint8_t> x8;
            std::vector<float> sx, yg, yu;            // f32 gate / up (separate launches)
            std::vector<std::uint16_t> hg, hu;        // fused gate + up, f16 out
        };
        auto run = [&](int np, int bn, Run& out) {
            Buf perm(static_cast<std::size_t>(np) * 4), inv(static_cast<std::size_t>(np) * 4);
            const int max_tiles = (np + bn - 1) / bn + R;
            Buf tiles(static_cast<std::size_t>(max_tiles) * 16), nt(4);
            hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, ids.p(), np, R, bn, perm.p(), inv.p(), tiles.p(), nt.p());
            if (!fp8) {  // routing only (the generic f16 expert GEMM below)
                c.sync();
                out.perm = perm.down<int>(static_cast<std::size_t>(np));
                out.inv = inv.down<int>(static_cast<std::size_t>(np));
                out.ntiles = nt.down<int>(1)[0];
                out.tiles = tiles.down<wk::Int4>(static_cast<std::size_t>(max_tiles));
                return;
            }
            Buf x8(static_cast<std::size_t>(np) * E), sx(static_cast<std::size_t>(np) * 4);
            hip::launch(c.k.moe_gather_fp8, {static_cast<unsigned>(np), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), x8.p(), sx.p(), E, K);
            Buf yg(static_cast<std::size_t>(np) * F * 4), yu(static_cast<std::size_t>(np) * F * 4);
            Buf hg(static_cast<std::size_t>(np) * F * 2), hu(static_cast<std::size_t>(np) * F * 2);
            for (Buf* b : {&yg, &yu, &hg, &hu}) b->fill(0xff);
            const hip::Function f32 = bn == 32 ? c.k.gemm8_moe32 : c.k.gemm8_moe;
            const hip::Function f16 = bn == 32 ? c.k.gemm8_moe32h : c.k.gemm8_moeh;
            const unsigned nby = cdiv(F, wk::kMoeBm);
            hip::launch(f32, {static_cast<unsigned>(max_tiles), nby, 1}, {256, 1, 1}, 0, c.s, dgate.p(), DevPtr{0}, mw.gate.row_bytes, rgate.p(),
                        DevPtr{0}, F, x8.p(), sx.p(), yg.p(), DevPtr{0}, E, tiles.p(), nt.p());
            hip::launch(f32, {static_cast<unsigned>(max_tiles), nby, 1}, {256, 1, 1}, 0, c.s, dup.p(), DevPtr{0}, mw.up.row_bytes, rup.p(),
                        DevPtr{0}, F, x8.p(), sx.p(), yu.p(), DevPtr{0}, E, tiles.p(), nt.p());
            hip::launch(f16, {static_cast<unsigned>(max_tiles), 2 * nby, 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), mw.gate.row_bytes, rgate.p(),
                        rup.p(), F, x8.p(), sx.p(), hg.p(), hu.p(), E, tiles.p(), nt.p());
            c.sync();
            out.perm = perm.down<int>(static_cast<std::size_t>(np));
            out.inv = inv.down<int>(static_cast<std::size_t>(np));
            out.ntiles = nt.down<int>(1)[0];
            out.tiles = tiles.down<wk::Int4>(static_cast<std::size_t>(max_tiles));
            out.x8 = x8.down<std::uint8_t>(static_cast<std::size_t>(np) * E);
            out.sx = sx.down<float>(static_cast<std::size_t>(np));
            out.yg = yg.down<float>(static_cast<std::size_t>(np) * F);
            out.yu = yu.down<float>(static_cast<std::size_t>(np) * F);
            out.hg = hg.down<std::uint16_t>(static_cast<std::size_t>(np) * F);
            out.hu = hu.down<std::uint16_t>(static_cast<std::size_t>(np) * F);
        };
        Run r64, r32, r40;
        run(pairs, 64, r64);
        run(pairs, 32, r32);
        // gather: CPU qact_fp8 of the routed token row
        if (fp8) {
            std::size_t bad = 0;
            std::vector<std::uint8_t> q(static_cast<std::size_t>(E));
            for (int pos = 0; pos < pairs; ++pos) {
                const int t = r64.perm[static_cast<std::size_t>(pos)] / K;
                float sx = 0;
                ref::qactFp8Row(h.data() + static_cast<std::size_t>(t) * E, E, q.data(), sx);
                bad += std::memcmp(q.data(), r64.x8.data() + static_cast<std::size_t>(pos) * E, static_cast<std::size_t>(E)) != 0;
                bad += sx != r64.sx[static_cast<std::size_t>(pos)];
            }
            Result r;
            r.name = "moe_gather_fp8 == CPU qact_fp8 of the routed token (bytes + scale)";
            r.n = static_cast<std::size_t>(pairs);
            r.mismatches = bad;
            r.pass = bad == 0;
            c.rep.add(r);
        }
        // gate (f32 out) vs CPU: exact MXFP4 weights x the GPU's fp8 activations
        if (fp8) {
            std::vector<float> got;
            std::vector<double> ref, sc;
            for (int p = 0; p < pairs; p += 37) {
                const int pos = r64.inv[static_cast<std::size_t>(p)], e = hids[static_cast<std::size_t>(p)];
                std::vector<double> x(static_cast<std::size_t>(E));
                for (int k = 0; k < E; ++k)
                    x[static_cast<std::size_t>(k)] = static_cast<double>(ref::e4m3ToF(r64.x8[static_cast<std::size_t>(pos) * E + k])) * r64.sx[static_cast<std::size_t>(pos)];
                for (int rr = 0; rr < F; rr += 5) {
                    double y, a;
                    dotRow(mw.gate, static_cast<std::size_t>(e) * F + rr, x, y, a);
                    got.push_back(r64.yg[static_cast<std::size_t>(pos) * F + rr]);
                    ref.push_back(y);
                    sc.push_back(a);
                    dotRow(mw.up, static_cast<std::size_t>(e) * F + rr, x, y, a);
                    got.push_back(r64.yu[static_cast<std::size_t>(pos) * F + rr]);
                    ref.push_back(y);
                    sc.push_back(a);
                }
            }
            c.rep.add(cmpTol("gemm8_moe gate / up (f32 out) vs CPU (exact MXFP4 weights, GPU fp8 activations)", got, ref, sc, 1e-4, 1e-6));
        }
        // tile 32 == tile 64 per (token, slot) row, f32 and fused f16 outputs
        if (fp8) {
            std::size_t bad = 0, badh = 0;
            for (int p = 0; p < pairs; ++p) {
                const std::size_t a = static_cast<std::size_t>(r64.inv[static_cast<std::size_t>(p)]) * F, b = static_cast<std::size_t>(r32.inv[static_cast<std::size_t>(p)]) * F;
                bad += std::memcmp(&r64.yg[a], &r32.yg[b], static_cast<std::size_t>(F) * 4) != 0;
                bad += std::memcmp(&r64.yu[a], &r32.yu[b], static_cast<std::size_t>(F) * 4) != 0;
                badh += std::memcmp(&r64.hg[a], &r32.hg[b], static_cast<std::size_t>(F) * 2) != 0;
                badh += std::memcmp(&r64.hu[a], &r32.hu[b], static_cast<std::size_t>(F) * 2) != 0;
            }
            c.rep.add(invariant("gemm8_moe32 == gemm8_moe per (token, slot) row (gate, up)", 2 * static_cast<std::size_t>(pairs), bad));
            c.rep.add(invariant("gemm8_moe32h == gemm8_moeh per (token, slot) row (fused gate + up)", 2 * static_cast<std::size_t>(pairs), badh));
        }
        // the first 40 tokens alone (tile 32) == the same rows of the 200-token batch
        if (fp8) {
            const int np40 = 40 * K;
            run(np40, 32, r40);
            std::size_t bad = 0;
            for (int p = 0; p < np40; ++p) {
                const std::size_t a = static_cast<std::size_t>(r40.inv[static_cast<std::size_t>(p)]) * F, b = static_cast<std::size_t>(r32.inv[static_cast<std::size_t>(p)]) * F;
                bad += std::memcmp(&r40.yg[a], &r32.yg[b], static_cast<std::size_t>(F) * 4) != 0;
                bad += std::memcmp(&r40.hu[a], &r32.hu[b], static_cast<std::size_t>(F) * 2) != 0;
            }
            c.rep.add(invariant("gemm8_moe32 / gemm8_moe32h: 40-token batch rows == 200-token batch rows", 2 * static_cast<std::size_t>(np40), bad));
        }
        // fused f16 outputs within one f16 rounding of the f32 results
        if (fp8) {
            std::vector<float> got;
            std::vector<double> ref;
            for (std::size_t i = 0; i < r64.yg.size(); i += 3) {
                got.push_back(h2f(r64.hg[i]));
                ref.push_back(r64.yg[i]);
                got.push_back(h2f(r64.hu[i]));
                ref.push_back(r64.yu[i]);
            }
            c.rep.add(cmpTolRel("gemm8_moeh fused gate + up (f16 out) vs gemm8_moe f32", got, ref, 1.0 / 2048, 1e-7));
        }
        // down projection: silu_mul_x8h(fused f16 g, u) -> fp8 rows; tile 64 vs CPU, tile 32 == 64
        if (fp8 && c.k.silu_mul_x8h) {
            std::vector<std::vector<float>> by_pair(2);
            std::vector<std::uint8_t> a8;
            std::vector<float> asx;
            for (int vi = 0; vi < 2; ++vi) {
                const Run& rr = vi == 0 ? r64 : r32;
                const int bn = vi == 0 ? 64 : 32;
                Buf hg(rr.hg), hu(rr.hu), x8(static_cast<std::size_t>(pairs) * F), sx(static_cast<std::size_t>(pairs) * 4);
                hip::launch(c.k.silu_mul_x8h, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, hg.p(), hu.p(), x8.p(), sx.p(), F);
                Buf tiles(rr.tiles), nt(std::vector<int>{rr.ntiles}), y(static_cast<std::size_t>(pairs) * E * 4);
                y.fill(0xff);
                hip::launch(bn == 32 ? c.k.gemm8_moe32 : c.k.gemm8_moe, {static_cast<unsigned>(rr.tiles.size()), cdiv(E, wk::kMoeBm), 1}, {256, 1, 1}, 0,
                            c.s, ddown.p(), DevPtr{0}, mw.down.row_bytes, rdown.p(), DevPtr{0}, E, x8.p(), sx.p(), y.p(), DevPtr{0}, F, tiles.p(), nt.p());
                c.sync();
                const auto yy = y.down<float>(static_cast<std::size_t>(pairs) * E);
                if (vi == 0) {
                    a8 = x8.down<std::uint8_t>(static_cast<std::size_t>(pairs) * F);
                    asx = sx.down<float>(static_cast<std::size_t>(pairs));
                    std::vector<float> got;
                    std::vector<double> ref, sc;
                    for (int p = 0; p < pairs; p += 41) {
                        const int pos = rr.inv[static_cast<std::size_t>(p)], e = hids[static_cast<std::size_t>(p)];
                        std::vector<double> x(static_cast<std::size_t>(F));
                        for (int k = 0; k < F; ++k)
                            x[static_cast<std::size_t>(k)] = static_cast<double>(ref::e4m3ToF(a8[static_cast<std::size_t>(pos) * F + k])) * asx[static_cast<std::size_t>(pos)];
                        for (int r = 0; r < E; r += 9) {
                            double yv, a;
                            dotRow(mw.down, static_cast<std::size_t>(e) * E + r, x, yv, a);
                            got.push_back(yy[static_cast<std::size_t>(pos) * E + r]);
                            ref.push_back(yv);
                            sc.push_back(a);
                        }
                    }
                    c.rep.add(cmpTol("gemm8_moe down (f32 out, vector epilogue) vs CPU", got, ref, sc, 1e-4, 1e-6));
                }
                by_pair[static_cast<std::size_t>(vi)].resize(yy.size());
                for (int p = 0; p < pairs; ++p)
                    std::memcpy(&by_pair[static_cast<std::size_t>(vi)][static_cast<std::size_t>(p) * E],
                                &yy[static_cast<std::size_t>(rr.inv[static_cast<std::size_t>(p)]) * E], static_cast<std::size_t>(E) * 4);
            }
            c.rep.add(cmpExact("gemm8_moe32 == gemm8_moe per pair (down projection)", by_pair[1], by_pair[0], Kind::invariant));
        }
        // the generic f16 path for MXFP4 experts (gemm_moe32_mxfp4 == gemm_moe_mxfp4 per row)
        if (c.k.gemm_moe[ti] && c.k.gemm_moe32[ti]) {
            std::vector<std::vector<float>> by_pair(2);
            for (int vi = 0; vi < 2; ++vi) {
                const Run& rr = vi == 0 ? r64 : r32;
                const int bn = vi == 0 ? 64 : 32;
                Buf perm(rr.perm), xg(static_cast<std::size_t>(pairs) * E * 2), tiles(rr.tiles), nt(std::vector<int>{rr.ntiles});
                hip::launch(c.k.moe_gather_f16, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), xg.p(), E, K);
                Buf y(static_cast<std::size_t>(pairs) * F * 4);
                y.fill(0xff);
                hip::launch((bn == 32 ? c.k.gemm_moe32 : c.k.gemm_moe)[ti], {static_cast<unsigned>(rr.tiles.size()), cdiv(F, wk::kMoeBm), 1}, {256, 1, 1}, 0,
                            c.s, dgate.p(), mw.gate.row_bytes, F, xg.p(), y.p(), E, tiles.p(), nt.p());
                c.sync();
                const auto yy = y.down<float>(static_cast<std::size_t>(pairs) * F);
                if (vi == 0) {
                    const auto xh = xg.down<std::uint16_t>(static_cast<std::size_t>(pairs) * E);
                    std::vector<float> got;
                    std::vector<double> ref, sc;
                    std::vector<std::uint16_t> wh(static_cast<std::size_t>(E));
                    for (int p = 0; p < pairs; p += 97) {
                        const int pos = rr.inv[static_cast<std::size_t>(p)], e = hids[static_cast<std::size_t>(p)];
                        for (int r = 0; r < F; r += 11) {
                            ref::dequantRowF16(QType::mxfp4, mw.gate.data.data() + (static_cast<std::size_t>(e) * F + r) * mw.gate.row_bytes, E, wh.data());
                            double yv = 0, a = 0;
                            for (int k = 0; k < E; ++k) {
                                const double pr = static_cast<double>(h2f(wh[static_cast<std::size_t>(k)])) * h2f(xh[static_cast<std::size_t>(pos) * E + k]);
                                yv += pr;
                                a += std::fabs(pr);
                            }
                            got.push_back(yy[static_cast<std::size_t>(pos) * F + r]);
                            ref.push_back(yv);
                            sc.push_back(a);
                        }
                    }
                    c.rep.add(cmpTol("gemm_moe_mxfp4 vs CPU", got, ref, sc, 1e-4, 1e-6));
                }
                by_pair[static_cast<std::size_t>(vi)].resize(yy.size());
                for (int p = 0; p < pairs; ++p)
                    std::memcpy(&by_pair[static_cast<std::size_t>(vi)][static_cast<std::size_t>(p) * F],
                                &yy[static_cast<std::size_t>(rr.inv[static_cast<std::size_t>(p)]) * F], static_cast<std::size_t>(F) * 4);
            }
            c.rep.add(cmpExact("gemm_moe32_mxfp4 == gemm_moe_mxfp4 per (token, slot) row", by_pair[1], by_pair[0], Kind::invariant));
        }
        moeGuChecks(c, mw.gate, mw.up, mw.down, R, K);
    }
}

}  // namespace kt
