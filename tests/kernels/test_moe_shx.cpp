// whirl-kernel-test: moeshx (fast item, feat028-moefast97; kernels/moe_shx.hip, gfx1201) - the shared expert as
// slot K of the int8 MXFP4 decode / verify experts. Called by testMoeMx (family moemx):
//   * moe_gus_mxfp4w routed slots == moe_gu_mxfp4w (bitwise: same per-row code); shared slot vs CPU (tol);
//   * moe_downs_mxfp4w vs CPU (x0 + sum_k w_k down_k + sg * shared down; tol) and vs moe_down_mxfp4w + the
//     separate shared down on the same int8 activations (tol: only the f32 order differs);
//   * 1-token launches == a 5-token launch (bitwise; the MTP == plain property inside fast);
//   * moeskip ids (-1): dropped slots do no work and are left out of the sum (vs CPU).
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>

#include "cpu_ref.h"

namespace kt {

namespace {

std::vector<double> deq8s(const std::vector<std::int8_t>& q, const std::vector<float>& d, std::size_t off, int n) {
    std::vector<double> x(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) x[static_cast<std::size_t>(i)] = static_cast<double>(q[off + i]) * ref::xdScale(d[(off + i) / 32]);
    return x;
}
void dotRowS(const HostMat& m, std::size_t row, const std::vector<double>& x, double& y, double& a) {
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
Result invariantS(const std::string& name, std::size_t n, std::size_t bad) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    return r;
}

}  // namespace

void moeShxChecks(Ctx& c, const HostMat& gate, const HostMat& up, const HostMat& down, int R, int K) {
    if (!c.k.moe_gus_mxw || !c.k.moe_downs_mxw || !c.k.moe_gu_mxw || !c.k.moe_down_mxw) {
        c.rep.skip("moemx", "moe_gus_mxfp4w / moe_downs_mxfp4w (moeshx)", "kernel not in this code object");
        return;
    }
    const int E = gate.ncols, F = down.ncols;
    // the shared expert: blk.0 of the model, else synthetic MXFP4 of the experts' shapes
    HostMat sg, su, sd;
    const whirl::gguf::File* f = c.open(c.models.moemx);
    const auto* tg = f ? f->tensor("blk.0.ffn_gate_shexp.weight") : nullptr;
    if (tg && tg->type == whirl::gguf::GgmlType::mxfp4 && static_cast<int>(tg->ne[0]) == E && static_cast<int>(tg->ne[1]) == F) {
        sg = loadMat(*f, "blk.0.ffn_gate_shexp.weight");
        su = loadMat(*f, "blk.0.ffn_up_shexp.weight");
        sd = loadMat(*f, "blk.0.ffn_down_shexp.weight");
    } else {
        c.rep.skip("moemx", "moeshx shared expert weights", "MXFP4 shared expert of the experts' shape not available, synthetic used");
        sg = randomMat(c, QType::mxfp4, F, E);
        su = randomMat(c, QType::mxfp4, F, E);
        sd = randomMat(c, QType::mxfp4, E, F);
    }
    Buf dgate(gate.data), dup(up.data), ddown(down.data), dsg(sg.data), dsu(su.data), dsd(sd.data);
    for (const bool skips : {false, true}) {
        const int n = 5, S = K + 1, nb = F / 32;
        const std::string tag = skips ? " (moeskip ids)" : "";
        // routing: K distinct experts per token, renormalized weights; skips: the last two slots id -1 / weight 0
        std::vector<int> hids(static_cast<std::size_t>(n) * K);
        std::vector<float> hw(hids.size());
        std::uniform_int_distribution<int> ur(0, R - 1);
        std::uniform_real_distribution<float> uw(0.05f, 1.f);
        for (int t = 0; t < n; ++t) {
            float s = 0;
            for (int k = 0; k < K; ++k) {
                int e;
                bool dup2;
                do {
                    e = ur(c.rng);
                    dup2 = false;
                    for (int j = 0; j < k; ++j) dup2 = dup2 || hids[static_cast<std::size_t>(t) * K + j] == e;
                } while (dup2);
                hids[static_cast<std::size_t>(t) * K + k] = e;
                s += hw[static_cast<std::size_t>(t) * K + k] = uw(c.rng);
            }
            for (int k = 0; k < K; ++k) hw[static_cast<std::size_t>(t) * K + k] /= s;
            if (skips && K > 2)
                for (int k = K - 2; k < K; ++k) {
                    hids[static_cast<std::size_t>(t) * K + k] = -1;
                    hw[static_cast<std::size_t>(t) * K + k] = 0.f;
                }
        }
        const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E);
        const std::vector<float> hsg = c.randu(static_cast<std::size_t>(n), 0.1f, 0.9f);
        Buf dh(h), ids(hids), w(hw), dsgw(hsg);
        Buf xq(static_cast<std::size_t>(n) * E), xd(static_cast<std::size_t>(n) * E / 32 * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(n) * E, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), xq.p(), xd.p(), n * E);
        const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(n) * E);
        const auto hd = xd.down<float>(static_cast<std::size_t>(n) * E / 32);
        // gate + up over n x (K + 1) pairs
        const std::size_t ng = static_cast<std::size_t>(n) * S * F;
        Buf yg(ng * 4), yu(ng * 4);
        yg.zero();
        yu.zero();
        hip::launch(c.k.moe_gus_mxw, {cdiv(2 * F, 8), static_cast<unsigned>(n * S), 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), dsg.p(), dsu.p(),
                    gate.row_bytes, F, 2 * F, xq.p(), xd.p(), ids.p(), yg.p(), yu.p(), E, K, DevPtr{0});
        c.sync();
        const auto g1 = yg.down<float>(ng), u1 = yu.down<float>(ng);
        if (!skips) {
            // routed slots == moe_gu_mxfp4w (bitwise)
            const std::size_t nr = static_cast<std::size_t>(n) * K * F;
            Buf rg(nr * 4), ru(nr * 4);
            hip::launch(c.k.moe_gu_mxw, {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), gate.row_bytes, F,
                        2 * F, xq.p(), xd.p(), ids.p(), rg.p(), ru.p(), E, K, DevPtr{0});
            c.sync();
            const auto rgh = rg.down<float>(nr), ruh = ru.down<float>(nr);
            std::size_t bad = 0;
            for (int t = 0; t < n; ++t)
                for (int k = 0; k < K; ++k) {
                    const std::size_t a = (static_cast<std::size_t>(t) * S + k) * F, b = (static_cast<std::size_t>(t) * K + k) * F;
                    bad += std::memcmp(g1.data() + a, rgh.data() + b, static_cast<std::size_t>(F) * 4) != 0;
                    bad += std::memcmp(u1.data() + a, ruh.data() + b, static_cast<std::size_t>(F) * 4) != 0;
                }
            c.rep.add(invariantS("moe_gus_mxfp4w routed slots == moe_gu_mxfp4w", 2 * static_cast<std::size_t>(n) * K, bad));
        }
        {
            // shared slot (and every kept routed slot) vs CPU
            std::vector<float> got;
            std::vector<double> rf, sc;
            for (int t = 0; t < n; ++t) {
                const auto x = deq8s(hq, hd, static_cast<std::size_t>(t) * E, E);
                for (int s = 0; s < S; ++s) {
                    const int e = s < K ? hids[static_cast<std::size_t>(t) * K + s] : 0;
                    if (s < K && e < 0) continue;
                    for (int r = 0; r < F; r += (s == K ? 1 : 11)) {
                        double y, a;
                        const HostMat& gm = s < K ? gate : sg;
                        const HostMat& um = s < K ? up : su;
                        const std::size_t row = s < K ? static_cast<std::size_t>(e) * F + r : static_cast<std::size_t>(r);
                        const std::size_t at = (static_cast<std::size_t>(t) * S + s) * F + r;
                        dotRowS(gm, row, x, y, a);
                        got.push_back(g1[at]);
                        rf.push_back(y);
                        sc.push_back(a);
                        dotRowS(um, row, x, y, a);
                        got.push_back(u1[at]);
                        rf.push_back(y);
                        sc.push_back(a);
                    }
                }
            }
            c.rep.add(cmpTol("moe_gus_mxfp4w (shared slot + routed) vs CPU" + tag, got, rf, sc, 1e-4, 1e-6));
        }
        // SwiGLU -> int8 over n x (K + 1) x F (the launch the host makes)
        Buf dg(g1), du(u1), aq(ng), ad(ng / 32 * 4);
        hip::launch(c.k.silu_mul_q8, {cdiv(ng, 256), 1, 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), aq.p(), ad.p(), static_cast<int>(ng));
        c.sync();
        const auto aqh = aq.down<std::int8_t>(ng);
        const auto adh = ad.down<float>(ng / 32);
        const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E);
        Buf dx(x0);
        hip::launch(c.k.moe_downs_mxw, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, ddown.p(), dsd.p(), down.row_bytes, E, aq.p(),
                    ad.p(), ids.p(), w.p(), dsgw.p(), dx.p(), F, K, DevPtr{0});
        c.sync();
        const auto gx = dx.down<float>(x0.size());
        {
            std::vector<float> got;
            std::vector<double> rf, sc;
            for (int t = 0; t < n; ++t)
                for (int r = 0; r < E; r += 5) {
                    double acc = 0, a = 0;
                    for (int k = 0; k < S; ++k) {
                        const int e = k < K ? hids[static_cast<std::size_t>(t) * K + k] : 0;
                        if (k < K && e < 0) continue;
                        const auto x = deq8s(aqh, adh, (static_cast<std::size_t>(t) * S + k) * F, F);
                        const double wt = k < K ? hw[static_cast<std::size_t>(t) * K + k] : hsg[static_cast<std::size_t>(t)];
                        double y, aa;
                        dotRowS(k < K ? down : sd, k < K ? static_cast<std::size_t>(e) * E + r : static_cast<std::size_t>(r), x, y, aa);
                        acc += wt * y;
                        a += wt * aa;
                    }
                    got.push_back(gx[static_cast<std::size_t>(t) * E + r]);
                    rf.push_back(x0[static_cast<std::size_t>(t) * E + r] + acc);
                    sc.push_back(std::fabs(x0[static_cast<std::size_t>(t) * E + r]) + a);
                }
            c.rep.add(cmpTol("moe_downs_mxfp4w (+ shared slot, residual) vs CPU" + tag, got, rf, sc, 1e-4, 1e-6));
        }
        if (!skips) {
            // the balance form on the same int8 activations: moe_down_mxfp4w over the routed slots + sg * ysh, the
            // shared down from the CPU on the shared slot's int8 activations (moe_down's own input layout: n x K)
            std::vector<std::int8_t> rq(static_cast<std::size_t>(n) * K * F);
            std::vector<float> rd(rq.size() / 32);
            std::vector<float> ysh(static_cast<std::size_t>(n) * E);
            for (int t = 0; t < n; ++t) {
                for (int k = 0; k < K; ++k) {
                    std::memcpy(rq.data() + (static_cast<std::size_t>(t) * K + k) * F, aqh.data() + (static_cast<std::size_t>(t) * S + k) * F,
                                static_cast<std::size_t>(F));
                    std::memcpy(rd.data() + (static_cast<std::size_t>(t) * K + k) * nb, adh.data() + (static_cast<std::size_t>(t) * S + k) * nb,
                                static_cast<std::size_t>(nb) * 4);
                }
                const auto x = deq8s(aqh, adh, (static_cast<std::size_t>(t) * S + K) * F, F);
                for (int r = 0; r < E; ++r) {
                    double y, aa;
                    dotRowS(sd, static_cast<std::size_t>(r), x, y, aa);
                    ysh[static_cast<std::size_t>(t) * E + r] = static_cast<float>(y);
                }
            }
            Buf drq(rq), drd(rd), dysh(ysh), dx2(x0);
            hip::launch(c.k.moe_down_mxw, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, ddown.p(), down.row_bytes, E, drq.p(), drd.p(),
                        ids.p(), w.p(), dsgw.p(), dysh.p(), dx2.p(), F, K, DevPtr{0});
            c.sync();
            const auto bx = dx2.down<float>(x0.size());
            std::vector<double> rf(bx.begin(), bx.end()), sc(bx.size());
            for (std::size_t i = 0; i < bx.size(); ++i) sc[i] = std::fabs(bx[i]) + std::fabs(x0[i]) + 1.0;
            c.rep.add(cmpTol("moe_downs_mxfp4w == balance form (moe_down_mxfp4w + sg * ysh) within f32 order", gx, rf, sc, 1e-5, 1e-6));
            // 1-token launches == the 5-token launch (gate + up and down), bitwise
            std::size_t bad = 0;
            for (int t = 0; t < n; ++t) {
                std::vector<int> id1(hids.begin() + t * K, hids.begin() + (t + 1) * K);
                std::vector<float> w1(hw.begin() + t * K, hw.begin() + (t + 1) * K);
                std::vector<float> xt(x0.begin() + static_cast<std::ptrdiff_t>(t) * E, x0.begin() + static_cast<std::ptrdiff_t>(t + 1) * E);
                Buf ids1(id1), dw1(w1), dx1(xt), y1g(static_cast<std::size_t>(S) * F * 4), y1u(static_cast<std::size_t>(S) * F * 4);
                hip::launch(c.k.moe_gus_mxw, {cdiv(2 * F, 8), static_cast<unsigned>(S), 1}, {256, 1, 1}, 0, c.s, dgate.p(), dup.p(), dsg.p(), dsu.p(),
                            gate.row_bytes, F, 2 * F, xq.p() + static_cast<std::uint64_t>(t) * E, xd.p() + static_cast<std::uint64_t>(t) * E / 32 * 4,
                            ids1.p(), y1g.p(), y1u.p(), E, K, DevPtr{0});
                hip::launch(c.k.moe_downs_mxw, {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, ddown.p(), dsd.p(), down.row_bytes, E,
                            aq.p() + static_cast<std::uint64_t>(t) * S * F, ad.p() + static_cast<std::uint64_t>(t) * S * F / 32 * 4, ids1.p(), dw1.p(),
                            dsgw.p() + static_cast<std::uint64_t>(t) * 4, dx1.p(), F, K, DevPtr{0});
                c.sync();
                const auto a = y1g.down<float>(static_cast<std::size_t>(S) * F), b = y1u.down<float>(a.size());
                const auto xx = dx1.down<float>(static_cast<std::size_t>(E));
                bad += std::memcmp(a.data(), g1.data() + static_cast<std::size_t>(t) * S * F, a.size() * 4) != 0;
                bad += std::memcmp(b.data(), u1.data() + static_cast<std::size_t>(t) * S * F, b.size() * 4) != 0;
                bad += std::memcmp(xx.data(), gx.data() + static_cast<std::size_t>(t) * E, xx.size() * 4) != 0;
            }
            c.rep.add(invariantS("moe_gus / moe_downs_mxfp4w: 1-token launches == 5-token launch", 3 * static_cast<std::size_t>(n), bad));
        }
    }
}

}  // namespace kt
