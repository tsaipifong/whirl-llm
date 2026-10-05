// whirl-kernel-test: Gated DeltaNet.
//   * conv (gdn_conv_seq, gdn_conv_par + gdn_conv_state), l2norm, gates,
//     gated norm vs CPU; the rolling conv state and its snapshot exact;
//   * the recurrence (gdn_seq_128, gdn_seq_128_l8, chunked gdn_chunk_prep +
//     gdn_chunk_scan, f16-WMMA gdn_wprep + gdn_wscan8) vs a double-precision
//     CPU delta rule (output, final state, snapshot);
//   * fused decode kernels (gdn_conv_l2, gdn_step_norm): a multi-row segment
//     == the rows run one launch at a time, snapshots == the per-row states,
//     replay of kept rows == running them, two segments in one launch ==
//     separate launches, gdn_step_norm == gdn_step_norm_v0 (all bitwise), and
//     vs CPU (tolerance).
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "cpu_ref.h"

namespace kt {

namespace {

constexpr int kNk = 16, kNv = 48, kDk = 128, kDv = 128;
constexpr int kCh = 2 * kNk * kDk + kNv * kDv;  // conv channels = qkv row (q | k | v)
constexpr int kDInner = kNv * kDv;
const float kEps = 1e-6f;
const float kScale = 1.0f / std::sqrt(128.0f);

double rmsOf(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s / std::max<std::size_t>(1, v.size())) + 1e-30;
}

// qkv rows with unit-norm q / k heads (what the conv + l2norm produce), v ~ N(0, 1).
std::vector<float> makeQkv(Ctx& c, int n) {
    std::vector<float> x = c.randn(static_cast<std::size_t>(n) * kCh);
    for (int t = 0; t < n; ++t)
        for (int hh = 0; hh < 2 * kNk; ++hh) {
            float* p = x.data() + static_cast<std::size_t>(t) * kCh + static_cast<std::size_t>(hh) * kDk;
            double s = 0;
            for (int i = 0; i < kDk; ++i) s += static_cast<double>(p[i]) * p[i];
            const float r = static_cast<float>(1.0 / std::sqrt(s));
            for (int i = 0; i < kDk; ++i) p[i] *= r;
        }
    return x;
}

// Delta rule in double. state: [nv][dk][dv]; out: [n][nv][dv]; snaps[t] = state after row t.
void deltaRef(const std::vector<float>& qkv, const std::vector<float>& g, const std::vector<float>& beta, int n,
              std::vector<double>& state, std::vector<double>& out, std::vector<std::vector<double>>* snaps) {
    out.assign(static_cast<std::size_t>(n) * kNv * kDv, 0.0);
    if (snaps) snaps->assign(static_cast<std::size_t>(n), {});
    for (int t = 0; t < n; ++t) {
        const float* row = qkv.data() + static_cast<std::size_t>(t) * kCh;
        for (int h = 0; h < kNv; ++h) {
            const int kh = h % kNk;
            const float* q = row + static_cast<std::size_t>(kh) * kDk;
            const float* k = row + static_cast<std::size_t>(kNk + kh) * kDk;
            const float* v = row + static_cast<std::size_t>(2 * kNk) * kDk + static_cast<std::size_t>(h) * kDv;
            double* S = state.data() + static_cast<std::size_t>(h) * kDk * kDv;
            const double decay = std::exp(static_cast<double>(g[static_cast<std::size_t>(t) * kNv + h]));
            const double b = beta[static_cast<std::size_t>(t) * kNv + h];
            for (int i = 0; i < kDk * kDv; ++i) S[i] *= decay;
            for (int j = 0; j < kDv; ++j) {
                double sk = 0;
                for (int i = 0; i < kDk; ++i) sk += S[i * kDv + j] * k[i];
                const double delta = (v[j] - sk) * b;
                double o = 0;
                for (int i = 0; i < kDk; ++i) {
                    S[i * kDv + j] += k[i] * delta;
                    o += S[i * kDv + j] * q[i];
                }
                out[(static_cast<std::size_t>(t) * kNv + h) * kDv + j] = o * kScale;
            }
        }
        if (snaps) (*snaps)[static_cast<std::size_t>(t)] = state;
    }
}

std::vector<double> toD(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

Result tolRms(const std::string& name, const std::vector<float>& got, const std::vector<double>& ref, double rtol) {
    const std::vector<double> sc(ref.size(), rmsOf(ref));
    return cmpTol(name, got, ref, sc, rtol, 0);
}

// gated RMSNorm per v head + silu(z) gate, in double
void gatedNormRef(const std::vector<double>& o, const std::vector<float>& z, const std::vector<float>& w, int n,
                  std::vector<double>& y) {
    y.assign(o.size(), 0.0);
    for (int t = 0; t < n; ++t)
        for (int h = 0; h < kNv; ++h) {
            const std::size_t b = (static_cast<std::size_t>(t) * kNv + h) * kDv;
            double ss = 0;
            for (int j = 0; j < kDv; ++j) ss += o[b + j] * o[b + j];
            const double r = 1.0 / std::sqrt(ss / kDv + kEps);
            for (int j = 0; j < kDv; ++j) {
                const double zz = z[b + j];
                y[b + j] = o[b + j] * r * w[static_cast<std::size_t>(j)] * (zz / (1.0 + std::exp(-zz)));
            }
        }
}

}  // namespace

void testGdn(Ctx& c) {
    c.rep.family = "gdn";
    const std::vector<float> convw = c.randn(static_cast<std::size_t>(kCh) * 4, 0.5f);
    Buf dconvw(convw);

    // ---- causal conv
    {
        const int n = 9;
        const std::vector<float> xin = c.randn(static_cast<std::size_t>(n) * kCh), st0 = c.randn(3 * kCh);
        // CPU
        std::vector<double> ref(static_cast<std::size_t>(n) * kCh), sc(ref.size());
        for (int ch = 0; ch < kCh; ++ch) {
            for (int t = 0; t < n; ++t) {
                double acc = 0, a = 0;
                for (int kk = 0; kk < 4; ++kk) {
                    const int tt = t - 3 + kk;
                    const double x = tt >= 0 ? xin[static_cast<std::size_t>(tt) * kCh + ch] : st0[static_cast<std::size_t>(3 + tt) * kCh + ch];
                    acc += x * convw[static_cast<std::size_t>(ch) * 4 + kk];
                    a += std::fabs(x * convw[static_cast<std::size_t>(ch) * 4 + kk]);
                }
                const double sig = 1.0 / (1.0 + std::exp(-acc));
                ref[static_cast<std::size_t>(t) * kCh + ch] = acc * sig;
                sc[static_cast<std::size_t>(t) * kCh + ch] = a + std::fabs(acc);
            }
        }
        std::vector<float> st_ref(3 * kCh);
        for (int kk = 0; kk < 3; ++kk)
            for (int ch = 0; ch < kCh; ++ch) st_ref[static_cast<std::size_t>(kk) * kCh + ch] = xin[static_cast<std::size_t>(n - 3 + kk) * kCh + ch];
        {
            Buf dx(xin), dst(st0), out(static_cast<std::size_t>(n) * kCh * 4);
            hip::launch(c.k.gdn_conv_seq, {cdiv(kCh, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dst.p(), dconvw.p(), out.p(),
                        kCh, n);
            c.sync();
            c.rep.add(cmpTol("gdn_conv_seq vs CPU", out.down<float>(ref.size()), ref, sc, 1e-5, 1e-7));
            c.rep.add(cmpExact("gdn_conv_seq rolling state", dst.down<float>(st_ref.size()), st_ref));
        }
        {
            Buf dx(xin), dst(st0), out(static_cast<std::size_t>(n) * kCh * 4), snap(3 * kCh * 4);
            hip::launch(c.k.gdn_conv_par, {cdiv(kCh, 256), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dx.p(), dst.p(),
                        dconvw.p(), out.p(), kCh, n);
            const int snap_after = 1;  // state right after token 1 (mixes the old state)
            hip::launch(c.k.gdn_conv_state, {cdiv(kCh, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), dst.p(), kCh, n, snap.p(),
                        snap_after);
            c.sync();
            c.rep.add(cmpTol("gdn_conv_par vs CPU", out.down<float>(ref.size()), ref, sc, 1e-5, 1e-7));
            c.rep.add(cmpExact("gdn_conv_state rolling state", dst.down<float>(st_ref.size()), st_ref));
            std::vector<float> sn(3 * kCh);
            for (int kk = 0; kk < 3; ++kk)
                for (int ch = 0; ch < kCh; ++ch) {
                    const int tt = snap_after - 2 + kk;
                    sn[static_cast<std::size_t>(kk) * kCh + ch] =
                        tt >= 0 ? xin[static_cast<std::size_t>(tt) * kCh + ch] : st0[static_cast<std::size_t>(3 + tt) * kCh + ch];
                }
            c.rep.add(cmpExact("gdn_conv_state snapshot", snap.down<float>(sn.size()), sn));
        }
    }

    // ---- prefill conv + q/k l2norm in one pass (gdn_conv_l2n / _h, gdn_conv_state_h)
    {
        const int n = 21;
        const std::vector<float> xin = c.randn(static_cast<std::size_t>(n) * kCh), st0 = c.randn(3 * kCh);
        Buf dx(xin), dst(st0);
        // reference: gdn_conv_par + l2norm of the 2 * n_k_heads q / k heads
        Buf ref(static_cast<std::size_t>(n) * kCh * 4);
        hip::launch(c.k.gdn_conv_par, {cdiv(kCh, 256), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dx.p(), dst.p(),
                    dconvw.p(), ref.p(), kCh, n);
        hip::launch(c.k.l2norm, {2 * kNk, static_cast<unsigned>(n), 1}, {128, 1, 1}, 0, c.s, ref.p(), kDk, kDk, kCh, kEps);
        c.sync();
        const auto want = ref.down<float>(static_cast<std::size_t>(n) * kCh);
        if (c.k.gdn_conv_l2n) {
            Buf out(want.size() * 4);
            hip::launch(c.k.gdn_conv_l2n, {kCh / 128, cdiv(n, 8), 1}, {128, 1, 1}, 0, c.s, dx.p(), dst.p(), dconvw.p(), out.p(), kCh, n,
                        2 * kNk, kEps);
            c.sync();
            c.rep.add(cmpExact("gdn_conv_l2n == gdn_conv_par + l2norm", out.down<float>(want.size()), want, Kind::invariant));
        }
        if (c.k.gdn_conv_l2n_h && c.k.gdn_conv_state_h) {
            // f16 input: the same as the f32 kernels on the f16-rounded input
            Buf x16(xin.size() * 2);
            hip::launch(c.k.f32_to_f16, {cdiv(xin.size() / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x16.p(), static_cast<int>(xin.size()));
            c.sync();
            std::vector<float> xr;
            for (auto h : x16.down<std::uint16_t>(xin.size())) xr.push_back(h2f(h));
            Buf dxr(xr), refh(want.size() * 4), out(want.size() * 4);
            hip::launch(c.k.gdn_conv_par, {cdiv(kCh, 256), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dxr.p(), dst.p(),
                        dconvw.p(), refh.p(), kCh, n);
            hip::launch(c.k.l2norm, {2 * kNk, static_cast<unsigned>(n), 1}, {128, 1, 1}, 0, c.s, refh.p(), kDk, kDk, kCh, kEps);
            hip::launch(c.k.gdn_conv_l2n_h, {kCh / 128, cdiv(n, 8), 1}, {128, 1, 1}, 0, c.s, x16.p(), dst.p(), dconvw.p(), out.p(), kCh,
                        n, 2 * kNk, kEps);
            c.sync();
            c.rep.add(cmpExact("gdn_conv_l2n_h (f16 x) == gdn_conv_par + l2norm on the f16 values", out.down<float>(want.size()),
                               refh.down<float>(want.size()), Kind::invariant));
            Buf sa(st0), sb(st0), na(3 * kCh * 4), nb(3 * kCh * 4);
            hip::launch(c.k.gdn_conv_state_h, {cdiv(kCh, 256), 1, 1}, {256, 1, 1}, 0, c.s, x16.p(), sa.p(), kCh, n, na.p(), 1);
            hip::launch(c.k.gdn_conv_state, {cdiv(kCh, 256), 1, 1}, {256, 1, 1}, 0, c.s, dxr.p(), sb.p(), kCh, n, nb.p(), 1);
            c.sync();
            const bool same = sa.down<float>(3 * kCh) == sb.down<float>(3 * kCh) && na.down<float>(3 * kCh) == nb.down<float>(3 * kCh);
            Result r;
            r.name = "gdn_conv_state_h (f16 x) == gdn_conv_state on the f16 values (state + snapshot)";
            r.kind = Kind::invariant;
            r.n = 6 * kCh;
            r.mismatches = same ? 0 : 1;
            r.pass = same;
            c.rep.add(r);
        }
    }

    // ---- l2norm of q / k heads (in place), gates
    {
        const int n = 4;
        std::vector<float> x = c.randn(static_cast<std::size_t>(n) * kCh);
        Buf dx(x);
        hip::launch(c.k.l2norm, {2 * kNk, static_cast<unsigned>(n), 1}, {128, 1, 1}, 0, c.s, dx.p(), kDk, kDk, kCh, kEps);
        c.sync();
        std::vector<double> ref = toD(x);
        for (int t = 0; t < n; ++t)
            for (int hh = 0; hh < 2 * kNk; ++hh) {
                const std::size_t b = static_cast<std::size_t>(t) * kCh + static_cast<std::size_t>(hh) * kDk;
                double s = 0;
                for (int i = 0; i < kDk; ++i) s += ref[b + i] * ref[b + i];
                const double r = 1.0 / std::sqrt(s + kEps);
                for (int i = 0; i < kDk; ++i) ref[b + i] *= r;
            }
        c.rep.add(cmpTolRel("l2norm vs CPU", dx.down<float>(ref.size()), ref, 1e-6, 1e-7));

        const int nh = n * kNv;
        std::vector<float> b = c.randn(static_cast<std::size_t>(nh), 2.f), a = c.randn(static_cast<std::size_t>(nh), 3.f);
        a[0] = 25.f;  // softplus linear branch
        const std::vector<float> dtb = c.randn(kNv), A = c.randu(kNv, -2.f, -0.1f);
        Buf db(b), da(a), ddt(dtb), dA(A);
        hip::launch(c.k.gdn_gates, {cdiv(nh, 64), 1, 1}, {64, 1, 1}, 0, c.s, db.p(), da.p(), ddt.p(), dA.p(), nh, kNv);
        c.sync();
        std::vector<double> rb(static_cast<std::size_t>(nh)), ra(static_cast<std::size_t>(nh));
        for (int i = 0; i < nh; ++i) {
            const int h = i % kNv;
            rb[static_cast<std::size_t>(i)] = 1.0 / (1.0 + std::exp(-static_cast<double>(b[static_cast<std::size_t>(i)])));
            const double z = static_cast<double>(a[static_cast<std::size_t>(i)]) + dtb[static_cast<std::size_t>(h)];
            ra[static_cast<std::size_t>(i)] = (z > 20 ? z : std::log1p(std::exp(z))) * A[static_cast<std::size_t>(h)];
        }
        c.rep.add(cmpTolRel("gdn_gates beta vs CPU", db.down<float>(rb.size()), rb, 1e-5, 1e-7));
        c.rep.add(cmpTolRel("gdn_gates g vs CPU", da.down<float>(ra.size()), ra, 1e-5, 1e-7));
        // gdn_gates_ba on [beta | alpha] per token == gdn_gates (bitwise)
        if (c.k.gdn_gates_ba != nullptr) {
            std::vector<float> ba(static_cast<std::size_t>(2 * nh));
            for (int t = 0; t < n; ++t)
                for (int h = 0; h < kNv; ++h) {
                    ba[static_cast<std::size_t>(t * 2 * kNv + h)] = b[static_cast<std::size_t>(t * kNv + h)];
                    ba[static_cast<std::size_t>(t * 2 * kNv + kNv + h)] = a[static_cast<std::size_t>(t * kNv + h)];
                }
            Buf dba(ba), db2(static_cast<std::size_t>(nh) * 4), da2(static_cast<std::size_t>(nh) * 4);
            db2.fill(0xff);
            da2.fill(0xff);
            hip::launch(c.k.gdn_gates_ba, {cdiv(nh, 64), 1, 1}, {64, 1, 1}, 0, c.s, dba.p(), db2.p(), da2.p(), ddt.p(), dA.p(), nh, kNv);
            c.sync();
            c.rep.add(cmpExact("gdn_gates_ba beta == gdn_gates", db2.down<float>(rb.size()), db.down<float>(rb.size()), Kind::invariant));
            c.rep.add(cmpExact("gdn_gates_ba g == gdn_gates", da2.down<float>(ra.size()), da.down<float>(ra.size()), Kind::invariant));
        } else {
            // gfx1151: not ported; the forward runs the separate beta / alpha GEMMs when it is null
            c.rep.skip("gdn", "gdn_gates_ba == gdn_gates", "kernel absent in this module (forward uses separate GEMMs)");
        }
    }

    // ---- recurrence
    auto inputs = [&](int n, std::vector<float>& qkv, std::vector<float>& g, std::vector<float>& beta) {
        qkv = makeQkv(c, n);
        g = c.randu(static_cast<std::size_t>(n) * kNv, -1.5f, -0.01f);
        beta = c.randu(static_cast<std::size_t>(n) * kNv, 0.05f, 0.95f);
    };
    const std::vector<float> state0 = c.randn(static_cast<std::size_t>(kNv) * kDk * kDv, 0.3f);
    {
        const int n = 40, snap_after = 17;
        std::vector<float> qkv, g, beta;
        inputs(n, qkv, g, beta);
        std::vector<double> st = toD(state0), out;
        std::vector<std::vector<double>> snaps;
        deltaRef(qkv, g, beta, n, st, out, &snaps);
        Buf dq(qkv), dg(g), db(beta);
        for (const char* name : {"gdn_seq_128", "gdn_seq_128_l8"}) {
            const bool l8 = std::string(name) == "gdn_seq_128_l8";
            Buf dst(state0), dout(out.size() * 4), dsnap(state0.size() * 4);
            hip::launch(c.fn(name), {kNv, l8 ? 8u : 4u, 1}, {128, 1, 1}, 0, c.s, dq.p(), dg.p(), db.p(), dst.p(), dout.p(), n,
                        kNk, kNv, kDv, kCh, kScale, l8 ? DevPtr{0} : dsnap.p(), snap_after);
            c.sync();
            c.rep.add(tolRms(std::string(name) + " out vs CPU (40 tokens)", dout.down<float>(out.size()), out, 1e-4));
            c.rep.add(tolRms(std::string(name) + " final state vs CPU", dst.down<float>(st.size()), st, 1e-4));
            if (!l8)
                c.rep.add(tolRms(std::string(name) + " snapshot (after token 17) vs CPU", dsnap.down<float>(st.size()),
                                 snaps[static_cast<std::size_t>(snap_after)], 1e-4));
        }
    }
    {
        const int n = c.quick ? 130 : 200, nchunks = (n + 63) / 64;
        std::vector<float> qkv, g, beta;
        inputs(n, qkv, g, beta);
        std::vector<double> st = toD(state0), out;
        deltaRef(qkv, g, beta, n, st, out, nullptr);
        Buf dq(qkv), dg(g), db(beta);
        const std::size_t gsz = static_cast<std::size_t>(nchunks) * kNv * 64;
        Buf W(gsz * 128 * 4), U(gsz * 128 * 4), M(gsz * 64 * 4), G(gsz * 4);
        {
            Buf dst(state0), dout(out.size() * 4);
            hip::launch(c.k.gdn_chunk_prep, {static_cast<unsigned>(nchunks), kNv, 1}, {256, 1, 1}, 0, c.s, dq.p(), dg.p(),
                        db.p(), W.p(), U.p(), M.p(), G.p(), n, kNk, kNv, kCh);
            hip::launch(c.k.gdn_chunk_scan, {kNv, kDv / 32, 1}, {256, 1, 1}, 0, c.s, dq.p(), W.p(), U.p(), M.p(), G.p(),
                        dst.p(), dout.p(), n, kNk, kNv, kCh, kScale);
            c.sync();
            c.rep.add(tolRms("gdn_chunk_prep + gdn_chunk_scan out vs CPU (" + std::to_string(n) + " tokens)",
                             dout.down<float>(out.size()), out, 1e-3));
            c.rep.add(tolRms("gdn_chunk_scan final state vs CPU", dst.down<float>(st.size()), st, 1e-3));
        }
        if (c.k.gdn_wprep && c.k.gdn_wscan8) {
            Buf dst(state0), dout(out.size() * 4);
            hip::launch(c.k.gdn_wprep, {kNv, static_cast<unsigned>(nchunks), 1}, {256, 1, 1}, 0, c.s, dq.p(), dg.p(), db.p(),
                        W.p(), G.p(), n, kNk, kNv, kCh);
            hip::launch(c.k.gdn_wscan8, {kNv, kDv / 128, 1}, {256, 1, 1}, 0, c.s, dq.p(), W.p(), G.p(), db.p(), dst.p(),
                        dout.p(), n, kNk, kNv, kCh, kScale);
            c.sync();
            c.rep.add(tolRms("gdn_wprep + gdn_wscan8 (f16 WMMA) out vs CPU", dout.down<float>(out.size()), out, 2e-2));
            c.rep.add(tolRms("gdn_wscan8 final state vs CPU", dst.down<float>(st.size()), st, 2e-2));
        }
    }

    // ---- gated norm
    const std::vector<float> wn = c.randu(kDv, 0.5f, 1.5f);
    Buf dwn(wn);
    {
        const int n = 3;
        const std::vector<float> o = c.randn(static_cast<std::size_t>(n) * kDInner), z = c.randn(static_cast<std::size_t>(n) * kDInner);
        Buf dO(o), dz(z);
        hip::launch(c.k.gdn_gated_norm, {static_cast<unsigned>(n * kNv), 1, 1}, {128, 1, 1}, 0, c.s, dO.p(), dz.p(), dwn.p(),
                    kDv, kEps);
        c.sync();
        std::vector<double> y;
        gatedNormRef(toD(o), z, wn, n, y);
        c.rep.add(cmpTolRel("gdn_gated_norm vs CPU", dO.down<float>(y.size()), y, 1e-5, 1e-6));
    }

    // ---- fused decode kernels (segments, snapshots, replay)
    {
        const int n = 6;
        const std::vector<float> xin = c.randn(static_cast<std::size_t>(n) * kCh);
        const std::vector<float> cst0 = c.randn(3 * kCh);
        std::vector<float> qkv, g, beta;
        inputs(n, qkv, g, beta);
        const std::vector<float> z = c.randn(static_cast<std::size_t>(n) * kDInner);
        Buf dxin(xin), dq(qkv), dg(g), db(beta), dz(z);
        const std::size_t sbytes = state0.size() * 4, cbytes = 3 * kCh * 4;

        // conv: one 6-row segment with snapshots
        Buf cs_a(cst0), out_a(static_cast<std::size_t>(n) * kCh * 4);
        std::vector<Buf> csnap;
        wk::GdnSegs segs;
        segs.s[0].state = cs_a.p();
        segs.s[0].row0 = 0;
        segs.s[0].nrows = n;
        for (int t = 0; t < n; ++t) {
            csnap.emplace_back(cbytes);
            segs.s[0].snap[t] = csnap.back().p();
        }
        hip::launch(c.k.gdn_conv_l2, {kCh / 128, 1, 1}, {128, 1, 1}, 0, c.s, dxin.p(), dconvw.p(), out_a.p(), kCh, 2 * kNk,
                    kEps, segs);
        // conv: row by row
        Buf cs_b(cst0), out_b(static_cast<std::size_t>(n) * kCh * 4);
        std::vector<std::vector<float>> cstates;
        for (int t = 0; t < n; ++t) {
            wk::GdnSegs s1;
            s1.s[0].state = cs_b.p();
            s1.s[0].row0 = t;
            s1.s[0].nrows = 1;
            hip::launch(c.k.gdn_conv_l2, {kCh / 128, 1, 1}, {128, 1, 1}, 0, c.s, dxin.p(), dconvw.p(), out_b.p(), kCh,
                        2 * kNk, kEps, s1);
            c.sync();
            cstates.push_back(cs_b.down<float>(3 * kCh));
        }
        c.sync();
        const auto oa = out_a.down<float>(static_cast<std::size_t>(n) * kCh), ob = out_b.down<float>(oa.size());
        c.rep.add(cmpExact("gdn_conv_l2 6-row segment == 6 single-row launches (out)", oa, ob, Kind::invariant));
        c.rep.add(cmpExact("gdn_conv_l2 segment state == row-by-row", cs_a.down<float>(3 * kCh), cstates.back(), Kind::invariant));
        {
            std::size_t bad = 0;
            for (int t = 0; t < n; ++t) bad += csnap[static_cast<std::size_t>(t)].down<float>(3 * kCh) != cstates[static_cast<std::size_t>(t)];
            Result r;
            r.name = "gdn_conv_l2 snapshots == per-row states";
            r.kind = Kind::invariant;
            r.n = static_cast<std::size_t>(n);
            r.mismatches = bad;
            r.pass = bad == 0;
            c.rep.add(r);
        }
        // conv + l2norm vs CPU
        {
            std::vector<double> ref(oa.size());
            for (int t = 0; t < n; ++t) {
                for (int ch = 0; ch < kCh; ++ch) {
                    double acc = 0;
                    for (int kk = 0; kk < 4; ++kk) {
                        const int tt = t - 3 + kk;
                        const double x = tt >= 0 ? xin[static_cast<std::size_t>(tt) * kCh + ch] : cst0[static_cast<std::size_t>(3 + tt) * kCh + ch];
                        acc += x * convw[static_cast<std::size_t>(ch) * 4 + kk];
                    }
                    ref[static_cast<std::size_t>(t) * kCh + ch] = acc / (1.0 + std::exp(-acc));
                }
                for (int hh = 0; hh < 2 * kNk; ++hh) {
                    const std::size_t b = static_cast<std::size_t>(t) * kCh + static_cast<std::size_t>(hh) * kDk;
                    double s = 0;
                    for (int i = 0; i < kDk; ++i) s += ref[b + i] * ref[b + i];
                    for (int i = 0; i < kDk; ++i) ref[b + i] /= std::sqrt(s + kEps);
                }
            }
            c.rep.add(cmpTolRel("gdn_conv_l2 vs CPU (conv + SiLU + q/k l2norm)", oa, ref, 1e-5, 1e-6));
        }
        // conv replay: record 6 rows (pend), then apply 4 kept rows
        if (!c.k.caps.gdn_replay) {
            c.rep.skip("gdn", "gdn_conv_l2 replay", "replay-mode fused DeltaNet kernels not in this code object");
        } else {
            Buf cs(cst0), pend(static_cast<std::size_t>(n) * kCh * 4), out(oa.size() * 4);
            wk::GdnSegs s;
            s.s[0].state = cs.p();
            s.s[0].nrows = n;
            s.s[0].pend = pend.p();
            hip::launch(c.k.gdn_conv_l2, {kCh / 128, 1, 1}, {128, 1, 1}, 0, c.s, dxin.p(), dconvw.p(), out.p(), kCh, 2 * kNk,
                        kEps, s);
            c.sync();
            const bool untouched = cs.down<float>(3 * kCh) == cst0;
            s.s[0].nrows = 0;
            s.s[0].npend = 4;
            hip::launch(c.k.gdn_conv_l2, {kCh / 128, 1, 1}, {128, 1, 1}, 0, c.s, dxin.p(), dconvw.p(), out.p(), kCh, 2 * kNk,
                        kEps, s);
            c.sync();
            Result r = cmpExact("gdn_conv_l2 replay of 4 kept rows == state after 4 rows", cs.down<float>(3 * kCh), cstates[3],
                                Kind::invariant);
            if (!untouched) {
                r.pass = false;
                r.note += " (recording run modified the state)";
            }
            c.rep.add(r);
        }

        // delta-rule step + gated norm + q8
        auto step = [&](hip::Function f, const wk::GdnSegs& s, unsigned nseg, DevPtr out, DevPtr xq, DevPtr xd) {
            hip::launch(f, {kNv, nseg, 1}, {512, 1, 1}, 0, c.s, dq.p(), dg.p(), db.p(), dz.p(), dwn.p(), out, xq, xd, kNk,
                        kScale, kEps, kNv, kCh, s);
        };
        const std::size_t no = static_cast<std::size_t>(n) * kDInner;
        Buf st_a(state0), y_a(no * 4), q_a(no), d_a(no / 32 * 4);
        std::vector<Buf> ssnap;
        wk::GdnSegs sa;
        sa.s[0].state = st_a.p();
        sa.s[0].nrows = n;
        for (int t = 0; t < n; ++t) {
            ssnap.emplace_back(sbytes);
            sa.s[0].snap[t] = ssnap.back().p();
        }
        step(c.k.gdn_step_norm, sa, 1, y_a.p(), q_a.p(), d_a.p());
        Buf st_b(state0), y_b(no * 4), q_b(no), d_b(no / 32 * 4);
        std::vector<std::vector<float>> sstates;
        for (int t = 0; t < n; ++t) {
            wk::GdnSegs s1;
            s1.s[0].state = st_b.p();
            s1.s[0].row0 = t;
            s1.s[0].nrows = 1;
            step(c.k.gdn_step_norm, s1, 1, y_b.p(), q_b.p(), d_b.p());
            c.sync();
            sstates.push_back(st_b.down<float>(state0.size()));
        }
        Buf st_c(state0), y_c(no * 4), q_c(no), d_c(no / 32 * 4);
        wk::GdnSegs sc;
        sc.s[0].state = st_c.p();
        sc.s[0].nrows = n;
        const hip::Function fv0 = c.fnOpt("gdn_step_norm_v0");  // gfx1201 only
        if (fv0) step(fv0, sc, 1, y_c.p(), q_c.p(), d_c.p());
        c.sync();
        const auto ya = y_a.down<float>(no);
        c.rep.add(cmpExact("gdn_step_norm 6-row segment == 6 single-row launches (out)", ya, y_b.down<float>(no), Kind::invariant));
        c.rep.add(cmpExact("gdn_step_norm segment xq == row-by-row", q_a.down<std::int8_t>(no), q_b.down<std::int8_t>(no), Kind::invariant));
        c.rep.add(cmpExact("gdn_step_norm segment xd == row-by-row", d_a.down<float>(no / 32), d_b.down<float>(no / 32), Kind::invariant));
        c.rep.add(cmpExact("gdn_step_norm segment state == row-by-row", st_a.down<float>(state0.size()), sstates.back(), Kind::invariant));
        {
            std::size_t bad = 0;
            for (int t = 0; t < n; ++t) bad += ssnap[static_cast<std::size_t>(t)].down<float>(state0.size()) != sstates[static_cast<std::size_t>(t)];
            Result r;
            r.name = "gdn_step_norm snapshots == per-row states";
            r.kind = Kind::invariant;
            r.n = static_cast<std::size_t>(n);
            r.mismatches = bad;
            r.pass = bad == 0;
            c.rep.add(r);
        }
        if (fv0) {
            c.rep.add(cmpExact("gdn_step_norm == gdn_step_norm_v0 (out)", ya, y_c.down<float>(no), Kind::invariant));
            c.rep.add(cmpExact("gdn_step_norm == gdn_step_norm_v0 (xq)", q_a.down<std::int8_t>(no), q_c.down<std::int8_t>(no), Kind::invariant));
            c.rep.add(cmpExact("gdn_step_norm == gdn_step_norm_v0 (state)", st_a.down<float>(state0.size()), st_c.down<float>(state0.size()), Kind::invariant));
        } else {
            c.rep.skip("gdn", "gdn_step_norm == gdn_step_norm_v0", "kernel not in this code object");
        }
        // xq / xd == quantize_q8(out)
        {
            std::vector<std::int8_t> rq(no);
            std::vector<float> rd(no / 32);
            ref::quantizeQ8(ya.data(), static_cast<int>(no), rq.data(), rd.data());
            c.rep.add(cmpExact("gdn_step_norm xq == quantize_q8(out)", q_a.down<std::int8_t>(no), rq));
            c.rep.add(cmpExact("gdn_step_norm xd == quantize_q8(out)", d_a.down<float>(no / 32), rd));
        }
        // vs CPU: delta rule + gated norm
        {
            std::vector<double> st = toD(state0), o, y;
            std::vector<std::vector<double>> snaps;
            deltaRef(qkv, g, beta, n, st, o, &snaps);
            gatedNormRef(o, z, wn, n, y);
            c.rep.add(cmpTolRel("gdn_step_norm out vs CPU", ya, y, 1e-4, 1e-5));
            c.rep.add(tolRms("gdn_step_norm state vs CPU", st_a.down<float>(state0.size()), st, 1e-4));
        }
        // replay: record 6 rows, apply 4 kept rows
        if (!c.k.caps.gdn_replay) {
            c.rep.skip("gdn", "gdn_step_norm replay", "replay-mode fused DeltaNet kernels not in this code object");
        } else {
            const int PW = kDk + kDv + 2;
            Buf st(state0), pend(static_cast<std::size_t>(n) * kNv * PW * 4), y(no * 4), q(no), d(no / 32 * 4);
            wk::GdnSegs s;
            s.s[0].state = st.p();
            s.s[0].nrows = n;
            s.s[0].pend = pend.p();
            step(c.k.gdn_step_norm, s, 1, y.p(), q.p(), d.p());
            c.sync();
            const bool untouched = st.down<float>(state0.size()) == state0;
            const bool same_out = y.down<float>(no) == ya;
            s.s[0].nrows = 0;
            s.s[0].npend = 4;
            step(c.k.gdn_step_norm, s, 1, y.p(), q.p(), d.p());
            c.sync();
            Result r = cmpExact("gdn_step_norm replay of 4 kept rows == state after 4 rows", st.down<float>(state0.size()),
                                sstates[3], Kind::invariant);
            if (!untouched || !same_out) {
                r.pass = false;
                r.note += !untouched ? " (recording run modified the state)" : " (recording run output differs)";
            }
            c.rep.add(r);
        }
        // two segments (rows 0..2 on state A, rows 3..5 on state B) in one launch == separate
        {
            Buf sA(state0), sB(state0), y(no * 4), q(no), d(no / 32 * 4);
            wk::GdnSegs s2;
            s2.s[0].state = sA.p();
            s2.s[0].row0 = 0;
            s2.s[0].nrows = 3;
            s2.s[1].state = sB.p();
            s2.s[1].row0 = 3;
            s2.s[1].nrows = 3;
            step(c.k.gdn_step_norm, s2, 2, y.p(), q.p(), d.p());
            Buf tA(state0), tB(state0), y2(no * 4), q2(no), d2(no / 32 * 4);
            for (int sgi = 0; sgi < 2; ++sgi) {
                wk::GdnSegs s1;
                s1.s[0].state = sgi ? tB.p() : tA.p();
                s1.s[0].row0 = 3 * sgi;
                s1.s[0].nrows = 3;
                step(c.k.gdn_step_norm, s1, 1, y2.p(), q2.p(), d2.p());
            }
            c.sync();
            const bool same = y.down<float>(no) == y2.down<float>(no) && sA.down<float>(state0.size()) == tA.down<float>(state0.size()) &&
                              sB.down<float>(state0.size()) == tB.down<float>(state0.size());
            Result r;
            r.name = "gdn_step_norm two segments in one launch == separate launches";
            r.kind = Kind::invariant;
            r.n = no;
            r.mismatches = same ? 0 : 1;
            r.pass = same;
            c.rep.add(r);
        }
    }
}

}  // namespace kt
