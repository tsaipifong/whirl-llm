// whirl-kernel-test family "i8qk" (feat027-i8qk, gfx1201): int8 Q.K^T prefill attention.
// SPDX-License-Identifier: Apache-2.0
//   * i8k_mean vs the CPU (i8qk_ref.h: same sampled rows, same float summation order);
//   * i8k_quant vs the CPU on the GPU's mu (int8 codes within 1, scales within 1e-6 relative);
//   * attn_kgsi6 / 4 / 2 over one key range and over ranges of 1024 keys (each range its own mu, the softmax state
//     carried) vs the CPU emulation of the same int8 numerics (i8qk::scoresI8 + softmax + f16 P), tight;
//   * the same outputs vs exact double attention and vs the f16 attn_kgs (the error the fast path adds), bounded.
// Data: Swift shapes (24 / 4 heads, hd 256), K with a per-channel bias and outlier channels, q and k
// Hadamard-rotated (what the q4 KV fast path stores), 64 queries at the end of 2048 keys; output gate +30.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include "i8qk_ref.h"
#include "kt.h"

namespace kt {

namespace {

void wht(float* x, int hd) {
    for (int len = 1; len < hd; len <<= 1)
        for (int i = 0; i < hd; i += 2 * len)
            for (int j = i; j < i + len; ++j) {
                const float a = x[j], b = x[j + len];
                x[j] = a + b;
                x[j + len] = a - b;
            }
    const float sc = 1.f / std::sqrt(static_cast<float>(hd));
    for (int d = 0; d < hd; ++d) x[d] *= sc;
}

double relL2(const std::vector<float>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        num += (a[i] - b[i]) * (a[i] - b[i]);
        den += b[i] * b[i];
    }
    return std::sqrt(num / std::max(den, 1e-300));
}

}  // namespace

void testAttnI8qk(Ctx& c) {
    c.rep.family = "i8qk";
    const auto fm = c.fnOpt("i8k_mean"), fq = c.fnOpt("i8k_quant");
    if (!fm || !fq || !c.fnOpt("attn_kgsi6")) {
        // not a missing-kernel skip: gfx1201 only by design (iu8 WMMA = f16 rate on gfx1151)
        c.rep.skip("i8qk", "attn_kgsi", "gfx1201 only by design (feat027-i8qk; no int8 Q.K^T gain on this GPU)");
        return;
    }
    i8qk::Shape s;
    s.n = 64;
    s.pos0 = 2048 - 64;
    s.n_head = 24;
    s.n_kv = 4;
    const int hd = s.hd, rows = s.pos0 + s.n, rel = s.n_kv * hd;
    // data
    std::normal_distribution<float> N(0.f, 1.f);
    std::vector<float> w(static_cast<std::size_t>(rel), 1.f), bias(static_cast<std::size_t>(rel));
    for (int x = 0; x < rel; ++x) {
        if (x % hd == 3 || x % hd == 77 || x % hd == 130 || x % hd == 200) w[static_cast<std::size_t>(x)] = 8.f;
        bias[static_cast<std::size_t>(x)] = 2.f * N(c.rng);
    }
    std::vector<float> k(static_cast<std::size_t>(rows) * rel), v(k.size());
    for (int r = 0; r < rows; ++r) {
        for (int x = 0; x < rel; ++x) {
            const std::size_t i = static_cast<std::size_t>(r) * rel + x;
            k[i] = bias[static_cast<std::size_t>(x)] + w[static_cast<std::size_t>(x)] * N(c.rng);
            v[i] = roundh(N(c.rng));
        }
        for (int g = 0; g < s.n_kv; ++g) {
            float* kr = &k[(static_cast<std::size_t>(r) * s.n_kv + g) * hd];
            wht(kr, hd);
            for (int d = 0; d < hd; ++d) kr[d] = roundh(kr[d]);
        }
    }
    std::vector<float> q(static_cast<std::size_t>(s.n) * s.n_head * s.q_stride);
    for (int i = 0; i < s.n; ++i)
        for (int h = 0; h < s.n_head; ++h) {
            float* qr = &q[(static_cast<std::size_t>(i) * s.n_head + h) * s.q_stride];
            for (int d = 0; d < hd; ++d) qr[d] = N(c.rng) * (d % 64 == 5 ? 8.f : 1.f);
            wht(qr, hd);
            for (int d = 0; d < hd; ++d) qr[hd + d] = 30.f;  // output gate sigmoid(30) = 1.0f
        }
    std::vector<std::uint16_t> k16(k.size()), v16(v.size());
    for (std::size_t i = 0; i < k.size(); ++i) {
        k16[i] = f2h(k[i]);
        v16[i] = f2h(v[i]);
    }
    Buf dk(k16), dv(v16), dq(q), dpos(std::vector<int>{s.pos0});
    std::vector<int> idt(static_cast<std::size_t>(rows / wk::kKvPage + 2));
    std::iota(idt.begin(), idt.end(), 0);
    Buf did(idt), st(static_cast<std::size_t>(s.n) * s.n_head * 2 * 4);
    const std::size_t on = static_cast<std::size_t>(s.n) * s.n_head * hd;

    // CPU exact attention (double) and the f16 attn_kgs6 on the same copy
    std::vector<double> exact(on);
    {
        std::vector<double> sc;
        for (int i = 0; i < s.n; ++i)
            for (int h = 0; h < s.n_head; ++h) {
                i8qk::scoresRef(s, q, k, i, h, false, roundh, sc);
                i8qk::softmaxPv(s, sc, v, h, false, roundh, &exact[(static_cast<std::size_t>(i) * s.n_head + h) * hd]);
            }
    }
    wk::KvArgs a{};
    a.k = dk.p();
    a.v = dv.p();
    a.ptab = did.p();
    std::vector<float> f16out;
    if (const auto fks = c.fnOpt("attn_kgs6")) {
        Buf ob(on * 4);
        ob.fill(0x7f);
        hip::launch(fks, {cdiv(s.n, 16), static_cast<unsigned>(s.n_head / 6), 1}, {384, 1, 1}, 0, c.s, dq.p(), a, ob.p(), s.n_head, s.n_kv, s.q_stride,
                    dpos.p(), s.n, s.scale, 0, 0, rows, st.p());
        c.sync();
        f16out = ob.down<float>(on);
    }

    for (const int seg : {1 << 20, 1024}) {
        const int kss = std::min(seg, rows) + 32;
        Buf kq(static_cast<std::size_t>(std::min(seg, rows)) * rel), ksc(static_cast<std::size_t>(s.n_kv) * kss * 4), mu(static_cast<std::size_t>(rel) * 4);
        ksc.fill(0);
        for (const auto& [name, np] : {std::pair{"attn_kgsi6", 6}, std::pair{"attn_kgsi4", 4}, std::pair{"attn_kgsi2", 2}}) {
            const auto fk = c.fnOpt(name);
            if (!fk || (s.n_head / s.n_kv) % np != 0) continue;
            Buf ob(on * 4);
            ob.fill(0x7f);
            std::vector<double> emu(on);
            std::vector<std::vector<double>> sc_all(static_cast<std::size_t>(s.n) * s.n_head);
            const std::string where = std::string(name) + (seg >= rows ? " (one key range)" : " (key ranges of 1024)");
            for (int r0 = 0; r0 < rows; r0 += seg) {
                const int r1 = std::min(rows, r0 + seg), nr = r1 - r0;
                const DevPtr kr0 = dk.p() + static_cast<DevPtr>(r0) * rel * 2;
                hip::launch(fm, {static_cast<unsigned>(rel / 64), 1, 1}, {256, 1, 1}, 0, c.s, kr0, mu.p(), nr, rel);
                hip::launch(fq, {cdiv(static_cast<std::uint64_t>(nr) * s.n_kv, 8), 1, 1}, {256, 1, 1}, 0, c.s, kr0, mu.p(), kq.p(), ksc.p(), nr, s.n_kv, kss);
                const DevPtr kqb = kq.p() - static_cast<DevPtr>(r0) * rel, kscb = ksc.p() - static_cast<DevPtr>(r0) * 4;
                hip::launch(fk, {cdiv(s.n, 16), static_cast<unsigned>(s.n_head / np), 1}, {static_cast<unsigned>(64 * np), 1, 1}, 0, c.s, dq.p(), a, ob.p(),
                            s.n_head, s.n_kv, s.q_stride, dpos.p(), s.n, s.scale, 0, r0, r1, st.p(), kqb, kscb, kss, mu.p());
                c.sync();
                // CPU on this range's copy: mean (vs the GPU's), quantization on the GPU's mu, scores
                const std::vector<float> kr(k.begin() + static_cast<std::ptrdiff_t>(r0) * rel, k.begin() + static_cast<std::ptrdiff_t>(r1) * rel);
                const std::vector<float> mug = mu.down<float>(static_cast<std::size_t>(rel));
                if (std::string(name) == "attn_kgsi6") {
                    const std::vector<float> muc = i8qk::meanSampled(kr, nr, rel);
                    std::vector<double> mud(muc.begin(), muc.end());
                    c.rep.add(cmpTolRel("i8k_mean vs CPU (range " + std::to_string(r0) + ")", mug, mud, 1e-6, 1e-6));
                }
                std::vector<std::int8_t> kqc;
                std::vector<float> kscc;
                i8qk::quantK(kr, mug, nr, s.n_kv, hd, kqc, kscc);
                const auto kqg = kq.down<std::int8_t>(static_cast<std::size_t>(nr) * rel);
                const auto kscg_all = ksc.down<float>(static_cast<std::size_t>(s.n_kv) * kss);
                if (std::string(name) == "attn_kgsi6") {
                    std::vector<float> kqf(kqg.begin(), kqg.end());
                    std::vector<double> kqd(kqc.begin(), kqc.end());
                    c.rep.add(cmpTolRel("i8k_quant codes vs CPU, within 1 (range " + std::to_string(r0) + ")", kqf, kqd, 0.0, 1.0));
                    std::vector<float> sg;
                    std::vector<double> sr;
                    for (int g = 0; g < s.n_kv; ++g)
                        for (int r = 0; r < nr; ++r) {
                            sg.push_back(kscg_all[static_cast<std::size_t>(g) * kss + r]);
                            sr.push_back(kscc[static_cast<std::size_t>(g) * nr + r]);
                        }
                    c.rep.add(cmpTolRel("i8k_quant scales vs CPU (range " + std::to_string(r0) + ")", sg, sr, 1e-6, 0.0));
                }
                // the GPU's codes and scales for the emulated scores (isolates the attention kernel)
                std::vector<float> kscg(static_cast<std::size_t>(s.n_kv) * nr);
                for (int g = 0; g < s.n_kv; ++g)
                    for (int r = 0; r < nr; ++r) kscg[static_cast<std::size_t>(g) * nr + r] = kscg_all[static_cast<std::size_t>(g) * kss + r];
                for (int i = 0; i < s.n; ++i) {
                    const int vis = s.pos0 + i + 1;  // keys 0 .. vis - 1
                    if (vis <= r0) continue;
                    i8qk::Shape sr = s;
                    sr.pos0 = std::min(vis, r1) - r0 - 1 - i;  // scoresI8 covers keys [0, sr.pos0 + i + 1) of the range
                    for (int h = 0; h < s.n_head; ++h) {
                        std::vector<double> sc;
                        i8qk::scoresI8(sr, q, kqg, kscg, mug, nr, i, h, sc);
                        auto& all = sc_all[static_cast<std::size_t>(i) * s.n_head + h];
                        all.insert(all.end(), sc.begin(), sc.end());
                    }
                }
            }
            for (int i = 0; i < s.n; ++i)
                for (int h = 0; h < s.n_head; ++h)
                    i8qk::softmaxPv(s, sc_all[static_cast<std::size_t>(i) * s.n_head + h], v, h, true, roundh,
                                    &emu[(static_cast<std::size_t>(i) * s.n_head + h) * hd]);
            const auto got = ob.down<float>(on);
            // vs the emulation of the same int8 numerics: differences are f32 epilogue rounding and the online
            // softmax's per-tile f16 P (vs the CPU's global max)
            c.rep.add(cmpTolRel(where + " vs CPU int8 emulation", got, emu, 5e-3, 2e-3));
            const double e_i8 = relL2(got, exact);
            Result r;
            r.family = "i8qk";
            r.kind = Kind::tol;
            r.n = got.size();
            if (!f16out.empty()) {
                const double e_f16 = relL2(f16out, exact);
                r.name = where + " error vs exact (rel L2 <= 5e-2; f16 attn_kgs6 shown)";
                r.metric = e_i8 / 5e-2;
                r.pass = e_i8 <= 5e-2;
                char buf[160];
                std::snprintf(buf, sizeof buf, "int8 %.3e, f16 attn_kgs6 %.3e", e_i8, e_f16);
                r.note = buf;
            } else {
                r.name = where + " error vs exact (rel L2 <= 5e-2)";
                r.metric = e_i8 / 5e-2;
                r.pass = e_i8 <= 5e-2;
            }
            std::printf("   %s: rel L2 vs exact %.3e%s\n", where.c_str(), e_i8, f16out.empty() ? "" : (" (f16 attn_kgs6 " + std::to_string(relL2(f16out, exact)) + ")").c_str());
            c.rep.add(r);
        }
    }
}

}  // namespace kt
