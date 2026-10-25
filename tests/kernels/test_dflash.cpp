// DFlash2 drafter kernels vs C++ CPU references (the formulas of
// tools/dflash_ref/ref_drafter.py in whirl-cloud, i.e. llama.cpp's DFlash2):
//   * dflash_conv: both sides, causal inside the block, grouped dynamic weights
//   * dflash_topk: key order with ties to the smaller index, NaN never chosen, repeatable
//   * dflash_lattice: f32 and Q8_0 codebooks, anchor row for position 0
//   * dflash_walk: same tokens as the CPU walk on the same lattice (p-min / n-min)
//   * dflash_attn: drafter attention over the context ring + block, vs CPU
//   * dflash_attn_long: the same past kDFlashAttnKeys keys (chunked); == dflash_attn bitwise on one chunk
//   * dflash_attn_part + dflash_attn_merge (feat029-dflashcost): split over keys, GQA-shared K / V; the same
//     cases vs CPU (1e-4) and vs dflash_attn on the same inputs
//   * dflash_topk_part + dflash_topk_merge: == dflash_topk bit for bit (ids and values)
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "kt.h"
#include "whirl/dflash_kernel_plan.h"

namespace kt {

namespace {

// keys: larger value first, then smaller index; NaN as -inf
bool better(float va, int ia, float vb, int ib) {
    if (std::isnan(va)) va = -std::numeric_limits<float>::infinity();
    if (std::isnan(vb)) vb = -std::numeric_limits<float>::infinity();
    return va > vb || (va == vb && ia < ib);
}

Result passFail(const std::string& name, bool ok, std::size_t n, const std::string& note = "") {
    Result r;
    r.name = name;
    r.n = n;
    r.mismatches = ok ? 0 : 1;
    r.pass = ok;
    if (!ok) r.note = note;
    return r;
}

// CPU walk on a lattice (first maximum; p-min over the K scores; n-min)
std::vector<int> walkRef(const std::vector<float>& lat, const std::vector<int>& cand, int P, int K, float p_min, int n_min) {
    std::vector<int> out;
    int pred = 0;
    for (int p = 0; p < P; ++p) {
        const float* s = lat.data() + (static_cast<std::size_t>(p) * K + pred) * K;
        int jb = 0;
        for (int j = 1; j < K; ++j)
            if (s[j] > s[jb]) jb = j;
        if (p_min > 0.f) {
            float sum = 0.f;
            for (int j = 0; j < K; ++j) sum += std::exp(s[j] - s[jb]);
            if (1.f / sum < p_min) break;
        }
        out.push_back(cand[static_cast<std::size_t>(p) * K + jb]);
        pred = jb;
    }
    if (static_cast<int>(out.size()) < n_min) out.clear();
    return out;
}

}  // namespace

void testDflash(Ctx& c) {
    c.rep.family = "dflash";
    if (!c.k.dflash_conv || !c.k.dflash_topk || !c.k.dflash_lattice || !c.k.dflash_walk) {
        c.rep.skip("dflash", "all", "dflash kernels not in this code object");
        return;
    }
    // ---- dynamic convolution: T = 8 (draft block), C = 5120, group 16, K = 2
    {
        const int T = 8, C = 5120, group = 16, K = 2, G = C / group, ds = 2 * K * G;
        const std::vector<float> x = c.randn(static_cast<std::size_t>(T) * C), dyn = c.randn(static_cast<std::size_t>(T) * ds, 0.1f),
                                 base = c.randn(static_cast<std::size_t>(2) * K * C, 0.5f);
        Buf dx(x), dd(dyn), db(base), dy(static_cast<std::size_t>(T) * C * 4);
        for (int side = 0; side < 2; ++side) {
            hip::launch(c.k.dflash_conv, {static_cast<unsigned>((C + 255) / 256), static_cast<unsigned>(T), 1}, {256, 1, 1}, 0, c.s, dx.p(),
                        dd.p(), db.p(), dy.p(), T, C, K, group, side, ds);
            c.sync();
            std::vector<double> ref(static_cast<std::size_t>(T) * C), sc(ref.size());
            for (int t = 0; t < T; ++t)
                for (int ch = 0; ch < C; ++ch) {
                    double a = 0, s = 0;
                    for (int tap = 0; tap < K && tap <= t; ++tap) {
                        const float w = base[static_cast<std::size_t>(side * K + tap) * C + ch] +
                                        dyn[static_cast<std::size_t>(t) * ds + ch / group + G * (tap + K * side)];
                        const double term = static_cast<double>(w) * x[static_cast<std::size_t>(t - tap) * C + ch];
                        a += term;
                        s += std::fabs(term);
                    }
                    ref[static_cast<std::size_t>(t) * C + ch] = a;
                    sc[static_cast<std::size_t>(t) * C + ch] = s;
                }
            c.rep.add(cmpTol("dflash_conv side " + std::to_string(side) + " vs CPU (T 8, C 5120, group 16, K 2)",
                             dy.down<float>(ref.size()), ref, sc, 1e-6, 1e-7));
        }
    }
    // ---- deterministic top-16 over a vocabulary with ties and a NaN
    {
        const int n = c.quick ? 50000 : 248320, rows = 3, k = 16;
        std::vector<float> x = c.randn(static_cast<std::size_t>(rows) * n);
        // row 0: a 3-way tie at the top, row 1: ties straddling rank 16, row 2: NaN + equal values
        for (int i : {n - 1, 7, 1000}) x[static_cast<std::size_t>(i)] = 10.f;
        for (int i = 0; i < 20; ++i) x[static_cast<std::size_t>(n) + static_cast<std::size_t>(i) * 997] = 8.f;
        x[2ull * n + 5] = std::numeric_limits<float>::quiet_NaN();
        for (int i : {n - 2, 3, 40000 % n}) x[2ull * n + static_cast<std::size_t>(i)] = 9.f;
        Buf dx(x), di(static_cast<std::size_t>(rows) * k * 4), dv(static_cast<std::size_t>(rows) * k * 4);
        std::vector<int> ref_i;
        std::vector<float> ref_v;
        for (int r = 0; r < rows; ++r) {
            std::vector<int> idx(static_cast<std::size_t>(n));
            std::iota(idx.begin(), idx.end(), 0);
            const float* row = x.data() + static_cast<std::size_t>(r) * n;
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return better(row[a], a, row[b], b); });
            for (int j = 0; j < k; ++j) {
                ref_i.push_back(idx[static_cast<std::size_t>(j)]);
                ref_v.push_back(row[idx[static_cast<std::size_t>(j)]]);
            }
        }
        std::vector<int> got_i[2];
        for (int run = 0; run < 2; ++run) {
            di.zero();
            hip::launch(c.k.dflash_topk, {static_cast<unsigned>(rows), 1, 1}, {1024, 1, 1}, 0, c.s, dx.p(), n, n, k, di.p(), dv.p());
            c.sync();
            got_i[run] = di.down<int>(ref_i.size());
        }
        c.rep.add(cmpExact("dflash_topk ids (key order, ties -> smaller index, NaN skipped)", got_i[0], ref_i));
        c.rep.add(cmpExact("dflash_topk values", dv.down<float>(ref_v.size()), ref_v));
        c.rep.add(cmpExact("dflash_topk repeatable (two runs)", got_i[1], got_i[0], Kind::invariant));
        if (c.k.dflash_topk_part && c.k.dflash_topk_merge) {
            // two-stage top-k: the slices' top-k lists merged; must equal dflash_topk bit for bit
            const std::vector<float> old_v = dv.down<float>(ref_v.size());
            const unsigned S = whirl::dflash::topkSlices(static_cast<unsigned>(n));
            Buf pid(static_cast<std::size_t>(rows) * S * k * 4), pval(static_cast<std::size_t>(rows) * S * k * 4);
            di.zero();
            dv.zero();
            hip::launch(c.k.dflash_topk_part, {static_cast<unsigned>(rows), S, 1}, {256, 1, 1}, 0, c.s, dx.p(), n, n, k, static_cast<int>(S),
                        pid.p(), pval.p());
            hip::launch(c.k.dflash_topk_merge, {static_cast<unsigned>(rows), 1, 1}, {256, 1, 1}, 0, c.s, pid.p(), pval.p(), static_cast<int>(S),
                        k, di.p(), dv.p());
            c.sync();
            const std::vector<int> si = di.down<int>(ref_i.size());
            const std::vector<float> sv = dv.down<float>(ref_v.size());
            c.rep.add(cmpExact("dflash_topk_part + merge ids == CPU (slices of 8192, ties, NaN)", si, ref_i));
            c.rep.add(cmpExact("dflash_topk_part + merge ids == dflash_topk bitwise", si, got_i[0], Kind::invariant));
            c.rep.add(cmpExact("dflash_topk_part + merge values == dflash_topk bitwise", sv, old_v, Kind::invariant));
        }
    }
    // ---- drafter attention: context ring (with empty rows, rows past the anchor and rows outside
    // the window) plus the block's own rows, GQA 32 / 8, head_dim 128. dflash_attn_long (C-52): on the
    // same ring it is one chunk and must equal dflash_attn bit for bit (its merge is l * e^0 + 0, o * e^0
    // + 0); past kDFlashAttnKeys (Ornith's 4096-row window, 16-row blocks) it is checked against the CPU,
    // also with a first chunk that has no visible key (skipped).
    auto attnCase = [&](const char* what, const auto& fn, int B, int W, int window, int qpos, auto cposOf, std::vector<float>* got_out) {
        const int nh = 32, nkv = 8, hd = 128;
        const std::vector<float> q = c.randn(static_cast<std::size_t>(B) * nh * hd), kb = c.randn(static_cast<std::size_t>(B) * nkv * hd),
                                 vb = c.randn(static_cast<std::size_t>(B) * nkv * hd), kc = c.randn(static_cast<std::size_t>(W) * nkv * hd),
                                 vc = c.randn(static_cast<std::size_t>(W) * nkv * hd);
        std::vector<int> cpos(static_cast<std::size_t>(W));
        for (int i = 0; i < W; ++i) cpos[static_cast<std::size_t>(i)] = cposOf(i);
        Buf dq(q), dkb(kb), dvb(vb), dkc(kc), dvc(vc), dcp(cpos), dout(q.size() * 4);
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        if (fn == nullptr) {
            // nullptr: the split pair dflash_attn_part + dflash_attn_merge (feat029-dflashcost)
            const unsigned S = whirl::dflash::attnSplits(static_cast<unsigned>(W + B));
            Buf part(whirl::dflash::attnPartFloats(static_cast<unsigned>(W + B), static_cast<unsigned>(B), static_cast<unsigned>(nh),
                                                   static_cast<unsigned>(hd)) *
                     4);
            hip::launch(c.k.dflash_attn_part, {static_cast<unsigned>(nkv), static_cast<unsigned>(B), S}, {whirl::dflash::kAttnThreads, 1, 1}, 0,
                        c.s, dq.p(), dkb.p(), dvb.p(), dkc.p(), dvc.p(), dcp.p(), W, B, nh, nkv, hd, qpos, window, scale, part.p());
            hip::launch(c.k.dflash_attn_merge, {static_cast<unsigned>(nh), static_cast<unsigned>(B), 1}, {static_cast<unsigned>(hd), 1, 1}, 0,
                        c.s, part.p(), static_cast<int>(S), B, nh, hd, dout.p());
        } else {
            hip::launch(fn, {static_cast<unsigned>(nh), static_cast<unsigned>(B), 1}, {static_cast<unsigned>(hd), 1, 1}, 0, c.s, dq.p(),
                        dkb.p(), dvb.p(), dkc.p(), dvc.p(), dcp.p(), W, B, nh, nkv, hd, qpos, window, scale, dout.p());
        }
        c.sync();
        std::vector<double> ref(q.size()), sc(q.size());
        for (int t = 0; t < B; ++t)
            for (int h = 0; h < nh; ++h) {
                const int g = h / (nh / nkv);
                const float* qv = q.data() + (static_cast<std::size_t>(t) * nh + h) * hd;
                std::vector<double> s(static_cast<std::size_t>(W + B), -INFINITY);
                double m = -INFINITY;
                for (int k = 0; k < W + B; ++k) {
                    const float* kr;
                    if (k < W) {
                        const int p = cpos[static_cast<std::size_t>(k)];
                        if (p < 0 || p >= qpos || (window != 0 && qpos + t - p >= window)) continue;
                        kr = kc.data() + (static_cast<std::size_t>(k) * nkv + g) * hd;
                    } else {
                        kr = kb.data() + (static_cast<std::size_t>(k - W) * nkv + g) * hd;
                    }
                    double d = 0;
                    for (int i = 0; i < hd; ++i) d += static_cast<double>(qv[i]) * kr[i];
                    s[static_cast<std::size_t>(k)] = d * scale;
                    m = std::max(m, s[static_cast<std::size_t>(k)]);
                }
                double l = 0;
                for (double& x : s) {
                    x = std::isinf(x) ? 0.0 : std::exp(x - m);
                    l += x;
                }
                for (int d = 0; d < hd; ++d) {
                    double o = 0, a = 0;
                    for (int k = 0; k < W + B; ++k) {
                        const double p = s[static_cast<std::size_t>(k)];
                        if (p == 0) continue;
                        const float v = k < W ? vc[(static_cast<std::size_t>(k) * nkv + g) * hd + d] : vb[(static_cast<std::size_t>(k - W) * nkv + g) * hd + d];
                        o += p * v;
                        a += p * std::fabs(v);
                    }
                    ref[(static_cast<std::size_t>(t) * nh + h) * hd + d] = o / l;
                    sc[(static_cast<std::size_t>(t) * nh + h) * hd + d] = a / l;
                }
            }
        const std::vector<float> got = dout.down<float>(ref.size());
        c.rep.add(cmpTol(what, got, ref, sc, 1e-4, 1e-6));
        if (got_out) *got_out = got;
    };
    // the original case: ring 256 with empty / future / out-of-window rows, block 8
    auto cpos256 = [](int i) { return i % 17 == 3 ? -1 : (i < 250 ? 50 + i : 300 + i - 250); };
    if (c.k.dflash_attn) {
        std::vector<float> a, b;
        c.rng.seed(77);
        attnCase("dflash_attn vs CPU (ring 256 with empty / future / out-of-window rows, block 8, GQA 32/8, hd 128)", c.k.dflash_attn, 8, 256,
                 200, 300, cpos256, &a);
        if (c.k.dflash_attn_long) {
            c.rng.seed(77);  // the same inputs
            attnCase("dflash_attn_long vs CPU (same ring: one chunk)", c.k.dflash_attn_long, 8, 256, 200, 300, cpos256, &b);
            c.rep.add(cmpExact("dflash_attn_long == dflash_attn bitwise when the keys fit one chunk", b, a, Kind::invariant));
        }
    }
    if (c.k.dflash_attn_long) {
        // Ornith's drafter: 4096-row window, 16-row blocks: 4112 keys > kDFlashAttnKeys (two chunks)
        attnCase("dflash_attn_long vs CPU (ring 4096, window 4096, block 16: 2 chunks, empty / future / out-of-window rows)",
                 c.k.dflash_attn_long, 16, 4096, 4096, 5000, [](int i) { return i % 29 == 5 ? -1 : (i < 4090 ? 950 + i : 5000 + i); }, nullptr);
        // a first chunk without any visible key (all ring rows of chunk 0 empty), so only chunk 1 and the block count
        attnCase("dflash_attn_long vs CPU (ring 4096, first chunk all empty: skipped)", c.k.dflash_attn_long, 16, 4096, 0, 9000,
                 [](int i) { return i < whirl::kernels::kDFlashAttnKeys ? -1 : 4000 + i; }, nullptr);
    }
    if (c.k.dflash_attn_part && c.k.dflash_attn_merge) {
        const hip::Function split = nullptr;
        std::vector<float> a, b;
        c.rng.seed(77);  // the inputs of the dflash_attn case
        attnCase("dflash_attn_part + merge vs CPU (ring 256, empty / future / out-of-window rows, block 8, GQA 32/8: 2 splits)", split, 8,
                 256, 200, 300, cpos256, &b);
        if (c.k.dflash_attn) {
            c.rng.seed(77);
            attnCase("dflash_attn (reference for the split pair)", c.k.dflash_attn, 8, 256, 200, 300, cpos256, &a);
            std::vector<double> ad(a.begin(), a.end()), one(a.size(), 1.0);
            c.rep.add(cmpTol("dflash_attn_part + merge vs dflash_attn (same inputs, 1e-4 abs: f32 summation order only)", b, ad, one, 1e-4, 1e-6));
        }
        // the R9700 / 8060S drafter shape: full 2048-row ring + block 8 (9 splits, last one only the block)
        attnCase("dflash_attn_part + merge vs CPU (ring 2048 full, window 2048, block 8: 9 splits)", split, 8, 2048, 2048, 5000,
                 [](int i) { return 5000 - 2048 + i; }, nullptr);
        // Ornith's drafter: ring 4096, window 4096, block 16 (17 splits) with empty / future / out-of-window rows
        attnCase("dflash_attn_part + merge vs CPU (ring 4096, window 4096, block 16: 17 splits, mixed rows)", split, 16, 4096, 4096, 5000,
                 [](int i) { return i % 29 == 5 ? -1 : (i < 4090 ? 950 + i : 5000 + i); }, nullptr);
        // whole splits without a visible key (the first 2304 ring rows empty): skipped in the merge
        attnCase("dflash_attn_part + merge vs CPU (ring 4096, first 9 splits all empty)", split, 16, 4096, 0, 9000,
                 [](int i) { return i < whirl::kernels::kDFlashAttnKeys ? -1 : 4000 + i; }, nullptr);
        // a cold drafter context: every ring row empty, only the block's own keys
        attnCase("dflash_attn_part + merge vs CPU (ring 2048 all empty: block keys only)", split, 8, 2048, 2048, 10,
                 [](int) { return -1; }, nullptr);
    }
    // ---- selector lattice (f32 and Q8_0 codebooks) and the walk
    {
        const int P = 7, K = 16, R = 256, V = 2000, anchor = 1234;
        std::vector<int> cand(static_cast<std::size_t>(P) * K);
        for (std::size_t i = 0; i < cand.size(); ++i) cand[i] = static_cast<int>((i * 7919 + 13) % V);
        const std::vector<float> unary = c.randn(cand.size()), gate = c.randn(static_cast<std::size_t>(P) * R, 0.5f);
        const std::vector<float> prev = c.randn(static_cast<std::size_t>(V) * R, 0.5f), next = c.randn(static_cast<std::size_t>(V) * R, 0.5f);
        // Q8_0 copies: d = amax / 127 rounded to f16, q = rint(x / d); values as the kernel decodes them
        auto q8 = [&](const std::vector<float>& f, std::vector<std::uint8_t>& blob, std::vector<float>& deq) {
            blob.assign(static_cast<std::size_t>(V) * (R / 32) * 34, 0);
            deq.assign(f.size(), 0.f);
            for (std::size_t b = 0; b < f.size() / 32; ++b) {
                float amax = 0;
                for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(f[b * 32 + i]));
                const std::uint16_t dh = f2h(amax / 127.f);
                const float d = h2f(dh);
                std::uint8_t* o = blob.data() + b * 34;
                std::memcpy(o, &dh, 2);
                for (int i = 0; i < 32; ++i) {
                    const int q = d > 0 ? std::clamp(static_cast<int>(std::lrint(f[b * 32 + i] / d)), -127, 127) : 0;
                    o[2 + i] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
                    deq[b * 32 + i] = d * static_cast<float>(q);
                }
            }
        };
        std::vector<std::uint8_t> pq, nq;
        std::vector<float> pdeq, ndeq;
        q8(prev, pq, pdeq);
        q8(next, nq, ndeq);
        Buf dc(cand), du(unary), dg(gate), dlat(static_cast<std::size_t>(P) * K * K * 4);
        std::vector<float> lat_f32;
        for (int mode = 0; mode < 2; ++mode) {
            const bool is_q8 = mode == 1;
            Buf dp = is_q8 ? Buf(pq) : Buf(prev);
            Buf dn = is_q8 ? Buf(nq) : Buf(next);
            const std::uint64_t rb = is_q8 ? static_cast<std::uint64_t>(R / 32) * 34 : static_cast<std::uint64_t>(R) * 4;
            const std::vector<float>& pv = is_q8 ? pdeq : prev;
            const std::vector<float>& nv = is_q8 ? ndeq : next;
            hip::launch(c.k.dflash_lattice, {static_cast<unsigned>(P), static_cast<unsigned>(K), 1}, {static_cast<unsigned>(32 * K), 1, 1}, 0, c.s,
                        dc.p(), du.p(), dg.p(), dp.p(), dn.p(), rb, is_q8 ? 1 : 0, R, K, anchor, dlat.p());
            c.sync();
            std::vector<double> ref(static_cast<std::size_t>(P) * K * K), sc(ref.size());
            for (int p = 0; p < P; ++p)
                for (int i = 0; i < K; ++i) {
                    const int pred = p == 0 ? anchor : cand[static_cast<std::size_t>(p - 1) * K + i];
                    for (int j = 0; j < K; ++j) {
                        const int nx = cand[static_cast<std::size_t>(p) * K + j];
                        double s = 0, a = 0;
                        for (int r = 0; r < R; ++r) {
                            const double t = static_cast<double>(pv[static_cast<std::size_t>(pred) * R + r]) * gate[static_cast<std::size_t>(p) * R + r] *
                                             nv[static_cast<std::size_t>(nx) * R + r];
                            s += t;
                            a += std::fabs(t);
                        }
                        const double u = unary[static_cast<std::size_t>(p) * K + j];
                        ref[(static_cast<std::size_t>(p) * K + i) * K + j] = s + u;
                        sc[(static_cast<std::size_t>(p) * K + i) * K + j] = a + std::fabs(u);
                    }
                }
            const std::vector<float> got = dlat.down<float>(ref.size());
            c.rep.add(cmpTol(std::string("dflash_lattice vs CPU (") + (is_q8 ? "Q8_0" : "f32") + " codebooks, anchor row at p = 0)", got, ref, sc,
                             2e-6, 1e-6));
            if (!is_q8) lat_f32 = got;
        }
        // walk: GPU == CPU on the same lattice, for a few p-min / n-min settings
        Buf dl(lat_f32), dout(static_cast<std::size_t>(1 + P) * 4);
        for (const auto& [p_min, n_min] : {std::pair<float, int>{0.f, 0}, {0.3f, 0}, {0.999f, 0}, {0.f, P + 1}, {0.3f, 3}}) {
            dout.zero();
            hip::launch(c.k.dflash_walk, {1, 1, 1}, {32, 1, 1}, 0, c.s, dl.p(), dc.p(), P, K, p_min, n_min, dout.p());
            c.sync();
            const std::vector<int> o = dout.down<int>(static_cast<std::size_t>(1 + P));
            const std::vector<int> want = walkRef(lat_f32, cand, P, K, p_min, n_min);
            bool ok = o[0] == static_cast<int>(want.size());
            for (std::size_t q = 0; ok && q < want.size(); ++q) ok = o[1 + q] == want[q];
            c.rep.add(passFail("dflash_walk == CPU walk (p_min " + std::to_string(p_min) + ", n_min " + std::to_string(n_min) + ", " +
                                   std::to_string(want.size()) + " drafts)",
                               ok, want.size() + 1, "gpu n " + std::to_string(o[0])));
        }
    }
}

}  // namespace kt
