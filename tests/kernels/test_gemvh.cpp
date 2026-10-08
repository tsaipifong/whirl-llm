// whirl-kernel-test families "gemvh" (bitwise, part of the default run) and "gemvhbench"
// (timing only, --only gemvhbench): the precise-decode f16-activation matmul gemvh / gemvh2.
// SPDX-License-Identifier: Apache-2.0
// gemvh: for every exported type, ncols (even stage count, odd stage count, 5120), nrows (not a
// multiple of 16 included), n_tok in {1,2,5,16,17,32} and accumulate 0/1, gemvh (n <= 16) and
// gemvh2 (n <= 32) must equal the existing gemms_c* kernels and gemm_c0 bit for bit (accumulate
// starts from the same random prefill).
// gemvhbench: us per call, GB/s of weight bytes and % of the memory-bandwidth peak for gemvh vs
// gemms (the path precise decode used before), real Qwen3.8-27B shapes.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kt.h"

namespace kt {

namespace {

Buf makeX16(Ctx& c, int nmax, int ncols) {
    const std::vector<float> x = c.randn(static_cast<std::size_t>(nmax) * ncols);
    Buf dx(x);
    Buf x16(static_cast<std::size_t>(nmax) * ncols * 2);
    hip::launch(c.k.f32_to_f16, {cdiv(static_cast<std::uint64_t>(nmax) * ncols / 4, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), x16.p(),
                nmax * ncols);
    c.sync();
    return x16;
}

}  // namespace

void testGemvh(Ctx& c) {
    c.rep.family = "gemvh";
    const std::vector<int> ncolsL = c.quick ? std::vector<int>{512, 768} : std::vector<int>{256, 512, 768, 1280, 5120};
    const std::vector<int> nrowsL = c.quick ? std::vector<int>{7, 37, 100} : std::vector<int>{1, 7, 16, 17, 37, 100, 1003, 4099};
    const int ns[] = {1, 2, 5, 16, 17, 32};
    for (QType t : wk::kAllTypes) {
        const int ti = static_cast<int>(t);
        if (t == QType::f32 || (!c.k.gemvh[ti] && !c.k.gemvh2[ti])) continue;
        std::size_t checked = 0, bad = 0;
        std::string first_bad;
        for (int ncols : ncolsL) {
            if (ncols % 256 != 0) continue;
            for (int nrows : nrowsL) {
                const HostMat m = randomMat(c, t, nrows, ncols);
                Buf w(m.data);
                Buf x16 = makeX16(c, 32, ncols);
                for (int nt : ns) {
                    for (int acc = 0; acc < 2; ++acc) {
                        const std::vector<float> init = c.randn(static_cast<std::size_t>(nt) * nrows, 3.0f);
                        auto run = [&](hip::Function f, int bm, int bn, int nth, bool tok_x) {
                            Buf y(init);
                            const hip::Dim3 grid = tok_x ? hip::Dim3{cdiv(nt, bn), cdiv(nrows, bm), 1} : hip::Dim3{cdiv(nrows, bm), cdiv(nt, bn), 1};
                            hip::launch(f, grid, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, w.p(), m.row_bytes, x16.p(), y.p(), ncols, nrows, nt,
                                        acc);
                            c.sync();
                            return y.down<float>(static_cast<std::size_t>(nt) * nrows);
                        };
                        const auto& c0 = wk::kGemmCfgs[0];
                        const std::vector<float> ref = run(c.k.gemmc[0][ti], c0.bm, c0.bn, c0.nth, true);
                        auto chk = [&](const std::vector<float>& y, const std::string& label) {
                            ++checked;
                            if (std::memcmp(y.data(), ref.data(), ref.size() * 4) != 0) {
                                ++bad;
                                if (first_bad.empty())
                                    first_bad = label + " ncols " + std::to_string(ncols) + " nrows " + std::to_string(nrows) + " n " +
                                                std::to_string(nt) + " acc " + std::to_string(acc);
                            }
                        };
                        for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                            const auto& cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                            if (c.k.gemms[si][ti]) chk(run(c.k.gemms[si][ti], cf.bm, cf.bn, cf.nth, true), "gemms_c" + std::to_string(si));
                        }
                        if (nt <= 16 && c.k.gemvh[ti]) chk(run(c.k.gemvh[ti], 16, 1 << 20, 32, false), "gemvh");
                        if (c.k.gemvh2[ti]) chk(run(c.k.gemvh2[ti], 16, 1 << 20, 32, false), "gemvh2");
                    }
                }
            }
        }
        Result r;
        r.name = std::string("gemvh/gemvh2 == gemm_c0 and gemms_c*, ncols x nrows x n x acc sweep (") + std::to_string(checked) + " runs) " +
                 wk::typeSuffix(t);
        r.kind = Kind::invariant;
        r.n = checked;
        r.mismatches = bad;
        r.pass = bad == 0 && checked > 0;
        if (!first_bad.empty()) r.note = "first " + first_bad;
        c.rep.add(r);
    }
}

// WHIRL_KT_PEAK_GBS (default 238): the bandwidth the % column refers to.
void benchGemvh(Ctx& c) {
    double peak = 238.0;
    if (const char* e = std::getenv("WHIRL_KT_PEAK_GBS")) peak = std::atof(e);
    struct Shape {
        QType t;
        int nrows, ncols;
        const char* what;
    };
    const Shape shapes[] = {
        {QType::q4_k, 5120, 5120, "attn proj 5120x5120"}, {QType::q4_k, 17408, 5120, "ffn gate/up 17408x5120"},
        {QType::q6_k, 5120, 17408, "ffn down 5120x17408"}, {QType::q6_k, 5120, 5120, "5120x5120"},
        {QType::q8_0, 5120, 5120, "5120x5120"},            {QType::q5_k, 5120, 5120, "5120x5120"},
        {QType::q6_k, 248320, 5120, "lm head 248320x5120"},
    };
    std::printf("   (peak reference %.0f GB/s)\n", peak);
    for (const Shape& sh : shapes) {
        const int ti = static_cast<int>(sh.t);
        if (!c.k.gemvh[ti]) continue;
        const HostMat m = randomMat(c, sh.t, sh.nrows, sh.ncols);
        // enough copies that the weights never stay in the 32 MB MALL between calls
        const int copies = std::max<int>(1, std::min<int>(16, static_cast<int>((192u << 20) / m.data.size())));
        std::vector<Buf> ws;
        for (int i = 0; i < copies; ++i) ws.emplace_back(m.data);
        Buf x16 = makeX16(c, 32, sh.ncols);
        for (int nt : {1, 16, 32}) {
            Buf y(static_cast<std::size_t>(nt) * sh.nrows * 4);
            struct V {
                std::string name;
                hip::Function f;
                int bm, bn, nth;
                bool tok_x;
            };
            std::vector<V> vs;
            if (nt <= 16 && c.k.gemvh[ti]) vs.push_back({"gemvh", c.k.gemvh[ti], 16, 1 << 20, 32, false});
            if (c.k.gemvh2[ti]) vs.push_back({"gemvh2", c.k.gemvh2[ti], 16, 1 << 20, 32, false});
            for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                const auto& cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                if (c.k.gemms[si][ti]) vs.push_back({"gemms_c" + std::to_string(si), c.k.gemms[si][ti], cf.bm, cf.bn, cf.nth, true});
            }
            for (const V& v : vs) {
                auto launch = [&](int i) {
                    const hip::Dim3 grid = v.tok_x ? hip::Dim3{cdiv(nt, v.bn), cdiv(sh.nrows, v.bm), 1}
                                                   : hip::Dim3{cdiv(sh.nrows, v.bm), cdiv(nt, v.bn), 1};
                    hip::launch(v.f, grid, {static_cast<unsigned>(v.nth), 1, 1}, 0, c.s, ws[static_cast<std::size_t>(i % copies)].p(), m.row_bytes,
                                x16.p(), y.p(), sh.ncols, sh.nrows, nt, 0);
                };
                for (int i = 0; i < copies; ++i) launch(i);
                c.sync();
                const int iters = std::max(copies * 2, 20);
                hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                hip::eventRecord(e0, c.s);
                for (int i = 0; i < iters; ++i) launch(i);
                hip::eventRecord(e1, c.s);
                hip::eventSync(e1);
                const double us = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
                hip::eventDestroy(e0);
                hip::eventDestroy(e1);
                const double gbs = static_cast<double>(m.data.size()) / (us * 1e3);
                std::printf("   %-6s %-24s n %2d %-9s %9.1f us %7.1f GB/s %5.1f%% of %.0f\n", wk::typeSuffix(sh.t), sh.what, nt, v.name.c_str(),
                            us, gbs, 100.0 * gbs / peak, peak);
                std::fflush(stdout);
            }
        }
    }
}

}  // namespace kt
