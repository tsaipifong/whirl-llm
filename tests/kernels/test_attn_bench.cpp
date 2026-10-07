// whirl-kernel-test family "attnbench" (only with --only attnbench): decode attention timing
// at long contexts (q4 / f16 KV, Ornith 16/2 and Qwen3.8-27B 24/4 head shapes).
// SPDX-License-Identifier: Apache-2.0
// Times attn_wsplit1 (f16), attn_wsplit1_q4 and attn_dq4 (gfx1151 q4 decode kernel) for one
// decode query at the end of the context, prints us / effective KV GB/s, and the largest
// difference of attn_dq4 vs attn_wsplit1_q4 after attn_combine (both lossy vs the CPU in
// different ways; the CPU tolerance tests are in test_attn.cpp).

#include <algorithm>
#include <cmath>
#include <numeric>

#include "kt.h"

namespace kt {

void benchAttn(Ctx& c) {
    c.rep.family = "attnbench";
    struct Shape {
        int heads, nkv;
    };
    const int hd = 256, qstride = 2 * hd;
    const float scale = 1.0f / 16.0f;
    const char* env_l = std::getenv("WHIRL_KT_BENCH_L");
    std::vector<int> lens = {65536, 131072};
    if (env_l) lens = {std::atoi(env_l)};
    const char* env_s = std::getenv("WHIRL_KT_BENCH_SPLITS");
    for (const Shape sh : {Shape{16, 2}, Shape{24, 4}}) {
        for (const int L : lens) {
            const int pages = (L + wk::kKvPage - 1) / wk::kKvPage;
            const std::size_t rows = static_cast<std::size_t>(pages) * wk::kKvPage, nel = rows * sh.nkv * hd;
            std::vector<int> pt(static_cast<std::size_t>(pages));
            std::iota(pt.begin(), pt.end(), 0);
            std::shuffle(pt.begin(), pt.end(), c.rng);
            Buf dpt(pt);
            // q4: random nibbles, scales 0.02..0.1; f16: N(0, 0.5)
            std::vector<std::uint8_t> k4(nel / 2), v4(nel / 2);
            for (auto& x : k4) x = static_cast<std::uint8_t>(c.rng());
            for (auto& x : v4) x = static_cast<std::uint8_t>(c.rng());
            std::vector<std::uint16_t> ks(nel / 32), vs(nel / 32);
            {
                const auto a = c.randu(nel / 32, 0.02f, 0.1f), b = c.randu(nel / 32, 0.02f, 0.1f);
                for (std::size_t i = 0; i < ks.size(); ++i) {
                    ks[i] = f2h(a[i]);
                    vs[i] = f2h(b[i]);
                }
            }
            Buf dk4(k4), dv4(v4), dks(ks), dvs(vs);
            Buf dk16(nel * 2), dv16(nel * 2);
            dk16.fill(0x30);  // 0x3030 = 0.1318 (f16); timing only
            dv16.fill(0x30);
            const std::vector<float> qp = c.randn(static_cast<std::size_t>(sh.heads) * qstride);
            Buf dq(qp), dpos(std::vector<int>{L - 1});
            int ns = std::min(256, static_cast<int>(cdiv(L, wk::kFdChunk)));
            if (env_s) ns = std::atoi(env_s);
            Buf ml(static_cast<std::size_t>(ns) * sh.heads * 2 * 4), acc(static_cast<std::size_t>(ns) * sh.heads * hd * 4);
            const std::size_t on = static_cast<std::size_t>(sh.heads) * hd;
            wk::AwGroups g1{};
            g1.first[0] = 0;
            g1.count[0] = 1;
            std::vector<float> out_ref;
            for (const char* name : {"attn_wsplit1", "attn_wsplit1_q4", "attn_dq4"}) {
                const auto f = c.fnOpt(name);
                if (!f) {
                    std::printf("   %s: not in this code object\n", name);
                    continue;
                }
                const bool q4 = std::string(name) != "attn_wsplit1";
                const unsigned threads = std::string(name) == "attn_dq4" ? 256 : 128;
                wk::KvArgs a{};
                a.k = q4 ? dk4.p() : dk16.p();
                a.v = q4 ? dv4.p() : dv16.p();
                a.ks = q4 ? dks.p() : 0;
                a.vs = q4 ? dvs.p() : 0;
                a.ptab = dpt.p();
                auto launch = [&]() {
                    hip::launch(f, {static_cast<unsigned>(sh.nkv), static_cast<unsigned>(ns), 1}, {threads, 1, 1}, 0, c.s, dq.p(), a, ml.p(),
                                acc.p(), sh.heads, sh.nkv, qstride, dpos.p(), scale, DevPtr{0}, g1);
                };
                for (int i = 0; i < 3; ++i) launch();
                c.sync();
                hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                const int iters = 20;
                hip::eventRecord(e0, c.s);
                for (int i = 0; i < iters; ++i) launch();
                hip::eventRecord(e1, c.s);
                hip::eventSync(e1);
                const double us = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
                hip::eventDestroy(e0);
                hip::eventDestroy(e1);
                const double bytes = static_cast<double>(L) * sh.nkv * (q4 ? (hd / 2 + hd / 32 * 2) * 2 : hd * 2 * 2);
                std::printf("   %-16s heads %d/%d L %6d splits %3d: %8.1f us  %6.1f GB/s\n", name, sh.heads, sh.nkv, L, ns, us, bytes / us / 1e3);
                if (q4) {
                    Buf out(on * 4);
                    launch();
                    hip::launch(c.k.attn_combine, {static_cast<unsigned>(sh.heads), 1, 1}, {256, 1, 1}, 0, c.s, ml.p(), acc.p(), dq.p(), out.p(),
                                sh.heads, hd, qstride, ns);
                    c.sync();
                    const auto o = out.down<float>(on);
                    if (out_ref.empty()) {
                        out_ref = o;
                    } else {
                        double md = 0, mr = 0;
                        for (std::size_t i = 0; i < on; ++i) {
                            md = std::max(md, static_cast<double>(std::fabs(o[i] - out_ref[i])));
                            mr = std::max(mr, static_cast<double>(std::fabs(out_ref[i])));
                        }
                        std::printf("   %-16s max |diff| vs attn_wsplit1_q4 %.3g (max |out| %.3g)\n", name, md, mr);
                    }
                }
            }
        }
    }
}

}  // namespace kt
