// whirl-kernel-test family "attnbench" (only with --only attnbench): decode attention timing
// at long contexts (q4 / f16 KV, Ornith 16/2 and Qwen3.8-27B 24/4 head shapes).
// SPDX-License-Identifier: Apache-2.0
// Times attn_wsplit1 (f16), attn_wsplit1_q4 and attn_dq4 (gfx1151 q4 decode kernel) for one
// decode query at the end of the context, prints us / effective KV GB/s, and the largest
// difference of attn_dq4 vs attn_wsplit1_q4 after attn_combine (both lossy vs the CPU in
// different ways; the CPU tolerance tests are in test_attn.cpp).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>

#include "kt.h"

namespace kt {

// WHIRL_KT_BENCH_PREFILL=1: prefill attention timing instead (Q8P): n = 4096 queries at the end of
// L keys (WHIRL_KT_BENCH_L, default 32768 and 131072), 24 / 4 heads, attn_kg6 for f16 / q8v / q8 KV
// (plus WHIRL_KT_BENCH_KERNELS, comma-separated "name:fmt" with fmt f16|q8|q8v, NP 6 grid).
static void benchPrefill(Ctx& c) {
    const int hd = 256, qstride = 2 * hd, heads = 24, nkv = 4, n = 4096;
    const float scale = 1.0f / 16.0f;
    std::vector<int> lens = {32768, 131072};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_L")) lens = {std::atoi(e)};
    struct K {
        std::string name, fmt;
        int np = 6, qt = 1;
    };
    std::vector<K> ks = {{"attn_kg6", "f16"}, {"attn_kg6_q8v", "q8v"}, {"attn_kg6_q8", "q8"}, {"attn_kg6", "dq"}};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_KERNELS")) {
        std::string s = e;
        std::size_t a = 0;
        while (a < s.size()) {
            std::size_t b = s.find(',', a);
            if (b == std::string::npos) b = s.size();
            const std::string it = s.substr(a, b - a);
            // name:fmt[:np:qt]
            std::vector<std::string> f;
            for (std::size_t x = 0;;) {
                const std::size_t y = it.find(':', x);
                f.push_back(it.substr(x, y == std::string::npos ? std::string::npos : y - x));
                if (y == std::string::npos) break;
                x = y + 1;
            }
            if (f.size() >= 2) ks.push_back({f[0], f[1], f.size() >= 4 ? std::atoi(f[2].c_str()) : 6, f.size() >= 4 ? std::atoi(f[3].c_str()) : 1});
            a = b + 1;
        }
    }
    // WHIRL_KT_BENCH_HSACO: extra kernels (WHIRL_KT_BENCH_KERNELS) from this code object (kernel development)
    std::vector<char> image;
    hip::Module xmod;
    bool have_x = false;
    if (const char* e = std::getenv("WHIRL_KT_BENCH_HSACO")) {
        std::FILE* fp = std::fopen(e, "rb");
        if (fp) {
            std::fseek(fp, 0, SEEK_END);
            image.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(image.data(), 1, image.size(), fp) != image.size()) image.clear();
            std::fclose(fp);
        }
        if (!image.empty()) {
            xmod = hip::Module::loadData(image.data(), c.mod.arch());
            have_x = true;
        }
    }
    auto fnX = [&](const std::string& nm) { return have_x && nm.find("attn_kg") != 0 ? xmod.getFunctionOpt(nm.c_str()) : c.fnOpt(nm); };
    for (const int L : lens) {
        const int pages = (L + wk::kKvPage - 1) / wk::kKvPage;
        const std::size_t rows = static_cast<std::size_t>(pages) * wk::kKvPage, nel = rows * nkv * hd;
        std::vector<int> pt(static_cast<std::size_t>(pages));
        std::iota(pt.begin(), pt.end(), 0);
        std::shuffle(pt.begin(), pt.end(), c.rng);
        Buf dpt(pt);
        std::vector<std::int8_t> k8(nel), v8(nel);
        for (auto& x : k8) x = static_cast<std::int8_t>(static_cast<int>(c.rng() % 255) - 127);
        for (auto& x : v8) x = static_cast<std::int8_t>(static_cast<int>(c.rng() % 255) - 127);
        std::vector<std::uint16_t> ksc(nel / 32), vsc(nel / 32);
        {
            const auto a = c.randu(nel / 32, 0.001f, 0.004f), b = c.randu(nel / 32, 0.002f, 0.01f);
            for (std::size_t i = 0; i < ksc.size(); ++i) {
                ksc[i] = f2h(a[i]);
                vsc[i] = f2h(b[i]);
            }
        }
        Buf dk8(k8), dv8(v8), dks(ksc), dvs(vsc);
        Buf dk16(nel * 2), dv16(nel * 2);
        dk16.fill(0x30);
        dv16.fill(0x30);
        const std::vector<float> qp = c.randn(static_cast<std::size_t>(n) * heads * qstride);
        Buf dq(qp), dpos(std::vector<int>{L - n});
        Buf out(static_cast<std::size_t>(n) * heads * hd * 4);
        double base_us = 0;
        std::vector<float> ref_q8;
        for (const K& kk : ks) {
            const std::string& name = kk.name;
            const std::string& fmt = kk.fmt;
            const auto f = fnX(name);
            if (!f) {
                std::printf("   %s: not in this code object\n", name.c_str());
                continue;
            }
            wk::KvArgs a{};
            a.k = fmt == "q8" ? dk8.p() : dk16.p();
            a.v = fmt == "f16" ? dv16.p() : dv8.p();
            a.ks = fmt == "q8" ? dks.p() : 0;
            a.vs = fmt == "f16" ? 0 : dvs.p();
            a.ptab = dpt.p();
            // fmt "dq" (Q8P): kv_dq_rows of the q8 cache into contiguous f16 rows, then the (f16) kernel on them
            const bool dqf = fmt == "dq";
            hip::Function fdq = dqf ? c.fnOpt("kv_dq_rows") : hip::Function{};
            if (dqf && !fdq) continue;
            // fmt "dqs" (FC-1c): kv_dq_rows_r of key ranges of WHIRL_KT_BENCH_SEG rows (default 139264, the
            // R9700 scratch at prefill batch 4096) + attn_kgs over each range (softmax state carried)
            const bool dqs = fmt == "dqs";
            hip::Function fdr = dqs ? c.fnOpt("kv_dq_rows_r") : hip::Function{};
            if (dqs && !fdr) continue;
            const int seg = std::getenv("WHIRL_KT_BENCH_SEG") ? std::atoi(std::getenv("WHIRL_KT_BENCH_SEG")) : 139264;
            Buf dsk(dqs ? static_cast<std::size_t>(std::min(seg, L)) * nkv * hd * 2 : 16), dsv(dqs ? static_cast<std::size_t>(std::min(seg, L)) * nkv * hd * 2 : 16);
            Buf dst(static_cast<std::size_t>(n) * heads * 2 * 4);
            std::vector<int> idt(static_cast<std::size_t>(pages) + 1);
            std::iota(idt.begin(), idt.end(), 0);
            Buf did(idt);
            wk::KvArgs a8 = a;
            if (dqf || dqs) {
                a8.k = dk8.p();
                a8.v = dv8.p();
                a8.ks = dks.p();
                a8.vs = dvs.p();
                a.k = dk16.p();
                a.v = dv16.p();
                a.ks = 0;
                a.vs = 0;
                a.ptab = did.p();
            }
            auto launch = [&]() {
                if (dqs) {
                    for (int r0 = 0; r0 < L; r0 += seg) {
                        const int r1 = std::min(L, r0 + seg), nr = r1 - r0;
                        hip::launch(fdr, {cdiv(static_cast<std::uint64_t>(nr) * nkv * hd, 2048), 1, 1}, {256, 1, 1}, 0, c.s, a8, dsk.p(), dsv.p(), r0, nr,
                                    nkv * hd);
                        wk::KvArgs g = a;
                        g.k = dsk.p() - static_cast<DevPtr>(r0) * nkv * hd * 2;
                        g.v = dsv.p() - static_cast<DevPtr>(r0) * nkv * hd * 2;
                        hip::launch(f, {static_cast<unsigned>(cdiv(n, 16)), static_cast<unsigned>(heads / kk.np), 1}, {static_cast<unsigned>(64 * kk.np), 1, 1},
                                    0, c.s, dq.p(), g, out.p(), heads, nkv, qstride, dpos.p(), n, scale, 0, r0, r1, dst.p());
                    }
                    return;
                }
                if (dqf)
                    hip::launch(fdq, {cdiv(static_cast<std::uint64_t>(L) * nkv * hd, 2048), 1, 1}, {256, 1, 1}, 0, c.s, a8, dk16.p(), dv16.p(), L,
                                nkv * hd);
                hip::launch(f, {static_cast<unsigned>(cdiv(n, 16 * kk.qt)), static_cast<unsigned>(heads / kk.np), 1},
                            {static_cast<unsigned>(64 * kk.np * kk.qt), 1, 1}, 0, c.s, dq.p(), a,
                            out.p(), heads, nkv, qstride, dpos.p(), n, scale, 0);
            };
            launch();
            c.sync();
            hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
            const int iters = 5;
            hip::eventRecord(e0, c.s);
            for (int i = 0; i < iters; ++i) launch();
            hip::eventRecord(e1, c.s);
            hip::eventSync(e1);
            const double us = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
            hip::eventDestroy(e0);
            hip::eventDestroy(e1);
            if (base_us == 0) base_us = us;
            std::printf("   prefill %-22s (%3s) n %d L %6d: %9.1f us  %+6.1f%% vs first\n", name.c_str(), fmt.c_str(), n, L, us,
                        100.0 * (us / base_us - 1.0));
            std::fflush(stdout);
            if (fmt == "q8" || dqf || dqs) {  // bitwise vs the first q8 kernel (attn_kg6_q8 or dq)
                launch();
                c.sync();
                const auto o = out.down<float>(static_cast<std::size_t>(n) * heads * hd);
                if (ref_q8.empty()) {
                    ref_q8 = o;
                } else {
                    std::size_t bad = 0;
                    for (std::size_t i = 0; i < o.size(); ++i) bad += std::memcmp(&o[i], &ref_q8[i], 4) != 0;
                    std::printf("   prefill %-22s bitwise mismatches vs first q8: %zu of %zu\n", name.c_str(), bad, o.size());
                }
            }
        }
    }
}

// gfx1151 (KG-3; no attn_kg): attn_prefill_wmma on f16 / q8 / q4 KV vs the dequantize-then-f16 paths:
// "dq" / "dq4" (kv_dq_rows_r[_q4] of all keys + attn_prefill_wmma) and "dqs" / "dqs4" (key ranges of
// WHIRL_KT_BENCH_SEG rows, default 32768, + attn_prefill_wmma_s); n = 4096 queries at the end of L
// keys (WHIRL_KT_BENCH_L, default 65536 and 131072), 24 / 4 heads, head-range launches as the host
// (~4096 x 128k query-key pairs per launch). The dq paths are checked bitwise against q8 / q4.
static void benchPrefill1151(Ctx& c) {
    const int hd = 256, qstride = 2 * hd, heads = 24, nkv = 4, n = 4096, rel = nkv * hd;
    const float scale = 1.0f / 16.0f;
    std::vector<int> lens = {65536, 131072};
    if (const char* e = std::getenv("WHIRL_KT_BENCH_L")) lens = {std::atoi(e)};
    const int seg = std::getenv("WHIRL_KT_BENCH_SEG") ? std::atoi(std::getenv("WHIRL_KT_BENCH_SEG")) : 32768;
    const bool lp = std::getenv("WHIRL_KT_BENCH_LP") != nullptr;  // launch per 4096 x 32k pairs (instead of 4096 x 128k)
    const auto fpw = c.fnOpt("attn_prefill_wmma"), fq8 = c.fnOpt("attn_prefill_wmma_q8"), fq4 = c.fnOpt("attn_prefill_wmma_q4"),
               fps = c.fnOpt("attn_prefill_wmma_s"), fdr = c.fnOpt("kv_dq_rows_r"), fdr4 = c.fnOpt("kv_dq_rows_r_q4");
    auto parts = [&](long long keys) {
        const long long budget = 4096ll * (lp ? 32768ll : 131072ll);
        int p = static_cast<int>(std::min<long long>(heads, (static_cast<long long>(n) * keys + budget - 1) / budget));
        while (p > 1 && heads % p != 0) ++p;
        return p;
    };
    for (const int L : lens) {
        const int pages = (L + wk::kKvPage - 1) / wk::kKvPage;
        const std::size_t rows = static_cast<std::size_t>(pages) * wk::kKvPage, nel = rows * rel;
        std::vector<int> pt(static_cast<std::size_t>(pages)), idt(static_cast<std::size_t>(pages) + 1);
        std::iota(pt.begin(), pt.end(), 0);
        std::shuffle(pt.begin(), pt.end(), c.rng);
        std::iota(idt.begin(), idt.end(), 0);
        Buf dpt(pt), did(idt);
        std::vector<std::int8_t> k8(nel), v8(nel);
        for (auto& x : k8) x = static_cast<std::int8_t>(static_cast<int>(c.rng() % 255) - 127);
        for (auto& x : v8) x = static_cast<std::int8_t>(static_cast<int>(c.rng() % 255) - 127);
        std::vector<std::uint16_t> ksc(nel / 32), vsc(nel / 32), k16h(nel), v16h(nel);
        {
            const auto a = c.randu(nel / 32, 0.001f, 0.004f), b = c.randu(nel / 32, 0.002f, 0.01f);
            for (std::size_t i = 0; i < ksc.size(); ++i) {
                ksc[i] = f2h(a[i]);
                vsc[i] = f2h(b[i]);
            }
            const auto kf = c.randn(nel, 0.3f), vf = c.randn(nel, 0.3f);
            for (std::size_t i = 0; i < nel; ++i) {
                k16h[i] = f2h(kf[i]);
                v16h[i] = f2h(vf[i]);
            }
        }
        // q4 cache: the same random bytes read as nibble pairs (first half of the int8 buffers)
        Buf dk8(k8), dv8(v8), dks(ksc), dvs(vsc), dkf(k16h), dvf(v16h);
        Buf sk(static_cast<std::size_t>(L) * rel * 2), sv(static_cast<std::size_t>(L) * rel * 2), st(static_cast<std::size_t>(n) * heads * 2 * 4);
        const std::vector<float> qp = c.randn(static_cast<std::size_t>(n) * heads * qstride);
        Buf dq(qp), dpos(std::vector<int>{L - n});
        const std::size_t on = static_cast<std::size_t>(n) * heads * hd;
        Buf out(on * 4);
        wk::KvArgs af{}, a8{};
        af.k = dkf.p();
        af.v = dvf.p();
        af.ptab = dpt.p();
        a8.k = dk8.p();
        a8.v = dv8.p();
        a8.ks = dks.p();
        a8.vs = dvs.p();
        a8.ptab = dpt.p();
        wk::KvArgs ad{};
        ad.ptab = did.p();
        const int P = parts(L), hp = heads / P;
        auto direct = [&](hip::Function f, const wk::KvArgs& a) {
            for (int p = 0; p < P; ++p)
                hip::launch(f, {cdiv(n, 128), static_cast<unsigned>(hp), 1}, {256, 1, 1}, 0, c.s, dq.p(), a, out.p(), heads, nkv, qstride, dpos.p(), n,
                            scale, p * hp);
        };
        auto dqAll = [&](hip::Function fd) {
            hip::launch(fd, {cdiv(static_cast<std::uint64_t>(L) * rel, 2048), 1, 1}, {256, 1, 1}, 0, c.s, a8, sk.p(), sv.p(), 0, L, rel);
            wk::KvArgs g = ad;
            g.k = sk.p();
            g.v = sv.p();
            direct(fpw, g);
        };
        auto dqSeg = [&](hip::Function fd) {
            for (int r0 = 0; r0 < L; r0 += seg) {
                const int r1 = std::min(L, r0 + seg), nr = r1 - r0;
                if (!fd) {  // f16 cache: key ranges straight through its page table
                    const int sp = parts(nr), shp = heads / sp;
                    for (int p = 0; p < sp; ++p)
                        hip::launch(fps, {cdiv(n, 128), static_cast<unsigned>(shp), 1}, {256, 1, 1}, 0, c.s, dq.p(), af, out.p(), heads, nkv, qstride,
                                    dpos.p(), n, scale, p * shp, r0, r1, st.p());
                    continue;
                }
                hip::launch(fd, {cdiv(static_cast<std::uint64_t>(nr) * rel, 2048), 1, 1}, {256, 1, 1}, 0, c.s, a8, sk.p(), sv.p(), r0, nr, rel);
                wk::KvArgs g = ad;
                g.k = sk.p() - static_cast<DevPtr>(r0) * rel * 2;
                g.v = sv.p() - static_cast<DevPtr>(r0) * rel * 2;
                const int sp = parts(nr), shp = heads / sp;
                for (int p = 0; p < sp; ++p)
                    hip::launch(fps, {cdiv(n, 128), static_cast<unsigned>(shp), 1}, {256, 1, 1}, 0, c.s, dq.p(), g, out.p(), heads, nkv, qstride, dpos.p(),
                                n, scale, p * shp, r0, r1, st.p());
            }
        };
        struct Run {
            const char* name;
            int ref;  // 0 f16, 1 q8 family, 2 q4 family
            std::function<void()> go;
            bool ok;
        };
        const std::vector<Run> runs = {
            {"attn_prefill_wmma (f16)", 0, [&] { direct(fpw, af); }, static_cast<bool>(fpw)},
            {"f16s: f16 cache ranges + _s", 0, [&] { dqSeg(hip::Function{}); }, static_cast<bool>(fps)},
            {"attn_prefill_wmma (f16) again", 0, [&] { direct(fpw, af); }, static_cast<bool>(fpw)},
            {"attn_prefill_wmma_q8", 1, [&] { direct(fq8, a8); }, static_cast<bool>(fq8)},
            {"dq: kv_dq_rows_r + f16", 1, [&] { dqAll(fdr); }, fdr && fpw},
            {"dqs: ranges + _s", 1, [&] { dqSeg(fdr); }, fdr && fps},
            {"attn_prefill_wmma_q4", 2, [&] { direct(fq4, a8); }, static_cast<bool>(fq4)},
            {"dq4: kv_dq_rows_r_q4 + f16", 2, [&] { dqAll(fdr4); }, fdr4 && fpw},
            {"dqs4: ranges + _s", 2, [&] { dqSeg(fdr4); }, fdr4 && fps},
        };
        double base_us = 0;
        std::vector<float> ref[3];
        for (const Run& r : runs) {
            if (!r.ok) {
                std::printf("   %s: not in this code object\n", r.name);
                continue;
            }
            r.go();
            c.sync();
            hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
            const int iters = 3;
            hip::eventRecord(e0, c.s);
            for (int i = 0; i < iters; ++i) r.go();
            hip::eventRecord(e1, c.s);
            hip::eventSync(e1);
            const double us = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
            hip::eventDestroy(e0);
            hip::eventDestroy(e1);
            if (base_us == 0) base_us = us;
            std::printf("   prefill %-28s n %d L %6d (seg %d): %10.1f us  %+6.1f%% vs f16\n", r.name, n, L, seg, us, 100.0 * (us / base_us - 1.0));
            const auto o = out.down<float>(on);
            if (ref[r.ref].empty()) {
                ref[r.ref] = o;
            } else {
                std::size_t bad = 0;
                for (std::size_t i = 0; i < o.size(); ++i) bad += std::memcmp(&o[i], &ref[r.ref][i], 4) != 0;
                std::printf("   prefill %-28s bitwise mismatches vs the cache kernel: %zu of %zu\n", r.name, bad, o.size());
            }
            std::fflush(stdout);
        }
    }
}

void benchAttn(Ctx& c) {
    c.rep.family = "attnbench";
    if (std::getenv("WHIRL_KT_BENCH_PREFILL")) {
        if (!c.fnOpt("attn_kg6") && c.fnOpt("attn_prefill_wmma_q8")) {
            benchPrefill1151(c);
            return;
        }
        benchPrefill(c);
        return;
    }
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
            // WHIRL_KT_BENCH_NQ: verify rows of one sequence (positions L - nq .. L - 1; attn_dq4 one row
            // per group, attn_wsplit1 up to 16 / grp rows per group)
            const int nq = std::getenv("WHIRL_KT_BENCH_NQ") ? std::max(1, std::min(16, std::atoi(std::getenv("WHIRL_KT_BENCH_NQ")))) : 1;
            const std::vector<float> qp = c.randn(static_cast<std::size_t>(nq) * sh.heads * qstride);
            std::vector<int> posv(static_cast<std::size_t>(nq));
            for (int i = 0; i < nq; ++i) posv[static_cast<std::size_t>(i)] = L - nq + i;
            Buf dq(qp), dpos(posv);
            int ns = std::min(256, static_cast<int>(cdiv(L, wk::kFdChunk)));
            if (env_s) ns = std::atoi(env_s);
            Buf ml(static_cast<std::size_t>(nq) * ns * sh.heads * 2 * 4), acc(static_cast<std::size_t>(nq) * ns * sh.heads * hd * 4);
            const std::size_t on = static_cast<std::size_t>(sh.heads) * hd;
            wk::AwGroups g1{}, gw{};
            int ng1 = 0;  // attn_dq4: one row per group
            for (; ng1 < nq; ++ng1) {
                g1.first[ng1] = ng1;
                g1.count[ng1] = 1;
            }
            int ngw = 0;
            for (int i = 0, per = std::max(1, 16 / (sh.heads / sh.nkv)); i < nq; i += per, ++ngw) {
                gw.first[ngw] = i;
                gw.count[ngw] = std::min(per, nq - i);
            }
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
                const bool one = std::string(name) == "attn_dq4";
                auto launch = [&]() {
                    hip::launch(f, {static_cast<unsigned>(sh.nkv), static_cast<unsigned>(ns), static_cast<unsigned>(one ? ng1 : ngw)}, {threads, 1, 1}, 0,
                                c.s, dq.p(), a, ml.p(), acc.p(), sh.heads, sh.nkv, qstride, dpos.p(), scale, DevPtr{0}, one ? g1 : gw);
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
                std::printf("   %-16s heads %d/%d L %6d splits %3d rows %d: %8.1f us  %6.1f GB/s (one pass)\n", name, sh.heads, sh.nkv, L, ns, nq, us,
                            bytes / us / 1e3);
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
