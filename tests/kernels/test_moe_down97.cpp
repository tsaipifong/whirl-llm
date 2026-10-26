// whirl-kernel-test family "moedown97" (feat026-moedown97): the gfx1201 MXFP4 decode expert load-schedule
// candidates of a development code object (WHIRL_KT_MD97_HSACO = a gfx1201 kernels build with -DWHIRL_MD97_DEV=1;
// exports md97dn<id>_mxfp4w, md97df<id>_mxfp4, md97gu<id>_mxfp4w) bit for bit against the SHIPPED kernels of the
// main code object (moe_down_mxfp4w, moe_down_f_mxfp4, moe_gu_mxfp4w), same launches, same inputs.
//   down (int8 and f32): Ornith 2048 x 512 K 8 (n 1, 2, 4, 5, 16), no shared expert, skip flag (int8 only),
//     F 256 / 768 / 1024 / 1536 / 2560 (nb 8 / 24 / 32 / 48 / 80: every dk / db branch of the k / b stepping),
//     K 1 / 3 / 6, E 2000 (partial last block).
//   gate / up: Ornith C 2048 ff 512 K 8 (n 1, 3, 16), C 768 / 2304 / 4096 (1 partial / 3 partial / 4 steps),
//     ff 100 (partial last block), skip flag.
// Synthetic MXFP4 weights (random valid blocks), int8 activations from quantize_q8. WHIRL_KT_MD97_IDS=dn3,gu1 limits
// the candidates (default: every exported one). One invariant per candidate plus a liveness check per reference.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kt.h"

namespace kt {

namespace {

struct Cand {
    std::string name;
    hip::Function f;
    std::size_t checked = 0, bad = 0;
    std::string first_bad;
};

void routing(Ctx& c, int n, int R, int K, std::vector<int>& hids, std::vector<float>& hw) {
    hids.assign(static_cast<std::size_t>(n) * K, 0);
    hw.assign(hids.size(), 0.f);
    std::uniform_int_distribution<int> ur(0, R - 1);
    std::uniform_real_distribution<float> uw(0.05f, 1.f);
    for (int t = 0; t < n; ++t) {
        float s = 0;
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = ur(c.rng);
                dup = false;
                for (int j = 0; j < k; ++j) dup = dup || hids[static_cast<std::size_t>(t) * K + j] == e;
            } while (dup);
            hids[static_cast<std::size_t>(t) * K + k] = e;
            hw[static_cast<std::size_t>(t) * K + k] = uw(c.rng);
            s += hw[static_cast<std::size_t>(t) * K + k];
        }
        for (int k = 0; k < K; ++k) hw[static_cast<std::size_t>(t) * K + k] /= s;
    }
}

void compare(Cand& cd, const std::vector<float>& y, const std::vector<float>& ref, const char* what, int n) {
    ++cd.checked;
    if (y.size() == ref.size() && std::memcmp(y.data(), ref.data(), ref.size() * 4) == 0) return;
    ++cd.bad;
    if (cd.first_bad.empty()) {
        std::size_t j = 0;
        while (j < y.size() && j < ref.size() && std::memcmp(&y[j], &ref[j], 4) == 0) ++j;
        char b[200];
        std::snprintf(b, sizeof(b), "%s n %d: [%zu] %.9g vs shipped %.9g", what, n, j, j < y.size() ? y[j] : 0.f, j < ref.size() ? ref[j] : 0.f);
        cd.first_bad = b;
    }
}

void report(Ctx& c, const std::vector<Cand>& cs, const std::string& shipped, const char* sweep) {
    for (const Cand& cd : cs) {
        Result r;
        r.name = cd.name + " == shipped " + shipped + ", bitwise (" + std::to_string(cd.checked) + " runs: " + sweep + ")";
        r.kind = Kind::invariant;
        r.n = cd.checked;
        r.mismatches = cd.bad;
        r.pass = cd.bad == 0 && cd.checked > 0;
        if (!cd.first_bad.empty()) r.note = "first " + cd.first_bad;
        c.rep.add(r);
    }
}

void live(Ctx& c, const std::string& shipped, std::size_t runs, std::size_t dead) {
    Result r;
    r.name = "moedown97: shipped " + shipped + " writes every run and honours skip (" + std::to_string(runs) + " runs; the reference is live)";
    r.kind = Kind::invariant;
    r.n = runs;
    r.mismatches = dead;
    r.pass = dead == 0 && runs > 0;
    c.rep.add(r);
}

}  // namespace

void testMoeDown97(Ctx& c) {
    c.rep.family = "moedown97";
    const char* hs = std::getenv("WHIRL_KT_MD97_HSACO");
    const int ti = static_cast<int>(QType::mxfp4);
    if (!hs || !*hs) {
        c.rep.skip("moedown97", "gfx1201 MXFP4 decode expert candidates", "WHIRL_KT_MD97_HSACO not set");
        return;
    }
    if (c.mod.arch() != hip::Arch::gfx1201 || !c.k.moe_down_mxw || !c.k.moe_gu_mxw || !c.k.moe_down_f[ti] || !c.k.quantize_q8) {
        c.rep.skip("moedown97", "gfx1201 MXFP4 decode expert candidates", "needs the gfx1201 code object (moe_*_mxfp4w, moe_down_f_mxfp4)");
        return;
    }
    std::vector<char> img;
    if (std::FILE* fp = std::fopen(hs, "rb")) {
        std::fseek(fp, 0, SEEK_END);
        img.resize(static_cast<std::size_t>(std::ftell(fp)));
        std::fseek(fp, 0, SEEK_SET);
        if (std::fread(img.data(), 1, img.size(), fp) != img.size()) img.clear();
        std::fclose(fp);
    }
    if (img.empty()) {
        Result r;
        r.name = std::string("moedown97: cannot read ") + hs;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    hip::Module xm = hip::Module::loadData(img.data(), c.mod.arch());
    // candidate list: dn1..dn10, df1..df6, gu1..gu4 (WHIRL_KT_MD97_IDS=dn3,df2,gu1 limits it)
    std::vector<std::string> want;
    if (const char* e = std::getenv("WHIRL_KT_MD97_IDS")) {
        for (const char* q = e; *q;) {
            const char* s = q;
            while (*q && *q != ',') ++q;
            want.emplace_back(s, q);
            if (*q == ',') ++q;
        }
    } else {
        for (int i = 1; i <= 10; ++i) want.push_back("dn" + std::to_string(i));
        for (int i = 1; i <= 6; ++i) want.push_back("df" + std::to_string(i));
        for (int i = 1; i <= 4; ++i) want.push_back("gu" + std::to_string(i));
    }
    std::vector<Cand> dn, df, gu;
    for (const std::string& id : want) {
        const std::string fam = id.substr(0, 2);
        const std::string nm = "md97" + id + (fam == "df" ? "_mxfp4" : "_mxfp4w");
        const hip::Function f = xm.getFunctionOpt(nm.c_str());
        if (!f) continue;
        (fam == "dn" ? dn : fam == "df" ? df : gu).push_back({nm, f});
    }
    if (dn.empty() && df.empty() && gu.empty()) {
        c.rep.skip("moedown97", "md97*", "not in the candidate code object");
        return;
    }

    // ---------------- down + combine (int8: moe_down_mxfp4w, f32: moe_down_f_mxfp4)
    struct DCase {
        int E, F, K, R;
        std::vector<int> ns;
        bool shared, skip;
        const char* what;
    };
    std::vector<DCase> dcases = {
        {2048, 512, 8, 32, {1, 2, 4, 5, 16}, true, false, "Ornith 2048x512 K8"},
        {2048, 512, 8, 32, {1, 4}, false, false, "Ornith, no shared expert"},
        {2048, 512, 8, 32, {2}, true, true, "Ornith, skip flag set"},
        {2048, 1024, 8, 16, {1, 3}, true, false, "F 1024 (nb 32)"},
        {2048, 256, 8, 16, {1, 3}, true, false, "F 256 (nb 8: 4 experts per lane step)"},
        {2048, 768, 8, 12, {1, 2}, true, false, "F 768 (nb 24)"},
        {2048, 1536, 8, 12, {1, 2}, true, false, "F 1536 (nb 48)"},
        {1024, 2560, 4, 8, {1, 2}, true, false, "F 2560 (nb 80)"},
        {2048, 512, 1, 8, {1, 4}, true, false, "K 1"},
        {2048, 512, 3, 8, {1, 5}, true, false, "K 3"},
        {2048, 512, 6, 16, {1, 2}, true, false, "K 6"},
        {2000, 512, 8, 32, {1, 3}, true, false, "E 2000 (partial last block)"},
    };
    if (c.quick) dcases.resize(4);
    std::size_t runs_q = 0, dead_q = 0, runs_f = 0, dead_f = 0;
    for (const DCase& cs : dcases) {
        const int E = cs.E, F = cs.F, K = cs.K, R = cs.R;
        const HostMat down = randomMat(c, QType::mxfp4, R * E, F);
        Buf dW(down.data);
        for (int n : cs.ns) {
            std::vector<int> hids;
            std::vector<float> hw;
            routing(c, n, R, K, hids, hw);
            const std::size_t na = static_cast<std::size_t>(n) * K * F;
            const std::vector<float> act = c.randn(na, 2.0f);
            Buf dact(act), aq(na), ad(na / 32 * 4);
            hip::launch(c.k.quantize_q8, {cdiv(na, 256), 1, 1}, {256, 1, 1}, 0, c.s, dact.p(), aq.p(), ad.p(), static_cast<int>(na));
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
            const std::vector<float> hsg = c.randu(static_cast<std::size_t>(n), 0.1f, 0.9f);
            Buf ids(hids), w(hw), sg(hsg), dysh(ysh), dskip(std::vector<int>{1});
            c.sync();
            const DevPtr psg = cs.shared ? sg.p() : DevPtr{0}, pysh = cs.shared ? dysh.p() : DevPtr{0};
            auto runq = [&](hip::Function f) {
                Buf dx(x0);
                hip::launch(f, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, aq.p(), ad.p(), ids.p(),
                            w.p(), psg, pysh, dx.p(), F, K, cs.skip ? dskip.p() : DevPtr{0});
                c.sync();
                return dx.down<float>(x0.size());
            };
            auto runf = [&](hip::Function f) {  // precise: no skip argument
                Buf dx(x0);
                hip::launch(f, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, dact.p(), ids.p(), w.p(),
                            psg, pysh, dx.p(), F, K);
                c.sync();
                return dx.down<float>(x0.size());
            };
            if (!dn.empty()) {
                const std::vector<float> ship = runq(c.k.moe_down_mxw);
                ++runs_q;
                const bool same = std::memcmp(ship.data(), x0.data(), x0.size() * 4) == 0;
                if (cs.skip ? !same : same) ++dead_q;
                for (Cand& cd : dn) compare(cd, runq(cd.f), ship, cs.what, n);
            }
            if (!df.empty() && !cs.skip) {
                const std::vector<float> ship = runf(c.k.moe_down_f[ti]);
                ++runs_f;
                if (std::memcmp(ship.data(), x0.data(), x0.size() * 4) == 0) ++dead_f;
                for (Cand& cd : df) compare(cd, runf(cd.f), ship, cs.what, n);
            }
        }
    }
    const char* dsweep = "Ornith n 1..16, F 256/768/1024/1536/2560, K 1/3/6, E 2000, no shared, skip";
    if (!dn.empty()) {
        live(c, "moe_down_mxfp4w", runs_q, dead_q);
        report(c, dn, "moe_down_mxfp4w", dsweep);
    }
    if (!df.empty()) {
        live(c, "moe_down_f_mxfp4", runs_f, dead_f);
        report(c, df, "moe_down_f_mxfp4", dsweep);
    }

    // ---------------- gate / up (moe_gu_mxfp4w)
    if (!gu.empty()) {
        struct GCase {
            int C, ff, K, R;
            std::vector<int> ns;
            bool skip;
            const char* what;
        };
        std::vector<GCase> gcases = {
            {2048, 512, 8, 32, {1, 3, 16}, false, "Ornith C 2048 ff 512 K8"},
            {2048, 512, 8, 32, {2}, true, "Ornith, skip flag set"},
            {768, 256, 4, 8, {1, 2}, false, "C 768 (1 partial step)"},
            {2304, 256, 4, 8, {1, 2}, false, "C 2304 (3 steps, last partial)"},
            {4096, 128, 4, 8, {1, 2}, false, "C 4096 (4 steps)"},
            {2048, 100, 8, 16, {1, 2}, false, "ff 100 (partial last block)"},
        };
        if (c.quick) gcases.resize(2);
        std::size_t runs = 0, dead = 0;
        for (const GCase& cs : gcases) {
            const int C = cs.C, ff = cs.ff, K = cs.K, R = cs.R;
            const HostMat gate = randomMat(c, QType::mxfp4, R * ff, C), up = randomMat(c, QType::mxfp4, R * ff, C);
            Buf dg(gate.data), du(up.data);
            for (int n : cs.ns) {
                std::vector<int> hids;
                std::vector<float> hw;
                routing(c, n, R, K, hids, hw);
                const std::size_t nh = static_cast<std::size_t>(n) * C;
                const std::vector<float> h = c.randn(nh, 2.0f);
                Buf dh(h), xq(nh), xd(nh / 32 * 4), ids(hids), dskip(std::vector<int>{1});
                hip::launch(c.k.quantize_q8, {cdiv(nh, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), xq.p(), xd.p(), static_cast<int>(nh));
                const std::size_t ny = static_cast<std::size_t>(n) * K * ff;
                const std::vector<float> y0 = c.randn(2 * ny);  // poison: both halves must be written
                c.sync();
                auto run = [&](hip::Function f) {
                    Buf y1(std::vector<float>(y0.begin(), y0.begin() + static_cast<std::ptrdiff_t>(ny)));
                    Buf y2(std::vector<float>(y0.begin() + static_cast<std::ptrdiff_t>(ny), y0.end()));
                    hip::launch(f, {cdiv(2 * ff, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, ff, 2 * ff,
                                xq.p(), xd.p(), ids.p(), y1.p(), y2.p(), C, K, cs.skip ? dskip.p() : DevPtr{0});
                    c.sync();
                    std::vector<float> a = y1.down<float>(ny);
                    const std::vector<float> b = y2.down<float>(ny);
                    a.insert(a.end(), b.begin(), b.end());
                    return a;
                };
                const std::vector<float> ship = run(c.k.moe_gu_mxw);
                ++runs;
                const bool same = std::memcmp(ship.data(), y0.data(), y0.size() * 4) == 0;
                if (cs.skip ? !same : same) ++dead;
                for (Cand& cd : gu) compare(cd, run(cd.f), ship, cs.what, n);
            }
        }
        live(c, "moe_gu_mxfp4w", runs, dead);
        report(c, gu, "moe_gu_mxfp4w", "Ornith n 1..16, C 768/2304/4096, ff 100, skip");
    }
}

}  // namespace kt
