// whirl-kernel-test family "moedown8" (feat026-moedown8s): the gfx1151 moe_down_mxfp4 load-schedule candidates of a
// development code object (WHIRL_KT_MD8_HSACO = a gfx1151 kernels build with -DWHIRL_MD8_DEV=1; exports
// md8d<id>_mxfp4) bit for bit against the SHIPPED moe_down_mxfp4 of the main code object, same launch
// ({ceil(E / 8), n}, 256 threads), same inputs. Sweep: Ornith 2048 x 512 K 8 (n 1, 2, 4, 5, 16), F 1024 / 1536
// (2 / 3 steps per expert; 1536 x 8 > the LDS limit -> global path), F 768 (nu % 32 != 0 -> shipped fallback),
// K 1 / 3 / 6, E 2000 (partial last block: dead waves around the LDS barrier), ysh = null (no shared expert),
// the skip flag set (nothing written). Synthetic MXFP4 weights (random valid blocks), int8 activations from
// quantize_q8 (valid packed scale words). WHIRL_KT_MD8_IDS=1,3 limits the ids (default 1..12). One invariant per id.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "kt.h"

namespace kt {

void testMoeDown8(Ctx& c) {
    c.rep.family = "moedown8";
    const char* hs = std::getenv("WHIRL_KT_MD8_HSACO");
    const int ti = static_cast<int>(QType::mxfp4);
    if (!hs || !*hs) {
        c.rep.skip("moedown8", "gfx1151 moe_down_mxfp4 candidates", "WHIRL_KT_MD8_HSACO not set");
        return;
    }
    if (c.mod.arch() != hip::Arch::gfx1151 || !c.k.moe_down[ti] || !c.k.quantize_q8) {
        c.rep.skip("moedown8", "gfx1151 moe_down_mxfp4 candidates", "needs the gfx1151 code object with moe_down_mxfp4");
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
        r.name = std::string("moedown8: cannot read ") + hs;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    hip::Module xm = hip::Module::loadData(img.data(), c.mod.arch());
    std::vector<int> idl = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    if (const char* e = std::getenv("WHIRL_KT_MD8_IDS")) {
        idl.clear();
        for (const char* q = e; *q;) {
            idl.push_back(std::atoi(q));
            while (*q && *q != ',') ++q;
            if (*q == ',') ++q;
        }
    }
    std::vector<std::pair<int, hip::Function>> fs;
    for (int id : idl)
        if (const hip::Function f = xm.getFunctionOpt((std::string("md8d") + std::to_string(id) + "_mxfp4").c_str())) fs.push_back({id, f});
    if (fs.empty()) {
        c.rep.skip("moedown8", "md8d*_mxfp4", "not in the candidate code object");
        return;
    }
    std::vector<std::size_t> checked(fs.size(), 0), bad(fs.size(), 0);
    std::vector<std::string> first_bad(fs.size());
    std::size_t ship_runs = 0, ship_changed = 0;

    struct Case {
        int E, F, K, R;
        std::vector<int> ns;
        bool shared, skip;
        const char* what;
    };
    std::vector<Case> cases = {
        {2048, 512, 8, 32, {1, 2, 4, 5, 16}, true, false, "Ornith 2048x512 K8"},
        {2048, 512, 8, 32, {1, 4}, false, false, "Ornith, no shared expert"},
        {2048, 512, 8, 32, {2}, true, true, "Ornith, skip flag set"},
        {2048, 1024, 8, 16, {1, 3}, true, false, "F 1024 (2 steps per expert)"},
        {2048, 1536, 8, 12, {1, 2}, true, false, "F 1536 (3 steps; K*F > LDS limit)"},
        {2048, 768, 8, 12, {1, 2}, true, false, "F 768 (nu % 32 != 0: shipped fallback)"},
        {2048, 512, 1, 8, {1, 4}, true, false, "K 1"},
        {2048, 512, 3, 8, {1, 5}, true, false, "K 3"},
        {2048, 512, 6, 16, {1, 2}, true, false, "K 6"},
        {2000, 512, 8, 32, {1, 3}, true, false, "E 2000 (partial last block)"},
    };
    if (c.quick) cases.resize(4);
    for (const Case& cs : cases) {
        const int E = cs.E, F = cs.F, K = cs.K, R = cs.R;
        const HostMat down = randomMat(c, QType::mxfp4, R * E, F);
        Buf dW(down.data);
        for (int n : cs.ns) {
            std::vector<int> hids(static_cast<std::size_t>(n) * K);
            std::vector<float> hw(hids.size());
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
            const std::size_t na = static_cast<std::size_t>(n) * K * F;
            const std::vector<float> act = c.randn(na, 2.0f);
            Buf dact(act), aq(na), ad(na / 32 * 4);
            hip::launch(c.k.quantize_q8, {cdiv(na, 256), 1, 1}, {256, 1, 1}, 0, c.s, dact.p(), aq.p(), ad.p(), static_cast<int>(na));
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
            const std::vector<float> hsg = c.randu(static_cast<std::size_t>(n), 0.1f, 0.9f);
            Buf ids(hids), w(hw), sg(hsg), dysh(ysh), dskip(std::vector<int>{1});
            c.sync();
            auto run = [&](hip::Function f) {
                Buf dx(x0);
                hip::launch(f, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, aq.p(), ad.p(), ids.p(),
                            w.p(), cs.shared ? sg.p() : DevPtr{0}, cs.shared ? dysh.p() : DevPtr{0}, dx.p(), F, K, cs.skip ? dskip.p() : DevPtr{0});
                c.sync();
                return dx.down<float>(x0.size());
            };
            const std::vector<float> ship = run(c.k.moe_down[ti]);
            ++ship_runs;
            if (!cs.skip && std::memcmp(ship.data(), x0.data(), x0.size() * 4) == 0) ++ship_changed;  // the reference must write
            if (cs.skip && std::memcmp(ship.data(), x0.data(), x0.size() * 4) != 0) ++ship_changed;   // ... and honour skip
            for (std::size_t i = 0; i < fs.size(); ++i) {
                const std::vector<float> y = run(fs[i].second);
                ++checked[i];
                if (std::memcmp(y.data(), ship.data(), ship.size() * 4) != 0) {
                    ++bad[i];
                    if (first_bad[i].empty()) {
                        std::size_t j = 0;
                        while (j < y.size() && std::memcmp(&y[j], &ship[j], 4) == 0) ++j;
                        char b[200];
                        std::snprintf(b, sizeof(b), "%s n %d: x[%zu] %.9g vs shipped %.9g", cs.what, n, j, j < y.size() ? y[j] : 0.f,
                                      j < ship.size() ? ship[j] : 0.f);
                        first_bad[i] = b;
                    }
                }
            }
        }
    }
    {
        Result r;
        r.name = "moedown8: shipped moe_down_mxfp4 writes every run and honours skip (" + std::to_string(ship_runs) + " runs; the reference is live)";
        r.kind = Kind::invariant;
        r.n = ship_runs;
        r.mismatches = ship_changed;
        r.pass = ship_changed == 0 && ship_runs > 0;
        c.rep.add(r);
    }
    for (std::size_t i = 0; i < fs.size(); ++i) {
        Result r;
        r.name = std::string("md8d") + std::to_string(fs[i].first) + "_mxfp4 == shipped moe_down_mxfp4, bitwise (" + std::to_string(checked[i]) +
                 " runs: Ornith n 1..16, F 1024/1536/768, K 1/3/6, E 2000, no shared, skip)";
        r.kind = Kind::invariant;
        r.n = checked[i];
        r.mismatches = bad[i];
        r.pass = bad[i] == 0 && checked[i] > 0;
        if (!first_bad[i].empty()) r.note = "first " + first_bad[i];
        c.rep.add(r);
    }
}

}  // namespace kt
