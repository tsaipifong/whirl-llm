// whirl-kernel-test family "moedown8" (feat026-moedown8s): the gfx1151 moe_down_mxfp4 load-schedule candidates of a
// development code object (WHIRL_KT_MD8_HSACO = a gfx1151 kernels build with -DWHIRL_MD8_DEV=1; exports
// md8d<id>_mxfp4) bit for bit against the SHIPPED moe_down_mxfp4 of the main code object, same launch
// ({ceil(E / 8), n}, 256 threads), same inputs. Sweep: Ornith 2048 x 512 K 8 (n 1, 2, 4, 5, 16), F 1024 / 1536
// (2 / 3 steps per expert; 1536 x 8 > the LDS limit -> global path), F 768 (nu % 32 != 0 -> shipped fallback),
// K 1 / 3 / 6, E 2000 (partial last block: dead waves around the LDS barrier), ysh = null (no shared expert),
// the skip flag set (nothing written). Synthetic MXFP4 weights (random valid blocks), int8 activations from
// quantize_q8 (valid packed scale words). WHIRL_KT_MD8_IDS=1,3 limits the ids (default 1..12). One invariant per id.
// feat026-kcand-r1 (review kcand-batch consider 2-4): K 6 x F 1024 (ns = 12: a full KB-8 batch, then a 4-step tail).
// Reference guard: a main object built with a tune id (WHIRL_MD8_MXFP4 != 0, e.g. WHIRL_CODE_OBJECT=cand\m<id>.hsaco
// or a promoted default) exports moe_down_mxfp4_shipped and its moe_down_mxfp4 IS a candidate; then the md8d<id>
// comparison would be candidate vs candidate, so it FAILS instead. Permanent invariant (no dev object needed): when the
// main object exports moe_down_mxfp4_shipped, moe_down_mxfp4 == moe_down_mxfp4_shipped bitwise on every case.
// feat026-moe-skipsafe (review moedown97-kcand-r1 A1): moeskip id -1 tails (feat027: a token's ids are a kept prefix,
// then -1 with weight 0): random kept count per token, every token K - 1 / 1 kept; Ornith (KB 8 / KB 4 batch tails,
// last step, NG), F 1024 (2 steps per expert), K 6 x F 1024, K 3, E 2000, LDS ids 6/8/11 included. Dropped slots carry
// NaN activation scale words (NaN after pk_d), so reading one is never 0 * finite. Reference: the SHIPPED body
// (moe_down_mxfp4, or _shipped of a tune-id main object) on the compacted problem (per token K' = kt, kept slots only),
// valid before feat027 (whose pre-moeskip shipped kernel would read before the weight tensor) and after it. With
// moeskip merged (main object exports moe_topk_skip) the shipped body also runs the -1 problem directly and must equal
// it, and the F 768 (shipped fallback) -1 case runs. Marker: a candidate object (and a tune-id main object) must export
// md8_skipsafe_v1, else it FAILS and its kernels do not run the -1 cases.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    if (c.mod.arch() != hip::Arch::gfx1151 || !c.k.moe_down[ti] || !c.k.quantize_q8) {
        c.rep.skip("moedown8", "gfx1151 moe_down_mxfp4 candidates", "needs the gfx1151 code object with moe_down_mxfp4");
        return;
    }
    // moe.hip renames the shipped entry to moe_down_mxfp4_shipped exactly when a tune id is compiled in
    const hip::Function twin = c.fnOpt("moe_down_mxfp4_shipped");
    const bool skipaware = c.fnOpt("moe_topk_skip") != nullptr;  // feat027-moeskip merged: the shipped body takes id -1
    const bool main_safe = !twin || c.fnOpt("md8_skipsafe_v1") != nullptr;
    bool cand_safe = true;
    if (!hs || !*hs) {
        if (!twin) {
            c.rep.skip("moedown8", "gfx1151 moe_down_mxfp4 candidates", "WHIRL_KT_MD8_HSACO not set (main object is the shipped build)");
            return;
        }
    } else if (twin) {
        Result r;
        r.name = "moedown8: the main code object is a candidate build (exports moe_down_mxfp4_shipped, i.e. WHIRL_MD8_MXFP4 != 0): "
                 "md8d<id> vs its moe_down_mxfp4 would compare a candidate with a candidate; run with the shipped main object "
                 "(WHIRL_CODE_OBJECT cleared)";
        r.kind = Kind::invariant;
        r.pass = false;
        c.rep.add(r);
        hs = nullptr;  // no candidate comparisons; the twin invariant below still runs
    }
    if (!main_safe) {
        Result r;
        r.name = "moedown8: the main code object has a WHIRL_MD8_MXFP4 tune id but no md8_skipsafe_v1 (built before feat026-moe-skipsafe: "
                 "its candidate reads before the weight tensor on a moeskip id -1); rebuild it; its -1 cases are not run";
        r.kind = Kind::invariant;
        r.pass = false;
        c.rep.add(r);
    }
    std::vector<std::pair<int, hip::Function>> fs;
    hip::Module xm;  // owns the candidate functions for the whole test
    if (hs) {
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
        xm = hip::Module::loadData(img.data(), c.mod.arch());
        cand_safe = xm.getFunctionOpt("md8_skipsafe_v1") != nullptr;
        if (!cand_safe) {
            Result r;
            r.name = std::string("moedown8: candidate object ") + hs +
                     " has no md8_skipsafe_v1 (built before feat026-moe-skipsafe); rebuild with -DWHIRL_MD8_DEV=1 from this branch; "
                     "its candidates do not run the id -1 cases";
            r.kind = Kind::invariant;
            r.pass = false;
            c.rep.add(r);
        }
        std::vector<int> idl = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
        if (const char* e = std::getenv("WHIRL_KT_MD8_IDS")) {
            idl.clear();
            for (const char* q = e; *q;) {
                idl.push_back(std::atoi(q));
                while (*q && *q != ',') ++q;
                if (*q == ',') ++q;
            }
        }
        for (int id : idl)
            if (const hip::Function f = xm.getFunctionOpt((std::string("md8d") + std::to_string(id) + "_mxfp4").c_str())) fs.push_back({id, f});
        if (fs.empty() && !twin) {
            c.rep.skip("moedown8", "md8d*_mxfp4", "not in the candidate code object");
            return;
        }
    }
    std::size_t twin_runs = 0, twin_bad = 0;
    std::string twin_first;
    std::vector<std::size_t> checked(fs.size(), 0), bad(fs.size(), 0);
    std::vector<std::string> first_bad(fs.size());
    std::size_t ship_runs = 0, ship_changed = 0, direct_runs = 0, direct_bad = 0;
    std::string direct_first;
    const hip::Function ship_fn = twin ? twin : c.k.moe_down[ti];  // the shipped body
    const float qnan = [] {
        const std::uint32_t b = 0xFFC00000u;  // NaN as a word and after pk_d ((b >> 13) << 12 = 0x7FE00000)
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }();

    struct Case {
        int E, F, K, R;
        std::vector<int> ns;
        bool shared, skip;
        const char* what;
        int drop = 0;  // moeskip tails: 1 random kept count, 2 every token K - 1, 3 every token 1
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
        {2048, 1024, 6, 16, {1, 2}, true, false, "K 6 F 1024 (ns 12: full KB-8 batch + tail)"},
        {2048, 512, 8, 32, {1, 2, 5, 16}, true, false, "Ornith, id -1 tails (random kept count)", 1},
        {2048, 512, 8, 32, {1, 4}, false, false, "Ornith, id -1 tails, no shared expert", 1},
        {2048, 512, 8, 32, {2}, true, false, "Ornith, every token K-1 kept (last step)", 2},
        {2048, 512, 8, 32, {3}, true, false, "Ornith, every token 1 kept", 3},
        {2048, 512, 8, 32, {2}, true, true, "Ornith, id -1 tails + skip flag", 1},
        {2048, 1024, 8, 16, {2, 16}, true, false, "F 1024, id -1 tails", 1},
        {2048, 1024, 6, 16, {3}, true, false, "K 6 F 1024, id -1 tails (NG tail)", 1},
        {2048, 512, 3, 8, {5}, true, false, "K 3, id -1 tails", 1},
        {2000, 512, 8, 32, {3}, true, false, "E 2000, id -1 tails", 1},
        {2048, 768, 8, 12, {2}, true, false, "F 768, id -1 tails (shipped fallback; moeskip merged only)", 1},
    };
    if (c.quick) {
        const Case d = cases[11];  // keep the first id -1 case
        cases.resize(4);
        cases.push_back(d);
    }
    for (const Case& cs : cases) {
        if (cs.drop && (cs.F / 16) % 32 != 0 && !skipaware) continue;  // the fallback is the pre-moeskip shipped body here
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
            std::vector<int> kts(static_cast<std::size_t>(n), K);
            if (cs.drop) {  // moeskip tail: kept prefix renormalized, then id -1 / weight 0
                std::uniform_int_distribution<int> uk(1, K);
                for (int t = 0; t < n; ++t) {
                    const int kt = cs.drop == 1 ? uk(c.rng) : cs.drop == 2 ? (K > 1 ? K - 1 : 1) : 1;
                    float s = 0;
                    for (int k = 0; k < kt; ++k) s += hw[static_cast<std::size_t>(t) * K + k];
                    for (int k = 0; k < K; ++k) {
                        const std::size_t p = static_cast<std::size_t>(t) * K + k;
                        if (k < kt) {
                            hw[p] /= s;
                        } else {
                            hids[p] = -1;
                            hw[p] = 0.f;
                        }
                    }
                    kts[static_cast<std::size_t>(t)] = kt;
                }
            }
            const std::size_t na = static_cast<std::size_t>(n) * K * F;
            const std::vector<float> act = c.randn(na, 2.0f);
            Buf dact(act), aq(na), ad(na / 32 * 4);
            hip::launch(c.k.quantize_q8, {cdiv(na, 256), 1, 1}, {256, 1, 1}, 0, c.s, dact.p(), aq.p(), ad.p(), static_cast<int>(na));
            c.sync();
            const int nbs = F / 32;
            const std::vector<std::int8_t> aqh = aq.down<std::int8_t>(na);
            std::vector<float> adh = ad.down<float>(na / 32);
            if (cs.drop) {  // poison the dropped slots' scale words
                for (std::size_t p = 0; p < hids.size(); ++p)
                    if (hids[p] < 0)
                        for (int b = 0; b < nbs; ++b) adh[p * nbs + b] = qnan;
                ad.up(adh);
            }
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
            // compacted problem: token t alone with K' = kt (its kept slots), so no id -1 reaches the kernel
            auto compact = [&](hip::Function f) {
                std::vector<float> out = x0;
                for (int t = 0; t < n; ++t) {
                    const int kt = kts[static_cast<std::size_t>(t)];
                    const std::size_t p0 = static_cast<std::size_t>(t) * K;
                    Buf id1(std::vector<int>(hids.begin() + static_cast<std::ptrdiff_t>(p0), hids.begin() + static_cast<std::ptrdiff_t>(p0 + kt)));
                    Buf w1(std::vector<float>(hw.begin() + static_cast<std::ptrdiff_t>(p0), hw.begin() + static_cast<std::ptrdiff_t>(p0 + kt)));
                    Buf x1(std::vector<float>(x0.begin() + static_cast<std::ptrdiff_t>(t) * E, x0.begin() + static_cast<std::ptrdiff_t>(t + 1) * E));
                    Buf y1(std::vector<float>(ysh.begin() + static_cast<std::ptrdiff_t>(t) * E, ysh.begin() + static_cast<std::ptrdiff_t>(t + 1) * E));
                    Buf s1(std::vector<float>{hsg[static_cast<std::size_t>(t)]});
                    Buf q1(std::vector<std::int8_t>(aqh.begin() + static_cast<std::ptrdiff_t>(p0 * F), aqh.begin() + static_cast<std::ptrdiff_t>((p0 + kt) * F)));
                    Buf d1(std::vector<float>(adh.begin() + static_cast<std::ptrdiff_t>(p0 * nbs), adh.begin() + static_cast<std::ptrdiff_t>((p0 + kt) * nbs)));
                    hip::launch(f, {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p(), down.row_bytes, E, q1.p(), d1.p(), id1.p(), w1.p(),
                                cs.shared ? s1.p() : DevPtr{0}, cs.shared ? y1.p() : DevPtr{0}, x1.p(), F, kt, cs.skip ? dskip.p() : DevPtr{0});
                    c.sync();
                    const std::vector<float> r = x1.down<float>(static_cast<std::size_t>(E));
                    std::copy(r.begin(), r.end(), out.begin() + static_cast<std::ptrdiff_t>(t) * E);
                }
                return out;
            };
            const std::vector<float> ship = cs.drop ? compact(ship_fn) : run(ship_fn);
            ++ship_runs;
            if (cs.drop && skipaware) {
                const std::vector<float> y = run(ship_fn);
                ++direct_runs;
                if (std::memcmp(y.data(), ship.data(), ship.size() * 4) != 0) {
                    ++direct_bad;
                    if (direct_first.empty()) {
                        std::size_t j = 0;
                        while (j < y.size() && std::memcmp(&y[j], &ship[j], 4) == 0) ++j;
                        char b[200];
                        std::snprintf(b, sizeof(b), "%s n %d: x[%zu] direct %.9g vs compacted %.9g", cs.what, n, j, j < y.size() ? y[j] : 0.f,
                                      j < ship.size() ? ship[j] : 0.f);
                        direct_first = b;
                    }
                }
            }
            if (!cs.skip && std::memcmp(ship.data(), x0.data(), x0.size() * 4) == 0) ++ship_changed;  // the reference must write
            if (cs.skip && std::memcmp(ship.data(), x0.data(), x0.size() * 4) != 0) ++ship_changed;   // ... and honour skip
            if (twin && (!cs.drop || main_safe)) {
                const std::vector<float> y = run(c.k.moe_down[ti]);  // the tune-id entry vs the shipped body
                ++twin_runs;
                if (std::memcmp(y.data(), ship.data(), ship.size() * 4) != 0) {
                    ++twin_bad;
                    if (twin_first.empty()) {
                        std::size_t j = 0;
                        while (j < y.size() && std::memcmp(&y[j], &ship[j], 4) == 0) ++j;
                        char b[200];
                        std::snprintf(b, sizeof(b), "%s n %d: x[%zu] moe_down_mxfp4 %.9g vs _shipped %.9g", cs.what, n, j, j < y.size() ? y[j] : 0.f,
                                      j < ship.size() ? ship[j] : 0.f);
                        twin_first = b;
                    }
                }
            }
            for (std::size_t i = 0; i < fs.size() && (!cs.drop || cand_safe); ++i) {
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
    if (skipaware) {
        Result r;
        r.name = "moedown8: shipped moe_down_mxfp4 on id -1 tails == the compacted problem (moeskip merged; " + std::to_string(direct_runs) + " runs)";
        r.kind = Kind::invariant;
        r.n = direct_runs;
        r.mismatches = direct_bad;
        r.pass = direct_bad == 0 && direct_runs > 0;
        if (!direct_first.empty()) r.note = "first " + direct_first;
        c.rep.add(r);
    }
    if (twin) {
        Result r;
        r.name = "moedown8: moe_down_mxfp4 (tune-id build) == moe_down_mxfp4_shipped, bitwise (" + std::to_string(twin_runs) +
                 " runs; id -1 cases vs the compacted _shipped)";
        r.kind = Kind::invariant;
        r.n = twin_runs;
        r.mismatches = twin_bad;
        r.pass = twin_bad == 0 && twin_runs > 0;
        if (!twin_first.empty()) r.note = "first " + twin_first;
        c.rep.add(r);
    }
    for (std::size_t i = 0; i < fs.size(); ++i) {
        Result r;
        r.name = std::string("md8d") + std::to_string(fs[i].first) + "_mxfp4 == shipped moe_down_mxfp4, bitwise (" + std::to_string(checked[i]) +
                 " runs: Ornith n 1..16, F 1024/1536/768, K 1/3/6, K 6 x F 1024, E 2000, no shared, skip, id -1 tails)";
        r.kind = Kind::invariant;
        r.n = checked[i];
        r.mismatches = bad[i];
        r.pass = bad[i] == 0 && checked[i] > 0;
        if (!first_bad[i].empty()) r.note = "first " + first_bad[i];
        c.rep.add(r);
    }
}

}  // namespace kt
