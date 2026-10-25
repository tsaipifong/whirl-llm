// whirl-kernel-test families "gemvh" (bitwise, part of the default run) and "gemvhbench"
// (timing only, --only gemvhbench): the precise-decode f16-activation matmul gemvh / gemvh2.
// SPDX-License-Identifier: Apache-2.0
// gemvh: for every exported type, ncols (even stage count, odd stage count, 5120), nrows (not a
// multiple of 16 included), n_tok in {1,2,5,16,17,32} and accumulate 0/1, gemvh (n <= 16) and
// gemvh2 (n <= 32) must equal the existing gemms_c* kernels and gemm_c0 bit for bit (accumulate
// starts from the same random prefill). Plus the 27B shapes at the production grid (FIX-RC4-B): 17408 x 5120 and
// 5120 x 17408 (ncols 17408) launched with kernels::gemvhGrid (gfx1151: k >= 2 tiles per block) and with small caps
// (k up to 11), every n in {1,5,16,17,32}, accumulate 0/1, against gemm_c0. Injected k-order / lane-role / phase
// faults (WHIRL_GH_INJECT=1..3 in kernels/gfx1151/gemv_h.hip, loaded with WHIRL_KT_CODE_OBJECT) fail this family.
// gemvhbench: us per call, GB/s of weight bytes and % of the memory-bandwidth peak for gemvh vs
// gemms (the path precise decode used before), real Qwen3.8-27B shapes.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

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
    if (c.mod.arch() == hip::Arch::gfx1151)
        std::printf("   gemvh production grid: 17408 rows -> %u blocks, 248320 rows -> %u blocks (multiProcessorCount %d)\n", wk::gemvhGrid(17408), wk::gemvhGrid(248320),
                    hip::describeDevice(hip::getDevice()).compute_units);
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
                        // gfx1151: the blocks loop over row tiles, so a grid of 1/2 / 1/3 of the tiles (and one block) must give the same bits
                        if (c.mod.arch() == hip::Arch::gfx1151) {
                            for (int div : {2, 3, 1 << 20}) {
                                const int bm = 16 * div > 16 ? 16 * div : 16;  // rows per launched block (div tiles each) for run()'s grid
                                if (nt <= 16 && c.k.gemvh[ti]) chk(run(c.k.gemvh[ti], bm, 1 << 20, 32, false), "gemvh/" + std::to_string(div));
                                if (c.k.gemvh2[ti]) chk(run(c.k.gemvh2[ti], bm, 1 << 20, 32, false), "gemvh2/" + std::to_string(div));
                            }
                        }
                    }
                }
            }
        }
        // production shapes and grids (the sweep above stays below 640 tiles, where gemvhGrid returns one tile per block)
        if (!c.quick) {
            struct PS {
                int nrows, ncols;
            };
            for (const PS ps : {PS{17408, 5120}, PS{5120, 17408}, PS{4099, 17408}}) {
                const HostMat m = randomMat(c, t, ps.nrows, ps.ncols);
                Buf w(m.data);
                Buf x16 = makeX16(c, 32, ps.ncols);
                const unsigned tiles = static_cast<unsigned>(cdiv(ps.nrows, 16));
                std::vector<std::pair<std::string, unsigned>> grids;
                if (c.mod.arch() == hip::Arch::gfx1151) {
                    grids.push_back({"grid gemvhGrid", wk::gemvhGrid(static_cast<unsigned>(ps.nrows))});
                    grids.push_back({"grid cap 100", wk::gemvhGridCap(static_cast<unsigned>(ps.nrows), 100)});
                    grids.push_back({"grid cap 7", wk::gemvhGridCap(static_cast<unsigned>(ps.nrows), 7)});
                } else {
                    grids.push_back({"grid tiles", tiles});
                }
                for (int nt : {1, 5, 16, 17, 32}) {
                    for (int acc = 0; acc < 2; ++acc) {
                        const std::vector<float> init = c.randn(static_cast<std::size_t>(nt) * ps.nrows, 3.0f);
                        auto run = [&](hip::Function f, hip::Dim3 grid, int nth) {
                            Buf y(init);
                            hip::launch(f, grid, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, w.p(), m.row_bytes, x16.p(), y.p(), ps.ncols, ps.nrows, nt,
                                        acc);
                            c.sync();
                            return y.down<float>(static_cast<std::size_t>(nt) * ps.nrows);
                        };
                        const auto& c0 = wk::kGemmCfgs[0];
                        const std::vector<float> ref =
                            run(c.k.gemmc[0][ti], hip::Dim3{cdiv(nt, c0.bn), cdiv(ps.nrows, c0.bm), 1}, c0.nth);
                        for (const auto& [glab, g] : grids) {
                            for (int which = 0; which < 2; ++which) {
                                const hip::Function f = which == 0 ? c.k.gemvh[ti] : c.k.gemvh2[ti];
                                if (!f || (which == 0 && nt > 16)) continue;
                                const std::vector<float> y = run(f, hip::Dim3{g, 1, 1}, 32);
                                ++checked;
                                if (std::memcmp(y.data(), ref.data(), ref.size() * 4) != 0) {
                                    ++bad;
                                    if (first_bad.empty())
                                        first_bad = std::string(which == 0 ? "gemvh " : "gemvh2 ") + glab + " (" + std::to_string(g) + " blocks) ncols " +
                                                    std::to_string(ps.ncols) + " nrows " + std::to_string(ps.nrows) + " n " + std::to_string(nt) +
                                                    " acc " + std::to_string(acc);
                                }
                            }
                        }
                    }
                }
            }
        }
        Result r;
        r.name = std::string("gemvh/gemvh2 == gemm_c0 and gemms_c*, ncols x nrows x n x acc sweep + 27B shapes at the production grid (") + std::to_string(checked) + " runs) " +
                 wk::typeSuffix(t);
        r.kind = Kind::invariant;
        r.n = checked;
        r.mismatches = bad;
        r.pass = bad == 0 && checked > 0;
        if (!first_bad.empty()) r.note = "first " + first_bad;
        c.rep.add(r);
    }
}

// gemvhv8 (feat026-gemvh8s, on request: --only gemvhv8): the gfx1151 gemvh_mxfp4 candidates of a development code
// object (WHIRL_KT_GEMVH8_HSACO = a gfx1151 kernels build with -DWHIRL_GVH8_DEV=1; exports gvh8d<id>_mxfp4, n <= 16)
// bit for bit against the SHIPPED gemvh_mxfp4 of the main code object (same grid) and against gemm_c0, over
// ncols 256 .. 5120 (unit counts 2, 4, 6, 10, 40: every ring tail incl. fewer units than the ring depth) x nrows
// (partial tiles) x n {1, 2, 4, 5, 16} x accumulate 0/1 at grid tiles, 1/2, 1/3 and one block, plus the Swift shapes
// 17408 x 5120, 5120 x 17408, 10240 x 5120, 6144 x 5120, 4099 x 17408 at kernels::gemvhGrid and caps 100 / 7.
// WHIRL_KT_GVH8_IDS=1,4 limits the ids (default 1..8). One invariant per candidate (runs, mismatches).
void testGemvhv8(Ctx& c) {
    c.rep.family = "gemvhv8";
    const char* hs = std::getenv("WHIRL_KT_GEMVH8_HSACO");
    const int ti = static_cast<int>(QType::mxfp4);
    if (!hs || !*hs) {
        c.rep.skip("gemvhv8", "gfx1151 gemvh_mxfp4 candidates", "WHIRL_KT_GEMVH8_HSACO not set");
        return;
    }
    if (c.mod.arch() != hip::Arch::gfx1151 || !c.k.gemvh[ti]) {
        c.rep.skip("gemvhv8", "gfx1151 gemvh_mxfp4 candidates", "needs the gfx1151 code object with gemvh_mxfp4");
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
        r.name = std::string("gemvhv8: cannot read ") + hs;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    hip::Module xm = hip::Module::loadData(img.data(), c.mod.arch());
    std::vector<int> ids = {1, 2, 3, 4, 5, 6, 7, 8};
    if (const char* e = std::getenv("WHIRL_KT_GVH8_IDS")) {
        ids.clear();
        for (const char* q = e; *q;) {
            ids.push_back(std::atoi(q));
            while (*q && *q != ',') ++q;
            if (*q == ',') ++q;
        }
    }
    std::vector<std::pair<int, hip::Function>> fs;
    for (int id : ids)
        if (const hip::Function f = xm.getFunctionOpt((std::string("gvh8d") + std::to_string(id) + "_mxfp4").c_str())) fs.push_back({id, f});
    if (fs.empty()) {
        c.rep.skip("gemvhv8", "gvh8d*_mxfp4", "not in the candidate code object");
        return;
    }
    std::vector<std::size_t> checked(fs.size(), 0), bad_ship(fs.size(), 0), bad_c0(fs.size(), 0);
    std::vector<std::string> first_bad(fs.size());
    std::size_t ship_vs_c0_bad = 0, ship_runs = 0;
    auto sweep = [&](int ncols, int nrows, const std::vector<int>& nlist, const std::vector<std::pair<std::string, unsigned>>& grids) {
        const HostMat m = randomMat(c, QType::mxfp4, nrows, ncols);
        Buf w(m.data);
        Buf x16 = makeX16(c, 32, ncols);
        for (int nt : nlist)
            for (int acc = 0; acc < 2; ++acc) {
                const std::vector<float> init = c.randn(static_cast<std::size_t>(nt) * nrows, 3.0f);
                auto run = [&](hip::Function f, hip::Dim3 grid, int nth) {
                    Buf y(init);
                    hip::launch(f, grid, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, w.p(), m.row_bytes, x16.p(), y.p(), ncols, nrows, nt, acc);
                    c.sync();
                    return y.down<float>(static_cast<std::size_t>(nt) * nrows);
                };
                const auto& c0 = wk::kGemmCfgs[0];
                const std::vector<float> ref = run(c.k.gemmc[0][ti], hip::Dim3{cdiv(nt, c0.bn), cdiv(nrows, c0.bm), 1}, c0.nth);
                for (const auto& gl : grids) {
                    const unsigned g = gl.second;
                    const std::vector<float> ship = run(c.k.gemvh[ti], hip::Dim3{g, 1, 1}, 32);
                    ++ship_runs;
                    if (std::memcmp(ship.data(), ref.data(), ref.size() * 4) != 0) ++ship_vs_c0_bad;
                    for (std::size_t i = 0; i < fs.size(); ++i) {
                        const std::vector<float> y = run(fs[i].second, hip::Dim3{g, 1, 1}, 32);
                        ++checked[i];
                        const bool bs = std::memcmp(y.data(), ship.data(), ship.size() * 4) != 0;
                        const bool bc = std::memcmp(y.data(), ref.data(), ref.size() * 4) != 0;
                        bad_ship[i] += bs ? 1 : 0;
                        bad_c0[i] += bc ? 1 : 0;
                        if ((bs || bc) && first_bad[i].empty())
                            first_bad[i] = "ncols " + std::to_string(ncols) + " nrows " + std::to_string(nrows) + " n " + std::to_string(nt) + " acc " +
                                           std::to_string(acc) + " " + gl.first + " (" + std::to_string(g) + ")";
                    }
                }
            }
    };
    const std::vector<int> ncolsL = c.quick ? std::vector<int>{256, 768} : std::vector<int>{256, 512, 768, 1280, 5120};
    const std::vector<int> nrowsL = c.quick ? std::vector<int>{7, 37} : std::vector<int>{1, 7, 16, 17, 37, 100, 1003};
    for (int ncols : ncolsL)
        for (int nrows : nrowsL) {
            const unsigned tiles = static_cast<unsigned>(cdiv(nrows, 16));
            sweep(ncols, nrows, {1, 2, 4, 5, 16},
                  {{"grid tiles", tiles}, {"grid 1/2", std::max(1u, tiles / 2)}, {"grid 1/3", std::max(1u, tiles / 3)}, {"grid 1", 1u}});
        }
    if (!c.quick) {
        struct PS {
            int nrows, ncols;
        };
        for (const PS ps : {PS{17408, 5120}, PS{5120, 17408}, PS{10240, 5120}, PS{6144, 5120}, PS{4099, 17408}}) {
            const unsigned nr = static_cast<unsigned>(ps.nrows);
            sweep(ps.ncols, ps.nrows, {1, 4, 16},
                  {{"grid gemvhGrid", wk::gemvhGrid(nr)}, {"grid cap 100", wk::gemvhGridCap(nr, 100)}, {"grid cap 7", wk::gemvhGridCap(nr, 7)}});
        }
    }
    {
        Result r;
        r.name = "gemvhv8: shipped gemvh_mxfp4 == gemm_c0 on the same sweep (" + std::to_string(ship_runs) + " runs; the reference is sound)";
        r.kind = Kind::invariant;
        r.n = ship_runs;
        r.mismatches = ship_vs_c0_bad;
        r.pass = ship_vs_c0_bad == 0 && ship_runs > 0;
        c.rep.add(r);
    }
    for (std::size_t i = 0; i < fs.size(); ++i) {
        Result r;
        r.name = std::string("gvh8d") + std::to_string(fs[i].first) + "_mxfp4 == shipped gemvh_mxfp4 and gemm_c0, sweep + Swift shapes + grids (" +
                 std::to_string(checked[i]) + " runs; vs gemm_c0 " + std::to_string(bad_c0[i]) + " bad)";
        r.kind = Kind::invariant;
        r.n = checked[i];
        r.mismatches = bad_ship[i] + bad_c0[i];
        r.pass = bad_ship[i] == 0 && bad_c0[i] == 0 && checked[i] > 0;
        if (!first_bad[i].empty()) r.note = "first " + first_bad[i];
        c.rep.add(r);
    }
}

// gemvhv9 (feat026-gemvh8s2, on request: --only gemvhv9): the gfx1151 gemvh2_mxfp4 / gemvh_q6_k / gemvh_q4_k candidates
// of the development code object (WHIRL_KT_GEMVH8_HSACO, built with -DWHIRL_GVH8_DEV=1: gvh92d<id>_mxfp4 (n <= 32),
// gvh9d<id>_q6_k, gvh9d<id>_q4_k (n <= 16)) bit for bit against the SHIPPED kernel of the main code object (gemvh2_mxfp4,
// gemvh_q6_k, gemvh_q4_k; same grid) and against gemm_c0: ncols sweep (every ring tail incl. fewer units than the ring
// depth) x nrows (partial tiles) x n x accumulate 0/1 at grid tiles, 1/2, 1/3 and one block, plus the Swift shapes
// (LM head rows > 640 tiles, MTP layer / dense shapes) at kernels::gemvhGrid and caps 100 / 7.
// WHIRL_KT_GVH9_ONLY=q6_k,q4_k,mxfp4 limits the kernels, WHIRL_KT_GVH9_IDS=1,4 the ids. One invariant per candidate.
void testGemvhv9(Ctx& c) {
    c.rep.family = "gemvhv9";
    const char* hs = std::getenv("WHIRL_KT_GEMVH8_HSACO");
    if (!hs || !*hs) {
        c.rep.skip("gemvhv9", "gfx1151 gemvh2_mxfp4 / gemvh_q6_k / gemvh_q4_k candidates", "WHIRL_KT_GEMVH8_HSACO not set");
        return;
    }
    if (c.mod.arch() != hip::Arch::gfx1151) {
        c.rep.skip("gemvhv9", "gfx1151 gemvh candidates", "needs the gfx1151 code object");
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
        r.name = std::string("gemvhv9: cannot read ") + hs;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    hip::Module xm = hip::Module::loadData(img.data(), c.mod.arch());
    auto csv = [](const char* e, std::vector<std::string> def) {
        if (!e || !*e) return def;
        std::vector<std::string> out;
        const std::string v = e;
        for (std::size_t a = 0; a <= v.size();) {
            std::size_t b = v.find(',', a);
            if (b == std::string::npos) b = v.size();
            if (b > a) out.push_back(v.substr(a, b - a));
            a = b + 1;
        }
        return out;
    };
    const std::vector<std::string> only = csv(std::getenv("WHIRL_KT_GVH9_ONLY"), {"mxfp4", "q6_k", "q4_k"});
    const std::vector<std::string> idss = csv(std::getenv("WHIRL_KT_GVH9_IDS"), {"1", "2", "3", "4", "5", "6", "7", "8"});
    struct KSpec {
        QType t;
        bool tn2;  // shipped kernel = gemvh2 (n <= 32) instead of gemvh (n <= 16)
        const char* prefix;
        std::vector<int> ncols;
        std::vector<int> ns;
        std::vector<std::pair<int, int>> shapes;  // nrows, ncols
    };
    const std::vector<KSpec> specs = {
        {QType::mxfp4, true, "gvh92d", {256, 512, 768, 1280, 5120}, {1, 5, 16, 17, 20, 32},
         {{17408, 5120}, {5120, 17408}, {10240, 5120}, {6144, 5120}, {4099, 17408}}},
        {QType::q6_k, false, "gvh9d", {256, 512, 768, 1024, 1280, 1536, 5120}, {1, 2, 4, 5, 16}, {{32771, 5120}, {17408, 5120}, {5120, 17408}}},
        {QType::q4_k, false, "gvh9d", {256, 512, 768, 1024, 1280, 1536, 5120}, {1, 2, 4, 5, 16},
         {{17408, 5120}, {5120, 17408}, {10240, 5120}, {6144, 5120}, {4099, 17408}}},
    };
    for (const KSpec& sp : specs) {
        const std::string suf = wk::typeSuffix(sp.t);
        bool want = false;
        for (const std::string& o : only) want = want || o == suf;
        if (!want) continue;
        const int ti = static_cast<int>(sp.t);
        const hip::Function ship_f = sp.tn2 ? c.k.gemvh2[ti] : c.k.gemvh[ti];
        const std::string ship_name = std::string(sp.tn2 ? "gemvh2_" : "gemvh_") + suf;
        if (!ship_f) {
            c.rep.skip("gemvhv9", ship_name, "not in the main code object");
            continue;
        }
        std::vector<std::pair<int, hip::Function>> fs;
        for (const std::string& is : idss) {
            const int id = std::atoi(is.c_str());
            if (const hip::Function f = xm.getFunctionOpt((std::string(sp.prefix) + std::to_string(id) + "_" + suf).c_str())) fs.push_back({id, f});
        }
        if (fs.empty()) {
            c.rep.skip("gemvhv9", std::string(sp.prefix) + "*_" + suf, "not in the candidate code object");
            continue;
        }
        std::vector<std::size_t> checked(fs.size(), 0), bad_ship(fs.size(), 0), bad_c0(fs.size(), 0);
        std::vector<std::string> first_bad(fs.size());
        std::size_t ship_vs_c0_bad = 0, ship_runs = 0;
        auto sweep = [&](int ncols, int nrows, const std::vector<int>& nlist, const std::vector<std::pair<std::string, unsigned>>& grids) {
            const HostMat m = randomMat(c, sp.t, nrows, ncols);
            Buf w(m.data);
            Buf x16 = makeX16(c, 32, ncols);
            for (int nt : nlist)
                for (int acc = 0; acc < 2; ++acc) {
                    const std::vector<float> init = c.randn(static_cast<std::size_t>(nt) * nrows, 3.0f);
                    auto run = [&](hip::Function f, hip::Dim3 grid, int nth) {
                        Buf y(init);
                        hip::launch(f, grid, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, w.p(), m.row_bytes, x16.p(), y.p(), ncols, nrows, nt, acc);
                        c.sync();
                        return y.down<float>(static_cast<std::size_t>(nt) * nrows);
                    };
                    const auto& c0 = wk::kGemmCfgs[0];
                    const std::vector<float> ref = run(c.k.gemmc[0][ti], hip::Dim3{cdiv(nt, c0.bn), cdiv(nrows, c0.bm), 1}, c0.nth);
                    for (const auto& gl : grids) {
                        const unsigned g = gl.second;
                        const std::vector<float> ship = run(ship_f, hip::Dim3{g, 1, 1}, 32);
                        ++ship_runs;
                        if (std::memcmp(ship.data(), ref.data(), ref.size() * 4) != 0) ++ship_vs_c0_bad;
                        for (std::size_t i = 0; i < fs.size(); ++i) {
                            const std::vector<float> y = run(fs[i].second, hip::Dim3{g, 1, 1}, 32);
                            ++checked[i];
                            const bool bs = std::memcmp(y.data(), ship.data(), ship.size() * 4) != 0;
                            const bool bc = std::memcmp(y.data(), ref.data(), ref.size() * 4) != 0;
                            bad_ship[i] += bs ? 1 : 0;
                            bad_c0[i] += bc ? 1 : 0;
                            if ((bs || bc) && first_bad[i].empty())
                                first_bad[i] = "ncols " + std::to_string(ncols) + " nrows " + std::to_string(nrows) + " n " + std::to_string(nt) + " acc " +
                                               std::to_string(acc) + " " + gl.first + " (" + std::to_string(g) + ")";
                        }
                    }
                }
        };
        const std::vector<int> ncolsL = c.quick ? std::vector<int>{256, 768} : sp.ncols;
        const std::vector<int> nrowsL = c.quick ? std::vector<int>{7, 37} : std::vector<int>{1, 7, 16, 17, 37, 100, 1003};
        for (int ncols : ncolsL)
            for (int nrows : nrowsL) {
                const unsigned tiles = static_cast<unsigned>(cdiv(nrows, 16));
                sweep(ncols, nrows, sp.ns,
                      {{"grid tiles", tiles}, {"grid 1/2", std::max(1u, tiles / 2)}, {"grid 1/3", std::max(1u, tiles / 3)}, {"grid 1", 1u}});
            }
        if (!c.quick) {
            const std::vector<int> nprod = sp.tn2 ? std::vector<int>{17, 32} : std::vector<int>{1, 4, 16};
            for (const auto& ps : sp.shapes) {
                const unsigned nr = static_cast<unsigned>(ps.first);
                sweep(ps.second, ps.first, nprod,
                      {{"grid gemvhGrid", wk::gemvhGrid(nr)}, {"grid cap 100", wk::gemvhGridCap(nr, 100)}, {"grid cap 7", wk::gemvhGridCap(nr, 7)}});
            }
        }
        {
            Result r;
            r.name = "gemvhv9: shipped " + ship_name + " == gemm_c0 on the same sweep (" + std::to_string(ship_runs) + " runs; the reference is sound)";
            r.kind = Kind::invariant;
            r.n = ship_runs;
            r.mismatches = ship_vs_c0_bad;
            r.pass = ship_vs_c0_bad == 0 && ship_runs > 0;
            c.rep.add(r);
        }
        for (std::size_t i = 0; i < fs.size(); ++i) {
            Result r;
            r.name = std::string(sp.prefix) + std::to_string(fs[i].first) + "_" + suf + " == shipped " + ship_name + " and gemm_c0, sweep + Swift shapes + grids (" +
                     std::to_string(checked[i]) + " runs; vs gemm_c0 " + std::to_string(bad_c0[i]) + " bad)";
            r.kind = Kind::invariant;
            r.n = checked[i];
            r.mismatches = bad_ship[i] + bad_c0[i];
            r.pass = bad_ship[i] == 0 && bad_c0[i] == 0 && checked[i] > 0;
            if (!first_bad[i].empty()) r.note = "first " + first_bad[i];
            c.rep.add(r);
        }
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
    std::vector<Shape> shapes = {
        {QType::q4_k, 5120, 5120, "5120x5120"},      {QType::q4_k, 17408, 5120, "17408x5120 ffn up"}, {QType::q4_k, 5120, 17408, "5120x17408 ffn down"},
        {QType::q5_k, 5120, 5120, "5120x5120"},      {QType::q5_k, 17408, 5120, "17408x5120 ffn up"}, {QType::q5_k, 5120, 17408, "5120x17408 ffn down"},
        {QType::q5_k, 10240, 5120, "10240x5120 qkv"},
        {QType::q6_k, 5120, 5120, "5120x5120"},      {QType::q6_k, 17408, 5120, "17408x5120 ffn up"}, {QType::q6_k, 5120, 17408, "5120x17408 ffn down"},
        {QType::q6_k, 248320, 5120, "lm head 248320x5120"},
        {QType::q8_0, 5120, 5120, "5120x5120"},      {QType::mxfp4, 5120, 5120, "5120x5120"},
        {QType::iq4_xs, 5120, 5120, "5120x5120"},    {QType::iq4_xs, 17408, 5120, "17408x5120 ffn up"}, {QType::iq4_xs, 5120, 17408, "5120x17408 ffn down"},
        {QType::iq4_nl, 17408, 5120, "17408x5120 ffn up"}, {QType::iq4_nl, 5120, 17408, "5120x17408 ffn down"},
        {QType::q3_k, 17408, 5120, "17408x5120 ffn up"}, {QType::q3_k, 5120, 17408, "5120x17408 ffn down"},
        {QType::iq3_s, 17408, 5120, "17408x5120 ffn up"}, {QType::iq3_s, 5120, 17408, "5120x17408 ffn down"},
    };
    // WHIRL_KT_GEMVH_SHAPES=<type>:<nrows>:<ncols>[,..] replaces the list (development)
    if (const char* e = std::getenv("WHIRL_KT_GEMVH_SHAPES")) {
        shapes.clear();
        static std::vector<std::string> names;
        names.reserve(64);
        const std::string vs = e;
        for (std::size_t a = 0; a < vs.size();) {
            std::size_t b = vs.find(',', a);
            if (b == std::string::npos) b = vs.size();
            const std::string it = vs.substr(a, b - a);
            const std::size_t p1 = it.find(':'), p2 = it.find(':', p1 + 1);
            if (p1 != std::string::npos && p2 != std::string::npos) {
                QType t = QType::q4_k;
                for (QType cand : wk::kAllTypes)
                    if (it.substr(0, p1) == wk::typeSuffix(cand)) t = cand;
                names.push_back(it.substr(p1 + 1));
                shapes.push_back({t, std::atoi(it.substr(p1 + 1, p2 - p1 - 1).c_str()), std::atoi(it.substr(p2 + 1).c_str()), names.back().c_str()});
            }
            a = b + 1;
        }
    }
    std::printf("   (peak reference %.0f GB/s)\n", peak);
    // kernel development: WHIRL_KT_GEMVH_HSACO=<code object> WHIRL_KT_GEMVH_VARS=<prefix>:<rows per block>:<threads>[,..]
    // times <prefix>_<type> from that code object next to the built-in kernels (and checks it bitwise against gemvh2);
    // WHIRL_KT_GEMVH_LITE=1 skips gemms_c1..c7, WHIRL_KT_GEMVH_TYPES=q4_k,q8_0 filters the types,
    // WHIRL_KT_GEMVH_NS=1,16 the token counts.
    struct XVar {
        std::string prefix;
        int bm, nth;
    };
    std::vector<XVar> xvars;
    std::vector<char> ximage;
    hip::Module xmod;
    if (const char* e = std::getenv("WHIRL_KT_GEMVH_HSACO")) {
        if (std::FILE* fp = std::fopen(e, "rb")) {
            std::fseek(fp, 0, SEEK_END);
            ximage.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(ximage.data(), 1, ximage.size(), fp) != ximage.size()) ximage.clear();
            std::fclose(fp);
        }
        if (!ximage.empty()) xmod = hip::Module::loadData(ximage.data(), c.mod.arch());
        const std::string vs = std::getenv("WHIRL_KT_GEMVH_VARS") ? std::getenv("WHIRL_KT_GEMVH_VARS") : "";
        for (std::size_t a = 0; a < vs.size();) {
            std::size_t b = vs.find(',', a);
            if (b == std::string::npos) b = vs.size();
            const std::string it = vs.substr(a, b - a);
            const std::size_t p1 = it.find(':'), p2 = it.find(':', p1 + 1);
            if (p1 != std::string::npos && p2 != std::string::npos)
                xvars.push_back({it.substr(0, p1), std::atoi(it.substr(p1 + 1, p2 - p1 - 1).c_str()), std::atoi(it.substr(p2 + 1).c_str())});
            a = b + 1;
        }
    }
    const bool lite = std::getenv("WHIRL_KT_GEMVH_LITE") != nullptr;
    const bool onlyx = std::getenv("WHIRL_KT_GEMVH_ONLYX") != nullptr;  // time only the development variants
    const std::string tfilter = std::getenv("WHIRL_KT_GEMVH_TYPES") ? std::string(",") + std::getenv("WHIRL_KT_GEMVH_TYPES") + "," : "";
    std::vector<int> nlist = {1, 16, 32};
    if (const char* e = std::getenv("WHIRL_KT_GEMVH_NS")) {
        nlist.clear();
        for (const char* q = e; *q;) {
            nlist.push_back(std::atoi(q));
            while (*q && *q != ',') ++q;
            if (*q == ',') ++q;
        }
    }
    for (const Shape& sh : shapes) {
        if (!tfilter.empty() && tfilter.find(std::string(",") + wk::typeSuffix(sh.t) + ",") == std::string::npos) continue;
        const int ti = static_cast<int>(sh.t);
        if (!c.k.gemvh[ti]) continue;
        const HostMat m = randomMat(c, sh.t, sh.nrows, sh.ncols);
        // enough copies that the weights never stay in the 32 MB MALL between calls
        const int copies = std::max<int>(1, std::min<int>(48, static_cast<int>((640u << 20) / m.data.size())));
        std::vector<Buf> ws;
        for (int i = 0; i < copies; ++i) ws.emplace_back(m.data);
        Buf x16 = makeX16(c, 32, sh.ncols);
        for (int nt : nlist) {
            Buf y(static_cast<std::size_t>(nt) * sh.nrows * 4);
            struct V {
                std::string name;
                hip::Function f;
                int bm, bn, nth;
                bool tok_x;
            };
            std::vector<V> vs;
            if (!onlyx && nt <= 16 && c.k.gemvh[ti]) vs.push_back({"gemvh", c.k.gemvh[ti], 16, 1 << 20, 32, false});
            if (!onlyx && c.k.gemvh2[ti]) vs.push_back({"gemvh2", c.k.gemvh2[ti], 16, 1 << 20, 32, false});
            for (int si = 0; si < static_cast<int>(wk::kGemmsCfgs.size()); ++si) {
                if (onlyx || (lite && si > 0)) break;
                const auto& cf = c.k.gemms_geom[static_cast<std::size_t>(si)];
                if (c.k.gemms[si][ti]) vs.push_back({"gemms_c" + std::to_string(si), c.k.gemms[si][ti], cf.bm, cf.bn, cf.nth, true});
            }
            for (const XVar& xv : xvars) {
                if (!ximage.empty() && nt <= 32) {
                    const hip::Function f = xmod.getFunctionOpt((xv.prefix + "_" + wk::typeSuffix(sh.t)).c_str());
                    if (f) vs.push_back({xv.prefix, f, xv.bm, 1 << 20, xv.nth, false});
                }
            }
            // bitwise check of the development variants against the built-in gemvh2 (any n <= 32)
            if (!xvars.empty() && c.k.gemvh2[ti]) {
                auto runv = [&](hip::Function f, int bm, int nth) {
                    Buf yy(static_cast<std::size_t>(nt) * sh.nrows * 4);
                    hip::launch(f, {cdiv(sh.nrows, bm), 1, 1}, {static_cast<unsigned>(nth), 1, 1}, 0, c.s, ws[0].p(), m.row_bytes, x16.p(), yy.p(),
                                sh.ncols, sh.nrows, nt, 0);
                    c.sync();
                    return yy.down<float>(static_cast<std::size_t>(nt) * sh.nrows);
                };
                const auto ref = runv(c.k.gemvh2[ti], 16, 32);
                for (const V& v : vs) {
                    if (v.tok_x || v.name.rfind("gemvh", 0) == 0 && (v.name == "gemvh" || v.name == "gemvh2")) continue;
                    const auto yv = runv(v.f, v.bm, v.nth);
                    std::size_t bad = 0;
                    for (std::size_t i = 0; i < yv.size(); ++i) bad += std::memcmp(&yv[i], &ref[i], 4) != 0;
                    std::printf("   check %s n %d %s: %zu mismatches\n", v.name.c_str(), nt, sh.what, bad);
                }
            }
            for (const V& v : vs) {
                auto launch = [&](int i) {
                    // WHIRL_KT_GEMVH_GRIDDIV=<d>: development variants (persistent tile loop) get 1/d of the blocks
                    const int gdiv = (!v.tok_x && v.name != "gemvh" && v.name != "gemvh2" && std::getenv("WHIRL_KT_GEMVH_GRIDDIV"))
                                         ? std::max(1, std::atoi(std::getenv("WHIRL_KT_GEMVH_GRIDDIV"))) : 1;
                    // WHIRL_KT_GEMVH_XGRID=prod (feat026-gemvh8s): development variants get the production gemvhGrid too (same blocks as gemvh)
                    const bool xprod = !v.tok_x && std::getenv("WHIRL_KT_GEMVH_XGRID") && std::string(std::getenv("WHIRL_KT_GEMVH_XGRID")) == "prod";
                    const bool shipped = ((v.name == "gemvh" || v.name == "gemvh2") || xprod) && c.mod.arch() == hip::Arch::gfx1151;
                    const hip::Dim3 grid = v.tok_x ? hip::Dim3{cdiv(nt, v.bn), cdiv(sh.nrows, v.bm), 1}
                                                   : hip::Dim3{shipped ? wk::gemvhGrid(static_cast<unsigned>(sh.nrows)) : cdiv(cdiv(sh.nrows, v.bm), gdiv),
                                                               cdiv(nt, v.bn), 1};
                    hip::launch(v.f, grid, {static_cast<unsigned>(v.nth), 1, 1}, 0, c.s, ws[static_cast<std::size_t>(i % copies)].p(), m.row_bytes,
                                x16.p(), y.p(), sh.ncols, sh.nrows, nt, 0);
                };
                for (int i = 0; i < copies; ++i) launch(i);
                c.sync();
                // timed in ~150 ms windows after a ~100 ms warm-up (short windows read the GPU clock ramp-up, not the kernel)
                auto timed = [&](int iters) {
                    hip::Event e0 = hip::eventCreate(true), e1 = hip::eventCreate(true);
                    hip::eventRecord(e0, c.s);
                    for (int i = 0; i < iters; ++i) launch(i);
                    hip::eventRecord(e1, c.s);
                    hip::eventSync(e1);
                    const double us1 = 1000.0 * hip::eventElapsedMs(e0, e1) / iters;
                    hip::eventDestroy(e0);
                    hip::eventDestroy(e1);
                    return us1;
                };
                const int base_iters = std::max(copies * 2, 20);
                double us = timed(base_iters);
                const int win = static_cast<int>(std::min(20000.0, std::max<double>(base_iters, 150000.0 / us)));
                timed(win * 2 / 3);
                us = 1e30;
                for (int rep = 0; rep < 3; ++rep) us = std::min(us, timed(win));
                const double gbs = static_cast<double>(m.data.size()) / (us * 1e3);
                std::printf("   %-6s %-24s n %2d %-9s %9.1f us %7.1f GB/s %5.1f%% of %.0f\n", wk::typeSuffix(sh.t), sh.what, nt, v.name.c_str(),
                            us, gbs, 100.0 * gbs / peak, peak);
                std::fflush(stdout);
            }
        }
    }
}

}  // namespace kt
