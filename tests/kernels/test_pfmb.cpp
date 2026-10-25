// whirl-kernel-test family "pfmb" (on request only: --only pfmb): the feat026-prefetch load-path microbenchmark.
// SPDX-License-Identifier: Apache-2.0
// progress\feat026-prefetch.PRIOR_ART.md, kernels: tests\kernels\mb\pfmb.hip (NOT part of the engine code object; built
// on its own by tests\kernels\mb\build_pfmb.bat, loaded from WHIRL_KT_PFMB_HSACO=<code object>; gfx1201 only).
//
// Per cell (one stream shape and grid): the baseline variant (plain global loads, as the engine kernels load today)
// is checked against the C++ model of its checksum (pfs / pfmx) and timed alone with kt::Bench (warm-up, blocked
// hipEvent timing, stop rule, rotation of input copies past the 64 MB Infinity Cache); then every candidate variant
// (raw buffer loads, scalar-path scales, s_prefetch_data) must give the baseline's output bit for bit, and is timed
// against the baseline in ABBA pairs (kt::Bench::runAB, judged number = paired B / A).
// Printed per side: us, GB/s, % of the spec roof (roofFor: 640 GB/s on gfx1201), % of the practical roof (618), and %
// of the MEASURED roof = the pfs_g full-grid median of the same run (the docs\guide\en\kernels.md section 0 pure-read
// pattern; 604-626 GB/s on the old driver).
// Options (environment): WHIRL_KT_PFMB_FAMS=pfs,pfmx,pfkv (default all), WHIRL_KT_PFMB_JSONL=<file> (rows for
// db_record.py jsonl: metric kern.pfmb.<variant>, plus .ab_ratio rows), WHIRL_KT_PFMB_CELL_S=<s> (kt::Bench max_cell_s,
// default 2), WHIRL_KT_PFMB_QUICK=1 (one shape per family).

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "kt.h"
#include "kt_bench.h"

#ifndef WHIRL_KT_VERSION
#define WHIRL_KT_VERSION "dev"
#endif

namespace kt {

namespace {

namespace wb = whirl::bench;

// ---------------------------------------------------------------- C++ models of the checksums (pfmb.hip)
inline std::uint32_t pfMix(std::uint32_t h, std::uint32_t x) { return (h ^ x) * 0x01000193u; }
inline std::uint32_t rd32(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

// pfs_impl: block 256 (8 waves), rows wave-major
std::vector<std::uint32_t> modelPfs(const std::vector<std::uint8_t>& src, std::uint32_t row_bytes, int nrows, int grid) {
    const int wpb = 8, nw = grid * wpb, np = static_cast<int>((row_bytes + 511) / 512);
    std::vector<std::uint32_t> out(static_cast<std::size_t>(grid) * 256);
    for (int b = 0; b < grid; ++b)
        for (int t = 0; t < 256; ++t) {
            const int lane = t & 31, w = t >> 5;
            std::uint32_t acc = 0x811c9dc5u;
            for (int row = b * wpb + w; row < nrows; row += nw) {
                const std::uint8_t* rp = src.data() + static_cast<std::size_t>(row) * row_bytes;
                for (int j = 0; j < (np + 3) / 4 * 4; ++j) {
                    const std::uint32_t o = static_cast<std::uint32_t>(j) * 512 + static_cast<std::uint32_t>(lane) * 16;
                    if (o + 16 > row_bytes) continue;
                    for (int q = 0; q < 4; ++q) acc = pfMix(acc, rd32(rp + o + 4 * q));
                }
            }
            out[static_cast<std::size_t>(b) * 256 + t] = acc;
        }
    return out;
}

// pfmx_impl: block 32, one 16-row tile per wave, tiles grid-strided; per unit: scale dword, pieces 0..3 (x, y)
std::vector<std::uint32_t> modelPfmx(const std::vector<std::uint8_t>& w, std::uint32_t row_bytes, int ncols, int nrows, int grid) {
    const int nu = ncols / 128;
    std::vector<std::uint32_t> out(static_cast<std::size_t>(grid) * 32);
    for (int b = 0; b < grid; ++b)
        for (int lane = 0; lane < 32; ++lane) {
            const int g0 = lane >> 4, rl = lane & 15;
            std::uint32_t acc = 0x811c9dc5u;
            for (int tile = b; tile * 16 < nrows; tile += grid) {
                const std::uint8_t* rp = w.data() + static_cast<std::size_t>(std::min(tile * 16 + rl, nrows - 1)) * row_bytes;
                for (int u = 0; u < nu; ++u) {
                    const std::uint8_t* sb = rp + static_cast<std::size_t>(u >> 1) * 136;
                    const int ph = u & 1;
                    acc = pfMix(acc, rd32(sb + 4 * ph));
                    for (int i = 0; i < 4; ++i) {
                        const std::uint8_t* q = sb + 8 + 16 * (4 * ph + i) + 8 * g0;
                        acc = pfMix(pfMix(acc, rd32(q)), rd32(q + 4));
                    }
                }
            }
            out[static_cast<std::size_t>(b) * 32 + lane] = acc;
        }
    return out;
}

// ---------------------------------------------------------------- environment
EventApi pfEvents() {
    EventApi a;
    a.create = [] { return static_cast<void*>(hip::eventCreate(true)); };
    a.destroy = [](void* e) { hip::eventDestroy(static_cast<hip::Event>(e)); };
    a.record = [](void* e, void* s) { hip::eventRecord(static_cast<hip::Event>(e), static_cast<hip::Stream>(s)); };
    a.sync = [](void* e) { hip::eventSync(static_cast<hip::Event>(e)); };
    a.stream_sync = [](void* s) { hip::streamSync(static_cast<hip::Stream>(s)); };
    a.elapsed_ms = [](void* e0, void* e1) { return static_cast<double>(hip::eventElapsedMs(static_cast<hip::Event>(e0), static_cast<hip::Event>(e1))); };
    return a;
}

struct PfEnv {
    Ctx& c;
    Bench bench;
    hip::Module mod;
    std::vector<char> img;
    std::string arch, pci, hash;
    Roof roof;
    double meas_roof = 0;  // bytes / s, pfs_g full grid of this run
    std::uint64_t target = 0;
    wb::JsonlWriter out;
    bool jsonl = false;
    std::shared_ptr<TimerStats> timer = std::make_shared<TimerStats>();
    explicit PfEnv(Ctx& cc, BenchCfg cfg) : c(cc), bench(cfg) {}
};

struct Case {
    std::string fam, cell, shape, settings;
    double bytes = 0;  // bytes read per call
    std::function<void(hip::Function, std::uint64_t)> call;
    std::function<std::vector<std::uint32_t>(hip::Function)> run;
    std::vector<std::uint32_t> model;  // empty: no C++ model (pfkv), the baseline is only checked for being written
};

BlockFn block(PfEnv& e, hip::Function f, const Case& k) {
    void* const s = static_cast<void*>(e.c.s);
    return eventBlock(pfEvents(), s, [&k, f](std::uint64_t rot, void*) { k.call(f, rot); }, nullptr, e.timer);
}

std::string pct(const PfEnv& e, double us, double bytes) {
    char b[160];
    const double gbs = bytes / (us * 1e3);
    std::snprintf(b, sizeof(b), "%8.1f GB/s %5.1f%% of %.0f, %5.1f%% of %.0f prac, %5.1f%% of measured %.0f", gbs, 100.0 * gbs * 1e9 / e.roof.bw,
                  e.roof.bw / 1e9, 100.0 * gbs * 1e9 / e.roof.bw_prac, e.roof.bw_prac / 1e9, e.meas_roof > 0 ? 100.0 * gbs * 1e9 / e.meas_roof : 0.0,
                  e.meas_roof / 1e9);
    return b;
}

CellInfo info(const PfEnv& e, const Case& k, const std::string& kernel) {
    CellInfo ci;
    ci.family = "pfmb";
    ci.kernel = kernel;
    ci.cell = k.cell;
    ci.shape = k.shape;
    ci.settings = k.settings;
    ci.code_hash = e.hash;
    ci.gpu_arch = e.arch;
    ci.gpu_pci = e.pci;
    ci.started = wb::iso_now();
    ci.bytes = k.bytes;
    return ci;
}

void fail(PfEnv& e, const std::string& what) {
    Result r;
    r.family = "pfmb";
    r.name = what;
    r.pass = false;
    e.c.rep.add(r);
}

// returns the baseline median (us), 0 on failure
double runCase(PfEnv& e, const Case& k, const std::string& a, const std::vector<std::string>& bs) {
    Ctx& c = e.c;
    const hip::Function fa = e.mod.getFunctionOpt(a.c_str());
    if (!fa) {
        c.rep.skip("pfmb", a + " " + k.cell, "kernel not in the pfmb code object");
        return 0;
    }
    const std::vector<std::uint32_t> ya = k.run(fa);
    if (!k.model.empty()) {
        Result r = cmpExact("pfmb " + a + " " + k.cell + " vs C++ model", ya, k.model);
        const bool ok = r.pass;
        c.rep.add(r);
        if (!ok) return 0;
    }
    e.bench.beforeCell();
    const CellStats sa = e.bench.run(block(e, fa, k));
    if (!sa.ok()) {
        fail(e, "pfmb timing " + a + " " + k.cell + ": " + sa.error);
        return 0;
    }
    std::printf("   %-10s %-28s A  %9.2f us  p10 %.2f p90 %.2f n %3lld %-8s %s\n", a.c_str(), k.cell.c_str(), sa.st.median, sa.st.p10, sa.st.p90,
                static_cast<long long>(sa.st.n), sa.stop_reason.c_str(), pct(e, sa.st.median, k.bytes).c_str());
    std::fflush(stdout);
    if (e.jsonl) e.out.write(cellRow(info(e, k, a), sa, WHIRL_KT_VERSION));
    for (const std::string& b : bs) {
        const hip::Function fb = e.mod.getFunctionOpt(b.c_str());
        if (!fb) {
            c.rep.skip("pfmb", b + " " + k.cell, "kernel not in the pfmb code object");
            continue;
        }
        const std::vector<std::uint32_t> yb = k.run(fb);
        Result r = cmpExact("pfmb " + b + " " + k.cell + " bitwise == " + a, yb, ya);
        const bool ok = r.pass;
        c.rep.add(r);
        if (!ok) {
            std::printf("   %-10s %-28s output differs from %s: not timed\n", b.c_str(), k.cell.c_str(), a.c_str());
            continue;
        }
        e.bench.beforeCell();
        const AbStats ab = e.bench.runAB(block(e, fa, k), block(e, fb, k));
        if (!ab.ratio.ok()) {
            fail(e, "pfmb --ab timing " + b + " " + k.cell + ": " + ab.ratio.error);
            continue;
        }
        std::printf("   %-10s %-28s B  %9.2f us (A %.2f)  B/A %.4f [p10 %.4f p90 %.4f, %d pairs, %s]  %s\n", b.c_str(), k.cell.c_str(), ab.b.st.median,
                    ab.a.st.median, ab.ratio.st.median, ab.ratio.st.p10, ab.ratio.st.p90, ab.pairs, ab.ratio.stop_reason.c_str(),
                    pct(e, ab.b.st.median, k.bytes).c_str());
        std::fflush(stdout);
        if (e.jsonl) {
            CellInfo ci = info(e, k, b);
            ci.settings += ";ab=abba;a=" + a;
            ci.variant = "B";
            wb::Row rb = cellRow(ci, ab.b, WHIRL_KT_VERSION);
            rb.extra.push_back({"b_bitwise", "true"});
            e.out.write(rb);
            wb::Row rr = abRow(ci, ab, WHIRL_KT_VERSION, e.hash);
            rr.extra.push_back({"b_bitwise", "true"});
            rr.extra.push_back({"a_kernel", "\"" + a + "\""});
            e.out.write(rr);
        }
    }
    return sa.st.median;
}

std::vector<std::uint8_t> randomBytes(Ctx& c, std::size_t n) {
    std::vector<std::uint8_t> v(n);
    for (std::size_t i = 0; i < n; i += 8) {
        const std::uint64_t r = c.rng();
        std::memcpy(v.data() + i, &r, std::min<std::size_t>(8, n - i));
    }
    return v;
}

struct Copies {
    std::vector<std::shared_ptr<Buf>> b;
    DevPtr at(std::uint64_t r) const { return b[static_cast<std::size_t>(r % b.size())]->p(); }
};
Copies upload(const PfEnv& e, const std::vector<std::uint8_t>& d) {
    Copies cp;
    const int n = rotationCopies(d.size(), e.target, 6ull << 30);
    for (int i = 0; i < n; ++i) cp.b.push_back(std::make_shared<Buf>(d));
    return cp;
}

std::string cellSettings(const PfEnv& e, const Copies& cp, std::size_t ws) {
    return "rot_bytes=" + std::to_string(ws * cp.b.size()) + ";rot_copies=" + std::to_string(cp.b.size()) + ";max_cell_s=" +
           std::to_string(e.bench.cfg().max_cell_s);
}

// ---------------------------------------------------------------- families
const std::vector<std::string> kPfsB = {"pfs_b", "pfs_gp0", "pfs_gp2k", "pfs_gp4k", "pfs_gp8k", "pfs_bp4k"};
const std::vector<std::string> kPfmxB = {"pfmx_b", "pfmx_gs", "pfmx_bs", "pfmx_gp2", "pfmx_gp8", "pfmx_gr8", "pfmx_bp2"};
const std::vector<std::string> kPfkvB = {"pfkv_b", "pfkv_gp1", "pfkv_gp2", "pfkv_bp1"};

void famPfs(PfEnv& e, bool quick) {
    struct S {
        const char* name;
        std::uint32_t row_bytes;
        int nrows;
    };
    // the two MXFP4 FFN row lengths of Swift 27B (5120 / 17408 columns), 47.3 MB each
    std::vector<S> shapes = {{"r2720", 2720, 17408}, {"r9248", 9248, 5120}};
    if (quick) shapes.resize(1);
    for (const S& sh : shapes) {
        const std::size_t ws = static_cast<std::size_t>(sh.row_bytes) * sh.nrows;
        const std::vector<std::uint8_t> h = randomBytes(e.c, ws);
        const Copies cp = upload(e, h);
        // grids: every row its own wave (full), and capped persistent grids of 128 / 512 waves (low occupancy:
        // where latency, not bandwidth, limits the bytes in flight)
        const int full = static_cast<int>(cdiv(sh.nrows, 8));
        for (const int grid : {full, 16, 64}) {
            Case k;
            k.fam = "pfs";
            k.cell = std::string("pfs.") + sh.name + (grid == full ? ".full" : ".w" + std::to_string(grid * 8));
            k.shape = std::to_string(sh.nrows) + "x" + std::to_string(sh.row_bytes) + "B";
            k.settings = cellSettings(e, cp, ws) + ";grid=" + std::to_string(grid) + "x256";
            k.bytes = static_cast<double>(ws);
            auto y = std::make_shared<Buf>(static_cast<std::size_t>(grid) * 256 * 4);
            const std::uint32_t wsb = static_cast<std::uint32_t>(ws), rb = sh.row_bytes;
            const int nr = sh.nrows;
            Ctx* cc = &e.c;
            k.call = [cc, cp, y, grid, wsb, rb, nr](hip::Function f, std::uint64_t r) {
                hip::launch(f, {static_cast<unsigned>(grid), 1, 1}, {256, 1, 1}, 0, cc->s, cp.at(r), wsb, rb, nr, y->p());
            };
            k.run = [cc, &k, y, grid](hip::Function f) {
                y->fill(0xa5);
                k.call(f, 0);
                cc->sync();
                return y->down<std::uint32_t>(static_cast<std::size_t>(grid) * 256);
            };
            k.model = modelPfs(h, sh.row_bytes, sh.nrows, grid);
            const double us = runCase(e, k, "pfs_g", kPfsB);
            if (grid == full && us > 0 && e.meas_roof == 0) {
                e.meas_roof = k.bytes / (us * 1e-6);
                std::printf("   measured pure-stream roof (pfs_g %s): %.1f GB/s\n", k.cell.c_str(), e.meas_roof / 1e9);
            }
        }
    }
}

void famPfmx(PfEnv& e, bool quick) {
    struct S {
        const char* name;
        int nrows, ncols;
    };
    std::vector<S> shapes = {{"ffn_gate", 17408, 5120}, {"ffn_down", 5120, 17408}, {"attn_qkv", 10240, 5120}};
    if (quick) shapes.resize(1);
    for (const S& sh : shapes) {
        const std::uint32_t rb = static_cast<std::uint32_t>(sh.ncols / 256 * 136);
        const std::size_t ws = static_cast<std::size_t>(rb) * sh.nrows;
        const std::vector<std::uint8_t> h = randomBytes(e.c, ws);
        const Copies cp = upload(e, h);
        const int tiles = static_cast<int>(cdiv(sh.nrows, 16));
        std::vector<int> grids = {tiles};
        if (std::string(sh.name) == "ffn_gate") grids.push_back(256);  // persistent, 256 waves (low occupancy)
        for (const int grid : grids) {
            Case k;
            k.fam = "pfmx";
            k.cell = std::string("pfmx.") + sh.name + (grid == tiles ? ".full" : ".w" + std::to_string(grid));
            k.shape = std::to_string(sh.nrows) + "x" + std::to_string(sh.ncols) + " mxfp4";
            k.settings = cellSettings(e, cp, ws) + ";grid=" + std::to_string(grid) + "x32";
            k.bytes = static_cast<double>(ws);
            auto y = std::make_shared<Buf>(static_cast<std::size_t>(grid) * 32 * 4);
            const std::uint32_t wsb = static_cast<std::uint32_t>(ws);
            const int nr = sh.nrows, nc = sh.ncols;
            Ctx* cc = &e.c;
            k.call = [cc, cp, y, grid, wsb, rb, nr, nc](hip::Function f, std::uint64_t r) {
                hip::launch(f, {static_cast<unsigned>(grid), 1, 1}, {32, 1, 1}, 0, cc->s, cp.at(r), wsb, rb, nc, nr, y->p());
            };
            k.run = [cc, &k, y, grid](hip::Function f) {
                y->fill(0xa5);
                k.call(f, 0);
                cc->sync();
                return y->down<std::uint32_t>(static_cast<std::size_t>(grid) * 32);
            };
            k.model = modelPfmx(h, rb, sh.ncols, sh.nrows, grid);
            runCase(e, k, "pfmx_g", kPfmxB);
        }
    }
}

void famPfkv(PfEnv& e, bool quick) {
    constexpr int nkv = 4, page = 256;
    constexpr std::uint32_t rowb = nkv * 256 * 2;  // bytes per pool row
    std::vector<int> lens = {2048, 32768, 131072};
    if (quick) lens = {2048};
    for (const int L : lens) {
        const int pages = static_cast<int>(cdiv(L, page));
        const std::uint64_t region = static_cast<std::uint64_t>(pages) * page * rowb * 2;  // K + V
        const int R = rotationCopies(region, e.target, 6ull << 30);
        const std::uint64_t pool = static_cast<std::uint64_t>(R) * pages * page * rowb;  // per K or V
        if (pool >= (1ull << 32)) {
            e.c.rep.skip("pfmb", "pfkv L" + std::to_string(L), "pool >= 4 GB (32-bit buffer window)");
            continue;
        }
        auto kp = std::make_shared<Buf>(pool), vp = std::make_shared<Buf>(pool);
        {
            const std::vector<std::uint8_t> chunk = randomBytes(e.c, std::size_t{1} << 23);
            for (std::uint64_t off = 0; off < pool; off += chunk.size()) {
                const std::size_t n = static_cast<std::size_t>(std::min<std::uint64_t>(chunk.size(), pool - off));
                hip::upload(kp->p() + off, chunk.data(), n);
                hip::upload(vp->p() + off, chunk.data() + (chunk.size() - n) / 2, n);
            }
            hip::sync();
        }
        std::vector<int> ptab(static_cast<std::size_t>(R) * pages);
        for (int r = 0; r < R; ++r) {
            std::vector<int> perm(static_cast<std::size_t>(pages));
            std::iota(perm.begin(), perm.end(), r * pages);
            std::shuffle(perm.begin(), perm.end(), e.c.rng);
            std::copy(perm.begin(), perm.end(), ptab.begin() + static_cast<std::ptrdiff_t>(r) * pages);
        }
        auto dpt = std::make_shared<Buf>(ptab);
        const int ns = std::min(wk::kFdMaxSplits, static_cast<int>(cdiv(L, wk::kFdChunk)));
        auto y = std::make_shared<Buf>(static_cast<std::size_t>(nkv) * ns * 32 * 4);
        Case k;
        k.fam = "pfkv";
        k.cell = "pfkv.f16.L" + std::to_string(L);
        k.shape = "kv4_hd256_L" + std::to_string(L) + "_splits" + std::to_string(ns);
        k.settings = "rot_bytes=" + std::to_string(region * static_cast<std::uint64_t>(R)) + ";rot_copies=" + std::to_string(R) +
                     ";kv=f16;paged=random;max_cell_s=" + std::to_string(e.bench.cfg().max_cell_s);
        k.bytes = static_cast<double>(L) * rowb * 2;
        const std::uint32_t pb = static_cast<std::uint32_t>(pool);
        Ctx* cc = &e.c;
        k.call = [cc, kp, vp, dpt, y, ns, pb, L, pages](hip::Function f, std::uint64_t r) {
            const DevPtr pt = dpt->p() + (r % (dpt->bytes() / 4 / static_cast<std::size_t>(pages))) * static_cast<std::uint64_t>(pages) * 4;
            const int nk = nkv;  // hip::launch binds arguments by reference: no constexpr local here
            hip::launch(f, {static_cast<unsigned>(nk), static_cast<unsigned>(ns), 1}, {128, 1, 1}, 0, cc->s, kp->p(), vp->p(), pb, pt, L, nk, y->p());
        };
        k.run = [cc, &k, y, ns](hip::Function f) {
            y->fill(0xa5);
            k.call(f, 0);
            cc->sync();
            return y->down<std::uint32_t>(static_cast<std::size_t>(nkv) * ns * 32);
        };
        // no C++ model: the baseline must at least have written every block's 32 words
        {
            const hip::Function fg = e.mod.getFunctionOpt("pfkv_g");
            if (fg) {
                const std::vector<std::uint32_t> y0 = k.run(fg);
                std::size_t unwritten = 0;
                for (std::uint32_t v : y0) unwritten += v == 0xa5a5a5a5u;
                Result r;
                r.family = "pfmb";
                r.kind = Kind::invariant;
                r.name = "pfmb pfkv_g " + k.cell + " every block wrote its checksum";
                r.n = y0.size();
                r.mismatches = unwritten;
                r.pass = unwritten == 0;
                e.c.rep.add(r);
            }
        }
        runCase(e, k, "pfkv_g", kPfkvB);
    }
}

}  // namespace

void benchPfmb(Ctx& c) {
    c.rep.family = "pfmb";
    const char* path = std::getenv("WHIRL_KT_PFMB_HSACO");
    if (!path || !*path) {
        c.rep.skip("pfmb", "pfmb", "WHIRL_KT_PFMB_HSACO not set (build tests\\kernels\\mb\\build_pfmb.bat)");
        return;
    }
    const hip::DeviceInfo di = hip::describeDevice(hip::getDevice());
    if (di.gcn_arch.rfind("gfx120", 0) != 0) {
        c.rep.skip("pfmb", "pfmb", "gfx12 only (device is " + di.gcn_arch + ")");
        return;
    }
    BenchCfg cfg;
    cfg.max_cell_s = 2.0;
    if (const char* s = std::getenv("WHIRL_KT_PFMB_CELL_S")) cfg.max_cell_s = std::atof(s);
    PfEnv e(c, cfg);
    if (std::FILE* fp = std::fopen(path, "rb")) {
        std::fseek(fp, 0, SEEK_END);
        e.img.resize(static_cast<std::size_t>(std::ftell(fp)));
        std::fseek(fp, 0, SEEK_SET);
        if (std::fread(e.img.data(), 1, e.img.size(), fp) != e.img.size()) e.img.clear();
        std::fclose(fp);
    }
    if (e.img.empty()) {
        fail(e, std::string("pfmb: cannot read ") + path);
        return;
    }
    e.mod = hip::Module::loadData(e.img.data(), c.mod.arch());
    e.arch = di.gcn_arch;
    e.pci = "pci:" + std::to_string(di.pci_bus) + ":" + std::to_string(di.pci_device);
    e.hash = sha256Hex(e.img.data(), e.img.size());
    e.roof = roofFor(e.arch);
    e.target = rotationTarget(e.arch);
    if (const char* j = std::getenv("WHIRL_KT_PFMB_JSONL")) {
        e.jsonl = e.out.open(j);
        if (!e.jsonl) fail(e, std::string("pfmb: cannot open ") + j);
    }
    const std::string fams = std::getenv("WHIRL_KT_PFMB_FAMS") ? std::string(",") + std::getenv("WHIRL_KT_PFMB_FAMS") + "," : ",pfs,pfmx,pfkv,";
    const bool quick = c.quick || std::getenv("WHIRL_KT_PFMB_QUICK") != nullptr;
    std::printf("   pfmb %s (sha256 %s) on %s %s; rotation target %.0f MB; roof %.0f / practical %.0f GB/s\n", path, e.hash.c_str(), di.name.c_str(),
                e.pci.c_str(), e.target / 1048576.0, e.roof.bw / 1e9, e.roof.bw_prac / 1e9);
    struct F {
        const char* name;
        void (*fn)(PfEnv&, bool);
    };
    const F fs[] = {{"pfs", famPfs}, {"pfmx", famPfmx}, {"pfkv", famPfkv}};
    for (const F& f : fs) {
        if (fams.find(std::string(",") + f.name + ",") == std::string::npos) continue;
        std::printf("== pfmb %s\n", f.name);
        try {
            f.fn(e, quick);
        } catch (const std::exception& ex) {
            fail(e, std::string("pfmb ") + f.name + " exception: " + ex.what());
        }
    }
    std::printf("   pfmb timer: %lld blocks, %lld retried, %lld invalid\n", static_cast<long long>(e.timer->blocks), static_cast<long long>(e.timer->retries),
                static_cast<long long>(e.timer->invalid));
}

}  // namespace kt
