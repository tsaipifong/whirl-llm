// whirl-kernel-test "silu" (v0252-silu, PLAN V1-77 follow-up; gfx1201 rows only): silu_mul_x8w / silu_mul_x8w4
// (kernels/moe_silu_w.hip, one wave32 per row) against silu_mul_x8h (kernels/gemm_fp8.hip, one 256-thread block per row),
// the fp8 MoE prefill activation of qwen35 ffn.cpp (MoePrefillKind::fp8: balance / fast on the R9700):
//   every fp8 byte of every row and every row scale sx bitwise (memcmp), and rows >= nrows untouched (guard bytes).
// Cases: random f16 g / u for F = 16 / 48 / 496 / 512 / 1024 and nrows 1 / 7 / 9 / 300 (partial blocks, one and two
// chunks, a chunk partly past F); crafted f16 edge values (NaN, +-inf, f16 denormals, +-0, all-zero rows, 65504 * 65504
// -> y inf -> sx inf, ties of the row max, the row max in lane 0 / 15 / 16 / 31 / the second chunk); real Ornith g / u:
// the 11-10 R9700 routing dumps (WHIRL_KT_SILU_RT, ';'-separated .i32 with the moesm .md beside each, default
// scratch\v0252_moesm\out\gpu\rt_code{2k,8k,32k}), last full forward, layers 0 / 20 / 39, moe_route + moe_gather_fp8 +
// gemm8r_moeh / gemm8r_moe32h (the plan's tile) = the f16 g / u silu_mul_x8h reads in production.
// Presence rows first: both entries in the gfx1201 object and the host table (RED before the kernel exists), absent
// from the gfx1151 object and table.
// Kernel mutants (scratch\v0252_silu\mk_mutants.ps1, -DSILU_MUT=1/2/3) must fail the bitwise rows.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "cpu_ref.h"

namespace kt {

namespace {

Result invRow(const std::string& name, std::size_t n, std::size_t bad, const std::string& note = "") {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = n > 0 && bad == 0;
    r.note = note;
    return r;
}

bool presence(Ctx& c) {
    const bool g1201 = c.mod.arch() == hip::Arch::gfx1201;
    struct P {
        const char* name;
        hip::Function host;
    };
    const P ps[] = {{"silu_mul_x8w", c.k.silu_mul_x8w}, {"silu_mul_x8w4", c.k.silu_mul_x8w4}};
    for (const P& p : ps) {
        const hip::Function f = c.fnOpt(p.name);
        Result r;
        r.kind = Kind::invariant;
        r.n = 1;
        if (g1201) {
            r.name = std::string(p.name) + " (v0252-silu) in the gfx1201 code object and loaded into the host table";
            r.pass = f != nullptr && p.host == f;
            if (!r.pass) r.note = f == nullptr ? "missing from the code object" : "not in the host table (kernels_abi.cpp)";
        } else {
            r.name = std::string(p.name) + " absent from this non-gfx1201 code object and from the host table";
            r.pass = f == nullptr && p.host == nullptr;
        }
        r.mismatches = r.pass ? 0 : 1;
        c.rep.add(r);
    }
    return g1201 && c.k.silu_mul_x8w != nullptr && c.k.silu_mul_x8w4 != nullptr;
}

// One input (f16 g / u, nrows x F, device): silu_mul_x8h once, then each variant; adds 3 rows per variant
// (fp8 rows, row scales, guard rows past nrows untouched).
void runCase(Ctx& c, const Buf& g, const Buf& u, int nrows, int F, const std::string& tag) {
    const std::size_t qbytes = static_cast<std::size_t>(nrows) * F;
    Buf qa(qbytes), sxa(static_cast<std::size_t>(nrows) * 4);
    qa.fill(0x5a);
    sxa.fill(0x5a);
    hip::launch(c.k.silu_mul_x8h, {static_cast<unsigned>(nrows), 1, 1}, {256, 1, 1}, 0, c.s, g.p(), u.p(), qa.p(), sxa.p(), F);
    c.sync();
    const auto qref = qa.down<std::uint8_t>(qbytes);
    const auto sref = sxa.down<std::uint32_t>(static_cast<std::size_t>(nrows));
    struct V {
        const char* name;
        hip::Function f;
        int wpb;
    };
    const V vs[] = {{"silu_mul_x8w", c.k.silu_mul_x8w, wk::siluWRowsPerBlock(false)}, {"silu_mul_x8w4", c.k.silu_mul_x8w4, wk::siluWRowsPerBlock(true)}};
    const int guard = 8;  // rows past nrows the kernel must not write (a grid that covers more than nrows)
    for (const V& v : vs) {
        Buf q((static_cast<std::size_t>(nrows) + guard) * F), sx((static_cast<std::size_t>(nrows) + guard) * 4);
        q.fill(0xa5);  // a byte the kernel never wrote shows as a mismatch
        sx.fill(0xa5);
        hip::launch(v.f, {cdiv(static_cast<std::uint64_t>(nrows), static_cast<std::uint64_t>(v.wpb)), 1, 1}, {static_cast<unsigned>(32 * v.wpb), 1, 1}, 0,
                    c.s, g.p(), u.p(), q.p(), sx.p(), F, nrows);
        c.sync();
        const auto qb = q.down<std::uint8_t>((static_cast<std::size_t>(nrows) + guard) * F);
        const auto sb = sx.down<std::uint32_t>(static_cast<std::size_t>(nrows) + guard);
        std::size_t badq = 0, bads = 0, badg = 0;
        for (int p = 0; p < nrows; ++p) {
            badq += std::memcmp(&qb[static_cast<std::size_t>(p) * F], &qref[static_cast<std::size_t>(p) * F], static_cast<std::size_t>(F)) != 0;
            bads += sb[static_cast<std::size_t>(p)] != sref[static_cast<std::size_t>(p)];
        }
        for (std::size_t i = qbytes; i < qb.size(); ++i) badg += qb[i] != 0xa5;
        for (std::size_t i = static_cast<std::size_t>(nrows); i < sb.size(); ++i) badg += sb[i] != 0xa5a5a5a5u;
        const std::string vn = std::string(v.name) + " == silu_mul_x8h, F " + std::to_string(F) + ", " + std::to_string(nrows) + " rows, " + tag;
        c.rep.add(invRow(vn + ": fp8 rows (bitwise, per row)", static_cast<std::size_t>(nrows), badq));
        c.rep.add(invRow(vn + ": row scales sx (bitwise)", static_cast<std::size_t>(nrows), bads));
        c.rep.add(invRow(vn + ": rows past nrows untouched (q and sx guard bytes)", static_cast<std::size_t>(guard) * (F + 4), badg));
    }
}

void runHost(Ctx& c, const std::vector<std::uint16_t>& g, const std::vector<std::uint16_t>& u, int nrows, int F, const std::string& tag) {
    const Buf dg(g), du(u);
    runCase(c, dg, du, nrows, F, tag);
}

std::vector<std::uint16_t> randh(Ctx& c, std::size_t n, float sigma) {
    const std::vector<float> f = c.randn(n, sigma);
    std::vector<std::uint16_t> h(n);
    for (std::size_t i = 0; i < n; ++i) h[i] = f2h(f[i]);
    return h;
}

struct Dump {
    std::vector<int> fwd;  // rows per forward
    std::vector<int> ids;  // forward-major, then 40 layers of rows x K
};

// <file>.i32 + <file>.md ("forwards [a, b, ...]"), 40 layers, K ids per row, forward-major (the swamax reader)
bool loadDump(const std::string& path, int K, Dump& d, std::string& why) {
    std::string md = path;
    const std::size_t dot = md.rfind(".i32");
    if (dot == std::string::npos) {
        why = "not an .i32 file";
        return false;
    }
    md.replace(dot, 4, ".md");
    std::ifstream fm(md);
    std::string line;
    if (!fm || !std::getline(fm, line)) {
        why = "cannot read " + md;
        return false;
    }
    const std::size_t a = line.find("forwards ["), b = line.find(']', a);
    if (a == std::string::npos || b == std::string::npos) {
        why = "no forwards list in " + md;
        return false;
    }
    std::stringstream ss(line.substr(a + 10, b - a - 10));
    std::string tok;
    while (std::getline(ss, tok, ',')) d.fwd.push_back(std::atoi(tok.c_str()));
    std::ifstream fi(path, std::ios::binary | std::ios::ate);
    if (!fi) {
        why = "cannot read " + path;
        return false;
    }
    const std::streamsize bytes = fi.tellg();
    fi.seekg(0);
    d.ids.resize(static_cast<std::size_t>(bytes / 4));
    fi.read(reinterpret_cast<char*>(d.ids.data()), bytes);
    std::size_t need = 0;
    for (int n : d.fwd) need += static_cast<std::size_t>(40) * n * K;
    if (need != d.ids.size()) {
        why = "layout mismatch (" + std::to_string(d.ids.size()) + " ids, forwards need " + std::to_string(need) + "; mtp dumps not supported)";
        return false;
    }
    return true;
}

}  // namespace

std::vector<std::string> siluDumpPaths() {
    const char* env = std::getenv("WHIRL_KT_SILU_RT");
    std::string s = env && *env ? env
                                : "C:\\ClaudeWorkspace\\whirl-work\\scratch\\v0252_moesm\\out\\gpu\\rt_code2k.i32;"
                                  "C:\\ClaudeWorkspace\\whirl-work\\scratch\\v0252_moesm\\out\\gpu\\rt_code8k.i32;"
                                  "C:\\ClaudeWorkspace\\whirl-work\\scratch\\v0252_moesm\\out\\gpu\\rt_code32k.i32";
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ';'))
        if (!tok.empty()) out.push_back(tok);
    return out;
}

// The f16 g / u rows silu_mul_x8h reads in production for one dump (its last full forward = the largest, latest) and
// one layer: Ornith MXFP4 weights of that layer, moe_route with the plan's token tile (ops::moeTile: n * K >= 48 * R ->
// 64, else 32), moe_gather_fp8 of Gaussian hidden rows, gemm8r_moeh / gemm8r_moe32h. false + why: no model / dump /
// kernels.
bool siluRealGu(Ctx& c, const std::string& path, int layer, SiluGu& out, std::string& why) {
    if (!c.k.gemm8r_moeh || !c.k.gemm8r_moe32h || !c.k.moe_route || !c.k.moe_gather_fp8) {
        why = "gemm8r_moeh / gemm8r_moe32h / moe_route / moe_gather_fp8 not in this code object";
        return false;
    }
    const whirl::gguf::File* f = c.models.moemx.empty() ? nullptr : c.open(c.models.moemx);
    const std::string lp = "blk." + std::to_string(layer) + ".";
    const auto* tg = f ? f->tensor(lp + "ffn_gate_exps.weight") : nullptr;
    if (!tg || tg->type != whirl::gguf::GgmlType::mxfp4) {
        why = "needs --moemx (Ornith-1.5-35B-A3B MXFP4)";
        return false;
    }
    const HostMat gate = loadMat(*f, lp + "ffn_gate_exps.weight"), up = loadMat(*f, lp + "ffn_up_exps.weight");
    const int E = gate.ncols, F = static_cast<int>(f->tensor(lp + "ffn_down_exps.weight")->ne[0]), R = gate.nrows / F;
    const int K = static_cast<int>(f->getUintOr("qwen35moe.expert_used_count", 8));
    Dump d;
    if (!loadDump(path, K, d, why)) return false;
    int j = 0;
    for (int i = 0; i < static_cast<int>(d.fwd.size()); ++i)
        if (d.fwd[static_cast<std::size_t>(i)] >= d.fwd[static_cast<std::size_t>(j)]) j = i;
    const int n = d.fwd[static_cast<std::size_t>(j)], pairs = n * K;
    std::size_t off = 0;
    for (int i = 0; i < j; ++i) off += static_cast<std::size_t>(40) * d.fwd[static_cast<std::size_t>(i)] * K;
    off += static_cast<std::size_t>(layer) * pairs;
    const std::vector<int> ids(d.ids.begin() + static_cast<std::ptrdiff_t>(off), d.ids.begin() + static_cast<std::ptrdiff_t>(off + pairs));
    const int bn = pairs < 48 * R ? 32 : 64;  // ops::moeTile
    const int max_tiles = (pairs + bn - 1) / bn + R;
    const Buf dg(gate.data), du(up.data), rg(gate.ref), ru(up.ref), dids(ids);
    Buf perm(static_cast<std::size_t>(pairs) * 4), inv(static_cast<std::size_t>(pairs) * 4), tiles(static_cast<std::size_t>(max_tiles) * 16), nt(4);
    Buf x8(static_cast<std::size_t>(pairs) * E), sx(static_cast<std::size_t>(pairs) * 4);
    Buf hg(static_cast<std::size_t>(pairs) * F * 2), hu(static_cast<std::size_t>(pairs) * F * 2);
    const Buf dh(c.randn(static_cast<std::size_t>(n) * E));
    hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, bn, perm.p(), inv.p(), tiles.p(), nt.p());
    hip::launch(c.k.moe_gather_fp8, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), x8.p(), sx.p(), E, K);
    const unsigned nby = cdiv(F, wk::kMoeBm);
    hip::launch(bn == 32 ? c.k.gemm8r_moe32h : c.k.gemm8r_moeh, {static_cast<unsigned>(max_tiles) * 2 * nby, 1, 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(),
                gate.row_bytes, rg.p(), ru.p(), F, x8.p(), sx.p(), hg.p(), hu.p(), E, tiles.p(), nt.p());
    c.sync();
    out.n = n;
    out.pairs = pairs;
    out.F = F;
    out.fwd = j;
    out.bn = bn;
    out.g = hg.down<std::uint16_t>(static_cast<std::size_t>(pairs) * F);
    out.u = hu.down<std::uint16_t>(static_cast<std::size_t>(pairs) * F);
    return true;
}

void testSiluW(Ctx& c) {
    c.rep.family = "silu";
    if (!presence(c)) return;  // gfx1151 (absent by design: rows above) or missing kernels (presence FAIL above: the RED)
    if (!c.k.silu_mul_x8h) {
        c.rep.add(invRow("silu: silu_mul_x8h (the reference) is in the gfx1201 object", 1, 1));
        return;
    }
    // shape helper: the kernel's chunk layout
    {
        std::size_t bad = 0;
        for (std::uint32_t F : {16u, 512u, 1024u}) bad += !wk::siluWShapeOk(F);
        for (std::uint32_t F : {0u, 8u, 520u + 4u, 1040u}) bad += wk::siluWShapeOk(F);
        c.rep.add(invRow("silu: siluWShapeOk accepts 16 / 512 / 1024, rejects 0 / 8 / 524 / 1040", 7, bad));
    }
    // random f16 g / u
    for (int F : {16, 48, 496, 512, 1024})
        for (int nrows : {1, 7, 9, 300}) {
            if (c.quick && nrows > 9 && F != 512) continue;
            const std::size_t n = static_cast<std::size_t>(nrows) * F;
            runHost(c, randh(c, n, 2.f), randh(c, n, 1.f), nrows, F, "random g, u");
        }
    // crafted edge values (F 512 = Ornith, and 1024 = two chunks)
    for (int F : {512, 1024}) {
        const int nrows = 64;
        const std::size_t n = static_cast<std::size_t>(nrows) * F;
        std::vector<std::uint16_t> g = randh(c, n, 1.f), u = randh(c, n, 1.f);
        const std::uint16_t specials[] = {0x7e00 /*NaN*/, 0x7c00 /*+inf*/, 0xfc00 /*-inf*/, 0x0001 /*min denormal*/, 0x83ff /*-denormal*/,
                                          0x0000, 0x8000 /*-0*/, 0x7bff /*65504*/, 0xfbff /*-65504*/, 0x3c00 /*1*/};
        const int ns = static_cast<int>(sizeof(specials) / sizeof(specials[0]));
        auto at = [&](std::vector<std::uint16_t>& v, int r, int i) -> std::uint16_t& { return v[static_cast<std::size_t>(r) * F + i]; };
        for (int r = 0; r < nrows; ++r) {
            switch (r % 8) {
                case 0:  // specials sprinkled into g and u
                    for (int i = r % 5; i < F; i += 37) at(g, r, i) = specials[(i / 37) % ns];
                    for (int i = r % 3; i < F; i += 41) at(u, r, i) = specials[(i / 41 + 3) % ns];
                    break;
                case 1:  // all-zero row: m 0 -> sx 0, inv 0
                    std::fill_n(g.begin() + static_cast<std::ptrdiff_t>(r) * F, F, std::uint16_t{0});
                    break;
                case 2:  // 65504 * 65504 -> y inf -> m inf, inv 0, inf * 0 NaN -> -448
                    at(g, r, (r * 13) % F) = 0x7bff;
                    at(u, r, (r * 13) % F) = 0x7bff;
                    break;
                case 3:  // f16 denormals only: tiny y (f32 denormal or 0 after the product), tiny or zero row max
                    for (int i = 0; i < F; ++i) {
                        at(g, r, i) = static_cast<std::uint16_t>(0x0001 + (i % 0x3ff));
                        at(u, r, i) = static_cast<std::uint16_t>(0x0001 + ((i * 7) % 0x3ff));
                    }
                    break;
                default: {  // the row max in a chosen lane / chunk: lane 0, 15, 16, 31, then the last 16 values; with a tie
                    const int lanes[] = {0, 15, 16, 31, F / 16 - 1};
                    const int lane = lanes[(r / 8) % 5];
                    const int i = lane * 16 + (r % 16);
                    at(g, r, i) = f2h(12.f);
                    at(u, r, i) = f2h(9.f);
                    if (r % 8 == 7) {  // tie: the same pair in another lane
                        const int i2 = ((lane + 9) % (F / 16)) * 16 + 3;
                        at(g, r, i2) = f2h(12.f);
                        at(u, r, i2) = f2h(-9.f);
                    }
                    break;
                }
            }
        }
        runHost(c, g, u, nrows, F, "edge values (NaN / inf / denormal / -0 / zero rows / overflow to inf / max lane 0-15-16-31-last / ties)");
    }
    // real Ornith g / u (11-10 R9700 routing dumps): layers 0 / 20 / 39 of the last full forward
    if (c.quick) return;
    for (const std::string& path : siluDumpPaths()) {
        const std::size_t sl = path.find_last_of("\\/");
        const std::string base = sl == std::string::npos ? path : path.substr(sl + 1);
        for (int l : {0, 20, 39}) {
            SiluGu gu;
            std::string why;
            if (!siluRealGu(c, path, l, gu, why)) {
                const char* env = std::getenv("WHIRL_KT_SILU_RT");
                if (!(env && *env)) {  // default dumps are this machine's scratch files: absent elsewhere = a visible skip
                    c.rep.skip("silu", "real routing " + base + " layer " + std::to_string(l), why);
                    continue;
                }
                c.rep.add(invRow("silu real routing " + path + " layer " + std::to_string(l) + ": " + why, 1, 1));  // named dumps: FAIL
                continue;
            }
            runHost(c, gu.g, gu.u, gu.pairs, gu.F,
                    "real Ornith g, u: " + base + " fwd " + std::to_string(gu.fwd) + " (" + std::to_string(gu.n) + " rows) layer " + std::to_string(l) + " tile bn " +
                        std::to_string(gu.bn));
        }
    }
}

}  // namespace kt
