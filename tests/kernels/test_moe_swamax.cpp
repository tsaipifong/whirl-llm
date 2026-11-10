// whirl-kernel-test "swamax" (v0252-swamax, PLAN V1-77; gfx1201 rows only): the fp8 MoE prefill gate + up GEMM with
// SwiGLU and the down projection's per-pos fp8 quantization in its epilogue (kernels/moe_swamax.hip) against the
// chain it replaces in qwen35 ffn.cpp (MoePrefillKind::fp8):
//   gemm8r_moeh / gemm8r_moe32h (f16 g, u) -> silu_mul_x8h (fp8 rows q [pos][F], sx[pos])
//   == gemm8r_moeswa (token half-tile 64) / gemm8r_moeswa32 (half-tile 32, halves = tile bn / 32)
// every fp8 byte of every pos and every row scale bitwise (memcmp), sx_out separate and in place (sx_out == sx).
// Cases: uniform and Zipf routing (heavy experts: many full tiles + a partial tail), 1 / 3 / 37 tokens (partial tiles
// only), real Ornith routing from the 11-10 R9700 dumps (WHIRL_KT_SWA_RT: ';'-separated .i32 files with the
// moesm .md next to them, default scratch\v0252_moesm\out\gpu\rt_code{2k,8k,32k}; layers 0 / 20 / 39 of one forward),
// and crafted fp8 activations / scales for the edge values (sx 0, an f32 denormal, 2^-30 (g, u underflow to f16
// denormals / zero), 2^20 (f16 overflow -> inf, then inf * 0 = NaN in SwiGLU), inf, NaN; all-zero rows; duplicated
// rows (ties)). Presence rows first: both entries in the gfx1201 object and the host table (RED before the kernel
// exists), absent from the gfx1151 object and table.
// Kernel mutants (scratch\v0252_swamax\mk_mutants.ps1, -DSWA_MUT=1/2/3) must fail the bitwise rows.
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
    const P ps[] = {{"gemm8r_moeswa", c.k.gemm8r_moeswa}, {"gemm8r_moeswa32", c.k.gemm8r_moeswa32}};
    for (const P& p : ps) {
        const hip::Function f = c.fnOpt(p.name);
        Result r;
        r.kind = Kind::invariant;
        r.n = 1;
        if (g1201) {
            r.name = std::string(p.name) + " (v0252-swamax) in the gfx1201 code object and loaded into the host table";
            r.pass = f != nullptr && p.host == f;
            if (!r.pass) r.note = f == nullptr ? "missing from the code object" : "not in the host table (kernels_abi.cpp)";
        } else {
            r.name = std::string(p.name) + " absent from this non-gfx1201 code object and from the host table";
            r.pass = f == nullptr && p.host == nullptr;
        }
        r.mismatches = r.pass ? 0 : 1;
        c.rep.add(r);
    }
    return g1201 && c.k.gemm8r_moeswa != nullptr && c.k.gemm8r_moeswa32 != nullptr;
}

struct Weights {
    HostMat gate, up;
    int E = 2048, R = 256, K = 8, F = 512;
};

Weights loadWeights(Ctx& c, int layer) {
    Weights w;
    const whirl::gguf::File* f = c.open(c.models.moemx);
    const std::string p = "blk." + std::to_string(layer) + ".";
    const auto* tg = f ? f->tensor(p + "ffn_gate_exps.weight") : nullptr;
    if (tg && tg->type == whirl::gguf::GgmlType::mxfp4) {
        w.gate = loadMat(*f, p + "ffn_gate_exps.weight");
        w.up = loadMat(*f, p + "ffn_up_exps.weight");
        w.E = w.gate.ncols;
        const auto* td = f->tensor(p + "ffn_down_exps.weight");
        w.F = td ? static_cast<int>(td->ne[0]) : 512;
        w.R = w.gate.nrows / w.F;
        w.K = static_cast<int>(f->getUintOr("qwen35moe.expert_used_count", 8));
    } else {
        w.R = 64;
        w.gate = randomMat(c, QType::mxfp4, w.R * w.F, w.E);
        w.up = randomMat(c, QType::mxfp4, w.R * w.F, w.E);
    }
    return w;
}

struct DevW {
    Buf g, u, rg, ru;
    explicit DevW(const Weights& w) : g(w.gate.data), u(w.up.data), rg(w.gate.ref), ru(w.up.ref) {}
};

// K distinct experts per token; zipf > 0: P(e) ~ 1 / (e + 1)^zipf (heavy low experts), else uniform
std::vector<int> routing(Ctx& c, int n, int R, int K, double zipf) {
    std::vector<double> cdf(static_cast<std::size_t>(R));
    double s = 0;
    for (int e = 0; e < R; ++e) cdf[static_cast<std::size_t>(e)] = (s += zipf > 0 ? 1.0 / std::pow(e + 1.0, zipf) : 1.0);
    std::uniform_real_distribution<double> ud(0.0, s);
    std::vector<int> ids(static_cast<std::size_t>(n) * K);
    for (int t = 0; t < n; ++t)
        for (int k = 0; k < K; ++k) {
            int e;
            bool dup;
            do {
                e = static_cast<int>(std::lower_bound(cdf.begin(), cdf.end(), ud(c.rng)) - cdf.begin());
                e = std::min(e, R - 1);
                dup = false;
                for (int j = 0; j < k; ++j) dup = dup || ids[static_cast<std::size_t>(t) * K + j] == e;
            } while (dup);
            ids[static_cast<std::size_t>(t) * K + k] = e;
        }
    return ids;
}

// fp8 activations of every pos: from h (moe_gather_fp8, the product path) or crafted (x8 / sx given)
struct Acts {
    std::vector<float> h;            // [n][E] (gather) or empty
    std::vector<std::uint8_t> x8;    // [pairs][E] crafted
    std::vector<float> sx;           // [pairs] crafted
};

// One routing, one tile bn: the old chain once, then each new variant (sx_out separate and in place); adds the rows.
void runCase(Ctx& c, const Weights& w, const DevW& dw, const std::vector<int>& ids, int n, int bn, const Acts& a, const std::string& tag) {
    const int E = w.E, F = w.F, R = w.R, K = w.K, pairs = n * K;
    const int max_tiles = (pairs + bn - 1) / bn + R;
    Buf dids(ids), perm(static_cast<std::size_t>(pairs) * 4), inv(static_cast<std::size_t>(pairs) * 4);
    Buf tiles(static_cast<std::size_t>(max_tiles) * 16), nt(4);
    hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dids.p(), pairs, R, bn, perm.p(), inv.p(), tiles.p(), nt.p());
    Buf x8(static_cast<std::size_t>(pairs) * E), sx(static_cast<std::size_t>(pairs) * 4);
    if (!a.h.empty()) {
        Buf dh(a.h);
        hip::launch(c.k.moe_gather_fp8, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), perm.p(), x8.p(), sx.p(), E, K);
        c.sync();
    } else {
        x8.up(a.x8);
        sx.up(a.sx);
    }
    const std::vector<float> sx0 = sx.down<float>(static_cast<std::size_t>(pairs));
    // old chain: gemm8r_moe{,32}h -> silu_mul_x8h
    const unsigned nby = cdiv(F, wk::kMoeBm);
    Buf hg(static_cast<std::size_t>(pairs) * F * 2), hu(static_cast<std::size_t>(pairs) * F * 2);
    Buf qa(static_cast<std::size_t>(pairs) * F), sxa(static_cast<std::size_t>(pairs) * 4);
    qa.fill(0x5a);
    sxa.fill(0x5a);
    hip::launch(bn == 32 ? c.k.gemm8r_moe32h : c.k.gemm8r_moeh, {static_cast<unsigned>(max_tiles) * 2 * nby, 1, 1}, {256, 1, 1}, 0, c.s, dw.g.p(),
                dw.u.p(), w.gate.row_bytes, dw.rg.p(), dw.ru.p(), F, x8.p(), sx.p(), hg.p(), hu.p(), E, tiles.p(), nt.p());
    hip::launch(c.k.silu_mul_x8h, {static_cast<unsigned>(pairs), 1, 1}, {256, 1, 1}, 0, c.s, hg.p(), hu.p(), qa.p(), sxa.p(), F);
    c.sync();
    const auto qref = qa.down<std::uint8_t>(static_cast<std::size_t>(pairs) * F);
    const auto sref = sxa.down<std::uint32_t>(static_cast<std::size_t>(pairs));
    const auto x8_before = x8.down<std::uint8_t>(static_cast<std::size_t>(pairs) * E);
    struct V {
        const char* name;
        hip::Function f;
        int bnk;
    };
    std::vector<V> vs;
    if (bn == 64) vs.push_back({"gemm8r_moeswa", c.k.gemm8r_moeswa, 64});
    vs.push_back({"gemm8r_moeswa32", c.k.gemm8r_moeswa32, 32});
    for (const V& v : vs) {
        const int halves = bn / v.bnk;
        for (int inplace = 0; inplace < 2; ++inplace) {
            Buf sxi(sx0), sxo(static_cast<std::size_t>(pairs) * 4), q(static_cast<std::size_t>(pairs) * F);
            sxo.fill(0xa5);
            q.fill(0xa5);  // a byte the kernel never wrote shows as a mismatch
            const DevPtr so = inplace ? sxi.p() : sxo.p();
            hip::launch(v.f, {static_cast<unsigned>(max_tiles * halves), 1, 1}, {256, 1, 1}, 0, c.s, dw.g.p(), dw.u.p(), w.gate.row_bytes, dw.rg.p(),
                        dw.ru.p(), F, x8.p(), sxi.p(), so, q.p(), E, tiles.p(), nt.p(), halves);
            c.sync();
            const auto qb = q.down<std::uint8_t>(static_cast<std::size_t>(pairs) * F);
            const auto sb = (inplace ? sxi : sxo).down<std::uint32_t>(static_cast<std::size_t>(pairs));
            std::size_t badq = 0, bads = 0;
            for (int p = 0; p < pairs; ++p) {
                badq += std::memcmp(&qb[static_cast<std::size_t>(p) * F], &qref[static_cast<std::size_t>(p) * F], static_cast<std::size_t>(F)) != 0;
                bads += sb[static_cast<std::size_t>(p)] != sref[static_cast<std::size_t>(p)];
            }
            const std::string vn = std::string(v.name) + " (tile bn " + std::to_string(bn) + ", halves " + std::to_string(halves) + (inplace ? ", sx in place" : "") +
                                   ") == gemm8r_moe" + (bn == 32 ? "32" : "") + "h + silu_mul_x8h, " + tag;
            c.rep.add(invRow(vn + ": fp8 rows (bitwise, per pos)", static_cast<std::size_t>(pairs), badq));
            c.rep.add(invRow(vn + ": row scales sx (bitwise)", static_cast<std::size_t>(pairs), bads));
        }
    }
    const auto x8_after = x8.down<std::uint8_t>(static_cast<std::size_t>(pairs) * E);
    c.rep.add(invRow("swamax variants leave the fp8 activations X8 unchanged, " + tag, 1, x8_after != x8_before ? 1 : 0));
}

}  // namespace

// real routing dumps: <file>.i32 + <file>.md ("forwards [a, b, ...]"), 40 layers, K ids per row, forward-major
bool swamaxLoadDump(const std::string& path, int K, SwaDump& d, std::string& why) {
    d.path = path;
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

// ids of layer l of forward j
std::vector<int> swamaxDumpLayer(const SwaDump& d, int K, int j, int l) {
    std::size_t off = 0;
    for (int i = 0; i < j; ++i) off += static_cast<std::size_t>(40) * d.fwd[static_cast<std::size_t>(i)] * K;
    const std::size_t n = static_cast<std::size_t>(d.fwd[static_cast<std::size_t>(j)]) * K;
    off += static_cast<std::size_t>(l) * n;
    return std::vector<int>(d.ids.begin() + static_cast<std::ptrdiff_t>(off), d.ids.begin() + static_cast<std::ptrdiff_t>(off + n));
}

// the default real-routing files (the 11-10 R9700 dump, scratch\v0252_moesm\out\gpu) and the forward each case takes:
// the last full 2048-row forward (deepest context with a full ubatch)
std::vector<std::string> swamaxDumpPaths() {
    const char* env = std::getenv("WHIRL_KT_SWA_RT");
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

int swamaxForward(const std::vector<int>& fwd) {
    int j = 0;
    for (int i = 0; i < static_cast<int>(fwd.size()); ++i)
        if (fwd[static_cast<std::size_t>(i)] >= fwd[static_cast<std::size_t>(j)]) j = i;
    return j;
}

void testMoeSwamax(Ctx& c) {
    c.rep.family = "swamax";
    if (!presence(c)) return;  // gfx1151 (absent by design: rows above) or missing kernels (presence FAIL above: the RED)
    if (!c.k.gemm8r_moeh || !c.k.gemm8r_moe32h || !c.k.silu_mul_x8h || !c.k.moe_route || !c.k.moe_gather_fp8) {
        Result r;
        r.name = "swamax: the old chain (gemm8r_moeh / gemm8r_moe32h, silu_mul_x8h, moe_route, moe_gather_fp8) is in the gfx1201 object";
        r.n = 1;
        r.mismatches = 1;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    const Weights w0 = loadWeights(c, 0);
    if (!wk::swamaxShapeOk(static_cast<std::uint32_t>(w0.F))) {
        Result r;
        r.name = "swamax: model F " + std::to_string(w0.F) + " passes swamaxShapeOk";
        r.n = 1;
        r.mismatches = 1;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    const DevW d0(w0);
    const int E = w0.E, R = w0.R, K = w0.K;
    // synthetic routing: uniform / Zipf, partial tiles only (1, 3, 37 tokens)
    struct S {
        int n;
        double zipf;
        const char* name;
    };
    const S ss[] = {{200, 0.0, "uniform n=200"}, {1024, 1.2, "zipf 1.2 n=1024"}, {1, 0.0, "n=1"}, {3, 0.0, "n=3"}, {37, 0.6, "zipf 0.6 n=37"}};
    for (const S& s : ss) {
        if (c.quick && s.n > 256) continue;
        const std::vector<int> ids = routing(c, s.n, R, K, s.zipf);
        Acts a;
        a.h = c.randn(static_cast<std::size_t>(s.n) * E);
        for (int bn : {64, 32}) runCase(c, w0, d0, ids, s.n, bn, a, s.name);
    }
    // edge values: crafted fp8 rows and scales (zipf routing so heavy and partial tiles both occur)
    {
        const int n = 160, pairs = n * K;
        const std::vector<int> ids = routing(c, n, R, K, 1.0);
        Acts a;
        a.x8.resize(static_cast<std::size_t>(pairs) * E);
        a.sx.resize(static_cast<std::size_t>(pairs));
        std::uniform_int_distribution<int> ub(0, 255);
        for (auto& b : a.x8) {
            int v;
            do v = ub(c.rng);
            while ((v & 0x7f) == 0x7f);  // e4m3fn NaN codes
            b = static_cast<std::uint8_t>(v);
        }
        const float sxs[] = {0.f, 1e-40f, std::ldexp(1.f, -30), std::ldexp(1.f, -12), 1e-3f, 1.f, std::ldexp(1.f, 12), std::ldexp(1.f, 20),
                             INFINITY, NAN};
        const int ns = static_cast<int>(sizeof(sxs) / sizeof(sxs[0]));
        for (int p = 0; p < pairs; ++p) a.sx[static_cast<std::size_t>(p)] = p % 3 == 0 ? sxs[(p / 3) % ns] : 1e-2f;
        for (int p = 5; p < pairs; p += 17) std::fill_n(a.x8.begin() + static_cast<std::ptrdiff_t>(p) * E, E, std::uint8_t{0});  // all-zero rows
        for (int p = 7; p + 1 < pairs; p += 23) {  // duplicated rows (equal row maxima in two pos)
            std::copy_n(a.x8.begin() + static_cast<std::ptrdiff_t>(p) * E, E, a.x8.begin() + static_cast<std::ptrdiff_t>(p + 1) * E);
            a.sx[static_cast<std::size_t>(p) + 1] = a.sx[static_cast<std::size_t>(p)];
        }
        for (int bn : {64, 32}) runCase(c, w0, d0, ids, n, bn, a, "edge values (sx 0 / denormal / 2^-30 / 2^20 / inf / NaN, zero and duplicated rows)");
    }
    // real Ornith routing (11-10 R9700 dumps): layers 0 / 20 / 39 of the last full forward, that layer's weights
    if (c.quick) return;
    const bool real_w = c.open(c.models.moemx) != nullptr;
    for (const std::string& path : swamaxDumpPaths()) {
        SwaDump d;
        std::string why;
        if (!swamaxLoadDump(path, K, d, why)) {
            const char* env = std::getenv("WHIRL_KT_SWA_RT");
            if (!(env && *env)) {  // the default dumps are this machine's scratch files: absent elsewhere = a visible skip
                c.rep.skip("swamax", "real routing " + path, "routing dump not available (" + why + "); WHIRL_KT_SWA_RT names dumps");
                continue;
            }
            Result r;  // dumps named explicitly (the kt job): unreadable = FAIL
            r.name = "swamax real routing " + path + ": " + why;
            r.n = 1;
            r.mismatches = 1;
            r.pass = false;
            c.rep.add(r);
            continue;
        }
        const int j = swamaxForward(d.fwd), n = d.fwd[static_cast<std::size_t>(j)];
        const std::size_t sl = path.find_last_of("\\/");
        const std::string base = sl == std::string::npos ? path : path.substr(sl + 1);
        for (int l : {0, 20, 39}) {
            const Weights wl = real_w ? loadWeights(c, l) : w0;
            const DevW dl(wl);
            std::vector<int> ids = swamaxDumpLayer(d, K, j, l);
            for (int& e : ids) e = std::min(std::max(e, 0), wl.R - 1);  // synthetic weights (no model): R 64
            Acts a;
            a.h = c.randn(static_cast<std::size_t>(n) * E);
            const std::string tag = "real routing " + base + " fwd " + std::to_string(j) + " (" + std::to_string(n) + " rows) layer " + std::to_string(l);
            // the plan's tile for this shape (ops::moeTile) and the other one
            for (int bn : {64, 32}) runCase(c, wl, dl, ids, n, bn, a, tag);
        }
    }
}

}  // namespace kt
