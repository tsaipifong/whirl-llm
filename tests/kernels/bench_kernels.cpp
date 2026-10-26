// whirl-kernel-test --bench FAMILY[,FAMILY] [--jsonl F] [--ab HSACO]: per-kernel timing on real production shapes
// (measure-tools P2; progress\measure-tools.DESIGN.md A1 and (d), progress\mt-p2.PRIOR_ART.md).
// SPDX-License-Identifier: Apache-2.0
//
// Every cell: output check first (vs the C++ CPU reference, the family's existing tolerance; a failed check writes no
// timing row and is a normal FAIL), then kt::Bench (blocked hipEvent timing, warm-up, stop rule, rotation of input
// copies past the last-level cache, anchor drift / cooldown, burst idle) and one JSONL row (tests\bench\bench_out.h ->
// `db_record.py jsonl`). With --ab HSACO the same kernel name from the candidate code object is checked (bitwise vs A,
// else vs the CPU) and timed against A in ABBA order; rows for A, B and the paired ratio B / A are written.
// Families (kernels):
//   gemv  gemvq_<T> (int8 activations, n = 1)                 Swift dense weights  bytes = weight bytes
//   gemvh gemvh_<T> (precise f16 activations, n = 1 / 4)      Swift dense weights  bytes = weight bytes
//   gemm  gemmhq_<T>, gemm_c0_<T>, gemm_c2_f16 on dequantized f16 weights (f16 activations, n = 2048) Swift dense
//         weights  flops = 2 n rows cols
//   attn  attn_wsplit1 + attn_combine (f16 KV, 24 / 4 heads, hd 256, one decode query at L = 2k / 32k / 128k)
//   moe   moe_gu_mxfp4[w], moe_down_mxfp4[w] (Ornith MXFP4 experts, n = 1, K routed experts per launch)
//   gdn   gdn_step_norm (one decode row, 48 v heads x 128 x 128 state)
// Weights: --mx (Swift-1.5 27B MXFP4-A) for gemv / gemvh / gemm / anchor, --moemx (Ornith-1.5-35B-A3B MXFP4) for moe;
// without a model the cells use synthetic MXFP4 blocks of the same shapes (cell names start with "synthetic.").

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <numeric>

#include "cpu_ref.h"
#include "kt_bench.h"

namespace kt {

namespace {

namespace wb = whirl::bench;

// shapes used inside launch lambdas (namespace scope: hip::launch takes its arguments by reference)
namespace at {  // Swift full attention
constexpr int heads = 24, nkv = 4, hd = 256, qstride = 2 * hd, row = nkv * hd;
constexpr float scale = 1.0f / 16.0f;
}  // namespace at
namespace gd {  // Swift Gated DeltaNet
constexpr int nk = 16, nv = 48, dk = 128, dv = 128, ch = 2 * nk * dk + nv * dv, dinner = nv * dv;
const float scale = 1.0f / std::sqrt(128.0f);
constexpr float eps = 1e-6f;
}  // namespace gd

// ---------------------------------------------------------------- shared environment
struct Env {
    Ctx& c;
    const BenchOpts& o;
    Bench bench;
    wb::JsonlWriter out;
    hip::Module xmod;  // --ab candidate
    std::vector<char> ximage;
    bool have_b = false;
    std::uint64_t target = 0;
    std::string settings0;
    int rows_written = 0, rows_lost = 0;
    bool anchor_off_noted = false;

    Env(Ctx& cc, const BenchOpts& oo) : c(cc), o(oo), bench(oo.cfg) {}

    std::string settings(std::uint64_t rot_bytes, int copies) const {
        return settings0 + ";rot_bytes=" + std::to_string(rot_bytes) + ";rot_copies=" + std::to_string(copies);
    }
    CellInfo info(const std::string& fam, const std::string& kernel, const std::string& cell, const std::string& shape) const {
        CellInfo ci;
        ci.family = fam;
        ci.kernel = kernel;
        ci.cell = cell;
        ci.shape = shape;
        ci.code_hash = o.code_hash;
        ci.gpu_arch = o.gpu_arch;
        ci.gpu_pci = o.gpu_pci;
        return ci;
    }
    void write(const wb::Row& r) {
        if (out.write(r)) {
            ++rows_written;
        } else {
            ++rows_lost;
            Result res;
            res.family = c.rep.family;
            res.name = "jsonl write failed: " + r.metric + " " + r.cell;
            res.pass = false;
            c.rep.add(res);
        }
    }
};

// Times a block of calls between two hipEvents on the test stream through kt::eventBlock (fresh event pair per block,
// both on c.s, sync, host-wall sanity check, one retry; whirl::hip throws on any HIP error). The first version reused
// one event pair for every block and accepted short blocks: the R9700 smoke (2026-10-25 14:15) read zero / negative
// intervals for most cells (scratch\mt_p2\smoke_dense_r9700_20261025-1415.log).
std::shared_ptr<TimerStats> g_timer = std::make_shared<TimerStats>();

EventApi hipEvents() {
    EventApi a;
    a.create = [] { return static_cast<void*>(hip::eventCreate(true)); };
    a.destroy = [](void* e) { hip::eventDestroy(static_cast<hip::Event>(e)); };
    a.record = [](void* e, void* s) { hip::eventRecord(static_cast<hip::Event>(e), static_cast<hip::Stream>(s)); };
    a.sync = [](void* e) { hip::eventSync(static_cast<hip::Event>(e)); };
    a.stream_sync = [](void* s) { hip::streamSync(static_cast<hip::Stream>(s)); };
    a.elapsed_ms = [](void* e0, void* e1) { return static_cast<double>(hip::eventElapsedMs(static_cast<hip::Event>(e0), static_cast<hip::Event>(e1))); };
    return a;
}

BlockFn hipBlock(Ctx& c, std::function<void(std::uint64_t rot)> call) {
    void* const s = static_cast<void*>(c.s);
    return eventBlock(hipEvents(), s,
                      [s, call = std::move(call)](std::uint64_t rot, void* stream) {
                          if (stream != s) throw std::runtime_error("bench: timer stream != launch stream");
                          call(rot);  // every cell launches on c.s
                      },
                      nullptr, g_timer);
}

std::string shapeStr(int nrows, int ncols) { return std::to_string(nrows) + "x" + std::to_string(ncols); }

void printCell(const CellInfo& ci, const CellStats& cs, const char* side = "") {
    const double med = cs.st.median;
    std::string work;
    char b[96] = "";
    if (cs.ok() && ci.bytes > 0) {
        const double g = ci.bytes / med / 1e3, roof = roofFor(ci.gpu_arch).bw;
        std::snprintf(b, sizeof(b), " %8.1f GB/s %5.1f%% of %.0f", g, roof > 0 ? 100.0 * g * 1e9 / roof : 0.0, roof / 1e9);
    } else if (cs.ok() && ci.flops > 0) {
        const double t = ci.flops / med / 1e6, roof = roofFor(ci.gpu_arch).f16_prac;
        std::snprintf(b, sizeof(b), " %7.2f TF/s %5.1f%% of %.0f", t, roof > 0 ? 100.0 * t * 1e12 / roof : 0.0, roof / 1e12);
    }
    std::printf("   %-5s %-22s %-34s %s%10.2f us  p10 %.2f p90 %.2f  n %3lld blk %4d %-8s%s%s\n", ci.family.c_str(), ci.kernel.c_str(),
                ci.cell.c_str(), side, med, cs.st.p10, cs.st.p90, static_cast<long long>(cs.st.n), cs.block, cs.stop_reason.c_str(), b,
                cs.cooldowns ? (" cooldowns " + std::to_string(cs.cooldowns)).c_str() : "");
    if (!cs.error.empty()) std::printf("         error: %s; last invalid reading: %s\n", cs.error.c_str(), g_timer->last_invalid.c_str());
    if (cs.invalid) std::printf("         %d invalid timer reading(s) dropped\n", cs.invalid);
    std::fflush(stdout);
}

// One cell: check A (and B), time, write rows. call(f, rot) issues one call of the cell with main kernel f; run(f) does
// one clean call and returns the output; verify(out) compares it with the CPU reference.
struct Cell {
    CellInfo ci;
    std::string kernel;  // name looked up in the candidate code object for --ab
    hip::Function fa;
    std::function<void(hip::Function, std::uint64_t)> call;
    std::function<std::vector<float>(hip::Function)> run;
    std::function<Result(const std::vector<float>&)> verify;
};

void runCell(Env& e, Cell& k) {
    Ctx& c = e.c;
    const std::vector<float> ya = k.run(k.fa);
    Result ra = k.verify(ya);
    ra.name = "bench " + k.kernel + " " + k.ci.cell + ": " + ra.name;
    const bool ok_a = ra.pass;
    c.rep.add(ra);
    if (!ok_a) {
        std::printf("   %s %s: output check FAILED, no timing row\n", k.kernel.c_str(), k.ci.cell.c_str());
        return;
    }
    hip::Function fb = e.have_b ? e.xmod.getFunctionOpt(k.kernel.c_str()) : nullptr;
    bool b_bitwise = false;
    if (fb) {
        const std::vector<float> yb = k.run(fb);
        b_bitwise = yb.size() == ya.size() && std::memcmp(yb.data(), ya.data(), ya.size() * 4) == 0;
        Result rb = b_bitwise ? Result{} : k.verify(yb);
        if (b_bitwise) {
            rb.kind = Kind::invariant;
            rb.pass = true;
            rb.n = ya.size();
            rb.name = "bitwise == A";
        }
        rb.name = "bench --ab " + k.kernel + " " + k.ci.cell + ": " + rb.name;
        const bool ok_b = rb.pass;
        c.rep.add(rb);
        if (!ok_b) {
            std::printf("   --ab %s %s: candidate output check FAILED, A timed alone\n", k.kernel.c_str(), k.ci.cell.c_str());
            fb = nullptr;
        }
    } else if (e.have_b) {
        std::printf("   --ab %s: not in the candidate code object, A timed alone\n", k.kernel.c_str());
    }
    e.bench.beforeCell();
    k.ci.started = wb::iso_now();
    const BlockFn ba = hipBlock(c, [&k, f = k.fa](std::uint64_t r) { k.call(f, r); });
    if (!fb) {
        const CellStats cs = e.bench.run(ba);
        printCell(k.ci, cs);
        if (e.bench.anchorSwitchedOff() && !e.anchor_off_noted) {
            std::printf("   anchor: %s (later cells: no drift control)\n", e.bench.anchorError().c_str());
            e.anchor_off_noted = true;
        }
        if (!cs.ok()) {
            Result r;
            r.name = "bench timing " + k.kernel + " " + k.ci.cell + ": " + cs.error;
            r.pass = false;
            c.rep.add(r);
            return;
        }
        e.write(cellRow(k.ci, cs, e.o.tool_ver));
        return;
    }
    const BlockFn bb = hipBlock(c, [&k, fb](std::uint64_t r) { k.call(fb, r); });
    const AbStats ab = e.bench.runAB(ba, bb);
    if (e.bench.anchorSwitchedOff()) std::printf("   anchor: %s\n", e.bench.anchorError().c_str());
    printCell(k.ci, ab.a, "A ");
    CellInfo cb = k.ci;
    cb.code_hash = e.o.code_hash_b;
    cb.variant = "B";
    printCell(cb, ab.b, "B ");
    std::printf("         B/A %.4f (p10 %.4f p90 %.4f, %d pairs ABBA, order AB %.4f BA %.4f, %s)%s\n", ab.ratio.st.median, ab.ratio.st.p10,
                ab.ratio.st.p90, ab.pairs, ab.order_ab, ab.order_ba, ab.ratio.stop_reason.c_str(), b_bitwise ? " B bitwise == A" : "");
    if (!ab.ratio.ok()) {
        Result r;
        r.name = "bench --ab timing " + k.kernel + " " + k.ci.cell + ": " + ab.ratio.error;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    CellInfo ca = k.ci;
    ca.variant = "A";
    // per-side rows carry the run's anchor / stop reason through the ratio
    CellStats sa = ab.a, sb = ab.b;
    for (CellStats* s : {&sa, &sb}) {
        s->anchor_ref_us = ab.ratio.anchor_ref_us;
        s->anchor_min_us = ab.ratio.anchor_min_us;
        s->anchor_max_us = ab.ratio.anchor_max_us;
        s->anchor_name = ab.ratio.anchor_name;
        s->cooldowns = ab.ratio.cooldowns;
        s->stop_reason = ab.ratio.stop_reason;
    }
    ca.settings += ";ab=abba";
    cb.settings += ";ab=abba";
    e.write(cellRow(ca, sa, e.o.tool_ver));
    wb::Row rb = cellRow(cb, sb, e.o.tool_ver);
    rb.extra.push_back({"b_bitwise", b_bitwise ? "true" : "false"});
    e.write(rb);
    wb::Row rr = abRow(k.ci, ab, e.o.tool_ver, e.o.code_hash_b);
    rr.extra.push_back({"b_bitwise", b_bitwise ? "true" : "false"});
    e.write(rr);
}

// ---------------------------------------------------------------- weights
struct Weights {
    std::string label;  // "swift", "ornith" or "synthetic"
    std::vector<std::pair<std::string, HostMat>> mats;
};

// Dense Swift tensors of one GDN layer (blk.0) and one full-attention layer (blk.3) plus the output head; absent names
// are skipped. No model: synthetic MXFP4 at the two FFN shapes.
Weights denseWeights(Ctx& c, const std::vector<std::string>& names) {
    Weights w;
    const whirl::gguf::File* f = c.open(c.models.mx);
    if (f) {
        w.label = "swift";
        for (const std::string& n : names)
            if (f->tensor(n)) w.mats.push_back({n, loadMat(*f, n)});
        if (!w.mats.empty()) return w;
    }
    w.label = "synthetic";
    w.mats.push_back({"ffn_gate", randomMat(c, QType::mxfp4, 17408, 5120)});
    w.mats.push_back({"ffn_down", randomMat(c, QType::mxfp4, 5120, 17408)});
    return w;
}

std::string shortName(const std::string& t) {
    std::string s = t;
    const std::string suf = ".weight";
    if (s.size() > suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0) s.resize(s.size() - suf.size());
    return s;
}

// copies of a host matrix on the device for rotation
struct DevCopies {
    std::vector<Buf> b;
    int n() const { return static_cast<int>(b.size()); }
    DevPtr at(std::uint64_t r) const { return b[static_cast<std::size_t>(r % b.size())].p(); }
};
DevCopies upload(const Env& e, const std::vector<std::uint8_t>& data) {
    DevCopies d;
    const int n = rotationCopies(data.size(), e.target, e.o.rot_cap);
    for (int i = 0; i < n; ++i) d.b.emplace_back(data);
    return d;
}

// ---------------------------------------------------------------- anchor: one GEMV of a real Swift matrix
// The anchor matrix is ROTATED like the cells (copies past the last-level cache): every anchor launch reads cold
// weights whatever the cell before it did. The first version timed one 47 MB copy: its reference was taken cache-hot in
// the R9700's 64 MB Infinity Cache (33 us = 1.4 TB/s) and every check inside a cell read it cold (~135 us), so every
// cell took 2 cooldowns x 60 s and ended noisy (smoke 2, scratch\mt_p2\smoke_dense_r9700_20261025-1427.jsonl).
std::vector<Buf> g_anchor_w;
std::unique_ptr<Buf> g_anchor_xq, g_anchor_xd, g_anchor_y;

void setAnchor(Env& e) {
    Ctx& c = e.c;
    if (e.o.no_anchor) return;
    const whirl::gguf::File* f = c.open(c.models.mx);
    HostMat m;
    std::string name;
    if (f && f->tensor("blk.0.ffn_down.weight")) {
        m = loadMat(*f, "blk.0.ffn_down.weight");
        name = "gemvq:swift.blk.0.ffn_down";
    } else {
        m = randomMat(c, QType::mxfp4, 5120, 17408);
        name = "gemvq:synthetic.ffn_down";
    }
    const hip::Function fq = c.k.gemvq[static_cast<int>(m.type)];
    if (!fq) {
        std::printf("   anchor: no gemvq kernel for %s, running without an anchor\n", wk::typeSuffix(m.type));
        return;
    }
    const int ncopies = rotationCopies(m.data.size(), e.target, e.o.rot_cap);
    g_anchor_w.clear();
    for (int i = 0; i < ncopies; ++i) g_anchor_w.emplace_back(m.data);
    const std::vector<float> x = c.randn(static_cast<std::size_t>(m.ncols));
    Buf dx(x);
    g_anchor_xq = std::make_unique<Buf>(static_cast<std::size_t>(m.ncols));
    g_anchor_xd = std::make_unique<Buf>(static_cast<std::size_t>(m.ncols) / 32 * 4);
    g_anchor_y = std::make_unique<Buf>(static_cast<std::size_t>(m.nrows) * 4);
    hip::launch(c.k.quantize_q8, {cdiv(m.ncols, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), g_anchor_xq->p(), g_anchor_xd->p(), m.ncols);
    c.sync();
    const int nrows = m.nrows, ncols = m.ncols;
    const std::uint64_t rb = m.row_bytes;
    e.bench.setAnchor(hipBlock(c,
                               [&c, fq, nrows, ncols, rb](std::uint64_t r) {
                                   const DevPtr wp = g_anchor_w[static_cast<std::size_t>(r % g_anchor_w.size())].p();
                                   const wk::GvArgs g = wk::gvSingle(wp, g_anchor_y->p(), rb, nrows);
                                   hip::launch(fq, {cdiv(nrows, 8), 1, 1}, {256, 1, 1}, 0, c.s, g, g_anchor_xq->p(), g_anchor_xd->p(), ncols, 0,
                                               DevPtr{0});
                               }),
                      name);
    if (e.bench.hasAnchor())
        std::printf("   anchor %s: %.2f us per launch (reference, %d rotated copies)\n", name.c_str(), e.bench.anchorRefUs(), ncopies);
    else
        std::printf("   anchor %s failed: %s (running without an anchor)\n", name.c_str(), e.bench.anchorError().c_str());
    name += ".rot" + std::to_string(ncopies);
    e.settings0 += ";anchor=" + name;
}

// ---------------------------------------------------------------- gemv
void benchGemv(Env& e) {
    Ctx& c = e.c;
    c.rep.family = "gemv";
    const Weights w = denseWeights(c, {"blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_out.weight", "blk.0.ffn_gate.weight",
                                       "blk.0.ffn_up.weight", "blk.0.ffn_down.weight", "blk.3.attn_q.weight", "blk.3.attn_k.weight",
                                       "blk.3.attn_v.weight", "blk.3.attn_output.weight", "output.weight"});
    for (const auto& [tn, m] : w.mats) {
        const int ti = static_cast<int>(m.type);
        const std::string kn = std::string("gemvq_") + wk::typeSuffix(m.type);
        if (!c.k.gemvq[ti]) {
            c.rep.skip("gemv", "bench " + kn, "kernel not in this code object");
            continue;
        }
        const int ncols = m.ncols, nrows = m.nrows;
        const std::vector<float> x = c.randn(static_cast<std::size_t>(ncols));
        Buf dx(x), xq(static_cast<std::size_t>(ncols)), xd(static_cast<std::size_t>(ncols) / 32 * 4), y(static_cast<std::size_t>(nrows) * 4);
        hip::launch(c.k.quantize_q8, {cdiv(ncols, 256), 1, 1}, {256, 1, 1}, 0, c.s, dx.p(), xq.p(), xd.p(), ncols);
        c.sync();
        const auto hq = xq.down<std::int8_t>(static_cast<std::size_t>(ncols));
        const auto hd = xd.down<float>(static_cast<std::size_t>(ncols) / 32);
        const DevCopies wc = upload(e, m.data);
        const std::uint64_t rb = m.row_bytes;
        const int crows = std::min(nrows, c.quick ? 64 : 256);
        Cell k;
        k.ci = e.info("gemv", kn, w.label + "." + shortName(tn) + ".n1", shapeStr(nrows, ncols));
        k.ci.settings = e.settings(static_cast<std::uint64_t>(wc.n()) * m.data.size(), wc.n());
        k.ci.bytes = static_cast<double>(m.data.size()) + ncols;
        k.kernel = kn;
        k.fa = c.k.gemvq[ti];
        k.call = [&](hip::Function f, std::uint64_t r) {
            const wk::GvArgs g = wk::gvSingle(wc.at(r), y.p(), rb, nrows);
            hip::launch(f, {cdiv(nrows, 8), 1, 1}, {256, 1, 1}, 0, c.s, g, xq.p(), xd.p(), ncols, 0, DevPtr{0});
        };
        k.run = [&](hip::Function f) {
            y.fill(0xff);
            k.call(f, 0);
            c.sync();
            return y.down<float>(static_cast<std::size_t>(crows));
        };
        k.verify = [&](const std::vector<float>& got) {
            const std::vector<float> wf = ref::dequantRows(m, 0, crows);
            std::vector<double> ry, rs;
            ref::gemvQ8F64(wf, crows, ncols, hq.data(), hd.data(), 1, ry, rs);
            return cmpTol("vs CPU (" + std::to_string(crows) + " rows)", got, ry, rs, 1e-4, 1e-6);
        };
        runCell(e, k);
    }
}

// ---------------------------------------------------------------- gemvh (precise decode)
void benchGemvh(Env& e) {
    Ctx& c = e.c;
    c.rep.family = "gemvh";
    const Weights w = denseWeights(c, {"blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.3.attn_q.weight",
                                       "output.weight"});
    const bool gfx1151 = c.mod.arch() == hip::Arch::gfx1151;
    for (const auto& [tn, m] : w.mats) {
        const int ti = static_cast<int>(m.type);
        const std::string kn = std::string("gemvh_") + wk::typeSuffix(m.type);
        if (!c.k.gemvh[ti]) {
            c.rep.skip("gemvh", "bench " + kn, "kernel not in this code object");
            continue;
        }
        if (m.ncols % 256 != 0) continue;
        const int ncols = m.ncols, nrows = m.nrows;
        const DevCopies wc = upload(e, m.data);
        const std::uint64_t rb = m.row_bytes;
        for (const int nt : {1, 4}) {
            const std::vector<float> x = c.randn(static_cast<std::size_t>(nt) * ncols);
            std::vector<std::uint16_t> xh(x.size());
            std::vector<float> xr(x.size());
            for (std::size_t i = 0; i < x.size(); ++i) {
                xh[i] = f2h(x[i]);
                xr[i] = h2f(xh[i]);
            }
            auto x16 = std::make_shared<Buf>(xh);
            auto y = std::make_shared<Buf>(static_cast<std::size_t>(nt) * nrows * 4);
            const int crows = std::min(nrows, c.quick ? 64 : 256);
            const unsigned gx = gfx1151 ? wk::gemvhGrid(static_cast<unsigned>(nrows)) : cdiv(nrows, 16);
            Cell k;
            k.ci = e.info("gemvh", kn, w.label + "." + shortName(tn) + ".n" + std::to_string(nt), shapeStr(nrows, ncols));
            k.ci.settings = e.settings(static_cast<std::uint64_t>(wc.n()) * m.data.size(), wc.n());
            k.ci.bytes = static_cast<double>(m.data.size()) + 2.0 * nt * ncols;
            k.kernel = kn;
            k.fa = c.k.gemvh[ti];
            k.call = [&c, &wc, x16, y, rb, ncols, nrows, nt, gx](hip::Function f, std::uint64_t r) {
                hip::launch(f, {gx, 1, 1}, {32, 1, 1}, 0, c.s, wc.at(r), rb, x16->p(), y->p(), ncols, nrows, nt, 0);
            };
            k.run = [&c, &k, y, nt, nrows, crows](hip::Function f) {
                y->fill(0xff);
                k.call(f, 0);
                c.sync();
                const auto all = y->down<float>(static_cast<std::size_t>(nt) * nrows);
                std::vector<float> got;
                for (int t = 0; t < nt; ++t)
                    got.insert(got.end(), all.begin() + static_cast<std::ptrdiff_t>(t) * nrows,
                               all.begin() + static_cast<std::ptrdiff_t>(t) * nrows + crows);
                return got;
            };
            k.verify = [&m, xr, crows, ncols, nt](const std::vector<float>& got) {
                const std::vector<float> wf = ref::dequantRows(m, 0, crows);
                std::vector<double> ry, rs;
                ref::gemvF64(wf, crows, ncols, xr.data(), nt, ncols, ry, rs);
                return cmpTol("vs CPU (" + std::to_string(crows) + " rows x " + std::to_string(nt) + " tokens)", got, ry, rs, 1e-4, 1e-6);
            };
            runCell(e, k);
        }
    }
}

// ---------------------------------------------------------------- gemm (prefill)
void benchGemm(Env& e) {
    Ctx& c = e.c;
    c.rep.family = "gemm";
    const Weights w = denseWeights(c, {"blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"});
    const int n = c.quick ? 512 : 2048;
    for (const auto& [tn, m] : w.mats) {
        const int ti = static_cast<int>(m.type);
        const std::string sfx = wk::typeSuffix(m.type);
        const int ncols = m.ncols, nrows = m.nrows;
        if (ncols % 32 != 0) continue;
        const std::vector<float> x = c.randn(static_cast<std::size_t>(n) * ncols);
        std::vector<std::uint16_t> xh(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) xh[i] = f2h(x[i]);
        Buf x16(xh), y(static_cast<std::size_t>(n) * nrows * 4);
        const DevCopies wc = upload(e, m.data);
        const std::uint64_t rb = m.row_bytes;
        const int cr = std::min(nrows, c.quick ? 32 : 64), ct = 8;
        struct K {
            std::string name;
            hip::Function f;
            int bm, bn, nth;
            const DevCopies* wp = nullptr;  // weights (nullptr: the quantized copies wc)
            std::uint64_t rbk = 0;          // their row bytes
        };
        const auto& c0 = wk::kGemmCfgs[0];
        std::vector<K> ks;
        if (c.k.gemmhq[ti]) ks.push_back({"gemmhq_" + sfx, c.k.gemmhq[ti], 128, 256, 256});
        if (c.k.gemmc[0][ti]) ks.push_back({"gemm_c0_" + sfx, c.k.gemmc[0][ti], c0.bm, c0.bn, c0.nth});
        // feat026-gemmc2spill: gemm_c2_f16 on the f16 weights dequant_f16 makes (the 8060S dense prefill choice
        // dequant + gemm_c2_f16 = tune choice G + 2); --ab with a WHIRL_C2SPILL candidate object times the candidate
        const int f16i = static_cast<int>(QType::f16);
        DevCopies w16c;
        const std::uint64_t rb16 = static_cast<std::uint64_t>(ncols) * 2;
        if (m.type != QType::f16 && c.k.gemmc[2][f16i] && wk::kGemmCfgs.size() > 2) {
            std::vector<std::uint8_t> h16(static_cast<std::size_t>(nrows) * rb16);
            for (int r = 0; r < nrows; ++r)
                ref::dequantRowF16(m.type, m.data.data() + static_cast<std::size_t>(r) * m.row_bytes, ncols,
                                   reinterpret_cast<std::uint16_t*>(h16.data() + static_cast<std::size_t>(r) * rb16));
            w16c = upload(e, h16);
            const auto& c2 = wk::kGemmCfgs[2];
            ks.push_back({"gemm_c2_f16", c.k.gemmc[2][f16i], c2.bm, c2.bn, c2.nth, &w16c, rb16});
        }
        if (ks.empty()) c.rep.skip("gemm", "bench gemm " + sfx, "kernel not in this code object");
        for (const K& kk : ks) {
            Cell k;
            k.ci = e.info("gemm", kk.name, w.label + "." + shortName(tn) + ".n" + std::to_string(n), shapeStr(nrows, ncols));
            const DevCopies& wk_ = kk.wp != nullptr ? *kk.wp : wc;
            const std::uint64_t rbk = kk.wp != nullptr ? kk.rbk : rb;
            const std::uint64_t wbytes = kk.wp != nullptr ? static_cast<std::uint64_t>(nrows) * rbk : m.data.size();
            k.ci.settings = e.settings(static_cast<std::uint64_t>(wk_.n()) * wbytes, wk_.n());
            k.ci.flops = 2.0 * n * nrows * ncols;
            k.kernel = kk.name;
            k.fa = kk.f;
            const K g = kk;
            k.call = [&c, &wk_, &x16, &y, g, rbk, ncols, nrows, n](hip::Function f, std::uint64_t r) {
                hip::launch(f, {cdiv(n, g.bn), cdiv(nrows, g.bm), 1}, {static_cast<unsigned>(g.nth), 1, 1}, 0, c.s, wk_.at(r), rbk, x16.p(), y.p(),
                            ncols, nrows, n, 0);
            };
            k.run = [&c, &k, &y, n, nrows, cr, ct](hip::Function f) {
                y.fill(0xff);
                k.call(f, 0);
                c.sync();
                const auto all = y.down<float>(static_cast<std::size_t>(n) * nrows);
                std::vector<float> got;
                for (int t = 0; t < ct; ++t)
                    got.insert(got.end(), all.begin() + static_cast<std::ptrdiff_t>(t) * nrows,
                               all.begin() + static_cast<std::ptrdiff_t>(t) * nrows + cr);
                return got;
            };
            k.verify = [&m, &xh, cr, ct, ncols](const std::vector<float>& got) {
                // the f16 weights the GEMM multiplies (dequant_f16 rounding), f16 activations, double accumulation
                std::vector<float> wf(static_cast<std::size_t>(cr) * ncols);
                std::vector<std::uint16_t> wh(static_cast<std::size_t>(ncols));
                for (int r = 0; r < cr; ++r) {
                    ref::dequantRowF16(m.type, m.data.data() + static_cast<std::size_t>(r) * m.row_bytes, ncols, wh.data());
                    for (int q = 0; q < ncols; ++q) wf[static_cast<std::size_t>(r) * ncols + q] = h2f(wh[static_cast<std::size_t>(q)]);
                }
                std::vector<float> xf(static_cast<std::size_t>(ct) * ncols);
                for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = h2f(xh[i]);
                std::vector<double> ry, rs;
                ref::gemvF64(wf, cr, ncols, xf.data(), ct, ncols, ry, rs);
                return cmpTol("vs CPU (" + std::to_string(cr) + " rows x " + std::to_string(ct) + " tokens)", got, ry, rs, 1e-4, 1e-6);
            };
            runCell(e, k);
        }
    }
}

// ---------------------------------------------------------------- attn (decode, f16 KV)
void benchAttnDecode(Env& e) {
    Ctx& c = e.c;
    c.rep.family = "attn";
    using namespace at;
    const hip::Function fw = c.fnOpt("attn_wsplit1");
    if (!fw || !c.k.attn_combine) {
        c.rep.skip("attn", "bench attn_wsplit1", "kernel not in this code object");
        return;
    }
    const std::vector<int> lens = c.quick ? std::vector<int>{2048, 8192} : std::vector<int>{2048, 32768, 131072};
    // one pool for every L: R disjoint regions of pages(L) pages each (rotation), filled from one random chunk
    std::uint64_t max_pages = 0;
    for (int L : lens) {
        const std::uint64_t pg = static_cast<std::uint64_t>(cdiv(L, wk::kKvPage));
        const int R = rotationCopies(pg * wk::kKvPage * row * 2 * 2, e.target, e.o.rot_cap);
        max_pages = std::max<std::uint64_t>(max_pages, pg * static_cast<std::uint64_t>(R));
    }
    const std::size_t pool_el = static_cast<std::size_t>(max_pages) * wk::kKvPage * row;
    constexpr std::size_t kChunk = std::size_t{1} << 22;  // 4 Mi f16 values (8 MB) of N(0, 1) K and V, tiled over the pool
    std::vector<std::uint16_t> ck(kChunk), cv(kChunk);
    {
        const auto a = c.randn(kChunk), b = c.randn(kChunk);
        for (std::size_t i = 0; i < kChunk; ++i) {
            ck[i] = f2h(a[i]);
            cv[i] = f2h(b[i]);
        }
    }
    Buf pk(pool_el * 2), pv(pool_el * 2);
    for (std::size_t off = 0; off < pool_el; off += kChunk) {
        const std::size_t nn = std::min(kChunk, pool_el - off);
        hip::upload(pk.p() + off * 2, ck.data(), nn * 2);
        hip::upload(pv.p() + off * 2, cv.data(), nn * 2);
    }
    hip::sync();
    std::vector<float> q = c.randn(static_cast<std::size_t>(heads) * qstride);
    for (int h = 0; h < heads; ++h)
        for (int d = 0; d < hd; ++d) q[static_cast<std::size_t>(h) * qstride + hd + d] = 30.0f;  // output gate sigmoid(30) = 1.0f
    Buf dq(q);
    for (const int L : lens) {
        const int pages = static_cast<int>(cdiv(L, wk::kKvPage));
        const std::uint64_t region_bytes = static_cast<std::uint64_t>(pages) * wk::kKvPage * row * 2 * 2;
        const int R = rotationCopies(region_bytes, e.target, e.o.rot_cap);
        std::vector<int> ptab(static_cast<std::size_t>(R) * pages);
        for (int r = 0; r < R; ++r) {
            std::vector<int> perm(static_cast<std::size_t>(pages));
            std::iota(perm.begin(), perm.end(), r * pages);
            std::shuffle(perm.begin(), perm.end(), c.rng);
            std::copy(perm.begin(), perm.end(), ptab.begin() + static_cast<std::ptrdiff_t>(r) * pages);
        }
        auto dpt = std::make_shared<Buf>(ptab);
        auto dpos = std::make_shared<Buf>(std::vector<int>{L - 1});
        const int ns = std::min(wk::kFdMaxSplits, static_cast<int>(cdiv(L, wk::kFdChunk)));
        auto ml = std::make_shared<Buf>(static_cast<std::size_t>(ns) * heads * 2 * 4);
        auto acc = std::make_shared<Buf>(static_cast<std::size_t>(ns) * heads * hd * 4);
        auto out = std::make_shared<Buf>(static_cast<std::size_t>(heads) * hd * 4);
        const DevPtr kp = pk.p(), vp = pv.p(), qp = dq.p();
        const hip::Function fc = c.k.attn_combine;
        Cell k;
        k.ci = e.info("attn", "attn_wsplit1", "swift.f16kv.L" + std::to_string(L), "heads24_kv4_hd256_L" + std::to_string(L));
        k.ci.settings = e.settings(region_bytes * static_cast<std::uint64_t>(R), R) + ";kv=f16;kv_values=randn";
        k.ci.bytes = static_cast<double>(L) * row * 2 * 2;
        k.ci.launches_per_call = 2;
        k.kernel = "attn_wsplit1";
        k.fa = fw;
        k.call = [&c, dpt, dpos, ml, acc, out, kp, vp, qp, fc, pages, ns](hip::Function f, std::uint64_t r) {
            wk::KvArgs a{};
            a.k = kp;
            a.v = vp;
            a.ptab = dpt->p() + (r % (dpt->bytes() / 4 / static_cast<std::size_t>(pages))) * static_cast<std::uint64_t>(pages) * 4;
            wk::AwGroups g;
            g.first[0] = 0;
            g.count[0] = 1;
            hip::launch(f, {nkv, static_cast<unsigned>(ns), 1}, {128, 1, 1}, 0, c.s, qp, a, ml->p(), acc->p(), heads, nkv, qstride, dpos->p(), scale,
                        DevPtr{0}, g);
            hip::launch(fc, {heads, 1, 1}, {256, 1, 1}, 0, c.s, ml->p(), acc->p(), qp, out->p(), heads, hd, qstride, ns);
        };
        k.run = [&c, &k, out](hip::Function f) {
            out->fill(0xff);
            k.call(f, 0);
            c.sync();
            return out->down<float>(static_cast<std::size_t>(heads) * hd);
        };
        // CPU softmax attention (double) of 2 heads (different KV heads) over region 0
        k.verify = [&ck, &cv, &q, ptab, L](const std::vector<float>& got) {
            std::vector<float> g;
            std::vector<double> ro, rs;
            for (const int h : {0, 13}) {
                const int kvh = h / (heads / nkv);
                const float* qv = q.data() + static_cast<std::size_t>(h) * qstride;
                std::vector<double> s(static_cast<std::size_t>(L));
                double mx = -INFINITY;
                auto el = [&](int p, int i) {
                    const std::size_t prow = static_cast<std::size_t>(ptab[static_cast<std::size_t>(p / wk::kKvPage)]) * wk::kKvPage + p % wk::kKvPage;
                    return ((prow * nkv + kvh) * hd + i) % kChunk;
                };
                for (int p = 0; p < L; ++p) {
                    double d = 0;
                    for (int i = 0; i < hd; ++i) d += static_cast<double>(qv[i]) * h2f(ck[el(p, i)]);
                    s[static_cast<std::size_t>(p)] = d * scale;
                    mx = std::max(mx, s[static_cast<std::size_t>(p)]);
                }
                double l = 0;
                for (auto& x : s) {
                    x = std::exp(x - mx);
                    l += x;
                }
                std::vector<double> o(hd, 0.0), a(hd, 0.0);
                for (int p = 0; p < L; ++p)
                    for (int d = 0; d < hd; ++d) {
                        const double vv = h2f(cv[el(p, d)]);
                        o[static_cast<std::size_t>(d)] += s[static_cast<std::size_t>(p)] * vv;
                        a[static_cast<std::size_t>(d)] += s[static_cast<std::size_t>(p)] * std::fabs(vv);
                    }
                for (int d = 0; d < hd; ++d) {
                    g.push_back(got[static_cast<std::size_t>(h) * hd + d]);
                    ro.push_back(o[static_cast<std::size_t>(d)] / l);
                    rs.push_back(a[static_cast<std::size_t>(d)] / l);
                }
            }
            return cmpTol("attn_wsplit1 + attn_combine vs CPU (heads 0, 13)", g, ro, rs, 1e-2, 1e-4);
        };
        runCell(e, k);
    }
}

// ---------------------------------------------------------------- moe (decode, Ornith MXFP4 experts)
void benchMoe(Env& e) {
    Ctx& c = e.c;
    c.rep.family = "moe";
    const int ti = static_cast<int>(QType::mxfp4);
    if (!c.k.moe_gu[ti] || !c.k.moe_down[ti] || !c.k.silu_mul_q8) {
        c.rep.skip("moe", "bench moe_gu_mxfp4", "kernel not in this code object");
        return;
    }
    struct Layer {
        HostMat gate, up, down;
    };
    std::vector<Layer> layers;
    int E = 2048, R = 64, K = 8, F = 512;
    std::string label = "synthetic";
    const whirl::gguf::File* f = c.open(c.models.moemx);
    const int nl = std::max(1, e.o.moe_layers);
    if (f && f->tensor("blk.0.ffn_gate_exps.weight") && f->tensor("blk.0.ffn_gate_exps.weight")->type == whirl::gguf::GgmlType::mxfp4) {
        label = "ornith";
        for (int l = 0; l < nl; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            if (!f->tensor(p + "ffn_gate_exps.weight")) break;
            layers.push_back({loadMat(*f, p + "ffn_gate_exps.weight"), loadMat(*f, p + "ffn_up_exps.weight"), loadMat(*f, p + "ffn_down_exps.weight")});
        }
        E = layers[0].gate.ncols;
        F = layers[0].down.ncols;
        R = layers[0].gate.nrows / F;
        K = static_cast<int>(f->getUintOr("qwen35moe.expert_used_count", 8));
    } else {
        for (int l = 0; l < nl; ++l)
            layers.push_back({randomMat(c, QType::mxfp4, R * F, E), randomMat(c, QType::mxfp4, R * F, E), randomMat(c, QType::mxfp4, R * E, F)});
    }
    struct DLayer {
        Buf g, u, d;
    };
    std::vector<DLayer> dl;
    std::uint64_t wbytes = 0;
    for (const Layer& L : layers) {
        dl.push_back({Buf(L.gate.data), Buf(L.up.data), Buf(L.down.data)});
        wbytes += L.gate.data.size() + L.up.data.size() + L.down.data.size();
    }
    // routing sets: S sets of K distinct experts (uniform), so consecutive launches read different experts
    const int S = 64;
    std::vector<int> ids(static_cast<std::size_t>(S) * K);
    {
        std::vector<int> all(static_cast<std::size_t>(R));
        std::iota(all.begin(), all.end(), 0);
        for (int s = 0; s < S; ++s) {
            std::shuffle(all.begin(), all.end(), c.rng);
            std::copy(all.begin(), all.begin() + K, ids.begin() + static_cast<std::ptrdiff_t>(s) * K);
        }
    }
    std::vector<float> wts(static_cast<std::size_t>(K), 1.0f / K);
    auto dids = std::make_shared<Buf>(ids);
    auto dw = std::make_shared<Buf>(wts);
    auto dsg = std::make_shared<Buf>(std::vector<float>{0.5f});
    const std::vector<float> h = c.randn(static_cast<std::size_t>(E));
    const std::vector<float> ysh = c.randn(static_cast<std::size_t>(E));
    const std::vector<float> x0 = c.randn(static_cast<std::size_t>(E));
    Buf dh(h);
    auto dysh = std::make_shared<Buf>(ysh);
    auto xq = std::make_shared<Buf>(static_cast<std::size_t>(E)), xd = std::make_shared<Buf>(static_cast<std::size_t>(E) / 32 * 4);
    hip::launch(c.k.quantize_q8, {cdiv(E, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), xq->p(), xd->p(), E);
    c.sync();
    const auto hq = xq->down<std::int8_t>(static_cast<std::size_t>(E));
    const auto hd = xd->down<float>(static_cast<std::size_t>(E) / 32);
    const std::size_t ng = static_cast<std::size_t>(K) * F;
    auto yg = std::make_shared<Buf>(ng * 4), yu = std::make_shared<Buf>(ng * 4);
    const bool mxw = c.k.moe_gu_mxw && c.k.moe_down_mxw;
    const hip::Function fgu = mxw ? c.k.moe_gu_mxw : c.k.moe_gu[ti];
    const hip::Function fdn = mxw ? c.k.moe_down_mxw : c.k.moe_down[ti];
    const std::string ngu = mxw ? "moe_gu_mxfp4w" : "moe_gu_mxfp4", ndn = mxw ? "moe_down_mxfp4w" : "moe_down_mxfp4";
    const std::uint64_t grb = layers[0].gate.row_bytes, drb = layers[0].down.row_bytes;
    const int nlay = static_cast<int>(dl.size());
    auto lay = [nlay](std::uint64_t r) { return static_cast<std::size_t>(r % static_cast<std::uint64_t>(nlay)); };
    auto set = [nlay, S](std::uint64_t r) { return static_cast<std::uint64_t>((r / static_cast<std::uint64_t>(nlay)) % static_cast<std::uint64_t>(S)); };
    const std::string settings = e.settings(wbytes, nlay) + ";routing_sets=" + std::to_string(S) + ";routing=uniform";
    // dequantized row of expert e of a [R * rows][cols] matrix
    auto dot = [](const HostMat& m, std::size_t rrow, const std::vector<double>& x, double& y, double& a) {
        std::vector<float> w(static_cast<std::size_t>(m.ncols));
        ref::dequantRow(m.type, m.data.data() + rrow * m.row_bytes, m.ncols, w.data());
        y = 0;
        a = 0;
        for (int i = 0; i < m.ncols; ++i) {
            const double p = w[static_cast<std::size_t>(i)] * x[static_cast<std::size_t>(i)];
            y += p;
            a += std::fabs(p);
        }
    };
    // gate + up
    {
        Cell k;
        k.ci = e.info("moe", ngu, label + ".gate_up.n1.k" + std::to_string(K), "E" + std::to_string(E) + "_F" + std::to_string(F) + "_R" + std::to_string(R));
        k.ci.settings = settings;
        k.ci.bytes = static_cast<double>(K) * 2.0 * F * static_cast<double>(grb);
        k.kernel = ngu;
        k.fa = fgu;
        k.call = [&c, &dl, dids, xq, xd, yg, yu, lay, set, grb, F, E, K](hip::Function fn, std::uint64_t r) {
            const DLayer& L = dl[lay(r)];
            hip::launch(fn, {cdiv(2 * F, 8), static_cast<unsigned>(K), 1}, {256, 1, 1}, 0, c.s, L.g.p(), L.u.p(), grb, F, 2 * F, xq->p(), xd->p(),
                        dids->p() + set(r) * static_cast<std::uint64_t>(K) * 4, yg->p(), yu->p(), E, K, DevPtr{0});
        };
        k.run = [&c, &k, yg, yu, ng](hip::Function fn) {
            yg->fill(0xff);
            yu->fill(0xff);
            k.call(fn, 0);
            c.sync();
            std::vector<float> o = yg->down<float>(ng);
            const auto u = yu->down<float>(ng);
            o.insert(o.end(), u.begin(), u.end());
            return o;
        };
        k.verify = [&layers, &ids, &hq, &hd, &dot, F, E, K, ng](const std::vector<float>& got) {
            std::vector<double> x(static_cast<std::size_t>(E));
            for (int i = 0; i < E; ++i) x[static_cast<std::size_t>(i)] = static_cast<double>(hq[static_cast<std::size_t>(i)]) * ref::xdScale(hd[static_cast<std::size_t>(i) / 32]);
            std::vector<float> g;
            std::vector<double> ro, rs;
            for (int kk = 0; kk < K; ++kk)
                for (int r = 0; r < F; r += 7) {
                    const std::size_t row = static_cast<std::size_t>(ids[static_cast<std::size_t>(kk)]) * F + r;
                    double y, a;
                    dot(layers[0].gate, row, x, y, a);
                    g.push_back(got[static_cast<std::size_t>(kk) * F + r]);
                    ro.push_back(y);
                    rs.push_back(a);
                    dot(layers[0].up, row, x, y, a);
                    g.push_back(got[ng + static_cast<std::size_t>(kk) * F + r]);
                    ro.push_back(y);
                    rs.push_back(a);
                }
            return cmpTol("gate / up vs CPU (every 7th row of the K experts)", g, ro, rs, 1e-4, 1e-6);
        };
        runCell(e, k);
    }
    // down (+ routing weights, shared expert, residual)
    {
        const std::vector<float> act = c.randn(ng, 0.5f);
        Buf dact(act);
        auto aq = std::make_shared<Buf>(ng), ad = std::make_shared<Buf>(ng / 32 * 4);
        hip::launch(c.k.quantize_q8, {cdiv(ng, 256), 1, 1}, {256, 1, 1}, 0, c.s, dact.p(), aq->p(), ad->p(), static_cast<int>(ng));
        c.sync();
        const auto aqh = aq->down<std::int8_t>(ng);
        const auto adh = ad->down<float>(ng / 32);
        auto dx = std::make_shared<Buf>(x0);
        Cell k;
        k.ci = e.info("moe", ndn, label + ".down.n1.k" + std::to_string(K), "E" + std::to_string(E) + "_F" + std::to_string(F) + "_R" + std::to_string(R));
        k.ci.settings = settings;
        k.ci.bytes = static_cast<double>(K) * E * static_cast<double>(drb);
        k.kernel = ndn;
        k.fa = fdn;
        k.call = [&c, &dl, dids, dw, dsg, dysh, aq, ad, dx, lay, set, drb, F, E, K](hip::Function fn, std::uint64_t r) {
            const DLayer& L = dl[lay(r)];
            hip::launch(fn, {cdiv(E, 8), 1, 1}, {256, 1, 1}, 0, c.s, L.d.p(), drb, E, aq->p(), ad->p(), dids->p() + set(r) * static_cast<std::uint64_t>(K) * 4,
                        dw->p(), dsg->p(), dysh->p(), dx->p(), F, K, DevPtr{0});
        };
        k.run = [&c, &k, dx, &x0, E](hip::Function fn) {
            dx->up(x0);  // the kernel adds onto x (residual): start from x0 for the check
            k.call(fn, 0);
            c.sync();
            return dx->down<float>(static_cast<std::size_t>(E));
        };
        k.verify = [&layers, &ids, &wts, &ysh, &x0, &dot, aqh, adh, F, E, K](const std::vector<float>& got) {
            std::vector<float> g;
            std::vector<double> ro, rs;
            for (int r = 0; r < E; r += 5) {
                double acc = 0, a = 0;
                for (int kk = 0; kk < K; ++kk) {
                    std::vector<double> x(static_cast<std::size_t>(F));
                    for (int i = 0; i < F; ++i) {
                        const std::size_t o = static_cast<std::size_t>(kk) * F + i;
                        x[static_cast<std::size_t>(i)] = static_cast<double>(aqh[o]) * ref::xdScale(adh[o / 32]);
                    }
                    double y, aa;
                    dot(layers[0].down, static_cast<std::size_t>(ids[static_cast<std::size_t>(kk)]) * E + r, x, y, aa);
                    acc += wts[static_cast<std::size_t>(kk)] * y;
                    a += wts[static_cast<std::size_t>(kk)] * aa;
                }
                const double shv = 0.5 * ysh[static_cast<std::size_t>(r)];
                g.push_back(got[static_cast<std::size_t>(r)]);
                ro.push_back(x0[static_cast<std::size_t>(r)] + acc + shv);
                rs.push_back(std::fabs(x0[static_cast<std::size_t>(r)]) + a + std::fabs(shv));
            }
            return cmpTol("down (+ weights, shared, residual) vs CPU (every 5th row)", g, ro, rs, 1e-4, 1e-6);
        };
        runCell(e, k);
    }
}

// ---------------------------------------------------------------- gdn (decode step + gated norm)
void benchGdn(Env& e) {
    Ctx& c = e.c;
    c.rep.family = "gdn";
    using namespace gd;
    if (!c.k.gdn_step_norm) {
        c.rep.skip("gdn", "bench gdn_step_norm", "kernel not in this code object");
        return;
    }
    // inputs of one decode row: unit-norm q / k heads, v ~ N(0, 1); g in [-1.5, -0.01], beta in [0.05, 0.95]
    std::vector<float> qkv = c.randn(static_cast<std::size_t>(ch));
    for (int hh = 0; hh < 2 * nk; ++hh) {
        float* p = qkv.data() + static_cast<std::size_t>(hh) * dk;
        double s = 0;
        for (int i = 0; i < dk; ++i) s += static_cast<double>(p[i]) * p[i];
        const float r = static_cast<float>(1.0 / std::sqrt(s));
        for (int i = 0; i < dk; ++i) p[i] *= r;
    }
    const std::vector<float> g = c.randu(nv, -1.5f, -0.01f), beta = c.randu(nv, 0.05f, 0.95f);
    const std::vector<float> z = c.randn(static_cast<std::size_t>(dinner)), wn = c.randu(dv, 0.5f, 1.5f);
    const std::vector<float> state0 = c.randn(static_cast<std::size_t>(nv) * dk * dv, 0.3f);
    auto dq = std::make_shared<Buf>(qkv), dg = std::make_shared<Buf>(g), db = std::make_shared<Buf>(beta), dz = std::make_shared<Buf>(z),
         dwn = std::make_shared<Buf>(wn);
    auto out = std::make_shared<Buf>(static_cast<std::size_t>(dinner) * 4), xq = std::make_shared<Buf>(static_cast<std::size_t>(dinner)),
         xd = std::make_shared<Buf>(static_cast<std::size_t>(dinner) / 32 * 4);
    const std::uint64_t sbytes = state0.size() * 4;
    const int R = rotationCopies(sbytes, e.target, e.o.rot_cap);
    auto states = std::make_shared<std::vector<Buf>>();
    for (int i = 0; i < R; ++i) states->emplace_back(state0);
    Cell k;
    k.ci = e.info("gdn", "gdn_step_norm", "swift.decode.n1", "nk16_nv48_dk128_dv128");
    k.ci.settings = e.settings(sbytes * static_cast<std::uint64_t>(R), R);
    k.ci.bytes = 2.0 * static_cast<double>(sbytes);
    k.kernel = "gdn_step_norm";
    k.fa = c.k.gdn_step_norm;
    k.call = [&c, dq, dg, db, dz, dwn, out, xq, xd, states](hip::Function f, std::uint64_t r) {
        wk::GdnSegs s;
        s.s[0].state = (*states)[static_cast<std::size_t>(r % states->size())].p();
        s.s[0].row0 = 0;
        s.s[0].nrows = 1;
        hip::launch(f, {nv, 1, 1}, {512, 1, 1}, 0, c.s, dq->p(), dg->p(), db->p(), dz->p(), dwn->p(), out->p(), xq->p(), xd->p(), nk, scale, eps, nv,
                    ch, s);
    };
    k.run = [&c, &k, out, states, &state0](hip::Function f) {
        (*states)[0].up(state0);
        out->fill(0xff);
        k.call(f, 0);
        c.sync();
        std::vector<float> o = out->down<float>(static_cast<std::size_t>(dinner));
        const auto st = (*states)[0].down<float>(state0.size());
        o.insert(o.end(), st.begin(), st.end());
        return o;
    };
    k.verify = [&qkv, &g, &beta, &z, &wn, &state0](const std::vector<float>& got) {
        // delta rule, one row, in double; then gated RMSNorm with silu(z)
        std::vector<double> S(state0.begin(), state0.end()), o(static_cast<std::size_t>(dinner));
        for (int h = 0; h < nv; ++h) {
            const int kh = h % nk;
            const float* q = qkv.data() + static_cast<std::size_t>(kh) * dk;
            const float* kk = qkv.data() + static_cast<std::size_t>(nk + kh) * dk;
            const float* v = qkv.data() + static_cast<std::size_t>(2 * nk) * dk + static_cast<std::size_t>(h) * dv;
            double* Sh = S.data() + static_cast<std::size_t>(h) * dk * dv;
            const double decay = std::exp(static_cast<double>(g[static_cast<std::size_t>(h)])), b = beta[static_cast<std::size_t>(h)];
            for (int i = 0; i < dk * dv; ++i) Sh[i] *= decay;
            for (int j = 0; j < dv; ++j) {
                double sk = 0;
                for (int i = 0; i < dk; ++i) sk += Sh[i * dv + j] * kk[i];
                const double delta = (v[j] - sk) * b;
                double acc = 0;
                for (int i = 0; i < dk; ++i) {
                    Sh[i * dv + j] += kk[i] * delta;
                    acc += Sh[i * dv + j] * q[i];
                }
                o[static_cast<std::size_t>(h) * dv + j] = acc * scale;
            }
        }
        std::vector<double> ref(static_cast<std::size_t>(dinner) + S.size());
        for (int h = 0; h < nv; ++h) {
            const std::size_t b0 = static_cast<std::size_t>(h) * dv;
            double ss = 0;
            for (int j = 0; j < dv; ++j) ss += o[b0 + j] * o[b0 + j];
            const double r = 1.0 / std::sqrt(ss / dv + eps);
            for (int j = 0; j < dv; ++j) {
                const double zz = z[b0 + j];
                ref[b0 + j] = o[b0 + j] * r * wn[static_cast<std::size_t>(j)] * (zz / (1.0 + std::exp(-zz)));
            }
        }
        std::copy(S.begin(), S.end(), ref.begin() + dinner);
        // out: 1e-5 + 1e-4 |ref| (test_gdn cmpTolRel "gdn_step_norm out vs CPU"); state: 1e-4 of the state's rms (tolRms)
        std::vector<double> sc(ref.size());
        double rms = 0;
        for (std::size_t i = dinner; i < ref.size(); ++i) rms += ref[i] * ref[i];
        rms = std::sqrt(rms / static_cast<double>(S.size())) + 1e-30;
        for (std::size_t i = 0; i < ref.size(); ++i) sc[i] = i < static_cast<std::size_t>(dinner) ? std::fabs(ref[i]) : rms;
        return cmpTol("out + state vs CPU delta rule + gated norm", got, ref, sc, 1e-4, 1e-5);
    };
    runCell(e, k);
}

}  // namespace

const char* const kBenchFamilies[] = {"gemv", "gemvh", "gemm", "attn", "moe", "gdn"};

bool benchFamilyKnown(const std::string& f) {
    for (const char* n : kBenchFamilies)
        if (f == n) return true;
    return false;
}

void runBench(Ctx& c, const BenchOpts& o) {
    Env e(c, o);
    if (!o.jsonl.empty() && !e.out.open(o.jsonl)) {
        Result r;
        r.family = "bench";
        r.name = "cannot open --jsonl " + o.jsonl;
        r.pass = false;
        c.rep.add(r);
        return;
    }
    e.target = o.rot_target ? o.rot_target : rotationTarget(o.gpu_arch);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "state=%s;burst_idle=%g;max_cell_s=%g;rel=%g;min_block_us=%g;warmup_ms=%g", o.state.c_str(), o.cfg.burst_idle_s,
                  o.cfg.max_cell_s, o.cfg.rel_target, o.cfg.min_block_us, o.cfg.warmup_ms);
    e.settings0 = buf;
    if (!o.ab_hsaco.empty()) {
        std::FILE* fp = std::fopen(o.ab_hsaco.c_str(), "rb");
        if (fp) {
            std::fseek(fp, 0, SEEK_END);
            e.ximage.resize(static_cast<std::size_t>(std::ftell(fp)));
            std::fseek(fp, 0, SEEK_SET);
            if (std::fread(e.ximage.data(), 1, e.ximage.size(), fp) != e.ximage.size()) e.ximage.clear();
            std::fclose(fp);
        }
        if (e.ximage.empty()) {
            Result r;
            r.family = "bench";
            r.name = "--ab: cannot read " + o.ab_hsaco;
            r.pass = false;
            c.rep.add(r);
            return;
        }
        e.xmod = hip::Module::loadData(e.ximage.data(), c.mod.arch());
        e.have_b = true;
        std::printf("   --ab candidate %s (sha256 %s)\n", o.ab_hsaco.c_str(), o.code_hash_b.c_str());
    }
    std::printf("   rotation target %.0f MB, cap %.0f MB; %s\n", e.target / 1048576.0, o.rot_cap / 1048576.0, e.settings0.c_str());
    setAnchor(e);
    struct F {
        const char* name;
        void (*fn)(Env&);
    };
    const F fams[] = {{"gemv", benchGemv}, {"gemvh", benchGemvh}, {"gemm", benchGemm}, {"attn", benchAttnDecode}, {"moe", benchMoe}, {"gdn", benchGdn}};
    for (const F& f : fams) {
        if (!o.families.count(f.name)) continue;
        std::printf("== bench %s\n", f.name);
        std::fflush(stdout);
        try {
            f.fn(e);
        } catch (const std::exception& ex) {
            Result r;
            r.family = f.name;
            r.name = std::string("bench exception: ") + ex.what();
            r.pass = false;
            c.rep.add(r);
        }
    }
    std::printf("bench: %d rows written%s, %d burst idles, %d groups past %g s\n", e.rows_written,
                o.jsonl.empty() ? " (no --jsonl: console only)" : (" to " + o.jsonl).c_str(), e.bench.bursts(), e.bench.overruns(), o.cfg.burst_s);
    std::printf("bench timer: %lld blocks, %lld retried, %lld invalid after the retry%s%s\n", static_cast<long long>(g_timer->blocks),
                static_cast<long long>(g_timer->retries), static_cast<long long>(g_timer->invalid), g_timer->last_invalid.empty() ? "" : "; last: ",
                g_timer->last_invalid.c_str());
    g_anchor_w.clear();
    g_anchor_xq.reset();
    g_anchor_xd.reset();
    g_anchor_y.reset();
}

}  // namespace kt
