// whirl-kernel-test: paged KV cache and attention.
//   * kv_store / kv_store_q8 / kv_store_q8v exact (CPU emulation of the f16
//     store and of the q8 quantizer), single-sequence and batched (kvbase);
//   * attn_decode, attn_split + attn_combine, attn_wsplit1 / attn_wsplit2 + attn_combine,
//     attn_prefill_wmma vs a double-precision CPU softmax attention over the
//     cache contents (tolerance; f16 WMMA paths looser);
//   * bitwise invariances: attn_wsplit1 / attn_wsplit2 grouped == per-query (the
//     prototype's checkAttnGroups), attn_kx and attn_kg (f16, q8, q8v) ==
//     attn_prefill_wmma, head-range
//     split launches == one launch, attn_combine_q8 == attn_combine (+ its
//     int8 copy == quantize_q8), attn_prep == rmsnorm + rope_neox + kv_store.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>

#include "cpu_ref.h"

namespace kt {

namespace {

constexpr int kHeads = 24, kKv = 4, kHd = 256, kRot = 64, kGrp = kHeads / kKv;
constexpr int kQStride = 2 * kHd;  // q then gate
constexpr int kRow = kKv * kHd;
constexpr int kPages = 8, kMaxCtx = 4096;
const float kScale = 1.0f / 16.0f;  // 1 / sqrt(256)

enum class Fmt { f16, q8, q8v };
const char* fmtSuffix(Fmt f) { return f == Fmt::f16 ? "" : (f == Fmt::q8 ? "_q8" : "_q8v"); }
bool kQ8(Fmt f) { return f == Fmt::q8; }
bool vQ8(Fmt f) { return f != Fmt::f16; }

struct Pool {
    Fmt fmt;
    Buf k, v, ks, vs, ptab;
    std::vector<int> ptab_h;
    Pool(Ctx& c, Fmt f) : fmt(f) {
        const std::size_t rows = static_cast<std::size_t>(kPages) * wk::kKvPage;
        k = Buf(rows * kRow * (kQ8(f) ? 1 : 2));
        v = Buf(rows * kRow * (vQ8(f) ? 1 : 2));
        ks = Buf(rows * kRow / 32 * 2);
        vs = Buf(rows * kRow / 32 * 2);
        ptab_h.resize(kPages);
        std::iota(ptab_h.begin(), ptab_h.end(), 0);
        std::shuffle(ptab_h.begin(), ptab_h.end(), c.rng);
        ptab = Buf(ptab_h);
        k.zero();
        v.zero();
        ks.zero();
        vs.zero();
    }
    wk::KvArgs args(DevPtr kvbase = 0, int tab0 = 0) const {
        wk::KvArgs a;
        a.k = k.p();
        a.v = v.p();
        a.ks = (kQ8(fmt) ? ks.p() : 0);
        a.vs = (vQ8(fmt) ? vs.p() : 0);
        a.ptab = ptab.p();
        a.kvbase = kvbase;
        a.tab0 = tab0;
        return a;
    }
    int prow(int p) const { return ptab_h[static_cast<std::size_t>(p / wk::kKvPage)] * wk::kKvPage + p % wk::kKvPage; }
};

// q8 store of one 32-value group as the kernels do it.
void q8Group(const float* x, std::int8_t* q, std::uint16_t& s) {
    float amax = 0;
    for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
    // (_Float16)(amax * (1.f / 127.f)) compiles to v_fma_mixlo_f16: the exact product rounded once to f16
    s = d2h(static_cast<double>(amax) * static_cast<double>(1.f / 127.f));
    const float sf = h2f(s);
    for (int i = 0; i < 32; ++i) {
        int qv = sf > 0.f ? static_cast<int>(std::nearbyint(x[i] / sf)) : 0;
        q[i] = static_cast<std::int8_t>(std::max(-127, std::min(127, qv)));
    }
}

// Host image of the cache after storing rows of k / v at pool rows.
struct HostKv {
    std::vector<std::uint16_t> k16, v16, ks, vs;
    std::vector<std::int8_t> k8, v8;
};

HostKv download(const Pool& p) {
    const std::size_t rows = static_cast<std::size_t>(kPages) * wk::kKvPage, n = rows * kRow;
    HostKv h;
    if (kQ8(p.fmt)) {
        h.k8 = p.k.down<std::int8_t>(n);
        h.ks = p.ks.down<std::uint16_t>(n / 32);
    } else {
        h.k16 = p.k.down<std::uint16_t>(n);
    }
    if (vQ8(p.fmt)) {
        h.v8 = p.v.down<std::int8_t>(n);
        h.vs = p.vs.down<std::uint16_t>(n / 32);
    } else {
        h.v16 = p.v.down<std::uint16_t>(n);
    }
    return h;
}

// Value the kernels read for element e (f16, or f16(q * s)).
double kvVal(const std::vector<std::uint16_t>& f16, const std::vector<std::int8_t>& q8, const std::vector<std::uint16_t>& sc,
             std::size_t e) {
    if (!f16.empty()) return h2f(f16[e]);
    return h2f(d2h(static_cast<double>(q8[e]) * h2f(sc[e >> 5])));
}

// CPU softmax attention of query rows (positions qpos[t]) against keys 0..qpos[t].
// win_lo > 0: MTP draft window, keys [256, win_lo) are not attended (sink 256 + window from win_lo)
void attnRef(const HostKv& h, const Pool& p, const std::vector<float>& q, const std::vector<int>& qpos,
             const std::vector<int>& rows, std::vector<double>& out, std::vector<double>& scale, int win_lo = 0) {
    const int n = static_cast<int>(rows.size());
    out.assign(static_cast<std::size_t>(n) * kHeads * kHd, 0.0);
    scale.assign(out.size(), 0.0);
    for (int ri = 0; ri < n; ++ri) {
        const int t = rows[static_cast<std::size_t>(ri)];
        const int np = qpos[static_cast<std::size_t>(t)] + 1;
        for (int hh = 0; hh < kHeads; ++hh) {
            const int kvh = hh / kGrp;
            const float* qv = q.data() + (static_cast<std::size_t>(t) * kHeads + hh) * kQStride;
            std::vector<double> s(static_cast<std::size_t>(np));
            double m = -INFINITY;
            for (int pp = 0; pp < np; ++pp) {
                const std::size_t base = (static_cast<std::size_t>(p.prow(pp)) * kKv + kvh) * kHd;
                double d = 0;
                for (int i = 0; i < kHd; ++i) d += static_cast<double>(qv[i]) * kvVal(h.k16, h.k8, h.ks, base + i);
                s[static_cast<std::size_t>(pp)] = pp >= 256 && pp < win_lo ? -INFINITY : d * kScale;
                m = std::max(m, s[static_cast<std::size_t>(pp)]);
            }
            double l = 0;
            for (auto& x : s) {
                x = std::exp(x - m);
                l += x;
            }
            for (int dd = 0; dd < kHd; ++dd) {
                double o = 0, a = 0;
                for (int pp = 0; pp < np; ++pp) {
                    const double vv =
                        kvVal(h.v16, h.v8, h.vs, (static_cast<std::size_t>(p.prow(pp)) * kKv + kvh) * kHd + dd);
                    o += s[static_cast<std::size_t>(pp)] * vv;
                    a += s[static_cast<std::size_t>(pp)] * std::fabs(vv);
                }
                const double g = 1.0 / (1.0 + std::exp(-static_cast<double>(qv[kHd + dd])));
                const std::size_t oi = (static_cast<std::size_t>(ri) * kHeads + hh) * kHd + dd;
                out[oi] = o / l * g;
                scale[oi] = a / l * g;
            }
        }
    }
}

std::vector<float> pickRows(const std::vector<float>& all, const std::vector<int>& rows) {
    std::vector<float> o;
    for (int t : rows)
        o.insert(o.end(), all.begin() + static_cast<std::ptrdiff_t>(t) * kHeads * kHd,
                 all.begin() + static_cast<std::ptrdiff_t>(t + 1) * kHeads * kHd);
    return o;
}

int nSplit() { return std::min(wk::kFdMaxSplits, (kMaxCtx + wk::kFdChunk - 1) / wk::kFdChunk); }

}  // namespace

void testAttn(Ctx& c) {
    c.rep.family = "attn";
    const int L = c.quick ? 600 : 1100;  // cached positions (crosses page boundaries)
    for (Fmt fmt : {Fmt::f16, Fmt::q8, Fmt::q8v}) {
        const std::string fs = fmtSuffix(fmt);
        const std::string tag = std::string(" [kv ") + (fmt == Fmt::f16 ? "f16" : (fmt == Fmt::q8 ? "q8" : "q8v")) + "]";
        if (fmt == Fmt::q8v && !c.k.caps.kv_q8v) {  // gfx1151: no q8v kernels
            c.rep.skip("attn", "kv q8v (kv_store / attn_decode / attn_split / attn_prefill_wmma)", "kernel not in this code object");
            continue;
        }
        Pool pool(c, fmt);
        // ---- kv_store of L rows (single sequence, positions 0..L-1)
        const std::vector<float> kin = c.randn(static_cast<std::size_t>(L) * kRow, 0.7f);
        const std::vector<float> vin = c.randn(static_cast<std::size_t>(L) * kRow, 0.7f);
        {
            Buf dk(kin), dv(vin), pos(std::vector<int>{0});
            hip::launch(c.fn("kv_store" + fs), {static_cast<unsigned>(L), 1, 1}, {256, 1, 1}, 0, c.s, dk.p(), dv.p(),
                        pool.args(), pos.p(), kRow);
            c.sync();
        }
        const HostKv h = download(pool);
        {
            std::size_t n = 0, bad = 0;
            for (int pp = 0; pp < L; ++pp) {
                const std::size_t base = static_cast<std::size_t>(pool.prow(pp)) * kRow;
                for (int g = 0; g < kRow / 32; ++g) {
                    const float* kx = kin.data() + static_cast<std::size_t>(pp) * kRow + 32 * g;
                    const float* vx = vin.data() + static_cast<std::size_t>(pp) * kRow + 32 * g;
                    std::int8_t q[32];
                    std::uint16_t s;
                    for (int which = 0; which < 2; ++which) {
                        const float* x = which ? vx : kx;
                        const bool q8 = which ? vQ8(fmt) : kQ8(fmt);
                        const auto& h16 = which ? h.v16 : h.k16;
                        const auto& h8 = which ? h.v8 : h.k8;
                        const auto& hs = which ? h.vs : h.ks;
                        if (q8) {
                            q8Group(x, q, s);
                            bad += hs[(base >> 5) + g] != s;
                            for (int i = 0; i < 32; ++i) bad += h8[base + 32 * g + i] != q[i];
                            n += 33;
                        } else {
                            for (int i = 0; i < 32; ++i) bad += h16[base + 32 * g + i] != f2h(x[i]);
                            n += 32;
                        }
                    }
                }
            }
            Result r;
            r.name = "kv_store" + fs + " (" + std::to_string(L) + " rows, paged)" + tag;
            r.n = n;
            r.mismatches = bad;
            r.pass = bad == 0;
            c.rep.add(r);
        }

        // queries (q then gate per head)
        const int nq = 3;
        const std::vector<float> q = c.randn(static_cast<std::size_t>(16) * kHeads * kQStride);
        Buf dq(q);
        std::vector<int> qpos(16);
        for (int t = 0; t < 16; ++t) qpos[static_cast<std::size_t>(t)] = L - nq + t;  // rows 0..2 valid
        // ---- attn_decode (scalar reference kernel): queries at L-3 .. L-1
        {
            Buf out(static_cast<std::size_t>(nq) * kHeads * kHd * 4), scores(static_cast<std::size_t>(nq) * kHeads * kMaxCtx * 4);
            Buf pos(std::vector<int>{L - nq});
            hip::launch(c.fn("attn_decode" + fs), {kHeads, nq, 1}, {256, 1, 1}, 0, c.s, dq.p(), pool.args(), out.p(),
                        scores.p(), kHeads, kKv, kHd, kQStride, kMaxCtx, pos.p(), kScale);
            c.sync();
            std::vector<double> ro, rs;
            const std::vector<int> rows = {0, 1, 2};
            attnRef(h, pool, q, qpos, rows, ro, rs);
            c.rep.add(cmpTol("attn_decode" + fs + " vs CPU" + tag, out.down<float>(ro.size()), ro, rs, 1e-4, 1e-6));
        }
        // ---- attn_split + attn_combine / attn_combine_q8
        const int ns = nSplit();
        Buf ml(static_cast<std::size_t>(16) * ns * kHeads * 2 * 4), acc(static_cast<std::size_t>(16) * ns * kHeads * kHd * 4);
        Buf posq(qpos);
        auto combine = [&](int n) {
            Buf out(static_cast<std::size_t>(n) * kHeads * kHd * 4);
            hip::launch(c.k.attn_combine, {kHeads, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, ml.p(), acc.p(),
                        dq.p(), out.p(), kHeads, kHd, kQStride, ns);
            c.sync();
            return out.down<float>(static_cast<std::size_t>(n) * kHeads * kHd);
        };
        {
            hip::launch(c.fn("attn_split" + fs), {kKv, static_cast<unsigned>(ns), nq}, {256, 1, 1}, 0, c.s, dq.p(),
                        pool.args(), ml.p(), acc.p(), kHeads, kKv, kHd, kQStride, posq.p(), kScale, 1, DevPtr{0});
            const std::vector<float> out = combine(nq);
            std::vector<double> ro, rs;
            attnRef(h, pool, q, qpos, {0, 1, 2}, ro, rs);
            c.rep.add(cmpTol("attn_split" + fs + " + attn_combine vs CPU" + tag, out, ro, rs, 1e-4, 1e-6));
            if (c.k.attn_combine_q8) {
                Buf o8(out.size() * 4), xq(out.size()), xd(out.size() / 32 * 4);
                hip::launch(c.k.attn_combine_q8, {kHeads, nq, 1}, {kHd, 1, 1}, 0, c.s, ml.p(), acc.p(), dq.p(), o8.p(),
                            kHeads, kHd, kQStride, ns, xq.p(), xd.p());
                c.sync();
                c.rep.add(cmpExact("attn_combine_q8 out == attn_combine" + tag, o8.down<float>(out.size()), out, Kind::invariant));
                std::vector<std::int8_t> rq(out.size());
                std::vector<float> rd(out.size() / 32);
                ref::quantizeQ8(out.data(), static_cast<int>(out.size()), rq.data(), rd.data());
                c.rep.add(cmpExact("attn_combine_q8 xq == quantize_q8(out)" + tag, xq.down<std::int8_t>(rq.size()), rq));
                c.rep.add(cmpExact("attn_combine_q8 xd == quantize_q8(out)" + tag, xd.down<float>(rd.size()), rd));
            }
        }
        // ---- attn_wsplit1: grouped == per-query (checkAttnGroups), and vs CPU
        // ---- attn_wsplit2 (<= 32 columns): grouped == per-query attn_wsplit1
        if (auto fw = c.fnOpt("attn_wsplit1" + fs))
        for (int wv : {1, 2}) {
            const auto fg = wv == 1 ? fw : c.fnOpt("attn_wsplit2" + fs);
            if (!fg) continue;
            const std::string kn = "attn_wsplit" + std::to_string(wv);
            const int ng_q = std::max(1, (16 * wv) / kGrp);  // queries per group (2 / 5 for 24 / 4 heads)
            for (int p0 : {L - 100, L - 77}) {  // a chunk start and a position inside a chunk
                std::vector<int> pv(16);
                for (int t = 0; t < 16; ++t) pv[static_cast<std::size_t>(t)] = p0 + t;
                Buf dpv(pv);
                std::vector<float> mla, acca;
                for (int mode = 0; mode < 2; ++mode) {
                    ml.zero();
                    acc.zero();
                    wk::AwGroups g;
                    unsigned ng = 0;
                    if (mode == 0) {
                        g.first[0] = 0;
                        g.count[0] = ng_q;
                        ng = 1;
                    } else {
                        for (int r = 0; r < ng_q; ++r) {
                            g.first[r] = r;
                            g.count[r] = 1;
                        }
                        ng = static_cast<unsigned>(ng_q);
                    }
                    hip::launch(mode == 0 ? fg : fw, {kKv, static_cast<unsigned>(ns), ng}, {128, 1, 1}, 0, c.s, dq.p(), pool.args(), ml.p(),
                                acc.p(), kHeads, kKv, kQStride, dpv.p(), kScale, DevPtr{0}, g);
                    c.sync();
                    auto m = ml.down<float>(static_cast<std::size_t>(ng_q) * ns * kHeads * 2);
                    auto a = acc.down<float>(static_cast<std::size_t>(ng_q) * ns * kHeads * kHd);
                    if (mode == 0) {
                        mla = m;
                        acca = a;
                        continue;
                    }
                    // compare (empty splits: both -inf)
                    std::size_t bad = 0, n = 0;
                    for (std::size_t i = 0; i < m.size(); i += 2) {
                        if (std::isinf(mla[i]) && mla[i] < 0 && std::isinf(m[i]) && m[i] < 0) continue;
                        n += 2 + kHd;
                        bad += std::memcmp(&mla[i], &m[i], 8) != 0;
                        bad += std::memcmp(&acca[i / 2 * kHd], &a[i / 2 * kHd], kHd * 4) != 0;
                    }
                    Result r;
                    r.name = kn + fs + " grouped == per-query (" + std::to_string(ng_q) + " q at pos " +
                             std::to_string(p0) + ")" + tag;
                    r.kind = Kind::invariant;
                    r.n = n;
                    r.mismatches = bad;
                    r.pass = bad == 0;
                    if (c.k.caps.attn_group1)  // gfx1151: the host never groups queries (not bitwise here)
                        c.rep.skip("attn", r.name, "caps.attn_group1: groups of one query only (" + std::to_string(bad) + " differing rows)");
                    else
                        c.rep.add(r);
                    // per-query result vs CPU
                    const std::vector<float> out = combine(ng_q);
                    std::vector<double> ro, rs;
                    std::vector<int> rows(static_cast<std::size_t>(ng_q));
                    std::iota(rows.begin(), rows.end(), 0);
                    attnRef(h, pool, q, pv, rows, ro, rs);
                    c.rep.add(cmpTol(kn + fs + " + combine vs CPU (pos " + std::to_string(p0) + ")" + tag, out, ro,
                                     rs, 1e-2, 1e-4));
                }
            }
        }
        // ---- MTP draft window (KvArgs::win = W / 64 | threshold / 1024 << 16): a window that reaches
        // the sink, or a context below the threshold, gives the same bits as no window; a narrow
        // window (sink 256 + last 256 from a 64-aligned start) matches the CPU over those keys
        if (!c.k.caps.draft_window) {  // gfx1151: attn_wsplit* ignore KvArgs::win
            c.rep.skip("attn", "attn_wsplit1 draft window" + tag, "caps.draft_window off: MTP draft window not implemented (drafts attend all positions)");
        } else if (auto fw1 = c.fnOpt("attn_wsplit1" + fs)) {
            const std::vector<int> pv1 = {L - 1};
            Buf dp1(pv1);
            wk::AwGroups g1;
            g1.first[0] = 0;
            g1.count[0] = 1;
            auto winArg = [](int w, int min_ctx) { return ((w + 63) / 64) | ((min_ctx / 1024) << 16); };
            auto run = [&](int win) {
                ml.zero();
                acc.zero();
                wk::KvArgs a = pool.args();
                a.win = win;
                hip::launch(fw1, {kKv, static_cast<unsigned>(ns), 1}, {128, 1, 1}, 0, c.s, dq.p(), a, ml.p(), acc.p(), kHeads, kKv,
                            kQStride, dp1.p(), kScale, DevPtr{0}, g1);
                c.sync();
                return std::pair<std::vector<float>, std::vector<float>>{ml.down<float>(static_cast<std::size_t>(ns) * kHeads * 2),
                                                                         acc.down<float>(static_cast<std::size_t>(ns) * kHeads * kHd)};
            };
            const auto base = run(0);
            const auto all = run(winArg(L, 0));            // W >= context: the window reaches the sink
            const auto below = run(winArg(256, L + 1024));  // context below the threshold
            c.rep.add(cmpExact("attn_wsplit1 draft window reaching the sink == off (ml)" + tag, all.first, base.first, Kind::invariant));
            c.rep.add(cmpExact("attn_wsplit1 draft window reaching the sink == off (acc)" + tag, all.second, base.second, Kind::invariant));
            c.rep.add(cmpExact("attn_wsplit1 draft window below threshold == off (ml)" + tag, below.first, base.first, Kind::invariant));
            c.rep.add(cmpExact("attn_wsplit1 draft window below threshold == off (acc)" + tag, below.second, base.second, Kind::invariant));
            const int w0 = std::max(256, (L - 256) & ~63);
            run(winArg(256, 0));
            const std::vector<float> out = combine(1);
            std::vector<double> ro, rs;
            attnRef(h, pool, q, pv1, {0}, ro, rs, w0);
            c.rep.add(cmpTol("attn_wsplit1 draft window (sink 256 + keys from " + std::to_string(w0) + ") + combine vs CPU" + tag, out, ro, rs,
                             1e-2, 1e-4));
        }
        // ---- prefill attention: n queries at L-n .. L-1
        {
            const int n = c.quick ? 130 : 200;
            const std::vector<float> qp = c.randn(static_cast<std::size_t>(n) * kHeads * kQStride);
            Buf dqp(qp), pos(std::vector<int>{L - n});
            auto run = [&](hip::Function f, unsigned threads, int h0, int nh) {
                Buf out(static_cast<std::size_t>(n) * kHeads * kHd * 4);
                out.zero();
                hip::launch(f, {cdiv(n, 128), static_cast<unsigned>(nh), 1}, {threads, 1, 1}, 0, c.s, dqp.p(),
                            pool.args(), out.p(), kHeads, kKv, kQStride, pos.p(), n, kScale, h0);
                return out;
            };
            Buf o1 = run(c.fn("attn_prefill_wmma" + fs), 256, 0, kHeads);
            c.sync();
            const auto ref1 = o1.down<float>(static_cast<std::size_t>(n) * kHeads * kHd);
            std::vector<int> qpp(static_cast<std::size_t>(n));
            for (int t = 0; t < n; ++t) qpp[static_cast<std::size_t>(t)] = L - n + t;
            std::vector<int> rows;
            for (int t = 0; t < n; t += (c.quick ? 37 : 13)) rows.push_back(t);
            rows.push_back(n - 1);
            std::vector<double> ro, rs;
            attnRef(h, pool, qp, qpp, rows, ro, rs);
            c.rep.add(cmpTol("attn_prefill_wmma" + fs + " vs CPU (" + std::to_string(rows.size()) + " of " +
                                 std::to_string(n) + " queries)" + tag,
                             pickRows(ref1, rows), ro, rs, 1e-2, 1e-4));
            if (auto fk = c.fnOpt("attn_kx" + fs)) {
                Buf o2 = run(fk, 512, 0, kHeads);
                c.sync();
                c.rep.add(cmpExact("attn_kx" + fs + " == attn_prefill_wmma" + fs + tag, o2.down<float>(ref1.size()), ref1,
                                   Kind::invariant));
            }
            // attn_kg (f16 / q8 / q8v KV): GQA-grouped kernel == attn_prefill_wmma, for
            // group 6 (24 / 4) and group 8 (24 / 3, the pool read with 3 KV heads),
            // one launch and two head-range launches
            for (const int nkv : {kKv, 3}) {
                Buf rb(static_cast<std::size_t>(n) * kHeads * kHd * 4);
                rb.zero();
                hip::launch(c.fn("attn_prefill_wmma" + fs), {cdiv(n, 128), static_cast<unsigned>(kHeads), 1}, {256, 1, 1}, 0, c.s,
                            dqp.p(), pool.args(), rb.p(), kHeads, nkv, kQStride, pos.p(), n, kScale, 0);
                c.sync();
                const auto ref = rb.down<float>(static_cast<std::size_t>(n) * kHeads * kHd);
                const int grp = kHeads / nkv;
                for (const auto& [name, np] : {std::pair{"attn_kg6", 6}, std::pair{"attn_kg4", 4}, std::pair{"attn_kg2", 2}}) {
                    if (grp % np != 0) continue;
                    auto fk = c.fnOpt(name + fs);
                    if (!fk) continue;
                    for (const int split : {1, 2}) {
                        const int hp = kHeads / split;
                        if (hp % np != 0) continue;
                        Buf ob(static_cast<std::size_t>(n) * kHeads * kHd * 4);
                        ob.zero();
                        for (int h0 = 0; h0 < kHeads; h0 += hp)
                            hip::launch(fk, {cdiv(n, 16), static_cast<unsigned>(hp / np), 1}, {static_cast<unsigned>(64 * np), 1, 1}, 0,
                                        c.s, dqp.p(), pool.args(), ob.p(), kHeads, nkv, kQStride, pos.p(), n, kScale, h0);
                        c.sync();
                        c.rep.add(cmpExact(std::string(name) + fs + " == attn_prefill_wmma" + fs + " (group " + std::to_string(grp) +
                                               (split > 1 ? ", 2 head-range launches)" : ")") + tag,
                                           ob.down<float>(ref.size()), ref, Kind::invariant));
                    }
                }
            }
            {  // two head-range launches (h0 = 0, 12) == one launch
                Buf out(static_cast<std::size_t>(n) * kHeads * kHd * 4);
                for (int h0 : {0, kHeads / 2})
                    hip::launch(c.fn("attn_prefill_wmma" + fs), {cdiv(n, 128), kHeads / 2, 1}, {256, 1, 1}, 0, c.s, dqp.p(),
                                pool.args(), out.p(), kHeads, kKv, kQStride, pos.p(), n, kScale, h0);
                c.sync();
                c.rep.add(cmpExact("attn_prefill_wmma" + fs + " head-split launches == one" + tag,
                                   out.down<float>(ref1.size()), ref1, Kind::invariant));
            }
        }
    }

    // ---- batched rows (kvbase): two sequences with their own page-table regions
    {
        Pool pool(c, Fmt::f16);
        // page table: sequence 0 uses tab entries [0, 4), sequence 1 entries [4, 8)
        const std::vector<int> base = {0, 4, 0, 4};
        const std::vector<int> pos = {5, 300, 6, 301};
        const std::vector<float> kin = c.randn(4 * kRow), vin = c.randn(4 * kRow);
        Buf dk(kin), dv(vin), dpos(pos), dbase(base);
        hip::launch(c.k.kv_store, {4, 1, 1}, {256, 1, 1}, 0, c.s, dk.p(), dv.p(), pool.args(dbase.p()), dpos.p(), kRow);
        c.sync();
        const HostKv h = download(pool);
        std::size_t bad = 0;
        for (int t = 0; t < 4; ++t) {
            const int pr = pool.ptab_h[static_cast<std::size_t>(base[static_cast<std::size_t>(t)] + pos[static_cast<std::size_t>(t)] / wk::kKvPage)] * wk::kKvPage +
                           pos[static_cast<std::size_t>(t)] % wk::kKvPage;
            for (int i = 0; i < kRow; ++i) {
                bad += h.k16[static_cast<std::size_t>(pr) * kRow + i] != f2h(kin[static_cast<std::size_t>(t) * kRow + i]);
                bad += h.v16[static_cast<std::size_t>(pr) * kRow + i] != f2h(vin[static_cast<std::size_t>(t) * kRow + i]);
            }
        }
        Result r;
        r.name = "kv_store batched rows (kvbase page-table offsets)";
        r.n = 4 * 2 * kRow;
        r.mismatches = bad;
        r.pass = bad == 0;
        c.rep.add(r);
    }

    // ---- attn_prep_q8h (q8 KV with Hadamard-rotated q / k): q vs CPU
    if (auto fh = c.fnOpt("attn_prep_q8h")) {
        const int n = 3;
        const float theta_scale = std::pow(1.0e7f, -2.0f / kRot), eps = 1e-6f;
        const std::vector<float> qf = c.randn(static_cast<std::size_t>(n) * kHeads * kQStride);
        const std::vector<float> kk = c.randn(static_cast<std::size_t>(n) * kRow), vv = c.randn(static_cast<std::size_t>(n) * kRow);
        const std::vector<float> qw = c.randu(kHd, 0.5f, 1.5f), kw = c.randu(kHd, 0.5f, 1.5f);
        const std::vector<int> pos = {10, 11, 12};
        Buf dq(qf), dk(kk), dv(vv), dqw(qw), dkw(kw), dpos(pos);
        Pool pool(c, Fmt::q8);
        hip::launch(fh, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dq.p(), dk.p(), dv.p(), dqw.p(), dkw.p(),
                    pool.args(), dpos.p(), kHeads, kKv, kHd, kRot, theta_scale, eps);
        c.sync();
        const auto got = dq.down<float>(qf.size());
        std::vector<float> g2;
        std::vector<double> ref;
        for (int t = 0; t < n; ++t)
            for (int h = 0; h < kHeads; ++h) {
                const float* x = qf.data() + (static_cast<std::size_t>(t) * kHeads + h) * kQStride;
                std::vector<double> v(kHd);
                double ss = 0;
                for (int i = 0; i < kHd; ++i) ss += static_cast<double>(x[i]) * x[i];
                const double r = 1.0 / std::sqrt(ss / kHd + eps);
                for (int i = 0; i < kHd; ++i) v[static_cast<std::size_t>(i)] = x[i] * r * qw[static_cast<std::size_t>(i)];
                for (int i = 0; i < kRot / 2; ++i) {
                    const double th = pos[static_cast<std::size_t>(t)] * std::pow(static_cast<double>(theta_scale), i);
                    const double x0 = v[static_cast<std::size_t>(i)], x1 = v[static_cast<std::size_t>(i + kRot / 2)];
                    v[static_cast<std::size_t>(i)] = x0 * std::cos(th) - x1 * std::sin(th);
                    v[static_cast<std::size_t>(i + kRot / 2)] = x0 * std::sin(th) + x1 * std::cos(th);
                }
                for (int len = 1; len < kHd; len <<= 1)
                    for (int i = 0; i < kHd / 2; ++i) {
                        const int a = (i / len) * 2 * len + i % len, b = a + len;
                        const double p0 = v[static_cast<std::size_t>(a)], p1 = v[static_cast<std::size_t>(b)];
                        v[static_cast<std::size_t>(a)] = p0 + p1;
                        v[static_cast<std::size_t>(b)] = p0 - p1;
                    }
                for (int i = 0; i < kHd; ++i) {
                    ref.push_back(v[static_cast<std::size_t>(i)] / std::sqrt(static_cast<double>(kHd)));
                    g2.push_back(got[(static_cast<std::size_t>(t) * kHeads + h) * kQStride + i]);
                }
            }
        c.rep.add(cmpTolRel("attn_prep_q8h rotated q vs CPU (rmsnorm, RoPE, Walsh-Hadamard)", g2, ref, 1e-4, 1e-5));
    }

    // ---- attn_prep == rmsnorm + rope_neox + kv_store (same arithmetic)
    for (Fmt fmt : {Fmt::f16, Fmt::q8, Fmt::q8v}) {
        const std::string fs = fmtSuffix(fmt);
        auto fp = c.fnOpt("attn_prep" + fs);
        if (!fp) continue;
        const int n = 5;
        const float theta_scale = std::pow(1.0e7f, -2.0f / kRot), eps = 1e-6f;
        const std::vector<float> qf = c.randn(static_cast<std::size_t>(n) * kHeads * kQStride);
        const std::vector<float> kk = c.randn(static_cast<std::size_t>(n) * kRow), vv = c.randn(static_cast<std::size_t>(n) * kRow);
        const std::vector<float> qw = c.randu(kHd, 0.5f, 1.5f), kw = c.randu(kHd, 0.5f, 1.5f);
        std::vector<int> pos(static_cast<std::size_t>(n));
        for (int t = 0; t < n; ++t) pos[static_cast<std::size_t>(t)] = 254 + t;  // crosses a page
        Buf dqw(qw), dkw(kw), dpos(pos);
        // A: fused
        Pool pa(c, fmt);
        Buf qa(qf), ka(kk), va(vv);
        hip::launch(fp, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, qa.p(), ka.p(), va.p(), dqw.p(),
                    dkw.p(), pa.args(), dpos.p(), kHeads, kKv, kHd, kRot, theta_scale, eps);
        // B: separate kernels (same page table)
        Pool pb(c, fmt);
        pb.ptab_h = pa.ptab_h;
        pb.ptab.up(pb.ptab_h);
        Buf qb(qf), kb(kk), vb(vv);
        hip::launch(c.k.rmsnorm, {static_cast<unsigned>(n * kHeads), 1, 1}, {256, 1, 1}, 0, c.s, qb.p(), dqw.p(), qb.p(), kHd,
                    kQStride, kQStride, eps);
        hip::launch(c.k.rmsnorm, {static_cast<unsigned>(n * kKv), 1, 1}, {256, 1, 1}, 0, c.s, kb.p(), dkw.p(), kb.p(), kHd, kHd,
                    kHd, eps);
        hip::launch(c.k.rope_neox, {kHeads, static_cast<unsigned>(n), 1}, {64, 1, 1}, 0, c.s, qb.p(), dpos.p(), kQStride,
                    kQStride * kHeads, kRot, theta_scale);
        hip::launch(c.k.rope_neox, {kKv, static_cast<unsigned>(n), 1}, {64, 1, 1}, 0, c.s, kb.p(), dpos.p(), kHd, kHd * kKv,
                    kRot, theta_scale);
        hip::launch(c.fn("kv_store" + fs), {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, kb.p(), vb.p(), pb.args(),
                    dpos.p(), kRow);
        c.sync();
        const std::size_t nqf = qf.size(), nk = kk.size();
        bool same = std::memcmp(qa.down<float>(nqf).data(), qb.down<float>(nqf).data(), nqf * 4) == 0 &&
                    std::memcmp(ka.down<float>(nk).data(), kb.down<float>(nk).data(), nk * 4) == 0;
        const HostKv ha = download(pa), hb = download(pb);
        same = same && ha.k16 == hb.k16 && ha.v16 == hb.v16 && ha.k8 == hb.k8 && ha.v8 == hb.v8 && ha.ks == hb.ks &&
               ha.vs == hb.vs;
        Result r;
        r.name = "attn_prep" + fs + " == rmsnorm + rope_neox + kv_store" + fs;
        r.kind = Kind::invariant;
        r.n = nqf + nk;
        r.mismatches = same ? 0 : 1;
        r.pass = same;
        c.rep.add(r);
    }

    // ---- vision: attn_prep_m* (multi-section RoPE) and set_rpos
    // (a) equal positions (rpos == null, rdelta = 0; and an rpos table of equal
    //     triples) == attn_prep of the same format, bitwise (q, k, KV pool);
    // (b) distinct {t, h, w} rows vs a double CPU reference (q; rotated for q8h);
    // (c) rdelta: positions pos - rdelta for RoPE, KV row still pos;
    // (d) set_rpos writes the by-value table exactly.
    const int secs[3] = {4, 3, 3};  // pairs i % 3 == 1 (i < 9) take h, i % 3 == 2 (i < 9) take w
    auto pairPos = [&](int i, int pt, int ph, int pw) {
        const int sec = i % 3;
        return (sec == 1 && i < 3 * secs[1]) ? ph : (sec == 2 && i < 3 * secs[2]) ? pw : pt;
    };
    struct MV {
        const char* plain;
        const char* m;
        Fmt fmt;
        bool rot;
    };
    const MV mvs[] = {{"attn_prep", "attn_prep_m", Fmt::f16, false},
                      {"attn_prep_q8", "attn_prep_m_q8", Fmt::q8, false},
                      {"attn_prep_q8h", "attn_prep_m_q8h", Fmt::q8, true},
                      {"attn_prep_q8v", "attn_prep_m_q8v", Fmt::q8v, false}};
    for (const MV& mv : mvs) {
        auto fm = c.fnOpt(mv.m);
        auto fp = c.fnOpt(mv.plain);
        if (!fm || !fp) {
            c.rep.skip("attn", mv.m, "kernel not in this code object");
            continue;
        }
        const int n = 5;
        const float theta_scale = std::pow(1.0e7f, -2.0f / kRot), eps = 1e-6f;
        const std::vector<float> qf = c.randn(static_cast<std::size_t>(n) * kHeads * kQStride);
        const std::vector<float> kk = c.randn(static_cast<std::size_t>(n) * kRow), vv = c.randn(static_cast<std::size_t>(n) * kRow);
        const std::vector<float> qw = c.randu(kHd, 0.5f, 1.5f), kw = c.randu(kHd, 0.5f, 1.5f);
        std::vector<int> pos(static_cast<std::size_t>(n));
        for (int t = 0; t < n; ++t) pos[static_cast<std::size_t>(t)] = 254 + t;  // crosses a page
        Buf dqw(qw), dkw(kw), dpos(pos);
        std::vector<int> eq3;
        for (int t = 0; t < n; ++t) eq3.insert(eq3.end(), {pos[static_cast<std::size_t>(t)], pos[static_cast<std::size_t>(t)], pos[static_cast<std::size_t>(t)]});
        Buf deq3(eq3);
        // (a) plain vs MR with rpos == null and with equal triples
        Pool pa(c, mv.fmt), pb(c, mv.fmt), pc(c, mv.fmt);
        pb.ptab_h = pa.ptab_h;
        pb.ptab.up(pb.ptab_h);
        pc.ptab_h = pa.ptab_h;
        pc.ptab.up(pc.ptab_h);
        Buf qa(qf), ka(kk), va(vv), qb(qf), kb(kk), vb(vv), qc(qf), kc(kk), vc(vv);
        hip::launch(fp, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, qa.p(), ka.p(), va.p(), dqw.p(), dkw.p(),
                    pa.args(), dpos.p(), kHeads, kKv, kHd, kRot, theta_scale, eps);
        hip::launch(fm, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, qb.p(), kb.p(), vb.p(), dqw.p(), dkw.p(),
                    pb.args(), dpos.p(), kHeads, kKv, kHd, kRot, theta_scale, eps, DevPtr(0), 0, secs[0], secs[1], secs[2]);
        hip::launch(fm, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, qc.p(), kc.p(), vc.p(), dqw.p(), dkw.p(),
                    pc.args(), dpos.p(), kHeads, kKv, kHd, kRot, theta_scale, eps, deq3.p(), 0, secs[0], secs[1], secs[2]);
        c.sync();
        const std::size_t nqf = qf.size(), nk = kk.size();
        const auto q_a = qa.down<float>(nqf), q_b = qb.down<float>(nqf), q_c = qc.down<float>(nqf);
        const auto k_a = ka.down<float>(nk), k_b = kb.down<float>(nk), k_c = kc.down<float>(nk);
        const HostKv ha = download(pa), hb = download(pb), hc = download(pc);
        auto kvSame = [](const HostKv& x, const HostKv& y) {
            return x.k16 == y.k16 && x.v16 == y.v16 && x.k8 == y.k8 && x.v8 == y.v8 && x.ks == y.ks && x.vs == y.vs;
        };
        const bool same_b = std::memcmp(q_a.data(), q_b.data(), nqf * 4) == 0 && std::memcmp(k_a.data(), k_b.data(), nk * 4) == 0 && kvSame(ha, hb);
        const bool same_c = std::memcmp(q_a.data(), q_c.data(), nqf * 4) == 0 && std::memcmp(k_a.data(), k_c.data(), nk * 4) == 0 && kvSame(ha, hc);
        {
            Result r;
            r.name = std::string(mv.m) + " (equal positions: rpos null / equal triples) == " + mv.plain;
            r.kind = Kind::invariant;
            r.n = 2 * (nqf + nk);
            r.mismatches = (same_b ? 0 : 1) + (same_c ? 0 : 1);
            r.pass = same_b && same_c;
            c.rep.add(r);
        }
        // (b) distinct {t, h, w}: q vs CPU
        std::vector<int> rp;
        for (int t = 0; t < n; ++t) rp.insert(rp.end(), {100 + t, 100 + t + 3 * t, 100 + t + 7 * (t % 2)});
        Buf drp(rp), qd(qf), kd(kk), vd(vv);
        Pool pd(c, mv.fmt);
        hip::launch(fm, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, qd.p(), kd.p(), vd.p(), dqw.p(), dkw.p(),
                    pd.args(), dpos.p(), kHeads, kKv, kHd, kRot, theta_scale, eps, drp.p(), 0, secs[0], secs[1], secs[2]);
        c.sync();
        const auto got = qd.down<float>(nqf);
        std::vector<float> g2;
        std::vector<double> ref;
        for (int t = 0; t < n; ++t)
            for (int h = 0; h < kHeads; ++h) {
                const float* x = qf.data() + (static_cast<std::size_t>(t) * kHeads + h) * kQStride;
                std::vector<double> v(kHd);
                double ss = 0;
                for (int i = 0; i < kHd; ++i) ss += static_cast<double>(x[i]) * x[i];
                const double r = 1.0 / std::sqrt(ss / kHd + eps);
                for (int i = 0; i < kHd; ++i) v[static_cast<std::size_t>(i)] = x[i] * r * qw[static_cast<std::size_t>(i)];
                for (int i = 0; i < kRot / 2; ++i) {
                    const int pi = pairPos(i, rp[3 * static_cast<std::size_t>(t)], rp[3 * static_cast<std::size_t>(t) + 1], rp[3 * static_cast<std::size_t>(t) + 2]);
                    const double th = pi * std::pow(static_cast<double>(theta_scale), i);
                    const double x0 = v[static_cast<std::size_t>(i)], x1 = v[static_cast<std::size_t>(i + kRot / 2)];
                    v[static_cast<std::size_t>(i)] = x0 * std::cos(th) - x1 * std::sin(th);
                    v[static_cast<std::size_t>(i + kRot / 2)] = x0 * std::sin(th) + x1 * std::cos(th);
                }
                if (mv.rot) {
                    for (int len = 1; len < kHd; len <<= 1)
                        for (int i = 0; i < kHd / 2; ++i) {
                            const int a = (i / len) * 2 * len + i % len, b = a + len;
                            const double p0 = v[static_cast<std::size_t>(a)], p1 = v[static_cast<std::size_t>(b)];
                            v[static_cast<std::size_t>(a)] = p0 + p1;
                            v[static_cast<std::size_t>(b)] = p0 - p1;
                        }
                    for (double& e : v) e /= std::sqrt(static_cast<double>(kHd));
                }
                for (int i = 0; i < kHd; ++i) {
                    ref.push_back(v[static_cast<std::size_t>(i)]);
                    g2.push_back(got[(static_cast<std::size_t>(t) * kHeads + h) * kQStride + i]);
                }
            }
        c.rep.add(cmpTolRel(std::string(mv.m) + " q vs CPU (multi-section RoPE {t,h,w}, sections 4/3/3)", g2, ref, 1e-4, 1e-5));
        // (c) rdelta: RoPE at pos - 37 (all sections), KV row pos: q == attn_prep at pos - 37
        if (!mv.rot) {
            std::vector<int> pos2(pos);
            for (int& p : pos2) p -= 37;
            Buf dpos2(pos2), qe(qf), ke(kk), ve(vv), qg(qf), kg(kk), vg(vv);
            Pool pe(c, mv.fmt), pg(c, mv.fmt);
            hip::launch(fm, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, qe.p(), ke.p(), ve.p(), dqw.p(), dkw.p(),
                        pe.args(), dpos.p(), kHeads, kKv, kHd, kRot, theta_scale, eps, DevPtr(0), 37, secs[0], secs[1], secs[2]);
            hip::launch(fp, {kHeads + kKv, static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, qg.p(), kg.p(), vg.p(), dqw.p(), dkw.p(),
                        pg.args(), dpos2.p(), kHeads, kKv, kHd, kRot, theta_scale, eps);
            c.sync();
            const bool same_e = qe.down<float>(nqf) == qg.down<float>(nqf) && ke.down<float>(nk) == kg.down<float>(nk);
            Result r;
            r.name = std::string(mv.m) + " (rdelta 37) q, k == " + mv.plain + " at pos - 37";
            r.kind = Kind::invariant;
            r.n = nqf + nk;
            r.mismatches = same_e ? 0 : 1;
            r.pass = same_e;
            c.rep.add(r);
        }
    }
    if (auto fr = c.fnOpt("set_rpos")) {
        struct Rpos16 {
            int v[48];
        } tab{};
        for (int i = 0; i < 48; ++i) tab.v[i] = 1000 * i - 7;
        for (int n : {1, 7, 16}) {
            Buf d(static_cast<std::size_t>(48) * 4);
            d.fill(0xab);
            hip::launch(fr, {1, 1, 1}, {64, 1, 1}, 0, c.s, d.p(), tab, n);
            c.sync();
            const auto got = d.down<int>(48);
            std::vector<int> ref(48);
            std::memset(ref.data(), 0xab, 48 * 4);
            for (int i = 0; i < 3 * n; ++i) ref[static_cast<std::size_t>(i)] = tab.v[i];
            c.rep.add(cmpExact("set_rpos n=" + std::to_string(n) + " (3n words, the rest untouched)", got, ref));
        }
    } else {
        c.rep.skip("attn", "set_rpos", "kernel not in this code object");
    }
}

}  // namespace kt
