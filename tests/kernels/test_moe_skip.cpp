// whirl-kernel-test "moeskip" (feat027, fast mode): kernels/moe_skip.hip and the id -1 handling.
//   * moe_topk_skip with thr = 0 == moe_topk (ids, weights, shared gate), bitwise;
//   * moe_topk_skip thr > 0: ids = moe_topk's kept prefix then -1, weights vs the host reference
//     (whirl/moe_skip.h selectRow on the same logits);
//   * decode experts (MXFP4 generic, and the whole-block kernels where present): moe_gu_* with skipped
//     ids leaves dropped pairs untouched and kept pairs bitwise equal to the full-id launch; moe_down_*
//     with skipped ids == the full-id launch with zero weights on the dropped slots (tolerance: the
//     dropped terms are absent instead of 0 * x);
//   * precise decode experts moe_gu_f_* / moe_down_f_* (MXFP4 whole-block and the generic Q4_K form,
//     feat027-moeskip-r1): the same checks, dropped down slots with NaN activations;
//   * prefill: moe_route (and moe_tiles where present) with dropped pairs == the host reference
//     (tiles exact, inv -1 for dropped, dropped pairs parked after the kept positions); moe_combine
//     with inv -1 and a NaN tail vs CPU.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <numeric>

#include "cpu_ref.h"
#include "whirl/moe_skip.h"

namespace kt {

namespace {

Result invariant(const std::string& name, std::size_t n, std::size_t bad) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    return r;
}

}  // namespace

void testMoeSkip(Ctx& c) {
    c.rep.family = "moeskip";
    namespace ms = whirl::moeskip;
    if (!c.k.moe_topk_skip) {
        c.rep.skip("moeskip", "moe_topk_skip", "kernel not in this code object");
        return;
    }
    const int R = 64, K = 8, E = 512, F = 256;
    const float thr = 0.06f;
    const int min_keep = 1;

    // ---------------- routing: thr 0 == moe_topk, thr > 0 vs the host reference
    const int n = 12;
    std::vector<float> lgv = c.randn(static_cast<std::size_t>(n) * R, 1.5f);
    const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E), shw = c.randn(static_cast<std::size_t>(E), 0.05f);
    Buf logits(lgv), dh(h), dshw(shw);
    const std::size_t nk = static_cast<std::size_t>(n) * K;
    Buf ids_f(nk * 4), w_f(nk * 4), sg_f(static_cast<std::size_t>(n) * 4);
    Buf ids_0(nk * 4), w_0(nk * 4), sg_0(static_cast<std::size_t>(n) * 4);
    Buf ids_s(nk * 4), w_s(nk * 4), sg_s(static_cast<std::size_t>(n) * 4);
    hip::launch(c.k.moe_topk, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids_f.p(), w_f.p(),
                sg_f.p(), R, K, E);
    hip::launch(c.k.moe_topk_skip, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids_0.p(),
                w_0.p(), sg_0.p(), R, K, E, 0.f, min_keep);
    hip::launch(c.k.moe_topk_skip, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids_s.p(),
                w_s.p(), sg_s.p(), R, K, E, thr, min_keep);
    c.sync();
    const auto hf = ids_f.down<int>(nk), h0 = ids_0.down<int>(nk), hs = ids_s.down<int>(nk);
    const auto wf = w_f.down<float>(nk), w0 = w_0.down<float>(nk), ws = w_s.down<float>(nk);
    c.rep.add(cmpExact("moe_topk_skip thr 0 ids == moe_topk", h0, hf));
    c.rep.add(cmpExact("moe_topk_skip thr 0 weights == moe_topk (bitwise)", w0, wf));
    c.rep.add(cmpExact("moe_topk_skip thr 0 shared gate == moe_topk (bitwise)", sg_0.down<float>(n), sg_f.down<float>(n)));
    c.rep.add(cmpExact("moe_topk_skip shared gate == moe_topk (bitwise)", sg_s.down<float>(n), sg_f.down<float>(n)));
    std::vector<int> want_ids;
    std::vector<double> want_w;
    int dropped = 0;
    for (int t = 0; t < n; ++t) {
        float v[64], wr[64];
        for (int k = 0; k < K; ++k) v[k] = lgv[static_cast<std::size_t>(t) * R + hf[static_cast<std::size_t>(t) * K + k]];
        const int keep = ms::selectRow(v, K, ms::Params{thr, min_keep}, wr);
        dropped += K - keep;
        for (int k = 0; k < K; ++k) {
            want_ids.push_back(k < keep ? hf[static_cast<std::size_t>(t) * K + k] : -1);
            want_w.push_back(wr[k]);
        }
    }
    c.rep.add(cmpExact("moe_topk_skip ids == moe_topk prefix + -1 (host rule)", hs, want_ids));
    c.rep.add(cmpTolRel("moe_topk_skip renormalized weights vs host", ws, want_w, 1e-5, 1e-7));
    if (dropped == 0 || dropped == static_cast<int>(nk))
        c.rep.skip("moeskip", "drop mix", "random logits dropped none / all slots: the decode checks below are weak");

    // ---------------- decode experts (MXFP4): skipped ids vs full ids
    const int ti = static_cast<int>(QType::mxfp4);
    if (c.k.moe_gu[ti] && c.k.moe_down[ti]) {
        HostMat gate = randomMat(c, QType::mxfp4, R * F, E), up = randomMat(c, QType::mxfp4, R * F, E), down = randomMat(c, QType::mxfp4, R * E, F);
        Buf dg(gate.data), du(up.data), dd(down.data);
        Buf xq(static_cast<std::size_t>(n) * E), xd(static_cast<std::size_t>(n) * E / 32 * 4);
        hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(n) * E, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), xq.p(), xd.p(), n * E);
        std::vector<float> wz = ws;  // full ids, zero weights on dropped slots
        Buf dwz(wz);
        const std::size_t ng = nk * F;
        for (const bool whole : {false, true}) {
            const hip::Function gu = whole ? c.k.moe_gu_mxw : c.k.moe_gu[ti];
            const hip::Function dn = whole ? c.k.moe_down_mxw : c.k.moe_down[ti];
            const std::string nm = whole ? "mxfp4w" : "mxfp4";
            if (!gu || !dn) {
                c.rep.skip("moeskip", "moe_gu_" + nm + " / moe_down_" + nm, "kernel not in this code object");
                continue;
            }
            Buf yg_s(ng * 4), yu_s(ng * 4), yg_f(ng * 4), yu_f(ng * 4);
            yg_s.fill(0xff);
            yu_s.fill(0xff);
            hip::launch(gu, {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, F, 2 * F, xq.p(),
                        xd.p(), ids_s.p(), yg_s.p(), yu_s.p(), E, K, DevPtr{0});
            hip::launch(gu, {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, F, 2 * F, xq.p(),
                        xd.p(), ids_f.p(), yg_f.p(), yu_f.p(), E, K, DevPtr{0});
            c.sync();
            const auto gs = yg_s.down<std::uint32_t>(ng), us = yu_s.down<std::uint32_t>(ng);
            const auto gf = yg_f.down<std::uint32_t>(ng), uf = yu_f.down<std::uint32_t>(ng);
            std::size_t bad = 0;
            for (std::size_t p = 0; p < nk; ++p)
                for (int r = 0; r < F; ++r) {
                    const std::size_t i = p * F + r;
                    if (hs[p] < 0) bad += gs[i] != 0xffffffffu || us[i] != 0xffffffffu;  // dropped: untouched
                    else bad += gs[i] != gf[i] || us[i] != uf[i];                         // kept: as the full launch
                }
            c.rep.add(invariant("moe_gu_" + nm + " skipped ids: dropped pairs untouched, kept == full ids (bitwise)", ng, bad));
            // down: same activations (from the full launch) for both
            Buf aq(ng), ad(ng / 32 * 4);
            hip::launch(c.k.silu_mul_q8, {cdiv(static_cast<std::uint64_t>(ng), 256), 1, 1}, {256, 1, 1}, 0, c.s, yg_f.p(), yu_f.p(), aq.p(), ad.p(),
                        static_cast<int>(ng));
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
            Buf x_s(x0), x_f(x0), dysh(ysh);
            hip::launch(dn, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, aq.p(), ad.p(), ids_s.p(),
                        w_s.p(), sg_s.p(), dysh.p(), x_s.p(), F, K, DevPtr{0});
            hip::launch(dn, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, aq.p(), ad.p(), ids_f.p(),
                        dwz.p(), sg_s.p(), dysh.p(), x_f.p(), F, K, DevPtr{0});
            c.sync();
            const auto gxs = x_s.down<float>(x0.size()), gxf = x_f.down<float>(x0.size());
            std::vector<double> ref(gxf.begin(), gxf.end());
            c.rep.add(cmpTolRel("moe_down_" + nm + " skipped ids == full ids with zero weights", gxs, ref, 1e-5, 1e-5));
        }
    } else {
        c.rep.skip("moeskip", "moe_gu_mxfp4 / moe_down_mxfp4", "kernel not in this code object");
    }

    // ---------------- precise decode experts moe_gu_f_* / moe_down_f_* (kernels/moe_f.hip) with id -1
    // feat027-moeskip-r1 (F1 of feat026-moe-skipsafe): the precise kernels must honour the same contract (moeskip is
    // refused outside fast, this is the kernel-level safety). The weight tensors hold exactly R experts, so a kernel
    // that forms the address of expert -1 reads before the tensor (out of bounds; garbage or a fault). Dropped
    // gate/up pairs must stay untouched (0xff fill), and dropped down slots get NaN activations, so a kernel that
    // reads them poisons the result instead of passing by luck.
    for (const QType qt : {QType::mxfp4, QType::q4_k}) {
        const int ti2 = static_cast<int>(qt);
        const std::string sfx = qt == QType::mxfp4 ? "mxfp4" : "q4_k";
        if (!c.k.moe_gu_f[ti2] || !c.k.moe_down_f[ti2]) {
            c.rep.skip("moeskip", "moe_gu_f_" + sfx + " / moe_down_f_" + sfx, "kernel not in this code object");
            continue;
        }
        HostMat gate = randomMat(c, qt, R * F, E), up = randomMat(c, qt, R * F, E), down = randomMat(c, qt, R * E, F);
        Buf dg(gate.data), du(up.data), dd(down.data);
        const std::size_t ng = nk * F;
        // gate + up (f32 activations h)
        Buf yg_s(ng * 4), yu_s(ng * 4), yg_f(ng * 4), yu_f(ng * 4);
        yg_s.fill(0xff);
        yu_s.fill(0xff);
        hip::launch(c.k.moe_gu_f[ti2], {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, F,
                    2 * F, dh.p(), ids_s.p(), yg_s.p(), yu_s.p(), E, K);
        hip::launch(c.k.moe_gu_f[ti2], {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, F,
                    2 * F, dh.p(), ids_f.p(), yg_f.p(), yu_f.p(), E, K);
        c.sync();
        {
            const auto gs = yg_s.down<std::uint32_t>(ng), us = yu_s.down<std::uint32_t>(ng);
            const auto gf = yg_f.down<std::uint32_t>(ng), uf = yu_f.down<std::uint32_t>(ng);
            std::size_t bad = 0;
            for (std::size_t p = 0; p < nk; ++p)
                for (int r = 0; r < F; ++r) {
                    const std::size_t i = p * F + r;
                    if (hs[p] < 0) bad += gs[i] != 0xffffffffu || us[i] != 0xffffffffu;
                    else bad += gs[i] != gf[i] || us[i] != uf[i];
                }
            c.rep.add(invariant("moe_gu_f_" + sfx + " skipped ids: dropped pairs untouched, kept == full ids (bitwise)", ng, bad));
        }
        // down + combine: f32 activations; dropped slots NaN in the skipped launch, finite with weight 0 in the full one
        std::vector<float> act = c.randn(ng), act_nan = act;
        for (std::size_t p = 0; p < nk; ++p)
            if (hs[p] < 0) std::fill(act_nan.begin() + static_cast<std::ptrdiff_t>(p * F), act_nan.begin() + static_cast<std::ptrdiff_t>((p + 1) * F), std::nanf(""));
        const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
        std::vector<float> wz = ws;
        Buf da(act), dan(act_nan), dwz(wz), x_s(x0), x_f(x0), dysh(ysh);
        hip::launch(c.k.moe_down_f[ti2], {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, dan.p(),
                    ids_s.p(), w_s.p(), sg_s.p(), dysh.p(), x_s.p(), F, K);
        hip::launch(c.k.moe_down_f[ti2], {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, da.p(),
                    ids_f.p(), dwz.p(), sg_s.p(), dysh.p(), x_f.p(), F, K);
        c.sync();
        const auto gxs = x_s.down<float>(x0.size()), gxf = x_f.down<float>(x0.size());
        std::vector<double> ref(gxf.begin(), gxf.end());
        c.rep.add(cmpTolRel("moe_down_f_" + sfx + " skipped ids (NaN dropped activations) == full ids with zero weights", gxs, ref, 1e-5, 1e-5));
        // without -1 the patched kernel is the shipped one: full ids twice, bitwise (no -1 -> kt == K)
        Buf x_f2(x0);
        hip::launch(c.k.moe_down_f[ti2], {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, da.p(),
                    ids_f.p(), dwz.p(), sg_s.p(), dysh.p(), x_f2.p(), F, K);
        c.sync();
        c.rep.add(cmpExact("moe_down_f_" + sfx + " full ids deterministic (bitwise)", x_f2.down<float>(x0.size()), gxf));
    }

    // ---------------- prefill grouping + combine
    {
        const int np_tok = 300;
        const std::vector<float> lg2 = c.randn(static_cast<std::size_t>(np_tok) * R, 1.5f);
        const std::vector<float> h2 = c.randn(static_cast<std::size_t>(np_tok) * E);
        Buf dl2(lg2), dh2(h2);
        const int pairs = np_tok * K;
        Buf ids(static_cast<std::size_t>(pairs) * 4), w(static_cast<std::size_t>(pairs) * 4), sg(static_cast<std::size_t>(np_tok) * 4);
        hip::launch(c.k.moe_topk_skip, {static_cast<unsigned>(np_tok), 1, 1}, {256, 1, 1}, 0, c.s, dl2.p(), R, dh2.p(), DevPtr{0}, ids.p(), w.p(),
                    sg.p(), R, K, E, thr, min_keep);
        c.sync();
        const auto hids = ids.down<int>(static_cast<std::size_t>(pairs));
        const auto hw = w.down<float>(static_cast<std::size_t>(pairs));
        for (const int bn : {32, 64}) {
            const ms::Route want = ms::route(hids, R, bn);
            const int max_tiles = (pairs + bn - 1) / bn + R;
            Buf perm(static_cast<std::size_t>(pairs) * 4), inv(static_cast<std::size_t>(pairs) * 4), tiles(static_cast<std::size_t>(max_tiles) * 16), nt(4);
            hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, ids.p(), pairs, R, bn, perm.p(), inv.p(), tiles.p(), nt.p());
            c.sync();
            const auto gp = perm.down<int>(static_cast<std::size_t>(pairs)), gi = inv.down<int>(static_cast<std::size_t>(pairs));
            const auto gt = tiles.down<wk::Int4>(static_cast<std::size_t>(max_tiles));
            const int gnt = nt.down<int>(1)[0];
            std::size_t bad = gnt == static_cast<int>(want.tiles.size()) ? 0 : 1;
            for (std::size_t i = 0; bad == 0 && i < want.tiles.size(); ++i) {
                const auto& a = want.tiles[i];
                bad += gt[i].x != a.expert || gt[i].y != a.first || gt[i].z != a.count;
            }
            std::vector<int> tail_got, tail_want;
            for (int p = 0; p < pairs; ++p) {
                const int q = gi[static_cast<std::size_t>(p)];
                if (hids[static_cast<std::size_t>(p)] < 0) {
                    bad += q != -1;
                    tail_want.push_back(p);
                } else {
                    bad += q < 0 || q >= want.n_kept || gp[static_cast<std::size_t>(q)] != p ||
                           (q > 0 && hids[static_cast<std::size_t>(gp[static_cast<std::size_t>(q - 1)])] > hids[static_cast<std::size_t>(p)]);
                }
            }
            for (int q = want.n_kept; q < pairs; ++q) tail_got.push_back(gp[static_cast<std::size_t>(q)]);
            std::sort(tail_got.begin(), tail_got.end());
            bad += tail_got != tail_want;
            c.rep.add(invariant("moe_route bn=" + std::to_string(bn) + " with dropped pairs == host reference", static_cast<std::size_t>(pairs), bad));
            if (bn == 64 && c.k.moe_tiles) {
                Buf t64(static_cast<std::size_t>(max_tiles) * 16), nt64(4);
                hip::launch(c.k.moe_tiles, {1, 1, 1}, {1024, 1, 1}, 0, c.s, ids.p(), pairs, R, 64, t64.p(), nt64.p());
                c.sync();
                const auto g64 = t64.down<wk::Int4>(static_cast<std::size_t>(max_tiles));
                std::size_t b64 = nt64.down<int>(1)[0] == static_cast<int>(want.tiles.size()) ? 0 : 1;
                for (std::size_t i = 0; b64 == 0 && i < want.tiles.size(); ++i)
                    b64 += g64[i].x != want.tiles[i].expert || g64[i].y != want.tiles[i].first || g64[i].z != want.tiles[i].count;
                c.rep.add(invariant("moe_tiles bn=64 with dropped pairs == host reference", want.tiles.size(), b64));
            }
            if (bn != 32) continue;
            // combine: kept positions random, tail NaN (never read)
            std::vector<float> yd(static_cast<std::size_t>(pairs) * E, std::nanf(""));
            const std::vector<float> yk = c.randn(static_cast<std::size_t>(want.n_kept) * E);
            std::copy(yk.begin(), yk.end(), yd.begin());
            const std::vector<float> x0 = c.randn(static_cast<std::size_t>(np_tok) * E);
            Buf dyd(yd), dx(x0);
            hip::launch(c.k.moe_combine, {cdiv(E, 256), static_cast<unsigned>(np_tok), 1}, {256, 1, 1}, 0, c.s, dx.p(), DevPtr{0}, DevPtr{0}, dyd.p(),
                        inv.p(), w.p(), E, K);
            c.sync();
            const auto gx = dx.down<float>(x0.size());
            std::vector<double> ref(x0.size());
            for (int t = 0; t < np_tok; ++t)
                for (int e = 0; e < E; ++e) {
                    double acc = 0;
                    for (int k = 0; k < K; ++k) {
                        const int q = gi[static_cast<std::size_t>(t) * K + k];
                        if (q >= 0) acc += static_cast<double>(hw[static_cast<std::size_t>(t) * K + k]) * yd[static_cast<std::size_t>(q) * E + e];
                    }
                    ref[static_cast<std::size_t>(t) * E + e] = x0[static_cast<std::size_t>(t) * E + e] + acc;
                }
            c.rep.add(cmpTolRel("moe_combine with dropped slots (inv -1, NaN tail) vs CPU", gx, ref, 1e-5, 1e-6));
        }
    }
}

}  // namespace kt
