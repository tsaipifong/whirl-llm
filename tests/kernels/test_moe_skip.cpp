// whirl-kernel-test "moeskip" (v0252-f5, fast mode; ported from feat027-moeskip-r1 4b6474b tests/kernels/test_moe_skip.cpp to
// the NEW entry points of kernels/moe_skip.hip (gfx1201), kernels/gfx1151/moe_skip.hip and kernels/shared/moe_skip.h; host
// reference include/whirl/moe_skip.h):
//   * presence: the entry points each code object must carry (and loadMoe loaded), and the ones it must not;
//   * moe_topk_skip thr 0 == moe_topk (ids, weights, shared gate; bitwise); thr > 0: == moe_topk followed by moe_skip_apply
//     (bitwise: the same rule on the same values), ids = moe_topk's kept prefix then -1, weights vs the host reference;
//     static top-k (thr 1, min_keep 6): exactly 6 kept per token;
//   * decode experts (the generic MXFP4 twins on both objects, the whole-block twins on gfx1201): gate + up with skipped ids
//     leaves the dropped pairs untouched (0xff fill) and the kept pairs bitwise == the shipped kernel on the full ids; down
//     with skipped ids and NaN-poisoned dropped activations == the shipped kernel on the full ids with zero weights on the
//     dropped slots (only the absent 0 * x terms differ; tol 1e-6) - a twin that reads a dropped slot turns NaN; without -1
//     the twins are bitwise the shipped kernels;
//   * prefill: moe_route_skip (bn 32 / 64 / 128) and, on gfx1151, moe_tiles_skip (bn 64 / 128) vs the host reference (tiles
//     exact, inv -1 for dropped, dropped pairs parked after the kept positions, kept positions grouped by expert); without
//     -1 they give moe_route's / moe_tiles' tile lists; moe_combine_skip with inv -1 and a NaN tail vs CPU, == moe_combine
//     without -1 (bitwise); on gfx1151 moe_combine4_skip == moe_combine_skip (bitwise).
// The kernel mutant of the CPU chain (moe_skip_rule without the renormalization) must fail the "vs host" rows.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

#include "cpu_ref.h"
#include "whirl/moe_skip.h"

namespace kt {

namespace {

Result invariantK(const std::string& name, std::size_t n, std::size_t bad, const std::string& note = {}) {
    Result r;
    r.name = name;
    r.kind = Kind::invariant;
    r.n = n;
    r.mismatches = bad;
    r.pass = bad == 0;
    r.note = note;
    return r;
}

}  // namespace

void testMoeSkip(Ctx& c) {
    c.rep.family = "moeskip";
    namespace ms = whirl::moeskip;
    const bool g1201 = c.mod.arch() == hip::Arch::gfx1201;
    const bool g1151 = c.mod.arch() == hip::Arch::gfx1151;
    // ---------------- presence (the code object and the host table)
    {
        struct P {
            const char* name;
            hip::Function host;
            bool want;
        };
        const P ps[] = {{"moe_topk_skip", c.k.moe_topk_skip, true},
                        {"moe_skip_apply", c.k.moe_skip_apply, true},
                        {"moe_route_skip", c.k.moe_route_skip, true},
                        {"moe_combine_skip", c.k.moe_combine_skip, true},
                        {"moe_gu_skip_mxfp4", c.k.moe_gu_skip_mx, true},
                        {"moe_down_skip_mxfp4", c.k.moe_down_skip_mx, true},
                        {"moe_gu_skip_mxfp4w", c.k.moe_gu_skip_mxw, g1201},
                        {"moe_down_skip_mxfp4w", c.k.moe_down_skip_mxw, g1201},
                        {"moe_tiles_skip", c.k.moe_tiles_skip, g1151},
                        {"moe_combine4_skip", c.k.moe_combine4_skip, g1151}};
        for (const P& p : ps) {
            const hip::Function f = c.fnOpt(p.name);
            const bool ok = p.want ? (f != nullptr && p.host == f) : (f == nullptr && p.host == nullptr);
            c.rep.add(invariantK(std::string(p.name) + (p.want ? " in this code object and the host table" : " absent from this code object"), 1, ok ? 0 : 1,
                                 ok ? std::string() : (f == nullptr ? "missing from the code object" : p.want ? "not in MoeKernels (loadMoe)" : "present")));
        }
    }
    if (!c.k.moe_topk_skip || !c.k.moe_skip_apply || !c.k.moe_route_skip || !c.k.moe_combine_skip || !c.k.moe_gu_skip_mx || !c.k.moe_down_skip_mx) {
        c.rep.skip("moeskip", "moe_topk_skip / moe_skip_apply / moe_route_skip / moe_combine_skip / moe_gu_skip_mxfp4 / moe_down_skip_mxfp4",
                   "kernel not in this code object");
        return;
    }
    const int R = 64, K = 8, E = 512, F = 256;
    const float thr = 0.06f;
    const int min_keep = 1;

    // ---------------- routing
    const int n = 12;
    const std::vector<float> lgv = c.randn(static_cast<std::size_t>(n) * R, 1.5f);
    const std::vector<float> h = c.randn(static_cast<std::size_t>(n) * E), shw = c.randn(static_cast<std::size_t>(E), 0.05f);
    Buf logits(lgv), dh(h), dshw(shw);
    const std::size_t nk = static_cast<std::size_t>(n) * K;
    Buf ids_f(nk * 4), w_f(nk * 4), sg_f(static_cast<std::size_t>(n) * 4);
    Buf ids_0(nk * 4), w_0(nk * 4), sg_0(static_cast<std::size_t>(n) * 4);
    Buf ids_s(nk * 4), w_s(nk * 4), sg_s(static_cast<std::size_t>(n) * 4);
    Buf ids_a(nk * 4), w_a(nk * 4), sg_a(static_cast<std::size_t>(n) * 4);
    hip::launch(c.k.moe_topk, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids_f.p(), w_f.p(), sg_f.p(), R,
                K, E);
    hip::launch(c.k.moe_topk_skip, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids_0.p(), w_0.p(),
                sg_0.p(), R, K, E, 0.f, min_keep);
    hip::launch(c.k.moe_topk_skip, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids_s.p(), w_s.p(),
                sg_s.p(), R, K, E, thr, min_keep);
    // moe_topk + moe_skip_apply (the fused-router path's form)
    hip::launch(c.k.moe_topk, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids_a.p(), w_a.p(), sg_a.p(), R,
                K, E);
    hip::launch(c.k.moe_skip_apply, {static_cast<unsigned>((n + 255) / 256), 1, 1}, {256, 1, 1}, 0, c.s, ids_a.p(), w_a.p(), n, K, thr, min_keep);
    c.sync();
    const auto hf = ids_f.down<int>(nk), h0 = ids_0.down<int>(nk), hs = ids_s.down<int>(nk), ha = ids_a.down<int>(nk);
    const auto wf = w_f.down<float>(nk), w0 = w_0.down<float>(nk), ws = w_s.down<float>(nk), wa = w_a.down<float>(nk);
    c.rep.add(cmpExact("moe_topk_skip thr 0 ids == moe_topk", h0, hf));
    c.rep.add(cmpExact("moe_topk_skip thr 0 weights == moe_topk (bitwise)", w0, wf));
    c.rep.add(cmpExact("moe_topk_skip thr 0 shared gate == moe_topk (bitwise)", sg_0.down<float>(n), sg_f.down<float>(n)));
    c.rep.add(cmpExact("moe_topk_skip shared gate == moe_topk (bitwise)", sg_s.down<float>(n), sg_f.down<float>(n)));
    c.rep.add(cmpExact("moe_topk_skip ids == moe_topk + moe_skip_apply", hs, ha));
    c.rep.add(cmpExact("moe_topk_skip weights == moe_topk + moe_skip_apply (bitwise)", ws, wa));
    std::vector<int> want_ids(hf);
    std::vector<float> want_wf(wf);
    int dropped = 0;
    for (int t = 0; t < n; ++t)
        dropped += K - ms::applyRow(want_ids.data() + static_cast<std::size_t>(t) * K, want_wf.data() + static_cast<std::size_t>(t) * K, K, ms::Params{thr, min_keep});
    c.rep.add(cmpExact("moe_topk_skip ids == moe_topk prefix + -1 (host rule)", hs, want_ids));
    c.rep.add(cmpTolRel("moe_topk_skip renormalized weights vs the host rule", ws, std::vector<double>(want_wf.begin(), want_wf.end()), 1e-6, 1e-9));
    c.rep.add(invariantK("moe_topk_skip random logits drop some but not all slots", 1, dropped > 0 && dropped < static_cast<int>(nk) - n ? 0 : 1,
                         "dropped " + std::to_string(dropped) + " of " + std::to_string(nk)));
    {
        // static top-k: thr 1 keeps exactly min_keep (6) per token, weights renormalized over the 6
        Buf ids6(nk * 4), w6(nk * 4), sg6(static_cast<std::size_t>(n) * 4);
        hip::launch(c.k.moe_topk_skip, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, logits.p(), R, dh.p(), dshw.p(), ids6.p(), w6.p(),
                    sg6.p(), R, K, E, 1.f, 6);
        c.sync();
        const auto i6 = ids6.down<int>(nk);
        const auto x6 = w6.down<float>(nk);
        std::size_t bad = 0;
        for (int t = 0; t < n; ++t) {
            double s = 0;
            for (int k = 0; k < K; ++k) {
                const std::size_t i = static_cast<std::size_t>(t) * K + k;
                bad += k < 6 ? (i6[i] != hf[i]) : (i6[i] != -1 || x6[i] != 0.f);
                if (k < 6) s += x6[i];
            }
            bad += std::fabs(s - 1.0) > 1e-5;
        }
        c.rep.add(invariantK("moe_topk_skip thr 1 min 6 = static top 6 (ids = moe_topk's first 6, weights sum 1)", static_cast<std::size_t>(n), bad));
    }

    // ---------------- decode experts (MXFP4): skipped ids vs full ids
    const int ti = static_cast<int>(QType::mxfp4);
    HostMat gate = randomMat(c, QType::mxfp4, R * F, E), up = randomMat(c, QType::mxfp4, R * F, E), down = randomMat(c, QType::mxfp4, R * E, F);
    Buf dg(gate.data), du(up.data), dd(down.data);
    Buf xq(static_cast<std::size_t>(n) * E), xd(static_cast<std::size_t>(n) * E / 32 * 4);
    hip::launch(c.k.quantize_q8, {cdiv(static_cast<std::uint64_t>(n) * E, 256), 1, 1}, {256, 1, 1}, 0, c.s, dh.p(), xq.p(), xd.p(), n * E);
    std::vector<float> wz = ws;  // full ids, zero weights on the dropped slots
    Buf dwz(wz);
    const std::size_t ng = nk * F;
    for (const bool whole : {false, true}) {
        const hip::Function gu0 = whole ? c.k.moe_gu_mxw : c.k.moe_gu[ti], dn0 = whole ? c.k.moe_down_mxw : c.k.moe_down[ti];
        const hip::Function gu1 = whole ? c.k.moe_gu_skip_mxw : c.k.moe_gu_skip_mx, dn1 = whole ? c.k.moe_down_skip_mxw : c.k.moe_down_skip_mx;
        const std::string nm = whole ? "mxfp4w" : "mxfp4";
        if (whole && !g1201) continue;  // whole-block decode experts: gfx1201 only (by design)
        if (!gu0 || !dn0 || !gu1 || !dn1) {
            c.rep.skip("moeskip", "moe_gu_skip_" + nm + " / moe_down_skip_" + nm, "kernel not in this code object");
            continue;
        }
        Buf yg_s(ng * 4), yu_s(ng * 4), yg_f(ng * 4), yu_f(ng * 4), yg_n(ng * 4), yu_n(ng * 4);
        yg_s.fill(0xff);
        yu_s.fill(0xff);
        hip::launch(gu1, {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, F, 2 * F, xq.p(), xd.p(),
                    ids_s.p(), yg_s.p(), yu_s.p(), E, K, DevPtr{0});
        hip::launch(gu0, {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, F, 2 * F, xq.p(), xd.p(),
                    ids_f.p(), yg_f.p(), yu_f.p(), E, K, DevPtr{0});
        hip::launch(gu1, {cdiv(2 * F, 8), static_cast<unsigned>(n * K), 1}, {256, 1, 1}, 0, c.s, dg.p(), du.p(), gate.row_bytes, F, 2 * F, xq.p(), xd.p(),
                    ids_f.p(), yg_n.p(), yu_n.p(), E, K, DevPtr{0});
        c.sync();
        const auto gs = yg_s.down<std::uint32_t>(ng), us = yu_s.down<std::uint32_t>(ng);
        const auto gf = yg_f.down<std::uint32_t>(ng), uf = yu_f.down<std::uint32_t>(ng);
        std::size_t bad = 0;
        for (std::size_t p = 0; p < nk; ++p)
            for (int r = 0; r < F; ++r) {
                const std::size_t i = p * F + r;
                if (hs[p] < 0) bad += gs[i] != 0xffffffffu || us[i] != 0xffffffffu;  // dropped: untouched
                else bad += gs[i] != gf[i] || us[i] != uf[i];                         // kept: the shipped kernel's bits
            }
        c.rep.add(invariantK("moe_gu_skip_" + nm + " skipped ids: dropped pairs untouched, kept == moe_gu (bitwise)", ng, bad));
        c.rep.add(cmpExact("moe_gu_skip_" + nm + " without -1 == moe_gu (bitwise, gate)", yg_n.down<std::uint32_t>(ng), gf));
        c.rep.add(cmpExact("moe_gu_skip_" + nm + " without -1 == moe_gu (bitwise, up)", yu_n.down<std::uint32_t>(ng), uf));
        // down: int8 activations from the full launch; the skipped launch gets NaN-poisoned activations on the dropped slots
        Buf aq(ng), ad(ng / 32 * 4);
        hip::launch(c.k.silu_mul_q8, {cdiv(static_cast<std::uint64_t>(ng), 256), 1, 1}, {256, 1, 1}, 0, c.s, yg_f.p(), yu_f.p(), aq.p(), ad.p(),
                    static_cast<int>(ng));
        c.sync();
        std::vector<float> adn = ad.down<float>(ng / 32);
        for (std::size_t p = 0; p < nk; ++p)
            if (hs[p] < 0)
                for (int b = 0; b < F / 32; ++b) adn[p * (F / 32) + static_cast<std::size_t>(b)] = std::nanf("");
        Buf dadn(adn);
        const std::vector<float> x0 = c.randn(static_cast<std::size_t>(n) * E), ysh = c.randn(static_cast<std::size_t>(n) * E);
        Buf x_s(x0), x_f(x0), x_n(x0), x_o(x0), dysh(ysh);
        hip::launch(dn1, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, aq.p(), dadn.p(), ids_s.p(), w_s.p(),
                    sg_s.p(), dysh.p(), x_s.p(), F, K, DevPtr{0});
        hip::launch(dn0, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, aq.p(), ad.p(), ids_f.p(), dwz.p(),
                    sg_s.p(), dysh.p(), x_f.p(), F, K, DevPtr{0});
        // without -1: the twin == the shipped kernel (bitwise)
        hip::launch(dn1, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, aq.p(), ad.p(), ids_f.p(), w_f.p(),
                    sg_s.p(), dysh.p(), x_n.p(), F, K, DevPtr{0});
        hip::launch(dn0, {cdiv(E, 8), static_cast<unsigned>(n), 1}, {256, 1, 1}, 0, c.s, dd.p(), down.row_bytes, E, aq.p(), ad.p(), ids_f.p(), w_f.p(),
                    sg_s.p(), dysh.p(), x_o.p(), F, K, DevPtr{0});
        c.sync();
        const auto gxs = x_s.down<float>(x0.size()), gxf = x_f.down<float>(x0.size());
        c.rep.add(cmpTolRel("moe_down_skip_" + nm + " skipped ids (NaN dropped activations) == moe_down with zero weights", gxs,
                            std::vector<double>(gxf.begin(), gxf.end()), 1e-6, 1e-6));
        c.rep.add(cmpExact("moe_down_skip_" + nm + " without -1 == moe_down (bitwise)", x_n.down<float>(x0.size()), x_o.down<float>(x0.size())));
    }

    // ---------------- prefill grouping + combine
    {
        const int np_tok = 300;
        const std::vector<float> lg2 = c.randn(static_cast<std::size_t>(np_tok) * R, 1.5f);
        const std::vector<float> h2 = c.randn(static_cast<std::size_t>(np_tok) * E);
        Buf dl2(lg2), dh2(h2);
        const int pairs = np_tok * K;
        Buf ids(static_cast<std::size_t>(pairs) * 4), w(static_cast<std::size_t>(pairs) * 4), sg(static_cast<std::size_t>(np_tok) * 4);
        Buf idsf(static_cast<std::size_t>(pairs) * 4), wfull(static_cast<std::size_t>(pairs) * 4), sgf(static_cast<std::size_t>(np_tok) * 4);
        hip::launch(c.k.moe_topk_skip, {static_cast<unsigned>(np_tok), 1, 1}, {256, 1, 1}, 0, c.s, dl2.p(), R, dh2.p(), DevPtr{0}, ids.p(), w.p(), sg.p(), R,
                    K, E, thr, min_keep);
        hip::launch(c.k.moe_topk, {static_cast<unsigned>(np_tok), 1, 1}, {256, 1, 1}, 0, c.s, dl2.p(), R, dh2.p(), DevPtr{0}, idsf.p(), wfull.p(), sgf.p(),
                    R, K, E);
        c.sync();
        const auto hids = ids.down<int>(static_cast<std::size_t>(pairs));
        const auto hw = w.down<float>(static_cast<std::size_t>(pairs));
        const auto hidsf = idsf.down<int>(static_cast<std::size_t>(pairs));
        auto routeCheck = [&](const hip::Function f, const char* fname, const std::vector<int>& hi, Buf& di, int bn, bool tiles_only) {
            const ms::Route want = ms::route(hi, R, bn);
            const int max_tiles = (pairs + bn - 1) / bn + R;
            Buf perm(static_cast<std::size_t>(pairs) * 4), inv(static_cast<std::size_t>(pairs) * 4), tiles(static_cast<std::size_t>(max_tiles) * 16), nt(16);
            nt.zero();
            if (tiles_only)
                hip::launch(f, {1, 1, 1}, {1024, 1, 1}, 0, c.s, di.p(), pairs, R, bn, tiles.p(), nt.p());
            else
                hip::launch(f, {1, 1, 1}, {1024, 1, 1}, 0, c.s, di.p(), pairs, R, bn, perm.p(), inv.p(), tiles.p(), nt.p());
            c.sync();
            const auto gt = tiles.down<wk::Int4>(static_cast<std::size_t>(max_tiles));
            const int gnt = nt.down<int>(1)[0];
            std::size_t bad = gnt == static_cast<int>(want.tiles.size()) ? 0 : 1;
            for (std::size_t i = 0; bad == 0 && i < want.tiles.size(); ++i) {
                const auto& a = want.tiles[i];
                bad += gt[i].x != a.expert || gt[i].y != a.first || gt[i].z != a.count;
            }
            if (!tiles_only) {
                const auto gp = perm.down<int>(static_cast<std::size_t>(pairs)), gi = inv.down<int>(static_cast<std::size_t>(pairs));
                std::vector<int> tail_got, tail_want;
                for (int p = 0; p < pairs; ++p) {
                    const int q = gi[static_cast<std::size_t>(p)];
                    if (hi[static_cast<std::size_t>(p)] < 0) {
                        bad += q != -1;
                        tail_want.push_back(p);
                    } else {
                        bad += q < 0 || q >= want.n_kept || gp[static_cast<std::size_t>(q)] != p ||
                               (q > 0 && hi[static_cast<std::size_t>(gp[static_cast<std::size_t>(q - 1)])] > hi[static_cast<std::size_t>(p)]);
                    }
                }
                for (int q = want.n_kept; q < pairs; ++q) tail_got.push_back(gp[static_cast<std::size_t>(q)]);
                std::sort(tail_got.begin(), tail_got.end());
                bad += tail_got != tail_want;
            }
            c.rep.add(invariantK(std::string(fname) + " bn=" + std::to_string(bn) + (want.n_kept < pairs ? " with dropped pairs" : " without -1") +
                                     " == host reference",
                                 static_cast<std::size_t>(pairs), bad));
            return std::make_pair(std::move(inv), want.n_kept);
        };
        Buf dfull(hidsf);
        for (const int bn : {32, 64, 128}) {
            routeCheck(c.k.moe_route_skip, "moe_route_skip", hidsf, dfull, bn, false);  // without -1: moe_route's grouping
            if (c.k.moe_tiles_skip && bn >= 64) {
                routeCheck(c.k.moe_tiles_skip, "moe_tiles_skip", hids, ids, bn, true);
                routeCheck(c.k.moe_tiles_skip, "moe_tiles_skip", hidsf, dfull, bn, true);
            }
        }
        for (const int bn : {64, 128}) routeCheck(c.k.moe_route_skip, "moe_route_skip", hids, ids, bn, false);
        auto [inv, n_kept] = routeCheck(c.k.moe_route_skip, "moe_route_skip", hids, ids, 32, false);
        const auto gi = inv.down<int>(static_cast<std::size_t>(pairs));
        // combine: kept positions random, the parked tail NaN (never read)
        std::vector<float> yd(static_cast<std::size_t>(pairs) * E, std::nanf(""));
        const std::vector<float> yk = c.randn(static_cast<std::size_t>(n_kept) * E);
        std::copy(yk.begin(), yk.end(), yd.begin());
        const std::vector<float> x0 = c.randn(static_cast<std::size_t>(np_tok) * E), ysh = c.randn(static_cast<std::size_t>(np_tok) * E);
        const std::vector<float> sgv = c.randu(static_cast<std::size_t>(np_tok), 0.1f, 0.9f);
        Buf dyd(yd), dx(x0), dx4(x0), dysh(ysh), dsg(sgv);
        hip::launch(c.k.moe_combine_skip, {cdiv(E, 256), static_cast<unsigned>(np_tok), 1}, {256, 1, 1}, 0, c.s, dx.p(), dysh.p(), dsg.p(), dyd.p(), inv.p(),
                    w.p(), E, K);
        if (c.k.moe_combine4_skip)
            hip::launch(c.k.moe_combine4_skip, {cdiv(E, 1024), static_cast<unsigned>(np_tok), 1}, {256, 1, 1}, 0, c.s, dx4.p(), dysh.p(), dsg.p(), dyd.p(),
                        inv.p(), w.p(), E, K);
        c.sync();
        const auto gx = dx.down<float>(x0.size());
        std::vector<float> ref(x0.size());
        for (int t = 0; t < np_tok; ++t)
            for (int e = 0; e < E; ++e)
                ref[static_cast<std::size_t>(t) * E + e] =
                    ms::combine(x0[static_cast<std::size_t>(t) * E + e], gi.data() + static_cast<std::size_t>(t) * K, hw.data() + static_cast<std::size_t>(t) * K, K,
                                [&](int q) { return yd[static_cast<std::size_t>(q) * E + e]; }, true, sgv[static_cast<std::size_t>(t)],
                                ysh[static_cast<std::size_t>(t) * E + e]);
        c.rep.add(cmpTolRel("moe_combine_skip (inv -1, NaN tail) vs the host reference (same f32 order; the device may fuse w * y + acc)", gx,
                            std::vector<double>(ref.begin(), ref.end()), 1e-6, 1e-6));
        if (c.k.moe_combine4_skip) c.rep.add(cmpExact("moe_combine4_skip == moe_combine_skip (bitwise)", dx4.down<float>(x0.size()), gx));
        {
            // without -1: moe_combine_skip == moe_combine (bitwise) on moe_route's grouping of the full ids
            const int bn = 32, max_tiles = (pairs + bn - 1) / bn + R;
            Buf perm(static_cast<std::size_t>(pairs) * 4), invf(static_cast<std::size_t>(pairs) * 4), tiles(static_cast<std::size_t>(max_tiles) * 16), nt(16);
            nt.zero();
            hip::launch(c.k.moe_route, {1, 1, 1}, {1024, 1, 1}, 0, c.s, dfull.p(), pairs, R, bn, perm.p(), invf.p(), tiles.p(), nt.p());
            const std::vector<float> ydf = c.randn(static_cast<std::size_t>(pairs) * E);
            Buf dydf(ydf), xa(x0), xb(x0);
            hip::launch(c.k.moe_combine_skip, {cdiv(E, 256), static_cast<unsigned>(np_tok), 1}, {256, 1, 1}, 0, c.s, xa.p(), dysh.p(), dsg.p(), dydf.p(), invf.p(),
                        wfull.p(), E, K);
            hip::launch(c.k.moe_combine, {cdiv(E, 256), static_cast<unsigned>(np_tok), 1}, {256, 1, 1}, 0, c.s, xb.p(), dysh.p(), dsg.p(), dydf.p(), invf.p(),
                        wfull.p(), E, K);
            c.sync();
            c.rep.add(cmpExact("moe_combine_skip without -1 == moe_combine (bitwise)", xa.down<float>(x0.size()), xb.down<float>(x0.size())));
        }
    }
}

}  // namespace kt
