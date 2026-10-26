// whirl-kernel-test "router" (feat026-router): moe_router_fused_i16 / _i24 (router logits + top-k + shared gate in
// one launch) against the two-kernel path moe_logits_f32 + moe_topk, bitwise (logits, ids, weights, shared gate), and
// against the CPU (logits vs f64 GEMV; ids == stable top-k of the same logits, ties -> lower index; weights / gate vs
// f64). Cases: the Ornith router (blk.0, or synthetic R 256 x E 2048) at n = 1..5 / 8 / 16; exact ties (duplicated
// router rows, all-zero rows, an all-zero input); a large-magnitude row set (weights underflow to 0, NaN-free); E 2000
// (partial last lane iteration) and E 3072 (i24); R 128 (padded lanes); K 1 / 32; no shared gate; skip = 1 (logits
// untouched, top-k still runs on the buffer, as the separate moe_topk launch); the block counter returns to 0.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <numeric>

#include "cpu_ref.h"

namespace kt {

namespace {

struct RouterCase {
    std::string name;
    int R = 256, E = 2048, K = 8, n = 2;
    bool shared = true;  // shared-expert gate row present
    bool skip = false;   // ctl_stop set
};

hip::Function fusedFor(const Ctx& c, int E) {
    const int it = (E + 127) / 128;
    return it == 16 ? c.k.moe_router_fused_i16 : it == 24 ? c.k.moe_router_fused_i24 : nullptr;
}

void runCase(Ctx& c, const RouterCase& rc, const std::vector<float>& W, const std::vector<float>& shw, const std::vector<float>& h) {
    const int R = rc.R, E = rc.E, K = rc.K, n = rc.n;
    const std::string tag = rc.name + " R" + std::to_string(R) + " E" + std::to_string(E) + " K" + std::to_string(K) + " n" + std::to_string(n);
    const hip::Function ff = fusedFor(c, E);
    if (!ff) {
        c.rep.skip("router", tag, "kernel not in this code object (moe_router_fused_i*)");
        return;
    }
    Buf dW(W), dh(h), dshw(shw.empty() ? std::vector<float>{0.f} : shw);
    const DevPtr shp = rc.shared ? dshw.p() : DevPtr{0};
    std::vector<int> one{1};
    Buf dskip(one);
    const DevPtr skp = rc.skip ? dskip.p() : DevPtr{0};
    // logits buffers start from the same pattern (skip = 1 runs top-k on it)
    std::vector<float> l0 = c.randn(static_cast<std::size_t>(n) * R);
    Buf la(l0), lb(l0);
    const std::size_t nk = static_cast<std::size_t>(n) * K;
    Buf ia(nk * 4), wa(nk * 4), sa(static_cast<std::size_t>(n) * 4), ib(nk * 4), wb(nk * 4), sb(static_cast<std::size_t>(n) * 4);
    for (Buf* b : {&ia, &wa, &sa, &ib, &wb, &sb}) b->fill(0x7f);
    Buf cnt(16);
    cnt.zero();
    // A: the production pair
    hip::launch(c.k.moe_logits_f32, {cdiv(R, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p(), dh.p(), la.p(), R, E, n, skp);
    hip::launch(c.k.moe_topk, {static_cast<unsigned>(n), 1, 1}, {256, 1, 1}, 0, c.s, la.p(), R, dh.p(), shp, ia.p(), wa.p(), sa.p(), R, K, E);
    // B: fused, twice (the second launch checks the counter reset)
    for (int rep = 0; rep < 2; ++rep)
        hip::launch(ff, {cdiv(R, 8), 1, 1}, {256, 1, 1}, 0, c.s, dW.p(), dh.p(), lb.p(), R, E, n, skp, shp, ib.p(), wb.p(), sb.p(), K, E, cnt.p());
    c.sync();
    const auto gla = la.down<float>(static_cast<std::size_t>(n) * R), glb = lb.down<float>(gla.size());
    const auto gia = ia.down<int>(nk), gib = ib.down<int>(nk);
    const auto gwa = wa.down<float>(nk), gwb = wb.down<float>(nk);
    const auto gsa = sa.down<float>(static_cast<std::size_t>(n)), gsb = sb.down<float>(static_cast<std::size_t>(n));
    c.rep.add(cmpExact("fused logits == moe_logits_f32 (bitwise) " + tag, glb, gla, Kind::invariant));
    c.rep.add(cmpExact("fused ids == moe_topk " + tag, gib, gia, Kind::invariant));
    c.rep.add(cmpExact("fused weights == moe_topk (bitwise) " + tag, gwb, gwa, Kind::invariant));
    if (rc.shared) c.rep.add(cmpExact("fused shared gate == moe_topk (bitwise) " + tag, gsb, gsa, Kind::invariant));
    c.rep.add(cmpExact("fused block counter back to 0 " + tag, cnt.down<unsigned>(1), std::vector<unsigned>{0u}, Kind::invariant));
    // CPU: logits (f64) unless skipped, then top-k of the GPU logits
    if (!rc.skip) {
        std::vector<double> ry, rs;
        ref::gemvF64(W, R, E, h.data(), n, E, ry, rs);
        c.rep.add(cmpTol("fused logits vs CPU " + tag, glb, ry, rs, 1e-5, 1e-7));
    } else {
        c.rep.add(cmpExact("skip: logits untouched " + tag, glb, l0, Kind::invariant));
    }
    std::vector<int> rid;
    std::vector<double> rw, rsg;
    for (int t = 0; t < n; ++t) {
        std::vector<int> idx(static_cast<std::size_t>(R));
        std::iota(idx.begin(), idx.end(), 0);
        const float* l = glb.data() + static_cast<std::size_t>(t) * R;
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
        double s = 0;
        for (int k = 0; k < K; ++k) s += std::exp(static_cast<double>(l[idx[static_cast<std::size_t>(k)]]) - l[idx[0]]);
        for (int k = 0; k < K; ++k) {
            rid.push_back(idx[static_cast<std::size_t>(k)]);
            rw.push_back(std::exp(static_cast<double>(l[idx[static_cast<std::size_t>(k)]]) - l[idx[0]]) / s);
        }
        if (rc.shared) {
            double d = 0;
            for (int e = 0; e < E; ++e) d += static_cast<double>(h[static_cast<std::size_t>(t) * E + e]) * shw[static_cast<std::size_t>(e)];
            rsg.push_back(1.0 / (1.0 + std::exp(-d)));
        }
    }
    c.rep.add(cmpExact("fused ids == CPU stable top-k " + tag, gib, rid));
    c.rep.add(cmpTolRel("fused weights vs CPU " + tag, gwb, rw, 1e-5, 1e-7));
    if (rc.shared) c.rep.add(cmpTolRel("fused shared gate vs CPU " + tag, gsb, rsg, 1e-5, 1e-6));
}

}  // namespace

void testMoeRouter(Ctx& c) {
    c.rep.family = "router";
    // Ornith router of blk.0 when the MoE model is given (f32 ffn_gate_inp [R][E]), else synthetic
    std::vector<float> W, shw;
    int R = 256, E = 2048, K = 8;
    std::string label = "synthetic";
    const whirl::gguf::File* f = c.open(!c.models.moemx.empty() ? c.models.moemx : c.models.moe);
    if (f && f->tensor("blk.0.ffn_gate_inp.weight") && f->tensor("blk.0.ffn_gate_inp_shexp.weight")) {
        HostMat r = loadMat(*f, "blk.0.ffn_gate_inp.weight");
        HostMat s = loadMat(*f, "blk.0.ffn_gate_inp_shexp.weight");
        if (r.type == QType::f32 && s.type == QType::f32) {
            E = r.ncols;
            R = r.nrows;
            W.resize(r.data.size() / 4);
            std::memcpy(W.data(), r.data.data(), r.data.size());
            shw.resize(s.data.size() / 4);
            std::memcpy(shw.data(), s.data.data(), s.data.size());
            K = static_cast<int>(f->getUintOr("qwen35moe.expert_used_count", 8));
            label = "ornith";
        }
    }
    if (W.empty()) {
        c.rep.skip("router", "weights", "MoE model not available, synthetic router used");
        W = c.randn(static_cast<std::size_t>(R) * E, 0.05f);
        shw = c.randn(static_cast<std::size_t>(E), 0.05f);
    }
    // 1. production shapes
    for (int n : {1, 2, 3, 4, 5, 8, 16}) {
        RouterCase rc;
        rc.name = label;
        rc.R = R, rc.E = E, rc.K = K, rc.n = n;
        runCase(c, rc, W, shw, c.randn(static_cast<std::size_t>(n) * E));
    }
    // 2. exact ties: rows 2j+1 = rows 2j (equal logits in pairs), rows 200..255 zero (equal 0 logits)
    {
        std::vector<float> Wt = c.randn(static_cast<std::size_t>(256) * 2048, 0.05f);
        for (int r = 1; r < 256; r += 2) std::copy_n(Wt.begin() + static_cast<std::ptrdiff_t>(r - 1) * 2048, 2048, Wt.begin() + static_cast<std::ptrdiff_t>(r) * 2048);
        std::fill(Wt.begin() + static_cast<std::ptrdiff_t>(200) * 2048, Wt.end(), 0.0f);
        const std::vector<float> sh = c.randn(2048, 0.05f);
        for (int n : {1, 2, 4, 7}) {
            RouterCase rc;
            rc.name = "ties";
            rc.n = n;
            runCase(c, rc, Wt, sh, c.randn(static_cast<std::size_t>(n) * 2048));
        }
        // all-zero input: every logit is 0 -> ids 0..K-1, weights 1/K
        RouterCase rz;
        rz.name = "zero-input";
        rz.n = 2;
        runCase(c, rz, Wt, sh, std::vector<float>(static_cast<std::size_t>(2) * 2048, 0.0f));
        // negative-dominated: half the rows strongly negative, top-8 among a few positives (renorm of tiny exps)
        std::vector<float> Wn = c.randn(static_cast<std::size_t>(256) * 2048, 1.0f);
        RouterCase rl;
        rl.name = "large";
        rl.n = 3;
        runCase(c, rl, Wn, sh, c.randn(static_cast<std::size_t>(3) * 2048, 2.0f));
    }
    // 3. other shapes: E 2000 (last lane iteration partial), E 3072 (i24), R 128 (padded lanes), K 1 / 32
    {
        struct S {
            int R, E, K, n;
        };
        for (const S s : {S{256, 2000, 8, 2}, S{256, 3072, 8, 2}, S{256, 3072, 8, 9}, S{128, 2048, 8, 3}, S{256, 2048, 1, 2},
                          S{256, 2048, 32, 4}, S{64, 2048, 8, 16}}) {
            RouterCase rc;
            rc.name = "shape";
            rc.R = s.R, rc.E = s.E, rc.K = s.K, rc.n = s.n;
            runCase(c, rc, c.randn(static_cast<std::size_t>(s.R) * s.E, 0.05f), c.randn(static_cast<std::size_t>(s.E), 0.05f),
                    c.randn(static_cast<std::size_t>(s.n) * s.E));
        }
    }
    // 4. no shared gate; skip = 1
    {
        RouterCase rc;
        rc.name = label + "-noshared";
        rc.R = R, rc.E = E, rc.K = K, rc.n = 2, rc.shared = false;
        runCase(c, rc, W, shw, c.randn(static_cast<std::size_t>(2) * E));
        RouterCase rs = rc;
        rs.name = label + "-skip";
        rs.shared = true;
        rs.skip = true;
        runCase(c, rs, W, shw, c.randn(static_cast<std::size_t>(2) * E));
    }
}

}  // namespace kt
