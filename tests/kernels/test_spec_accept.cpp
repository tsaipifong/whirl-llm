// whirl-kernel-test families "specacc" and "specaccz" (v0252-e1p1, PLAN V1-93 step P1a; port of the feat026-acc kernel
// test 4ac9da6 + feat026-acc-r1 C4 c797b27): the device accept decision spec_accept_rows (kernels/spec_accept.hip, both code
// objects) against kernels/spec_accept.h run on the CPU, BIT FOR BIT, launched the way each WHIRL_SPEC_ACC mode launches it:
//   specacc   hand    3 sequences at control areas 2, 0, 3 of 4 (MTP full / partial accept, literal drafts with an end-of-turn
//                     cut, a host-decided sequence): control words, the host-layout copy into a DEVICE buffer, next-cycle rows,
//                     and the hand-checked acc / chosen values
//             gpu     (WHIRL_SPEC_ACC=2) random cycles: host = null, then the pinned read the engine makes after it
//                     (Context::readCtlPinned: hip::downloadAsync into hip::hostMalloc memory on the stream + stream sync)
//             check   (WHIRL_SPEC_ACC=4) random cycles: the same launch, then the next-cycle rows read back the way
//                     Context::readSpecTab does (stream sync + blocking download) next to the control words
//   specaccz  gpu-zc  (WHIRL_SPEC_ACC=3, on request: --only specaccz) the hand case and random cycles: the host-layout buffer is
//                     pinned host memory from hip::hostMalloc (hipHostMalloc flags 0 = Engine's device::PinnedBuf), its HOST
//                     pointer goes to the kernel as the device address (SampleBuffers::readCycleCtl), the stream is synced
//                     and the CPU reads the words straight from that memory (no copy). Pre-filled with -1, so a write that lands
//                     elsewhere or nowhere fails. If this fails where "specacc" passes, the host pointer is not a device
//                     address on this GPU / driver: map it the llama.cpp way (hipHostMalloc Portable | Mapped +
//                     hipHostGetDevicePointer, refsrc llama.cpp ggml/src/ggml-cuda/allreduce.cu:268-300) before mode 3 is used.
//                     Kept apart (on request) because a wrong address there is a GPU page fault, not a failed check.
// Random cycles: 1..16 sequences at a random permutation of 16 control areas, MTP or literal drafts, p-min kept counts (stale
// / -1), matching prefixes of every length, end-of-turn drafts, ignore_eos, max_tokens budgets, some host-decided sequences.
// SPDX-License-Identifier: Apache-2.0

#include "kt.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "spec_accept.h"  // kernels/ (feat026-acc)

template <>
struct whirl::hip::LaunchArgPtrWords<SpecAccArgs> : whirl::hip::LaunchArgNoPtrWords {};

namespace kt {

namespace {

constexpr int kAreas = 16;                 // control areas (server slots) of a random cycle
constexpr int kW = WHIRL_SA_CTL_WORDS;     // words per area

// what spec_accept_rows writes, computed on the CPU with the same header: the control words after the kernel, the
// host-layout copy (only the verify's areas; -1 elsewhere) and the next-cycle rows
struct Want {
    std::vector<int> ctl, host;
    std::vector<int> tab;  // SpecNextTab as words
    SpecAccResult res[WHIRL_SA_MAX_SEG];
};

Want reference(const SpecAccArgs& a, const std::vector<int>& ctl_in, int areas) {
    Want w;
    w.ctl = ctl_in;
    for (int k = 0; k < a.n; ++k) {
        int* cw = w.ctl.data() + a.seq[k] * kW;
        // v0252-e1p1b: a host-decided sequence through spec_copy_one (its position is resolved like a decided one's)
        const SpecAccResult r = a.dev[k] ? spec_accept_one(a, k, cw) : spec_copy_one(a, k, cw);
        spec_store(a, k, r, cw);
        w.res[k] = r;
    }
    SpecNextTab t{};
    spec_next_rows(a, w.ctl.data(), w.res, &t);
    w.tab.resize(sizeof(SpecNextTab) / 4);
    std::memcpy(w.tab.data(), &t, sizeof(SpecNextTab));
    w.host.assign(static_cast<std::size_t>(areas) * kW, -1);
    for (int k = 0; k < a.n; ++k)
        for (int i = 0; i < kW; ++i) w.host[a.seq[k] * kW + i] = w.ctl[a.seq[k] * kW + i];
    return w;
}

// the feat026-acc hand case (4ac9da6 tests/kernels/test_misc.cpp)
void handCase(SpecAccArgs& a, std::vector<int>& ctl) {
    ctl.assign(4 * kW, 0);
    for (int i = 0; i < 4 * kW; ++i) ctl[i] = 7 + (i * 37) % 900;
    a = SpecAccArgs{};
    a.n = 3;
    a.n_vocab = 1000;
    a.stop0 = 998;
    a.stop1 = 999;
    const int seqs[3] = {2, 0, 3};
    for (int k = 0; k < 3; ++k) {
        a.seq[k] = seqs[k];
        a.dev[k] = k < 2 ? 1 : 0;
        a.eos[k] = 1;
        a.budget[k] = 100;
        a.pos[k] = 1000 * (k + 1);
        a.row0[k] = 5 * k;
        a.kvbase[k] = 64 * k;
    }
    // seq 2: MTP, 4 drafts, p-min kept 3, rows match drafts 0..1
    a.nd[0] = 4;
    ctl[2 * kW + 48] = 3;
    for (int r = 0; r < 4; ++r) ctl[2 * kW + 16 + r] = 100 + r;
    ctl[2 * kW + 0] = 100;
    ctl[2 * kW + 1] = 101;
    // seq 0: literal drafts {5, 998, 6}, all match: cut at the end-of-turn draft
    a.nd[1] = 3;
    a.lit[1] = 1;
    a.lit_d[1][0] = 5;
    a.lit_d[1][1] = 998;
    a.lit_d[1][2] = 6;
    ctl[0] = 5;
    ctl[1] = 998;
    ctl[2] = 6;
    a.nd[2] = 2;
}

// one random cycle over kAreas control areas (the shape of the unit test accept_policy testSpecAccept)
void randomCycle(Ctx& c, SpecAccArgs& a, std::vector<int>& ctl) {
    auto rnd = [&](std::uint32_t n) { return static_cast<std::uint32_t>(c.rng() % n); };
    const std::uint32_t V = 1000, IM_END = 998, EOT = 999;
    a = SpecAccArgs{};
    ctl.assign(static_cast<std::size_t>(kAreas) * kW, 0);
    for (int& v : ctl) v = static_cast<int>(rnd(V));
    const int A = 1 + static_cast<int>(rnd(kAreas));
    a.n = A;
    a.n_vocab = static_cast<int>(V - (rnd(4) == 0 ? 10 : 0));
    a.stop0 = IM_END;
    a.stop1 = rnd(5) == 0 ? 0xffffffffu : EOT;
    int ids[kAreas];
    for (int i = 0; i < kAreas; ++i) ids[i] = i;
    for (int i = kAreas - 1; i > 0; --i) std::swap(ids[i], ids[rnd(static_cast<std::uint32_t>(i) + 1)]);
    int row = 0;
    for (int k = 0; k < A; ++k) {
        const int sq = ids[k];
        int* cw = ctl.data() + sq * kW;
        const int nd_cap = std::min(WHIRL_SA_MAX_ND, WHIRL_SA_MAX_ROWS - row - (A - k));
        const std::uint32_t nd = nd_cap > 0 ? rnd(static_cast<std::uint32_t>(nd_cap) + 1) : 0;
        const bool lit = rnd(3) == 0;
        const std::uint32_t n_gen = rnd(50);
        const std::uint32_t max_tok = rnd(3) == 0 ? n_gen + rnd(6) : n_gen + 100;
        a.seq[k] = sq;
        a.nd[k] = static_cast<int>(nd);
        a.lit[k] = lit ? 1 : 0;
        a.dev[k] = rnd(6) == 0 ? 0 : 1;  // some sequences decide on the host (sampling / relaxacc): copy only
        a.eos[k] = rnd(4) == 0 ? 0 : 1;
        a.budget[k] = static_cast<int>(max_tok > n_gen ? max_tok - n_gen : 0);
        a.pos[k] = static_cast<int>(rnd(100000));
        a.row0[k] = row;
        row += static_cast<int>(nd) + 1;
        a.kvbase[k] = static_cast<int>(rnd(1u << 20));
        cw[48] = static_cast<int>(rnd(17)) - 1;  // p-min kept count (may be stale / -1)
        const std::uint32_t match = rnd(nd + 2);
        for (std::uint32_t r = 0; r < nd; ++r) {
            std::uint32_t d = rnd(V);
            if (rnd(8) == 0) d = rnd(2) ? IM_END : EOT;
            if (lit) a.lit_d[k][r] = static_cast<int>(d);
            else cw[16 + r] = static_cast<int>(d);
            if (r < match) cw[r] = static_cast<int>(d);
        }
    }
}

// mode 2 (gpu) and mode 4 (check): host = null; then the engine's reads of that cycle
void runDeviceMode(Ctx& c, const char* mode, int rounds) {
    hip::Function f = c.k.spec_accept_rows;
    const std::size_t words = static_cast<std::size_t>(kAreas) * kW;
    auto* pin = static_cast<int*>(hip::hostMalloc(words * sizeof(int)));
    std::size_t bad_ctl = 0, bad_pin = 0, bad_tab = 0;
    std::size_t n_eos = 0, n_budget = 0, n_host = 0;
    std::vector<int> got_all, want_all, gp_all, wp_all, gt_all, wt_all;
    for (int it = 0; it < rounds; ++it) {
        SpecAccArgs a{};
        std::vector<int> ctl;
        randomCycle(c, a, ctl);
        const Want w = reference(a, ctl, kAreas);
        Buf dctl(ctl), dtab(sizeof(SpecNextTab));
        dtab.zero();
        hip::launch(f, {1, 1, 1}, {64, 1, 1}, 0, c.s, dctl.p(), a, DevPtr{0}, dtab.p());
        // the engine's read after the kernel: async copy into pinned memory on the stream + stream sync
        for (std::size_t i = 0; i < words; ++i) pin[i] = -7;
        hip::downloadAsync(pin, dctl.p(), words * sizeof(int), c.s);
        hip::streamSync(c.s);
        const std::vector<int> got = dctl.down<int>(words);
        const std::vector<int> gp(pin, pin + words);
        const std::vector<int> gt = dtab.down<int>(sizeof(SpecNextTab) / 4);
        bad_ctl += got != w.ctl;
        bad_pin += gp != w.ctl;  // the pinned read holds every area (the verify's with the results)
        bad_tab += gt != w.tab;
        got_all.insert(got_all.end(), got.begin(), got.end());
        want_all.insert(want_all.end(), w.ctl.begin(), w.ctl.end());
        gp_all.insert(gp_all.end(), gp.begin(), gp.end());
        wp_all.insert(wp_all.end(), w.ctl.begin(), w.ctl.end());
        gt_all.insert(gt_all.end(), gt.begin(), gt.end());
        wt_all.insert(wt_all.end(), w.tab.begin(), w.tab.end());
        for (int k = 0; k < a.n; ++k) {
            n_eos += (w.res[k].flags & WHIRL_SA_F_EOS) != 0;
            n_budget += (w.res[k].flags & WHIRL_SA_F_BUDGET) != 0;
            n_host += a.dev[k] == 0;
        }
    }
    hip::hostFree(pin);
    const std::string m(mode);
    c.rep.add(cmpExact("spec_accept_rows " + m + " control words (" + std::to_string(rounds) + " cycles)", got_all, want_all));
    c.rep.add(cmpExact("spec_accept_rows " + m + " pinned read after the kernel", gp_all, wp_all));
    c.rep.add(cmpExact("spec_accept_rows " + m + " next-cycle rows", gt_all, wt_all));
    // the random cycles reached the branches (a generator change that stops hitting them fails here, not silently)
    Result r;
    r.family = "specacc";
    r.name = "spec_accept_rows " + m + " branches hit (eos " + std::to_string(n_eos) + ", budget " + std::to_string(n_budget) +
             ", host-decided " + std::to_string(n_host) + "; cycles failing ctl / pinned / rows: " + std::to_string(bad_ctl) + " / " +
             std::to_string(bad_pin) + " / " + std::to_string(bad_tab) + ")";
    r.pass = n_eos > 0 && n_budget > 0 && n_host > 0;
    r.n = static_cast<std::size_t>(rounds);
    c.rep.add(r);
}

}  // namespace

void testSpecAcc(Ctx& c) {
    if (!c.k.spec_accept_rows) {
        c.rep.skip("specacc", "spec_accept_rows", "kernel not present");
        return;
    }
    hip::Function f = c.k.spec_accept_rows;
    // the hand case: control words, host copy (device buffer), next-cycle rows, hand-checked values
    {
        SpecAccArgs a{};
        std::vector<int> ctl;
        handCase(a, ctl);
        const Want w = reference(a, ctl, 4);
        Buf dctl(ctl), dhost(std::vector<int>(4 * kW, -1)), dtab(sizeof(SpecNextTab));
        dtab.zero();  // feat026-acc-r1 (review N2): compared whole with a zero-initialised SpecNextTab
        hip::launch(f, {1, 1, 1}, {64, 1, 1}, 0, c.s, dctl.p(), a, dhost.p(), dtab.p());
        c.sync();
        c.rep.add(cmpExact("spec_accept_rows hand control words", dctl.down<int>(4 * kW), w.ctl));
        c.rep.add(cmpExact("spec_accept_rows hand host copy", dhost.down<int>(4 * kW), w.host));
        c.rep.add(cmpExact("spec_accept_rows hand next-cycle rows", dtab.down<int>(sizeof(SpecNextTab) / 4), w.tab));
        // seq 2 acc 2 chosen ctl[2]; seq 0 cut to acc 1 chosen 998 (by hand from the case above)
        c.rep.add(cmpExact("spec_accept_rows hand acc / chosen", std::vector<int>{w.res[0].acc, w.res[0].chosen, w.res[1].acc, w.res[1].chosen},
                           std::vector<int>{2, ctl[2 * kW + 2], 1, 998}));
    }
    const int rounds = c.quick ? 64 : 400;
    runDeviceMode(c, "gpu", rounds);
    runDeviceMode(c, "check", rounds);
}

void testSpecAccZc(Ctx& c) {
    if (!c.k.spec_accept_rows) {
        c.rep.skip("specaccz", "spec_accept_rows", "kernel not present");
        return;
    }
    hip::Function f = c.k.spec_accept_rows;
    const int rounds = c.quick ? 64 : 400;
    const std::size_t words = static_cast<std::size_t>(kAreas) * kW;
    std::vector<int> gz_all, wz_all, gc_all, wc_all, gt_all, wt_all;
    for (int it = 0; it <= rounds; ++it) {
        SpecAccArgs a{};
        std::vector<int> ctl;
        int areas = kAreas;
        if (it == 0) {
            handCase(a, ctl);  // the feat026-acc-r1 C4 case first
            areas = 4;
        } else {
            randomCycle(c, a, ctl);
        }
        const std::size_t n = static_cast<std::size_t>(areas) * kW;
        const Want w = reference(a, ctl, areas);
        Buf dctl(ctl), dtab(sizeof(SpecNextTab));
        dtab.zero();
        auto* hp = static_cast<int*>(hip::hostMalloc(words * sizeof(int)));
        for (std::size_t i = 0; i < words; ++i) hp[i] = -1;
        hip::launch(f, {1, 1, 1}, {64, 1, 1}, 0, c.s, dctl.p(), a, reinterpret_cast<DevPtr>(hp), dtab.p());
        hip::streamSync(c.s);
        const std::vector<int> zc(hp, hp + n);
        hip::hostFree(hp);
        gz_all.insert(gz_all.end(), zc.begin(), zc.end());
        wz_all.insert(wz_all.end(), w.host.begin(), w.host.end());
        const std::vector<int> got = dctl.down<int>(n);
        gc_all.insert(gc_all.end(), got.begin(), got.end());
        wc_all.insert(wc_all.end(), w.ctl.begin(), w.ctl.end());
        const std::vector<int> gt = dtab.down<int>(sizeof(SpecNextTab) / 4);
        gt_all.insert(gt_all.end(), gt.begin(), gt.end());
        wt_all.insert(wt_all.end(), w.tab.begin(), w.tab.end());
    }
    c.rep.add(cmpExact("spec_accept_rows gpu-zc pinned host copy (hand + " + std::to_string(rounds) + " cycles)", gz_all, wz_all));
    c.rep.add(cmpExact("spec_accept_rows gpu-zc control words", gc_all, wc_all));
    c.rep.add(cmpExact("spec_accept_rows gpu-zc next-cycle rows", gt_all, wt_all));
}

}  // namespace kt

// ---------------------------------------------------------------------------------------------------------------------
// whirl-kernel-test family "specaccx" (v0252-e1fast, PLAN V1-93 P3; gfx1151 object only): spec_accept_relax_rows
// (kernels/gfx1151/spec_accept_relax.hip) against kernels/spec_accept_relax.h run on the CPU (the repetition guard on one thread),
// BIT FOR BIT, launched as the engine launches it:
//   chain   per sequence a chain of cycles, one relaxacc sequence per launch (an E1 cycle): the first on host inputs (the host
//           wrote the window into the ring), the rest on device inputs (NPOS / CHOSEN words; the ring only as the kernel
//           appended it); small vocabularies so the 6-gram guard fires; rank words with near-boundary alpha gaps; EOS and
//           max_tokens cuts. Every cycle: all 64 control words, the sequence's history ring (pinned host memory, its host
//           pointer as the device address - the engine's SampleBuffers::relaxHistAll), the rank-word copy (pinned) and the
//           next-cycle rows.
//   mixed   one launch with 1..6 sequences: relaxed ones, greedy exact ones (spec_accept_one) and host-decided ones (copy)
//           at random control areas - the block loops over them with the header's result per sequence.
#include "spec_accept_relax.h"  // kernels/ (v0252-e1fast)

template <>
struct whirl::hip::LaunchArgPtrWords<SpecRelaxArgs> : whirl::hip::LaunchArgNoPtrWords {};

namespace kt {

namespace {

constexpr int kXS = 32;  // rank-word stride (max rows)

struct XWant {
    std::vector<int> ctl, hist, rcopy, tab;
    int relaxed = 0, guarded = 0, cuts = 0;
};

// the CPU side of one launch (the header, as the mock and the engine's check mode run it)
XWant xReference(const SpecAccArgs& a, const SpecRelaxArgs& r, const std::vector<int>& ctl_in, const std::vector<int>& rank,
                 const std::vector<int>& hist_in, const std::vector<int>& rcopy_in) {
    XWant w;
    w.ctl = ctl_in;
    w.hist = hist_in;
    w.rcopy = rcopy_in;
    SpecAccResult res[WHIRL_SA_MAX_SEG];
    for (int k = 0; k < a.n; ++k) {
        int* cw = w.ctl.data() + a.seq[k] * kW;
        if (a.dev[k] && r.on[k]) {
            int* h = w.hist.data() + a.seq[k] * WHIRL_SR_HIST;
            const int pos = spec_pos(a, k, cw), next = spec_relax_next(a, r, k, cw);
            const int win = r.win < WHIRL_SR_WIN ? r.win : WHIRL_SR_WIN;
            SpecRelaxScalarGuard g{h, pos, pos < win ? pos : win, next, &a, k, cw, r.ngram};
            SpecRelaxOut ro{};
            const SpecAccResult o = spec_accept_relax_one(a, r, k, cw, rank.data(), g, ro);
            spec_store(a, k, o, cw);
            spec_relax_store(ro, cw);
            spec_relax_append(h, pos, next, a, k, cw, o.acc);
            for (int q = 0; q <= a.nd[k]; ++q)
                for (int t = 0; t < 3; ++t) w.rcopy[t * kXS + a.row0[k] + q] = rank[t * kXS + a.row0[k] + q];
            res[k] = o;
            for (unsigned m = ro.mask; m; m &= m - 1) ++w.relaxed;
            w.guarded += ro.guard > 0;
            w.cuts += (o.flags & (WHIRL_SA_F_EOS | WHIRL_SA_F_BUDGET)) != 0;
        } else {
            res[k] = a.dev[k] ? spec_accept_one(a, k, cw) : spec_copy_one(a, k, cw);
            spec_store(a, k, res[k], cw);
        }
    }
    SpecNextTab t{};
    spec_next_rows(a, w.ctl.data(), res, &t);
    w.tab.resize(sizeof(SpecNextTab) / 4);
    std::memcpy(w.tab.data(), &t, sizeof(SpecNextTab));
    return w;
}

// one relaxacc cycle's rows / drafts / rank words for sequence k (area sq), draft count nd, first verify row row0
void xRows(Ctx& c, SpecAccArgs& a, int k, int sq, int nd, int row0, std::uint32_t V, std::vector<int>& ctl, std::vector<int>& rank) {
    auto rnd = [&](std::uint32_t n) { return static_cast<std::uint32_t>(c.rng() % n); };
    int* cw = ctl.data() + sq * kW;
    a.seq[k] = sq;
    a.nd[k] = nd;
    a.row0[k] = row0;
    a.dev[k] = 1;
    for (int q = 0; q <= nd; ++q) cw[q] = static_cast<int>(rnd(V));
    for (int q = 0; q < nd; ++q) cw[16 + q] = rnd(2) ? cw[q] : static_cast<int>(rnd(V));
    cw[48] = rnd(5) == 0 ? static_cast<int>(rnd(static_cast<std::uint32_t>(nd) + 1)) : nd;
    for (int q = 0; q <= nd; ++q) {
        const float m = 5.0f + static_cast<float>(rnd(1000)) / 37.0f;
        float ld = m - static_cast<float>(rnd(400)) / 100.0f;
        if (rnd(4) == 0) {  // at the alpha boundary (0.05: ln = -2.9957), a few float steps either way
            ld = m + static_cast<float>(std::log(0.05));
            for (int s = static_cast<int>(rnd(7)) - 3; s != 0; s += s > 0 ? -1 : 1) ld = std::nextafter(ld, s > 0 ? 1e30f : -1e30f);
        }
        rank[row0 + q] = static_cast<int>(rnd(9));
        rank[kXS + row0 + q] = spec_relax_bits(m);
        rank[2 * kXS + row0 + q] = spec_relax_bits(ld);
    }
}

}  // namespace

void testSpecAccRelax(Ctx& c) {
    if (!c.k.spec_accept_relax_rows) {
        c.rep.skip("specaccx", "spec_accept_relax_rows", "kernel not present (gfx1151 object only)");
        return;
    }
    hip::Function f = c.k.spec_accept_relax_rows;
    auto rnd = [&](std::uint32_t n) { return static_cast<std::uint32_t>(c.rng() % n); };
    const std::size_t hist_words = static_cast<std::size_t>(kAreas) * WHIRL_SR_HIST;
    auto* hist = static_cast<int*>(hip::hostMalloc(hist_words * sizeof(int)));
    auto* rcopy = static_cast<int*>(hip::hostMalloc(3 * kXS * sizeof(int)));
    SpecRelaxArgs r{};
    r.dstar_bits = spec_relax_dbits(spec_relax_dstar(0.05));
    r.topk = 6;
    r.stride = kXS;
    r.win = WHIRL_SR_WIN;
    r.ngram = WHIRL_SR_NGRAM;
    std::vector<int> g_ctl, w_ctl, g_hist, w_hist, g_rc, w_rc, g_tab, w_tab;
    int relaxed = 0, guarded = 0, cuts = 0, cycles = 0;
    const int seqs = c.quick ? 6 : 24, chain = c.quick ? 12 : 30;
    // ---- chain: one relaxacc sequence per launch, cycles in a row
    for (int sqi = 0; sqi < seqs; ++sqi) {
        const std::uint32_t V = 3 + rnd(10);
        const int sq = static_cast<int>(rnd(kAreas));
        std::vector<int> ctl(static_cast<std::size_t>(kAreas) * kW, 0), hv(hist_words, 0);
        for (int& v : ctl) v = static_cast<int>(rnd(V));
        // the host's history (cache_tokens) and its window in the ring
        std::vector<int> cache(rnd(9000));
        for (int& t : cache) t = static_cast<int>(rnd(V));
        for (std::size_t i = cache.size() > WHIRL_SR_WIN ? cache.size() - WHIRL_SR_WIN : 0; i < cache.size(); ++i)
            hv[static_cast<std::size_t>(sq) * WHIRL_SR_HIST + (i & (WHIRL_SR_HIST - 1))] = cache[i];
        std::memcpy(hist, hv.data(), hist_words * sizeof(int));
        const int next = static_cast<int>(rnd(V));
        for (int cyc = 0; cyc < chain; ++cyc) {
            SpecAccArgs a{};
            a.n = 1;
            a.n_vocab = static_cast<int>(V);
            a.stop0 = rnd(3) == 0 ? 1u : 0xffffffffu;
            a.stop1 = 0xffffffffu;
            a.eos[0] = a.stop0 != 0xffffffffu ? 1 : 0;
            a.budget[0] = rnd(4) == 0 ? 1 + static_cast<int>(rnd(4)) : 1 << 30;
            a.kvbase[0] = 64;
            std::vector<int> rank(3 * kXS, 0);
            xRows(c, a, 0, sq, 1 + static_cast<int>(rnd(4)), static_cast<int>(rnd(3)), V, ctl, rank);
            SpecRelaxArgs rr = r;
            rr.on[0] = 1;
            a.pos[0] = cyc == 0 ? static_cast<int>(cache.size()) : -1;
            rr.next[0] = cyc == 0 ? next : -12345;
            for (int i = 0; i < 3 * kXS; ++i) rcopy[i] = -1;
            const std::vector<int> rc_in(rcopy, rcopy + 3 * kXS), hist_in(hist, hist + hist_words);
            const XWant w = xReference(a, rr, ctl, rank, hist_in, rc_in);
            Buf dctl(ctl), drank(rank), dtab(sizeof(SpecNextTab));
            dtab.zero();
            hip::launch(f, {1, 1, 1}, {256, 1, 1}, 0, c.s, dctl.p(), a, DevPtr{0}, dtab.p(), rr, drank.p(), reinterpret_cast<DevPtr>(hist),
                        reinterpret_cast<DevPtr>(rcopy));
            hip::streamSync(c.s);
            const std::vector<int> got = dctl.down<int>(ctl.size());
            const std::vector<int> gh(hist, hist + hist_words), grc(rcopy, rcopy + 3 * kXS);
            const std::vector<int> gt = dtab.down<int>(sizeof(SpecNextTab) / 4);
            g_ctl.insert(g_ctl.end(), got.begin(), got.end());
            w_ctl.insert(w_ctl.end(), w.ctl.begin(), w.ctl.end());
            g_hist.insert(g_hist.end(), gh.begin() + sq * WHIRL_SR_HIST, gh.begin() + (sq + 1) * WHIRL_SR_HIST);
            w_hist.insert(w_hist.end(), w.hist.begin() + sq * WHIRL_SR_HIST, w.hist.begin() + (sq + 1) * WHIRL_SR_HIST);
            g_rc.insert(g_rc.end(), grc.begin(), grc.end());
            w_rc.insert(w_rc.end(), w.rcopy.begin(), w.rcopy.end());
            g_tab.insert(g_tab.end(), gt.begin(), gt.end());
            w_tab.insert(w_tab.end(), w.tab.begin(), w.tab.end());
            relaxed += w.relaxed;
            guarded += w.guarded;
            cuts += w.cuts;
            ++cycles;
            // the next cycle reads the words this one left (the CPU reference's, so one device miss does not cascade)
            ctl = w.ctl;
            std::memcpy(hist, w.hist.data(), hist_words * sizeof(int));
        }
    }
    // ---- mixed: several sequences of one launch (relaxed / exact / host-decided)
    std::vector<int> m_ctl, mw_ctl, m_hist, mw_hist;
    const int mixes = c.quick ? 16 : 64;
    for (int it = 0; it < mixes; ++it) {
        const std::uint32_t V = 4 + rnd(8);
        std::vector<int> ctl(static_cast<std::size_t>(kAreas) * kW, 0), rank(3 * kXS, 0);
        for (int& v : ctl) v = static_cast<int>(rnd(V));
        std::vector<int> hv(hist_words);
        for (int& v : hv) v = static_cast<int>(rnd(V));
        std::memcpy(hist, hv.data(), hist_words * sizeof(int));
        SpecAccArgs a{};
        SpecRelaxArgs rr = r;
        const int A = 1 + static_cast<int>(rnd(6));
        a.n = A;
        a.n_vocab = static_cast<int>(V);
        a.stop0 = 0xffffffffu;
        a.stop1 = 0xffffffffu;
        int ids[kAreas];
        for (int i = 0; i < kAreas; ++i) ids[i] = i;
        for (int i = kAreas - 1; i > 0; --i) std::swap(ids[i], ids[rnd(static_cast<std::uint32_t>(i) + 1)]);
        int row = 0;
        for (int k = 0; k < A; ++k) {
            const int nd = 1 + static_cast<int>(rnd(3));
            xRows(c, a, k, ids[k], nd, row, V, ctl, rank);
            row += nd + 1;
            a.budget[k] = 1 << 30;
            a.pos[k] = static_cast<int>(rnd(12000));
            const std::uint32_t kind = rnd(3);
            rr.on[k] = kind == 0 ? 1 : 0;
            a.dev[k] = kind == 2 ? 0 : 1;
            rr.next[k] = static_cast<int>(rnd(V));
        }
        const std::vector<int> hist_in(hist, hist + hist_words), rc_in(3 * kXS, -1);
        for (int i = 0; i < 3 * kXS; ++i) rcopy[i] = -1;
        const XWant w = xReference(a, rr, ctl, rank, hist_in, rc_in);
        Buf dctl(ctl), drank(rank), dtab(sizeof(SpecNextTab));
        dtab.zero();
        hip::launch(f, {1, 1, 1}, {256, 1, 1}, 0, c.s, dctl.p(), a, DevPtr{0}, dtab.p(), rr, drank.p(), reinterpret_cast<DevPtr>(hist),
                    reinterpret_cast<DevPtr>(rcopy));
        hip::streamSync(c.s);
        const std::vector<int> got = dctl.down<int>(ctl.size());
        m_ctl.insert(m_ctl.end(), got.begin(), got.end());
        mw_ctl.insert(mw_ctl.end(), w.ctl.begin(), w.ctl.end());
        m_hist.insert(m_hist.end(), hist, hist + hist_words);
        mw_hist.insert(mw_hist.end(), w.hist.begin(), w.hist.end());
        relaxed += w.relaxed;
        guarded += w.guarded;
    }
    hip::hostFree(hist);
    hip::hostFree(rcopy);
    const std::string n = std::to_string(cycles);
    c.rep.add(cmpExact("spec_accept_relax_rows chain control words (" + n + " cycles)", g_ctl, w_ctl));
    c.rep.add(cmpExact("spec_accept_relax_rows chain history ring (pinned)", g_hist, w_hist));
    c.rep.add(cmpExact("spec_accept_relax_rows chain rank-word copy (pinned)", g_rc, w_rc));
    c.rep.add(cmpExact("spec_accept_relax_rows chain next-cycle rows", g_tab, w_tab));
    c.rep.add(cmpExact("spec_accept_relax_rows mixed control words (" + std::to_string(mixes) + " launches)", m_ctl, mw_ctl));
    c.rep.add(cmpExact("spec_accept_relax_rows mixed history rings", m_hist, mw_hist));
    Result res;
    res.family = "specaccx";
    res.name = "spec_accept_relax_rows branches hit (relaxed keeps " + std::to_string(relaxed) + ", guard refusals " + std::to_string(guarded) +
               ", EOS / budget cuts " + std::to_string(cuts) + ")";
    res.pass = relaxed > 0 && guarded > 0 && cuts > 0;
    res.n = static_cast<std::size_t>(cycles);
    c.rep.add(res);
}

}  // namespace kt
