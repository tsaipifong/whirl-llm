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
