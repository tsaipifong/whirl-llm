// WHIRL v0252-e1fast (PLAN V1-93 P3): the relaxacc accept decision of a verify ON THE DEVICE, so a fast-mode cycle can be
// submitted one cycle ahead (E1, WHIRL_E1=2). Shared by the device kernel spec_accept_relax_rows (kernels/gfx1151/spec_accept_relax.hip,
// gfx1151 set only), the host (the engine's check mode, the mock model of the server tests, the unit tests).
// SPDX-License-Identifier: Apache-2.0
//
// The rule is the host's, unchanged (fast mode's task accuracy was measured with it): sampling::Acceptor::greedy with relax
// (src/sampling/acceptor.cpp) on the v0252-f1 one-pass rank words, then the EOS / max_tokens cuts (as spec_accept_one):
//   draft r == the row's argmax                                  -> kept (exact)
//   else rank(r) < topk and exp(double(l_d) - double(m)) >= alpha -> kept (relaxed), unless the repetition guard says the
//        kept draft would end a 6-gram that already occurs in the last 4096 tokens of the history (relax::extendsRepeat)
//   else stop.
// Two parts need care to be EXACTLY the host's decision:
//  - the alpha test: the device has no copy of the host's libm exp. The host finds d* = the smallest double whose std::exp is
//    >= alpha (spec_relax_dstar, bisection over the double bit pattern); the device compares d = double(l_d) - double(m) (the
//    same IEEE subtraction as the host) with d*. Equal for every d while the host's exp is monotone (the unit test sweeps it).
//  - the repetition guard needs the slot's token history, which only the host kept: a per-slot ring of WHIRL_SR_HIST tokens
//    indexed by history position (= the slot's cache_tokens index; pos == cache_tokens.size() is a precondition) in pinned
//    host memory. The host writes the window before a cycle with host inputs (a.pos >= 0); every relax cycle the kernel
//    decides appends what the host appends to cache_tokens (next and the kept drafts, after the cuts), so a cycle on device
//    inputs (a.pos < 0, the lookahead) reads a complete window. Prior art: TensorRT-LLM's mtpRelaxedAcceptanceKernel keeps its
//    per-slot relaxed state on the device and updates it from the accepted tokens (progress/v0252-e1fast.PRIOR_ART.md).
#pragma once

#include "spec_accept.h"

// result words of a relaxed sequence (spec_accept_rows leaves 58 .. 63 unused)
#define WHIRL_SR_CTL_RMASK 58  // drafts kept by the relaxed rule, before the cuts (bit r: draft r)
#define WHIRL_SR_CTL_GUARD 59  // 1 + the draft whose relaxed keep the repetition guard refused (0: none)
#define WHIRL_SR_CTL_GAP 60    // [60, 64): float bits of m - l_d of the first WHIRL_SR_GAP_N relaxed drafts (the host's relax_gap)
#define WHIRL_SR_GAP_N 4
#define WHIRL_SR_F_RELAX 8     // FLAGS: decided by the relaxed rule on the device
// the history ring of one slot: a power of 2 >= window + drafts + 2 (positions pos - window .. pos + nd never alias)
#define WHIRL_SR_HIST 8192
#define WHIRL_SR_WIN 4096      // relax::extendsRepeat's window
#define WHIRL_SR_NGRAM 6       // relax::extendsRepeat's n

// kernel argument (by value; plain integers, no device address: the pointers are separate kernel arguments). No padding and no
// float member (whirl/hip.h LaunchArgNoPadding: the launch log digests every byte): d* travels as its 64-bit pattern; the
// layout (152 bytes, offsets) is pinned in tests/server/e1_fast_tests.inc and read by kt specaccx
struct SpecRelaxArgs {
    long long dstar_bits;          // d* as a double's bit pattern: keep iff double(l_d) - double(m) >= d* (spec_relax_dval)
    int on[WHIRL_SA_MAX_SEG];      // 1: relaxacc applies to this sequence (greedy, MTP drafts) and it is decided here
    int next[WHIRL_SA_MAX_SEG];    // the token `next` of a sequence on host inputs (a.pos[k] >= 0); else the CHOSEN word
    int topk;                      // relax::Params::topk()
    int stride;                    // rank words: [stride] i32 rank, [stride] f32 row max, [stride] f32 draft logit
    int win;                       // WHIRL_SR_WIN (the tests shrink it)
    int ngram;                     // WHIRL_SR_NGRAM
};

// per-sequence outputs besides SpecAccResult
struct SpecRelaxOut {
    unsigned mask;                 // WHIRL_SR_CTL_RMASK
    int guard;                     // WHIRL_SR_CTL_GUARD
    int n_gap;
    float gap[WHIRL_SR_GAP_N];
};

WHIRL_SA_HD inline float spec_relax_f(int bits) {
    union {
        int i;
        float f;
    } u;
    u.i = bits;
    return u.f;
}
WHIRL_SA_HD inline int spec_relax_bits(float f) {
    union {
        int i;
        float f;
    } u;
    u.f = f;
    return u.i;
}

// ---- private copies of the spec_accept.h helpers for the relaxed entry. A second caller of the SHARED inline helpers changed
// the code of the shipping spec_accept_rows (pre-compare 2026-11-25 14:12, scratch/v0252_e1fast/co/pre_compare_1403.txt: same
// instruction count, other register allocation) - the e1s8 lesson (gdn_select.hip keeps private bodies). Token for token the
// spec_accept.h functions; e1_fast_rule pins each copy against its original on random inputs, kt specaccx pins the kernel.
WHIRL_SA_HD inline unsigned sr_draft(const SpecAccArgs& a, int k, const int* ctl, int r) {
    return a.lit[k] ? (unsigned)a.lit_d[k][r] : (unsigned)ctl[WHIRL_SA_CTL_DRAFTS + r];
}
WHIRL_SA_HD inline int sr_pos(const SpecAccArgs& a, int k, const int* ctl) { return a.pos[k] >= 0 ? a.pos[k] : ctl[WHIRL_SA_CTL_NPOS]; }
WHIRL_SA_HD inline SpecAccResult sr_copy_one(const SpecAccArgs& a, int k, const int* ctl) {
    SpecAccResult o;
    o.acc0 = 0;
    o.acc = 0;
    o.chosen = ctl[WHIRL_SA_CTL_ROWS];
    o.nd_dev = 0;
    o.flags = 0;
    o.pos = sr_pos(a, k, ctl);
    return o;
}
WHIRL_SA_HD inline SpecAccResult sr_accept_one(const SpecAccArgs& a, int k, const int* ctl) {
    SpecAccResult o;
    o.pos = sr_pos(a, k, ctl);
    const int nd_k = a.nd[k];
    int nd_dev = nd_k;
    if (!a.lit[k]) {
        const int kept = ctl[WHIRL_SA_CTL_ND] > 0 ? ctl[WHIRL_SA_CTL_ND] : 0;
        nd_dev = nd_k > 0 ? (nd_k < kept ? nd_k : kept) : 0;
    }
    int acc = 0;
    while (acc < nd_dev && (unsigned)ctl[WHIRL_SA_CTL_ROWS + acc] == sr_draft(a, k, ctl, acc)) ++acc;
    o.acc0 = acc;
    unsigned chosen = (unsigned)ctl[WHIRL_SA_CTL_ROWS + acc];
    int flags = WHIRL_SA_F_DEV;
    if (a.eos[k]) {
        for (int r = 0; r < acc; ++r) {
            const unsigned d = sr_draft(a, k, ctl, r);
            if (d == a.stop0 || d == a.stop1) {
                chosen = d;
                acc = r;
                flags |= WHIRL_SA_F_EOS;
                break;
            }
        }
    }
    const int budget = a.budget[k];
    if (budget >= 1 && acc + 1 > budget) {
        chosen = sr_draft(a, k, ctl, budget - 1);
        acc = budget - 1;
        flags |= WHIRL_SA_F_BUDGET;
    }
    o.acc = acc;
    o.chosen = (int)chosen;
    o.nd_dev = nd_dev;
    o.flags = flags;
    return o;
}
WHIRL_SA_HD inline int sr_lookup(const SpecAccArgs& a, unsigned t) { return t < (unsigned)a.n_vocab ? (int)t : 0; }
WHIRL_SA_HD inline void sr_next_rows(const SpecAccArgs& a, const int* ctl_all, const SpecAccResult* res, SpecNextTab* t) {
    int n = 0;
    for (int k = 0; k < a.n; ++k) {
        const int* ctl = ctl_all + a.seq[k] * WHIRL_SA_CTL_WORDS;
        const int acc = res[k].acc;
        for (int j = 0; j <= acc && n < WHIRL_SA_MAX_ROWS; ++j) {
            const unsigned tok = j < acc ? sr_draft(a, k, ctl, j) : (unsigned)res[k].chosen;
            t->mtp_tok[n] = sr_lookup(a, tok);
            t->mtp_pos[n] = res[k].pos + 1 + j;
            t->mtp_base[n] = a.kvbase[k];
            ++n;
        }
        t->mtp_last[k] = n - 1;
        t->ver_tok[k] = sr_lookup(a, (unsigned)res[k].chosen);
        t->ver_pos[k] = res[k].pos + acc + 1;
        t->ver_base[k] = a.kvbase[k];
    }
    t->n_mtp = n;
}
WHIRL_SA_HD inline void sr_store(const SpecAccArgs& a, int k, const SpecAccResult& r, int* ctl) {
    ctl[WHIRL_SA_CTL_ACC] = r.acc;
    ctl[WHIRL_SA_CTL_CHOSEN] = r.chosen;
    ctl[WHIRL_SA_CTL_NDDEV] = r.nd_dev;
    ctl[WHIRL_SA_CTL_FLAGS] = r.flags;
    ctl[WHIRL_SA_CTL_NPOS] = r.pos + r.acc + 1;
    ctl[WHIRL_SA_CTL_NROW] = a.row0[k] + r.acc;
    ctl[WHIRL_SA_CTL_ACC0] = r.acc0;
    ctl[WHIRL_SA_CTL_NPEND] = r.acc + 1;
}

// d* <-> its bit pattern (SpecRelaxArgs::dstar_bits)
WHIRL_SA_HD inline double spec_relax_dval(long long bits) {
    union {
        long long i;
        double d;
    } u;
    u.i = bits;
    return u.d;
}
WHIRL_SA_HD inline long long spec_relax_dbits(double d) {
    union {
        long long i;
        double d;
    } u;
    u.d = d;
    return u.i;
}

// the relaxed test of verify row gr (relax::acceptRank with the alpha test as d >= d*); gap = m - l_d (float, as the host's)
WHIRL_SA_HD inline bool spec_relax_row(const SpecRelaxArgs& r, const int* rank, int gr, float* gap) {
    const int rk = rank[gr];
    if (rk < 0 || rk >= r.topk) return false;
    const float m = spec_relax_f(rank[r.stride + gr]);
    const float ld = spec_relax_f(rank[2 * r.stride + gr]);
    *gap = m - ld;
    return (double)ld - (double)m >= spec_relax_dval(r.dstar_bits);
}

// the token `next` of sequence k: by value (host inputs), or the CHOSEN word its last accept wrote (read before spec_store)
WHIRL_SA_HD inline int spec_relax_next(const SpecAccArgs& a, const SpecRelaxArgs& r, int k, const int* ctl) {
    return a.pos[k] >= 0 ? r.next[k] : ctl[WHIRL_SA_CTL_CHOSEN];
}

// element j of the guard's sequence: the last hn history tokens (ring), then next, then the drafts
WHIRL_SA_HD inline int spec_relax_v(const int* h, int pos, int hn, int next, const SpecAccArgs& a, int k, const int* ctl, int j) {
    if (j < hn) return h[(pos - hn + j) & (WHIRL_SR_HIST - 1)];
    if (j == hn) return next;
    return (int)sr_draft(a, k, ctl, j - hn - 1);
}

// the repetition guard on one thread (host, mock, tests): relax::extendsRepeat(window of the history, [next, d0 .. d_acc])
struct SpecRelaxScalarGuard {
    const int* h;
    int pos, hn, next;
    const SpecAccArgs* a;
    int k;
    const int* ctl;
    int ngram;
    WHIRL_SA_HD bool operator()(int acc) const {
        const int s = hn + acc + 2, n = ngram;
        if (s < n + 1) return false;
        for (int i = 0; i + n < s; ++i) {
            int j = 0;
            while (j < n && spec_relax_v(h, pos, hn, next, *a, k, ctl, i + j) == spec_relax_v(h, pos, hn, next, *a, k, ctl, s - n + j)) ++j;
            if (j == n) return true;
        }
        return false;
    }
};

// relaxacc greedy accept of sequence k (spec_accept_one + the relaxed branch); guard(acc) = the repetition guard for draft acc
template <class Guard>
WHIRL_SA_HD inline SpecAccResult spec_accept_relax_one(const SpecAccArgs& a, const SpecRelaxArgs& r, int k, const int* ctl, const int* rank,
                                                       Guard& guard, SpecRelaxOut& ro) {
    SpecAccResult o;
    o.pos = sr_pos(a, k, ctl);
    ro.mask = 0;
    ro.guard = 0;
    ro.n_gap = 0;
    const int nd_k = a.nd[k];
    int nd_dev = nd_k;
    if (!a.lit[k]) {
        const int kept = ctl[WHIRL_SA_CTL_ND] > 0 ? ctl[WHIRL_SA_CTL_ND] : 0;
        nd_dev = nd_k > 0 ? (nd_k < kept ? nd_k : kept) : 0;
    }
    int acc = 0;
    while (acc < nd_dev) {
        if ((unsigned)ctl[WHIRL_SA_CTL_ROWS + acc] == sr_draft(a, k, ctl, acc)) {
            ++acc;
            continue;
        }
        if (rank == nullptr || a.lit[k]) break;
        float g = 0.0f;
        if (!spec_relax_row(r, rank, a.row0[k] + acc, &g)) break;
        if (guard(acc)) {
            ro.guard = acc + 1;
            break;
        }
        ro.mask |= 1u << acc;
        if (ro.n_gap < WHIRL_SR_GAP_N) ro.gap[ro.n_gap++] = g;
        ++acc;
    }
    o.acc0 = acc;
    unsigned chosen = (unsigned)ctl[WHIRL_SA_CTL_ROWS + acc];
    int flags = WHIRL_SA_F_DEV | WHIRL_SR_F_RELAX;
    if (a.eos[k]) {
        for (int q = 0; q < acc; ++q) {
            const unsigned d = sr_draft(a, k, ctl, q);
            if (d == a.stop0 || d == a.stop1) {
                chosen = d;
                acc = q;
                flags |= WHIRL_SA_F_EOS;
                break;
            }
        }
    }
    const int budget = a.budget[k];
    if (budget >= 1 && acc + 1 > budget) {
        chosen = sr_draft(a, k, ctl, budget - 1);
        acc = budget - 1;
        flags |= WHIRL_SA_F_BUDGET;
    }
    o.acc = acc;
    o.chosen = (int)chosen;
    o.nd_dev = nd_dev;
    o.flags = flags;
    return o;
}

WHIRL_SA_HD inline void spec_relax_store(const SpecRelaxOut& ro, int* ctl) {
    ctl[WHIRL_SR_CTL_RMASK] = (int)ro.mask;
    ctl[WHIRL_SR_CTL_GUARD] = ro.guard;
    for (int i = 0; i < WHIRL_SR_GAP_N; ++i) ctl[WHIRL_SR_CTL_GAP + i] = i < ro.n_gap ? spec_relax_bits(ro.gap[i]) : 0;
}

// what the host appends to cache_tokens after the cycle (next, then the kept drafts after the cuts) into the ring
WHIRL_SA_HD inline void spec_relax_append(int* h, int pos, int next, const SpecAccArgs& a, int k, const int* ctl, int acc) {
    h[pos & (WHIRL_SR_HIST - 1)] = next;
    for (int j = 0; j < acc; ++j) h[(pos + 1 + j) & (WHIRL_SR_HIST - 1)] = (int)sr_draft(a, k, ctl, j);
}

#if !defined(__HIP_DEVICE_COMPILE__)
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

// The smallest double d with std::exp(d) >= alpha (host libm), so that "double(l_d) - double(m) >= d*" is the host's
// "std::exp(double(l_d) - double(m)) >= alpha" for every d while std::exp is monotone. alpha <= 0: -inf (always); NaN or an
// alpha no exp reaches: +inf (never; the comparison with a NaN d is false on both sides).
inline double spec_relax_dstar(double alpha) {
    if (alpha != alpha) return std::numeric_limits<double>::infinity();
    if (!(alpha > 0.0)) return -std::numeric_limits<double>::infinity();
    // order-preserving map double <-> uint64
    auto key = [](double x) {
        std::uint64_t b;
        std::memcpy(&b, &x, 8);
        return (b >> 63) ? ~b : (b | 0x8000000000000000ull);
    };
    auto val = [](std::uint64_t k) {
        const std::uint64_t b = (k >> 63) ? (k & 0x7fffffffffffffffull) : ~k;
        double x;
        std::memcpy(&x, &b, 8);
        return x;
    };
    std::uint64_t lo = key(-std::numeric_limits<double>::infinity()), hi = key(std::numeric_limits<double>::infinity());
    if (!(std::exp(val(hi)) >= alpha)) return std::numeric_limits<double>::infinity();
    if (std::exp(val(lo)) >= alpha) return -std::numeric_limits<double>::infinity();
    while (hi - lo > 1) {  // invariant: exp(val(lo)) < alpha <= exp(val(hi))
        const std::uint64_t mid = lo + (hi - lo) / 2;
        if (std::exp(val(mid)) >= alpha) hi = mid;
        else lo = mid;
    }
    return val(hi);
}

// the relaxed launch's arguments without the per-sequence part (on / next): d*, the rank limit, the rank-word stride
inline SpecRelaxArgs spec_relax_args(double dstar, int topk, int stride) {
    SpecRelaxArgs r{};
    r.dstar_bits = spec_relax_dbits(dstar);
    r.topk = topk;
    r.stride = stride;
    r.win = WHIRL_SR_WIN;
    r.ngram = WHIRL_SR_NGRAM;
    return r;
}

// a slot whose history index is its position (the rings are indexed by position; pos == cache_tokens.size()) and whose
// positions fit the kernel's int arithmetic
inline bool spec_relax_pos_ok(std::uint64_t pos, std::uint64_t hist_n) { return pos == hist_n && pos < (1ull << 30); }

// the host's part of a slot's ring h (WHIRL_SR_HIST words): the last WHIRL_SR_WIN tokens of its history at their positions.
// Only before a cycle on host inputs (SpecAccArgs::pos >= 0), when no accept of the slot is in flight; a cycle on device
// inputs reads what the accepts before it appended
inline void spec_relax_hist_write(int* h, const std::uint32_t* toks, std::size_t n) {
    if (h == nullptr) return;
    const std::size_t lo = n > WHIRL_SR_WIN ? n - WHIRL_SR_WIN : 0;
    for (std::size_t i = lo; i < n; ++i) h[i & (WHIRL_SR_HIST - 1)] = (int)toks[i];
}
#endif
