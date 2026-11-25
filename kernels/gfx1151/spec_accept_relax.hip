// WHIRL device code (gfx1151 set only; v0252-e1fast, PLAN V1-93 P3): spec_accept_rows with the relaxacc rule decided on the
// device, so a fast-mode cycle can be submitted one cycle ahead (E1). The rule and its exactness argument: spec_accept_relax.h.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included LAST by whirl_kernels_gfx1151.hip (this directory) (no existing kernel moves; spec_accept_rows and
// every other shipping entry stay byte-identical - this file only adds the entry).
//
// One block of 256 threads; the sequences of the verify one after the other (an E1 cycle has one). Per sequence:
//  - not decided here (a.dev[k] == 0): copied as spec_accept_rows does; greedy exact (r.on[k] == 0): spec_accept_one (the
//    header's private copies sr_*: no code shared with spec_accept_rows, so its code object entry stays byte-identical);
//  - relaxacc (r.on[k] == 1): spec_accept_relax_one, the repetition guard over the history ring with the whole block: the
//    window (<= r.win tokens), next and the drafts are loaded into LDS once (only when a relaxed keep needs the guard), every
//    thread tests a share of the start positions, __syncthreads_or joins them;
//  - thread 0 stores the result words (spec_store + spec_relax_store), appends the history ring and (rhost != null, the
//    lookahead's check mode) copies the sequence's rank words for the host.
// Then the next-cycle rows (tab) and the zero-copy control areas (host), as spec_accept_rows.

#include "../spec_accept_relax.h"

struct SpecRelaxBlockGuard {
    const int* h;
    int pos, hn, next;
    const SpecAccArgs* a;
    int k;
    const int* ctl;
    int ngram;
    int* sv;
    bool loaded;
    __device__ bool operator()(int acc) {
        if (!loaded) {
            const int len = hn + 1 + a->nd[k];
            for (int j = threadIdx.x; j < len; j += blockDim.x) sv[j] = spec_relax_v(h, pos, hn, next, *a, k, ctl, j);
            __syncthreads();
            loaded = true;
        }
        const int s = hn + acc + 2, n = ngram;
        int hit = 0;
        if (s >= n + 1) {
            for (int i = threadIdx.x; i + n < s; i += blockDim.x) {
                int j = 0;
                while (j < n && sv[i + j] == sv[s - n + j]) ++j;
                if (j == n) hit = 1;
            }
        }
        return __syncthreads_or(hit) != 0;
    }
};

extern "C" __global__ void __launch_bounds__(256) spec_accept_relax_rows(int* ctl_all, SpecAccArgs a, int* host, SpecNextTab* tab, SpecRelaxArgs r,
                                                                          const int* rank, int* hist, int* rhost) {
    __shared__ int sv[WHIRL_SR_WIN + 1 + WHIRL_SA_MAX_ND];
    __shared__ SpecAccResult res[WHIRL_SA_MAX_SEG];
    for (int k = 0; k < a.n; ++k) {
        int* ctl = ctl_all + a.seq[k] * WHIRL_SA_CTL_WORDS;
        const bool rel = a.dev[k] != 0 && r.on[k] != 0;
        SpecAccResult o;
        SpecRelaxOut ro;
        ro.mask = 0;
        ro.guard = 0;
        ro.n_gap = 0;
        int pos = 0, next = 0;
        int* h = hist + a.seq[k] * WHIRL_SR_HIST;
        if (rel) {
            pos = sr_pos(a, k, ctl);
            next = spec_relax_next(a, r, k, ctl);
            const int win = r.win < WHIRL_SR_WIN ? r.win : WHIRL_SR_WIN;
            SpecRelaxBlockGuard g{h, pos, pos < win ? pos : win, next, &a, k, ctl, r.ngram, sv, false};
            o = spec_accept_relax_one(a, r, k, ctl, rank, g, ro);
        } else {
            o = a.dev[k] ? sr_accept_one(a, k, ctl) : sr_copy_one(a, k, ctl);
        }
        __syncthreads();  // every thread has read the words (NPOS, CHOSEN, drafts) before thread 0 overwrites them
        if (threadIdx.x == 0) {
            sr_store(a, k, o, ctl);
            if (rel) {
                spec_relax_store(ro, ctl);
                spec_relax_append(h, pos, next, a, k, ctl, o.acc);
                if (rhost != nullptr && rank != nullptr) {
                    for (int q = 0; q <= a.nd[k]; ++q) {
                        const int gr = a.row0[k] + q;
                        rhost[gr] = rank[gr];
                        rhost[r.stride + gr] = rank[r.stride + gr];
                        rhost[2 * r.stride + gr] = rank[2 * r.stride + gr];
                    }
                }
            }
            res[k] = o;
        }
        __syncthreads();
    }
    if (tab != nullptr && threadIdx.x == 0) sr_next_rows(a, ctl_all, res, tab);
    if (host != nullptr) {
        for (int i = threadIdx.x; i < a.n * WHIRL_SA_CTL_WORDS; i += blockDim.x) {
            const int off = a.seq[i / WHIRL_SA_CTL_WORDS] * WHIRL_SA_CTL_WORDS + i % WHIRL_SA_CTL_WORDS;
            host[off] = ctl_all[off];
        }
    }
    __threadfence_system();  // the zero-copy words, the history ring and the rank copy (pinned host memory)
}
