// WHIRL device code (gfx1201 and gfx1151): feat026-acc accept decision of a batched verify on the GPU (ported to int0252 by
// v0252-e1p1, PLAN V1-93 step P1a; unchanged from feat026-acc 33c5da1 / feat026-acc-p25 4ac9da6).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included last by whirl_kernels.hip (both kernel sets; plain int code; last, so no existing kernel
// moves in either code object).
//
// spec_accept_rows runs right after the verify's argmax (one block, thread k = sequence k):
//  - greedy exact sequences (a.dev[k]): accepted count, next token, EOS / max_tokens cuts (spec_accept_one, the host path
//    in kernels/spec_accept.h) into the result words 50 .. 56 of the sequence's control area;
//  - tab (may be null): the next cycle's MTP step-0 rows and verify row 0 (spec_next_rows) in a device SpecNextTab, for the
//    host-free cycle (E1 P1b / P1c);
//  - host (may be null): the control areas of the verify's sequences copied into pinned host memory (zero copy; same layout
//    as Context::readCtl), so the host needs no D2H copy command (WHIRL_SPEC_ACC=gpu-zc).

#include "spec_accept.h"

extern "C" __global__ void __launch_bounds__(64) spec_accept_rows(int *ctl_all, SpecAccArgs a, int *host, SpecNextTab *tab) {
    __shared__ SpecAccResult res[WHIRL_SA_MAX_SEG];
    const int k = threadIdx.x;
    if (k < a.n) {
        int *ctl = ctl_all + a.seq[k] * WHIRL_SA_CTL_WORDS;
        SpecAccResult r;
        if (a.dev[k]) {
            r = spec_accept_one(a, k, ctl);
        } else {
            r.acc0 = 0;
            r.acc = 0;
            r.chosen = ctl[WHIRL_SA_CTL_ROWS];
            r.nd_dev = 0;
            r.flags = 0;
        }
        spec_store(a, k, r, ctl);
        res[k] = r;
    }
    __syncthreads();
    if (tab != nullptr && threadIdx.x == 0) spec_next_rows(a, ctl_all, res, tab);
    if (host != nullptr) {
        for (int i = threadIdx.x; i < a.n * WHIRL_SA_CTL_WORDS; i += blockDim.x) {
            const int off = a.seq[i / WHIRL_SA_CTL_WORDS] * WHIRL_SA_CTL_WORDS + i % WHIRL_SA_CTL_WORDS;
            host[off] = ctl_all[off];
        }
        __threadfence_system();
    }
}
