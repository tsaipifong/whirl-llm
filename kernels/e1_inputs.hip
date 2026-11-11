// WHIRL device code (gfx1201 and gfx1151): E1 next-cycle inputs on the device (v0252-e1p1b, PLAN V1-93 step P1b).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included LAST by whirl_kernels.hip (both kernel sets, after spec_accept.hip: no existing kernel moves
// in either code object). Each is the device-value twin of a by-value kernel of kernels/shared/sample_pick_set_tokens.h; a
// launch whose device words hold the values the by-value launch passes writes the same bits (kt family "e1in").
//  set_rows_dp          set_rows plus, per row, a device position base: pos[i] = tab.pos[i] + dev_src[pdev.v[i]] (pdev.v[i] < 0:
//                       tab.pos[i] alone, = set_rows). E1: the verify rows and the MTP draft steps r >= 1 sit at the NPOS word
//                       of the sequence's last spec_accept_rows + r; tokens keep RowTab's negative index (row 0 = the CHOSEN
//                       word, MTP step 0 = the previous verify's argmax words).
//  rmsnorm_q8_rows_dv   rmsnorm_q8_rows with output row b = the norm of x row src.v[b] + dev[add.v[b]] (add.v[b] < 0: src.v[b]
//                       alone, = rmsnorm_q8_rows). E1: the MTP draft head reads the last ACCEPTED row of the padded step 0 (the
//                       segment's first row + the ACC word).
// (llama.cpp reads every row index / position from device tensors the same way: ggml-cuda/set-rows.cu:57, getrows.cu:23.)
// struct RowTab / RowIdx: kernels/kernel_args.h

extern "C" __global__ void set_rows_dp(int *ids, int *pos, int *kvbase, const int *dev_src, RowTab tab, int n, RowIdx pdev) {
    const int i = threadIdx.x;
    if (i >= n) return;
    const int t = tab.tok[i];
    ids[i] = t >= 0 ? t : dev_src[-t - 1];
    const int w = pdev.v[i];
    pos[i] = tab.pos[i] + (w >= 0 ? dev_src[w] : 0);
    kvbase[i] = tab.base[i];
}

extern "C" __global__ void __launch_bounds__(1024)
    rmsnorm_q8_rows_dv(const float *x, const float *w, float *out, int8_t *xq, float *xd, int n, float eps, RowIdx src,
                       const int *dev, RowIdx add) {
    const int a = add.v[blockIdx.x];
    x += (size_t)(src.v[blockIdx.x] + (a >= 0 ? dev[a] : 0)) * n;
#include "shared/rmsnorm_q8_row.h"  // norm + int8 copy of output row blockIdx.x (ADR-0005), as rmsnorm_q8_rows
}
