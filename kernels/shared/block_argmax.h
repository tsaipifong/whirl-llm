    float best = -INFINITY;  // shared block argmax (ADR-0005); see the end of this file
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }

// kernels/shared/block_argmax.h -- block-wide greedy argmax: each thread scans x[threadIdx.x::blockDim.x],
// then a tree reduction in shared memory; ties go to the lower index, so the result is the first index of
// the maximum. Leaves the max in bv[0] and its index in bi[0] (after a barrier). #included inside argmax
// (kernels/shared/misc_kv_argmax_kv_store.h) and argmax_rows, argmax_rows_to, draft_pick_body, argmax_prob
// (kernels/shared/sample_pick_set_tokens.h); ADR-0005 / arch025-sc11d. The includer provides x, n and the
// __shared__ arrays bv (float) / bi (int) of blockDim.x entries; blockDim.x is a power of two.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
