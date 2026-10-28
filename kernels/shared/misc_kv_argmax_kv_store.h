// WHIRL device code, shared by both kernel sets (gfx1201 kernels/misc_kv_argmax.hip, gfx1151 kernels/gfx1151/gdn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

template <int Q8>
__device__ __forceinline__ void kv_store_impl(const float *k, const float *v, const KvArgs &kv, const int *pos, int row) {
    const int t = blockIdx.x;
    const int lp = kv.kvbase == nullptr ? pos[0] + t : pos[t];
    const size_t prow = (size_t)kv_row(kv, kv_tb(kv, t), lp);
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        kv_st32<kv_kf(Q8)>(kv.k, kv.ks, prow * row + i, k[(size_t)t * row + i]);
        kv_st32<kv_vf(Q8)>(kv.v, kv.vs, prow * row + i, v[(size_t)t * row + i]);
    }
}
extern "C" __global__ void kv_store(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<false>(k, v, kv, pos, row);
}
extern "C" __global__ void kv_store_q8(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<true>(k, v, kv, pos, row);
}
extern "C" __global__ void kv_store_q8v(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<2>(k, v, kv, pos, row);
}
// q4: no Hadamard rotation here (attn_prep_q4 rotates); the host always uses attn_prep for q4
extern "C" __global__ void kv_store_q4(const float *k, const float *v, KvArgs kv, const int *pos, int row) {
    kv_store_impl<3>(k, v, kv, pos, row);
}

// Greedy argmax over n logits -> out[0]; also feeds the next decode step:
// ids[0] = token, pos[0] += advance.
extern "C" __global__ void __launch_bounds__(1024) argmax(const float *x, int n, int *out, int *ids, int *pos,
                                                          int advance) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float best = -INFINITY;
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
    if (threadIdx.x == 0) {
        out[0] = bi[0];
        ids[0] = bi[0];
        pos[0] += advance;
    }
}
