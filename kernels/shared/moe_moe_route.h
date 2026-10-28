// WHIRL device code, shared by both kernel sets (gfx1201 kernels/moe.hip, gfx1151 kernels/gfx1151/moe.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void __launch_bounds__(1024)
    moe_route(const int *__restrict__ ids, int n_pairs, int R, int bn, int *__restrict__ perm, int *__restrict__ inv,
              int4 *__restrict__ tiles, int *__restrict__ n_tiles) {
    __shared__ int cnt[1024], off[1024], fill[1024];
    for (int r = threadIdx.x; r < R; r += blockDim.x) {
        cnt[r] = 0;
        fill[r] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) atomicAdd(&cnt[ids[p]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int o = 0, nt = 0;
        for (int r = 0; r < R; ++r) {
            off[r] = o;
            for (int s = 0; s < cnt[r]; s += bn) tiles[nt++] = make_int4(r, o + s, min(bn, cnt[r] - s), 0);
            o += cnt[r];
        }
        *n_tiles = nt;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < n_pairs; p += blockDim.x) {
        const int e = ids[p];
        const int pos = off[e] + atomicAdd(&fill[e], 1);
        perm[pos] = p;
        inv[p] = pos;
    }
}

// xg[pos][c] = f16(x[perm[pos] / K][c]); one block per pos, C % 4 == 0.
extern "C" __global__ void moe_gather_f16(const float *__restrict__ x, const int *__restrict__ perm,
                                          _Float16 *__restrict__ xg, int C, int K) {
    const int pos = blockIdx.x;
    const int t = perm[pos] / K;
    for (int c = threadIdx.x * 4; c < C; c += blockDim.x * 4) {
        const float4 v = *(const float4 *)(x + (size_t)t * C + c);
        _Float16 *o = xg + (size_t)pos * C + c;
        o[0] = (_Float16)v.x;
        o[1] = (_Float16)v.y;
        o[2] = (_Float16)v.z;
        o[3] = (_Float16)v.w;
    }
}

// a[i] = f16(silu(g[i]) * u[i])
extern "C" __global__ void moe_act_f16(const float *__restrict__ g, const float *__restrict__ u,
                                       _Float16 *__restrict__ a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = g[i];
    a[i] = (_Float16)(v / (1.f + __expf(-v)) * u[i]);
}

// x[t][c] += sum_k w[t,k] * yd[inv[t*K+k]][c] (+ sg[t] * ysh[t][c]); grid (C/256, n).
extern "C" __global__ void moe_combine(float *__restrict__ x, const float *__restrict__ ysh, const float *__restrict__ sg,
                                       const float *__restrict__ yd, const int *__restrict__ inv,
                                       const float *__restrict__ w, int C, int K) {
    const int t = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += w[(size_t)t * K + k] * yd[(size_t)inv[(size_t)t * K + k] * C + c];
    if (ysh != nullptr) acc += sg[t] * ysh[(size_t)t * C + c];
    x[(size_t)t * C + c] += acc;
}
