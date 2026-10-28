// WHIRL device code, shared by both kernel sets (gfx1201 kernels/prefill_fused.hip, gfx1151 kernels/gfx1151/prefill_fused.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

#define SM_PER4 24
template <bool FP8, bool TL = false>
__device__ void silu_mul_x_impl(const float *g, const float *u, uint8_t *q, float *sx, int ncols) {
    const float *gr = g + (size_t)blockIdx.x * ncols, *ur = u + (size_t)blockIdx.x * ncols;
    const int tid = threadIdx.x;
    float4 y[SM_PER4];
    float m = 0.f;
#pragma unroll
    for (int k = 0; k < SM_PER4; ++k) {
        const int i = (tid + k * 256) * 4;
        y[k] = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < ncols) {
            const float4 a = *(const float4 *)(gr + i), b = *(const float4 *)(ur + i);
            y[k].x = a.x / (1.f + __expf(-a.x)) * b.x;
            y[k].y = a.y / (1.f + __expf(-a.y)) * b.y;
            y[k].z = a.z / (1.f + __expf(-a.z)) * b.z;
            y[k].w = a.w / (1.f + __expf(-a.w)) * b.w;
            // materialize the f32 products (no fused multiply + f16 round: bitwise == silu_mul)
            asm volatile("" : "+v"(y[k].x), "+v"(y[k].y), "+v"(y[k].z), "+v"(y[k].w));
            m = fmaxf(m, fmaxf(fmaxf(fabsf(y[k].x), fabsf(y[k].y)), fmaxf(fabsf(y[k].z), fabsf(y[k].w))));
        }
    }
    if (FP8) {
        m = block_max(m);
        const float inv = m > 0.f ? 448.f / m : 0.f;
        if (tid == 0) sx[blockIdx.x] = m > 0.f ? m / 448.f : 0.f;
        uint8_t *qr = q + (size_t)blockIdx.x * ncols;
#pragma unroll
        for (int k = 0; k < SM_PER4; ++k) {
            const int i = (tid + k * 256) * 4;
            if (i < ncols) {
                const float a = fminf(fmaxf(y[k].x * inv, -448.f), 448.f), b = fminf(fmaxf(y[k].y * inv, -448.f), 448.f);
                const float c = fminf(fmaxf(y[k].z * inv, -448.f), 448.f), d = fminf(fmaxf(y[k].w * inv, -448.f), 448.f);
                int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
                p = __builtin_amdgcn_cvt_pk_fp8_f32(c, d, p, true);
                if (TL) *(int *)(q + x8t_off(blockIdx.x, i, ncols)) = p;
                else *(int *)(qr + i) = p;
            }
        }
    } else {
        _Float16 *qr = (_Float16 *)q + (size_t)blockIdx.x * ncols;
#pragma unroll
        for (int k = 0; k < SM_PER4; ++k) {
            const int i = (tid + k * 256) * 4;
            if (i < ncols) {
                qr[i] = (_Float16)y[k].x;
                qr[i + 1] = (_Float16)y[k].y;
                qr[i + 2] = (_Float16)y[k].z;
                qr[i + 3] = (_Float16)y[k].w;
            }
        }
    }
}
