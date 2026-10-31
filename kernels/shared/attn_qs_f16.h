        const int c = i / (AW_D / 8), d8 = (i % (AW_D / 8)) * 8;  // shared Q^T f16 LDS staging step (ADR-0005); see the end of this file
        half8 v8;
        if (c < ncol) {
            const float *qr = q + ((size_t)(first + c / grp) * n_head + g * grp + c % grp) * q_stride + d8;
            const float4 a = *(const float4 *)qr, b = *(const float4 *)(qr + 4);
            v8 = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                         (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) v8[e] = (_Float16)0.f;
        }
        *(half8 *)&Qs[c][d8] = v8;

// kernels/shared/attn_qs_f16.h -- one step i of the Q^T staging loop of the wave-split decode attention:
// column c = i / (AW_D / 8) = (query c / grp, head c % grp), dims d8..d8+8 of q (f32) -> f16 in Qs[c][d8];
// columns past ncol are zero. #included inside the staging loops of kernels/attn.hip (attn_wsplit_impl),
// kernels/gfx1151/attn.hip and kernels/gfx1151/attn_ws4.hip (two kernels); ADR-0005 / arch025-sc11d. The
// includer provides i, q, q_stride, first, grp, g, n_head, ncol and the __shared__ Qs.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
