    half8 qf[FA_D / 16];  // shared flash-prefill Q^T fragment load (ADR-0005); see the end of this file
    {
        const float *qr = q + ((size_t)my_qc * n_head + h) * q_stride;
#pragma unroll
        for (int st = 0; st < FA_D / 16; ++st) {
            const float4 a = *(const float4 *)(qr + 16 * st + 8 * (lane / 16));
            const float4 b = *(const float4 *)(qr + 16 * st + 8 * (lane / 16) + 4);
            qf[st] = (half8){(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
                             (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
        }
    }

// kernels/shared/attn_fa_qfrag.h -- Q^T fragments (WMMA B operand) of the flash prefill attention: column
// = this lane's query my_qc, k = dims 16 st + 8 (lane / 16) .. +8, f32 q -> f16. #included in
// kernels/attn.hip (attn_fa_impl), kernels/attn_prefill_kp.hip (two kernels) and kernels/gfx1151/attn.hip
// (two kernels); ADR-0005 / arch025-sc11d. The includer provides q, q_stride, n_head, h, my_qc, lane.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
