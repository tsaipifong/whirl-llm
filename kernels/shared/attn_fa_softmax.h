        float mx = -INFINITY;  // shared flash-prefill mask + online softmax step (ADR-0005); see the end of this file
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
#if defined(__GFX11__)
                const int key = k0 + kt * 16 + CROW(lane, v);
#else
                const int key = k0 + kt * 16 + 8 * (lane / 16) + v;
#endif
                const float x = (key <= my_pos) ? sc[kt][v] * scale : -INFINITY;
                sc[kt][v] = x;
                mx = fmaxf(mx, x);
            }
        mx = fmaxf(mx, __shfl_xor(mx, 16, WARP));
        const float m_new = fmaxf(m, mx);
        const float alpha = (m == -INFINITY) ? 0.f : __expf(m - m_new);
        float rs = 0.f;
        half8 pt[FA_K / 16];
#pragma unroll
        for (int kt = 0; kt < FA_K / 16; ++kt)
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const float p = (sc[kt][v] == -INFINITY) ? 0.f : __expf(sc[kt][v] - m_new);
                rs += p;
                pt[kt][v] = (_Float16)p;
            }
        rs += __shfl_xor(rs, 16, WARP);
        l = l * alpha + rs;
        m = m_new;

// kernels/shared/attn_fa_softmax.h -- one FA_K key tile of the flash prefill attention: causal mask (keys past
// my_pos -> -inf) with the scale, tile max over the lane pair (lane ^ 16), running max m_new and rescale
// alpha, P = e^(s - m_new) as f16 into pt[FA_K / 16], row sum into l, m = m_new. The accumulator row of
// element v is 8 (lane / 16) + v on gfx12 and CROW(lane, v) on gfx11 (__GFX11__, the gfx1151 set). #included
// inside the key loops of kernels/attn.hip (attn_fa_impl), kernels/attn_prefill_kp.hip and kernels/gfx1151/attn.hip
// (two kernels); ADR-0005 / arch025-sc11d. The includer provides sc[FA_K / 16], k0, lane, my_pos, scale, m, l;
// the O rescale by alpha that follows differs per kernel and stays outside.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
