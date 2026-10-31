    if (tid < GCH) {  // shared DeltaNet chunk g / beta load + scan (ADR-0005); see the end of this file
        const int tok = t0 + tid;
        G[tid] = tok < n_tok ? gl[(size_t)tok * n_v_heads + h] : 0.f;
        Bt[tid] = tok < n_tok ? beta[(size_t)tok * n_v_heads + h] : 0.f;
    }
    __syncthreads();
    if (tid < WARP) {  // inclusive scan of g over 64 values (2 per lane)
        float a0 = G[2 * tid], a1 = G[2 * tid + 1];
        a1 += a0;
        float x = a1;
#pragma unroll
        for (int o = 1; o < WARP; o <<= 1) {
            const float y = __shfl_up(x, o, WARP);
            if (tid >= o) x += y;
        }
        const float base = x - a1;
        G[2 * tid] = base + a0;
        G[2 * tid + 1] = x;
    }

// kernels/shared/gdn_gscan.h -- chunked Gated DeltaNet prefill: load the chunk's g (log decay) and beta for
// head h into G / Bt (zeros past n_tok), then G = inclusive prefix sum of g over the GCH = 64 chunk tokens (one
// wave, two values per lane, __shfl_up scan). #included inside the chunk kernels of kernels/gdn_prefill.hip,
// kernels/gdn_prefill_wmma.hip and kernels/gfx1151/gdn.hip (both kernel sets); ADR-0005 / arch025-sc11d.
// The includer provides tid, t0, n_tok, h, n_v_heads, gl, beta and the __shared__ arrays G, Bt [GCH].
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
