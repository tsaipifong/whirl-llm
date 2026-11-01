    out += (size_t)blockIdx.x * n;  // shared RMSNorm + int8 row body (ADR-0005); see the end of this file
    xq += (size_t)blockIdx.x * n;
    xd += (size_t)blockIdx.x * (n / 32);
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += x[i] * x[i];
    ss = block_sum(ss);
    const float r = rsqrtf(ss / n + eps);
    const int wid = threadIdx.x / WARP, lane = threadIdx.x % WARP, nw = blockDim.x / WARP;
    for (int b = wid; b < n / 32; b += nw) {
        const int i = b * 32 + lane;
        const float v = x[i] * r * w[i];
        out[i] = v;
        wave_q8(v, xq, xd, b);
    }

// kernels/shared/rmsnorm_q8_row.h -- body of rmsnorm_q8 (kernels/shared/fused_decode_rmsnorm_q8.h) and
// rmsnorm_q8_rows (kernels/shared/sample_pick_set_tokens.h): output row blockIdx.x of out / xq / xd, RMS norm of
// the input row x (already offset by the includer: row blockIdx.x, or the gathered row src.v[blockIdx.x]) times
// w, written as f32 and as the int8 copy (wave_q8, 32-value blocks). #included inside both bodies;
// ADR-0005 / arch025-sc11d. The includer provides x, w, out, xq, xd, n, eps.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
