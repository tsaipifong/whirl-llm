// WHIRL device code (gfx1201 / RDNA 4): moeshx (fast item, feat028-moefast97) - the shared expert of a
// qwen35moe MoE block as slot K of the int8 MXFP4 decode / verify expert launches.
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included after moe_mxfp4.hip by whirl_kernels.hip (uses mxw_dot / mxw_scale, warp_sum).
//
// Balance / fast decode today runs, per MoE layer: moe_gu_mxfp4w (routed gate + up, grid.y = n x K),
// the shared expert's grouped gemvq gate + up, silu_mul_q8 (shared), gemvq shared down, silu_mul_q8 (routed),
// moe_down_mxfp4w (routed down + sg * ysh + residual). The shared chain is latency-bound (512 rows,
// ~76 GB/s, 0.83 ms/token on the R9700, audit\MOE-DEC-PERF.md). Here the shared expert is one more slot:
//   moe_gus_mxfp4w: grid.y = n x (K + 1); pair p -> token t = p / (K + 1), slot s = p % (K + 1); slots
//     s < K read routed expert ids[t * K + s] (rows of W1 / W2), slot K reads the shared gate / up rows
//     (SG / SU, same MXFP4 load-time layout and row bytes); output row p of y1 / y2 (stride ff).
//   silu_mul_q8 over n x (K + 1) x ff (the host launch, unchanged kernel).
//   moe_downs_mxfp4w: per token, the sum over routed slots (weight w[t * K + s]) and the shared slot
//     (weight sg[t] = sigmoid(shw . h), from moe_topk) of the down rows, + residual.
// Same per-row code as moe_gu_mxfp4w / moe_down_mxfp4w (lane owns whole 32-value blocks, warp_sum), so the
// routed slots' gate / up values are bit-identical to moe_gu_mxfp4w; the shared slot's values differ from the
// gemvq path only in the f32 summation order (same int8 activations, same weights). Not a precise path.
// id < 0 (moeskip, feat027): the slot is dropped (no gate / up work; skipped in the down sum).
// Old kernels are untouched (this file only adds entry points).

extern "C" __global__ void __launch_bounds__(256)
    moe_gus_mxfp4w(const uint8_t *__restrict__ W1, const uint8_t *__restrict__ W2, const uint8_t *__restrict__ SG,
                   const uint8_t *__restrict__ SU, uint64_t row_bytes, int ff, int nrows, const int8_t *__restrict__ xq,
                   const float *__restrict__ xd, const int *__restrict__ ids, float *__restrict__ y1, float *__restrict__ y2,
                   int C, int K, const int *skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= nrows) return;
    const int p = blockIdx.y;
    const int t = p / (K + 1), s = p - t * (K + 1);
    const bool second = row >= ff;
    const int r = second ? row - ff : row;
    const uint8_t *wr;
    if (s < K) {
        const int e = ids[t * K + s];
        if (e < 0) return;  // moeskip: dropped slot
        wr = (second ? W2 : W1) + ((size_t)e * ff + r) * row_bytes;
    } else {
        wr = (second ? SU : SG) + (size_t)r * row_bytes;
    }
    const int8_t *xt = xq + (size_t)t * C;
    const float *dt = xd + (size_t)t * (C / 32);
    float acc = 0.f;
    for (int b = lane; b < C / 32; b += WARP) acc += mxw_scale(wr, b) * dt[b] * (float)mxw_dot(wr, b, xt);
    acc = warp_sum(acc);
    if (lane == 0) (second ? y2 : y1)[(size_t)p * ff + r] = acc;
}

extern "C" __global__ void __launch_bounds__(256)
    moe_downs_mxfp4w(const uint8_t *__restrict__ W, const uint8_t *__restrict__ SD, uint64_t row_bytes, int E,
                     const int8_t *__restrict__ xq, const float *__restrict__ xd, const int *__restrict__ ids,
                     const float *__restrict__ w, const float *__restrict__ sg, float *__restrict__ x, int F, int K,
                     const int *skip) {
    if (skip != nullptr && *skip) return;
    const int lane = threadIdx.x % WARP;
    const int row = blockIdx.x * (blockDim.x / WARP) + threadIdx.x / WARP;
    if (row >= E) return;
    const int t = blockIdx.y;
    const int nb = F / 32;
    const int S = K + 1;
    float acc = 0.f;
    for (int u = lane; u < S * nb; u += WARP) {
        const int k = u / nb, b = u - k * nb;
        const int p = t * S + k;
        const uint8_t *wr;
        float wt;
        if (k < K) {
            const int e = ids[t * K + k];
            if (e < 0) continue;  // moeskip: dropped slot
            wr = W + ((size_t)e * E + row) * row_bytes;
            wt = w[t * K + k];
        } else {
            wr = SD + (size_t)row * row_bytes;
            wt = sg[t];
        }
        acc += wt * (mxw_scale(wr, b) * xd[(size_t)p * nb + b] * (float)mxw_dot(wr, b, xq + (size_t)p * F));
    }
    acc = warp_sum(acc);
    if (lane == 0) x[(size_t)t * E + row] += acc;
}
