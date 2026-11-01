__device__ float block_sum(float v) {  // shared block-wide sum (ADR-0005); see the end of this file
    __shared__ float sh[32];
    const int lane = threadIdx.x % WARP, wid = threadIdx.x / WARP;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    const int nw = (blockDim.x + WARP - 1) / WARP;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : 0.0f;
    if (wid == 0) v = warp_sum(v);
    if (threadIdx.x == 0) sh[0] = v;
    __syncthreads();
    return sh[0];
}

// kernels/shared/block_sum.h -- block-wide float sum over all threads of the block (warp_sum per wave, one LDS slot
// per wave, warp_sum of the wave sums by wave 0; every thread returns the total). One definition for the whirl code
// object (both kernel sets, via kernels/shared/common_part_2.h) and the vision code object (kernels/vision/vision.hip
// had a token-identical copy); ADR-0005 / arch025-sc11e. The includer provides WARP and warp_sum before the #include.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
