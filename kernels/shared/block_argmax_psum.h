    const float m = bv[0];  // shared argmax softmax normaliser (ADR-0005); see the end of this file
    const int top = bi[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += __expf(x[i] - m);
    bv[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) bv[threadIdx.x] += bv[threadIdx.x + s];
        __syncthreads();
    }

// kernels/shared/block_argmax_psum.h -- after kernels/shared/block_argmax.h: m = max, top = its index,
// then bv[0] = sum_i e^(x[i] - m) (block tree sum), so the top token's softmax probability is 1 / bv[0].
// #included inside draft_pick_body and argmax_prob (kernels/shared/sample_pick_set_tokens.h);
// ADR-0005 / arch025-sc11d.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
