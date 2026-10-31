    const int h = blockIdx.x;  // shared split-attention combine prologue (ADR-0005); see the end of this file
    const int tq = blockIdx.y;  // query token
    part_ml += (size_t)tq * n_split * n_head * 2;
    part_acc += (size_t)tq * n_split * n_head * head_dim;
    q += (size_t)tq * n_head * q_stride;
    out += (size_t)tq * n_head * head_dim;
    float M = -INFINITY;
    for (int i = 0; i < n_split; ++i) M = fmaxf(M, part_ml[((size_t)i * n_head + h) * 2]);
    float L = 0.f;
    for (int i = 0; i < n_split; ++i) {
        const float m = part_ml[((size_t)i * n_head + h) * 2];
        if (m > -INFINITY) L += __expf(m - M) * part_ml[((size_t)i * n_head + h) * 2 + 1];
    }
    const float inv = 1.f / L;
    const float *gate = q + (size_t)h * q_stride + head_dim;

// kernels/shared/attn_combine_ml.h -- prologue of the split-attention combine kernels attn_combine
// (kernels/shared/attn_attn_combine.h) and attn_combine_q8 (kernels/shared/attn_set_rpos.h): head h = blockIdx.x,
// query token tq = blockIdx.y, the per-token pointer offsets, the global max M of the split maxima, the
// normaliser L = sum_i e^(m_i - M) l_i, inv = 1 / L and the sigmoid-gate row. #included inside both bodies;
// ADR-0005 / arch025-sc11d. The includer's parameters are part_ml, part_acc, q, out, n_head, head_dim, q_stride,
// n_split.
// Line 1 stays a code line and this comment stays at the end (clang -E -P whitespace, see gemmh_impl.h).
