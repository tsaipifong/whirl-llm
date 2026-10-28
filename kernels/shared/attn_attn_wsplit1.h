// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit1(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<1, false>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit1_q8(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<1, true>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit1_q8v(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<1, 2>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
extern "C" __global__ void __launch_bounds__(128)
    attn_wsplit1_q4(const float *q, KvArgs kv, float *part_ml, float *part_acc, int n_head, int n_kv,
                 int q_stride, const int *pos, float scale, const int *skip, AwGroups groups) {
    attn_wsplit_impl<1, 3>(q, kv, part_ml, part_acc, n_head, n_kv, q_stride, pos, scale, skip, groups);
}
