// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void __launch_bounds__(256)
    attn_prep(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    attn_prep_impl<false>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);
}
extern "C" __global__ void __launch_bounds__(256)
    attn_prep_q8(float *qf, float *kk, const float *vv, const float *qw, const float *kw, KvArgs kv,
              const int *pos, int n_head, int n_kv, int hd, int n_rot, float theta_scale, float eps) {
    attn_prep_impl<true>(qf, kk, vv, qw, kw, kv, pos, n_head, n_kv, hd, n_rot, theta_scale, eps);
}
