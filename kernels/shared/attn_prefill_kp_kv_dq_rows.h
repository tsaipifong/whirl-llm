// WHIRL device code, shared by both kernel sets (gfx1201 kernels/attn_prefill_kp.hip, gfx1151 kernels/gfx1151/attn.hip).
// SPDX-License-Identifier: Apache-2.0
// Not self-contained: included at one place in each set, in the order of whirl_kernels.hip (arch025 N5,
// tools/kernels/kshare.py); the gfx1151 copy was token-identical (its comments may have differed).

extern "C" __global__ void __launch_bounds__(256) kv_dq_rows(KvArgs kv, _Float16 *k16, _Float16 *v16, int n_rows, int row_el) {
    kv_dq_rows_impl<1>(kv, k16, v16, 0, n_rows, row_el);
}
// FC-1c: keys [r0, r0 + n_rows) into rows 0 .. n_rows - 1 of k16 / v16 (a key range of a long
// prefill, see Model::prefillAttn); _q4: the q4 cache (kv_dq4, the q4 readers' dequantization)
extern "C" __global__ void __launch_bounds__(256) kv_dq_rows_r(KvArgs kv, _Float16 *k16, _Float16 *v16, int r0, int n_rows, int row_el) {
    kv_dq_rows_impl<1>(kv, k16, v16, r0, n_rows, row_el);
}
extern "C" __global__ void __launch_bounds__(256) kv_dq_rows_r_q4(KvArgs kv, _Float16 *k16, _Float16 *v16, int r0, int n_rows, int row_el) {
    kv_dq_rows_impl<2>(kv, k16, v16, r0, n_rows, row_el);
}
