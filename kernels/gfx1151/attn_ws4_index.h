// WHIRL attn_ws4 index math (gfx1151 4-wave split-K decode attention), shared by the
// kernel (kernels/gfx1151/attn_ws4.hip) and the host checks (tests / CPU emulator).
// SPDX-License-Identifier: Apache-2.0
// Ported from the ATTN8S-T1 CPU emulator (whirl-work attn8s/emu/attn_ws4_emu.py:
// block_range, vt_col, crow, k_pieces, v_owner, vt_addr "prop", sc_addr "prop"), which
// proved the layout hazard-free, bank-conflict-free for Vt / Sc and bitwise equal per column
// to attn_wsplit_impl (audit/DES-ATTN-8S.md "T1 result" F4).
// Plain constexpr C++ (no HIP types) so a host translation unit can include it.
#pragma once

#if defined(__HIPCC__) || defined(__HIP__)
#define WS4_HD __host__ __device__
#else
#define WS4_HD
#endif

namespace ws4 {

constexpr int kKey = 32;    // keys per chunk (AW_K)
constexpr int kD = 256;     // head dim (AW_D)
constexpr int kSink = 256;  // draft-window sink (AW_SINK)
constexpr int kNs = 64;     // split rule constant (design 3.2): splits depend on n_pos only
constexpr int kQs = kD + 8;  // Qs / Ks row pitch (f16)
constexpr int kVtPitch = kKey + 8;                  // Vt row pitch (f16)
constexpr int kVtSize = kD * kVtPitch + (kD / 8) * 8;  // + 16 B every 8 rows (F4)

// split size for n_pos keys over ns splits (multiple of 64)
WS4_HD constexpr int per(int n_pos, int ns) { return (n_pos + ns * 64 - 1) / (ns * 64) * 64; }

struct Range {
    int b0, b1;
};
// key range [b0, b1) of split `split` (the attn_wsplit_impl rule incl. the MTP draft window,
// kv.win = W / 64 | (threshold / 1024) << 16); ns = number of splits of the rule
WS4_HD constexpr Range block_range(int n_pos, int split, int ns, int win) {
    int p = per(n_pos, ns);
    int b0 = split * p, b1 = b0 + p < n_pos ? b0 + p : n_pos;
    if (win != 0 && ns > 1) {
        const int w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
        const int wl = (n_pos - w) & ~63;
        const int w0 = wl > kSink ? wl : kSink;
        if (n_pos >= min_pos && w0 > kSink) {
            if (split == 0) {
                b0 = 0;
                b1 = kSink;
            } else {
                p = (n_pos - w0 + (ns - 1) * 64 - 1) / ((ns - 1) * 64) * 64;
                b0 = w0 + (split - 1) * p;
                b1 = b0 + p < n_pos ? b0 + p : n_pos;
            }
        }
    }
    return Range{b0, b1};
}

// v0252-8sws: block_range with the split count as a template constant, for the BENCH-ONLY kernel entries (attn_ws4.hip
// <name>_ns<S>) only. Their own copy on purpose: when every device call of block_range passes ns = kNs the compiler folds 64
// into it, and one call with another ns changed the shipping attn_ws4 / attn_ws4_q8 code (measured: code-object comparison).
// Same arithmetic as per / block_range; kt_bench_tests pins block_range_ns<S>(n, s, w) == block_range(n, s, S, w).
template <int NS>
WS4_HD constexpr Range block_range_ns(int n_pos, int split, int win) {
    int p = (n_pos + NS * 64 - 1) / (NS * 64) * 64;
    int b0 = split * p, b1 = b0 + p < n_pos ? b0 + p : n_pos;
    if (win != 0 && NS > 1) {
        const int w = (win & 0xffff) * 64, min_pos = (win >> 16) * 1024;
        const int wl = (n_pos - w) & ~63;
        const int w0 = wl > kSink ? wl : kSink;
        if (n_pos >= min_pos && w0 > kSink) {
            if (split == 0) {
                b0 = 0;
                b1 = kSink;
            } else {
                p = (n_pos - w0 + (NS - 1) * 64 - 1) / ((NS - 1) * 64) * 64;
                b0 = w0 + (split - 1) * p;
                b1 = b0 + p < n_pos ? b0 + p : n_pos;
            }
        }
    }
    return Range{b0, b1};
}

// accumulator element v of lane -> row in the 16x16 tile (CROW)
WS4_HD constexpr int crow(int lane, int v) { return 2 * v + lane / 16; }

// Vt column of staging key-block kb (8 keys), row r (attn_wsplit_impl key permutation:
// even rows at tb, odd rows at tb + 8, tb = (kb / 2) * 16 + (kb % 2) * 4)
WS4_HD constexpr int vt_tb(int kb) { return (kb / 2) * 16 + (kb % 2) * 4; }
WS4_HD constexpr int vt_col(int kb, int r) { return vt_tb(kb) + r / 2 + (r % 2 ? 8 : 0); }
// inverse: key (0..31 in the chunk) held by Vt column col (vt_key(vt_col(kb, r)) == 8 kb + r);
// the multi-row kernel masks V^T fragment element e of tile kt (column 16 kt + e) by this key
WS4_HD constexpr int vt_key(int col) { return 8 * (2 * (col / 16) + (col % 8) / 4) + 2 * (col % 4) + (col % 16) / 8; }
// multi-row (verify) blocks: query rows per block of CG 16-column tiles for grp GQA heads
WS4_HD constexpr int rows_per_block(int cg, int grp) { return 16 * cg / grp; }
// f16 offset of Vt[d][col] (row pitch 40, +8 halfs every 8 rows: 16-row reads and the
// staging writes are bank-conflict free, F4)
WS4_HD constexpr int vt_addr(int d, int col) { return d * kVtPitch + (d / 8) * 8 + col; }

// V staging owner of thread tv (0..127): key block kb (keys 8 kb .. 8 kb + 7) and dims
// [d8, d8 + 8) (F4 map: kb = 2 (w % 2) + (L / 8) % 2, d8 = 8 (8 (2 (w / 2) + L / 16) + L % 8))
WS4_HD constexpr int v_kb(int tv) { return 2 * ((tv / 32) % 2) + ((tv % 32) / 8) % 2; }
WS4_HD constexpr int v_d8(int tv) { return 8 * (8 * (2 * ((tv / 32) / 2) + (tv % 32) / 16) + (tv % 32) % 8); }

// K staging pieces of thread tv: f16 (F == 0): 8 pieces p of 8 values, key p * 4 + tv / 32,
// dims (tv % 32) * 8; int8 / int4 (F != 0): 4 pieces of 16 values (half a scale group),
// key p * 8 + tv / 16, dims (tv % 16) * 16
WS4_HD constexpr int k_pieces(int F) { return F == 0 ? 8 : 4; }
WS4_HD constexpr int k_len(int F) { return F == 0 ? 8 : 16; }
WS4_HD constexpr int k_key(int F, int tv, int p) { return F == 0 ? p * 4 + tv / 32 : p * 8 + tv / 16; }
WS4_HD constexpr int k_d0(int F, int tv) { return F == 0 ? (tv % 32) * 8 : (tv % 16) * 16; }

// Sc (QK scores, f32) offset in floats: [kt][v / 4][lane][v % 4] (F4: 1-way)
WS4_HD constexpr int sc_addr(int kt, int lane, int v) { return ((kt * 2 + v / 4) * 32 + lane) * 4 + v % 4; }
constexpr int kScSize = 2 * 2 * 32 * 4;

}  // namespace ws4
