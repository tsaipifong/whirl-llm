// WHIRL host check (no GPU): the lane -> byte mapping of the gfx1201 "direct" gemvh candidates
// (kernels/gemv_hv_map.h, used by kernels/gemv_hv.hip) reads, for every lane half g0, every 16-k step and
// every element, exactly the bytes / nibble / bit shift / scale byte that dec_group_h<T>(row, 2 * kk + g0)
// reads (kernels/gemm_prefill.hip dec_group_h<QT_MXFP4>, kernels/common.hip Q<QT_Q6_K>::dec8 via dec_group).
// The reference side below transcribes those decoders' address arithmetic line by line; the candidate side
// composes the map functions exactly as GvhD<T>::load / frag do. Returns the number of mismatches.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdio>

#include "../../kernels/gemv_hv_map.h"

namespace whirl_gvh_check {

struct Pick {
    int byte;   // superblock-relative byte of the quantized value (MXFP4 nibble byte / Q6_K ql byte)
    int shift;  // nibble shift (0 / 4)
    int hbyte;  // Q6_K qh byte (-1: n/a)
    int hshift; // Q6_K qh shift
    int sbyte;  // scale byte
    bool operator!=(const Pick& o) const { return byte != o.byte || shift != o.shift || hbyte != o.hbyte || hshift != o.hshift || sbyte != o.sbyte; }
};

// reference: dec_group_h<QT_MXFP4>(row, g), element e (gemm_prefill.hip: b = row + (g >> 5) * 136, p = (g & 31) * 8,
// ib = p >> 5, s = p & 31, scale b[ib], q = b + 8 + 16 * ib + (s & 15), high nibbles iff s >= 16)
inline Pick refMx(int g, int e) {
    const int sb = g >> 5, p = (g & 31) * 8, ib = p >> 5, s = p & 31;
    return {sb * 136 + 8 + 16 * ib + (s & 15) + e, s < 16 ? 0 : 4, -1, 0, sb * 136 + ib};
}
// candidate: unit u = 8 steps, step kq, lane half g0 (GvhD<QT_MXFP4>::load / frag)
inline Pick candMx(int u, int kq, int g0, int e) {
    const int sb = u >> 1, ph = u & 1, i = gvh_mx_piece(kq);
    return {sb * 136 + gvh_mx_q_off(ph, i, g0) + e, gvh_mx_hi(kq) ? 4 : 0, -1, 0, sb * 136 + gvh_mx_e_off(ph) + i};
}
// reference: dec_group<QT_Q6_K>(row, g) -> Q<QT_Q6_K>::dec8(row + (g / 32) * 210, (g % 32) * 8): n = p >> 7, r = p & 127,
// quarter = r >> 5, l = r & 31; ql += 64n, qh += 32n, sc += 8n; scale sc[(l >> 4) + 2 quarter];
// ql byte l + (quarter & 1 ? 32 : 0) + i, lshift quarter >> 1 ? 4 : 0; qh byte l + i, hshift 2 quarter
inline Pick refQ6(int g, int e) {
    const int sb = g / 32, p = (g % 32) * 8, n = p >> 7, r = p & 127, quarter = r >> 5, l = r & 31;
    const int base = sb * 210;
    return {base + 64 * n + l + ((quarter & 1) ? 32 : 0) + e, (quarter >> 1) ? 4 : 0, base + 128 + 32 * n + l + e, 2 * quarter,
            base + 192 + 8 * n + (l >> 4) + 2 * quarter};
}
inline Pick candQ6(int u, int kq, int g0, int e) {
    const int sb = u >> 1, ph = u & 1, base = sb * 210;
    return {base + gvh_q6_ql_off(ph, gvh_q6_ql_piece(kq), g0) + e, gvh_q6_lshift(kq), base + gvh_q6_qh_off(ph, gvh_q6_qh_piece(kq), g0) + e,
            gvh_q6_hshift(kq), base + gvh_q6_sc_off(ph) + kq};
}

// every unit of 3 superblocks, every step, both lane halves, all 8 elements; plus: each lane's pieces are disjoint
// 8-byte runs inside the superblock (no byte past the superblock is read)
inline int run(bool verbose) {
    int bad = 0, n = 0;
    for (int u = 0; u < 6; ++u)
        for (int kq = 0; kq < 8; ++kq)
            for (int g0 = 0; g0 < 2; ++g0)
                for (int e = 0; e < 8; ++e) {
                    const int kk = 8 * u + kq, g = 2 * kk + g0;
                    ++n;
                    if (refMx(g, e) != candMx(u, kq, g0, e)) {
                        if (verbose && bad < 5) std::printf("  mxfp4 mismatch u %d kq %d g0 %d e %d\n", u, kq, g0, e);
                        ++bad;
                    }
                    ++n;
                    if (refQ6(g, e) != candQ6(u, kq, g0, e)) {
                        if (verbose && bad < 5) std::printf("  q6_k mismatch u %d kq %d g0 %d e %d\n", u, kq, g0, e);
                        ++bad;
                    }
                }
    // load ranges inside the superblock (MXFP4 136 B: pieces 8 + 16 i + 8 g0 .. + 8 <= 136; Q6_K 210 B)
    for (int ph = 0; ph < 2; ++ph)
        for (int g0 = 0; g0 < 2; ++g0) {
            for (int i = 0; i < 4; ++i) {
                ++n;
                if (gvh_mx_q_off(ph, i, g0) + 8 > 136 || gvh_q6_ql_off(ph, i, g0) + 8 > 128) ++bad;
            }
            for (int c = 0; c < 2; ++c) {
                ++n;
                if (gvh_q6_qh_off(ph, c, g0) < 128 || gvh_q6_qh_off(ph, c, g0) + 8 > 192) ++bad;
            }
            ++n;
            if (gvh_q6_sc_off(ph) + 8 > 208 || gvh_mx_e_off(ph) + 4 > 8) ++bad;
        }
    if (verbose) std::printf("  gemvh direct map: %d checks, %d mismatches\n", n, bad);
    return bad;
}

}  // namespace whirl_gvh_check
