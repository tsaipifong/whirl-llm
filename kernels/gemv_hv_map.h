// WHIRL: lane -> byte mapping of the gfx1201 "direct" precise gemvh candidates (kernels/gemv_hv.hip),
// shared by the device code and the host unit test (tests/unit/unit_tests.cpp, "gemvh direct map").
// SPDX-License-Identifier: Apache-2.0
//
// gfx12 wave32 WMMA 16x16x16 f16: lane l holds A row l % 16 and k 8 * (l / 16) .. + 8 of every 16-k step.
// The current gemvh (kernels/gemv_h.hip) builds that fragment as dec_group_h<T>(row, 2 * kk + g0) with
// g0 = l / 16, kk = the global 16-k step. The direct kernels load, per lane, only the bytes that call reads,
// as whole pieces of a "unit" = half a 256-k superblock (8 steps), and decode them with the same arithmetic.
// These functions give, for step kq (0..7) of unit half ph (0/1) and lane half g0 (0/1), the superblock byte
// offsets the piece comes from; the unit test checks them against the dec8 addressing of kernels/common.hip
// for every (ph, kq, g0, element).
#pragma once

#include <stdint.h>

#if defined(__HIP__) || defined(__HIPCC__)
#define WHIRL_GVH_HD __host__ __device__
#else
#define WHIRL_GVH_HD
#endif

// ---- MXFP4 (WHIRL load-time layout: e[8] then qs[8][16], 136 bytes / 256 k) ----
// step kk = 2 * ib + hi of the superblock (ib = 32-value block, hi = high nibbles). Unit half ph holds ib 4ph..4ph+3.
// piece i (0..3) of unit ph: 8 bytes at gvh_mx_q_off(ph, i, g0); scale dword at gvh_mx_e_off(ph) (byte i = e[4ph + i]).
WHIRL_GVH_HD constexpr int gvh_mx_q_off(int ph, int i, int g0) { return 8 + 16 * (4 * ph + i) + 8 * g0; }
WHIRL_GVH_HD constexpr int gvh_mx_e_off(int ph) { return 4 * ph; }
// step kq (0..7) of the unit -> piece i = kq / 2, nibble hi = kq % 2
WHIRL_GVH_HD constexpr int gvh_mx_piece(int kq) { return kq >> 1; }
WHIRL_GVH_HD constexpr int gvh_mx_hi(int kq) { return kq & 1; }

// ---- Q6_K (ql[128] qh[64] sc[16] d, 210 bytes / 256 k, 2-byte aligned) ----
// unit half ph = n (128 values). Step kq (0..7): quarter = kq / 2, c = kq % 2.
// ql piece: 8 bytes at gvh_q6_ql_off(ph, kq, g0), nibble shift 4 * (quarter >> 1)
// qh piece: 8 bytes at gvh_q6_qh_off(ph, kq, g0), bit shift 2 * quarter
// scale: int8 byte kq of the 8 bytes at gvh_q6_sc_off(ph); d (f16) at byte 208
WHIRL_GVH_HD constexpr int gvh_q6_ql_off(int ph, int kq, int g0) { return 64 * ph + 32 * ((kq >> 1) & 1) + 16 * (kq & 1) + 8 * g0; }
WHIRL_GVH_HD constexpr int gvh_q6_qh_off(int ph, int kq, int g0) { return 128 + 32 * ph + 16 * (kq & 1) + 8 * g0; }
WHIRL_GVH_HD constexpr int gvh_q6_sc_off(int ph) { return 192 + 8 * ph; }
WHIRL_GVH_HD constexpr int gvh_q6_lshift(int kq) { return ((kq >> 1) >> 1) ? 4 : 0; }
WHIRL_GVH_HD constexpr int gvh_q6_hshift(int kq) { return 2 * (kq >> 1); }
// ql pieces of a unit: index (quarter & 1) * 2 + c, i.e. the 4 distinct offsets; qh pieces: index c
WHIRL_GVH_HD constexpr int gvh_q6_ql_piece(int kq) { return ((kq >> 1) & 1) * 2 + (kq & 1); }
WHIRL_GVH_HD constexpr int gvh_q6_qh_piece(int kq) { return kq & 1; }
