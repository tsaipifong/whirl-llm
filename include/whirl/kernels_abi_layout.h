// The one definition of the by-value kernel argument layouts (arch025 N13).
// SPDX-License-Identifier: Apache-2.0
//
// The kernel argument structs exist on two sides that cannot share one declaration: the host side
// (include/whirl/kernels_abi.h: DevPtr fields with zero initializers, compiled by MSVC) and the device side
// (kernels/kernel_args.h: typed pointers, compiled by HIP clang --cuda-device-only for every kernel set). This header
// is the single source of their layout: byte size and the offset of every field. Both sides include it and
// static_assert their struct against it, so a change on one side alone fails that side's compile
// (tools/gates/kernel_abi_check.py runs both compiles and mutation-tests the asserts).
//
// No includes and no types beyond the built-in ones: it is compiled as host C++20 and as device HIP C++17.
#pragma once

namespace whirl::kernels::abi {

// shared constants (host: whirl::kernels::k*, device: the #defines in kernels/*.hip)
inline constexpr unsigned kKvPage = 256;       // KV_PAGE
inline constexpr unsigned kGdnMaxSeg = 16;     // GDN_MAX_SEG
inline constexpr unsigned kGdnMaxSnap = 15;    // GDN_MAX_SNAP
inline constexpr unsigned kMaxVerifyRows = 32;  // rows of one batched verify / MTP step

// KvArgs: k v ks vs ptab kvbase (8-byte device addresses), tab0, win
inline constexpr unsigned kKvArgsSize = 56;
inline constexpr unsigned kKvArgsK = 0, kKvArgsV = 8, kKvArgsKs = 16, kKvArgsVs = 24, kKvArgsPtab = 32,
                          kKvArgsKvbase = 40, kKvArgsTab0 = 48, kKvArgsWin = 52;

// GvArgs: w[3] (device W0..W2), y[3] (y0..y2), rb[3] (rb0..rb2), n[3] (n0..n2), b1, b2; 4 bytes tail padding
// (the host names them pad_ and zeroes them; the kernels never read them)
inline constexpr unsigned kGvArgsSize = 96;
inline constexpr unsigned kGvArgsW = 0, kGvArgsY = 24, kGvArgsRb = 48, kGvArgsN = 72, kGvArgsB1 = 84, kGvArgsB2 = 88,
                          kGvArgsTail = 92;

// Tok16: t[16]
inline constexpr unsigned kTok16Size = 64;

// RowTab: tok[32] pos[32] base[32]; RowIdx: v[32]; AwGroups: first[32] count[32]
inline constexpr unsigned kRowTabSize = 384, kRowTabTok = 0, kRowTabPos = 128, kRowTabBase = 256;
inline constexpr unsigned kRowIdxSize = 128;
inline constexpr unsigned kAwGroupsSize = 256, kAwGroupsFirst = 0, kAwGroupsCount = 128;

// DSampArgs: on top_k qoff inv_t top_p min_p key_lo key_hi, [32] each, 4-byte elements
inline constexpr unsigned kDSampArgsSize = 1024;
inline constexpr unsigned kDSampOn = 0, kDSampTopK = 128, kDSampQoff = 256, kDSampInvT = 384, kDSampTopP = 512,
                          kDSampMinP = 640, kDSampKeyLo = 768, kDSampKeyHi = 896;

// GdnSeg: state, snap[15], row0, nrows, pend, npend, pad_; GdnSegs: s[16]
inline constexpr unsigned kGdnSegSize = 152;
inline constexpr unsigned kGdnSegState = 0, kGdnSegSnap = 8, kGdnSegRow0 = 128, kGdnSegNrows = 132, kGdnSegPend = 136,
                          kGdnSegNpend = 144, kGdnSegPad = 148;
inline constexpr unsigned kGdnSegsSize = kGdnSegSize * kGdnMaxSeg;  // 2432

}  // namespace whirl::kernels::abi
