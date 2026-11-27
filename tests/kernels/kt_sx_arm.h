// whirl-kernel-test attnbench SX: arm B chosen by WHIRL_KT_BENCH_SX_B (v0252-attnlat, PLAN V1-148; spec
// progress\v0252-attnlim.HANDOFF.md sec. 4). HIP-free, unit-tested on the CPU (tests\kernels\unit\kt_bench_tests.cpp,
// target whirl-kt-bench-tests).
// SPDX-License-Identifier: Apache-2.0
//
// Unset / empty: arm B is attn_sx<NP> (the v0252-strixc2 A/B, unchanged). Set: arm B is that kernel, launched with arm A's
// (attn_prefill_wmma's) geometry and arguments - {cdiv(n, 128), heads per launch, 1} x 256 - so it must be an FA_ENTRY-
// signature kernel of kernels\gfx1151\attn.hip:
//   * same_bits: a candidate that must reproduce attn_prefill_wmma bit for bit (the numerics row is a bitwise check);
//     attn_prefill_wmma itself is the A/A control;
//   * probe: fa_probe_* (a phase removed, output wrong by design): the numerics row is information only (a skip row);
//   * invalid: any other name - the bench refuses it (a skip row with the reason), nothing is timed.
#pragma once

#include <string>

namespace kt::sxarm {

enum class Kind { sx, same_bits, probe, invalid };

struct Arm {
    Kind kind = Kind::sx;
    std::string kernel;  // empty for Kind::sx
};

inline constexpr const char* kSameBits[] = {"attn_prefill_wmma", "attn_pfx_f16r"};
inline constexpr const char* kProbes[] = {"fa_probe_noload", "fa_probe_nopv", "fa_probe_norescale"};

inline Arm armFromEnv(const char* v) {
    Arm a;
    if (v == nullptr || *v == '\0') return a;
    a.kernel = v;
    a.kind = Kind::invalid;
    for (const char* k : kSameBits)
        if (a.kernel == k) a.kind = Kind::same_bits;
    for (const char* k : kProbes)
        if (a.kernel == k) a.kind = Kind::probe;
    return a;
}

inline const char* kindName(Kind k) {
    switch (k) {
    case Kind::sx: return "sx";
    case Kind::same_bits: return "same_bits";
    case Kind::probe: return "probe";
    default: return "invalid";
    }
}

}  // namespace kt::sxarm
