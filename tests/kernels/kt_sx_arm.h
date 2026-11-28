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
// v0252-attnsx (PLAN V1-148, kernel SOP step 2 for attn_sx8): kernels of attn_sx8's signature AND geometry - arm A becomes
// attn_sx8 itself (cells whose attn_sx pick is not NP 8 are skipped) and arm B is launched with attn_sx8's grid
// {cdiv(n, 8) x heads per launch / 8} x 256:
//   * sx_same_bits: must reproduce attn_sx8 bit for bit (a bitwise check row): attn_sx8 itself (the A/A control) and
//     attn_sxp_nosel8 (the clamped rows kept as loaded; masked keys give P = 0);
//   * sx_probe: attn_sxp_noload8 / _norescale8 / _skel8 (a phase removed, output wrong by design: info row + the bitwise count).
// The attn_sxp_* entries exist only in a code object built with -DWHIRL_ATTNSX_PROBES (kernels/gfx1151/attn_sx.hip), loaded
// through WHIRL_KT_CODE_OBJECT; the product object does not have them (the bench then skips: "kernel not in this code object").
#pragma once

#include <string>

namespace kt::sxarm {

enum class Kind { sx, same_bits, probe, sx_same_bits, sx_probe, invalid };

struct Arm {
    Kind kind = Kind::sx;
    std::string kernel;  // empty for Kind::sx
};

inline constexpr const char* kSameBits[] = {"attn_prefill_wmma", "attn_pfx_f16r"};
inline constexpr const char* kProbes[] = {"fa_probe_noload", "fa_probe_nopv", "fa_probe_norescale"};
// v0252-attnsx round 2: + the product forms attn_sxn2 / 4 / 8 (A = attn_sx of the same NP)
inline constexpr const char* kSxSameBits[] = {"attn_sx8", "attn_sxp_nosel8", "attn_sxn2", "attn_sxn4", "attn_sxn8"};
inline constexpr const char* kSxProbes[] = {"attn_sxp_noload8", "attn_sxp_norescale8", "attn_sxp_skel8"};

// arm A is attn_sx<NP> and arm B runs at attn_sx<NP>'s geometry
inline bool sxGeometry(Kind k) { return k == Kind::sx_same_bits || k == Kind::sx_probe; }
// the NP of an attn_sx-geometry arm = the kernel name's last digit (2, 4 or 8); 0 for any other name
inline int sxArmNp(const std::string& k) {
    if (k.empty()) return 0;
    const char d = k.back();
    return d == '2' ? 2 : d == '4' ? 4 : d == '8' ? 8 : 0;
}

inline Arm armFromEnv(const char* v) {
    Arm a;
    if (v == nullptr || *v == '\0') return a;
    a.kernel = v;
    a.kind = Kind::invalid;
    for (const char* k : kSameBits)
        if (a.kernel == k) a.kind = Kind::same_bits;
    for (const char* k : kProbes)
        if (a.kernel == k) a.kind = Kind::probe;
    for (const char* k : kSxSameBits)
        if (a.kernel == k) a.kind = Kind::sx_same_bits;
    for (const char* k : kSxProbes)
        if (a.kernel == k) a.kind = Kind::sx_probe;
    return a;
}

inline const char* kindName(Kind k) {
    switch (k) {
    case Kind::sx: return "sx";
    case Kind::same_bits: return "same_bits";
    case Kind::probe: return "probe";
    case Kind::sx_same_bits: return "sx_same_bits";
    case Kind::sx_probe: return "sx_probe";
    default: return "invalid";
    }
}

}  // namespace kt::sxarm
