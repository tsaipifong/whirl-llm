// WHIRL unit tests of module kernels.gemm (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-kernels-gemm-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "../../kernels/gemm_small_addr.h"
#include "../../kernels/gfx1151/mx_inject.h"  // v0251-a3port: the gemmfq decode helpers (host + device)
#include "../../kernels/gfx1151/fq_raster.h"  // v0252-8sbal: the gemmfq grouped-raster block order (host + device)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// ---------------------------------------------------------------------------
// Kernel fetch bounds (FIX-OVR): replay the W-stage addressing of the raw-stage GEMMs
// (gfx1201 gemms_impl / gemmsd_impl / gemvh_impl, gfx1151 gemms11_impl) and assert that no
// 16-byte load starts at or past the end of the matrix and none ends past the end rounded
// up to 16 bytes (a 16-byte aligned block holding a matrix byte never crosses a page).
// The pre-fix addressing (every chunk loaded) must be caught by the same check.
struct FetchBound {
    std::uint64_t max_start = 0, max_end = 0;
};
// s: super-block stages per round (gemvh_impl S = 2 when ncols % 512 == 0: SB = 2 * sbb, nst = ncols / 512)
FetchBound replayStageFetch(int sb, std::uint64_t row_bytes, int nrows, int ncols, bool guarded, int s = 1) {
    const int ch = gemms_stage_chunks(sb), nst = ncols / (256 * s);
    FetchBound b;
    // rows past nrows are clamped to nrows - 1 by every kernel; the pattern only depends on
    // the row start, so the first rows, the last two and the clamped row cover every case
    std::vector<int> rows = {0, 1, nrows - 2, nrows - 1, nrows + 5};
    for (int r0 : rows) {
        if (r0 < 0) continue;
        const int row = std::min(r0, nrows - 1);
        for (int js = 0; js < nst; ++js) {
            const std::uint64_t st = static_cast<std::uint64_t>(row) * row_bytes + static_cast<std::uint64_t>(js) * sb;
            const std::uint32_t st32 = static_cast<std::uint32_t>(st);  // gemmsd_impl's 32-bit offsets
            for (int c = 0; c < ch; ++c) {
                const bool need = gemms_chunk_needed(st, c, sb);
                if (gemms_chunk_needed(st32, c, sb) != need) b.max_end = ~0ull;  // the 32-bit form must agree
                if (guarded && !need) continue;
                const std::uint64_t a = gemms_chunk_addr(st, c);
                b.max_start = std::max(b.max_start, a);
                b.max_end = std::max(b.max_end, a + 16);
            }
        }
    }
    return b;
}

void testKernelFetchBounds() {
    struct T {
        const char* name;
        int blk, bytes;
    };
    const T types[] = {{"f16", 8, 16},     {"q8_0", 32, 34},    {"iq4_nl", 32, 18}, {"q3_k", 256, 110}, {"q4_k", 256, 144},
                       {"q5_k", 256, 176}, {"q6_k", 256, 210},  {"iq3_s", 256, 110}, {"iq4_xs", 256, 136}, {"mxfp4", 256, 136}};
    int old_caught = 0, s2_cases = 0, s2_old_caught = 0;
    for (const T& t : types)
        for (int st : {1, 2}) {  // FIX-K97 (FR-K97 P3-1): also gemvh_impl's S = 2 stage (2 super-blocks per LDS round)
            const int sb = st * (256 / t.blk * t.bytes);
            CHECK(gemms_stage_chunks(sb) * 16 >= sb + 15);  // CH covers the worst misalignment
            for (int ncols : {256, 512, 1536, 2048, 4096, 5120, 8192, 12288, 17408})
                for (int nrows : {1, 17, 37, 40, 47, 48, 128, 2048, 4097}) {
                    if (ncols % (256 * st) != 0) continue;  // S = 2 only when ncols % 512 == 0 (gemvh_<T> / gemvh2_<T>)
                    const std::uint64_t rb = static_cast<std::uint64_t>(ncols) / t.blk * t.bytes;
                    const std::uint64_t total = rb * nrows, total16 = (total + 15) & ~15ull;
                    const FetchBound b = replayStageFetch(sb, rb, nrows, ncols, true, st);
                    const bool ok = b.max_start < total && b.max_end <= total16;
                    if (!ok)
                        std::printf("FAIL fetch bound %s S=%d %dx%d: max start %llu end %llu, matrix %llu bytes\n", t.name, st, nrows, ncols,
                                    static_cast<unsigned long long>(b.max_start), static_cast<unsigned long long>(b.max_end),
                                    static_cast<unsigned long long>(total));
                    CHECK(ok);
                    // also within the allocation tail pad for any matrix size
                    CHECK(b.max_end <= total + 256);
                    const FetchBound o = replayStageFetch(sb, rb, nrows, ncols, false, st);
                    if (o.max_end > total16) ++(st == 1 ? old_caught : s2_old_caught);
                    s2_cases += st == 2;
                }
        }
    CHECK(old_caught > 0);     // the check does see the pre-fix over-read (every type reads >= 16 B past)
    CHECK(s2_cases > 0);       // the S = 2 addressing was replayed
    CHECK(s2_old_caught > 0);  // and the check bites there too
}

// v0251-a3port (feat026-fdq c15f51f testMxInject, verbatim): the integer part of the gfx1151 fused-dequant GEMM's
// MXFP4 decode (kernels/gfx1151/mx_inject.h, the same functions gemm_fq.hip runs) with the f16 FMA emulated exactly,
// against WHIRL's LUT decode math (dec_group_h<QT_MXFP4>: f16(kvalues[n]) * f16(e8m0_half(e)) + 0): every code x every
// block exponent of the injected range bitwise, the range predicate for all 256 exponents, and the k order of a
// 32-value sub-block.
namespace mxi_test {
std::uint16_t f2h(float f) {  // round to nearest even, subnormals, inf / nan
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    const std::uint32_t sign = (u >> 16) & 0x8000u, ex = (u >> 23) & 0xFFu, mant = u & 0x7FFFFFu;
    if (ex == 255) return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
    const int e = static_cast<int>(ex) - 127 + 15;
    if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7C00u);
    if (e <= 0) {
        if (ex == 0) return static_cast<std::uint16_t>(sign);
        const std::uint32_t m = mant | 0x800000u;
        const int sh = 14 - e;
        if (sh > 24) return static_cast<std::uint16_t>(sign);
        std::uint32_t h = m >> sh;
        const std::uint32_t rem = m & ((1u << sh) - 1u), half = 1u << (sh - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return static_cast<std::uint16_t>(sign | h);
    }
    std::uint32_t h = (static_cast<std::uint32_t>(e) << 10) | (mant >> 13);
    const std::uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return static_cast<std::uint16_t>(sign | h);
}
float h2f(std::uint16_t h) {
    const std::uint32_t sign = (h & 0x8000u) ? 1u : 0u, ex = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    float v = ex == 0 ? std::ldexp(static_cast<float>(m), -24) : ex == 31 ? (m ? NAN : INFINITY) : std::ldexp(static_cast<float>(m | 0x400u), static_cast<int>(ex) - 25);
    return sign ? -v : v;
}
float e8m0Half(std::uint32_t e) {  // kernels/common.hip e8m0_half
    const std::uint32_t b = e < 2 ? (0x00200000u << e) : ((e - 1) << 23);
    float f;
    std::memcpy(&f, &b, 4);
    return f;
}
const int kKv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// LUT path: (1024 + u) - (1024 + 128) is exactly kv; f16 product (one rounding; exact in float), then + 0
std::uint16_t lutRef(int code, std::uint32_t e) {
    const float dl = h2f(f2h(e8m0Half(e)));
    return f2h(h2f(f2h(static_cast<float>(kKv[code]) * dl)) + 0.0f);
}
// injected path: f16 fma(inj, S, +0) = RN(inj * S + 0); inj * S is exact in float
std::uint16_t injOf(std::uint32_t half_bits, std::uint32_t e) {
    return f2h(h2f(static_cast<std::uint16_t>(half_bits)) * h2f(static_cast<std::uint16_t>(mxi_scale_bits(e))) + 0.0f);
}
}  // namespace mxi_test

void testMxInject() {
    using namespace mxi_test;
    // range predicate and the scale's f16 bits
    int range_bad = 0;
    for (std::uint32_t e = 0; e < 256; ++e) range_bad += mxi_fast_ok(e) != (e >= 104 && e <= 128);
    CHECK(range_bad == 0);
    CHECK(h2f(static_cast<std::uint16_t>(mxi_scale_bits(113))) == 1.0f && h2f(static_cast<std::uint16_t>(mxi_scale_bits(104))) == std::ldexp(1.0f, -9) &&
          h2f(static_cast<std::uint16_t>(mxi_scale_bits(128))) == 32768.0f);
    // every code, every exponent of the range: low and high nibble paths, both f16 halves of both spreads
    int bad = 0, checked = 0;
    for (std::uint32_t e = 104; e <= 128; ++e) {
        for (int c = 0; c < 16; ++c) {
            const std::uint32_t q = 0x11111111u * static_cast<std::uint32_t>(c);
            for (const std::uint32_t t : {mxi_hib_lo(q), mxi_hib_hi(q)}) {
                for (int j = 0; j < 2; ++j) {
                    const std::uint32_t w = mxi_spread_ref(t, j);
                    for (int hh = 0; hh < 2; ++hh) {
                        ++checked;
                        bad += injOf((w >> (16 * hh)) & 0xFFFFu, e) != lutRef(c, e);
                    }
                }
            }
        }
    }
    CHECK(checked == 25 * 16 * 8 && bad == 0);
    // -0 (code 8) becomes +0 like the LUT, and no product of the range rounds (every value is exact)
    CHECK(injOf(mxi_spread_ref(mxi_hib_lo(8u), 0) & 0xFFFFu, 120) == 0 && lutRef(8, 120) == 0);
    int inexact = 0;
    for (std::uint32_t e = 104; e <= 128; ++e)
        for (int c = 0; c < 16; ++c)
            inexact += static_cast<double>(h2f(lutRef(c, e))) != static_cast<double>(kKv[c]) * std::ldexp(1.0, static_cast<int>(e) - 128);
    CHECK(inexact == 0);
    // the k order of one sub-block as gemmfq's fq_decode_sb builds it: o[0] = low nibbles of bytes 0-7, o[1] = low of
    // 8-15, o[2] = high of 0-7, o[3] = high of 8-15; value v of the sub-block is (v < 16 ? low : high) nibble of byte v % 16
    std::uint32_t rng = 12345u;
    int order_bad = 0;
    for (int trial = 0; trial < 64; ++trial) {
        std::uint8_t qs[16];
        for (auto& b : qs) {
            rng = rng * 1664525u + 1013904223u;
            b = static_cast<std::uint8_t>(rng >> 24);
        }
        const std::uint32_t e = 104u + static_cast<std::uint32_t>(trial) % 25u;
        std::uint32_t dw[4];
        std::memcpy(dw, qs, 16);
        const std::uint32_t ts[4][2] = {{mxi_hib_lo(dw[0]), mxi_hib_lo(dw[1])}, {mxi_hib_lo(dw[2]), mxi_hib_lo(dw[3])},
                                        {mxi_hib_hi(dw[0]), mxi_hib_hi(dw[1])}, {mxi_hib_hi(dw[2]), mxi_hib_hi(dw[3])}};
        for (int g = 0; g < 4; ++g) {
            std::uint16_t got[8];
            for (int d = 0; d < 2; ++d)
                for (int j = 0; j < 2; ++j) {
                    const std::uint32_t w = mxi_spread_ref(ts[g][d], j);
                    got[4 * d + 2 * j] = injOf(w & 0xFFFFu, e);
                    got[4 * d + 2 * j + 1] = injOf(w >> 16, e);
                }
            for (int i = 0; i < 8; ++i) {
                const int v = 8 * g + i;
                const int code = v < 16 ? (qs[v] & 0xF) : (qs[v - 16] >> 4);
                order_bad += got[i] != lutRef(code, e);
            }
        }
    }
    CHECK(order_bad == 0);
}

// v0252-8sbal: the grouped raster of the gfx1151 gemmfq prototypes (kernels/gfx1151/fq_raster.h, the function the
// kernels run). For every grid nx x ny up to 40 x 70 and grp 0 / 1 / 2 / 3 / 4 / 8 / 16 / 64: (1) every (row, token)
// tile has exactly one owner (a bijection of the grid); (2) grp 0 is the plain order (row = y, token = x); (3) inside a
// full group the row tile changes fastest: linear ids k and k + 1 of one group map to rows r and r + 1 of the same
// token tile, and the first grp ids of a group are grp distinct rows (the property the screen is about); (4) a group
// never leaves its grp row tiles.
void testFqRaster() {
    long bad_owner = 0, bad_plain = 0, bad_fast = 0, bad_span = 0;
    for (unsigned nx = 1; nx <= 40; nx += (nx < 9 ? 1 : 7))
        for (unsigned ny = 1; ny <= 70; ny += (ny < 12 ? 1 : 11))
            for (const unsigned grp : {0u, 1u, 2u, 3u, 4u, 8u, 16u, 64u}) {
                std::vector<int> owner(static_cast<std::size_t>(nx) * ny, 0);
                for (unsigned by = 0; by < ny; ++by)
                    for (unsigned bx = 0; bx < nx; ++bx) {
                        unsigned r = 0, t = 0;
                        fq_group_tile(bx, by, nx, ny, grp, &r, &t);
                        if (r >= ny || t >= nx) {
                            ++bad_owner;
                            continue;
                        }
                        ++owner[static_cast<std::size_t>(r) * nx + t];
                        if (grp == 0 && (r != by || t != bx)) ++bad_plain;
                        const unsigned lin = by * nx + bx;
                        if (grp > 0) {
                            const unsigned first = lin / (grp * nx) * grp;
                            if (r < first || r >= first + grp) ++bad_span;
                            const bool full = first + grp <= ny;
                            const unsigned within = lin - first * nx;
                            if (full && within + 1 < grp * nx && (within + 1) % grp != 0) {
                                const unsigned l2 = lin + 1;
                                unsigned r2 = 0, t2 = 0;
                                fq_group_tile(l2 % nx, l2 / nx, nx, ny, grp, &r2, &t2);
                                if (r2 != r + 1 || t2 != t) ++bad_fast;
                            }
                        }
                    }
                for (const int o : owner) bad_owner += o != 1;
            }
    CHECK_EQ(bad_owner, 0L);
    CHECK_EQ(bad_plain, 0L);
    CHECK_EQ(bad_fast, 0L);
    CHECK_EQ(bad_span, 0L);
    // the product shapes: Swift ffn_gate_up t2 at 2048 / 4096 rows (136 row tiles x 8 / 16 token tiles), grp 8: the
    // first 8 linear ids are row tiles 0..7 of token tile 0 (Gufo's order)
    for (unsigned k = 0; k < 8; ++k) {
        unsigned r = 0, t = 0;
        fq_group_tile(k % 16, k / 16, 16, 136, 8, &r, &t);
        CHECK(r == k && t == 0);
    }
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_kernels_gemm() {
    static const whirl::test::Suite s{"kernels.gemm",
                                      {
                                          {"testKernelFetchBounds", [] { testKernelFetchBounds(); }},
                                          {"testMxInject", [] { testMxInject(); }},  // v0251-a3port
                                          {"testFqRaster", [] { testFqRaster(); }},  // v0252-8sbal
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_kernels_gemm()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
