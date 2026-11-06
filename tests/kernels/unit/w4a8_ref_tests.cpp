// v0252-f16 (F16 P1a): CPU tests of the W4A8 prefill math - the host / device helpers the gfx1151 kernel uses
// (kernels/gfx1151/w4a8_math.h) and the CPU reference the GPU kernel is checked against (tests/kernels/w4a8_ref.h).
// SPDX-License-Identifier: Apache-2.0
//
// Written from the spec (progress/v0252-f16.PRIOR_ART.md sec 3, the brief): MXFP4 e2m1 x 2 -> int8 kvalues {0, 1, 2, 3,
// 4, 6, 8, 12, -0 -> 0, -1, ..., -12}; scale 2^(e - 128) (ggml GGML_E8M0_TO_FP32_HALF bits, f32 subnormal for e < 2);
// activations per 32 along K: d = amax / 127, q = rint(x * (1 / d)) (WHIRL quantize_q8, tests/kernels/cpu_ref.h:30),
// a block with a non-finite value -> d = NaN, q = 0 (NaN reaches the output like the f16 path); the tiled Q8_1 layout of
// Gufo (16-token tiles x 32 K: [b0 16x16][b1 16x16][16 f32 d] = 576 bytes, tile (tt, kb) at (tt * nkb + kb) * 576);
// int32 dot per 32-block; epilogue per block in ascending kb: s = dw * dx (f32), acc = fma(float(c), s, acc), then
// Y = acc or Y + acc. Expectations are literals worked out by hand (comments), never computed by the code under test.
// Build: scratch/v0252_f16/build_cpu_tests.bat (cl, C++17, no HIP). Exit 0 = every check passed.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../../../kernels/gfx1151/w4a8_math.h"
#include "../w4a8_ref.h"

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (cond) {                                                                 \
            ++g_pass;                                                               \
        } else {                                                                    \
            ++g_fail;                                                               \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
        }                                                                           \
    } while (0)

static std::uint32_t fbits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
static float bitsf(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
// one MXFP4 superblock (256 values) in WHIRL's load-time layout: e[8], then qs[8][16]; value j of sub-block ib is the
// low nibble of qs[ib][j] (j < 16) or the high nibble of qs[ib][j - 16]
static void setCode(std::uint8_t* sb, int k, int code) {
    const int ib = k / 32, j = k % 32;
    std::uint8_t& b = sb[8 + 16 * ib + (j & 15)];
    if (j < 16) b = static_cast<std::uint8_t>((b & 0xF0) | code);
    else b = static_cast<std::uint8_t>((b & 0x0F) | (code << 4));
}

// a wrong layout helper must fail checks, not crash: buffers have a floor size and reads are bounds-checked
static std::vector<std::uint8_t> xqBuf(int nt, int k, std::uint8_t fill) {
    const std::size_t want = w4a8_xq_bytes(nt, k), floor_ = static_cast<std::size_t>(1) << 16;
    return std::vector<std::uint8_t>(want > floor_ ? want : floor_, fill);
}
static std::uint8_t at(const std::vector<std::uint8_t>& v, std::size_t off) { return off < v.size() ? v[off] : 0xEE; }
static float atF(const std::vector<std::uint8_t>& v, std::size_t off) {
    std::uint8_t b[4] = {at(v, off), at(v, off + 1), at(v, off + 2), at(v, off + 3)};
    float f;
    std::memcpy(&f, b, 4);
    return f;
}

// 1. e2m1 x 2 -> int8: the literal kvalues table, -0 (code 8) -> 0
static void testKvTable() {
    const int want[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    for (int c = 0; c < 16; ++c) CHECK(w4a8_kv(static_cast<std::uint32_t>(c)) == want[c]);
}

// 2. the device decode (two-table v_perm select; the host form emulates v_perm) gives kvalues in every byte lane
static void testLut4() {
    // bytes (lane 0 first): codes 1, 7, 8, 15 -> 1, 12, 0, -12 = 0x01, 0x0C, 0x00, 0xF4
    CHECK(w4a8_lut4(0x0F080701u) == 0xF4000C01u);
    // codes 0, 5, 9, 14 -> 0, 6, -1, -8 = 0x00, 0x06, 0xFF, 0xF8
    CHECK(w4a8_lut4(0x0E090500u) == 0xF8FF0600u);
    const int want[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int bad = 0;
    for (int lane = 0; lane < 4; ++lane)
        for (int c = 0; c < 16; ++c)
            for (int other = 0; other < 16; other += 5) {
                std::uint32_t nib = 0;
                for (int l = 0; l < 4; ++l) nib |= static_cast<std::uint32_t>(l == lane ? c : other) << (8 * l);
                const std::uint32_t r = w4a8_lut4(nib);
                if (static_cast<std::int8_t>((r >> (8 * lane)) & 0xFF) != want[c]) ++bad;
            }
    CHECK(bad == 0);
    // the emulated v_perm itself: selector bytes 0..7 pick bytes of {s1 (0..3), s0 (4..7)}, 0x0C -> 0x00
    CHECK(w4a8_perm_ref(0x07060504u, 0x03020100u, 0x0C040300u) == 0x00040300u);
}

// 3. e8m0 -> 2^(e - 128) as f32 bits (ggml GGML_E8M0_TO_FP32_HALF), f32 subnormal for e < 2
static void testE8m0() {
    CHECK(w4a8_e8m0_bits(128) == 0x3F800000u);  // 1.0
    CHECK(w4a8_e8m0_bits(127) == 0x3F000000u);  // 0.5
    CHECK(w4a8_e8m0_bits(130) == 0x40800000u);  // 4.0
    CHECK(w4a8_e8m0_bits(1) == 0x00400000u);    // 2^-127 (subnormal)
    CHECK(w4a8_e8m0_bits(0) == 0x00200000u);    // 2^-128 (subnormal)
    CHECK(w4a8_e8m0_bits(255) == 0x7F000000u);  // 2^127 (finite: no NaN encoding, like ggml)
    int bad = 0;
    for (int e = 0; e < 256; ++e)
        if (bitsf(w4a8_e8m0_bits(static_cast<std::uint32_t>(e))) != std::ldexp(1.0f, e - 128)) ++bad;
    CHECK(bad == 0);
}

// 4. tiled activation layout offsets (k = 64 -> 2 blocks per token row of tiles)
static void testLayout() {
    CHECK(w4a8_xq_bytes(17, 64) == 2u * 2u * 576u);  // 2 token tiles x 2 blocks
    CHECK(w4a8_xq_bytes(16, 256) == 1u * 8u * 576u);
    CHECK(w4a8_xq_qoff(0, 0, 64) == 0u);
    CHECK(w4a8_xq_qoff(1, 0, 64) == 16u);    // token 1: next 16-byte row of b0
    CHECK(w4a8_xq_qoff(0, 15, 64) == 15u);
    CHECK(w4a8_xq_qoff(0, 16, 64) == 256u);  // second half of the block: b1
    CHECK(w4a8_xq_qoff(0, 32, 64) == 576u);  // block 1 of tile 0
    // token 17 (tile 1, row 1), k 40 (block 1, half 0, pos 8): tile index 1 * 2 + 1 = 3 -> 3 * 576 + 16 + 8
    CHECK(w4a8_xq_qoff(17, 40, 64) == 1752u);
    CHECK(w4a8_xq_doff(0, 0, 64) == 512u);
    CHECK(w4a8_xq_doff(17, 1, 64) == 3u * 576u + 512u + 4u);
}

// 5. quantizer (one 32-block)
static void testQuantBlock() {
    float x[32] = {};
    std::int8_t q[32];
    float d = -1.f;
    // amax 127 -> d = 1 exactly, q = rint(x): ties to even
    x[0] = 127.f; x[1] = -127.f; x[2] = 2.5f; x[3] = -2.5f; x[4] = 0.49f; x[5] = -0.5f; x[6] = 1.5f; x[7] = 3.5f;
    w4a8_quant_block(x, &d, q);
    CHECK(fbits(d) == 0x3F800000u);
    CHECK(q[0] == 127 && q[1] == -127 && q[2] == 2 && q[3] == -2 && q[4] == 0 && q[5] == 0 && q[6] == 2 && q[7] == 4);
    CHECK(q[31] == 0);
    // all zero (and -0): d = +0, q = 0, no NaN
    for (int i = 0; i < 32; ++i) x[i] = (i & 1) ? -0.0f : 0.0f;
    w4a8_quant_block(x, &d, q);
    CHECK(fbits(d) == 0u);
    int nz = 0;
    for (int i = 0; i < 32; ++i) nz += q[i] != 0;
    CHECK(nz == 0);
    // f16 max magnitude: d = 65504 / 127, q = +-127
    for (int i = 0; i < 32; ++i) x[i] = (i & 1) ? -65504.f : 65504.f;
    w4a8_quant_block(x, &d, q);
    CHECK(d == 65504.f / 127.f);
    CHECK(q[0] == 127 && q[1] == -127 && q[30] == 127 && q[31] == -127);
    // amax 127 * 2^-3: d = 0.125 exactly
    for (int i = 0; i < 32; ++i) x[i] = 0.f;
    x[9] = -15.875f;  // 127 / 8
    x[10] = 0.0625f;  // 0.5 * d -> rint(0.5) = 0
    x[11] = 0.1875f;  // 1.5 * d -> 2
    w4a8_quant_block(x, &d, q);
    CHECK(fbits(d) == 0x3E000000u);
    CHECK(q[9] == -127 && q[10] == 0 && q[11] == 2);
    // NaN or Inf anywhere in the block: d = NaN, every q = 0
    for (int i = 0; i < 32; ++i) x[i] = 1.f;
    x[17] = std::nanf("");
    w4a8_quant_block(x, &d, q);
    CHECK(std::isnan(d));
    nz = 0;
    for (int i = 0; i < 32; ++i) nz += q[i] != 0;
    CHECK(nz == 0);
    x[17] = -INFINITY;
    w4a8_quant_block(x, &d, q);
    CHECK(std::isnan(d));
}

// 6. tiled quantize of [n_tok][k]: padding tokens of the last tile are zero (q = 0, d = 0)
static void testQuantTiled() {
    const int nt = 17, k = 64;
    std::vector<float> x(static_cast<std::size_t>(nt) * k);
    for (int t = 0; t < nt; ++t)
        for (int c = 0; c < k; ++c) x[static_cast<std::size_t>(t) * k + c] = (c == 5) ? 127.f : static_cast<float>((t + c) % 7) - 3.f;
    std::vector<std::uint8_t> xq = xqBuf(nt, k, 0xAB);
    w4a8ref::quantizeTiled(x.data(), nt, k, xq.data());
    // token 3, element 5 -> 127 (d = 1); element 6 -> (3 + 6) % 7 - 3 = -1
    CHECK(static_cast<std::int8_t>(at(xq, w4a8_xq_qoff(3, 5, k))) == 127);
    CHECK(static_cast<std::int8_t>(at(xq, w4a8_xq_qoff(3, 6, k))) == -1);
    CHECK(atF(xq, w4a8_xq_doff(3, 0, k)) == 1.f);
    // token 16 = row 0 of tile 1, element 40: (16 + 40) % 7 - 3 = -3; block 1 amax 3 (no 127 there): d = 3 / 127
    CHECK(static_cast<std::int8_t>(at(xq, w4a8_xq_qoff(16, 40, k))) == -127);
    // padding tokens 17..31: zero bytes and zero scales
    int bad = 0;
    for (int t = 17; t < 32; ++t) {
        for (int c = 0; c < k; ++c) bad += at(xq, w4a8_xq_qoff(t, c, k)) != 0;
        for (int b = 0; b < k / 32; ++b) bad += fbits(atF(xq, w4a8_xq_doff(t, b, k))) != 0u;
    }
    CHECK(bad == 0);
}

// 7. int32 dot of one block and the full reference GEMM on literal cases
static void testDotAndGemm() {
    const int k = 256;
    std::vector<std::uint8_t> w(136, 0);  // one row, one superblock, every code 0, every e 0
    // block 0: e 130 (dw 4); code 7 (12) at k 0. block 1: e 126 (dw 0.25); code 15 (-12) at k 37
    w[0] = 130;
    w[1] = 126;
    setCode(w.data(), 0, 7);
    setCode(w.data(), 37, 15);
    std::vector<float> x(k, 0.f);
    x[0] = 127.f;    // block 0: d 1, q 127
    x[37] = -127.f;  // block 1: d 1, q -127
    std::vector<std::uint8_t> xq = xqBuf(1, k, 0);
    w4a8ref::quantizeTiled(x.data(), 1, k, xq.data());
    CHECK(w4a8ref::blockDot(w.data(), 136, 0, xq.data(), 0, 0, k) == 1524);  // 12 * 127
    CHECK(w4a8ref::blockDot(w.data(), 136, 0, xq.data(), 0, 1, k) == 1524);  // -12 * -127
    CHECK(w4a8ref::blockDot(w.data(), 136, 0, xq.data(), 0, 2, k) == 0);
    float y = 0.f;
    w4a8ref::gemm(w.data(), 136, xq.data(), &y, k, 1, 1, 0);
    CHECK(y == 6477.f);  // 1524 * 4 + 1524 * 0.25
    y = 0.5f;
    w4a8ref::gemm(w.data(), 136, xq.data(), &y, k, 1, 1, 1);
    CHECK(y == 6477.5f);
    // a full block of code 2 (kv 2, e 128 -> w = 2) against q = 1..32 (d = 1: x[31] = 127 sets amax; the others 1..31):
    // sum 2 * (1 + ... + 31 + 127) = 2 * (496 + 127) = 1246
    std::fill(w.begin(), w.end(), 0);
    w[0] = 128;
    for (int j = 0; j < 32; ++j) setCode(w.data(), j, 2);
    std::fill(x.begin(), x.end(), 0.f);
    for (int j = 0; j < 31; ++j) x[j] = static_cast<float>(j + 1);
    x[31] = 127.f;
    std::fill(xq.begin(), xq.end(), 0);
    w4a8ref::quantizeTiled(x.data(), 1, k, xq.data());
    CHECK(w4a8ref::blockDot(w.data(), 136, 0, xq.data(), 0, 0, k) == 1246);
}

// 8. the epilogue order is pinned: ascending kb, one rounding per block (fma). Block 0 adds 2^24, blocks 1 and 2 add 1
// each: 2^24 + 1 rounds back to 2^24 (tie to even) twice -> 16777216; any other order or a pre-summed int gives 16777218
static void testEpilogueOrder() {
    const int k = 256;
    std::vector<std::uint8_t> w(136, 0);
    w[0] = 145;  // dw 2^17
    w[1] = 121;  // dw 2^-7
    w[2] = 121;
    for (int b = 0; b < 3; ++b) {
        setCode(w.data(), 32 * b, 1);
        setCode(w.data(), 32 * b + 1, 1);
    }
    std::vector<float> x(k, 0.f);
    for (int b = 0; b < 3; ++b) {
        x[32 * b] = 127.f;  // q 127, d 1
        x[32 * b + 1] = 1.f;  // q 1 -> c = 128 per block
    }
    std::vector<std::uint8_t> xq = xqBuf(1, k, 0);
    w4a8ref::quantizeTiled(x.data(), 1, k, xq.data());
    float y = 0.f;
    w4a8ref::gemm(w.data(), 136, xq.data(), &y, k, 1, 1, 0);
    CHECK(y == 16777216.f);
}

// 9. non-finite activations: the token whose block holds a NaN gets NaN in every row; the other tokens stay finite
static void testNanPolicy() {
    const int k = 256, nr = 3, nt = 2;
    std::vector<std::uint8_t> w(static_cast<std::size_t>(nr) * 136, 0);
    for (int r = 0; r < nr; ++r) {
        for (int b = 0; b < 8; ++b) w[136 * r + b] = 128;
        for (int c = 0; c < k; ++c) setCode(&w[136 * r], c, 1 + (c + r) % 7);
    }
    std::vector<float> x(static_cast<std::size_t>(nt) * k, 0.5f);
    x[static_cast<std::size_t>(k) + 100] = std::nanf("");  // token 1, block 3
    std::vector<std::uint8_t> xq = xqBuf(nt, k, 0);
    w4a8ref::quantizeTiled(x.data(), nt, k, xq.data());
    std::vector<float> y(static_cast<std::size_t>(nt) * nr, 0.f);
    w4a8ref::gemm(w.data(), 136, xq.data(), y.data(), k, nr, nt, 0);
    int fin0 = 0, nan1 = 0;
    for (int r = 0; r < nr; ++r) {
        fin0 += std::isfinite(y[r]) && y[r] != 0.f;
        nan1 += std::isnan(y[nr + r]);
    }
    CHECK(fin0 == nr);
    CHECK(nan1 == nr);
}

// 10. weight decode from the layout (exact values) and the double-precision f16-path reference agree on a literal case
static void testWeightValueAndF64() {
    std::vector<std::uint8_t> w(136, 0);
    w[3] = 124;  // block 3: 2^-4
    setCode(w.data(), 3 * 32 + 17, 13);  // -6 * 2^-4 = -0.375
    CHECK(w4a8ref::weightValue(w.data(), 3 * 32 + 17) == -0.375);
    CHECK(w4a8ref::weightValue(w.data(), 3 * 32 + 16) == 0.0);
    std::vector<float> x(256, 0.f);
    x[3 * 32 + 17] = 2.f;
    double y = 0;
    w4a8ref::gemmF64(w.data(), 136, x.data(), &y, 256, 1, 1);
    CHECK(y == -0.75);
}

int main() {
    testKvTable();
    testLut4();
    testE8m0();
    testLayout();
    testQuantBlock();
    testQuantTiled();
    testDotAndGemm();
    testEpilogueOrder();
    testNanPolicy();
    testWeightValueAndF64();
    std::printf("w4a8_ref_tests: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
