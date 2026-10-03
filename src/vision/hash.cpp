// SHA-256 (FIPS 180-4) and Wyhash (final v4.2, as Zig's std.hash.Wyhash) for
// image content ids.
// SPDX-License-Identifier: Apache-2.0
//
// Written from the published algorithm descriptions (FIPS 180-4; the wyhash
// reference description by Wang Yi, public domain / unlicense).

#include "whirl/vision.h"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <intrin.h>

namespace whirl::vision {

namespace {

constexpr std::uint32_t k_sha[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline std::uint32_t rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

// SHA-256 with the x86 SHA extensions (SHA-NI) when the CPU has them: same
// result as block() (the standard message schedule / round mapping of the
// sha256rnds2 / sha256msg1 / sha256msg2 instructions).
bool haveShaNi() {
    static const bool v = [] {
        int r[4];
        __cpuid(r, 0);
        if (r[0] < 7) return false;
        __cpuidex(r, 7, 0);
        const bool sha = (r[1] >> 29) & 1;
        __cpuid(r, 1);
        const bool ssse3 = (r[2] >> 9) & 1, sse41 = (r[2] >> 19) & 1;
        return sha && ssse3 && sse41;
    }();
    return v;
}

void shaNiBlocks(std::uint32_t state[8], const std::uint8_t* data, std::size_t nblk) {
    const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i TMP = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
    __m128i STATE1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
    TMP = _mm_shuffle_epi32(TMP, 0xB1);           // CDAB
    STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);     // EFGH
    __m128i STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);  // ABEF
    STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0);  // CDGH
    while (nblk-- > 0) {
        const __m128i abef = STATE0, cdgh = STATE1;
        __m128i M[4];
        for (int j = 0; j < 16; ++j) {
            if (j < 4) {
                M[j] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * j)), MASK);
            } else {
                __m128i t = _mm_sha256msg1_epu32(M[j % 4], M[(j - 3) % 4]);
                t = _mm_add_epi32(t, _mm_alignr_epi8(M[(j - 1) % 4], M[(j - 2) % 4], 4));
                M[j % 4] = _mm_sha256msg2_epu32(t, M[(j - 1) % 4]);
            }
            __m128i msg = _mm_add_epi32(M[j % 4], _mm_loadu_si128(reinterpret_cast<const __m128i*>(&k_sha[4 * j])));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, msg);
        }
        STATE0 = _mm_add_epi32(STATE0, abef);
        STATE1 = _mm_add_epi32(STATE1, cdgh);
        data += 64;
    }
    TMP = _mm_shuffle_epi32(STATE0, 0x1B);        // FEBA
    STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);     // DCHG
    STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);  // DCBA
    STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);     // HGFE
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), STATE0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), STATE1);
}

struct Sha256 {
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    std::uint8_t buf[64] = {};
    std::size_t buf_len = 0;
    std::uint64_t total = 0;
    bool ni = haveShaNi();

    void block(const std::uint8_t* p) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16) | (std::uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = hh + S1 + ch + k_sha[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + mj;
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    }

    void update(const std::uint8_t* p, std::size_t n) {
        total += n;
        if (buf_len > 0) {
            const std::size_t take = std::min<std::size_t>(64 - buf_len, n);
            std::memcpy(buf + buf_len, p, take);
            buf_len += take;
            p += take;
            n -= take;
            if (buf_len == 64) {
                if (ni) shaNiBlocks(h, buf, 1);
                else block(buf);
                buf_len = 0;
            }
        }
        if (ni && n >= 64) {
            const std::size_t nb = n / 64;
            shaNiBlocks(h, p, nb);
            p += nb * 64;
            n -= nb * 64;
        }
        while (n >= 64) {
            block(p);
            p += 64;
            n -= 64;
        }
        if (n > 0) {
            std::memcpy(buf, p, n);
            buf_len = n;
        }
    }

    std::array<std::uint8_t, 32> final() {
        const std::uint64_t bits = total * 8;
        const std::uint8_t pad80 = 0x80;
        update(&pad80, 1);
        const std::uint8_t zero = 0;
        while (buf_len != 56) update(&zero, 1);
        std::uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = static_cast<std::uint8_t>(bits >> (56 - 8 * i));
        update(len, 8);
        std::array<std::uint8_t, 32> out{};
        for (int i = 0; i < 8; ++i) {
            out[4 * i] = static_cast<std::uint8_t>(h[i] >> 24);
            out[4 * i + 1] = static_cast<std::uint8_t>(h[i] >> 16);
            out[4 * i + 2] = static_cast<std::uint8_t>(h[i] >> 8);
            out[4 * i + 3] = static_cast<std::uint8_t>(h[i]);
        }
        return out;
    }
};

// ---- wyhash final v4.2 (Zig std.hash.Wyhash)

constexpr std::uint64_t k_wy[4] = {0xa0761d6478bd642full, 0xe7037ed1a0b428dbull, 0x8ebc6af09c88c6e3ull, 0x589965cc75374cc3ull};

inline void mum(std::uint64_t& a, std::uint64_t& b) {
    const unsigned __int64 lo = _umul128(a, b, reinterpret_cast<unsigned __int64*>(&b));
    a = lo;
}
inline std::uint64_t mix(std::uint64_t a, std::uint64_t b) {
    mum(a, b);
    return a ^ b;
}
inline std::uint64_t rd8(const std::uint8_t* p) {
    std::uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}
inline std::uint64_t rd4(const std::uint8_t* p) {
    std::uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

}  // namespace

std::array<std::uint8_t, 32> sha256Impl(std::span<const std::span<const std::uint8_t>> parts, bool allow_ni) {
    Sha256 s;
    s.ni = s.ni && allow_ni;
    for (const auto& p : parts) s.update(p.data(), p.size());
    return s.final();
}

std::array<std::uint8_t, 32> sha256(std::span<const std::span<const std::uint8_t>> parts) {
    Sha256 s;
    for (const auto& p : parts) s.update(p.data(), p.size());
    return s.final();
}

std::uint64_t wyhash(std::uint64_t seed, std::span<const std::uint8_t> data) {
    const std::uint8_t* in = data.data();
    const std::size_t len = data.size();
    std::uint64_t st[3];
    st[0] = seed ^ mix(seed ^ k_wy[0], k_wy[1]);
    st[1] = st[0];
    st[2] = st[0];
    std::uint64_t a = 0, b = 0;
    if (len <= 16) {
        if (len >= 4) {
            const std::size_t end = len - 4;
            const std::size_t quarter = (len >> 3) << 2;
            a = (rd4(in) << 32) | rd4(in + quarter);
            b = (rd4(in + end) << 32) | rd4(in + end - quarter);
        } else if (len > 0) {
            a = (std::uint64_t(in[0]) << 16) | (std::uint64_t(in[len >> 1]) << 8) | in[len - 1];
            b = 0;
        }
    } else {
        std::size_t i = 0;
        if (len >= 48) {
            while (i + 48 < len) {
                for (int r = 0; r < 3; ++r) {
                    const std::uint64_t x = rd8(in + i + 8 * (2 * r));
                    const std::uint64_t y = rd8(in + i + 8 * (2 * r + 1));
                    st[r] = mix(x ^ k_wy[r + 1], y ^ st[r]);
                }
                i += 48;
            }
            st[0] ^= st[1] ^ st[2];
        }
        // final1
        const std::uint8_t* tail = in + i;
        const std::size_t tl = len - i;
        std::size_t j = 0;
        while (j + 16 < tl) {
            st[0] = mix(rd8(tail + j) ^ k_wy[1], rd8(tail + j + 8) ^ st[0]);
            j += 16;
        }
        a = rd8(in + len - 16);
        b = rd8(in + len - 8);
    }
    // final2
    a ^= k_wy[1];
    b ^= st[0];
    mum(a, b);
    return mix(a ^ k_wy[0] ^ static_cast<std::uint64_t>(len), b ^ k_wy[1]);
}

}  // namespace whirl::vision
