// whirl-kernel-test: CPU references for the quant formats and GEMV.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstring>

#include "cpu_ref.h"

#define __constant__
#include "iq_tables.h"  // k_iq3s_grid (ggml, MIT; see the header)
#undef __constant__

namespace kt::ref {

namespace {

const std::int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
const std::int8_t kMxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

float hf(const std::uint8_t* p) {
    std::uint16_t h;
    std::memcpy(&h, p, 2);
    return h2f(h);
}

// K-quant 6-bit scale / min of sub-block j (0..7) from the 12 packed bytes.
void scaleMinK4(int j, const std::uint8_t* q, int& sc, int& m) {
    if (j < 4) {
        sc = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Q3_K: 16 signed 6-bit scales from the 12 packed bytes.
void q3kScales(const std::uint8_t* s, int out[16]) {
    for (int k = 0; k < 16; ++k) {
        const int lo = k < 8 ? (s[k] & 0xF) : (s[k - 8] >> 4);
        const int hi = (s[8 + (k & 3)] >> (2 * (k >> 2))) & 3;
        out[k] = (lo | (hi << 4)) - 32;
    }
}

float e8m0Half(std::uint8_t e) { return std::ldexp(1.0f, static_cast<int>(e) - 128); }

}  // namespace

void dequantRow(QType t, const std::uint8_t* b, int ncols, float* y) {
    switch (t) {
        case QType::f32: std::memcpy(y, b, static_cast<std::size_t>(ncols) * 4); return;
        case QType::f16:
            for (int i = 0; i < ncols; ++i) y[i] = hf(b + 2 * i);
            return;
        case QType::q8_0:
            for (int blk = 0; blk < ncols / 32; ++blk, b += 34, y += 32) {
                const float d = hf(b);
                for (int i = 0; i < 32; ++i) y[i] = d * static_cast<float>(static_cast<std::int8_t>(b[2 + i]));
            }
            return;
        case QType::iq4_nl:
            for (int blk = 0; blk < ncols / 32; ++blk, b += 18, y += 32) {
                const float d = hf(b);
                for (int i = 0; i < 16; ++i) {
                    y[i] = d * kIq4nl[b[2 + i] & 0xF];
                    y[i + 16] = d * kIq4nl[b[2 + i] >> 4];
                }
            }
            return;
        case QType::q4_k:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 144) {
                const float d = hf(b), dmin = hf(b + 2);
                const std::uint8_t* qs = b + 16;
                for (int j = 0; j < 4; ++j, qs += 32) {
                    int s0, m0, s1, m1;
                    scaleMinK4(2 * j, b + 4, s0, m0);
                    scaleMinK4(2 * j + 1, b + 4, s1, m1);
                    const float d0 = d * s0, mm0 = dmin * m0, d1 = d * s1, mm1 = dmin * m1;
                    for (int l = 0; l < 32; ++l) *y++ = d0 * (qs[l] & 0xF) - mm0;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * (qs[l] >> 4) - mm1;
                }
            }
            return;
        case QType::q5_k:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 176) {
                const float d = hf(b), dmin = hf(b + 2);
                const std::uint8_t *qh = b + 16, *qs = b + 48;
                for (int j = 0; j < 4; ++j, qs += 32) {
                    int s0, m0, s1, m1;
                    scaleMinK4(2 * j, b + 4, s0, m0);
                    scaleMinK4(2 * j + 1, b + 4, s1, m1);
                    const float d0 = d * s0, mm0 = dmin * m0, d1 = d * s1, mm1 = dmin * m1;
                    const int u0 = 1 << (2 * j), u1 = 2 << (2 * j);
                    for (int l = 0; l < 32; ++l) *y++ = d0 * ((qs[l] & 0xF) + ((qh[l] & u0) ? 16 : 0)) - mm0;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * ((qs[l] >> 4) + ((qh[l] & u1) ? 16 : 0)) - mm1;
                }
            }
            return;
        case QType::q6_k:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 210) {
                const std::uint8_t *ql = b, *qh = b + 128;
                const auto* sc = reinterpret_cast<const std::int8_t*>(b + 192);
                const float d = hf(b + 208);
                for (int n = 0; n < 2; ++n, ql += 64, qh += 32, sc += 8, y += 128) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l] = d * sc[is + 0] * q1;
                        y[l + 32] = d * sc[is + 2] * q2;
                        y[l + 64] = d * sc[is + 4] * q3;
                        y[l + 96] = d * sc[is + 6] * q4;
                    }
                }
            }
            return;
        case QType::q3_k:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 110) {
                const std::uint8_t *hm = b, *q = b + 32;
                int sc[16];
                q3kScales(b + 96, sc);
                const float d = hf(b + 108);
                int is = 0;
                std::uint8_t m = 1;
                for (int n = 0; n < 2; ++n, q += 32) {
                    for (int j = 0, shift = 0; j < 4; ++j, shift += 2, m = static_cast<std::uint8_t>(m << 1)) {
                        float dl = d * sc[is++];
                        for (int l = 0; l < 16; ++l) *y++ = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                        dl = d * sc[is++];
                        for (int l = 0; l < 16; ++l)
                            *y++ = dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
                    }
                }
            }
            return;
        case QType::iq4_xs:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 136) {
                const float d = hf(b);
                const int sh = b[2] | (b[3] << 8);
                const std::uint8_t* qs = b + 8;
                for (int ib = 0; ib < 8; ++ib, qs += 16, y += 32) {
                    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                    const float dl = d * (ls - 32);
                    for (int j = 0; j < 16; ++j) {
                        y[j] = dl * kIq4nl[qs[j] & 0xF];
                        y[j + 16] = dl * kIq4nl[qs[j] >> 4];
                    }
                }
            }
            return;
        case QType::iq3_s:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 110) {
                const float d = hf(b);
                const std::uint8_t *qs = b + 2, *qh = b + 66, *signs = b + 74, *scales = b + 106;
                for (int ib = 0; ib < 8; ++ib, qs += 8, signs += 4) {
                    const float db = d * (1 + 2 * ((scales[ib / 2] >> (4 * (ib % 2))) & 0xF));
                    for (int l = 0; l < 4; ++l, y += 8) {
                        const unsigned g1 = k_iq3s_grid[qs[2 * l] | ((qh[ib] << (8 - 2 * l)) & 256)];
                        const unsigned g2 = k_iq3s_grid[qs[2 * l + 1] | ((qh[ib] << (7 - 2 * l)) & 256)];
                        for (int j = 0; j < 4; ++j) {
                            y[j] = db * static_cast<float>((g1 >> (8 * j)) & 0xFF) * ((signs[l] >> j) & 1 ? -1.f : 1.f);
                            y[j + 4] =
                                db * static_cast<float>((g2 >> (8 * j)) & 0xFF) * ((signs[l] >> (j + 4)) & 1 ? -1.f : 1.f);
                        }
                    }
                }
            }
            return;
        case QType::mxfp4:  // kernel layout: e[8] then qs[8][16] per 256 values
            for (int blk = 0; blk < ncols / 256; ++blk, b += 136) {
                for (int ib = 0; ib < 8; ++ib, y += 32) {
                    const float dl = e8m0Half(b[ib]);
                    const std::uint8_t* q = b + 8 + 16 * ib;
                    for (int j = 0; j < 16; ++j) {
                        y[j] = dl * kMxfp4[q[j] & 0xF];
                        y[j + 16] = dl * kMxfp4[q[j] >> 4];
                    }
                }
            }
            return;
    }
}

std::vector<float> dequantRows(const HostMat& m, int r0, int nrows) {
    std::vector<float> out(static_cast<std::size_t>(nrows) * m.ncols);
    for (int r = 0; r < nrows; ++r)
        dequantRow(m.type, m.data.data() + static_cast<std::size_t>(r0 + r) * m.row_bytes, m.ncols,
                   out.data() + static_cast<std::size_t>(r) * m.ncols);
    return out;
}

void dequantRowF16(QType t, const std::uint8_t* b, int ncols, std::uint16_t* y) {
    auto fma16 = [](int q, std::uint16_t s, std::uint16_t a) {
        return d2h(static_cast<double>(q) * h2f(s) + h2f(a));  // exact in double, one f16 rounding
    };
    switch (t) {
        case QType::f16: std::memcpy(y, b, static_cast<std::size_t>(ncols) * 2); return;
        case QType::q4_k:
        case QType::q5_k: {
            const int bytes = t == QType::q4_k ? 144 : 176;
            for (int blk = 0; blk < ncols / 256; ++blk, b += bytes) {
                const float d = hf(b), dmin = hf(b + 2);
                const std::uint8_t* qh = b + 16;
                const std::uint8_t* qs = b + (t == QType::q4_k ? 16 : 48);
                for (int j = 0; j < 4; ++j, qs += 32) {
                    for (int h = 0; h < 2; ++h) {
                        int sc, m;
                        scaleMinK4(2 * j + h, b + 4, sc, m);
                        const std::uint16_t s16 = f2h(d * sc), a16 = f2h(-dmin * m);
                        for (int l = 0; l < 32; ++l) {
                            int q = h ? (qs[l] >> 4) : (qs[l] & 0xF);
                            if (t == QType::q5_k && (qh[l] >> (2 * j + h)) & 1) q += 16;
                            *y++ = fma16(q, s16, a16);
                        }
                    }
                }
            }
            return;
        }
        case QType::iq4_xs:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 136) {
                const float d = hf(b);
                const int sh = b[2] | (b[3] << 8);
                const std::uint8_t* qs = b + 8;
                for (int ib = 0; ib < 8; ++ib, qs += 16, y += 32) {
                    const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                    const std::uint16_t s16 = f2h(d * (ls - 32));
                    for (int j = 0; j < 16; ++j) {
                        y[j] = fma16(kIq4nl[qs[j] & 0xF], s16, 0);
                        y[j + 16] = fma16(kIq4nl[qs[j] >> 4], s16, 0);
                    }
                }
            }
            return;
        case QType::mxfp4:
            for (int blk = 0; blk < ncols / 256; ++blk, b += 136) {
                for (int ib = 0; ib < 8; ++ib, y += 32) {
                    const std::uint16_t s16 = f2h(e8m0Half(b[ib]));
                    const std::uint8_t* q = b + 8 + 16 * ib;
                    for (int j = 0; j < 16; ++j) {
                        y[j] = fma16(kMxfp4[q[j] & 0xF], s16, 0);
                        y[j + 16] = fma16(kMxfp4[q[j] >> 4], s16, 0);
                    }
                }
            }
            return;
        default: {
            // Generic path: (_Float16)(scale * q). For Q8_0 / Q3_K / Q6_K the compiler emits
            // v_fma_mix(scale, q, +0): the exact product rounded once to f16, but a -0 product
            // becomes +0 (x + +0); the other types convert with v_cvt_f16_f32 (sign kept).
            const bool plus0 = t == QType::q8_0 || t == QType::q3_k || t == QType::q6_k;
            std::vector<float> f(static_cast<std::size_t>(ncols));
            dequantRow(t, b, ncols, f.data());
            for (int i = 0; i < ncols; ++i) {
                const float v = f[static_cast<std::size_t>(i)];
                y[i] = (plus0 && v == 0.f) ? std::uint16_t{0} : f2h(v);
            }
            return;
        }
    }
}

void quantizeQ8(const float* x, int n, std::int8_t* xq, float* xd) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[32 * b + i]));
        const float d = amax / 127.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        xd[b] = d;
        for (int i = 0; i < 32; ++i) xq[32 * b + i] = static_cast<std::int8_t>(std::nearbyint(x[32 * b + i] * id));
    }
}

namespace {
template <class X>
void gemvImpl(const std::vector<float>& W, int nrows, int ncols, const X* x, int ntok, int x_stride,
              std::vector<double>& y, std::vector<double>& scale) {
    y.assign(static_cast<std::size_t>(ntok) * nrows, 0.0);
    scale.assign(y.size(), 0.0);
    for (int t = 0; t < ntok; ++t) {
        const X* xt = x + static_cast<std::size_t>(t) * x_stride;
        for (int r = 0; r < nrows; ++r) {
            const float* w = W.data() + static_cast<std::size_t>(r) * ncols;
            double s = 0, a = 0;
            for (int c = 0; c < ncols; ++c) {
                const double p = static_cast<double>(w[c]) * static_cast<double>(xt[c]);
                s += p;
                a += std::fabs(p);
            }
            y[static_cast<std::size_t>(t) * nrows + r] = s;
            scale[static_cast<std::size_t>(t) * nrows + r] = a;
        }
    }
}
}  // namespace

void gemvF64(const std::vector<float>& W, int nrows, int ncols, const float* x, int ntok, int x_stride,
             std::vector<double>& y, std::vector<double>& scale) {
    gemvImpl(W, nrows, ncols, x, ntok, x_stride, y, scale);
}

void gemvQ8F64(const std::vector<float>& W, int nrows, int ncols, const std::int8_t* xq, const float* xd, int ntok,
               std::vector<double>& y, std::vector<double>& scale) {
    std::vector<double> x(static_cast<std::size_t>(ntok) * ncols);
    for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<double>(xq[i]) * xd[i / 32];  // exact in double
    gemvImpl(W, nrows, ncols, x.data(), ntok, ncols, y, scale);
}

std::uint8_t e4m3(float v) {
    const std::uint8_t sign = std::signbit(v) ? 0x80 : 0;
    const double a = std::fabs(static_cast<double>(v));
    if (a == 0) return sign;
    if (a < std::ldexp(1.0, -6)) return static_cast<std::uint8_t>(sign | static_cast<int>(std::nearbyint(a * 512.0)));
    int E = 0;
    (void)std::frexp(a, &E);
    int e = E - 1;
    double m = std::nearbyint(std::ldexp(a, 3 - e));  // [8, 16]
    if (m >= 16) {
        m = 8;
        ++e;
    }
    return static_cast<std::uint8_t>(sign | ((e + 7) << 3) | (static_cast<int>(m) - 8));
}

float e4m3ToF(std::uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    const float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(static_cast<float>(8 + m), e - 10);
    return (b & 0x80) ? -v : v;
}

void qactFp8Row(const float* x, int n, std::uint8_t* q, float& sx) {
    float m = 0;
    for (int i = 0; i < n; ++i) m = std::max(m, std::fabs(x[i]));
    const float inv = m > 0.f ? 448.f / m : 0.f;
    sx = m > 0.f ? m / 448.f : 0.f;
    for (int i = 0; i < n; ++i) q[i] = e4m3(std::min(std::max(x[i] * inv, -448.f), 448.f));
}

}  // namespace kt::ref
