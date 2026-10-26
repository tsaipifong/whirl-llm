// whirl-gemvk-order-tests (CPU only, feat026-ggemv): the float-order contract of the gfx1151 gemvk kernels
// (kernels/gfx1151/gemv_gk.hip) against the 1-token gemv1s_impl and the production gemvw_impl, emulated on the host
// with the device's exact expressions (pk_d / pk_s / e8m0_half / wfold with fmaf, xor butterfly, LDS tree).
//   1. for nsb = 1..80 superblocks, KW = 8 / 4 / 2, PF on / off, 1..8 tokens: every (row, token) of the gemvk
//      wave/group walk (main rounds + tail, group index i = k mod (8 / KW), red[kw + KW i]) == gemv1s bitwise
//      == gemvw_impl (KW 8 walk sb = w; sb < nsb; sb += 8) bitwise;
//   2. the 1-token result vs a double-precision dot (|err| <= 1e-4 * sum |w x| + 1e-6);
//   3. the e8m0 header extraction (one 8-byte load, ee of unit q = bits [16 q, 16 q + 16)) == the u16 load at
//      blk + 2 q of wunit_half, for random headers;
//   4. the token-slot rule: C row m of the WMMA depends only on A row m (emulated 16x16x16 int dot), so lanes of
//      slots >= NT may load any token's x.
// Exit code 0 = all pass. SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

float u2f(std::uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
std::uint32_t f2u(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
// kernels/gfx1151/common.hip, quant.hip (same bit expressions)
float pk_d(std::uint32_t p) { return u2f((p >> 13) << 12); }
int pk_s(std::uint32_t p) { return static_cast<int>(p & 0x1FFFu) - 4096; }
std::uint32_t pk_make(float d, int sum) {
    const std::uint32_t b = f2u(d) + 0x800u;
    return ((b >> 12) << 13) | static_cast<std::uint32_t>(sum + 4096);
}
float e8m0_half(std::uint32_t e) { return u2f(e < 2 ? (0x00200000u << e) : ((e - 1) << 23)); }
const int kMx[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// gemv_wunit.hip wfold: acc + d_x * (s * dot - m * sum_x) with explicit fmas
float wfold(float acc, std::uint32_t xp, float s, float m, int ci) {
    return std::fmaf(pk_d(xp), std::fmaf(s, static_cast<float>(ci), -(m * static_cast<float>(pk_s(xp)))), acc);
}

struct Problem {
    int nsb = 0, ncols = 0, nt = 0;
    std::vector<std::uint8_t> row;   // one MXFP4 row: nsb x 136 B (8 e8m0 + 8 x 16 B quants)
    std::vector<std::int8_t> xq;     // nt x ncols
    std::vector<std::uint32_t> xd;   // nt x ncols / 32 pk words
};

// exact integer dot of sub-block ib (32 values) of superblock sb with token t
int subDot(const Problem& p, int sb, int ib, int t) {
    const std::uint8_t* blk = p.row.data() + static_cast<std::size_t>(sb) * 136;
    const std::uint8_t* q = blk + 8 + 16 * ib;
    const std::int8_t* x = p.xq.data() + static_cast<std::size_t>(t) * p.ncols + sb * 256 + 32 * ib;
    int d = 0;
    for (int j = 0; j < 16; ++j) {
        d += kMx[q[j] & 0xF] * x[j];
        d += kMx[q[j] >> 4] * x[16 + j];
    }
    return d;
}
// fold of unit u = 4 sb + q (sub-blocks 2q, 2q + 1) into acc, as every kernel does
float unitFold(const Problem& p, int sb, int q, int t, float acc, bool header64) {
    const std::uint8_t* blk = p.row.data() + static_cast<std::size_t>(sb) * 136;
    std::uint32_t ee;
    if (header64) {  // gemv_gk.hip mx_fetch / mx_sb
        std::uint32_t ex, ey;
        std::memcpy(&ex, blk, 4);
        std::memcpy(&ey, blk + 4, 4);
        ee = (q < 2 ? ex >> (16 * q) : ey >> (16 * (q - 2))) & 0xFFFF;
    } else {  // wunit_half: *(const uint16_t *)(blk + 2 pp)
        std::uint16_t e16;
        std::memcpy(&e16, blk + 2 * q, 2);
        ee = e16;
    }
    const float s0 = e8m0_half(ee & 0xFF), s1 = e8m0_half(ee >> 8);
    const int xi = (sb * 256 + 64 * q) >> 5;
    const std::uint32_t px = p.xd[static_cast<std::size_t>(t) * (p.ncols / 32) + xi];
    const std::uint32_t py = p.xd[static_cast<std::size_t>(t) * (p.ncols / 32) + xi + 1];
    return wfold(wfold(acc, px, s0, 0.f, subDot(p, sb, 2 * q, t)), py, s1, 0.f, subDot(p, sb, 2 * q + 1, t));
}

// gemv1s_impl: lane l = slice, units u = l, l + 32, ...; xor butterfly 1, 2, 4, 8, 16; lane 0's value
float ref1s(const Problem& p, int t) {
    float v[32];
    const int nu = p.nsb * 4;
    for (int l = 0; l < 32; ++l) {
        float acc = 0.f;
        for (int u = l; u < nu; u += 32) acc = unitFold(p, u >> 2, u & 3, t, acc, false);
        v[l] = acc;
    }
    for (int o = 1; o < 32; o <<= 1) {
        float n[32];
        for (int l = 0; l < 32; ++l) n[l] = v[l] + v[l ^ o];
        std::memcpy(v, n, sizeof v);
    }
    return v[0];
}
float tree8(const float (&g)[8]) { return ((g[0] + g[1]) + (g[2] + g[3])) + ((g[4] + g[5]) + (g[6] + g[7])); }

// gemvw_impl (production gemvq_nt*r2): wave w, sb = w; sb < nsb; sb += 8; red[w] = (a0 + a1) + (a2 + a3)
float refW(const Problem& p, int t) {
    float red[8];
    for (int w = 0; w < 8; ++w) {
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int sb = w; sb < p.nsb; sb += 8)
            for (int q = 0; q < 4; ++q) acc[q] = unitFold(p, sb, q, t, acc[q], false);
        red[w] = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    return tree8(red);
}

// gemvk_mx_impl<NT, KW, PF>: the literal wave walk (PF only changes when loads are issued; the fold order is the
// same, so it is emulated by fetching the next header early and checking it is the one used)
float refK(const Problem& p, int t, int KW, int PF) {
    const int NG = 8 / KW;
    float red[8];
    for (int g = 0; g < 8; ++g) red[g] = -12345.f;  // every group slot must be written
    for (int kw = 0; kw < KW; ++kw) {
        const int cnt = p.nsb > kw ? (p.nsb - kw + KW - 1) / KW : 0;
        std::vector<float> acc(static_cast<std::size_t>(NG) * 4, 0.f);
        int cur_sb = cnt > 0 ? kw : -1;  // PF: the superblock whose words are in "cur"
        auto step = [&](int k, int i) {
            const int sb = kw + KW * k;
            if (PF) {
                if (cur_sb != sb) std::printf("  PF walk mismatch kw=%d k=%d\n", kw, k);
                const int nxt = kw + KW * (k + 1);
                cur_sb = nxt < p.nsb - 1 ? nxt : p.nsb - 1;  // min(.., nsb - 1)
                if (k + 1 < cnt && cur_sb != nxt) std::printf("  PF clamp hit a used superblock kw=%d k=%d\n", kw, k);
            }
            if (sb % 8 != kw + KW * i) std::printf("  group index mismatch sb=%d i=%d kw=%d KW=%d\n", sb, i, kw, KW);
            for (int q = 0; q < 4; ++q)
                acc[static_cast<std::size_t>(i) * 4 + q] = unitFold(p, sb, q, t, acc[static_cast<std::size_t>(i) * 4 + q], true);
        };
        int k = 0;
        for (; k + NG <= cnt; k += NG)
            for (int i = 0; i < NG; ++i) step(k + i, i);
        for (int i = 0; i < NG - 1; ++i)
            if (k + i < cnt) step(k + i, i);
        for (int i = 0; i < NG; ++i) {
            const float* a = &acc[static_cast<std::size_t>(i) * 4];
            red[kw + KW * i] = (a[0] + a[1]) + (a[2] + a[3]);
        }
    }
    for (int g = 0; g < 8; ++g)
        if (red[g] == -12345.f) std::printf("  group %d never written (KW=%d nsb=%d)\n", g, KW, p.nsb);
    return tree8(red);
}

double refF64(const Problem& p, int t, double& mag) {
    double s = 0;
    mag = 0;
    for (int sb = 0; sb < p.nsb; ++sb)
        for (int ib = 0; ib < 8; ++ib) {
            const std::uint8_t* blk = p.row.data() + static_cast<std::size_t>(sb) * 136;
            const double sc = e8m0_half(blk[ib]);
            const double dx = pk_d(p.xd[static_cast<std::size_t>(t) * (p.ncols / 32) + sb * 8 + ib]);
            const std::uint8_t* q = blk + 8 + 16 * ib;
            const std::int8_t* x = p.xq.data() + static_cast<std::size_t>(t) * p.ncols + sb * 256 + 32 * ib;
            for (int j = 0; j < 16; ++j) {
                const double a = sc * kMx[q[j] & 0xF] * dx * x[j], b = sc * kMx[q[j] >> 4] * dx * x[16 + j];
                s += a + b;
                mag += std::fabs(a) + std::fabs(b);
            }
        }
    return s;
}

Problem make(std::mt19937& rng, int nsb, int nt) {
    Problem p;
    p.nsb = nsb;
    p.ncols = nsb * 256;
    p.nt = nt;
    p.row.resize(static_cast<std::size_t>(nsb) * 136);
    std::uniform_int_distribution<int> byte(0, 255), ex(118, 136), xv(-127, 127);
    std::normal_distribution<float> dd(0.f, 1.f);
    for (int sb = 0; sb < nsb; ++sb) {
        for (int i = 0; i < 8; ++i) p.row[static_cast<std::size_t>(sb) * 136 + i] = static_cast<std::uint8_t>(ex(rng));
        for (int i = 8; i < 136; ++i) p.row[static_cast<std::size_t>(sb) * 136 + i] = static_cast<std::uint8_t>(byte(rng));
    }
    p.xq.resize(static_cast<std::size_t>(nt) * p.ncols);
    p.xd.resize(static_cast<std::size_t>(nt) * p.ncols / 32);
    for (int t = 0; t < nt; ++t)
        for (int b = 0; b < p.ncols / 32; ++b) {
            int sum = 0;
            for (int j = 0; j < 32; ++j) {
                const int v = xv(rng);
                p.xq[static_cast<std::size_t>(t) * p.ncols + b * 32 + j] = static_cast<std::int8_t>(v);
                sum += v;
            }
            p.xd[static_cast<std::size_t>(t) * (p.ncols / 32) + b] = pk_make(std::fabs(dd(rng)) * 0.01f + 1e-4f, sum);
        }
    return p;
}

}  // namespace

int main() {
    std::mt19937 rng(20261026);
    long long checks = 0, bad_k = 0, bad_w = 0, bad_tol = 0, seq_checks = 0, seq_diff = 0;
    const int nsbs[] = {1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 20, 23, 24, 31, 32, 33, 40, 64, 68, 80};
    for (int nsb : nsbs) {
        for (int rep = 0; rep < 3; ++rep) {
            const int nt = 8;
            const Problem p = make(rng, nsb, nt);
            for (int t = 0; t < nt; ++t) {
                const float r1 = ref1s(p, t);
                const float rw = refW(p, t);
                if (f2u(rw) != f2u(r1)) ++bad_w;
                {  // discriminating power: a plain sequential order over the same folds must differ somewhere
                    float s = 0.f;
                    for (int sb = 0; sb < p.nsb; ++sb)
                        for (int q = 0; q < 4; ++q) s += unitFold(p, sb, q, t, 0.f, false);
                    ++seq_checks;
                    seq_diff += f2u(s) != f2u(r1);
                }
                for (int KW : {8, 4, 2})
                    for (int PF : {0, 1}) {
                        const float rk = refK(p, t, KW, PF);
                        ++checks;
                        if (f2u(rk) != f2u(r1)) {
                            ++bad_k;
                            if (bad_k <= 5) std::printf("  MISMATCH nsb=%d t=%d KW=%d PF=%d: %.9g vs %.9g\n", nsb, t, KW, PF, rk, r1);
                        }
                    }
                double mag = 0;
                const double rd = refF64(p, t, mag);
                if (std::fabs(static_cast<double>(r1) - rd) > 1e-4 * mag + 1e-6) {
                    ++bad_tol;
                    if (bad_tol <= 3) std::printf("  TOL nsb=%d t=%d: %.9g vs %.9g (mag %.3g)\n", nsb, t, r1, rd, mag);
                }
            }
        }
    }
    // 3. header extraction equivalence
    long long bad_hdr = 0;
    for (int i = 0; i < 100000; ++i) {
        std::uint8_t blk[8];
        for (auto& b : blk) b = static_cast<std::uint8_t>(rng());
        std::uint32_t ex, ey;
        std::memcpy(&ex, blk, 4);
        std::memcpy(&ey, blk + 4, 4);
        for (int q = 0; q < 4; ++q) {
            std::uint16_t e16;
            std::memcpy(&e16, blk + 2 * q, 2);
            const std::uint32_t ee = (q < 2 ? ex >> (16 * q) : ey >> (16 * (q - 2))) & 0xFFFF;
            if (ee != e16) ++bad_hdr;
        }
    }
    // 4. WMMA row independence (C[m][n] = sum_k A[m][k] B[k][n]): changing A rows >= NT leaves rows < NT unchanged
    long long bad_rows = 0;
    for (int it = 0; it < 2000; ++it) {
        int A[16][16], B[16][16], C0[16][16], C1[16][16];
        const int NT = 2 + static_cast<int>(rng() % 15);
        for (auto& r : A)
            for (int& v : r) v = static_cast<int>(rng() % 255) - 127;
        for (auto& r : B)
            for (int& v : r) v = static_cast<int>(rng() % 25) - 12;
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 16; ++n) {
                C0[m][n] = 0;
                for (int k = 0; k < 16; ++k) C0[m][n] += A[m][k] * B[k][n];
            }
        for (int m = NT; m < 16; ++m)
            for (int k = 0; k < 16; ++k) A[m][k] = A[0][k];  // slots >= NT load token 0 (gemvk) instead of zeros
        for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 16; ++n) {
                C1[m][n] = 0;
                for (int k = 0; k < 16; ++k) C1[m][n] += A[m][k] * B[k][n];
            }
        for (int m = 0; m < NT; ++m)
            for (int n = 0; n < 16; ++n) bad_rows += C0[m][n] != C1[m][n];
    }
    std::printf("gemvk order: %lld (nsb, token, KW, PF) checks, %lld gemvk != gemv1s, %lld gemvw != gemv1s, %lld outside tolerance\n",
                checks, bad_k, bad_w, bad_tol);
    std::printf("e8m0 header extraction: %lld mismatches; WMMA token-slot rule: %lld mismatches\n", bad_hdr, bad_rows);
    std::printf("discriminating power: a sequential order differs from gemv1s in %lld of %lld results (must be > 0)\n", seq_diff,
                seq_checks);
    const bool ok = bad_k == 0 && bad_w == 0 && bad_tol == 0 && bad_hdr == 0 && bad_rows == 0 && checks > 0 && seq_diff > 0;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
