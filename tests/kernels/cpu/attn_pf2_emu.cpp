// feat026-attn8s: CPU emulation of the gfx1151 prefill attention kernels, wave-lane exact.
// SPDX-License-Identifier: Apache-2.0
//
// Emulates attn_prefill_wmma / attn_prefill_wmma_s (kernels/gfx1151/attn.hip, attn_prefill_impl<0, 0, SEG>) and the
// ours-v2 candidates attn_pf2_<cfg>[_s] (kernels/gfx1151/attn_pf2.hip) lane by lane: the same K / V loaders, LDS
// layouts (padded Ks, XOR-swizzled V^T), WMMA fragment layouts (gfx11 wave32: A / B lanes l and l + 16 hold the same
// row / column, D lane holds column lane % 16, rows 2 v + lane / 16), cross-half exchanges, softmax and the v2 pair
// exchanges through LDS. Code between two barriers is run for all waves of the block in turn, which is exact because
// waves only communicate through LDS across barriers.
//
// WMMA is emulated as one fixed-order f32 sum (the hardware's internal order is unknown, but it is a deterministic
// function of (A, B, C)): equality of the emulated outputs proves that both kernels issue the same WMMA operand
// chains per output element and the same scalar operations, i.e. the routing / indexing of v2 is right. The f32
// results are also compared with a double-precision softmax attention (tolerance), and the emulator asserts the
// gfx11 rule that both half-waves pass the same A / B values.
//
// Checks (exit code = number of failures):
//   1. v1 vs double reference (tolerance)          4. v2 (each cfg) == v1, bitwise
//   2. v1 SEG series == v1, bitwise                 5. v2 SEG series == v1, bitwise
//   3. no NaN read: pool rows outside the keys hold NaN (a read of them where P = 0 would still poison O)
// Build: tests\kernels\cpu\build_emu.bat (MSVC, /fp:precise, no contraction).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int WARP = 32, FA_K = 32, FA_D = 256, FA_VS = 40, KV_PAGE = 256, NKB = 128;

// ---- f16 <-> f32 (round to nearest even), bit exact
uint16_t f2h(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x200u : 0u));
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);
        mant |= 0x800000u;
        const int shift = 14 - exp;
        uint32_t h = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1u))) ++h;
        return static_cast<uint16_t>(sign | h);
    }
    uint32_t h = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) ++h;
    return static_cast<uint16_t>(sign | h);
}
float h2f(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1f, mant = h & 0x3ffu, x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign;
        } else {
            exp = 127 - 15 + 1;
            while (!(mant & 0x400u)) {
                mant <<= 1;
                --exp;
            }
            x = sign | (exp << 23) | ((mant & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        x = sign | 0x7f800000u | (mant << 13);
    } else {
        x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

int CROW(int lane, int v) { return 2 * v + lane / 16; }
int fa_vt_off(int row, int h) { return row * FA_VS + ((((h >> 3) ^ (row >> 3)) & 3) << 3) + (h & 7); }

struct Prob {
    int n_head = 4, n_kv = 2, n_tok = 0, pos0 = 0, q_stride = 2 * FA_D, tab0 = 0;
    float scale = 1.f / 16.f;
    std::vector<float> q;          // [n_tok][n_head][q_stride] (q then gate)
    std::vector<uint16_t> K, V;    // pool [row][n_kv][FA_D]
    std::vector<int> ptab;
    int kv_row(int p) const { return ptab[tab0 + p / KV_PAGE] * KV_PAGE + p % KV_PAGE; }
    int kv_delta(int p) const {
        const int pg = p / KV_PAGE;
        return (ptab[tab0 + pg] - pg) * KV_PAGE;
    }
};

int g_fail = 0, g_halfviol = 0;
// negative controls (argv[1]): 1 = v2 row sum in the other order, 2 = v2 n_keys from its own block (not the 128 block)
int g_mut = 0;
// v2 rescale policy (attn_pf2.hip RS): 0 every tile, 1 skip when all alpha == 1 (bitwise), 2 lazy max (lossy)
int g_rs = 0;
constexpr float TLAZY = 5.545177444479562f;

using Frag = uint16_t[WARP][16];  // A / B fragments per lane
using Acc = float[WARP][8];

// D = C + A B; A row m from lane m (lane m + 16 must hold the same), B column n from lane n
void wmma(const Frag& a, const Frag& b, Acc& c) {
    for (int l = 0; l < 16; ++l)
        for (int k = 0; k < 16; ++k)
            if (a[l][k] != a[l + 16][k] || b[l][k] != b[l + 16][k]) ++g_halfviol;
    Acc d;
    for (int lane = 0; lane < WARP; ++lane)
        for (int v = 0; v < 8; ++v) {
            const int mrow = CROW(lane, v), n = lane % 16;
            float s = 0.f;
            for (int k = 0; k < 16; ++k) s += h2f(a[mrow][k]) * h2f(b[n][k]);
            d[lane][v] = c[lane][v] + s;
        }
    std::memcpy(c, d, sizeof(Acc));
}
// join_halves: k [0, 8) = half 0's piece, k [8, 16) = half 1's piece, of the same lane % 16
void join_halves(const uint16_t own[WARP][8], Frag& f) {
    for (int lane = 0; lane < WARP; ++lane)
        for (int i = 0; i < 8; ++i) {
            f[lane][i] = own[lane % 16][i];
            f[lane][8 + i] = own[lane % 16 + 16][i];
        }
}

struct Lds {
    std::vector<uint16_t> Ks = std::vector<uint16_t>(FA_K * (FA_D + 8), 0x7e00);  // NaN fill: catches unwritten reads
    std::vector<uint16_t> Vt = std::vector<uint16_t>(FA_D * FA_VS, 0x7e00);
    uint16_t& K(int r, int c) { return Ks[r * (FA_D + 8) + c]; }
};

// The attn_prefill_wmma K / V tile loader + LDS store of a 256-thread geometry, thread t (both roles).
void stage(const Prob& P, Lds& L, int k0, int n_keys, int kvh, int t, bool doK, bool doV) {
    const int dl = P.kv_delta(k0), rstep = P.n_kv * FA_D;
    if (doK) {
        const int kro = kvh * FA_D + (t % 32) * 8;
        for (int e = 0; e < 4; ++e) {
            const int key = k0 + t / 32 + 8 * e, kc = std::min(key, n_keys - 1);
            const size_t src = (size_t)(unsigned)(kc + dl) * rstep + kro;
            const int i = t + e * 256;
            for (int j = 0; j < 8; ++j) L.K(i / 32, (i % 32) * 8 + j) = key < n_keys ? P.K[src + j] : 0;
        }
    }
    if (doV) {
        const int vd4 = (t % 64) * 4, vkb = t / 64, vtb = (vkb / 2) * 16 + (vkb % 2) * 4;
        const int vsA = fa_vt_off(vd4, vtb), vsB = fa_vt_off(vd4, vtb + 8), vro = kvh * FA_D + vd4;
        uint16_t vr[8][4];
        for (int r = 0; r < 8; ++r) {
            const int key = k0 + vkb * 8 + r, kc = std::min(key, n_keys - 1);
            const size_t src = (size_t)(unsigned)(kc + dl) * rstep + vro;
            for (int c = 0; c < 4; ++c) vr[r][c] = key < n_keys ? P.V[src + c] : 0;
        }
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) {
                L.Vt[vsA + c * FA_VS + r] = vr[2 * r][c];
                L.Vt[vsB + c * FA_VS + r] = vr[2 * r + 1][c];
            }
    }
}

// Q^T pieces of a wave: lane holds dims 16 st + 8 (lane / 16) .. + 7 of query my_qc
void loadQ(const Prob& P, int h, const int my_qc[WARP], uint16_t qf[16][WARP][8]) {
    for (int lane = 0; lane < WARP; ++lane) {
        const float* qr = &P.q[((size_t)my_qc[lane] * P.n_head + h) * P.q_stride];
        for (int st = 0; st < 16; ++st)
            for (int i = 0; i < 8; ++i) qf[st][lane][i] = f2h(qr[16 * st + 8 * (lane / 16) + i]);
    }
}
float shfl16(const float x[WARP], int lane) { return x[lane ^ 16]; }

// expf stands in for __expf in both kernels (same function -> equality check unaffected; tolerance vs double)
float ex(float x) { return std::exp(x); }

struct SegArgs {
    bool seg = false;
    int kb0 = 0, kb1 = 0;
    std::vector<float>* st_ml = nullptr;
};

// ---------------- v1: attn_prefill_impl<0, 0, SEG> (8 waves x 16 queries, 256 threads)
void v1_block(const Prob& P, std::vector<float>& out, int bx, int h, const SegArgs& S) {
    const int kvh = h / (P.n_head / P.n_kv), q0 = bx * 128, pos0 = P.pos0;
    const int last_q = std::min(q0 + 128, P.n_tok) - 1, n_keys = pos0 + last_q + 1;
    if (S.seg && S.kb0 >= n_keys) return;
    Lds L;
    struct W {
        int my_q[WARP], my_qc[WARP], my_pos[WARP];
        uint16_t qf[16][WARP][8];
        float o[16][WARP][8];
        float m[WARP], l[WARP];
    };
    std::vector<W> w(8);
    for (int wi = 0; wi < 8; ++wi) {
        W& x = w[wi];
        for (int lane = 0; lane < WARP; ++lane) {
            x.my_q[lane] = q0 + wi * 16 + lane % 16;
            x.my_qc[lane] = std::min(x.my_q[lane], P.n_tok - 1);
            x.my_pos[lane] = pos0 + x.my_qc[lane];
            x.m[lane] = -INFINITY;
            x.l[lane] = 0.f;
        }
        loadQ(P, h, x.my_qc, x.qf);
        std::memset(x.o, 0, sizeof x.o);
        if (S.seg && S.kb0 > 0)
            for (int lane = 0; lane < WARP; ++lane) {
                const size_t b = ((size_t)x.my_qc[lane] * P.n_head + h);
                x.m[lane] = (*S.st_ml)[b * 2];
                x.l[lane] = (*S.st_ml)[b * 2 + 1];
                for (int dt = 0; dt < 16; ++dt)
                    for (int v = 0; v < 8; ++v) x.o[dt][lane][v] = out[b * FA_D + dt * 16 + CROW(lane, v)];
            }
    }
    const int k_beg = S.seg ? S.kb0 : 0, k_end = S.seg ? std::min(n_keys, S.kb1) : n_keys;
    for (int k0 = k_beg; k0 < k_end; k0 += FA_K) {
        for (int t = 0; t < 256; ++t) stage(P, L, k0, n_keys, kvh, t, true, true);
        for (int wi = 0; wi < 8; ++wi) {
            W& x = w[wi];
            Acc sc[2] = {};
            for (int st = 0; st < 16; ++st) {
                Frag qb;
                join_halves(x.qf[st], qb);
                for (int kt = 0; kt < 2; ++kt) {
                    Frag a;
                    for (int lane = 0; lane < WARP; ++lane)
                        for (int i = 0; i < 16; ++i) a[lane][i] = L.K(kt * 16 + lane % 16, 16 * st + i);
                    wmma(a, qb, sc[kt]);
                }
            }
            float mx[WARP], mx2[WARP], m_new[WARP], alpha[WARP], rs[WARP], rs2[WARP];
            uint16_t pt[2][WARP][8];
            for (int lane = 0; lane < WARP; ++lane) {
                mx[lane] = -INFINITY;
                for (int kt = 0; kt < 2; ++kt)
                    for (int v = 0; v < 8; ++v) {
                        const int key = k0 + kt * 16 + CROW(lane, v);
                        const float xv = (key <= x.my_pos[lane]) ? sc[kt][lane][v] * P.scale : -INFINITY;
                        sc[kt][lane][v] = xv;
                        mx[lane] = std::fmax(mx[lane], xv);
                    }
            }
            for (int lane = 0; lane < WARP; ++lane) mx2[lane] = std::fmax(mx[lane], shfl16(mx, lane));
            for (int lane = 0; lane < WARP; ++lane) {
                m_new[lane] = std::fmax(x.m[lane], mx2[lane]);
                alpha[lane] = (x.m[lane] == -INFINITY) ? 0.f : ex(x.m[lane] - m_new[lane]);
                rs[lane] = 0.f;
                for (int kt = 0; kt < 2; ++kt)
                    for (int v = 0; v < 8; ++v) {
                        const float p = (sc[kt][lane][v] == -INFINITY) ? 0.f : ex(sc[kt][lane][v] - m_new[lane]);
                        rs[lane] += p;
                        pt[kt][lane][v] = f2h(p);
                    }
            }
            for (int lane = 0; lane < WARP; ++lane) rs2[lane] = rs[lane] + shfl16(rs, lane);
            for (int lane = 0; lane < WARP; ++lane) {
                x.l[lane] = x.l[lane] * alpha[lane] + rs2[lane];
                x.m[lane] = m_new[lane];
                for (int dt = 0; dt < 16; ++dt)
                    for (int v = 0; v < 8; ++v) x.o[dt][lane][v] *= alpha[lane];
            }
            for (int kt = 0; kt < 2; ++kt) {
                Frag pb;
                join_halves(pt[kt], pb);
                for (int dt = 0; dt < 16; ++dt) {
                    Frag a;
                    const int C = (2 * kt) ^ (2 * (dt & 1));
                    for (int lane = 0; lane < WARP; ++lane) {
                        const int s0 = (lane % 16) >> 3;
                        const int lo = (lane % 16) * FA_VS + s0 * 8 + dt * 16 * FA_VS + C * 8;
                        const int hi = (lane % 16) * FA_VS + 8 - s0 * 8 + dt * 16 * FA_VS + C * 8;
                        for (int i = 0; i < 8; ++i) {
                            a[lane][i] = L.Vt[lo + i];
                            a[lane][8 + i] = L.Vt[hi + i];
                        }
                    }
                    Acc acc;
                    std::memcpy(acc, x.o[dt], sizeof acc);
                    wmma(a, pb, acc);
                    std::memcpy(x.o[dt], acc, sizeof acc);
                }
            }
        }
    }
    for (int wi = 0; wi < 8; ++wi) {
        W& x = w[wi];
        for (int lane = 0; lane < WARP; ++lane) {
            if (x.my_q[lane] >= P.n_tok) continue;
            const size_t b = ((size_t)x.my_q[lane] * P.n_head + h);
            if (S.seg && k_end < n_keys) {
                if (lane < 16) {
                    (*S.st_ml)[b * 2] = x.m[lane];
                    (*S.st_ml)[b * 2 + 1] = x.l[lane];
                }
                for (int dt = 0; dt < 16; ++dt)
                    for (int v = 0; v < 8; ++v) out[b * FA_D + dt * 16 + CROW(lane, v)] = x.o[dt][lane][v];
                continue;
            }
            const float inv = 1.f / x.l[lane];
            const float* gate = &P.q[b * P.q_stride + FA_D];
            for (int dt = 0; dt < 16; ++dt)
                for (int v = 0; v < 8; ++v) {
                    const int d = dt * 16 + CROW(lane, v);
                    out[b * FA_D + d] = x.o[dt][lane][v] * inv / (1.f + ex(-gate[d]));
                }
        }
    }
}

// ---------------- v2: attn_pf2_impl<NQT, PF, BAR, SEG> (wave pairs; PF / BAR change timing only)
void v2_block(const Prob& P, std::vector<float>& out, int bx, int h, int NQT, const SegArgs& S) {
    const int NTHR = 64 * NQT, FQ = 16 * NQT, NW = 2 * NQT;
    const int kvh = h / (P.n_head / P.n_kv), q0 = bx * FQ, pos0 = P.pos0;
    const int qb0 = g_mut == 2 ? q0 : (q0 / NKB) * NKB, last_q = std::min(qb0 + (g_mut == 2 ? FQ : NKB), P.n_tok) - 1,
              n_keys = pos0 + last_q + 1;
    if (S.seg && S.kb0 >= n_keys) return;
    Lds L;
    std::vector<float> xmx(NQT * 2 * 16, NAN), xrs(NQT * 32, NAN);
    std::vector<uint16_t> xpt(NQT * 2 * 32 * 8, 0x7e00);
    struct W {
        int my_q[WARP], my_qc[WARP], my_pos[WARP];
        uint16_t qf[16][WARP][8];
        float o[8][WARP][8];
        float m[WARP], l[WARP];
        Acc sc;
        float mx[WARP], alpha[WARP], rs[WARP];
        uint16_t pt[WARP][8];
        bool resc = true;
    };
    std::vector<W> w(NW);
    for (int wi = 0; wi < NW; ++wi) {
        W& x = w[wi];
        const int qt = wi >> 1, hf = wi & 1;
        for (int lane = 0; lane < WARP; ++lane) {
            x.my_q[lane] = q0 + qt * 16 + lane % 16;
            x.my_qc[lane] = std::min(x.my_q[lane], P.n_tok - 1);
            x.my_pos[lane] = pos0 + x.my_qc[lane];
            x.m[lane] = -INFINITY;
            x.l[lane] = 0.f;
        }
        loadQ(P, h, x.my_qc, x.qf);
        std::memset(x.o, 0, sizeof x.o);
        if (S.seg && S.kb0 > 0)
            for (int lane = 0; lane < WARP; ++lane) {
                const size_t b = ((size_t)x.my_qc[lane] * P.n_head + h);
                x.m[lane] = (*S.st_ml)[b * 2];
                x.l[lane] = (*S.st_ml)[b * 2 + 1];
                for (int j = 0; j < 8; ++j)
                    for (int v = 0; v < 8; ++v) x.o[j][lane][v] = out[b * FA_D + (hf * 8 + j) * 16 + CROW(lane, v)];
            }
    }
    const int k_beg = S.seg ? S.kb0 : 0, k_end = S.seg ? std::min(n_keys, S.kb1) : n_keys;
    const bool both = NTHR == 256;
    for (int k0 = k_beg; k0 < k_end; k0 += FA_K) {
        // barrier; stage; barrier
        for (int tid = 0; tid < NTHR; ++tid) {
            const bool doV = both || tid < 256, doK = both || tid >= 256;
            stage(P, L, k0, n_keys, kvh, both ? tid : (tid & 255), doK, doV);
        }
        // phase A: scores of key tile hf, mask, tile max of my keys -> xmx
        for (int wi = 0; wi < NW; ++wi) {
            W& x = w[wi];
            const int qt = wi >> 1, hf = wi & 1;
            std::memset(x.sc, 0, sizeof x.sc);
            for (int st = 0; st < 16; ++st) {
                Frag qb, a;
                join_halves(x.qf[st], qb);
                for (int lane = 0; lane < WARP; ++lane)
                    for (int i = 0; i < 16; ++i) a[lane][i] = L.K(hf * 16 + lane % 16, 16 * st + i);
                wmma(a, qb, x.sc);
            }
            float mx[WARP];
            for (int lane = 0; lane < WARP; ++lane) {
                mx[lane] = -INFINITY;
                for (int v = 0; v < 8; ++v) {
                    const int key = k0 + hf * 16 + CROW(lane, v);
                    const float xv = (key <= x.my_pos[lane]) ? x.sc[lane][v] * P.scale : -INFINITY;
                    x.sc[lane][v] = xv;
                    mx[lane] = std::fmax(mx[lane], xv);
                }
            }
            for (int lane = 0; lane < WARP; ++lane) x.mx[lane] = std::fmax(mx[lane], shfl16(mx, lane));
            for (int lane = 0; lane < 16; ++lane) xmx[(qt * 2 + hf) * 16 + lane] = x.mx[lane];
        }
        // barrier; phase B: m_new, alpha, p, P^T -> xpt, wave 0 partial sum -> xrs
        for (int wi = 0; wi < NW; ++wi) {
            W& x = w[wi];
            const int qt = wi >> 1, hf = wi & 1;
            float mxa[WARP];
            bool any = false;
            for (int lane = 0; lane < WARP; ++lane) {
                mxa[lane] = std::fmax(x.mx[lane], xmx[(qt * 2 + (hf ^ 1)) * 16 + lane % 16]);
                if (g_rs == 2) any |= mxa[lane] > x.m[lane] + TLAZY;
            }
            x.resc = g_rs == 2 ? any : true;
            for (int lane = 0; lane < WARP; ++lane) {
                const float mx = mxa[lane];
                float m_new = std::fmax(x.m[lane], mx);
                if (g_rs == 2 && !x.resc) m_new = x.m[lane];
                x.alpha[lane] = (g_rs == 2 && !x.resc) ? 1.f : (x.m[lane] == -INFINITY) ? 0.f : ex(x.m[lane] - m_new);
                for (int v = 0; v < 8; ++v) {
                    const float p = (x.sc[lane][v] == -INFINITY) ? 0.f : ex(x.sc[lane][v] - m_new);
                    x.sc[lane][v] = p;
                    x.pt[lane][v] = f2h(p);
                }
                if (hf == 0) {
                    float rs = 0.f;
                    for (int v = 0; v < 8; ++v) rs += x.sc[lane][v];
                    xrs[qt * 32 + lane] = rs;
                }
                for (int v = 0; v < 8; ++v) xpt[((qt * 2 + hf) * 32 + lane) * 8 + v] = x.pt[lane][v];
                x.m[lane] = m_new;
            }
            if (g_rs == 1) {
                bool ne = false;
                for (int lane = 0; lane < WARP; ++lane) ne |= x.alpha[lane] != 1.f;
                x.resc = ne;
            }
        }
        // barrier; phase C: wave 1 row sum + l, rescale, PV
        for (int wi = 0; wi < NW; ++wi) {
            W& x = w[wi];
            const int qt = wi >> 1, hf = wi & 1;
            if (hf == 1) {
                float rs[WARP];
                for (int lane = 0; lane < WARP; ++lane) {
                    rs[lane] = xrs[qt * 32 + lane];
                    if (g_mut == 1) {
                        float r = 0.f;
                        for (int v = 0; v < 8; ++v) r += x.sc[lane][v];
                        rs[lane] = r + xrs[qt * 32 + lane];
                    } else {
                        for (int v = 0; v < 8; ++v) rs[lane] += x.sc[lane][v];
                    }
                }
                for (int lane = 0; lane < WARP; ++lane) {
                    const float r2 = rs[lane] + shfl16(rs, lane);
                    x.l[lane] = x.l[lane] * x.alpha[lane] + r2;
                }
            }
            if (x.resc)
                for (int lane = 0; lane < WARP; ++lane)
                    for (int j = 0; j < 8; ++j)
                        for (int v = 0; v < 8; ++v) x.o[j][lane][v] *= x.alpha[lane];
            for (int kt = 0; kt < 2; ++kt) {
                uint16_t pk[WARP][8];
                for (int lane = 0; lane < WARP; ++lane)
                    for (int v = 0; v < 8; ++v) pk[lane][v] = (kt == hf) ? x.pt[lane][v] : xpt[((qt * 2 + kt) * 32 + lane) * 8 + v];
                Frag pb;
                join_halves(pk, pb);
                for (int j = 0; j < 8; ++j) {
                    const int dt = hf * 8 + j;
                    const int C = (2 * kt) ^ (2 * (dt & 1));
                    Frag a;
                    for (int lane = 0; lane < WARP; ++lane) {
                        const int s0 = (lane % 16) >> 3;
                        const int lo = (lane % 16) * FA_VS + s0 * 8 + dt * 16 * FA_VS + C * 8;
                        const int hi = (lane % 16) * FA_VS + 8 - s0 * 8 + dt * 16 * FA_VS + C * 8;
                        for (int i = 0; i < 8; ++i) {
                            a[lane][i] = L.Vt[lo + i];
                            a[lane][8 + i] = L.Vt[hi + i];
                        }
                    }
                    Acc acc;
                    std::memcpy(acc, x.o[j], sizeof acc);
                    wmma(a, pb, acc);
                    std::memcpy(x.o[j], acc, sizeof acc);
                }
            }
        }
    }
    // l exchange: wave 1 -> wave 0
    for (int wi = 1; wi < NW; wi += 2)
        for (int lane = 0; lane < WARP; ++lane) xrs[(wi >> 1) * 32 + lane] = w[wi].l[lane];
    for (int wi = 0; wi < NW; wi += 2)
        for (int lane = 0; lane < WARP; ++lane) w[wi].l[lane] = xrs[(wi >> 1) * 32 + lane];
    for (int wi = 0; wi < NW; ++wi) {
        W& x = w[wi];
        const int hf = wi & 1;
        for (int lane = 0; lane < WARP; ++lane) {
            if (x.my_q[lane] >= P.n_tok) continue;
            const size_t b = ((size_t)x.my_q[lane] * P.n_head + h);
            if (S.seg && k_end < n_keys) {
                if (hf == 1 && lane < 16) {
                    (*S.st_ml)[b * 2] = x.m[lane];
                    (*S.st_ml)[b * 2 + 1] = x.l[lane];
                }
                for (int j = 0; j < 8; ++j)
                    for (int v = 0; v < 8; ++v) out[b * FA_D + (hf * 8 + j) * 16 + CROW(lane, v)] = x.o[j][lane][v];
                continue;
            }
            const float inv = 1.f / x.l[lane];
            const float* gate = &P.q[b * P.q_stride + FA_D];
            for (int j = 0; j < 8; ++j)
                for (int v = 0; v < 8; ++v) {
                    const int d = (hf * 8 + j) * 16 + CROW(lane, v);
                    out[b * FA_D + d] = x.o[j][lane][v] * inv / (1.f + ex(-gate[d]));
                }
        }
    }
}

// one launch (all blocks) or the SEG series over key ranges of `seg` keys
std::vector<float> run(const Prob& P, int kind /*0 v1, 4 / 8 v2 NQT*/, int seg) {
    std::vector<float> out((size_t)P.n_tok * P.n_head * FA_D, NAN), st_ml((size_t)P.n_tok * P.n_head * 2, NAN);
    const int FQ = kind == 0 ? 128 : 16 * kind;
    const int nb = (P.n_tok + FQ - 1) / FQ, rows = P.pos0 + P.n_tok;
    auto launch = [&](const SegArgs& S) {
        for (int h = 0; h < P.n_head; ++h)
            for (int bx = 0; bx < nb; ++bx) {
                if (kind == 0)
                    v1_block(P, out, bx, h, S);
                else
                    v2_block(P, out, bx, h, kind, S);
            }
    };
    if (seg <= 0) {
        launch(SegArgs{});
    } else {
        for (int r0 = 0; r0 < rows; r0 += seg) {
            SegArgs S;
            S.seg = true;
            S.kb0 = r0;
            S.kb1 = std::min(rows, r0 + seg);
            S.st_ml = &st_ml;
            launch(S);
        }
    }
    return out;
}

std::vector<double> reference(const Prob& P) {
    std::vector<double> out((size_t)P.n_tok * P.n_head * FA_D);
    for (int t = 0; t < P.n_tok; ++t)
        for (int h = 0; h < P.n_head; ++h) {
            const int kvh = h / (P.n_head / P.n_kv), np = P.pos0 + t + 1;
            const float* qr = &P.q[((size_t)t * P.n_head + h) * P.q_stride];
            std::vector<double> s(np);
            double mx = -INFINITY;
            for (int k = 0; k < np; ++k) {
                const size_t kr = ((size_t)P.kv_row(k) * P.n_kv + kvh) * FA_D;
                double d = 0;
                for (int i = 0; i < FA_D; ++i) d += (double)h2f(f2h(qr[i])) * h2f(P.K[kr + i]);
                s[k] = d * P.scale;
                mx = std::max(mx, s[k]);
            }
            double sum = 0;
            for (int k = 0; k < np; ++k) sum += (s[k] = std::exp(s[k] - mx));
            for (int i = 0; i < FA_D; ++i) {
                double a = 0;
                for (int k = 0; k < np; ++k) a += s[k] * h2f(P.V[((size_t)P.kv_row(k) * P.n_kv + kvh) * FA_D + i]);
                out[((size_t)t * P.n_head + h) * FA_D + i] = a / sum / (1.0 + std::exp(-(double)qr[FA_D + i]));
            }
        }
    return out;
}

void check(bool ok, const std::string& what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++g_fail;
}
bool bitsEqual(const std::vector<float>& a, const std::vector<float>& b, size_t* first = nullptr) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            if (first) *first = i;
            return false;
        }
    return true;
}

Prob make(int n_head, int n_kv, int n_tok, int pos0, unsigned seed) {
    Prob P;
    P.n_head = n_head;
    P.n_kv = n_kv;
    P.n_tok = n_tok;
    P.pos0 = pos0;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    const int rows = pos0 + n_tok, pages = (rows + KV_PAGE - 1) / KV_PAGE, pool_pages = pages + 2;
    P.tab0 = 3;
    P.ptab.assign(P.tab0 + pages, -1);
    std::vector<int> perm(pool_pages);
    for (int i = 0; i < pool_pages; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < pages; ++i) P.ptab[P.tab0 + i] = perm[i];
    const size_t pool = (size_t)pool_pages * KV_PAGE * n_kv * FA_D;
    P.K.assign(pool, 0x7e00);  // NaN everywhere no key lives
    P.V.assign(pool, 0x7e00);
    for (int p = 0; p < rows; ++p)
        for (int kh = 0; kh < n_kv; ++kh)
            for (int i = 0; i < FA_D; ++i) {
                const size_t e = ((size_t)P.kv_row(p) * n_kv + kh) * FA_D + i;
                P.K[e] = f2h(nd(rng) * 1.5f);
                P.V[e] = f2h(nd(rng));
            }
    P.q.resize((size_t)n_tok * n_head * P.q_stride);
    for (auto& x : P.q) x = nd(rng);
    return P;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1) g_mut = std::atoi(argv[1]);
    if (g_mut) std::printf("negative control %d (failures expected for 1)\n", g_mut);
    struct Case {
        int n_head, n_kv, n_tok, pos0;
    };
    // GQA 2 / 4 / 6 (Swift 6, Ornith 8 scaled down), partial blocks (n_tok % 64, % 128), pos0 inside a page,
    // across pages, and n_tok < 64.
    const Case cases[] = {{4, 2, 77, 0}, {4, 1, 200, 300}, {6, 1, 130, 513}, {8, 1, 40, 700}, {2, 1, 256, 31}};
    unsigned seed = 1234;
    for (const Case& c : cases) {
        const Prob P = make(c.n_head, c.n_kv, c.n_tok, c.pos0, seed++);
        const std::string tag = "heads " + std::to_string(c.n_head) + "/" + std::to_string(c.n_kv) + " n_tok " +
                                std::to_string(c.n_tok) + " pos0 " + std::to_string(c.pos0);
        g_halfviol = 0;
        const std::vector<float> v1 = run(P, 0, 0);
        const std::vector<double> ref = reference(P);
        double num = 0, den = 0, mad = 0;
        bool finite = true;
        for (size_t i = 0; i < v1.size(); ++i) {
            if (!std::isfinite(v1[i])) finite = false;
            const double d = v1[i] - ref[i];
            num += d * d;
            den += ref[i] * ref[i];
            mad = std::max(mad, std::fabs(d));
        }
        const double nmse = num / den;
        check(finite && nmse < 1e-5 && mad < 5e-3, "v1 vs double reference, " + tag + " (nmse " + std::to_string(nmse) +
                                                        ", max abs " + std::to_string(mad) + ", all finite " +
                                                        (finite ? "yes" : "no") + ")");
        const std::vector<float> v1s = run(P, 0, 256);
        check(bitsEqual(v1s, v1), "v1 SEG series (256-key ranges) == v1, " + tag);
        for (int rsm : {1, 2}) {  // v2 q8 with the rescale skip (bitwise) and the lazy max (lossy, tolerance only)
            g_rs = rsm;
            const std::vector<float> v2 = run(P, 8, 0);
            double num2 = 0, den2 = 0, mad2 = 0;
            for (size_t i = 0; i < v2.size(); ++i) {
                const double d = v2[i] - ref[i];
                num2 += d * d;
                den2 += ref[i] * ref[i];
                mad2 = std::max(mad2, std::fabs(d));
            }
            if (rsm == 1) {
                check(bitsEqual(v2, v1) && bitsEqual(run(P, 8, 256), v1), "v2 q8 RS1 (rescale skip) == v1 bitwise (plain + SEG 256), " + tag);
            } else {
                size_t diff = 0;
                for (size_t i = 0; i < v2.size(); ++i) diff += std::memcmp(&v2[i], &v1[i], 4) != 0;
                check(num2 / den2 < 1e-5 && mad2 < 5e-3, "v2 q8 RS2 (lazy max, lossy) vs double reference, " + tag + " (nmse " +
                                                             std::to_string(num2 / den2) + ", max abs " + std::to_string(mad2) +
                                                             ", values != v1: " + std::to_string(diff) + " of " + std::to_string(v2.size()) + ")");
                check(bitsEqual(run(P, 8, 256), v2), "v2 q8 RS2 SEG 256 == v2 q8 RS2 (state carried), " + tag);
            }
            g_rs = 0;
        }
        for (int nqt : {4, 8}) {
            size_t first = 0;
            const std::vector<float> v2 = run(P, nqt, 0);
            const bool eq = bitsEqual(v2, v1, &first);
            check(eq, "v2 q" + std::to_string(nqt) + " == v1 bitwise, " + tag +
                          (eq ? "" : " (first diff at " + std::to_string(first) + ")"));
            for (int seg : {256, 512}) {
                const std::vector<float> v2s = run(P, nqt, seg);
                check(bitsEqual(v2s, v1), "v2 q" + std::to_string(nqt) + " SEG " + std::to_string(seg) + " == v1, " + tag);
            }
        }
        check(g_halfviol == 0, "WMMA half-wave operand rule, " + tag + " (violations " + std::to_string(g_halfviol) + ")");
    }
    std::printf("%s: %d failure(s)\n", g_fail == 0 ? "ALL PASS" : "FAILED", g_fail);
    return g_fail;
}
