// whirl-i8qk-ref-tests (CPU only, v0252-8sattn / PLAN V1-148): the int8 Q.K^T of the balance gfx1151 prefill attention
// (kernels/gfx1151/attn.hip attn_prefill_wmma_i8 / _s_i8, math in kernels/shared/attn_i8qk.h), emulated on the host:
//   1. codes: every code -127..127 packs and unpacks (bits [8 i, 8 i + 8) = value i, two's complement); |code| <= 127
//      for any finite x with |x| <= amax, rounding error <= half a step; an all-zero row has inv 0, codes 0, scale 0;
//   2. fragment layout: the kernel's int8 K rows in LDS (stride FA_K8S = 68 words; words 2 j, 2 j + 1 of a key = dims
//      8 j .. 8 j + 7, written by lane j of the wave that holds the key) and its Q pieces (2 words per 16-dim step: dims
//      16 st + 8 (lane / 16) .. +7) joined as the kernel joins them (lane < 16: {own0, own1, partner0, partner1}, lane >=
//      16: {partner0, partner1, own0, own1}) fed to an emulated v_wmma_i32_16x16x16_iu8 (byte m of the 4-dword operand =
//      K element m on both operands; C row = A row = key, column = B column = query; A fragment of key row r, step st =
//      words 4 st .. 4 st + 3 of the row) give exactly sum_d code_q[d] code_k[d] for every (key, query) of a 32-key tile;
//      a negative control (B halves not swapped on the upper lanes) must fail;
//   3. error model (diagnostic, printed; checked against the analytic rounding-noise expectation only): the score error of
//      s_q s_k int-dot against the f64 dot on Gaussian rows (the Hadamard-rotated q8h case), and the attention output
//      error of a 512-key softmax for int8 against the int4 rule of the fast kernel. No model-quality claim is made
//      here: balance acceptance is task accuracy (>= 99 % of precise per family) on the GPU.
// Exit code 0 = all pass. SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "../../../kernels/shared/attn_i8qk.h"

namespace {

int g_fail = 0;
#define I8CHECK(c, ...)                         \
    do {                                        \
        if (!(c)) {                             \
            ++g_fail;                           \
            std::printf("FAIL %d: ", __LINE__); \
            std::printf(__VA_ARGS__);           \
            std::printf("\n");                  \
        }                                       \
    } while (0)

constexpr int D = 256;
constexpr int KS = 68;  // FA_K8S in kernels/gfx1151/attn.hip

float amaxOf(const float* x, int n) {
    float a = 0.f;
    for (int i = 0; i < n; ++i) a = std::fmax(a, std::fabs(x[i]));
    return a;
}

void testCodes() {
    for (int c = -127; c <= 127; ++c) {
        float x[4];
        for (int i = 0; i < 4; ++i) x[i] = static_cast<float>(i % 2 ? c : -c);
        const uint32_t w = i8qk_pack4(x, 1.f);
        for (int i = 0; i < 4; ++i) I8CHECK(i8qk_byte(w, i) == (i % 2 ? c : -c), "code %d lane %d -> %d", c, i, i8qk_byte(w, i));
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    for (int t = 0; t < 4000; ++t) {
        float x[4];
        for (float& v : x) v = nd(rng);
        const float am = amaxOf(x, 4), inv = i8qk_inv(am);
        for (int i = 0; i < 4; ++i) {
            const int c = i8qk_code(x[i], inv);
            I8CHECK(c >= -127 && c <= 127, "code out of range %d", c);
            I8CHECK(std::fabs(c * i8qk_scale(am) - x[i]) <= 0.5f * i8qk_scale(am) * 1.0001f + 1e-6f, "rounding error too large x %g c %d", x[i], c);
        }
        // the row maximum maps to +-127 exactly
        int imax = 0;
        for (int i = 1; i < 4; ++i)
            if (std::fabs(x[i]) > std::fabs(x[imax])) imax = i;
        I8CHECK(std::abs(i8qk_code(x[imax], inv)) == 127, "amax element -> %d", i8qk_code(x[imax], inv));
    }
    const float z[4] = {0, 0, 0, 0};
    I8CHECK(i8qk_inv(0.f) == 0.f && i8qk_scale(0.f) == 0.f && i8qk_pack4(z, 0.f) == 0u, "zero row");
    // the int32 sum of 256 products is exact: |sum| <= 256 * 127^2 < 2^24 (exact as f32 too)
    I8CHECK(256LL * 127 * 127 < (1LL << 24), "int dot bound");
}

// v_wmma_i32_16x16x16_iu8 on wave32 gfx11: lane l supplies A row l % 16 (16 K bytes in 4 dwords) and B column l % 16;
// lanes l and l + 16 must agree. C element v of lane l = row 2 v + l / 16, column l % 16.
int wmmaIu8(const uint32_t a[32][4], const uint32_t b[32][4], int32_t c[32][8]) {
    int disagree = 0;
    for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 4; ++j) disagree += (a[l][j] != a[l % 16][j]) + (b[l][j] != b[l % 16][j]);
    for (int l = 0; l < 32; ++l)
        for (int v = 0; v < 8; ++v) {
            const int row = 2 * v + l / 16, col = l % 16;
            int32_t s = 0;
            for (int m = 0; m < 16; ++m) s += i8qk_byte(a[row][m / 4], m % 4) * i8qk_byte(b[col][m / 4], m % 4);
            c[l][v] += s;
        }
    return disagree;
}

// returns the number of tile elements that differ from the code dot (swap = the kernel's join; !swap = negative control)
int fragmentRun(bool swap, int& disagree) {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int NK = 32, NQ = 16;
    std::vector<float> K(NK * D), Q(NQ * D);
    for (float& v : K) v = nd(rng);
    for (float& v : Q) v = nd(rng);
    // the kernel's LDS rows: lane j (of the wave holding key r) packs dims 8 j .. +7 into words 2 j, 2 j + 1
    std::vector<uint32_t> Ks8(NK * KS, 0u);
    for (int r = 0; r < NK; ++r) {
        const float inv = i8qk_inv(amaxOf(&K[r * D], D));
        for (int j = 0; j < 32; ++j) {
            Ks8[r * KS + 2 * j] = i8qk_pack4(&K[r * D + 8 * j], inv);
            Ks8[r * KS + 2 * j + 1] = i8qk_pack4(&K[r * D + 8 * j + 4], inv);
        }
    }
    // per lane: query l % 16 of the wave, own piece = dims 16 st + 8 (l / 16) .. +7 (2 words)
    std::vector<uint32_t> own(32 * 16 * 2);
    for (int l = 0; l < 32; ++l) {
        const float* q = &Q[(l % 16) * D];
        const float inv = i8qk_inv(amaxOf(q, D));
        for (int st = 0; st < 16; ++st) {
            own[(l * 16 + st) * 2] = i8qk_pack4(q + 16 * st + 8 * (l / 16), inv);
            own[(l * 16 + st) * 2 + 1] = i8qk_pack4(q + 16 * st + 8 * (l / 16) + 4, inv);
        }
    }
    int32_t acc[2][32][8] = {};
    disagree = 0;
    for (int st = 0; st < 16; ++st)
        for (int kt = 0; kt < 2; ++kt) {
            uint32_t a[32][4], b[32][4];
            for (int l = 0; l < 32; ++l) {
                const uint32_t* ar = &Ks8[(kt * 16 + l % 16) * KS + 4 * st];
                for (int j = 0; j < 4; ++j) a[l][j] = ar[j];
                const uint32_t o0 = own[(l * 16 + st) * 2], o1 = own[(l * 16 + st) * 2 + 1];
                const uint32_t p0 = own[((l ^ 16) * 16 + st) * 2], p1 = own[((l ^ 16) * 16 + st) * 2 + 1];
                const bool lo = swap ? l < 16 : true;
                b[l][0] = lo ? o0 : p0;
                b[l][1] = lo ? o1 : p1;
                b[l][2] = lo ? p0 : o0;
                b[l][3] = lo ? p1 : o1;
            }
            disagree += wmmaIu8(a, b, acc[kt]);
        }
    int bad = 0;
    for (int kt = 0; kt < 2; ++kt)
        for (int l = 0; l < 32; ++l)
            for (int v = 0; v < 8; ++v) {
                const int key = kt * 16 + 2 * v + l / 16, qi = l % 16;
                const float ik = i8qk_inv(amaxOf(&K[key * D], D)), iq = i8qk_inv(amaxOf(&Q[qi * D], D));
                int32_t want = 0;
                for (int d = 0; d < D; ++d) want += i8qk_code(K[key * D + d], ik) * i8qk_code(Q[qi * D + d], iq);
                bad += acc[kt][l][v] != want;
            }
    return bad;
}

void testFragments() {
    int dis = 0;
    const int bad = fragmentRun(true, dis);
    I8CHECK(bad == 0 && dis == 0, "fragment layout: %d of 512 tile elements differ from the code dot, %d lane disagreements", bad, dis);
    int dis_n = 0;
    const int bad_n = fragmentRun(false, dis_n);
    I8CHECK(bad_n > 0 || dis_n > 0, "negative control (upper lanes not swapped) passed: the check does not discriminate");
}

double scoreErr(const std::vector<float>& Q, const std::vector<float>& K, int nq, int nk, int levels) {
    double se = 0;
    for (int i = 0; i < nq; ++i) {
        const float* q = &Q[i * D];
        const float aq = amaxOf(q, D), iq = aq > 0 ? levels / aq : 0, sq = aq / levels;
        for (int j = 0; j < nk; ++j) {
            const float* k = &K[j * D];
            const float ak = amaxOf(k, D), ik = ak > 0 ? levels / ak : 0, sk = ak / levels;
            double ex = 0, qd = 0;
            for (int d = 0; d < D; ++d) {
                ex += static_cast<double>(q[d]) * k[d];
                qd += static_cast<double>(std::nearbyint(q[d] * iq)) * std::nearbyint(k[d] * ik);
            }
            se += (qd * sq * sk - ex) * (qd * sq * sk - ex);
        }
    }
    return std::sqrt(se / (static_cast<double>(nq) * nk));
}

// attention output rel rms error of one rule (levels 127 = int8, 7 = int4) against exact softmax(P).V
double attnErr(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V, int nq, int nk, int levels) {
    double oerr = 0, onorm = 0;
    for (int i = 0; i < nq; ++i) {
        const float* q = &Q[i * D];
        const float aq = amaxOf(q, D), iq = aq > 0 ? levels / aq : 0, sq = aq / levels;
        std::vector<double> se(nk), sa(nk);
        double me = -1e30, ma = -1e30;
        for (int j = 0; j < nk; ++j) {
            const float* k = &K[j * D];
            const float ak = amaxOf(k, D), ik = ak > 0 ? levels / ak : 0, sk = ak / levels;
            double ex = 0, c = 0;
            for (int d = 0; d < D; ++d) {
                ex += static_cast<double>(q[d]) * k[d];
                c += std::nearbyint(q[d] * iq) * std::nearbyint(k[d] * ik);
            }
            se[j] = ex * 4.0 / 16.0;  // a 4x sharper head than unit rows give
            sa[j] = static_cast<double>(static_cast<float>(c) * (sk * sq)) * 4.0 / 16.0;
            me = std::fmax(me, se[j]);
            ma = std::fmax(ma, sa[j]);
        }
        double le = 0, la = 0;
        for (int j = 0; j < nk; ++j) {
            se[j] = std::exp(se[j] - me);
            sa[j] = std::exp(sa[j] - ma);
            le += se[j];
            la += sa[j];
        }
        for (int d = 0; d < D; ++d) {
            double oe = 0, oa = 0;
            for (int j = 0; j < nk; ++j) {
                oe += se[j] * V[j * D + d];
                oa += sa[j] * V[j * D + d];
            }
            oe /= le;
            oa /= la;
            oerr += (oa - oe) * (oa - oe);
            onorm += oe * oe;
        }
    }
    return std::sqrt(oerr / onorm);
}

void testErrorModel() {
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int NQ = 32, NK = 512;
    std::vector<float> Q(NQ * D), K(NK * D), V(NK * D);
    for (float& v : Q) v = nd(rng);
    for (float& v : K) v = nd(rng);
    for (float& v : V) v = nd(rng);
    const double e8 = scoreErr(Q, K, NQ, NK, 127), e4 = scoreErr(Q, K, NQ, NK, 7);
    // analytic: Delta = amax / 127 per row (amax of 256 unit Gaussians ~ 2.9), dot error var = D (Delta_q^2 + Delta_k^2) / 12
    const double am = 2.9, pred8 = std::sqrt(D * 2.0 * (am / 127) * (am / 127) / 12.0);
    std::printf("score error (unit Gaussian rows, d 256): int8 rms %.4f (analytic %.4f; dot spread 16), int4 rms %.3f\n", e8, pred8, e4);
    I8CHECK(e8 > 0.5 * pred8 && e8 < 2.0 * pred8, "int8 score error %.4f far from the analytic %.4f", e8, pred8);
    I8CHECK(e8 < e4 / 8, "int8 error %.4f not well below int4 %.3f", e8, e4);
    const double a8 = attnErr(Q, K, V, NQ, NK, 127), a4 = attnErr(Q, K, V, NQ, NK, 7);
    std::printf("attention output rel rms error, 512 keys, 4x-sharp scores: int8 %.5f, int4 %.4f (diagnostic)\n", a8, a4);
    I8CHECK(std::isfinite(a8) && std::isfinite(a4) && a8 < a4, "attention emulation: int8 %.5f not below int4 %.4f", a8, a4);
}

}  // namespace

int main() {
    testCodes();
    testFragments();
    testErrorModel();
    std::printf("i8qk_ref_tests: %s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
