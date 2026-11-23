// whirl-i4qk-ref-tests (CPU only, v0252-8sfp / PLAN V1-144): the int4 Q.K^T of the fast gfx1151 prefill attention
// (kernels/gfx1151/attn.hip attn_prefill_wmma_i4 / _s_i4, math in kernels/shared/attn_i4qk.h), emulated on the host:
//   1. codes: every code -7..7 packs and unpacks (bits [4 i, 4 i + 4) = value i, two's complement); |code| <= 7 for any
//      finite x with |x| <= amax; an all-zero row has inv 0, codes 0, scale 0;
//   2. fragment layout: the kernel's int4 K rows in LDS (word j of a key = dims 8 j .. 8 j + 7) and its Q fragments
//      (lane < 16: {own word, partner word}, lane >= 16: {partner, own}; own = dims 16 st + 8 (lane / 16) ..) fed to an
//      emulated v_wmma_i32_16x16x16_iu4 (nibble m = K element m on both operands; C row = A row = key, column = B column
//      = query: hipfire gemm_mq4g256v2_residual_mmq_iu4.gfx11.hip:21-26) give exactly sum_d code_q[d] code_k[d] for every
//      (key, query) of a 32-key tile and the C element (row 2 v + lane / 16, column lane % 16) of each lane;
//   3. error model (diagnostic, printed): on Gaussian rows (the Hadamard-rotated q8h case) the score error of
//      s_q s_k int-dot against the f64 dot, and the attention output error of a 4096-key softmax, against the int8
//      per-row variant of the same rule. Checked only against the analytic expectation (uniform rounding noise
//      Delta^2 / 12 per element, Delta = amax / 7), so a broken scale or code rule fails, a model-quality claim is not made.
// Exit code 0 = all pass. SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "../../../kernels/shared/attn_i4qk.h"

namespace {

int g_fail = 0;
#define I4CHECK(c, ...)                         \
    do {                                        \
        if (!(c)) {                             \
            ++g_fail;                           \
            std::printf("FAIL %d: ", __LINE__); \
            std::printf(__VA_ARGS__);           \
            std::printf("\n");                  \
        }                                       \
    } while (0)

constexpr int D = 256;

float amaxOf(const float* x, int n) {
    float a = 0.f;
    for (int i = 0; i < n; ++i) a = std::fmax(a, std::fabs(x[i]));
    return a;
}

void testCodes() {
    for (int c = -7; c <= 7; ++c) {
        float x[8];
        for (int i = 0; i < 8; ++i) x[i] = static_cast<float>(i % 2 ? c : -c);
        const uint32_t w = i4qk_pack8(x, 1.f);
        for (int i = 0; i < 8; ++i) I4CHECK(i4qk_nib(w, i) == (i % 2 ? c : -c), "code %d lane %d -> %d", c, i, i4qk_nib(w, i));
    }
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 3.f);
    for (int t = 0; t < 2000; ++t) {
        float x[8];
        for (float& v : x) v = nd(rng);
        const float am = amaxOf(x, 8), inv = i4qk_inv(am);
        for (int i = 0; i < 8; ++i) {
            const int c = i4qk_code(x[i], inv);
            I4CHECK(c >= -7 && c <= 7, "code out of range %d", c);
            I4CHECK(std::fabs(c * i4qk_scale(am) - x[i]) <= 0.5f * i4qk_scale(am) * 1.0001f + 1e-6f, "rounding error too large x %g c %d", x[i], c);
        }
    }
    const float z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    I4CHECK(i4qk_inv(0.f) == 0.f && i4qk_scale(0.f) == 0.f && i4qk_pack8(z, 0.f) == 0u, "zero row");
}

// v_wmma_i32_16x16x16_iu4 on wave32 gfx11: lane l supplies A row l % 16 (16 K nibbles) and B column l % 16 (16 K
// nibbles); lanes l and l + 16 must agree. C element v of lane l = row 2 v + l / 16, column l % 16.
void wmmaIu4(const uint32_t a[32][2], const uint32_t b[32][2], int32_t c[32][8]) {
    for (int l = 0; l < 32; ++l) {
        I4CHECK(a[l][0] == a[l % 16][0] && a[l][1] == a[l % 16][1], "A lanes %d / %d differ", l, l % 16);
        I4CHECK(b[l][0] == b[l % 16][0] && b[l][1] == b[l % 16][1], "B lanes %d / %d differ", l, l % 16);
    }
    for (int l = 0; l < 32; ++l)
        for (int v = 0; v < 8; ++v) {
            const int row = 2 * v + l / 16, col = l % 16;
            int32_t s = 0;
            for (int m = 0; m < 16; ++m) s += i4qk_nib(a[row][m / 8], m % 8) * i4qk_nib(b[col][m / 8], m % 8);
            c[l][v] += s;
        }
}

void testFragments() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int NK = 32, NQ = 16;
    std::vector<float> K(NK * D), Q(NQ * D);
    for (float& v : K) v = nd(rng);
    for (float& v : Q) v = nd(rng);
    // the kernel's LDS rows: lane j (of the wave holding key r) packs dims 8 j .. +7 into word j
    std::vector<uint32_t> Ks4(NK * 34, 0u);
    for (int r = 0; r < NK; ++r) {
        const float inv = i4qk_inv(amaxOf(&K[r * D], D));
        for (int j = 0; j < 32; ++j) Ks4[r * 34 + j] = i4qk_pack8(&K[r * D + 8 * j], inv);
    }
    // per lane: query l % 16 of the wave, own word = dims 16 st + 8 (l / 16) .. +7
    std::vector<uint32_t> own(32 * 16);
    for (int l = 0; l < 32; ++l) {
        const float* q = &Q[(l % 16) * D];
        const float inv = i4qk_inv(amaxOf(q, D));
        for (int st = 0; st < 16; ++st) own[l * 16 + st] = i4qk_pack8(q + 16 * st + 8 * (l / 16), inv);
    }
    int32_t acc[2][32][8] = {};
    for (int st = 0; st < 16; ++st)
        for (int kt = 0; kt < 2; ++kt) {
            uint32_t a[32][2], b[32][2];
            for (int l = 0; l < 32; ++l) {
                const uint32_t* ar = &Ks4[(kt * 16 + l % 16) * 34 + 2 * st];
                a[l][0] = ar[0];
                a[l][1] = ar[1];
                const uint32_t w = own[l * 16 + st], pw = own[(l ^ 16) * 16 + st];
                b[l][0] = l < 16 ? w : pw;
                b[l][1] = l < 16 ? pw : w;
            }
            wmmaIu4(a, b, acc[kt]);
        }
    int bad = 0;
    for (int kt = 0; kt < 2; ++kt)
        for (int l = 0; l < 32; ++l)
            for (int v = 0; v < 8; ++v) {
                const int key = kt * 16 + 2 * v + l / 16, qi = l % 16;
                const float ik = i4qk_inv(amaxOf(&K[key * D], D)), iq = i4qk_inv(amaxOf(&Q[qi * D], D));
                int32_t want = 0;
                for (int d = 0; d < D; ++d) want += i4qk_code(K[key * D + d], ik) * i4qk_code(Q[qi * D + d], iq);
                bad += acc[kt][l][v] != want;
            }
    I4CHECK(bad == 0, "fragment layout: %d of 512 tile elements differ from the code dot", bad);
}

// int8 per-row variant of the same rule (diagnostic comparison only)
double scoreErr(const std::vector<float>& Q, const std::vector<float>& K, int nq, int nk, int levels, double& rel_spread) {
    double se = 0, sv = 0;
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
            sv += ex * ex;
        }
    }
    rel_spread = std::sqrt(se / sv);
    return std::sqrt(se / (static_cast<double>(nq) * nk));
}

void testErrorModel() {
    std::mt19937 rng(23);
    std::normal_distribution<float> nd(0.f, 1.f);
    constexpr int NQ = 64, NK = 512;
    std::vector<float> Q(NQ * D), K(NK * D);
    for (float& v : Q) v = nd(rng);
    for (float& v : K) v = nd(rng);
    double r4 = 0, r8 = 0;
    const double e4 = scoreErr(Q, K, NQ, NK, 7, r4), e8 = scoreErr(Q, K, NQ, NK, 127, r8);
    // analytic: Delta = amax / L per row (amax of 256 unit Gaussians ~ 2.9), dot error var = D (Delta_q^2 + Delta_k^2) / 12
    const double am = 2.9, pred4 = std::sqrt(D * 2.0 * (am / 7) * (am / 7) / 12.0);
    std::printf("score error (unit Gaussian rows, d 256): int4 rms %.3f (analytic %.3f, %.1f %% of the dot spread 16), int8 rms %.4f\n", e4,
                pred4, 100.0 * r4, e8);
    I4CHECK(e4 > 0.5 * pred4 && e4 < 2.0 * pred4, "int4 score error %.3f far from the analytic %.3f", e4, pred4);
    I4CHECK(e8 < e4 / 8, "int8 error %.4f not well below int4 %.3f", e8, e4);
    // attention output of one sharp-ish row (scores scaled 1 / 16 like the model): int4 vs exact softmax(P).V
    std::vector<float> V(NK * D);
    for (float& v : V) v = nd(rng);
    double oerr = 0, onorm = 0;
    for (int i = 0; i < NQ; ++i) {
        const float* q = &Q[i * D];
        const float aq = amaxOf(q, D), iq = i4qk_inv(aq), sq = i4qk_scale(aq);
        std::vector<double> se(NK), sa(NK);
        double me = -1e30, ma = -1e30;
        for (int j = 0; j < NK; ++j) {
            const float* k = &K[j * D];
            const float ak = amaxOf(k, D), ik = i4qk_inv(ak), sk = i4qk_scale(ak);
            double ex = 0;
            int32_t c = 0;
            for (int d = 0; d < D; ++d) {
                ex += static_cast<double>(q[d]) * k[d];
                c += i4qk_code(q[d], iq) * i4qk_code(k[d], ik);
            }
            se[j] = ex * 4.0 / 16.0;  // a 4x sharper head than unit rows give
            sa[j] = static_cast<double>(static_cast<float>(c) * (sk * sq)) * 4.0 / 16.0;
            me = std::fmax(me, se[j]);
            ma = std::fmax(ma, sa[j]);
        }
        double le = 0, la = 0;
        for (int j = 0; j < NK; ++j) {
            se[j] = std::exp(se[j] - me);
            sa[j] = std::exp(sa[j] - ma);
            le += se[j];
            la += sa[j];
        }
        for (int d = 0; d < D; ++d) {
            double oe = 0, oa = 0;
            for (int j = 0; j < NK; ++j) {
                oe += se[j] * V[j * D + d];
                oa += sa[j] * V[j * D + d];
            }
            oe /= le;
            oa /= la;
            oerr += (oa - oe) * (oa - oe);
            onorm += oe * oe;
        }
    }
    std::printf("attention output rel rms error, int4 Q.K^T, 512 keys, 4x-sharp scores: %.3f (diagnostic)\n", std::sqrt(oerr / onorm));
    I4CHECK(std::isfinite(oerr) && onorm > 0, "attention emulation not finite");
}

}  // namespace

int main() {
    testCodes();
    testFragments();
    testErrorModel();
    std::printf("i4qk_ref_tests: %s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
