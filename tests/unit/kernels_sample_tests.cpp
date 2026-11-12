// v0252-f1arg: CPU tests of the multi-block verify argmax (kernels/argmax_key.h, kernels/sample_argmax_mb.hip; no GPU).
//   * testArgmaxKeys (feat026-argmax's key check, ported with the key from feat026-argmax-p25 f4fa25b): the 64-bit keys
//     combined over random block partitions in random order (atomicMax is order-free) == the legacy argmax scan
//     (kernels/shared/block_argmax.h: best = -inf, idx = 0, x[i] > best; the tree takes the lower index on equal values)
//     on rows with ties, -inf, NaN (incl. index 0), -0 / +0, +inf, denormals;
//   * testArgmaxRankModel: a replay of argmax_rank_rows_to_mb's arithmetic (tiles of any size dealt to any number of blocks
//     the way the kernel's grid loop deals them, scanned in any order, keys max-combined, the F1 counts summed only for a
//     draft inside the row, the row's last block: index = am_idx, row max = x[index] or -inf when the key's high word is
//     AM_ORD_NEG_INF, l_d = x[d] or -inf, rank or INT_MAX) == what argmax_rank_rows_to
//     (kernels/shared/sample_pick_set_tokens.h) computes in one block, bitwise (max / l_d as bits); a draft id n / n - 1;
//   * the launch geometry AM_TILE / AM_THREADS / AM_TILES (v0252-f1arg2, REVIEW C1) and the workspace layout AmWs (512
//     bytes: key / cnt / rank per row of 32).
// The GPU side (the twins bitwise == the one-block kernels on the R9700) is tests/kernels/test_misc.cpp.
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"

#include "../../kernels/argmax_key.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

std::uint64_t g_st = 0x9E3779B97F4A7C15ull;
std::uint64_t rnd() {
    g_st ^= g_st << 13;
    g_st ^= g_st >> 7;
    g_st ^= g_st << 17;
    return g_st;
}

// the legacy one-block scan: first index of the maximum over values above -inf, 0 when none; m = x[index] or -inf
int amLegacy(const std::vector<float>& x, float& m) {
    float best = -std::numeric_limits<float>::infinity();
    int idx = 0;
    for (int i = 0; i < static_cast<int>(x.size()); ++i)
        if (x[static_cast<std::size_t>(i)] > best) {
            best = x[static_cast<std::size_t>(i)];
            idx = i;
        }
    m = best;
    return idx;
}

std::uint32_t bitsOf(float v) {
    std::uint32_t u;
    std::memcpy(&u, &v, 4);
    return u;
}

// a random row: values that stress the order (signed zeros, infinities, NaNs, denormals, near-equal values), many
// exact ties, no value above -inf at all, or plain spread values
std::vector<float> randomRow(int n, int mode) {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float den = std::numeric_limits<float>::denorm_min();
    const float pool[] = {0.f, -0.f, inf, -inf, nan, -nan, den, -den, 1.f, -1.f, 1.0000001f, 0.99999994f, 3.4e38f, -3.4e38f, 2.5f, -2.5f};
    std::vector<float> x(static_cast<std::size_t>(n));
    for (float& v : x) {
        const std::uint64_t r = rnd();
        if (mode == 0)
            v = pool[r % (sizeof pool / sizeof pool[0])];
        else if (mode == 1)
            v = static_cast<float>(static_cast<int>(r % 7) - 3);  // many exact ties
        else if (mode == 2)
            v = (r % 3 == 0) ? nan : -inf;  // no value above -inf
        else if (mode == 3)
            v = (r % 5 == 0) ? -0.f : ((r % 5 == 1) ? 0.f : -static_cast<float>(r % 100) - 1.f);  // a +-0 maximum
        else
            v = static_cast<float>(static_cast<std::int64_t>(r % 2000001) - 1000000) * 1e-3f;
    }
    return x;
}

void testArgmaxKeys() {
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float den = std::numeric_limits<float>::denorm_min();
    int cases = 0, bad = 0;
    for (int it = 0; it < 4000; ++it) {
        const int n = 1 + static_cast<int>(rnd() % (it < 2000 ? 64 : 9000));
        const std::vector<float> x = randomRow(n, static_cast<int>(rnd() % 5));
        // blocks: random tile size; each block folds its tile in any order, then blocks combine in random order
        const int tile = 1 + static_cast<int>(rnd() % 300);
        std::vector<am_key_t> part;
        for (int b = 0; b < n; b += tile) {
            am_key_t k = 0;
            const int e = std::min(n, b + tile);
            for (int i = e - 1; i >= b; --i) k = am_max(k, am_key(x[static_cast<std::size_t>(i)], i));  // reverse order on purpose
            part.push_back(k);
        }
        for (std::size_t i = part.size(); i > 1; --i) std::swap(part[i - 1], part[rnd() % i]);
        am_key_t g = 0;  // the zeroed workspace key
        for (const am_key_t k : part) g = am_max(g, k);
        ++cases;
        float m = 0.f;
        const int want = amLegacy(x, m);
        if (am_idx(g) != want) {
            if (bad < 5) std::printf("FAIL argmax key: n %d tile %d: key idx %d, legacy %d\n", n, tile, am_idx(g), want);
            ++bad;
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(cases == 4000);
    // fixed cases
    CHECK_EQ(am_idx(am_key(-inf, 0)), 0);
    CHECK(am_key(-inf, 7) > 0ull);                     // every key beats the zeroed workspace
    CHECK(am_key(0.f, 3) == am_key(-0.f, 3));          // -0 == +0
    CHECK(am_key(nan, 5) == am_key(-inf, 5));          // NaN never wins
    CHECK(am_key(1.f, 9) > am_key(1.f, 10));           // lower index wins a tie
    CHECK(am_key(inf, 1000) > am_key(3.4e38f, 0));
    CHECK(am_key(-den, 0) < am_key(den, 99));
    CHECK(am_key(den, 5) > am_key(0.f, 0));            // a denormal is above zero (IEEE order, no flush)
    CHECK_EQ(am_idx(am_key(2.f, 248319)), 248319);
    // the high word of -inf / NaN is AM_ORD_NEG_INF, and nothing else maps there (the row max is -inf exactly then)
    CHECK_EQ(static_cast<std::uint32_t>(am_key(-inf, 0) >> 32), AM_ORD_NEG_INF);
    CHECK_EQ(static_cast<std::uint32_t>(am_key(nan, 9) >> 32), AM_ORD_NEG_INF);
    CHECK_EQ(static_cast<std::uint32_t>(am_key(-std::numeric_limits<float>::max(), 0) >> 32), AM_ORD_NEG_INF + 1u);
}

// argmax_rank_rows_to in one block (kernels/shared/sample_pick_set_tokens.h:72-102): what it writes for one row. The
// predicate is written out here (the kernel's literal expression), not taken from argmax_key.h's am_ahead, so a broken
// helper cannot hide behind its own oracle.
struct RankOut {
    int tok, rank;
    std::uint32_t max_bits, ld_bits;
};
RankOut rankLegacy(const std::vector<float>& x, int d) {
    const int n = static_cast<int>(x.size());
    const bool has = d >= 0 && d < n;
    const float ld = has ? x[static_cast<std::size_t>(d)] : -std::numeric_limits<float>::infinity();
    float m = 0.f;
    const int tok = amLegacy(x, m);
    int cnt = 0;
    for (int i = 0; i < n; ++i) {  // the one-block kernel counts with d = -1 too
        const float v = x[static_cast<std::size_t>(i)];
        cnt += (v > ld || (v == ld && i < d)) ? 1 : 0;
    }
    return {tok, has ? cnt : std::numeric_limits<int>::max(), bitsOf(m), bitsOf(ld)};
}

// argmax_rank_rows_to_mb's arithmetic: tiles of `tile` values (the kernel: AM_TILE; any size here) dealt to `blocks` blocks
// the way the kernel's grid loop deals them (block b: tiles b, b + blocks, ...; blocks <= 0: one block per tile = the
// launchers' grid AM_TILES(n); more blocks than tiles: empty blocks with key 0), each block's key max and count, combined
// in random order, then the row's last block
RankOut rankMb(const std::vector<float>& x, int d, int tile, int blocks = 0) {
    const int n = static_cast<int>(x.size());
    const bool has = d >= 0 && d < n;
    const float ld = has ? x[static_cast<std::size_t>(d)] : -std::numeric_limits<float>::infinity();
    const int tiles = tile == AM_TILE ? AM_TILES(n) : (n + tile - 1) / tile;
    const int nb = blocks > 0 ? blocks : tiles;
    std::vector<std::pair<am_key_t, int>> part;
    for (int blk = 0; blk < nb; ++blk) {
        am_key_t k = 0;
        int c = 0;
        for (int t = blk; t < tiles; t += nb) {
            const int b = t * tile, e = std::min(n, b + tile);
            for (int i = b; i < e; ++i) {
                k = am_max(k, am_key(x[static_cast<std::size_t>(i)], i));
                if (has) c += am_ahead(x[static_cast<std::size_t>(i)], i, ld, d);
            }
        }
        part.push_back({k, c});
    }
    for (std::size_t i = part.size(); i > 1; --i) std::swap(part[i - 1], part[rnd() % i]);
    am_key_t g = 0;
    int rank = 0;
    for (const auto& [k, c] : part) {
        g = am_max(g, k);
        rank += c;
    }
    const int idx = am_idx(g);
    const float m = static_cast<std::uint32_t>(g >> 32) == AM_ORD_NEG_INF ? -std::numeric_limits<float>::infinity() : x[static_cast<std::size_t>(idx)];
    return {idx, has ? rank : std::numeric_limits<int>::max(), bitsOf(m), bitsOf(ld)};
}

void testArgmaxRankModel() {
    int cases = 0, bad = 0;
    for (int it = 0; it < 3000; ++it) {
        const int n = 1 + static_cast<int>(rnd() % (it < 1500 ? 64 : 12000));
        const std::vector<float> x = randomRow(n, static_cast<int>(rnd() % 5));
        int d;
        switch (static_cast<int>(rnd() % 6)) {
            case 0: d = -1; break;                                            // no draft
            case 1: d = n + static_cast<int>(rnd() % 5); break;               // id outside the row
            case 2: d = -2 - static_cast<int>(rnd() % 5); break;              // a negative id in the draft word
            default: d = static_cast<int>(rnd() % static_cast<std::uint64_t>(n)); break;  // inside
        }
        const int tile = (it % 3 == 0) ? AM_TILE : 1 + static_cast<int>(rnd() % 700);
        // the launchers' grid (one block per tile) on two of three rows, else any grid from 1 block to tiles + 3 (C1)
        const int ntile = (n + tile - 1) / tile;
        const int blocks = (it % 3 == 1) ? 1 + static_cast<int>(rnd() % static_cast<std::uint64_t>(ntile + 3)) : 0;
        const RankOut a = rankLegacy(x, d), b = rankMb(x, d, tile, blocks);
        ++cases;
        if (a.tok != b.tok || a.rank != b.rank || a.max_bits != b.max_bits || a.ld_bits != b.ld_bits) {
            if (bad < 5)
                std::printf("FAIL argmax rank model: n %d d %d tile %d blocks %d: tok %d/%d rank %d/%d max %08x/%08x ld %08x/%08x\n", n, d, tile, blocks,
                            a.tok, b.tok, a.rank, b.rank, static_cast<unsigned>(a.max_bits), static_cast<unsigned>(b.max_bits),
                            static_cast<unsigned>(a.ld_bits), static_cast<unsigned>(b.ld_bits));
            ++bad;
        }
    }
    CHECK_EQ(bad, 0);
    CHECK(cases == 3000);
    // the launch geometry the kernel and every launcher share (kernels/argmax_key.h, REVIEW-v0252-f1arg C1): 61 tiles of
    // 4096 for the 248,320-entry vocabulary of both product models, a partial last tile counts, 4 float4 loads per thread
    CHECK_EQ(AM_TILES(248320), 61);
    CHECK_EQ(AM_TILES(4096), 1);
    CHECK_EQ(AM_TILES(4097), 2);
    CHECK_EQ(AM_TILES(1), 1);
    CHECK_EQ(AM_TILE % (4 * AM_THREADS), 0);
    CHECK_EQ(AM_TILE / (4 * AM_THREADS), 4);
    // a draft id at the row's end (REVIEW-v0252-f1arg B2): n is outside the row (no rank, l_d -inf), n - 1 is the last
    // element (a real rank); every grid gives the same
    {
        const std::vector<float> x = randomRow(9000, 4);
        for (const int blocks : {0, 1, 2, 5}) {
            const RankOut o = rankMb(x, 9000, AM_TILE, blocks), l = rankMb(x, 8999, AM_TILE, blocks), ll = rankLegacy(x, 8999);
            CHECK_EQ(o.rank, std::numeric_limits<int>::max());
            CHECK_EQ(o.ld_bits, bitsOf(-std::numeric_limits<float>::infinity()));
            CHECK(l.rank != std::numeric_limits<int>::max());
            CHECK(l.tok == ll.tok && l.rank == ll.rank && l.max_bits == ll.max_bits && l.ld_bits == ll.ld_bits);
        }
    }
    // fixed rows: a -0 maximum keeps its sign in the row max (the key does not: it is read back from x[index])
    {
        const std::vector<float> x = {-1.f, -0.f, 0.f, -0.f, -3.f};
        const RankOut a = rankLegacy(x, 2), b = rankMb(x, 2, 2);
        CHECK_EQ(a.tok, 1);
        CHECK_EQ(b.tok, 1);
        CHECK_EQ(b.max_bits, 0x80000000u);
        CHECK_EQ(b.rank, 1);  // -0 at id 1 is equal to the +0 draft and has a lower id
        CHECK(a.rank == b.rank && a.max_bits == b.max_bits && a.ld_bits == b.ld_bits);
    }
    {  // nothing above -inf: index 0, max -inf (even with a NaN at index 0), the draft's NaN logit has nothing ahead
        const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
        const std::vector<float> x = {nan, -inf, nan, -inf};
        const RankOut a = rankLegacy(x, 2), b = rankMb(x, 2, 1);
        CHECK_EQ(b.tok, 0);
        CHECK_EQ(b.max_bits, bitsOf(-inf));
        CHECK_EQ(b.rank, 0);
        CHECK(a.tok == b.tok && a.rank == b.rank && a.max_bits == b.max_bits && a.ld_bits == b.ld_bits);
    }
    // the workspace layout the kernels and the host share
    CHECK_EQ(sizeof(AmWs), static_cast<std::size_t>(AM_WS_BYTES));
    CHECK_EQ(offsetof(AmWs, cnt), static_cast<std::size_t>(256));
    CHECK_EQ(offsetof(AmWs, rank), static_cast<std::size_t>(384));
}

}  // namespace

// tests/unit/unit_tests.cpp ORDER lists all modules
const whirl::test::Suite& unitSuite_kernels_sample() {
    static const whirl::test::Suite s{"kernels.sample",
                                      {
                                          {"testArgmaxKeys", [] { testArgmaxKeys(); }},
                                          {"testArgmaxRankModel", [] { testArgmaxRankModel(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_kernels_sample()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
