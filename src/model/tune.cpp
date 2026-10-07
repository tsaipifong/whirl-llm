// Prefill GEMM autotune + tune cache, runtime switches, matmul microbenchmarks
// and the bitwise self-checks of the GEMV / GEMM / attention kernels.
// SPDX-License-Identifier: Apache-2.0
// Reimplements the WHIRL Zig research prototype's model/qwen35.zig (autotuneGemm, writeTune,
// readTune, matmulsOnly, checkGemvq, checkGemvBitwise, checkPrefillInvariance,
// checkAttnGroups) and gguf_cli.zig (loadOrTune).

#include "whirl/model.h"

#include "whirl/common.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <tuple>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }

double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string fmt(const char* f, ...) {
    char buf[1024];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof buf, f, ap);
    va_end(ap);
    return buf;
}

std::string tyName(GgmlType t) {
    std::string s = gguf::typeName(t);
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::vector<float> randomNormal(std::size_t n, unsigned seed, float scale = 1.0f) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng) * scale;
    return v;
}

// fast autotune: a candidate whose first timed launch takes this many times the best
// candidate's first timed launch is dropped (timings on the 8060S vary by a few percent)
constexpr double kTuneRejectFactor = 1.5;
// ...only at batch >= this: below it a launch is short enough that the single launch's
// sync overhead dominates, and early rejection dropped the faster small-batch GEMMs
// (Swift MXFP4 n=96: fused gemm_c picked over gemms, prefill 88 -6.9%)
constexpr u32 kTuneRejectMinN = 512;

}  // namespace

// The kernel(s) Model::matmul launches for tune choice cj of w at batch n (mirrors its
// dispatch): {0, cfg} fused gemm_c, {1, cfg} dequant + gemm_c f16, {2, slot} gemms,
// {3, 0} fused dequant gemmhq, {4, 0} dequant + gemmh_f16, {5, 0} gemmh_f16 on f16 weights.
static std::pair<int, int> tuneKernelKey(const Model& m, const Mat& w, u32 cj, u32 n) {
    constexpr u32 gemmh_min = 512;  // = forward.cpp
    const u32 nc = static_cast<u32>(gemm_cfgs.size());
    if (cj >= 2 * nc) return {2, static_cast<int>(cj - 2 * nc)};
    const bool deq = cj >= nc && w.ty != GgmlType::f16;
    const bool hok = m.gemmh_on && n >= gemmh_min && w.ncols % 32 == 0;
    if (deq && hok && m.gemmhq_on && m.k.gemmhq[ti(w.ty)] != nullptr) return {3, 0};
    if (hok && m.k.gemmh_f16 != nullptr && (deq || (w.ty == GgmlType::f16 && w.row_bytes == static_cast<u64>(w.ncols) * 2)))
        return {deq ? 4 : 5, 0};
    return {deq ? 1 : 0, static_cast<int>(cj % nc)};
}

// Pick the fastest prefill GEMM configuration for every distinct (type, rows,
// cols) at batch tune_sizes[bucket]; writes the choice into the layers' Mats.
void Model::autotuneGemm(std::size_t bucket, u32 reps, std::string* log) {
    const u32 n = tune_sizes[bucket];
    std::map<std::tuple<u32, u32, u32>, std::uint8_t> chosen;
    Profile* saved_prof = prof;
    prof = nullptr;
    const bool saved_fp8 = fp8_prefill;
    fp8_prefill = false;
    const bool saved_prec = prec_dec;  // (tune sizes are >= 32 > gvMax(); kept off regardless)
    prec_dec = false;
    double total_best = 0, total_default = 0;
    // Radeon 8060S: the candidate list is long (24 fused + 24 dequant + 8 small-batch GEMMs)
    // and the GPU is slow, so the first tune of a 27B model took ~8.5 min. There each kernel
    // is timed once per shape (choices that launch the same kernel are skipped) and a
    // candidate whose first timed launch is far behind the best so far is not timed further
    // (batch >= kTuneRejectMinN only).
    // Every choice computes the same bits, so this only affects which fast kernel is found.
    const bool fast = arch == hip::Arch::gfx1151 && !tune_cold;
    const bool early_reject = fast && n >= kTuneRejectMinN;  // else time like the full tune
    for (const Layer& L : layers) {
        const MatList all = layerMats(L);
        for (const Mat& w : all.slice()) {
            const auto key = std::make_tuple(static_cast<u32>(w.ty), w.nrows, w.ncols);
            if (chosen.count(key)) continue;
            std::uint8_t best_ci = 0;
            double best_ms = std::numeric_limits<double>::infinity();
            double best_one = std::numeric_limits<double>::infinity();  // best's first timed launch (early_reject)
            double ms_default = 0;
            const DevPtr xin = w.ncols == cfg.n_embd ? h : ffn_g;
            std::set<std::pair<int, int>> timed;  // kernels already timed for this shape (fast mode)
            for (u32 cj = 0; cj < n_choices; ++cj) {
                if (cj >= 64 || (tune_mask & (1ull << cj)) == 0) continue;
                if (cj >= 2 * gemm_cfgs.size() && (k.gemms[cj - 2 * gemm_cfgs.size()][ti(w.ty)] == nullptr || w.ncols % 256 != 0)) continue;
                if (fast && !timed.insert(tuneKernelKey(*this, w, cj, n)).second) continue;
                Mat probe = w;
                probe.tune.fill(static_cast<std::uint8_t>(cj));
                matmul(probe, xin, ffn_u, n, false);  // warm-up
                double ms_sum = 0, one = 0;
                if (early_reject) {
                    // one timed launch first; a candidate already far behind the best stops there
                    hip::sync();
                    const double t0 = nowMs();
                    matmul(probe, xin, ffn_u, n, false);
                    hip::sync();
                    one = nowMs() - t0;
                    if (one > kTuneRejectFactor * best_one) continue;  // like with like: single launch vs single launch
                    const double t1 = nowMs();
                    for (u32 r = 1; r < reps; ++r) matmul(probe, xin, ffn_u, n, false);
                    hip::sync();
                    ms_sum = one + (nowMs() - t1);
                } else if (!tune_cold) {
                    // back-to-back launches in one timed window, like real inference
                    hip::sync();
                    const double t0 = nowMs();
                    for (u32 r = 0; r < reps; ++r) matmul(probe, xin, ffn_u, n, false);
                    hip::sync();
                    ms_sum = nowMs() - t0;
                } else {
                    for (u32 r = 0; r < reps; ++r) {
                        hip::memset(gc_w, 0, 96u * 1024 * 1024);  // evict the Infinity Cache (gc_w is idle scratch)
                        hip::sync();
                        const double t0 = nowMs();
                        matmul(probe, xin, ffn_u, n, false);
                        hip::sync();
                        ms_sum += nowMs() - t0;
                    }
                }
                const double ms = ms_sum / reps;
                if (cj == 0) ms_default = ms;
                if (ms < best_ms) {
                    best_ms = ms;
                    best_one = one;
                    best_ci = static_cast<std::uint8_t>(cj);
                }
            }
            chosen[key] = best_ci;
            total_best += best_ms;
            total_default += ms_default;
            if (log) {
                const bool small = best_ci >= 2 * gemm_cfgs.size();
                *log += fmt("    %-7s %6ux%-6u -> cfg %u%s (%.3f ms, cfg0 %.3f ms)\n", tyName(w.ty).c_str(), w.nrows, w.ncols,
                            small ? best_ci - static_cast<u32>(2 * gemm_cfgs.size()) : best_ci % static_cast<u32>(gemm_cfgs.size()),
                            small ? " small" : best_ci >= gemm_cfgs.size() ? " +deq" : "", best_ms, ms_default);
            }
        }
    }
    for (Layer& L : layers)
        for (Mat* mp : tuneMats(L)) mp->tune[bucket] = chosen.at(std::make_tuple(static_cast<u32>(mp->ty), mp->nrows, mp->ncols));
    prof = saved_prof;
    fp8_prefill = saved_fp8;
    prec_dec = saved_prec;
    if (log) *log += fmt("    distinct shapes %zu: sum of best %.1f ms vs cfg0 %.1f ms\n", chosen.size(), total_best, total_default);
}

// Tune table as text: one line per distinct (bucket, type, rows, cols).
std::string Model::writeTune() const {
    std::string out;
    std::map<std::tuple<u32, u32, u32, u32>, bool> seen;
    for (const Layer& L : layers) {
        Layer& LL = const_cast<Layer&>(L);
        for (Mat* m : const_cast<Model*>(this)->tuneMats(LL))
            for (u32 b = 0; b < n_tune; ++b) {
                const auto key = std::make_tuple(b, static_cast<u32>(m->ty), m->nrows, m->ncols);
                if (seen.count(key)) continue;
                seen[key] = true;
                out += fmt("%u %u %u %u %u\n", b, static_cast<u32>(m->ty), m->nrows, m->ncols, static_cast<u32>(m->tune[b]));
            }
    }
    return out;
}

// Apply buckets [0, n_buckets) of a table written by writeTune. Returns false if
// any matrix shape is missing (stale cache).
bool Model::readTune(std::string_view text, std::size_t n_buckets) {
    std::map<std::tuple<u32, u32, u32, u32>, std::uint8_t> map;
    std::size_t pos = 0;
    while (pos < text.size()) {
        std::size_t e = text.find('\n', pos);
        if (e == std::string_view::npos) e = text.size();
        std::string line(text.substr(pos, e - pos));
        pos = e + 1;
        for (char& c : line)
            if (c == '\r') c = ' ';
        std::istringstream is(line);
        u64 v[5];
        int got = 0;
        for (; got < 5 && (is >> v[got]); ++got) {
        }
        if (got == 0) continue;
        if (got != 5) return false;
        if (v[0] >= n_buckets) continue;
        if (v[4] >= n_choices) return false;
        map[std::make_tuple(static_cast<u32>(v[0]), static_cast<u32>(v[1]), static_cast<u32>(v[2]), static_cast<u32>(v[3]))] =
            static_cast<std::uint8_t>(v[4]);
    }
    for (Layer& L : layers)
        for (Mat* m : tuneMats(L))
            for (u32 b = 0; b < n_buckets; ++b) {
                auto it = map.find(std::make_tuple(b, static_cast<u32>(m->ty), m->nrows, m->ncols));
                if (it == map.end()) return false;
                m->tune[b] = it->second;
            }
    return true;
}

void Model::benchMat(const Mat& w, u32 n) {
    const DevPtr xin = w.ncols == cfg.n_embd ? h : ffn_g;
    matmul(w, xin, ffn_u, n, false);
}

std::uint64_t Model::matmulsOnly(u32 n) {
    u64 bytes = 0;
    for (const Layer& L : layers)
        for (const Mat& w : layerMats(L).slice()) {
            benchMat(w, n);
            bytes += w.row_bytes * w.nrows;
        }
    return bytes;
}

std::uint64_t Model::matmulsOfType(GgmlType ty, u32 n) {
    u64 bytes = 0;
    for (const Layer& L : layers)
        for (const Mat& w : layerMats(L).slice()) {
            if (w.ty != ty) continue;
            benchMat(w, n);
            bytes += w.row_bytes * w.nrows;
        }
    return bytes;
}

// Per quant type: int8-activation GEMV vs f32 GEMV on one real matrix.
bool Model::checkGemvq(std::string& log) {
    struct PrecOff {
        bool& p;
        bool s;
        ~PrecOff() { p = s; }
    } prec_off{prec_dec, prec_dec};
    prec_dec = false;  // the int8 kernels (balance / fast), whatever the mode
    const std::vector<float> xs = randomNormal(ff_scratch, 42);
    bool done[n_types] = {};
    bool ok_all = true;
    for (const Layer& L : layers)
        for (const Mat& w : layerMats(L).slice()) {
            const std::size_t t = ti(w.ty) % n_types;
            if (done[t] || k.gemvq[t] == nullptr) continue;
            done[t] = true;
            hip::upload(ffn_g, xs.data(), w.ncols * 4ull);
            std::vector<float> a(w.nrows), b(w.nrows);
            float_gemv = true;
            xq_src = 0;
            x16_src = 0;
            matmul(w, ffn_g, ffn_u, 1, false);
            hip::download(a.data(), ffn_u, a.size() * 4);
            float_gemv = false;
            xq_src = 0;
            x16_src = 0;
            matmul(w, ffn_g, ffn_u, 1, false);
            hip::download(b.data(), ffn_u, b.size() * 4);
            double num = 0, den = 0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                num += (static_cast<double>(a[i]) - b[i]) * (static_cast<double>(a[i]) - b[i]);
                den += static_cast<double>(a[i]) * a[i];
            }
            const double rel = std::sqrt(num / den);
            if (!(rel < 0.03)) ok_all = false;
            log += fmt("    %-7s %ux%u: rel err %.4f %s\n", tyName(w.ty).c_str(), w.nrows, w.ncols, rel, rel < 0.03 ? "ok" : "FAIL");
        }
    return ok_all;
}

// Multi-row and multi-token GEMV must reproduce the one-token GEMV bitwise
// (MTP greedy == plain greedy): every type, token counts 2..16, R = 1, 2, 4 and
// the WMMA variants, compared per token row with the n = 1 result.
bool Model::checkGemvBitwise(std::string& log) {
    struct PrecOff {
        bool& p;
        bool s;
        ~PrecOff() { p = s; }
    } prec_off{prec_dec, prec_dec};
    prec_dec = false;  // the int8 kernels (balance / fast), whatever the mode
    const std::size_t cols_max = ff_scratch;
    const std::vector<float> xs = randomNormal(max_small_batch * cols_max, 7);
    bool done[n_types] = {};
    bool all_ok = true;
    const GemvR saved = gemv_r;
    const GemvW saved_w = gemv_w;
    for (std::size_t li = 0; li <= layers.size(); ++li) {
        MatList all;
        if (li < layers.size())
            all = layerMats(layers[li]);
        else
            all.add(output);
        for (const Mat& w : all.slice()) {
            const std::size_t t = ti(w.ty) % n_types;
            // f32 / f16 have no int8 GEMV: n = 1 runs gemv_<t>_1 (f32 activations), n = 2..16 the
            // f16 GEMM. They are checked too (C-13): the check FAILS on such files until the
            // multi-token path is made bitwise (known bug; WHIRL_GEMV_BITWISE_FLOAT=warn reports
            // without failing).
            const bool flt = w.ty == GgmlType::f32 || w.ty == GgmlType::f16;
            if (li < layers.size() && done[t]) continue;
            if (k.gemvq[t] == nullptr && !flt) continue;
            done[t] = true;
            const std::size_t nr = w.nrows;
            std::vector<float> ref(max_small_batch * nr), got(max_small_batch * nr);
            for (u32 tk = 0; tk < max_small_batch; ++tk) {
                hip::upload(ffn_g, xs.data() + tk * w.ncols, w.ncols * 4ull);
                xq_src = 0;
                matmul(w, ffn_g, ffn_u, 1, false);
                hip::download(ref.data() + tk * nr, ffn_u, nr * 4);
            }
            hip::upload(ffn_g, xs.data(), static_cast<std::size_t>(max_small_batch) * w.ncols * 4);
            u32 bad = 0;
            for (u32 nt = 2; nt <= max_small_batch; ++nt) {
                for (int r : {1, 2, 4, 11, 12, 13, 14, 15, 16, 17, 18}) {
                    if (r < 10) {
                        if (r > 1 && k.gemvq_mr[r / 2 - 1][nt - 2][t] == nullptr) continue;
                        gemv_r[nt] = static_cast<std::uint8_t>(r);
                        gemv_w[t][nt] = 0;
                    } else {
                        if (k.gemvw[r - 10][nt - 2][t] == nullptr) continue;
                        gemv_w[t][nt] = static_cast<std::uint8_t>(r - 10);
                    }
                    xq_src = 0;
                    matmul(w, ffn_g, ffn_u, nt, false);
                    hip::download(got.data(), ffn_u, nt * nr * 4);
                    if (std::memcmp(got.data(), ref.data(), nt * nr * 4) != 0) {
                        bad += 1;
                        if (bad <= 3) {
                            std::size_t nd = 0;
                            float maxd = 0;
                            for (std::size_t i = 0; i < nt * nr; ++i) {
                                if (got[i] != ref[i]) nd += 1;
                                maxd = std::max(maxd, std::fabs(got[i] - ref[i]));
                            }
                            log += fmt("    %-7s nt=%u %s=%d: MISMATCH (%zu of %zu differ, max |d| %.3e)\n", tyName(w.ty).c_str(), nt, r < 10 ? "R" : "W",
                                       r < 10 ? r : r - 10, nd, static_cast<std::size_t>(nt) * nr, maxd);
                        }
                    }
                }
                gemv_r = saved;
                gemv_w = saved_w;
            }
            const char* fw = std::getenv("WHIRL_GEMV_BITWISE_FLOAT");
            const bool warn_only = flt && fw != nullptr && std::strcmp(fw, "warn") == 0;
            if (bad > 0 && !warn_only) all_ok = false;
            log += fmt("    %-7s %ux%u: multi-token / multi-row GEMV bitwise == 1-token: %s\n", tyName(w.ty).c_str(), w.nrows, w.ncols,
                       bad == 0 ? "ok" : (warn_only ? "FAIL (known bug C-13, warn only)" : "FAIL"));
        }
    }
    gemv_r = saved;
    gemv_w = saved_w;
    xq_src = 0;
    return all_ok;
}

// Prefill-batch invariance: a row's GEMM result must not depend on the GEMM
// config, and a token's MoE output not on the expert token tile or the other
// rows. Random inputs, bitwise.
bool Model::checkPrefillInvariance(std::string& log) {
    const u32 n = 200;
    const std::vector<float> xs = randomNormal(static_cast<std::size_t>(n) * ff_scratch, 7);
    bool done[n_types] = {};
    bool all_ok = true;
    for (const Layer& L : layers)
        for (const Mat& w0 : layerMats(L).slice()) {
            const std::size_t t = ti(w0.ty) % n_types;
            if (done[t] || k.gemmc[0][t] == nullptr) continue;
            done[t] = true;
            Mat w = w0;
            hip::upload(ffn_g, xs.data(), static_cast<std::size_t>(n) * w.ncols * 4);
            std::vector<float> ref(static_cast<std::size_t>(n) * w.nrows), got(ref.size());
            u32 bad_cfg = 0;
            int first_bad = -1;
            for (u32 c = 0; c < n_choices; ++c) {
                if (c >= gemm_cfgs.size() && c < 2 * gemm_cfgs.size() && w.ty == GgmlType::f16) continue;
                if (c >= 2 * gemm_cfgs.size()) {
                    if (k.gemms[c - 2 * gemm_cfgs.size()][t] == nullptr || w.ncols % 256 != 0) continue;
                } else if (k.gemmc[c % gemm_cfgs.size()][t] == nullptr) {
                    continue;
                }
                w.tune.fill(static_cast<std::uint8_t>(c));
                xq_src = 0;
                x16_src = 0;
                matmul(w, ffn_g, ffn_u, n, false);
                hip::download(c == 0 ? ref.data() : got.data(), ffn_u, ref.size() * 4);
                if (c == 0) continue;
                if (std::memcmp(ref.data(), got.data(), ref.size() * 4) != 0) {
                    bad_cfg += 1;
                    if (first_bad < 0) first_bad = static_cast<int>(c);
                }
            }
            w.tune.fill(0);
            xq_src = 0;
            x16_src = 0;
            matmul(w, ffn_g, ffn_u, 40, false);
            hip::download(got.data(), ffn_u, 40ull * w.nrows * 4);
            const bool sub_ok = std::memcmp(ref.data(), got.data(), 40ull * w.nrows * 4) == 0;
            if (!(bad_cfg == 0 && sub_ok)) all_ok = false;
            log += fmt("    GEMM %-7s %ux%u, %u rows: configs differing from cfg 0: %u (first %d), 40-row subset %s %s\n", tyName(w.ty).c_str(), w.nrows,
                       w.ncols, n, bad_cfg, first_bad, sub_ok ? "same" : "DIFFERENT", bad_cfg == 0 && sub_ok ? "ok" : "FAIL");
        }
    // MoE: tile 32 vs 64, and the first 40 tokens alone
    for (const Layer& L : layers) {
        if (!L.moe) continue;
        const u32 E = cfg.n_embd;
        std::vector<float> a(static_cast<std::size_t>(n) * E), b(a.size());
        const u32 saved = moe_bn_force;
        const u32 bns[3] = {32, 64, 32};
        for (int pass = 0; pass < 3; ++pass) {
            const u32 rows = pass == 2 ? 40 : n;
            moe_bn_force = bns[pass];
            hip::upload(x, xs.data(), static_cast<std::size_t>(rows) * E * 4);
            moeBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, *L.moe, rows);
            hip::download((pass == 0 ? a : b).data(), x, static_cast<std::size_t>(rows) * E * 4);
            if (pass == 0) continue;
            const bool same = std::memcmp(a.data(), b.data(), static_cast<std::size_t>(rows) * E * 4) == 0;
            if (!same) all_ok = false;
            log += fmt("    MoE tile 32 vs %s: %s\n", pass == 1 ? "64 (200 rows)" : "32 (first 40 rows alone)", same ? "same ok" : "DIFFERENT");
        }
        moe_bn_force = saved;
        break;
    }
    return all_ok;
}

// Precise decode (P-8): per weight type, every row of a small batch (n = 2..16, and 17 / 32
// as in a wide verify) bitwise == the same token alone (n = 1), and (not the head) == the
// same row inside a 40-row prefill GEMM; MoE: a token's block output alone == inside a
// batch of 16 == inside the 40-row prefill path. Random inputs.
bool Model::checkPreciseDecode(std::string& log) {
    const bool saved_prec = prec_dec;
    const u32 saved_small = small_max;
    prec_dec = true;
    const u32 NP = 40;  // > every gvMax(): the prefill GEMM
    const std::vector<float> xs = randomNormal(static_cast<std::size_t>(NP) * ff_scratch, 13);
    bool done[n_types] = {};
    bool all_ok = true;
    const std::string miss = precMissing();
    if (!miss.empty()) {
        log += "    precise decode: missing " + miss + " - FAIL\n";
        prec_dec = saved_prec;
        return false;
    }
    for (std::size_t li = 0; li <= layers.size(); ++li) {
        MatList all;
        if (li < layers.size())
            all = layerMats(layers[li]);
        else
            all.add(output);
        for (const Mat& w : all.slice()) {
            const bool head = w.ptr == output.ptr;
            const std::size_t t = ti(w.ty) % n_types;
            if (!head && done[t]) continue;
            if (!head) done[t] = true;
            const std::size_t nr = w.nrows;
            const DevPtr yb = head ? logits : ffn_u;
            std::vector<float> solo(32 * nr), got(32 * nr), pre;
            hip::upload(ffn_g, xs.data(), static_cast<std::size_t>(NP) * w.ncols * 4);
            small_max = max_small_batch;
            for (u32 tk = 0; tk < 32; ++tk) {
                x16_src = 0;
                xq_src = 0;
                matmul(w, ffn_g + static_cast<DevPtr>(tk) * w.ncols * 4, yb, 1, false);
                hip::download(solo.data() + tk * nr, yb, nr * 4);
            }
            u32 bad = 0;
            std::string where;
            for (u32 nt : {2u, 3u, 4u, 8u, 13u, 16u, 17u, 32u}) {
                small_max = nt > max_small_batch ? max_verify_rows : max_small_batch;
                x16_src = 0;
                xq_src = 0;
                matmul(w, ffn_g, yb, nt, false);
                hip::download(got.data(), yb, nt * nr * 4);
                if (std::memcmp(got.data(), solo.data(), nt * nr * 4) != 0) {
                    bad += 1;
                    where += fmt(" n=%u", nt);
                }
            }
            small_max = max_small_batch;
            bool pre_ok = true;
            if (!head) {
                pre.resize(static_cast<std::size_t>(NP) * nr);
                x16_src = 0;
                matmul(w, ffn_g, yb, NP, false);
                hip::download(pre.data(), yb, pre.size() * 4);
                pre_ok = std::memcmp(pre.data(), solo.data(), 32 * nr * 4) == 0;
            }
            // accuracy vs a double-precision CPU product of the same f16 activations is in the
            // kernel test; here: the solo row vs the f32-activation reference GEMV (sanity)
            double num = 0, den = 0;
            {
                std::vector<float> ref(nr);
                const bool fg = float_gemv, pd = prec_dec;
                float_gemv = true;
                prec_dec = false;
                x16_src = 0;
                xq_src = 0;
                if (k.gemv1[t] != nullptr) {
                    matmul(w, ffn_g, yb, 1, false);
                    hip::download(ref.data(), yb, nr * 4);
                    for (std::size_t i = 0; i < nr; ++i) {
                        const double d = static_cast<double>(ref[i]) - solo[i];
                        num += d * d;
                        den += static_cast<double>(ref[i]) * ref[i];
                    }
                }
                float_gemv = fg;
                prec_dec = pd;
            }
            const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
            const bool ok = bad == 0 && pre_ok && rel < 0.01;
            if (!ok) all_ok = false;
            log += fmt("    precise %-7s %ux%u%s: batch rows == solo: %s%s; == prefill GEMM rows: %s; rel err vs f32 GEMV %.2e %s\n",
                       tyName(w.ty).c_str(), w.nrows, w.ncols, head ? " (head)" : "", bad == 0 ? "yes" : "NO", where.c_str(),
                       head ? "n/a" : (pre_ok ? "yes" : "NO"), rel, ok ? "ok" : "FAIL");
        }
    }
    for (const Layer& L : layers) {
        if (!L.moe) continue;
        const u32 E = cfg.n_embd;
        const std::vector<float> xm = randomNormal(static_cast<std::size_t>(NP) * E, 17);
        std::vector<float> solo(32ull * E), got(static_cast<std::size_t>(NP) * E);
        for (u32 tk = 0; tk < 32; ++tk) {
            hip::upload(x, xm.data() + static_cast<std::size_t>(tk) * E, E * 4ull);
            moeBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, *L.moe, 1);
            hip::download(solo.data() + static_cast<std::size_t>(tk) * E, x, E * 4ull);
        }
        std::string res;
        bool ok = true;
        for (u32 rows : {2u, 3u, 4u, 8u, 16u, 17u, 24u, 32u}) {  // 17..32: wide verify (n-gram drafts)
            hip::upload(x, xm.data(), static_cast<std::size_t>(rows) * E * 4);
            moeBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, *L.moe, rows);
            hip::download(got.data(), x, static_cast<std::size_t>(rows) * E * 4);
            const bool same = std::memcmp(got.data(), solo.data(), static_cast<std::size_t>(rows) * E * 4) == 0;
            if (!same) ok = false;
            res += fmt(" n=%u %s", rows, same ? "same" : "DIFFERENT");
        }
        {
            // sanity: the decode experts vs the prefill (f16 GEMM) path on the same tokens
            hip::upload(x, xm.data(), static_cast<std::size_t>(NP) * E * 4);
            moeBlock(L.post_norm, L.ffn_gate, L.ffn_up, L.ffn_down, *L.moe, NP);
            hip::download(got.data(), x, static_cast<std::size_t>(NP) * E * 4);
            double num = 0, den = 0;
            for (std::size_t i = 0; i < 16ull * E; ++i) {
                const double dx = static_cast<double>(got[i]) - xm[i], ds = static_cast<double>(solo[i]) - xm[i];
                num += (dx - ds) * (dx - ds);
                den += ds * ds;
            }
            const double rel = den > 0 ? std::sqrt(num / den) : 0.0;
            if (!(rel < 0.02)) ok = false;
            res += fmt("; MoE output vs prefill path rel diff %.2e", rel);
        }
        if (!ok) all_ok = false;
        log += fmt("    precise MoE block, a token alone vs in a batch:%s %s\n", res.c_str(), ok ? "ok" : "FAIL");
        break;
    }
    small_max = saved_small;
    prec_dec = saved_prec;
    xq_src = 0;
    x16_src = 0;
    return all_ok;
}

// attn_wsplit consistency: the grouped (verify) launch must give every query
// the same partials, bit for bit, as its own one-query launch.
bool Model::checkAttnGroups(std::string& log, u32 p0, u32 n, bool wide) {
    const u32 hd = cfg.head_dim;
    if (k.attn_wsplit1 == nullptr || hd != 256) return true;
    if (wide && k.attn_wsplit2 == nullptr) return true;
    if (kv_kf16 || kv_q4) return true;  // probe: q8 or f16 only
    std::size_t li_attn = 0;
    for (std::size_t i = 0; i < layers.size(); ++i)
        if (layers[i].kind == LayerKind::attn) {
            li_attn = i;
            break;
        }
    std::mt19937_64 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const std::size_t qn = static_cast<std::size_t>(n) * cfg.n_head * 2 * hd;
    std::vector<float> qh(qn);
    for (float& v : qh) v = nd(rng);
    hip::upload(qf, qh.data(), qn * 4);
    const std::size_t kvn = static_cast<std::size_t>(p0 + n) * cfg.n_head_kv * hd;
    auto f16bits = [](float f) {
        // round-to-nearest-even float -> half (finite, normal range inputs)
        std::uint32_t u;
        std::memcpy(&u, &f, 4);
        const std::uint32_t sign = (u >> 16) & 0x8000;
        int e = static_cast<int>((u >> 23) & 0xff) - 127 + 15;
        std::uint32_t mant = u & 0x7fffff;
        if (e <= 0) return static_cast<std::uint16_t>(sign);
        if (e >= 31) return static_cast<std::uint16_t>(sign | 0x7c00);
        std::uint32_t hm = mant >> 13;
        const std::uint32_t rem = mant & 0x1fff;
        if (rem > 0x1000 || (rem == 0x1000 && (hm & 1))) {
            hm += 1;
            if (hm == 0x400) {
                hm = 0;
                e += 1;
            }
        }
        return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e) << 10) | hm);
    };
    if (kv_q8) {
        std::uniform_int_distribution<int> qd(-127, 127);
        std::uniform_real_distribution<float> sd(0.0f, 1.0f);
        std::vector<std::int8_t> q8(kvn);
        std::vector<std::uint16_t> sh(kvn / 32);
        const DevPtr dsts[2] = {kcache[li_attn], vcache[li_attn]};
        const DevPtr sdst[2] = {kscale[li_attn], vscale[li_attn]};
        for (int j = 0; j < 2; ++j) {
            for (auto& v : q8) v = static_cast<std::int8_t>(qd(rng));
            for (auto& v : sh) v = f16bits(0.02f + 0.01f * sd(rng));
            hip::upload(dsts[j], q8.data(), q8.size());
            hip::upload(sdst[j], sh.data(), sh.size() * 2);
        }
    } else {
        std::vector<std::uint16_t> kvh(kvn);
        for (auto& v : kvh) v = f16bits(nd(rng) * 0.5f);
        hip::upload(kcache[li_attn], kvh.data(), kvn * 2);
        for (auto& v : kvh) v = f16bits(nd(rng) * 0.5f);
        hip::upload(vcache[li_attn], kvh.data(), kvn * 2);
    }
    std::vector<i32> posv(n);
    for (u32 r = 0; r < n; ++r) posv[r] = static_cast<i32>(p0 + r);
    hip::upload(pos_buf, posv.data(), n * 4);
    const u32 n_split = std::min<u32>(64, (max_ctx + 63) / 64);
    const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    const std::size_t ml_n = static_cast<std::size_t>(n) * n_split * cfg.n_head * 2;
    const std::size_t acc_n = static_cast<std::size_t>(n) * n_split * cfg.n_head * hd;
    std::vector<float> ml_a(ml_n), acc_a(acc_n), ml_b(ml_n), acc_b(acc_n);
    const KvArgs kva = kvArgs(kvLayer(li_attn), 0, 0);
    for (int mode = 0; mode < 2; ++mode) {
        hip::memset(part_ml, 0, ml_n * 4);
        hip::memset(part_acc, 0, acc_n * 4);
        AwGroups groups;
        u32 ng = 0;
        if (mode == 0) {
            groups.first[0] = 0;
            groups.count[0] = static_cast<i32>(n);
            ng = 1;
        } else {
            for (u32 r = 0; r < n; ++r) {
                groups.first[r] = static_cast<i32>(r);
                groups.count[r] = 1;
            }
            ng = n;
        }
        hip::launch(mode == 0 && wide ? k.attn_wsplit2 : k.attn_wsplit1, hip::Dim3{cfg.n_head_kv, n_split, ng}, hip::Dim3{128}, 0, stream, qf,
                    kva, part_ml, part_acc,
                    static_cast<i32>(cfg.n_head), static_cast<i32>(cfg.n_head_kv), static_cast<i32>(2 * hd), pos_buf, scale, u64(0), groups);
        hip::sync();
        hip::download((mode == 0 ? ml_a : ml_b).data(), part_ml, ml_n * 4);
        hip::download((mode == 0 ? acc_a : acc_b).data(), part_acc, acc_n * 4);
    }
    std::size_t bad_ml = 0, bad_acc = 0;
    float maxd = 0;
    const std::size_t per_q_ml = static_cast<std::size_t>(n_split) * cfg.n_head * 2;
    for (u32 t = 0; t < n; ++t)
        for (u32 sp = 0; sp < n_split; ++sp)
            for (u32 hh = 0; hh < cfg.n_head; ++hh) {
                const std::size_t i = t * per_q_ml + (static_cast<std::size_t>(sp) * cfg.n_head + hh) * 2;
                if (std::isinf(ml_a[i]) && ml_a[i] < 0 && std::isinf(ml_b[i]) && ml_b[i] < 0) continue;
                if (std::memcmp(&ml_a[i], &ml_b[i], 8) != 0) bad_ml += 1;
                const std::size_t ai = (static_cast<std::size_t>(t) * n_split * cfg.n_head + static_cast<std::size_t>(sp) * cfg.n_head + hh) * hd;
                for (u32 d = 0; d < hd; ++d)
                    if (std::memcmp(&acc_a[ai + d], &acc_b[ai + d], 4) != 0) {
                        bad_acc += 1;
                        maxd = std::max(maxd, std::fabs(acc_a[ai + d] - acc_b[ai + d]));
                    }
            }
    const bool ok = bad_ml == 0 && bad_acc == 0;
    log += fmt("  attn_wsplit%s grouped vs per-query, %u queries at pos %u: ml mismatches %zu, acc mismatches %zu (max |d| %.3e) %s\n",
               wide ? "2" : "1", n, p0, bad_ml, bad_acc, maxd, ok ? "ok" : "FAIL");
    return ok;
}

// ---------------------------------------------------------------------------
// tune cache + runtime switches

namespace {

std::string localAppData() {
    char* buf = nullptr;
    std::size_t len = 0;
    if (_dupenv_s(&buf, &len, "LOCALAPPDATA") == 0 && buf != nullptr) {
        std::string v(buf);
        std::free(buf);
        return v;
    }
    return {};
}

bool tryRead(const std::string& path, std::string& out) {
    try {
        std::error_code ec;
        if (!std::filesystem::exists(widen(path), ec)) return false;
        out = readFile(path);
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace

void loadOrTune(Model& m, const std::string& model_path, std::string& log) {
    namespace nu = whirl::numerics;
    const bool has_mx = m.type_bytes[ti(GgmlType::mxfp4)] > 0;
    const nu::Request& rq = m.num_req;
    // MXFP4 prefill with fp8 activations: a balance item (m.fp8_prefill from the mode at load);
    // WHIRL_FP8=0|1 overrides. WHIRL_FP8=0 also turns the MoE expert fp8 path off (as before).
    const auto fp8_env = envGet("FP8");
    if (fp8_env) m.fp8_prefill = *fp8_env != "0";
    if (fp8_env && *fp8_env == "0") m.moe_fp8 = false;
    if (auto v = envGet("FP8_MASK")) {
        try {
            m.fp8_mask = static_cast<std::uint8_t>(std::stoul(*v, nullptr, 0));
        } catch (...) {
            m.fp8_mask = 7;
        }
    }
    if (has_mx) {
        if (m.fp8_prefill && m.k.gemm8[0] != nullptr)
            log += fmt("  MXFP4 prefill: fp8 activations (%llu folded blocks rounded)\n", static_cast<unsigned long long>(m.mx_fold_lossy));
        else
            log += "  MXFP4 prefill: f16 activations\n";
        // fragment-tiled fp8 GEMM (bitwise == gemm8), default on; WHIRL_G8T=0 keeps gemm8
        const bool g8t_on = envFlag("G8T", true);
        if (g8t_on && m.fp8_prefill && m.k.gemm8[0] != nullptr && m.useTiledFp8()) log += "  fp8 prefill GEMM: fragment-tiled (gemm8t)\n";
    }
    // f16-WMMA chunked DeltaNet prefill: a balance item, MXFP4 models (WHIRL_GDN_WMMA overrides)
    m.gdn_wmma = has_mx && rq.has(nu::Item::gdnwmma);
    const auto wmma_env = envGet("GDN_WMMA");
    if (wmma_env) m.gdn_wmma = *wmma_env != "0";
    if (auto v = envGet("ACT_FUSE")) m.act_fuse = *v != "0";
    if (auto v = envGet("GEMMH")) m.gemmh_on = *v != "0";
    if (auto v = envGet("GEMMHQ")) m.gemmhq_on = *v != "0";
    if (auto v = envGet("ATTN_KX")) m.attn_kx_on = *v != "0";
    if (auto v = envGet("ATTN_KG")) m.attn_kg_on = *v != "0";
    if (auto v = envGet("ATTN_DQF")) m.attn_dqf_on = *v != "0";
    if (auto v = envGet("GDN_BA")) m.gdn_ba_on = *v != "0";
    if (auto v = envGet("GDN_IN2")) m.gdn_in2_on = *v != "0";
    // f16 FFN / DeltaNet GEMM outputs: a balance item, MXFP4 models (WHIRL_FFN_H16 overrides)
    m.ffn_h16 = has_mx && rq.has(nu::Item::h16);
    // WHIRL_Q4_RELAXED=1: every non-bitwise prefill speedup for non-MXFP4 models too
    const auto relaxed_env = envGet("Q4_RELAXED");
    const bool relaxed = relaxed_env && *relaxed_env != "0";
    if (relaxed) {
        m.gdn_wmma = true;
        m.ffn_h16 = true;
    }
    const auto h16_env = envGet("FFN_H16");
    if (h16_env) m.ffn_h16 = *h16_env != "0";
    if (m.gdn_wmma && m.gdn_chunked && m.k.gdn_wprep != nullptr) log += "  DeltaNet prefill: f16 WMMA chunks\n";
    {
        // numerics mode: capability table, per-item environment overrides, one log line
        nu::Target t;
        t.gfx1151 = m.arch == hip::Arch::gfx1151;
        t.has_mxfp4 = has_mx;
        t.moe = m.cfg.moe;
        for (const Layer& L : m.layers)
            if (L.moe) {
                t.moe_mxfp4 = L.moe->gate.ty == GgmlType::mxfp4 && L.moe->up.ty == GgmlType::mxfp4 && L.moe->down.ty == GgmlType::mxfp4;
                break;
            }
        t.k_gemm8 = m.k.gemm8[0] != nullptr;
        t.k_gemm8_moe = m.k.gemm8_moe != nullptr && m.k.gemm8_moe32 != nullptr && m.k.gemm8_moeh != nullptr && m.k.gemm8_moe32h != nullptr &&
                        m.k.moe_gather_fp8 != nullptr;
        t.k_gdn_wmma = m.k.gdn_wprep != nullptr;
        t.k_kv_q8v = m.k.caps.kv_q8v;
        t.k_kv_q8h = m.k.caps.kv_q8h;
        t.mtp = m.mtp.has_value();
        t.k_spec_sample = m.k.draft_sample_rows != nullptr;
        t.k_kv_q4 = m.k.caps.kv_q4;
        t.kv_explicit = m.kv_mode != KvMode::automatic;
        nu::Plan p = nu::plan(rq, t);
        if (fp8_env) nu::applyOverride(p, nu::Item::fp8, "FP8", *fp8_env, m.fp8_prefill, t.has_mxfp4 && t.k_gemm8);
        const bool moe_ok = t.moe && t.moe_mxfp4 && t.k_gemm8_moe;
        if (const auto v = envGet("MOE_FP8"); v && !(fp8_env && *fp8_env == "0"))
            nu::applyOverride(p, nu::Item::moefp8, "MOE_FP8", *v, m.moe_fp8, moe_ok);
        else if (fp8_env && *fp8_env == "0")
            nu::applyOverride(p, nu::Item::moefp8, "FP8", *fp8_env, m.moe_fp8, moe_ok);
        if (relaxed) nu::applyOverride(p, nu::Item::gdnwmma, "Q4_RELAXED", *relaxed_env, m.gdn_wmma, t.k_gdn_wmma && m.gdn_chunked);
        else if (wmma_env) nu::applyOverride(p, nu::Item::gdnwmma, "GDN_WMMA", *wmma_env, m.gdn_wmma, t.k_gdn_wmma && m.gdn_chunked);
        if (h16_env) nu::applyOverride(p, nu::Item::h16, "FFN_H16", *h16_env, m.ffn_h16, true);
        else if (relaxed) nu::applyOverride(p, nu::Item::h16, "Q4_RELAXED", *relaxed_env, m.ffn_h16, true);
        if (const auto v = envGet("KV"); v && !v->empty() && *v != "auto")
            nu::applyOverride(p, *v == "q4" ? nu::Item::kvq4 : nu::Item::kvq8, "KV", *v, *v != "f16", !t.moe || *v != "f16");
        // decode / verify activations (P-8): precise -> f16 activations on the f16 GEMM
        // (f32 accumulation), balance / fast -> int8 (q8_1 class). WHIRL_Q8DEC=0|1 overrides.
        m.prec_dec = rq.mode == nu::Mode::precise;
        const auto q8dec_env = envGet("Q8DEC");
        if (q8dec_env && !q8dec_env->empty()) m.prec_dec = *q8dec_env == "0";
        nu::setQ8dec(p, !m.prec_dec, q8dec_env ? *q8dec_env : std::string());
        if (m.prec_dec) {
            const std::string miss = m.precMissing();
            if (!miss.empty())
                throw ModelError("PreciseKernels",
                                 "precise mode: this GPU's kernels have no " + miss +
                                     ", and precise mode never falls back to int8 decode activations. Use --balance (int8 decode / verify "
                                     "activations, as llama.cpp's q8_1)");
        }
        if (const auto v = envGet("RELAX"); v && !v->empty())
            nu::applyOverride(p, nu::Item::relaxacc, "RELAX", *v, *v != "0", true);
        m.relax = whirl::relax::Params{};
        if (p.on(nu::Item::relaxacc)) {
            m.relax.on = true;
            if (const auto v = envGet("RELAX_K"); v && !v->empty()) m.relax.k = static_cast<std::uint32_t>(std::stoul(*v));
            if (const auto v = envGet("RELAX_ALPHA"); v && !v->empty()) m.relax.alpha = std::stof(*v);
            if (const auto v = envGet("RELAX_EPS"); v && !v->empty()) m.relax.eps = std::stof(*v);
            if (const auto v = envGet("RELAX_DELTA"); v && !v->empty()) m.relax.delta = std::stof(*v);
            p.items[static_cast<std::size_t>(nu::Item::relaxacc)].note +=
                std::format(" [k {}, alpha {:.2f}, eps {:.2f}, delta {:.2f}]", m.relax.topk(), m.relax.alpha, m.relax.eps, m.relax.delta);
        }
        if (const auto v = envGet("SPEC_SAMPLE"); v && !v->empty())
            nu::applyOverride(p, nu::Item::specsample, "SPEC_SAMPLE", *v, *v != "0", t.mtp && t.k_spec_sample);
        m.num_plan = p;
        log += "  " + nu::logLine(p) + "\n";
        if (!m.load_note.empty()) log += "  " + m.load_note;
    }

    std::string cache_path;
    std::vector<std::string> read_paths4, read_paths_old;
    const std::string base = localAppData();
    if (!base.empty()) {
        std::error_code ec;
        const auto fsz = std::filesystem::file_size(widen(model_path), ec);
        if (!ec) {
            // the R9700 (gfx1201) keeps the plain file name; other code objects get their own.
            // gfx1151 table revision 2: the small-batch WMMA GEMMs (gemms) became candidates,
            // so tables written before them are tuned again.
            const std::string sfx = m.arch == hip::Arch::gfx1151 ? "-gfx1151r2" : "";
            const std::string bname = narrow(std::filesystem::path(widen(model_path)).filename().wstring());
            const std::string stem = bname + "-" + std::to_string(fsz) + sfx + ".txt";
            const std::string dir = base + "\\whirl";
            std::filesystem::create_directories(widen(dir), ec);
            cache_path = dir + "\\tune4-" + stem;
            // tables written by the research prototype are read as well (same format)
            for (const char* d : {"\\whirl\\", "\\whirl-r9700\\"}) read_paths4.push_back(base + d + "tune4-" + stem);
            for (const char* pfx : {"tune3-", "tune-"})
                for (const char* d : {"\\whirl\\", "\\whirl-r9700\\"}) read_paths_old.push_back(base + d + pfx + stem);
        }
    }
    for (const std::string& p : read_paths4) {
        std::string text;
        if (tryRead(p, text) && m.readTune(text, n_tune)) {
            log += "  prefill GEMM tune: cached (" + p + ")\n";
            return;
        }
    }
    // only an older table: keep its buckets 0 / 1 (512 / 4096), tune the small-batch ones
    bool have_old = false;
    for (const std::string& p : read_paths_old) {
        if (have_old) break;
        std::string text;
        if (tryRead(p, text)) have_old = m.readTune(text, 2);
    }
    // one-time notice (stderr): the tuning pause would otherwise look like a hang
    // (measured: R9700 ~100 s for a 27B Q4_K_M model, a few s for MXFP4; Radeon 8060S
    // ~460 s for the 27B Q4_K_M)
    std::fprintf(stderr,
                 "First run with this model on this GPU: tuning the prefill kernels. This takes up to about\n"
                 "%s and happens only once%s%s.\n",
                 m.arch == hip::Arch::gfx1151 ? "8 minutes on the Radeon 8060S (less for MXFP4 models)"
                                              : "2 minutes (a few seconds for MXFP4 models)",
                 cache_path.empty() ? "" : "; the result is cached in ", cache_path.c_str());
    std::fflush(stderr);
    const double t0 = nowMs();
    hip::sync();
    hip::memset(m.h, 0, 4ull * m.max_batch * m.cfg.n_embd);
    hip::memset(m.ffn_g, 0, 4ull * m.max_batch * m.ff_scratch);
    if (!have_old) {
        m.autotuneGemm(0, 3, nullptr);
        m.autotuneGemm(1, 3, nullptr);
    }
    for (std::size_t b = 2; b < n_tune; ++b) m.autotuneGemm(b, 3, nullptr);
    hip::sync();
    log += fmt("  prefill GEMM autotune: %.1f s\n", (nowMs() - t0) / 1000.0);
    if (!cache_path.empty()) {
        try {
            writeFile(cache_path, m.writeTune());
        } catch (...) {
        }
    }
}

}  // namespace whirl::qwen35
