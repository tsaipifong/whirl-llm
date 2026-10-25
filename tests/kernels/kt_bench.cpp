// kt::Bench: the shared kernel timer of whirl-kernel-test --bench (see kt_bench.h). No HIP here.
// SPDX-License-Identifier: Apache-2.0
// Written for WHIRL. SHA-256 follows FIPS 180-4 (public algorithm, written from the specification).

#include "kt_bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>

namespace kt {

namespace {

class RealClock : public Clock {
public:
    double now() override {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void sleep(double s) override {
        if (s > 0) std::this_thread::sleep_for(std::chrono::duration<double>(s));
    }
};

bool good(double ms) { return std::isfinite(ms) && ms > 0; }

std::string relLabel(double rel) {
    char b[32];
    std::snprintf(b, sizeof(b), "rel<=%g%%", rel * 100.0);
    return b;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return whirl::bench::quantile_sorted(v, 0.5);
}

}  // namespace

Clock& realClock() {
    static RealClock c;
    return c;
}

double relIqr(std::vector<double> v) {
    if (v.size() < 2) return 0;
    std::sort(v.begin(), v.end());
    const double med = whirl::bench::quantile_sorted(v, 0.5);
    if (med == 0) return 0;
    return (whirl::bench::quantile_sorted(v, 0.75) - whirl::bench::quantile_sorted(v, 0.25)) / std::fabs(med);
}

double CellStats::rel() const { return relIqr(samples); }

Bench::Bench(BenchCfg cfg, Clock* clock) : cfg_(cfg), clock_(clock ? clock : &realClock()) {}

double Bench::sample(const BlockFn& f, int block, std::uint64_t& rot, std::string& err) {
    const double ms = f(block, rot);
    rot += static_cast<std::uint64_t>(block);
    if (!good(ms)) {
        err = std::isfinite(ms) ? "zero-duration or negative block time (nothing was timed)" : "non-finite block time";
        return -1;
    }
    return 1000.0 * ms / block;
}

// calibrate (block doubles until >= min_block_us), warm up with that block (>= warmup_ms GPU time, calibration
// included), then re-check the block on the warm clock (a cold clock makes the kernel slower: the block may be short)
Bench::Cal Bench::warmCalibrate(const BlockFn& f, std::uint64_t& rot) {
    Cal c;
    double warm = 0;
    int block = 1;
    double us = 0;
    for (;;) {
        us = sample(f, block, rot, c.error);
        if (us <= 0) return c;
        warm += us * block / 1000.0;
        if (us * block >= cfg_.min_block_us || block >= cfg_.max_block) break;
        block = std::min(block * 2, cfg_.max_block);
    }
    while (warm < cfg_.warmup_ms) {
        us = sample(f, block, rot, c.error);
        if (us <= 0) return c;
        warm += us * block / 1000.0;
    }
    while (us * block < cfg_.min_block_us && block < cfg_.max_block) {
        block = std::min(block * 2, cfg_.max_block);
        us = sample(f, block, rot, c.error);
        if (us <= 0) return c;
        warm += us * block / 1000.0;
    }
    c.block = block;
    last_warm_ms_ = warm;
    return c;
}

void Bench::setAnchor(BlockFn f, std::string name) {
    anchor_ = std::move(f);
    anchor_name_ = std::move(name);
    const double t0 = clock_->now();
    const Cal c = warmCalibrate(anchor_, anchor_rot_);
    if (c.block <= 0) {
        anchor_ = nullptr;
        anchor_error_ = c.error;
        return;
    }
    anchor_block_ = c.block;
    std::string err;
    anchor_ref_us_ = sample(anchor_, anchor_block_, anchor_rot_, err);
    if (anchor_ref_us_ <= 0) {
        anchor_ = nullptr;
        anchor_error_ = err;
    }
    account(clock_->now() - t0);
}

double Bench::anchorTime() {
    std::string err;
    return sample(anchor_, anchor_block_, anchor_rot_, err);
}

// one anchor reading; on drift: cooldown (sleeps excluded from the cell's measuring time) and false
bool Bench::anchorCheck(CellStats& cs, bool& noisy) {
    auto note = [&](double us) {
        if (us <= 0) return;
        cs.anchor_min_us = cs.anchor_min_us == 0 ? us : std::min(cs.anchor_min_us, us);
        cs.anchor_max_us = std::max(cs.anchor_max_us, us);
    };
    double a = anchorTime();
    note(a);
    if (a <= 0) {
        cs.error = "anchor: zero-duration block";
        return false;
    }
    if (a < anchor_ref_us_) anchor_ref_us_ = a;  // cooler than any reading so far: the new reference
    cs.anchor_ref_us = anchor_ref_us_;
    if ((a / anchor_ref_us_ - 1.0) * 100.0 <= cfg_.drift_pct) return true;
    if (++cs.cooldowns > cfg_.max_cooldowns) {
        noisy = true;
        return false;
    }
    double slept = 0, step = cfg_.cool_first_s;
    while (slept < cfg_.cool_total_s - 1e-12) {
        const double s = std::min(step, cfg_.cool_total_s - slept);
        clock_->sleep(s);
        slept += s;
        a = anchorTime();
        note(a);
        if (a > 0 && a < anchor_ref_us_) anchor_ref_us_ = a;
        if (a > 0 && (a / anchor_ref_us_ - 1.0) * 100.0 <= cfg_.drift_pct) break;
        step = std::min(step * 2, cfg_.cool_max_step_s);
    }
    cs.cooldown_s += slept;
    cs.anchor_ref_us = anchor_ref_us_;
    return false;
}

void Bench::beforeCell() {
    if (cfg_.burst_idle_s > 0 && busy_ > 0 && busy_ + last_cell_s_ > cfg_.burst_s) {
        clock_->sleep(cfg_.burst_idle_s);
        ++bursts_;
        busy_ = 0;
        overrun_counted_ = false;
    }
}

void Bench::account(double busy_s) {
    busy_ += busy_s;
    last_cell_s_ = busy_s;
    if (busy_ > cfg_.burst_s && !overrun_counted_) {
        ++overruns_;
        overrun_counted_ = true;
    }
}

CellStats Bench::run(const BlockFn& f) {
    CellStats cs;
    const double t0 = clock_->now();
    std::uint64_t rot = 0;
    auto finish = [&](const std::string& why) {
        cs.stop_reason = why;
        if (why == "error") cs.samples.clear();
        cs.st = whirl::bench::summarize(cs.samples);
        cs.measure_s = clock_->now() - t0 - cs.cooldown_s;
        account(cs.measure_s);
        return cs;
    };
    const Cal cal = warmCalibrate(f, rot);
    if (cal.block <= 0) {
        cs.error = cal.error;
        return finish("error");
    }
    cs.block = cal.block;
    cs.warm_ms = last_warm_ms_;
    if (anchor_) cs.anchor_name = anchor_name_;
    bool noisy = false;
    std::size_t clean = 0;  // samples [0, clean) are covered by a clean anchor reading
    int since_anchor = 0;
    if (anchor_) {
        if (!anchorCheck(cs, noisy)) {
            if (noisy) return finish("noisy");
            if (!cs.error.empty()) return finish("error");
        }
    }
    auto drifted = [&]() {  // discard the samples since the last clean anchor
        cs.discarded += static_cast<int>(cs.samples.size() - clean);
        cs.samples.resize(clean);
        since_anchor = 0;
    };
    for (;;) {
        std::string err;
        const double us = sample(f, cs.block, rot, err);
        if (us <= 0) {
            cs.error = err;
            return finish("error");
        }
        cs.samples.push_back(us);
        ++since_anchor;
        if (anchor_ && since_anchor >= cfg_.anchor_every) {
            if (anchorCheck(cs, noisy)) {
                clean = cs.samples.size();
                since_anchor = 0;
            } else {
                if (noisy) return finish("noisy");
                if (!cs.error.empty()) return finish("error");
                drifted();
                continue;
            }
        }
        const int n = static_cast<int>(cs.samples.size());
        const double elapsed = clock_->now() - t0 - cs.cooldown_s;
        const char* why = nullptr;
        if (n >= cfg_.min_samples && relIqr(cs.samples) <= cfg_.rel_target) why = "rel";
        else if (n >= cfg_.max_samples) why = "n";
        else if (elapsed >= cfg_.max_cell_s) why = "time";
        if (!why) continue;
        if (anchor_ && clean < cs.samples.size()) {  // the tail must be covered by a clean anchor too
            if (anchorCheck(cs, noisy)) {
                clean = cs.samples.size();
            } else {
                if (noisy) return finish("noisy");
                if (!cs.error.empty()) return finish("error");
                drifted();
                if (std::string(why) == "time") return finish("time");
                continue;
            }
        }
        return finish(std::string(why) == "rel" ? relLabel(cfg_.rel_target) : why);
    }
}

AbStats Bench::runAB(const BlockFn& fa, const BlockFn& fb) {
    AbStats ab;
    const double t0 = clock_->now();
    std::uint64_t ra = 0, rb = 0;
    std::vector<int> first_a;  // per kept pair: 1 = A ran first
    auto finish = [&](const std::string& why) {
        ab.ratio.stop_reason = ab.a.stop_reason = ab.b.stop_reason = why;
        if (why == "error") ab.ratio.samples.clear(), ab.a.samples.clear(), ab.b.samples.clear(), first_a.clear();
        ab.a.st = whirl::bench::summarize(ab.a.samples);
        ab.b.st = whirl::bench::summarize(ab.b.samples);
        ab.ratio.st = whirl::bench::summarize(ab.ratio.samples);
        std::vector<double> rab, rba;
        for (std::size_t i = 0; i < first_a.size(); ++i) (first_a[i] ? rab : rba).push_back(ab.ratio.samples[i]);
        ab.order_ab = median(rab);
        ab.order_ba = median(rba);
        ab.pairs = static_cast<int>(ab.ratio.samples.size());
        ab.ratio.measure_s = clock_->now() - t0 - ab.ratio.cooldown_s;
        ab.a.measure_s = ab.b.measure_s = ab.ratio.measure_s;
        account(ab.ratio.measure_s);
        return ab;
    };
    const Cal ca = warmCalibrate(fa, ra);
    if (ca.block <= 0) {
        ab.ratio.error = "A: " + ca.error;
        return finish("error");
    }
    const double warm_a = last_warm_ms_;
    const Cal cb = warmCalibrate(fb, rb);
    if (cb.block <= 0) {
        ab.ratio.error = "B: " + cb.error;
        return finish("error");
    }
    ab.a.block = ca.block;
    ab.b.block = cb.block;
    ab.a.warm_ms = warm_a;
    ab.b.warm_ms = last_warm_ms_;
    CellStats& cs = ab.ratio;  // anchor bookkeeping lives on the ratio
    if (anchor_) cs.anchor_name = anchor_name_;
    bool noisy = false;
    std::size_t clean = 0;
    int since_anchor = 0;
    if (anchor_ && !anchorCheck(cs, noisy)) {
        if (noisy) return finish("noisy");
        if (!cs.error.empty()) return finish("error");
    }
    auto drifted = [&]() {
        cs.discarded += static_cast<int>(cs.samples.size() - clean);
        cs.samples.resize(clean);
        ab.a.samples.resize(clean);
        ab.b.samples.resize(clean);
        first_a.resize(clean);
        since_anchor = 0;
    };
    for (std::uint64_t pair = 0;; ++pair) {
        const bool a_first = pair % 2 == 0;
        std::string err;
        double sa = 0, sb = 0;
        if (a_first) {
            sa = sample(fa, ca.block, ra, err);
            if (sa > 0) sb = sample(fb, cb.block, rb, err);
        } else {
            sb = sample(fb, cb.block, rb, err);
            if (sb > 0) sa = sample(fa, ca.block, ra, err);
        }
        if (sa <= 0 || sb <= 0) {
            cs.error = err;
            return finish("error");
        }
        ab.a.samples.push_back(sa);
        ab.b.samples.push_back(sb);
        cs.samples.push_back(sb / sa);
        first_a.push_back(a_first ? 1 : 0);
        ++since_anchor;
        if (anchor_ && since_anchor >= cfg_.anchor_every) {
            if (anchorCheck(cs, noisy)) {
                clean = cs.samples.size();
                since_anchor = 0;
            } else {
                if (noisy) return finish("noisy");
                if (!cs.error.empty()) return finish("error");
                drifted();
                continue;
            }
        }
        const int n = static_cast<int>(cs.samples.size());
        if (n % 2 != 0) continue;  // both orders equally often (abjudge rule: even rounds)
        const double elapsed = clock_->now() - t0 - cs.cooldown_s;
        const char* why = nullptr;
        if (n >= cfg_.min_samples && relIqr(cs.samples) <= cfg_.rel_target) why = "rel";
        else if (n >= cfg_.max_samples) why = "n";
        else if (elapsed >= 2 * cfg_.max_cell_s) why = "time";
        if (!why) continue;
        if (anchor_ && clean < cs.samples.size()) {
            if (anchorCheck(cs, noisy)) {
                clean = cs.samples.size();
            } else {
                if (noisy) return finish("noisy");
                if (!cs.error.empty()) return finish("error");
                drifted();
                if (std::string(why) == "time") return finish("time");
                continue;
            }
        }
        return finish(std::string(why) == "rel" ? relLabel(cfg_.rel_target) : why);
    }
}

int rotationCopies(std::uint64_t ws, std::uint64_t target, std::uint64_t cap, int max_copies) {
    if (ws == 0) return 1;
    std::uint64_t n = (target + ws - 1) / ws;
    n = std::min<std::uint64_t>(n, cap / ws);
    n = std::min<std::uint64_t>(n, static_cast<std::uint64_t>(std::max(1, max_copies)));
    return static_cast<int>(std::max<std::uint64_t>(1, n));
}

std::uint64_t rotationTarget(const std::string& arch) {
    if (arch.rfind("gfx1151", 0) == 0) return 2ull << 30;
    return 512ull << 20;
}

Roof roofFor(const std::string& arch) {
    // tools\quick\sol.py HW (audit\KGAP-1.md, PEAK-1.md, KLIB-1.md): 8060S practical f16 22 TF/s (KLIB-1 §2)
    if (arch.rfind("gfx1201", 0) == 0) return {640e9, 618e9, 189e12, 130e12};
    if (arch.rfind("gfx1151", 0) == 0) return {256e9, 238e9, 59.4e12, 22e12};
    return {};
}

// ---------------------------------------------------------------- SHA-256 (FIPS 180-4)
std::string sha256Hex(const void* data, std::size_t n) {
    static const std::uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
        0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
        0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
        0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
        0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](std::uint32_t x, int r) { return (x >> r) | (x << (32 - r)); };
    auto block = [&](const unsigned char* p) {
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
            const std::uint32_t t1 = hh + S1 + ch + K[i] + w[i];
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
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
    };
    const unsigned char* p = static_cast<const unsigned char*>(data);
    std::size_t i = 0;
    for (; i + 64 <= n; i += 64) block(p + i);
    unsigned char tail[128] = {};
    const std::size_t rest = n - i;
    if (rest) std::memcpy(tail, p + i, rest);
    tail[rest] = 0x80;
    const std::size_t tl = rest + 1 + 8 <= 64 ? 64 : 128;
    const std::uint64_t bits = static_cast<std::uint64_t>(n) * 8;
    for (int k = 0; k < 8; ++k) tail[tl - 1 - k] = static_cast<unsigned char>(bits >> (8 * k));
    block(tail);
    if (tl == 128) block(tail + 64);
    static const char hex[] = "0123456789abcdef";
    std::string o;
    for (std::uint32_t v : h)
        for (int k = 7; k >= 0; --k) o += hex[(v >> (4 * k)) & 15];
    return o;
}

// ---------------------------------------------------------------- rows
namespace {
std::string jarr(const std::vector<double>& v) {
    std::string s = "[";
    for (std::size_t i = 0; i < v.size(); ++i) s += (i ? ", " : "") + whirl::bench::json_number(v[i]);
    return s + "]";
}
std::string withKv(std::string s, const std::string& kv) { return s.empty() ? kv : s + ";" + kv; }
}  // namespace

whirl::bench::Row cellRow(const CellInfo& ci, const CellStats& cs, const std::string& tool_ver) {
    namespace wb = whirl::bench;
    wb::Row r;
    r.tool = "whirl-kernel-test";
    r.tool_ver = tool_ver;
    r.component = "K-" + ci.family;
    r.cell = ci.cell;
    r.metric = "kern." + ci.family + "." + ci.kernel;
    r.unit = "us";
    if (cs.ok()) r.set_stats(cs.st);
    r.stop_reason = cs.stop_reason;
    wb::Work w;
    w.launches = ci.launches_per_call;
    r.work = w;
    r.code_hash = ci.code_hash;
    r.shape = ci.shape;
    // the calibrated block is adaptive (it follows the GPU clock), so it is an extra key, not part of settings: a
    // settings change makes db_lookup re-measure, a block change must not
    r.settings = ci.settings;
    if (!ci.variant.empty()) r.settings = withKv(r.settings, "variant=" + ci.variant);
    if (cs.anchor_ref_us > 0) {
        wb::Anchor a;
        a.ref = cs.anchor_name.empty() ? "anchor" : cs.anchor_name;
        a.min = cs.anchor_min_us;
        a.max = cs.anchor_max_us;
        a.drift_pct = (cs.anchor_max_us - cs.anchor_ref_us) / cs.anchor_ref_us * 100.0;
        a.cooldowns = cs.cooldowns;
        r.anchor = a;
    }
    r.gpu_arch = ci.gpu_arch;
    r.gpu_pci = ci.gpu_pci;
    r.started = ci.started;
    r.ended = wb::iso_now();
    r.extra.push_back({"block", wb::json_number(static_cast<std::int64_t>(cs.block))});
    r.extra.push_back({"discarded", wb::json_number(static_cast<std::int64_t>(cs.discarded))});
    if (cs.cooldown_s > 0) r.extra.push_back({"cooldown_s", wb::json_number(cs.cooldown_s)});
    r.extra.push_back({"rel_iqr", wb::json_number(cs.rel())});
    if (!cs.error.empty()) r.extra.push_back({"error", wb::json_string(cs.error)});
    const Roof roof = roofFor(ci.gpu_arch);
    if (cs.ok() && cs.st.median > 0) {
        if (ci.bytes > 0) {
            const double gbps = ci.bytes / cs.st.median / 1e3;
            r.extra.push_back({"bytes", wb::json_number(ci.bytes)});
            r.extra.push_back({"gbps", wb::json_number(gbps)});
            if (roof.bw > 0) {
                r.extra.push_back({"pct_of_roof", wb::json_number(100.0 * gbps * 1e9 / roof.bw)});
                r.extra.push_back({"roof", wb::json_string("bw")});
            }
        } else if (ci.flops > 0) {
            const double tf = ci.flops / cs.st.median / 1e6;
            r.extra.push_back({"flops", wb::json_number(ci.flops)});
            r.extra.push_back({"tflops", wb::json_number(tf)});
            if (roof.f16_prac > 0) {
                r.extra.push_back({"pct_of_roof", wb::json_number(100.0 * tf * 1e12 / roof.f16_prac)});
                r.extra.push_back({"roof", wb::json_string("f16_prac")});
            }
        }
    }
    r.extra.push_back({"samples", jarr(cs.samples)});
    return r;
}

whirl::bench::Row abRow(const CellInfo& ci, const AbStats& ab, const std::string& tool_ver, const std::string& code_hash_b) {
    namespace wb = whirl::bench;
    CellInfo c2 = ci;
    c2.kernel = ci.kernel + ".ab_ratio";
    c2.bytes = c2.flops = 0;
    c2.settings = withKv(ci.settings, "ab=paired_abba");
    wb::Row r = cellRow(c2, ab.ratio, tool_ver);
    r.unit = "ratio";
    r.extra.erase(std::remove_if(r.extra.begin(), r.extra.end(), [](const auto& kv) { return kv.first == "block"; }), r.extra.end());
    r.extra.push_back({"block_a", wb::json_number(static_cast<std::int64_t>(ab.a.block))});
    r.extra.push_back({"block_b", wb::json_number(static_cast<std::int64_t>(ab.b.block))});
    r.extra.push_back({"code_hash_b", wb::json_string(code_hash_b)});
    r.extra.push_back({"pairs", wb::json_number(static_cast<std::int64_t>(ab.pairs))});
    r.extra.push_back({"order_ab", wb::json_number(ab.order_ab)});
    r.extra.push_back({"order_ba", wb::json_number(ab.order_ba)});
    if (ab.a.st.n > 0) r.extra.push_back({"a_median_us", wb::json_number(ab.a.st.median)});
    if (ab.b.st.n > 0) r.extra.push_back({"b_median_us", wb::json_number(ab.b.st.median)});
    return r;
}

}  // namespace kt
