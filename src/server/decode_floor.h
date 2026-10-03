// Decode floor (--decode-min-tps N): prefill / decode fairness while slots decode.
//
// Without it the main loop runs one prefill forward (up to the prefill batch, e.g.
// 3-4 combined 1024-row chunks, ~1.5 s) per decode cycle, so a streaming request
// advances one cycle per prefill forward and drops to a few tok/s while other
// requests prefill long prompts. With a floor N > 0, and only while at least one
// slot is decoding:
//   - a prefill forward is limited to a row budget (whole schedule chunks; the
//     oldest request's next chunk always runs, so prefill always progresses), and
//   - after each prefill forward, decode cycles run alone until every decoding
//     slot has produced >= N tokens per second over the period that began with
//     that forward ("period" = one prefill forward + the decode cycles after it).
// The number of decode cycles therefore adapts every cycle to the measured tokens
// (MTP acceptance, number of decoding slots) and prefill forward time. If the floor
// is not reachable even by pure decoding, decode-only time per period is capped at
// max_decode_ratio x the period's prefill time (prefill keeps >= ~20% of the GPU).
// The row budget comes from EMAs: prefill rows/ms and decode-cycle time, so that a
// prefill forward's stall stays near gap_tokens tokens' worth at the floor rate.
//
// Pure bookkeeping on caller-supplied times (ms on any monotonic clock): no device
// calls, no clock reads, so it is unit-tested on the host. Changing the schedule
// only changes which rows share a forward and when decode cycles run; every GEMM /
// MoE tile is row-invariant, so outputs are bit-identical for any N.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace whirl::server {

class DecodeFloor {
public:
    struct SlotTok {
        std::uint32_t slot = 0;
        std::uint64_t job = 0;    // request id (a new request in the slot restarts its accounting)
        std::uint64_t n_gen = 0;  // tokens produced so far by that request
    };
    struct Plan {
        std::size_t rows = 0;       // row budget of the next prefill forward
        std::uint32_t cycles = 0;   // predicted decode cycles after it (0 = unknown / not needed)
        bool reachable = true;      // the floor is reachable by pure decoding (EMA estimate)
    };

    static constexpr double ema_alpha = 0.25;
    static constexpr double gap_tokens = 10;         // target stall per prefill forward, in tokens at the floor rate
    static constexpr double max_decode_ratio = 4;    // decode-only time per period <= 4 x its prefill time
    static constexpr double min_decode_ms = 50;      // ... but at least this much
    static constexpr double unknown_pf_decode_ms = 1000;  // cap while the period's prefill time is not measured yet

    explicit DecodeFloor(double min_tps = 0) { configure(min_tps); }
    void configure(double min_tps) { tps_ = min_tps > 0 ? min_tps : 0; }
    bool on() const { return tps_ > 0; }
    double minTps() const { return tps_; }

    // ---- measurements
    // One decode cycle finished (dur_ms: its wall time; with_prefill: a prefill forward
    // ran in the same loop iteration, so dur_ms includes it and is not a cycle time).
    void noteCycle(double dur_ms, bool with_prefill, std::span<const SlotTok> dec) {
        if (!with_prefill && dur_ms > 0) cyc_ms_ = ema(cyc_ms_, dur_ms);
        for (const SlotTok& s : dec) {
            Acc* a = find(s.slot);
            if (a == nullptr || a->job != s.job) continue;
            if (a->has_last) a->tpc = ema(a->tpc, static_cast<double>(s.n_gen - std::min(s.n_gen, a->last)));
            a->last = s.n_gen;
            a->has_last = true;
        }
    }
    // A prefill forward of `rows` rows took pf_ms (measured to the end of the decode
    // cycle that followed, minus the decode-cycle EMA).
    void notePrefill(std::size_t rows, double pf_ms) {
        if (rows == 0 || pf_ms <= 0) return;
        rows_per_ms_ = ema(rows_per_ms_, static_cast<double>(rows) / pf_ms);
        period_pf_ms_ = pf_ms;
    }
    double cycleMs() const { return cyc_ms_; }
    double rowsPerMs() const { return rows_per_ms_; }

    // ---- scheduling
    // May a prefill forward start now? false = run a decode cycle alone first.
    // dec: every decoding slot now. Slots that started decoding during the period
    // are accounted from the first call that sees them.
    bool prefillAllowed(double now_ms, std::span<const SlotTok> dec) {
        if (!on() || dec.empty() || !in_period_) return true;
        sync(now_ms, dec);
        // starvation guard: the floor may be out of reach (too many decoding slots,
        // low acceptance); prefill still gets its share
        const double pf = period_pf_ms_ > 0 ? period_pf_ms_ : 0;
        const double dec_only = now_ms - period_t0_ - pf;
        const double cap = pf > 0 ? std::max(min_decode_ms, max_decode_ratio * pf) : unknown_pf_decode_ms;
        if (dec_only >= cap) return true;
        for (const Acc& a : acc_) {
            if (!a.seen) continue;
            const double need = tps_ * (now_ms - a.t0) / 1000.0;
            if (static_cast<double>(a.n_now - a.n0) < need) return false;
        }
        return true;
    }
    // A prefill forward starts now (call only while slots decode): opens a period.
    void startPeriod(double now_ms, std::span<const SlotTok> dec) {
        in_period_ = true;
        period_t0_ = now_ms;
        period_pf_ms_ = 0;
        for (Acc& a : acc_) a.seen = false;
        for (const SlotTok& s : dec) {
            Acc* a = find(s.slot);
            if (a == nullptr) {
                acc_.push_back(Acc{});
                a = &acc_.back();
                a->slot = s.slot;
            }
            if (a->job != s.job) {
                a->job = s.job;
                a->tpc = 0;
                a->has_last = false;
            }
            a->n0 = a->n_now = s.n_gen;
            a->t0 = now_ms;
            a->seen = true;
        }
        drop();
    }
    // No slot decodes: the next prefill is unrestricted and starts a fresh period.
    void idle() {
        in_period_ = false;
        acc_.clear();
    }

    // Row budget for a prefill forward whose oldest chunk has `first` rows (always
    // allowed) with at most `cap` rows per forward, and the predicted decode cycles.
    Plan plan(std::size_t first, std::size_t cap) const {
        Plan p;
        p.rows = std::min(first, cap);
        if (!on()) {
            p.rows = cap;
            return p;
        }
        if (rows_per_ms_ > 0) {
            const double gap_ms = 1000.0 * gap_tokens / tps_;
            const double r = rows_per_ms_ * gap_ms;
            p.rows = std::max(p.rows, std::min(cap, static_cast<std::size_t>(r)));
        }
        // decode cycles needed after a forward of p.rows: k * tpc >= N * (T_pf + k * T_c)
        double tpc = 0;
        bool any = false;
        for (const Acc& a : acc_) {
            if (!a.seen || a.tpc <= 0) continue;
            tpc = any ? std::min(tpc, a.tpc) : a.tpc;
            any = true;
        }
        if (any && cyc_ms_ > 0 && rows_per_ms_ > 0) {
            const double t_pf = static_cast<double>(p.rows) / rows_per_ms_;
            const double per_cycle = tpc - tps_ * cyc_ms_ / 1000.0;  // net tokens per cycle above the floor
            if (per_cycle <= 0) {
                p.reachable = false;
                p.cycles = static_cast<std::uint32_t>(std::ceil(max_decode_ratio * t_pf / cyc_ms_));
            } else {
                p.cycles = static_cast<std::uint32_t>(std::ceil(tps_ * t_pf / 1000.0 / per_cycle));
            }
        }
        return p;
    }

private:
    struct Acc {
        std::uint32_t slot = 0;
        std::uint64_t job = 0;
        std::uint64_t n0 = 0, n_now = 0;  // tokens at the period start (or when first seen) / now
        double t0 = 0;
        bool seen = false;                // decoding in the current period
        double tpc = 0;                   // EMA tokens per decode cycle
        std::uint64_t last = 0;
        bool has_last = false;
    };
    static double ema(double cur, double x) { return cur <= 0 ? x : cur + ema_alpha * (x - cur); }
    Acc* find(std::uint32_t slot) {
        for (Acc& a : acc_)
            if (a.slot == slot) return &a;
        return nullptr;
    }
    void sync(double now_ms, std::span<const SlotTok> dec) {
        for (Acc& a : acc_) a.seen = false;
        for (const SlotTok& s : dec) {
            Acc* a = find(s.slot);
            if (a == nullptr) {
                acc_.push_back(Acc{});
                a = &acc_.back();
                a->slot = s.slot;
                a->job = s.job + 1;  // force the restart below
            }
            if (a->job != s.job || a->t0 < period_t0_) {
                // new in this period (started decoding after the forward, or a new request)
                if (a->job != s.job) {
                    a->tpc = 0;
                    a->has_last = false;
                }
                a->job = s.job;
                a->n0 = s.n_gen;
                a->t0 = now_ms;
            }
            a->n_now = s.n_gen;
            a->seen = true;
        }
        drop();
    }
    void drop() {
        acc_.erase(std::remove_if(acc_.begin(), acc_.end(), [](const Acc& a) { return !a.seen; }), acc_.end());
    }

    double tps_ = 0;
    double cyc_ms_ = 0;
    double rows_per_ms_ = 0;
    bool in_period_ = false;
    double period_t0_ = 0;
    double period_pf_ms_ = 0;
    std::vector<Acc> acc_;
};

}  // namespace whirl::server
