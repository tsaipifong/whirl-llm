// SPDX-License-Identifier: Apache-2.0
// DFlash2 speculative decoding (`--draft dflash`): the host-side policy shared by whirl chat /
// bench (src/cli/main.cpp), whirl-server (src/server/server_main.cpp, engine.cpp) and the drafter
// forward (src/model/dflash.cpp). Header-only and GPU-free so tests/unit covers it.
//
// Design (DFLASH-1, owner-approved 2026-10-05, v0.2.0):
//   * `--draft mtp|dflash` picks the draft source; default mtp. dflash needs the drafter GGUF
//     (--draft-model FILE, else WHIRL_DRAFT_MODEL) that fits the target (DFlashConfig::validate, headers only) and the dflash_*
//     kernels in the GPU's code object; otherwise a warning and the MTP drafts (choose()).
//   * dflash mode does not load the MTP block (no weights, no MTP KV layer): the KV pool is sized
//     after the drafter's weights, capture and rings, and the startup log prints its capacity.
//   * n-gram drafts stay on in both modes; per cycle the cost model picks n-gram or DFlash2.
//   * DFlash2 drafts are literal drafts (like n-gram): the target verifies them with the same
//     bitwise verify as plain decoding, and acceptance is EXACT in every numerics mode
//     (precise / balance / fast). relaxacc (fast) relaxes MTP drafts only (relax_accept.h), so with
//     --draft dflash the output equals plain greedy decoding of the same numerics mode; for
//     sampling the reply is a function of (target logits, seed) only. relaxAppliesTo() is the rule
//     (the server engine calls it per slot; whirl chat's dflashDecode has no relaxed acceptance).
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::dflash {

enum class Source { mtp, dflash };

// "mtp" / "dflash"; anything else: nullopt (the caller reports the usage error)
inline std::optional<Source> parseSource(std::string_view s) {
    if (s == "mtp") return Source::mtp;
    if (s == "dflash") return Source::dflash;
    return std::nullopt;
}

inline const char* sourceName(Source s) { return s == Source::dflash ? "dflash" : "mtp"; }

// The drafter GGUF: --draft-model FILE wins; the WHIRL_DRAFT_MODEL environment variable is the
// fallback (a debug override when no --draft-model is given), so a stale variable never overrides the
// command line. An empty value counts as not set.
inline std::optional<std::string> resolveDraftModel(const std::optional<std::string>& cli, const std::optional<std::string>& env) {
    if (cli && !cli->empty()) return cli;
    if (env && !env->empty()) return env;
    return std::nullopt;
}
// Log line about where the drafter came from ("" = nothing to report: none set, or only --draft-model).
inline std::string draftModelNote(const std::optional<std::string>& cli, const std::optional<std::string>& env) {
    const bool c = cli && !cli->empty(), e = env && !env->empty();
    if (c && e && *cli != *env) return "WHIRL_DRAFT_MODEL=" + *env + " ignored: --draft-model " + *cli + " given";
    if (!c && e) return "DFlash2 drafter from WHIRL_DRAFT_MODEL=" + *env;
    return "";
}

// The draft source actually used, and the warning to print when a dflash request falls back.
struct Choice {
    Source source = Source::mtp;
    std::string warning;  // empty: nothing to report
};

// requested: --draft; draft_model: resolveDraftModel(); incompatible: dflashIncompatible() of the
// two headers ("" = fits; ignored without a draft model); kernels_ok: the code object has every
// dflash_* kernel (false only for an old / foreign WHIRL_CODE_OBJECT once both kernel sets carry
// them; pass true before the module is loaded and check again after).
inline Choice choose(Source requested, const std::optional<std::string>& draft_model, std::string_view incompatible, bool kernels_ok) {
    Choice c;
    if (requested == Source::mtp) {
        if (draft_model) c.warning = "a DFlash2 drafter is set (--draft-model / WHIRL_DRAFT_MODEL) but --draft is mtp: the DFlash2 drafter is not loaded";
        return c;
    }
    if (!draft_model) {
        c.warning = "--draft dflash needs the DFlash2 drafter GGUF (--draft-model FILE or WHIRL_DRAFT_MODEL): using MTP drafts";
        return c;
    }
    if (!incompatible.empty()) {
        c.warning = "--draft dflash: " + *draft_model + " does not fit this model (" + std::string(incompatible) + "): using MTP drafts";
        return c;
    }
    if (!kernels_ok) {
        c.warning = "--draft dflash: the GPU code object has no DFlash2 kernels (dflash_*): using MTP drafts";
        return c;
    }
    c.source = Source::dflash;
    return c;
}

// relaxacc (fast mode) relaxes MTP drafts only; n-gram and DFlash2 drafts keep exact acceptance.
// The server engine asks relaxAppliesTo() for every slot's drafts (engine.cpp decodeCycle: the top-k
// launch, greedy relaxed acceptance and typical sampling acceptance), so this is the one rule.
// DraftKind values are the engine's per-slot draft source codes.
enum class DraftKind : std::uint8_t { mtp = 0, ngram = 1, dflash = 2 };
inline bool relaxAppliesTo(DraftKind k, bool relax_on) { return relax_on && k == DraftKind::mtp; }
inline bool relaxApplies(Source s, bool relax_on) { return relaxAppliesTo(s == Source::dflash ? DraftKind::dflash : DraftKind::mtp, relax_on); }

// Fewest capture rows accepted (WHIRL_DFLASH_CAPTURE_ROWS): one verify of a full draft block of the
// largest block size validate() allows (16). Fewer rows than a verify left whirl chat's ingest
// pointing before the capture buffer (FR-DFL P2-2); every ingest also goes through planIngest now.
inline constexpr std::uint32_t kMinCaptureRows = 16;

// ---- Draft width (feat029-dflash2, DFLASH-2a T5): how many of the drafter's block - 1 drafts a cycle verifies.
// fixed (default, the v0.2.0 behaviour): every draft the walk returns, up to the limit.
// auto (WHIRL_DFLASH_WIDTH=auto): per cycle n* = argmax over n = 0 .. lim of E(n) / T(n), where E(n) = 1 + sum over
// k < n of prod_{j <= k} alpha[j] (the per-position acceptance of this sequence's DFlash2 drafts) and T(n) the
// measured DFlash2 cycle time with n drafts (drafter included for n >= 1; n = 0 = the drafter is skipped, a one-row
// verify; the features are still ingested, so the drafter context stays unbroken). Prior art (progress\
// feat029-dflash2.PRIOR_ART.md): ik_llama spec-tuner (arm 0 = target only, recovery probes), gufo
// DFlashLengthController (full acceptance is censored: probe wider), hipfire block modulation.
// Output: acceptance is exact for DFlash2 drafts in every mode (relaxAppliesTo), so the width changes only how
// many rows are verified, never the tokens (the GPU proof is the per-width hash cell of the feat029 runner).
enum class Width : std::uint8_t { fixed, automatic };

struct WidthParams {
    Width mode = Width::fixed;
    std::uint32_t probe_every = 16;  // auto: every probe_every-th DFlash2-eligible cycle verifies the full limit
    float row_rel = 0.025f;          // auto: prior relative cost of one more verify row (R9700 Q4_K_M: 0.84 / 33.7 ms)
    std::uint32_t max_n = 0;         // both modes: at most this many drafts (WHIRL_DFLASH_NMAX, like llama.cpp
                                     // --spec-draft-n-max; 0 = the drafter's block - 1); the per-width sweep of the runner
};

// Per sequence (CLI: the one sequence; server: per slot; reset per request).
struct WidthState {
    std::uint32_t since_probe = 0;  // eligible cycles since the last full-limit cycle
    std::uint32_t climb = 0;        // > 0: the last cycle accepted all of its `climb - 1` drafts (censored): try wider
    std::uint64_t skipped = 0, probes = 0, cycles = 0;  // statistics (drafter skipped / full-limit probes / decided)
};

struct WidthChoice {
    std::uint32_t n = 0;  // drafts to verify (<= lim)
    bool draft = false;   // run the drafter this cycle (false when n == 0)
    bool probe = false;   // a full-limit probe (the alphas above a truncated width are refreshed)
};

// E(n) with the per-position acceptance alpha (missing positions count as 0).
inline float expectedTokens(const float* alpha, std::uint32_t n_alpha, std::uint32_t n) {
    float e = 1, p = 1;
    for (std::uint32_t k = 0; k < n && k < n_alpha; ++k) {
        p *= alpha[k];
        e += p;
    }
    return e;
}

// measured(n) -> std::optional<float>: the measured cycle ms with n drafts, nullopt when not measured yet (only real
// measurements: an unmeasured width is estimated here from the widest measured one with row_rel per row; for n = 0
// the drafter's cost is not subtracted, so the estimate is conservative and the drafter is skipped only when even
// that wins, until a real n = 0 cycle was timed).
template <class Measured>
WidthChoice chooseWidth(const WidthParams& wp, WidthState& st, const float* alpha, std::uint32_t n_alpha, std::uint32_t lim,
                        Measured&& measured) {
    WidthChoice c;
    if (wp.max_n > 0) lim = std::min(lim, wp.max_n);
    if (lim == 0) return c;
    if (wp.mode == Width::fixed) {
        c.n = lim;
        c.draft = true;
        return c;
    }
    st.cycles += 1;
    st.since_probe += 1;
    if (wp.probe_every > 0 && st.since_probe >= wp.probe_every) {
        st.since_probe = 0;
        st.probes += 1;
        st.climb = 0;
        c.n = lim;
        c.draft = true;
        c.probe = true;
        return c;
    }
    // the widest measured width with drafts is the anchor of the estimates; none yet: draft the full limit
    std::uint32_t anchor = 0;
    float t_anchor = 0;
    for (std::uint32_t n = lim; n >= 1; --n) {
        if (auto t = measured(n); t && *t > 0) {
            anchor = n;
            t_anchor = *t;
            break;
        }
    }
    if (anchor == 0) {
        st.climb = 0;
        c.n = lim;
        c.draft = true;
        return c;
    }
    auto est = [&](std::uint32_t n) -> float {
        if (auto t = measured(n); t && *t > 0) return *t;
        const float f = 1.0f + wp.row_rel * (static_cast<float>(n) - static_cast<float>(anchor));
        return t_anchor * std::max(0.25f, f);
    };
    std::uint32_t best = lim;
    float best_s = -1;
    for (std::uint32_t n = 0; n <= lim; ++n) {
        const float s = expectedTokens(alpha, n_alpha, n) / est(n);
        if (s > best_s) {  // ties keep the narrower width (fewer rows for the same tokens per ms)
            best_s = s;
            best = n;
        }
    }
    // censored full acceptance (gufo): the run may be longer than the width verified, so go one wider
    if (st.climb > 0) best = std::max(best, std::min(lim, st.climb));
    c.n = best;
    c.draft = best > 0;
    if (!c.draft) st.skipped += 1;
    return c;
}

// After the verify of a cycle that verified `drafted` DFlash2 drafts (of the width chosen) and accepted `acc`.
inline void observeWidth(WidthState& st, std::uint32_t drafted, std::uint32_t acc, std::uint32_t lim) {
    st.climb = (drafted > 0 && acc >= drafted && drafted < lim) ? drafted + 1 : 0;
}

// Settings from the environment (WHIRL_DFLASH_PMIN / _NMIN / _WARM / _CAPTURE_ROWS / _WIDTH / _PROBE / _NMAX).
struct Params {
    float p_min = 0;                           // walk stops at a token with probability below p_min
    std::uint32_t n_min = 0;                   // fewer drafts than n_min: none
    std::uint32_t warm = 64;                   // server: rows after a restore before drafting again
    std::optional<std::uint32_t> capture_rows; // default: the drafter's window (2048 without one)
    WidthParams width;                         // WHIRL_DFLASH_WIDTH=fixed|auto, WHIRL_DFLASH_PROBE=N (>= 2, 0 = never), WHIRL_DFLASH_NMAX=N
};

namespace detail {
inline bool parseU32(const std::string& s, std::uint32_t& out) {
    if (s.empty() || s.size() > 10) return false;
    std::uint64_t v = 0;
    for (char ch : s) {
        if (ch < '0' || ch > '9') return false;
        v = v * 10 + static_cast<std::uint64_t>(ch - '0');
    }
    if (v > 0xffffffffull) return false;
    out = static_cast<std::uint32_t>(v);
    return true;
}
inline bool parseProb(const std::string& s, float& out) {
    // decimal fraction in [0, 1]: digits, at most one '.', no sign / exponent
    if (s.empty() || s.size() > 16) return false;
    double v = 0, scale = 0;
    bool digit = false;
    for (char ch : s) {
        if (ch == '.') {
            if (scale != 0) return false;
            scale = 1;
            continue;
        }
        if (ch < '0' || ch > '9') return false;
        digit = true;
        if (scale == 0) v = v * 10 + (ch - '0');
        else v += (ch - '0') * (scale /= 10);
    }
    if (!digit || v > 1.0) return false;
    out = static_cast<float>(v);
    return true;
}
}  // namespace detail

// env(name) -> std::optional<std::string> for the name without the WHIRL_ prefix ("DFLASH_PMIN").
// An invalid value keeps the default and adds a warning.
template <class Env>
Params parseParams(Env&& env, std::vector<std::string>& warnings) {
    Params p;
    if (auto v = env("DFLASH_PMIN")) {
        if (!detail::parseProb(*v, p.p_min)) warnings.push_back("WHIRL_DFLASH_PMIN=" + *v + " ignored (a probability 0..1)");
    }
    if (auto v = env("DFLASH_NMIN")) {
        if (!detail::parseU32(*v, p.n_min)) warnings.push_back("WHIRL_DFLASH_NMIN=" + *v + " ignored (a draft count)");
    }
    if (auto v = env("DFLASH_WARM")) {
        if (!detail::parseU32(*v, p.warm)) warnings.push_back("WHIRL_DFLASH_WARM=" + *v + " ignored (a row count)");
    }
    if (auto v = env("DFLASH_CAPTURE_ROWS")) {
        std::uint32_t r = 0;
        if (detail::parseU32(*v, r) && r >= kMinCaptureRows) p.capture_rows = r;
        else warnings.push_back("WHIRL_DFLASH_CAPTURE_ROWS=" + *v + " ignored (a row count >= " + std::to_string(kMinCaptureRows) + ")");
    }
    if (auto v = env("DFLASH_WIDTH")) {
        if (*v == "fixed") p.width.mode = Width::fixed;
        else if (*v == "auto") p.width.mode = Width::automatic;
        else warnings.push_back("WHIRL_DFLASH_WIDTH=" + *v + " ignored (fixed or auto)");
    }
    if (auto v = env("DFLASH_PROBE")) {
        std::uint32_t r = 0;
        if (detail::parseU32(*v, r) && r != 1) p.width.probe_every = r;
        else warnings.push_back("WHIRL_DFLASH_PROBE=" + *v + " ignored (a cycle count >= 2, or 0 for never)");
    }
    if (auto v = env("DFLASH_NMAX")) {
        std::uint32_t r = 0;
        if (detail::parseU32(*v, r) && r >= 1 && r <= 64) p.width.max_n = r;
        else warnings.push_back("WHIRL_DFLASH_NMAX=" + *v + " ignored (a draft count 1..64)");
    }
    return p;
}

// Capture rows per target forward: requested (or the drafter's window, 2048 without one), at most
// the forward's batch (Model::enableCaptureLayers clamps the same way).
inline std::uint32_t captureRows(const Params& p, std::uint32_t window, std::uint32_t max_batch) {
    const std::uint32_t want = p.capture_rows ? *p.capture_rows : (window > 0 ? window : 2048u);
    return std::min(want, max_batch);
}

// Ring rows of a slot's drafter context: the drafter's window, 2048 without one.
inline std::uint32_t ringRows(std::uint32_t window) { return window > 0 ? window : 2048u; }

// VRAM of --draft dflash beyond the target (FR-DFL P3-1: whirl-server's pre-load fit check), following
// Model::loadDrafter / enableCapture / dflashSetup and the replay-less snapshot sets.
struct MemInput {
    std::uint64_t weights = 0;  // dflashWeightBytes()
    std::uint32_t n_layer = 0, n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_ff = 0;
    std::uint32_t block = 0, window = 0, conv_k = 0, conv_group = 1, sel_topk = 0, sel_rank = 0;
    std::uint32_t n_target_layers = 0, target_embd = 0, vocab = 0;
    std::uint32_t capture_rows = 0;  // captureRows()
    std::uint32_t slots = 1;
    std::uint32_t snap_sets = 0;           // server::dflashSnapSets() (0 with replay)
    std::uint64_t snap_set_bytes = 0;      // one set: every DeltaNet layer's conv + ssm state
};
struct Mem {
    std::uint64_t weights = 0, capture = 0, rings = 0, scratch = 0, snapshots = 0;
    std::uint64_t total() const { return weights + capture + rings + scratch + snapshots; }
};
inline Mem memory(const MemInput& in) {
    using u64 = std::uint64_t;
    Mem m;
    const u64 E = in.n_embd, B = in.block, P = B > 0 ? B - 1 : 0, kvw = static_cast<u64>(in.n_head_kv) * in.head_dim;
    const u64 qw = static_cast<u64>(in.n_head) * in.head_dim, ds = 2ull * in.conv_k * (E / (in.conv_group ? in.conv_group : 1));
    const u64 ring = ringRows(in.window), K = in.sel_topk, R = in.sel_rank;
    const u64 ir = std::max<u64>(1, std::min<u64>(in.capture_rows, ring));  // dflashSetup's ingest rows
    m.weights = in.weights;
    m.capture = static_cast<u64>(in.n_target_layers) * in.capture_rows * in.target_embd * 4;
    m.rings = static_cast<u64>(in.slots) * (static_cast<u64>(in.n_layer) * 2 * ring * kvw * 4 + ring * 4);
    m.scratch = B * 4 * (2 + 5 * E + ds + 2 * qw + 2 * kvw + 2ull * in.n_ff) + P * in.vocab * 4 + P * K * 8 + P * R * 4 + P * K * K * 4 +
                (1 + P) * 4 + ir * (E + 2 * kvw + 1) * 4;
    m.snapshots = static_cast<u64>(in.snap_sets) * in.snap_set_bytes;
    return m;
}

// Positions p0 .. p0 + m - 1 go to ring rows p % ring: [start, start + first) then [0, second).
struct RingPieces {
    std::uint32_t start = 0, first = 0, second = 0;
};
inline RingPieces ringPieces(std::uint32_t p0, std::uint32_t m, std::uint32_t ring) {
    RingPieces r;
    if (ring == 0 || m == 0) return r;
    m = std::min(m, ring);
    r.start = p0 % ring;
    r.first = std::min(m, ring - r.start);
    r.second = m - r.first;
    return r;
}

// DFlash2 drafts per slot in a verify of `slots` slots: block - 1, at most the literal-draft
// limit, the snapshot sets per slot and the verify rows per slot minus the slot's own row; the
// unfused decode path snapshots row 0 only (1 draft).
inline std::uint32_t draftLimit(std::uint32_t block, std::uint32_t max_literal, std::uint32_t snap_cap, std::uint32_t vrows,
                                std::uint32_t slots, bool fused) {
    if (block < 2 || slots == 0) return 0;
    const std::uint32_t per_rows = vrows / slots > 0 ? vrows / slots - 1 : 0u;
    std::uint32_t lim = std::min({block - 1, max_literal, snap_cap / slots, per_rows});
    if (!fused) lim = std::min(lim, 1u);
    return lim;
}

// The drafter context of one server slot: positions lo .. hi - 1 form one unbroken run.
struct Range {
    std::uint32_t lo = 0, hi = 0;
};

// Drafts with DFlash2 once the run ends at pos (the next position) and is at least
// min(pos, warm) rows long (after a restore the earlier rows have no features).
inline bool ready(const Range& r, std::uint32_t pos, std::uint32_t warm) { return r.hi == pos && pos - r.lo >= std::min(pos, warm); }

// Prefix reuse at `reuse` (the slot keeps positions < reuse, the rest is rewritten): the context rows
// at positions >= reuse go, and of the run [lo, hi) only what the ring still holds stays usable: the
// ring keeps the last `ring` positions ingested (hi - ring .. hi - 1; ring 0 = no limit). When nothing
// below reuse is left (e.g. a 2048-row ring after 5000 positions, reuse 2000) the context restarts
// cold at reuse (reset: the caller drops every row) instead of claiming [lo, reuse) (FR-DFL P3-3).
struct Trim {
    Range r;
    bool reset = false;
};
inline Trim trimForReuse(Range r, std::uint32_t reuse, std::uint32_t ring) {
    Trim t;
    const std::uint32_t old_hi = r.hi;
    t.r.hi = std::min(r.hi, reuse);
    t.r.lo = std::min(r.lo, t.r.hi);
    if (ring > 0 && old_hi > ring) t.r.lo = std::max(t.r.lo, old_hi - ring);
    if (t.r.lo >= t.r.hi && r.lo < r.hi) {
        t.r = Range{reuse, reuse};
        t.reset = true;
    }
    return t;
}

// One ingest after a forward / verify of run_rows rows whose last capture_n rows were captured:
// rows r0 .. r0 + n - 1 of it are positions pos0 .. of the slot.
struct Ingest {
    bool any = false;            // some of these rows were captured
    std::uint32_t cap_row = 0;   // first capture row to ingest
    std::uint32_t n = 0;         // rows to ingest
    std::uint32_t pos = 0;       // position of cap_row
    std::uint32_t gap = 0;       // positions without features (counted, the run restarts)
};

// Plans the ingest and updates the slot's range (hi = pos0 + n whatever was captured).
inline Ingest planIngest(Range& r, std::uint32_t capture_n, std::uint32_t run_rows, std::uint32_t r0, std::uint32_t n, std::uint32_t pos0) {
    Ingest g;
    if (n == 0) return g;
    const std::uint32_t cn = std::min(capture_n, run_rows);
    const std::uint32_t c0 = run_rows - cn;  // first captured row
    const std::uint32_t a = std::max(r0, c0), b = r0 + n;
    if (a >= b) {
        g.gap = n;  // nothing of this slot's rows in the capture
        return g;
    }
    const std::uint32_t p = pos0 + (a - r0);
    if (p > r.hi) {
        g.gap = p - r.hi;
        r.lo = p;
    } else if (p < r.lo) {
        r.lo = p;
    }
    g.any = true;
    g.cap_row = a - c0;
    g.n = b - a;
    g.pos = p;
    r.hi = pos0 + n;
    return g;
}

// whirl chat / bench (one sequence): after a verify of verify_rows rows (the anchor + drafts) whose
// last capture_n rows were captured, the first keep rows stay; their features go into the drafter
// context at positions pos ... The capture may hold fewer rows than the verify (then only the kept
// rows inside it are ingested, the earlier ones are a gap): never a row before the capture buffer.
inline Ingest verifyIngest(std::uint32_t capture_n, std::uint32_t verify_rows, std::uint32_t keep, std::uint32_t pos) {
    Range r{0, pos};
    return planIngest(r, capture_n, verify_rows, 0, keep, pos);
}

}  // namespace whirl::dflash
