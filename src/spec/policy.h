// MTP / n-gram speculative-decoding policy (arch025-M4): the per-device MTP defaults, the draft-head vocabulary
// choice, draft-count control (DraftAccept, DraftTiming, pickDrafts), the T4-1 per-slot draft allocation (CycleCost,
// SlotAccept, allocDrafts) and prompt-lookup drafting (Ngram, NgramPolicy).
// SPDX-License-Identifier: Apache-2.0
//
// Host-only library whirl_spec (cmake/spec.cmake): the server core and the model library both link it; nothing here
// needs HIP or the Model class (like llama.cpp's common/speculative.* and ngram-*.* next to libllama). The
// declarations were moved verbatim from include/whirl/model.h and include/whirl/model_types.h, in their order; the
// namespace stays whirl::qwen35.

#pragma once

#include "whirl/device_profile.h"  // hip::Arch, device::k_default_arch (mtpDefaults)
#include "whirl/hip.h"             // hip::Arch (mtpDefaults)
#include "whirl/model_types.h"     // max_ng_drafts

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace whirl::qwen35 {

// ---------------------------------------------------------------------------
// MTP draft-count control and n-gram (prompt-lookup) drafting

struct MtpDefaults {
    std::uint32_t drafts;
    float p_min;
    std::uint32_t adapt;
    bool automatic;
    std::uint32_t ngram_min = 3;  // n-gram drafting: shortest history suffix match (WHIRL_NGRAM_MIN)
    float ngram_slope = 0.015f;   // NgramPolicy::verify_slope (WHIRL_NGRAM_SLOPE)
};
// Per-device defaults: gfx1201 (R9700) is the reference; gfx1151 (Radeon 8060S, UMA,
// bandwidth-bound decode) has its own values measured on that device.
MtpDefaults mtpDefaults(bool moe, hip::Arch arch = device::k_default_arch);

// ---- draft-head vocabulary subset (T5-1b): WHIRL_DRAFT_VOCAB = 48k | 64k | <file> | off | N
// File for a WHIRL_DRAFT_VOCAB value: "48k" / "64k" / ... -> <dir>/subset_48k.bin, anything else
// that is not "off" or a plain number (the old first-N-rows experiment) is a path; else nullopt.
std::optional<std::string> draftVocabFile(std::string_view spec, const std::string& dir);
// uint32 little-endian token ids (throws on a missing file or a size that is not a multiple of 4)
std::vector<std::uint32_t> readDraftVocab(const std::string& path);
// Checks a subset (strictly ascending, every id < n_rows; throws otherwise) and adds the missing
// `required` ids (special and byte tokens), counting them in *added.
std::vector<std::uint32_t> draftVocabIds(std::span<const std::uint32_t> ids, std::uint32_t n_rows,
                                         std::span<const std::uint32_t> required, std::uint32_t* added = nullptr);
// directory of the running executable ("" if unknown)
std::string exeDirectory();
// What WHIRL_DRAFT_VOCAB selects. Unset = the embedded 64k subset by default (only applied to a
// model draftVocabDefaultFits accepts); "64k" = the embedded subset explicitly; "off" / "0" / "" =
// the full vocabulary; N = the first N rows (old experiment); "48k" etc. = <dir>/draft_vocab/
// subset_48k.bin; anything else = a file path.
struct DraftVocabChoice {
    enum class Kind { full, first_n, embedded_64k, file } kind = Kind::full;
    std::uint32_t n = 0;
    std::string file;
    bool by_default = false;  // WHIRL_DRAFT_VOCAB unset
};
DraftVocabChoice draftVocabChoice(const std::optional<std::string>& env_value, const std::string& dir);
// vocabulary size of the qwen35 tokenizer the embedded subset was built for
inline constexpr std::uint32_t draft_vocab_embedded_n_vocab = 248320;
// the default (embedded 64k) subset is used only for dense qwen35 models with that vocabulary
inline bool draftVocabDefaultFits(bool moe, std::uint32_t n_vocab) { return !moe && n_vocab == draft_vocab_embedded_n_vocab; }
// Whether the subset can be applied to this model (arch025-N10 / N6, lane 1 leg 6: the decision whirl.exe and
// whirl-server both wrote, the shape of llama.cpp common_speculative_are_compatible: the decision here, the message and
// what to do in each front end). none: full / first_n (nothing to apply here; first_n is applied by the caller at its
// own time); not_fit: the embedded subset does not fit this vocabulary (by_default: draftVocabDefaultFits, explicit:
// the 248320-token vocabulary); no_d2: it fits but the model has no 2-bit draft head; subset: build and set it.
enum class DraftVocabGate { none, not_fit, no_d2, subset };
DraftVocabGate draftVocabGate(const DraftVocabChoice& dv, bool moe, std::uint32_t n_vocab, bool draft_d2);
// The ids the subset must keep: every special token and every byte token (the tokenizer's specials() / byteToken(b),
// -1 = none), in that order.
template <class Tok>
std::vector<std::uint32_t> draftVocabRequired(const Tok& tok) {
    std::vector<std::uint32_t> req;
    for (const auto t : tok.specials())
        if (t >= 0) req.push_back(static_cast<std::uint32_t>(t));
    for (int b = 0; b < 256; ++b)
        if (tok.byteToken(b) >= 0) req.push_back(static_cast<std::uint32_t>(tok.byteToken(b)));
    return req;
}
// The subset of a file / embedded choice: the ids (readDraftVocab(dv.file), else embedded(): the embedded 64k list,
// qwen35::embeddedDraftVocab64k in the model library) through draftVocabIds, the added count and the log label.
struct DraftVocabSubset {
    std::vector<std::uint32_t> ids;
    std::uint32_t added = 0;
    std::string label;
};
DraftVocabSubset draftVocabSubset(const DraftVocabChoice& dv, std::vector<std::uint32_t> (*embedded)(), std::uint32_t n_rows,
                                  std::span<const std::uint32_t> required);

// whirl-server's max drafts per cycle by number of decoding slots (index = slots; arch025-N10 leg 6: was server_main.cpp's
// defaultBatchDrafts, whose `wide` argument was unused): dense {0, 8, 7, 4, 3, 2, 1, 1, 1, 0...}, MoE 1 up to 8 slots.
// The cost model picks within the cap.
using BatchDrafts = std::array<std::uint32_t, gdn_max_seg + 1>;
BatchDrafts defaultBatchDrafts(bool moe);
// WHIRL_MTP_BATCH_DRAFTS=d1,d2,...: entry k (from 1) for k slots; empty fields are skipped, a value that is not a number
// is 1, each is capped at max_drafts, the slots after the last value keep the last value.
BatchDrafts batchDraftsFrom(BatchDrafts base, const std::string& text);
// The automatic MTP draft count (cost model) on or off, whirl-server's and the CLI's one rule (arch025-lane1 leg 8: each
// wrote it, 5728296 server_main.cpp:744 / cli/decode.cpp:458-463): WHIRL_MTP_ADAPT set -> on exactly when it is "auto";
// else the device default (MtpDefaults::automatic), unless the draft count was given (an explicit count stays fixed).
bool mtpAutoMode(bool def_automatic, bool drafts_given, bool adapt_set, bool adapt_auto);
// whirl-server --draft dflash and the MTP-only settings (arch025-lane1 leg 9; were ServerRun::makeEngine rules).
// relaxacc relaxes MTP drafts only (whirl/dflash_policy.h relaxApplies): with DFlash2 it is turned off (off), with the
// info line (note) and the reason numerics::skipItem records (skip_why); otherwise nothing changes.
struct DflashRelax {
    bool off = false;
    std::string note;
    std::string skip_why;
};
DflashRelax dflashRelax(bool use_dflash, bool relax_on);
// With --draft dflash: WHIRL_SLOT_DRAFTS (the per-slot MTP draft allocation) is 0, with a warning when it was set
// (warn), and an info line when the draft-head settings WHIRL_DRAFT_VOCAB / _WINDOW / _WINDOW_MIN are set (note). Only
// called with --draft dflash.
struct DflashMtpOnly {
    std::uint32_t slot_drafts = 0;
    std::string warn;
    std::string note;
};
DflashMtpOnly dflashMtpOnly(std::uint32_t slot_drafts, bool draft_head_env);

struct DraftAccept {
    std::array<float, max_ng_drafts> alpha;
    explicit DraftAccept(float a = 0.75f) { alpha.fill(a); }
    void update(std::uint32_t nd_dev, std::uint32_t acc) { updateRate(nd_dev, acc, 0.1f); }
    void updateRate(std::uint32_t nd_dev, std::uint32_t acc, float r);
    float expected(std::uint32_t nd) const;
};

struct DraftTiming {
    std::array<float, max_ng_drafts + 1> t{};
    std::array<std::uint32_t, max_ng_drafts + 1> n{};
    float prior_slope = 0.12f;
    bool skip_first = false;
    std::array<bool, max_ng_drafts + 1> seen{};
    void update(std::uint32_t nd, float ms);
    std::optional<float> estimate(std::uint32_t nd) const;
};

// Draft count in [1, max] maximizing sum(E) / T (see the prototype notes).
std::uint32_t pickDrafts(std::span<const DraftAccept* const> accepts, const DraftTiming& timing, std::uint32_t max,
                         std::uint32_t cycle, std::uint32_t prev);

// ---- per-slot draft allocation (T4-1): pure model code, not wired into the engine yet

// One verify cycle as the cost model sees it.
struct CycleFeat {
    float rows = 0;    // R: verify rows of all sequences
    float steps = 0;   // S: sequential MTP draft steps (the longest MTP draft)
    float row_ctx = 0; // sum over sequences of rows_i * context_i, in k tokens
};

// Cycle time for one number of decoding slots: T ~ a + b R + c S + e [R > 16] + f sum(rows_i ctx_i),
// fitted by recursive least squares with forgetting factor 0.98.
class CycleCost {
public:
    static constexpr std::uint32_t n_par = 5;
    static constexpr double forget = 0.98;
    CycleCost() { reset(); }
    void reset();
    // seed from the uniform-allocation timing of `slots` decoding slots (DraftTiming points or its prior)
    void prior(const DraftTiming& tm, std::uint32_t slots, float ctx_k = 0);
    void observe(const CycleFeat& x, float ms);
    float predict(const CycleFeat& x) const;
    std::uint32_t samples() const { return n_obs; }

private:
    static std::array<double, n_par> feat(const CycleFeat& x);
    void rls(const std::array<double, n_par>& z, double y);
    std::array<double, n_par> th{};
    std::array<std::array<double, n_par>, n_par> P{};
    std::uint32_t n_obs = 0;
};

// Per-slot acceptance with the T4-1 additions on top of DraftAccept: faster start (r = 0.2 for 16
// cycles), unobserved positions drift towards the last observed rate x 0.95 (rate 0.02; a fully
// accepted chain is censored, not a ceiling, so it drifts without the 0.95), observation counts
// for shrinkage towards a global pool.
struct SlotAccept {
    static constexpr std::uint32_t n_cap = 32;
    static constexpr float pool_weight = 4.0f;
    static constexpr float drift_rate = 0.02f;
    static constexpr float drift_decay = 0.95f;
    DraftAccept a;
    std::array<std::uint32_t, max_ng_drafts> n{};
    std::uint32_t cycles = 0;
    explicit SlotAccept(float a0 = 0.75f) : a(a0) {}
    void observe(std::uint32_t nd_dev, std::uint32_t acc);
    // alpha_eff[k] = (n_k alpha_k + 4 alpha_pool[k]) / (n_k + 4)
    DraftAccept effective(const DraftAccept& pool) const;
};

struct AllocSlot {
    const DraftAccept* acc = nullptr; // MTP slot: acceptance (alpha_eff); unused for n-gram slots
    std::uint32_t cap = 1;            // max drafts of this slot
    float ctx_k = 0;                  // context length, k tokens
    std::uint32_t ng_rows = 0;        // > 0: n-gram slot with this many verify rows (fixed)
    float ng_e = 0;                   // n-gram slot: expected tokens of its fixed draft
};

struct AllocBudget {
    std::uint32_t rows = 16;           // verify rows available in total
    std::uint32_t snaps = 0xffffffffu; // snapshot sets free for drafts (0xffffffff: replay, no limit)
};

struct AllocResult {
    std::vector<std::uint32_t> d;     // drafts per slot (0 for n-gram slots)
    float score = 0;                  // sum(E) / T (tokens per ms)
    std::uint32_t rounds = 0;         // local-search rounds used
    bool uniform = true;              // true: the uniform allocation u was returned
};

// Per-slot draft counts maximizing sum(E_i) / T, starting from the uniform count `uniform`
// (capped per slot): local search over +1 / -1 / swap moves, at most 16 rounds, subject to
// sum(d) <= min(rows free, snapshots free), 1 <= d_i <= cap_i and each slot keeping at least
// (1 - x) of its uniform-allocation throughput. Keeps `prev` when it scores >= 0.98 of the best,
// and returns u unless the best beats it by more than 1%, with < 2 MTP slots, < 16 cost samples,
// or acceptances so close that u is already optimal.
AllocResult allocDrafts(std::span<const AllocSlot> slots, const CycleCost& cost, AllocBudget budget, std::uint32_t uniform,
                        std::span<const std::uint32_t> prev, float x = 0.03f);

class Ngram {
public:
    static constexpr std::uint32_t none = 0xffffffffu;
    static constexpr std::size_t long_len = 12;
    static constexpr std::uint32_t max_cands = 16;
    static constexpr std::uint32_t max_match = 256;
    struct Match {
        std::size_t n = 0;
        std::uint32_t mlen = 0;
    };
    // token id -> the same text with CR LF as LF (empty: none)
    std::vector<std::uint32_t> norm;

    void reset();
    Match lookup(std::span<const std::uint32_t> toks, std::uint32_t min_match, std::span<std::uint32_t> out);

private:
    std::uint32_t nt(std::uint32_t t) const { return t < norm.size() ? norm[t] : t; }
    std::unordered_map<std::uint64_t, std::uint32_t> map_s, map_l;
    std::vector<std::uint32_t> prev_s, prev_l, ntoks;
    std::size_t indexed = 2;
    std::size_t cur_q = 0, cur_n = 0;
    std::size_t gen_from = 0;
    bool gen_cr = false;
};

class NgramPolicy {
public:
    Ngram ng;
    // prior cost of one more verify row relative to a 1-draft MTP cycle, used until the n-gram
    // cycles are timed (per device: MtpDefaults::ngram_slope; set it before reset())
    float verify_slope = 0.015f;
    std::array<DraftAccept, 3> acc = priorAcc();
    DraftTiming timing = priorTiming(verify_slope);
    static constexpr float acc_rate = 0.2f;

    static std::uint32_t bucket(std::uint32_t mlen) { return mlen >= 24 ? 2 : (mlen >= 8 ? 1 : 0); }
    void reset();
    void observe(std::span<const std::uint32_t> toks);
    Ngram::Match propose(std::span<const std::uint32_t> toks, std::uint32_t min_match, std::span<std::uint32_t> out);
    std::uint32_t choose(std::size_t n, std::uint32_t mlen, float mtp_score, const DraftTiming& mtp_timing) const;

private:
    struct Pending {
        std::size_t at = 0;
        std::uint32_t n = 0, b = 0;
        std::array<std::uint32_t, max_ng_drafts> toks{};
    };
    static std::array<DraftAccept, 3> priorAcc() { return {DraftAccept(0.35f), DraftAccept(0.7f), DraftAccept(0.85f)}; }
    static DraftTiming priorTiming(float slope) {
        DraftTiming t;
        t.prior_slope = slope;
        t.skip_first = true;
        return t;
    }
    std::array<Pending, 16> pend{};
    std::uint32_t n_pend = 0;
};

}  // namespace whirl::qwen35
