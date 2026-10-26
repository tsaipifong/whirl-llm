// Numerics modes: precise, balance (default), fast.
// SPDX-License-Identifier: Apache-2.0
//
// One mode per process, decided at load (KV format, kernel set and tuning are load-time):
//  - precise (opt-in, --precise): GGUF weights dequantized to f16, f16/f32 activations and accumulation in
//    prefill, f32 DeltaNet chunks, f16 KV on every card (never quantized unless the user sets
//    WHIRL_KV). If f16 KV does not fit, the context shrinks (when not given explicitly) or the load
//    stops with a message pointing to --balance / a shorter context.
//  - balance: the speed-oriented defaults (MXFP4 fp8 prefill activations, f16-WMMA DeltaNet,
//    f16 GEMM intermediates, int8 decode activations q8dec); KV q8h on dense models, f16 on MoE models.
//    Balance is defined by its allowed items (owner policy 2026-10-16, MODES-MXFP4): fp8, moefp8,
//    q8dec, f16 intermediates (h16, gdnwmma), int8 KV - near-lossless, not bitwise. MoE models get
//    the same items (fp8 + moefp8 back in MoE balance; MOE-BAL 2026-10-13 had made them fast-only).
//  - fast: balance plus aggressive gated items (4-bit KV, relaxed speculative acceptance, ...).
//    Implemented: kvq4 (4-bit KV where the code object has it: Radeon 8060S; elsewhere fast keeps
//    f16 KV until there are q4 kernels) and relaxacc. The others are listed as skipped.
// KV format: fixed per mode and model type (chooseKv), never switched by what fits: a format that
// does not fit shrinks the context (when not given) or refuses. WHIRL_KV is a debug override.
// Items not applicable to the device or model are listed as skipped, never an error.
// The pure parts (parsing, the capability table, KV / context decisions) live here for unit tests.

#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "whirl/kv_format.h"

namespace whirl::numerics {

enum class Mode : std::uint8_t { precise, balance, fast };
const char* modeName(Mode m);

enum class Item : std::uint8_t {
    // balance items (implemented)
    fp8,      // MXFP4 dense GEMM with fp8 (e4m3) activations, incl. exponent folding of the weights
    moefp8,   // MoE routed experts (MXFP4) with fp8 activations in prefill
    gdnwmma,  // f16-WMMA DeltaNet chunked prefill (vs the f32 chunk path)
    h16,      // f16 FFN / DeltaNet GEMM outputs before the elementwise ops
    kvq8,     // dense models: q8h KV (int8 K and V + f16 scale / 32, Hadamard-rotated q / k)
    specsample,  // temperature > 0: MTP drafts drawn from the draft distribution, accepted with min(1, p/q)
                 // (same output distribution as plain sampling, not the same tokens for a seed)
    // fast items (kvq4 and relaxacc implemented, the rest not yet)
    kvq4,     // 4-bit KV (gfx1151: q4 = int4 + f16 scale / 32, Hadamard-rotated q/k; dense and MoE)
    relaxacc, // relaxed / typical speculative acceptance (whirl/relax_accept.h; changes greedy output)
    headq,    // low-bit output head in the main decode
    moeskip,  // skip low-weight MoE experts
    a8,       // int8 activations for Q4_K / Q5_K / Q6_K prefill (W4A8)
    a4,       // W4A4 prefill
    n_items
};
inline constexpr std::size_t n_items = static_cast<std::size_t>(Item::n_items);
const char* itemName(Item it);
const char* itemDesc(Item it);
std::optional<Item> itemFromName(std::string_view s);
inline constexpr std::uint32_t bit(Item it) { return 1u << static_cast<unsigned>(it); }
inline constexpr std::uint32_t balance_items = bit(Item::fp8) | bit(Item::moefp8) | bit(Item::gdnwmma) | bit(Item::h16) | bit(Item::kvq8) | bit(Item::specsample);
inline constexpr std::uint32_t fast_only_items =
    bit(Item::kvq4) | bit(Item::relaxacc) | bit(Item::headq) | bit(Item::moeskip) | bit(Item::a8) | bit(Item::a4);
// MoE models: balance items that only the fast mode enables (an explicit item list still enables
// them). Empty since MODES-MXFP4 (owner 2026-10-16): MoE balance has fp8 + moefp8 again
// (MOE-BAL 2026-10-13 had moved them here). Kept as the per-model-type hook of forModel.
inline constexpr std::uint32_t moe_fast_items = 0;
// implemented items (the rest are "not yet implemented": skipped, the mode runs as balance)
inline constexpr std::uint32_t implemented_items = balance_items | bit(Item::kvq4) | bit(Item::relaxacc);

// ---- arch025-N2a (R-5): the per-variable overrides as Request fields (raw values, parsed in plan())
//
// Every WHIRL_* value that decides a numerics / execution flag of the Model is a Request field, holding the raw value
// (std::nullopt = not set, "" = set to the empty string), exactly as config::EnvSnapshot (arch025-N1, plan C6) keeps
// it. plan() parses each one with the rule its old call site had (file:line next to each field). Not wired yet: the
// loader, loadOrTune, the CLI and the server still decide the Model's flags themselves and pass none of these (N2b
// fills them and deletes the old code). numerics is in the core library, so it cannot see EnvSnapshot (whirl_model)
// or cache::fingerprintEnvKept (whirl_cache); the tests (model_tests, cache_tests) hold the two name lists together.
using Raw = std::optional<std::string>;

// Load-time values (loader + loadOrTune; CLI and server alike: Model::load / loadOrTune run for both).
struct EnvValues {
    // src/model/tune.cpp:852-982 (loadOrTuneImpl)
    Raw fp8, fp8_mask, g8t, gdn_wmma, act_fuse, gemmh, gemmhq, attn_kx, attn_kg, attn_dqf, attn_seg, gdn_ba, gdn_in2, q4_relaxed,
        ffn_h16, gdn_strix, gdn_strix_min, kv, q8dec, relax, relax_k, relax_alpha, relax_eps, relax_delta, spec_sample;
    // src/model/loader.cpp:550-553
    Raw moe_fp8, moe_mxw, moe_wide, moe_rbf;
    // numerics::noteSwitches (k_numerics_switches, same order): recorded as overrides only
    std::array<Raw, 13> switches{};

    struct Entry {
        const char* name;  // without the WHIRL_ prefix
        Raw EnvValues::* field;
    };
    static constexpr std::size_t n_fields = 29;
    // every Raw field once (switches are k_numerics_switches); the reflection test checks the count against sizeof
    static const std::array<Entry, n_fields>& names();
    // by name, switches included (nullptr for an unknown name)
    const Raw* find(std::string_view name) const;
    Raw* find(std::string_view name);
};

// Which front end applies its after-load knobs, and with which parse rules. none: no front end (host-only tests,
// --dump-only cells: the S0b-2 dumps are taken after loadOrTune, before these knobs).
enum class Front : std::uint8_t { none, cli, server };

// After-load knobs of the front ends: src/cli/main.cpp:1030-1048 (applyRuntimeEnv) and :1340 (GDN_V0, chat only);
// src/server/server_main.cpp:1040-1120. A front end fills only the names it applies (the server sends none of the
// CLI-only ones: DBG, GEMV_MAX, NO_GRAPH, GV_GROUP, GV_NMAX).
struct RuntimeValues {
    Raw float_gemv, dbg, gemv_max, no_graph, no_fuse, draft_window, draft_window_min, naive_attn, attn_wide, attn_dq4, dq4_dyn,
        attn_ws4, moe_bn, gdn_seq, gv_group, gv_nmax, gdn_v0, wide_verify, head_chunk;
    struct Entry {
        const char* name;
        Raw RuntimeValues::* field;
    };
    static constexpr std::size_t n_fields = 19;
    static const std::array<Entry, n_fields>& names();
};

// The requested mode and items.
struct Request {
    Mode mode = Mode::precise;
    std::uint32_t items = 0;  // requested lossy items
    bool custom = false;      // an explicit item list (--balance=fp8,kvq8)
    std::string source = "default";
    std::uint32_t moved = 0;  // items the mode would request but forModel moved to fast for this model
    // arch025-N2a: per-variable overrides (empty: the mode's choice; plan() then decides exactly as before N2a)
    EnvValues env;
    // true: plan() also does loadOrTune's part on the items (per-item overrides, WHIRL_Q8DEC, the relaxacc parameter
    // note, noteSwitches; tune.cpp:939-982). false (every caller until N2b): the items exactly as before N2a, and
    // loadOrTune adds that part itself. Plan::flags are computed either way.
    bool load_overrides = false;
    Front front = Front::none;
    RuntimeValues runtime;  // read only when front != none
    bool has(Item it) const { return (items & bit(it)) != 0; }
};

// "precise" | "balance" ("balanced") | "fast", optionally ":item,item" ("=item,item" for the
// --balance= / --fast= flags is split off by the caller). Throws std::invalid_argument.
Request parseMode(std::string_view spec, std::string source);
// The request for a model type: balance (no explicit item list) on a MoE model drops
// moe_fast_items (recorded in `moved`, reported as skipped; empty since MODES-MXFP4). Precise, fast, custom lists and dense
// models are unchanged. Idempotent; the loader applies it
// once the model type is known.
Request forModel(const Request& r, bool moe);
// The mode's default decode / verify / small-batch activation form for a model type: true = int8
// (q8dec). precise -> f16; balance and fast -> int8 on dense and MoE models alike (MOE-BAL-3).
bool q8decDefault(const Request& r, bool moe);
// Command line (flags already collected: --mode V / --precise / --balance[=items] / --fast[=items];
// several flags -> invalid_argument) first, else WHIRL_MODE, else precise.
Request resolve(std::optional<std::string> cli_spec, std::optional<std::string> env_mode);
// One command-line argument: true (and *spec set) if it is a mode flag. "--mode" takes the next
// argument (*takes_next = true). Throws std::invalid_argument for a malformed flag.
bool modeFlag(std::string_view arg, std::string* spec, bool* takes_next);

// What the device and model support (filled by the loader; plain bools for tests).
struct Target {
    bool gfx1151 = false;            // Radeon 8060S code object
    bool has_mxfp4 = false;          // any MXFP4 tensor
    bool moe = false;
    bool moe_mxfp4 = false;          // routed experts are MXFP4
    bool k_gemm8 = false;            // fp8 WMMA GEMMs present
    bool k_gemm8_moe = false;        // MXFP4 x fp8 grouped expert GEMMs present
    bool k_gdn_wmma = false;         // f16-WMMA DeltaNet kernels present
    bool k_kv_q8v = false, k_kv_q8h = false;
    bool mtp = true;                 // MTP head present
    bool k_spec_sample = true;       // draft_sample_rows kernel present
    bool k_kv_q4 = false;      // q4 KV kernels present (gfx1151)
    bool kv_explicit = false;  // WHIRL_KV set to a format (the KV items then do not choose)
    bool fast_kv_q8h = false;  // fast takes q8h for kvq4 on dense models (KvCaps::fast_q8h, gfx1151)
    // ---- arch025-N2a: what Plan::flags needs besides the above. arch025-N2b: the gdn_strix input is gone; plan()
    // decides flags.gdn_strix from the fields below (gdnwmma runs as the gfx1151 strix-llama.cpp chunked prefill, GDN-8S).
    bool strix_profile = false;  // device::Profile::gdn_strix (gfx1151 row)
    bool k_gdn_sx = false;       // gdn_sx_prep and gdn_sx_scan present
    bool gdn_strix_shape = false;  // d_state 128, head V 128, n_k_heads > 0, n_v_heads % n_k_heads == 0, ratio <= 4
    bool k_gemm8t = false;       // the whole fragment-tiled fp8 set (Model::useTiledFp8, forward.cpp:269-277)
    bool k_gdn_v0 = false;       // gdn_step_norm_v0 present (WHIRL_GDN_V0)
};

// gdnwmma / h16 for a model without MXFP4 tensors (KG-1): dense models on the R9700 only
// (MoE non-MXFP4 models and the Radeon 8060S keep the f32 paths).
inline bool denseQuantRelaxed(const Target& t) { return !t.has_mxfp4 && !t.moe && !t.gfx1151; }

struct ItemState {
    Item item{};
    bool requested = false;
    bool moved = false;    // not requested: a fast-only item for this model type (Request::moved)
    bool enabled = false;  // requested (or user-forced) and applicable
    std::string note;      // enabled: detail; skipped: reason
};

// ---- arch025-N2a: the concrete execution flags (the Model's numerics members, one field each)
//
// X(kind, name, default, dump key): kind B = bool, U = std::uint32_t, F = float; default = the Model member's
// initializer (include/whirl/model.h; gv_group / gv_nmax: src/model/config.cpp:23-24); dump key = the "D exec:" key of
// WHIRL_DUMP_EXEC (src/model/exec_dump.cpp), "" = not in the dump (gdn_v0 swaps a kernel: the launch log pins it).
// Every flag is in bitsKey(): none is proven not to change the bits (use_graph replays the same launches, but that is
// not pinned at level 5; the CLI and the server differ there, the tier fingerprint is the server's only).
// Not here (decided elsewhere, listed in tests/unit numerics_flags_scope): the KV format and its bools (kv_q8, kv_rot,
// kv_kf16, kv_q4, kv_mode, fd_splits: numerics::chooseKv / resolveFdSplits, N3), the tune results (gemv_r, gemv_w,
// tune table, WHIRL_GEMV_R / _W / _WH: config.cpp applyGemvEnv), per-forward state (out_h16, mm_class, small_max,
// head_phase, mtp_batch, mrope_on, rpos_rows) and sizes.
#define WHIRL_NUMERICS_FLAGS(X)                                  \
    X(B, use_graph, true, "use_graph")                           \
    X(B, no_fuse, false, "no_fuse")                              \
    X(U, dbg_flags, 0, "dbg_flags")                              \
    X(B, naive_attn, false, "naive_attn")                        \
    X(B, attn_wide, true, "attn_wide")                           \
    X(B, attn_dq4, true, "attn_dq4")                             \
    X(B, dq4_dyn, true, "dq4_dyn")                               \
    X(B, attn_ws4, true, "attn_ws4")                             \
    X(B, wide_verify, true, "wide_verify")                       \
    X(B, moe_wide, false, "moe_wide")                            \
    X(U, head_chunk, 248320, "head_chunk")                       \
    X(B, float_gemv, false, "float_gemv")                        \
    X(B, prec_dec, false, "prec_dec")                            \
    X(U, gemv_max, 16, "gemv_max")                               \
    X(B, fp8_prefill, true, "fp8_prefill")                       \
    X(U, fp8_mask, 7, "fp8_mask")                                \
    X(B, relax_on, false, "relax.on")                            \
    X(U, relax_k, 4, "relax.k")                                  \
    X(F, relax_alpha, 0.1f, "relax.alpha")                       \
    X(F, relax_eps, 0.09f, "relax.eps")                          \
    X(F, relax_delta, 0.3f, "relax.delta")                       \
    X(B, gdn_chunked, true, "gdn_chunked")                       \
    X(B, gdn_wmma, false, "gdn_wmma")                            \
    X(B, gdn_strix, false, "gdn_strix")                          \
    X(U, gdn_strix_min, 256, "gdn_strix_min")                    \
    X(B, act_fuse, true, "act_fuse")                             \
    X(B, attn_kx_on, true, "attn_kx_on")                         \
    X(B, gdn_ba_on, true, "gdn_ba_on")                           \
    X(B, gdn_in2_on, true, "gdn_in2_on")                         \
    X(B, attn_dqf_on, true, "attn_dqf_on")                       \
    X(U, attn_seg, 0, "attn_seg")                                \
    X(B, attn_kg_on, true, "attn_kg_on")                         \
    X(B, ffn_h16, false, "ffn_h16")                              \
    X(B, h16_dq, false, "h16_dq")                                \
    X(B, g8t, false, "g8t")                                      \
    X(B, gemmh_on, true, "gemmh_on")                             \
    X(B, gemmhq_on, false, "gemmhq_on")                          \
    X(U, draft_window, 16384, "draft_window")                    \
    X(U, draft_window_min, 65536, "draft_window_min")            \
    X(U, moe_bn_force, 0, "moe_bn_force")                        \
    X(B, moe_fp8, true, "moe_fp8")                               \
    X(B, moe_mxw, true, "moe_mxw")                               \
    X(B, moe_rbf, true, "moe_rbf")                               \
    X(B, gv_group, true, "gv_group")                             \
    X(U, gv_nmax, 0, "gv_nmax")                                  \
    X(B, gdn_v0, false, "")

struct Flags {
#define WHIRL_NF_B bool
#define WHIRL_NF_U std::uint32_t
#define WHIRL_NF_F float
#define WHIRL_NF_FIELD(k, n, d, key) WHIRL_NF_##k n = d;
    WHIRL_NUMERICS_FLAGS(WHIRL_NF_FIELD)
#undef WHIRL_NF_FIELD
#undef WHIRL_NF_B
#undef WHIRL_NF_U
#undef WHIRL_NF_F
    bool operator==(const Flags&) const = default;
};

enum class FlagKind : std::uint8_t { b, u, f };
struct FlagInfo {
    const char* name;
    const char* dump_key;  // "" = not in WHIRL_DUMP_EXEC
    FlagKind kind;
    bool Flags::* b;
    std::uint32_t Flags::* u;
    float Flags::* f;
};
// every Flags field once, in declaration order (generated from WHIRL_NUMERICS_FLAGS)
const std::vector<FlagInfo>& flagTable();
// the field's value as bitsKey / the dump print it: bool 0/1, u32 decimal, float {:.9g} (exec_dump.cpp:53)
std::string flagValue(const Flags& fl, const FlagInfo& fi);

struct Plan {
    Mode mode = Mode::precise;
    std::string source = "default";  // where the mode came from: default (balance), command line, WHIRL_MODE
    std::array<ItemState, n_items> items{};
    std::vector<std::string> overrides;  // per-item environment variables that changed the mode's choice
    bool lossy_override = false;         // an override made precise lossy
    // decode / verify / small-batch matmul activations: int8 (q8_1 class; balance, fast) or f16
    // into the f16 GEMM with f32 accumulation (precise). Set by the loader (setQ8dec).
    bool q8dec = true;
    bool q8dec_default = true;  // the mode's choice (q8decDefault) before WHIRL_Q8DEC
    // arch025-N2a: the Model's execution flags this plan decides; arch025-N2b: the Model reads them (Model::nf())
    Flags flags;
    bool on(Item it) const { return items[static_cast<std::size_t>(it)].enabled; }
    // Everything here that decides the bits of the output, as "key=value;" in a fixed order: the mode, every item's
    // enabled state, q8dec and every Flags field (flagTable order). Descriptions (notes, overrides, source) are not in
    // it. Not used by the tier fingerprint yet (freeze, plan O6).
    std::string bitsKey() const;
};

// Capability table: requested items -> enabled / skipped with the reason. arch025-N2a: also every execution flag
// (Plan::flags), from the Request's per-variable overrides (r.env, and r.runtime for r.front), with the rules of
// loader.cpp:548-553 / :743, tune.cpp:850-982, cli main.cpp:1030-1048 / :1340 and server_main.cpp:1040-1120. With no
// overrides set the items, notes and q8dec are exactly what plan() gave before N2a (loadOrTune applies its own
// overrides afterwards until N2b). Throws std::invalid_argument / std::out_of_range where loadOrTune threw
// (WHIRL_RELAX_K / _ALPHA / _EPS / _DELTA that std::stoul / std::stof reject, with relaxacc on).
Plan plan(const Request& r, const Target& t);
// arch025-N2b: the front end's after-load knobs (the CLI's applyRuntimeEnv, the server's knob block) on a plan made with
// Front::none: the same rules plan() applies for r.front / r.runtime (plan(r) with a front == plan(r without) then this).
// k_gdn_v0 = Target::k_gdn_v0. Front::none: no change.
void applyFront(Plan& p, Front f, const RuntimeValues& v, bool k_gdn_v0);
// Per-item override from the environment: `forced` = the effective value after the variable.
// Records "WHIRL_X=v: item on (lossy, user-requested)" or "... item off" when it differs.
void applyOverride(Plan& p, Item it, const char* var, const std::string& value, bool forced, bool applicable);

// The decode activation form actually used: q8 = int8 (default: q8dec_default). env = the
// WHIRL_Q8DEC value ("" = unset); a value that changes the mode's default is recorded as an
// override (int8 in precise: lossy, user-requested).
void setQ8dec(Plan& p, bool q8, const std::string& env);

// An item the run cannot use after all (e.g. relaxacc with --draft dflash): skipped with `why`.
void skipItem(Plan& p, Item it, const std::string& why);

// WHIRL_* A/B switches that change which kernels run (WHIRL_FD_SPLITS, WHIRL_GEMV_MAX, ...):
// each one set is recorded as an override ("+overrides" label), so the log and /props show that
// the run's bits are not the mode's tested paths. get(name) reads WHIRL_<name>.
extern const char* const k_numerics_switches[];
void noteSwitches(Plan& p, const std::function<std::optional<std::string>(const char*)>& get);

// "numerics: balance - enabled: fp8 (...), ...; skipped: gdnwmma (...); always: q8dec (...)"
std::string logLine(const Plan& p);
// JSON object for GET /props ("numerics"); kv = the KV format name
std::string propsJson(const Plan& p, std::string_view kv);
// the mode label: "precise", "precise+overrides", "balance", "fast (runs as balance)"
std::string modeLabel(const Plan& p);

// ---- KV format (BAL-Q8): one fixed format per mode x model type, CLI and server alike

// (the format is whirl::KvFormat, whirl/kv_format.h; names kvFormatName / kvFormatFromName, arch025-N3)
// KV kernels in the code object (f16 and q8 are always there)
struct KvCaps {
    bool q8h = false;
    bool q8v = false;
    bool q4 = false;
    // RC6: this GPU's fast mode takes q8h instead of q4 KV on dense models (gfx1151: the q8h decode
    // attention is faster than q4's at 32k-128k and the error 4-10x smaller, FAST-8S; the R9700 keeps q4)
    bool fast_q8h = false;
};
// The KV caps the pre-load VRAM estimates assume, before any code object is loaded (Model::load's UMA check,
// the server's dGPU and UMA estimates; arch025-N3: were the literals {true, true, true(, true)} at each site):
// every KV kernel (both shipped code objects have q8h / q8v / q4 since FAST-1c / FC-1c) and fast_q8h on an
// integrated GPU (the Radeon 8060S, device::Profile::numerics_8060s). whirl-tests testKvEstimateCaps checks the
// choice made with these equals the choice Model::load makes with each GPU's real caps.
constexpr KvCaps kvCapsEstimate(bool integrated) { return KvCaps{true, true, true, integrated}; }
struct KvChoice {
    KvFormat kv = KvFormat::f16;
    bool debug = false;  // WHIRL_KV chose it (debug override: no fit check, sizing as before)
    std::string why;     // "balance, dense model: q8h"
};
// The three pre-load KV estimates (arch025-N3-r1, review C1: each site wrote out caps -> chooseKv -> "D kv:" line
// itself). The site decides both its "D kv:" name and which GPU kind it sizes for: the server's dGPU branch runs on
// a discrete GPU, Model::load's UMA check and the server's UMA branch on an integrated one (the Radeon 8060S).
enum class KvEstSite : std::uint8_t { loader_uma, server_dgpu, server_uma };
constexpr std::string_view kvEstSiteName(KvEstSite s) {
    switch (s) {
        case KvEstSite::loader_uma: return "loader.uma_est";
        case KvEstSite::server_dgpu: return "server.est_dgpu";
        case KvEstSite::server_uma: return "server.est_uma";
    }
    return "?";
}
constexpr bool kvEstSiteIntegrated(KvEstSite s) { return s != KvEstSite::server_dgpu; }
// The server estimates' WHIRL_KV reading (both branches): unset, "auto" and a name kvFormatFromName does not know
// are no request; Model::load refuses the unknown name afterwards (kvRequestFromEnv), as before arch025-N3.
// Model::load's own UMA estimate reads the request strictly (loader.cpp kvRequestOf), the same as its load.
std::optional<KvFormat> kvEstimateRequest(const std::optional<std::string>& kv_env);
// The KV format a pre-load estimate sizes for: chooseKv(r, moe, kvCapsEstimate(kvEstSiteIntegrated(site)), req),
// with one "D kv: site=<kvEstSiteName(site)>" line under WHIRL_DUMP_KV (dump::kv). whirl-tests testKvEstimateCaps
// checks it per site against the load's choice with that site's GPU caps.
KvFormat kvEstimate(KvEstSite site, const Request& r, bool moe, std::optional<KvFormat> req);
// The request's KV format: WHIRL_KV (forced) wins; else kvq4 -> q4 (where the kernels exist,
// else f16: R9700 has no q4 KV kernels yet, and int8 is no stand-in), dense and MoE, except a dense
// model where caps.fast_q8h (gfx1151) -> q8h; else kvq8 on
// a dense model -> q8h (q8 where the code object has no q8h kernels);
// else f16 (precise, MoE in balance). Independent of free memory and card size.
KvChoice chooseKv(const Request& r, bool moe, const KvCaps& caps, std::optional<KvFormat> forced);

// Wording of the "does not fit" messages for the chosen format
struct KvFitLabel {
    std::string mode = "precise";
    std::string fmt = "f16";
    std::string alt = "--balance allows int8 KV";  // a lower-precision alternative the user may choose ("" = none)
};
KvFitLabel kvFitLabel(const Request& r, bool moe, KvFormat kv);

// ---- KV fit (every mode: the chosen format only, never a lower-precision one)

// CLI / bench: f16 KV for `ctx` tokens next to `later` bytes in `free` bytes.
// Fits -> ctx unchanged. Else when ctx was not given explicitly: the largest page-aligned context
// that fits, if >= min_ctx (shrunk = true); else refuse (with the message).
struct CtxFit {
    std::uint32_t ctx = 0;
    bool shrunk = false;
    bool refuse = false;
    std::string msg;
};
CtxFit cliCtxFit(std::uint32_t ctx, bool ctx_explicit, std::uint32_t min_ctx, std::uint64_t free, std::uint64_t later,
                 std::uint64_t per_token, std::uint32_t page, const KvFitLabel& lb = {});

// Server: f16 pool from the usable VRAM. pool_req = --ctx (explicit) or the UMA default pool;
// slot_ctx = per-request context; *_explicit = given on the command line.
struct PoolFit {
    std::uint32_t pool = 0;
    std::uint32_t slot_ctx = 0;
    bool shrunk_pool = false;
    bool shrunk_ctx = false;
    bool refuse = false;
    std::string msg;  // warning (shrunk) or error (refuse)
};
PoolFit serverPoolFit(std::uint64_t avail, std::uint64_t per_token, std::optional<std::uint32_t> pool_req, bool pool_explicit,
                      std::uint32_t slot_ctx, bool slot_explicit, std::uint32_t page, std::uint32_t pool_cap, const KvFitLabel& lb = {});

}  // namespace whirl::numerics
