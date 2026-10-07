// Numerics modes: precise, balance (default), fast.
// SPDX-License-Identifier: Apache-2.0
//
// One mode per process, decided at load (KV format, kernel set and tuning are load-time):
//  - precise (opt-in, --precise): GGUF weights dequantized to f16, f16/f32 activations and accumulation in
//    prefill, f32 DeltaNet chunks, f16 KV on every card (never quantized unless the user sets
//    WHIRL_KV). If f16 KV does not fit, the context shrinks (when not given explicitly) or the load
//    stops with a message pointing to --balance / a shorter context.
//  - balance: the speed-oriented defaults (MXFP4 fp8 prefill activations, MoE expert fp8,
//    f16-WMMA DeltaNet, f16 GEMM intermediates); KV q8h on dense models, f16 on MoE models.
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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

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
// implemented items (the rest are "not yet implemented": skipped, the mode runs as balance)
inline constexpr std::uint32_t implemented_items = balance_items | bit(Item::kvq4) | bit(Item::relaxacc);

// The requested mode and items.
struct Request {
    Mode mode = Mode::precise;
    std::uint32_t items = 0;  // requested lossy items
    bool custom = false;      // an explicit item list (--balance=fp8,kvq8)
    std::string source = "default";
    bool has(Item it) const { return (items & bit(it)) != 0; }
};

// "precise" | "balance" ("balanced") | "fast", optionally ":item,item" ("=item,item" for the
// --balance= / --fast= flags is split off by the caller). Throws std::invalid_argument.
Request parseMode(std::string_view spec, std::string source);
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
};

// gdnwmma / h16 for a model without MXFP4 tensors (KG-1): dense models on the R9700 only
// (MoE non-MXFP4 models and the Radeon 8060S keep the f32 paths).
inline bool denseQuantRelaxed(const Target& t) { return !t.has_mxfp4 && !t.moe && !t.gfx1151; }

struct ItemState {
    Item item{};
    bool requested = false;
    bool enabled = false;  // requested (or user-forced) and applicable
    std::string note;      // enabled: detail; skipped: reason
};

struct Plan {
    Mode mode = Mode::precise;
    std::string source = "default";  // where the mode came from: default (balance), command line, WHIRL_MODE
    std::array<ItemState, n_items> items{};
    std::vector<std::string> overrides;  // per-item environment variables that changed the mode's choice
    bool lossy_override = false;         // an override made precise lossy
    // decode / verify / small-batch matmul activations: int8 (q8_1 class; balance / fast) or
    // f16 into the f16 GEMM with f32 accumulation (precise). Set by the loader (setQ8dec).
    bool q8dec = true;
    bool on(Item it) const { return items[static_cast<std::size_t>(it)].enabled; }
};

// Capability table: requested items -> enabled / skipped with the reason.
Plan plan(const Request& r, const Target& t);
// Per-item override from the environment: `forced` = the effective value after the variable.
// Records "WHIRL_X=v: item on (lossy, user-requested)" or "... item off" when it differs.
void applyOverride(Plan& p, Item it, const char* var, const std::string& value, bool forced, bool applicable);

// The decode activation form actually used: q8 = int8 (balance / fast default). env = the
// WHIRL_Q8DEC value ("" = unset); a value that changes the mode's default is recorded as an
// override (int8 in precise: lossy, user-requested).
void setQ8dec(Plan& p, bool q8, const std::string& env);

// "numerics: balance - enabled: fp8 (...), ...; skipped: gdnwmma (...); always: q8dec (...)"
std::string logLine(const Plan& p);
// JSON object for GET /props ("numerics"); kv = the KV format name
std::string propsJson(const Plan& p, std::string_view kv);
// the mode label: "precise", "precise+overrides", "balance", "fast (runs as balance)"
std::string modeLabel(const Plan& p);

// ---- KV format (BAL-Q8): one fixed format per mode x model type, CLI and server alike

enum class KvKind : std::uint8_t { f16, q8, q8h, q8v, q4 };
const char* kvKindName(KvKind k);
// "f16" | "q8" | "q8h" | "q8v" | "q4" (nullopt for anything else, including "auto")
std::optional<KvKind> kvKindFromName(std::string_view s);
// KV kernels in the code object (f16 and q8 are always there)
struct KvCaps {
    bool q8h = false;
    bool q8v = false;
    bool q4 = false;
};
struct KvChoice {
    KvKind kv = KvKind::f16;
    bool debug = false;  // WHIRL_KV chose it (debug override: no fit check, sizing as before)
    std::string why;     // "balance, dense model: q8h"
};
// The request's KV format: WHIRL_KV (forced) wins; else kvq4 -> q4 (where the kernels exist,
// else f16: R9700 has no q4 KV kernels yet, and int8 is no stand-in), dense and MoE; else kvq8 on
// a dense model -> q8h (q8 where the code object has no q8h kernels);
// else f16 (precise, MoE in balance). Independent of free memory and card size.
KvChoice chooseKv(const Request& r, bool moe, const KvCaps& caps, std::optional<KvKind> forced);

// Wording of the "does not fit" messages for the chosen format
struct KvFitLabel {
    std::string mode = "precise";
    std::string fmt = "f16";
    std::string alt = "--balance allows int8 KV";  // a lower-precision alternative the user may choose ("" = none)
};
KvFitLabel kvFitLabel(const Request& r, bool moe, KvKind kv);

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
