// Run settings shared by the two front ends (arch025-N15): the WHIRL_* names that `whirl` (src/cli) and whirl-server
// (src/server/server_main.cpp) both read, with ONE parse rule each, so the CLI and the server can no longer read the
// same variable two ways. The shape of llama.cpp's common_params / common_arg (one option table with its env name,
// used by llama-cli and llama-server; progress\arch025-lane1.PRIOR_ART.md section 1). Host-only, links whirl_platform
// only: no Model / HIP / log type. A bad value is reported through the caller's Warn (the server: logW; the CLI:
// stderr), never thrown.
//
// The differences of the two former readers and the side kept are listed in progress\arch025-lane1.HANDOFF.md (N15
// table); with every variable unset (the default) both front ends read exactly what they read before.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace whirl::settings {

using Warn = std::function<void(const std::string&)>;

// WHIRL_<name> (UTF-8 of the wide value, as envGet). nullopt when unset.
std::optional<std::string> envGet(std::string_view name);

// WHIRL_<name> set and not "0"; `def` when unset (the former envFlag / envOn).
bool envFlag(std::string_view name, bool def);

// The ONE unsigned-number rule (the server's envU32, FR-SRV P3-10): the whole value must be a decimal number that fits
// in 32 bits; anything else is a warning "WHIRL_<name>=<v> is not a number; using <def>" and `def`.
std::uint32_t envU32(std::string_view name, std::uint32_t def, const Warn& warn);
// The same rule without a default: nullopt when unset or bad ("...; ignored").
std::optional<std::uint32_t> envU32(std::string_view name, const Warn& warn);
// The rule on a given text (tests, and a value read earlier).
std::optional<std::uint32_t> parseU32(std::string_view v);

// ---- the shared names (N15: CLI and server both read them) ----------------------------------------------------------

// WHIRL_MTP: speculative decoding with the checkpoint's MTP head (default on; 0 = off).
inline bool mtp() { return envFlag("MTP", true); }
// WHIRL_MTP_Q4: requantize the MTP layer to Q4 (default on).
inline bool mtpQ4() { return envFlag("MTP_Q4", true); }
// WHIRL_TAIL_CK: the prefill's tail checkpoint forward end (default on; 0 = off, debug knob).
inline bool tailCk() { return envFlag("TAIL_CK", true); }
// WHIRL_DRAFT_HEAD=q4: the older Q4_K draft head instead of the 2-bit one.
inline bool draftHeadQ4() { return envGet("DRAFT_HEAD").value_or("") == "q4"; }
// WHIRL_DRAFT_MODEL / WHIRL_DRAFT_VOCAB / WHIRL_MODE: passed as text to their resolvers
// (dflash::resolveDraftModel, qwen35::draftVocabChoice, numerics::resolve), unchanged.
inline std::optional<std::string> draftModel() { return envGet("DRAFT_MODEL"); }
inline std::optional<std::string> draftVocab() { return envGet("DRAFT_VOCAB"); }
inline std::optional<std::string> mode() { return envGet("MODE"); }

// WHIRL_MTP_ADAPT: "auto" = the automatic draft count; a number N = adaptive count ceil(acc_ema + N); anything else
// (not a number) = 0 (the server's rule: std::stoul, a leading number counts).
struct MtpAdapt {
    bool set = false;        // the variable exists
    bool automatic = false;  // == "auto"
    std::uint32_t adapt = 0;
};
MtpAdapt mtpAdapt();
MtpAdapt parseMtpAdapt(const std::optional<std::string>& v);

// WHIRL_PREFILL_BATCH / WHIRL_PREFILL_CHUNK: the u32 rule (nullopt when unset or bad). Each front end applies its own
// floor / clamp to the value (the prefill plan, N15 step 2). "Set at all" (a bad value too) is envGet(name).has_value().
inline std::optional<std::uint32_t> prefillBatch(const Warn& warn) { return envU32("PREFILL_BATCH", warn); }
inline std::optional<std::uint32_t> prefillChunk(const Warn& warn) { return envU32("PREFILL_CHUNK", warn); }
// WHIRL_SYS_MIN: the smallest system message that gets its own prefill forward end; `def` when unset or bad.
inline std::uint32_t sysMin(std::uint32_t def, const Warn& warn) { return envU32("SYS_MIN", def, warn); }

// ---- whirl-server's KV-room settings (arch025-N10, lane 1 leg 6: were written out by hand in server_main.cpp) ----------
// WHIRL_VRAM_HEADROOM_MB: VRAM kept free for the desktop on any card (vram::CardInput::headroom_mb_env). Set = the u32
// rule with default 0, so a bad value warns and gives 0 (it is not "unset" as with the optional envU32).
inline std::optional<std::uint32_t> vramHeadroomMb(const Warn& warn) {
    if (!envGet("VRAM_HEADROOM_MB")) return std::nullopt;
    return envU32("VRAM_HEADROOM_MB", 0, warn);
}
// WHIRL_POOL_RESERVE_MB: set at all (a bad value too) = the user's own reserve (the free-VRAM rule, no WDDM budget, no
// default headroom); its value is the u32 rule with the caller's default (poolbudget::defaultReserveMb).
inline bool poolReserveGiven() { return envGet("POOL_RESERVE_MB").has_value(); }
inline std::uint32_t poolReserveMb(std::uint32_t def, const Warn& warn) { return envU32("POOL_RESERVE_MB", def, warn); }

// ---- the other names both front ends read (arch025-N15 rest, lane 1 leg 7; DISCOVERY review 7: 26 shared names, the 11
// above + HIP_DEVICE in config.device + these 14). Each read stays at its old place; with a variable unset both front ends
// read what they read before. The differences of the former readers (bad values only) are in DESIGN 3.9.
// WHIRL_MTP_DRAFTS: the fixed draft count. nullopt when unset (so "given at all" = has_value()); a text std::stoul cannot
// read gives `def`; clamped to [1, max] (both former readers).
std::optional<std::uint32_t> parseMtpDrafts(const std::optional<std::string>& v, std::uint32_t def, std::uint32_t max);
inline std::optional<std::uint32_t> mtpDrafts(std::uint32_t def, std::uint32_t max) { return parseMtpDrafts(envGet("MTP_DRAFTS"), def, max); }
// WHIRL_MTP_PMIN: std::stof; unset or unreadable = `def` (the server's rule; the CLI used 0 for an unreadable value).
float parsePmin(const std::optional<std::string>& v, float def);
inline float mtpPmin(float def) { return parsePmin(envGet("MTP_PMIN"), def); }
// WHIRL_MTP_NMIN / WHIRL_NGRAM_MIN / WHIRL_NGRAM_MAX: the u32 rule with the caller's default (the server's; the CLI read
// them with strtoul).
inline std::uint32_t mtpNmin(std::uint32_t def, const Warn& warn) { return envU32("MTP_NMIN", def, warn); }
inline std::uint32_t ngramMin(std::uint32_t def, const Warn& warn) { return envU32("NGRAM_MIN", def, warn); }
inline std::uint32_t ngramMax(std::uint32_t def, const Warn& warn) { return envU32("NGRAM_MAX", def, warn); }
// WHIRL_MTP_FULLHEAD set at all: draft with the full LM head (no draft head).
inline bool mtpFullHead() { return envGet("MTP_FULLHEAD").has_value(); }
// WHIRL_GDN_REPLAY (default on), WHIRL_NGRAM (the caller's default: on), WHIRL_NGRAM_FORCE: set and not "0".
inline bool gdnReplay() { return envFlag("GDN_REPLAY", true); }
inline bool ngram(bool def) { return envFlag("NGRAM", def); }
inline bool ngramForce(bool def) { return envFlag("NGRAM_FORCE", def); }
// WHIRL_NGRAM_SLOPE: std::strtof of the text (both former readers); `def` when unset.
float parseNgramSlope(const std::optional<std::string>& v, float def);
inline float ngramSlope(float def) { return parseNgramSlope(envGet("NGRAM_SLOPE"), def); }
// WHIRL_PROFILE: 0 unset, 2 for "2", 1 for any other value (the CLI profiles when it is not 0).
std::uint32_t parseProfile(const std::optional<std::string>& v);
inline std::uint32_t profile() { return parseProfile(envGet("PROFILE")); }
// WHIRL_VIS_MODE / WHIRL_VIS_DUMP: passed as text (vision::parseMode / the dump prefix), unchanged.
inline std::optional<std::string> visMode() { return envGet("VIS_MODE"); }
inline std::optional<std::string> visDump() { return envGet("VIS_DUMP"); }
// WHIRL_VIS_CKPT_MIN: the u32 rule, nullopt when unset or bad ("ignored"; the CLI's; the server read it with std::stoul).
inline std::optional<std::uint32_t> visCkptMin(const Warn& warn) { return envU32("VIS_CKPT_MIN", warn); }

}  // namespace whirl::settings
