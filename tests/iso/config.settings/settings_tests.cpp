// whirl-iso-config-settings-tests: src/config/run_settings.cpp built alone with its one declared dependency
// (platform.env) (ARCH-1 8.5 isolation, arch025-N15). Pins the shared settings rules against verbatim copies of the two
// former readers (3eef475): the server side unchanged on every value, the CLI side == the server rule (owner 2026-10-28:
// CLI / server duplicates merged; the differences are listed in progress\arch025-lane1.HANDOFF.md, N15 table).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "config/run_settings.h"
#include "platform/env.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <optional>
#include <string>
#include <vector>

namespace st = whirl::settings;

namespace {

// ---- verbatim pre-change code (3eef475) ----------------------------------------------------------------------------
// src/server/server_main.cpp:184-196 envU32 (the parse part)
std::optional<std::uint32_t> old_srv_u32(const std::string& v) {
    try {
        std::size_t used = 0;
        const unsigned long x = std::stoul(v, &used);
        if (used == v.size() && x <= 0xffffffffull) return static_cast<std::uint32_t>(x);
    } catch (const std::exception&) {
    }
    return std::nullopt;
}
// src/cli/common.cpp:84-92 envU32
std::optional<std::uint32_t> old_cli_u32(const std::string& v) {
    try {
        return static_cast<std::uint32_t>(std::stoul(v));
    } catch (...) {
    }
    return std::nullopt;
}
// src/server/server_main.cpp:815-823 (WHIRL_MTP_ADAPT)
st::MtpAdapt old_srv_adapt(const std::string& v) {
    st::MtpAdapt r{true, v == "auto", 0};
    try {
        r.adapt = static_cast<std::uint32_t>(std::stoul(v));
    } catch (const std::exception&) {
        r.adapt = 0;
    }
    return r;
}
// src/cli/decode.cpp:472-475 (WHIRL_MTP_ADAPT)
st::MtpAdapt old_cli_adapt(const std::string& v) {
    return st::MtpAdapt{true, v == "auto", static_cast<std::uint32_t>(std::strtoul(v.c_str(), nullptr, 10))};
}

const std::vector<std::string> k_vals = {"0",  "1",   "7",  "4096", "8192", "65536", "4294967295", "4294967296", "",    " 12",
                                         "12 ", "12x", "x12", "-1",   "+5",   "0x10",  "auto",       "3.5",        "99999999999999999999"};

// the unsigned rule == the server's on every text; the CLI's former rule differed only on malformed values
void testU32() {
    std::vector<std::string> cli_diff;
    for (const auto& v : k_vals) {
        ISO_CHECK(st::parseU32(v) == old_srv_u32(v));
        if (st::parseU32(v) != old_cli_u32(v)) cli_diff.push_back(v);
    }
    // the CLI took a leading number ("12 " / "12x" / "3.5" / "0x10" -> 12 / 12 / 3 / 0); these now warn and are not
    // used. Every other value (well-formed ones, leading blanks, signs, out of range) reads as before:
    ISO_CHECK((cli_diff == std::vector<std::string>{"12 ", "12x", "0x10", "3.5"}));
}

// WHIRL_MTP_ADAPT: the server's rule; the CLI's strtoul agreed on the values in use (auto, 0..4, junk)
void testMtpAdapt() {
    for (const auto& v : k_vals) {
        const st::MtpAdapt a = st::parseMtpAdapt(v), s = old_srv_adapt(v);
        ISO_CHECK(a.set && a.automatic == s.automatic && a.adapt == s.adapt);
    }
    for (const char* v : {"0", "1", "2", "3", "4", "auto", "12x", "x"}) {
        const st::MtpAdapt a = st::parseMtpAdapt(std::string(v)), c = old_cli_adapt(v);
        ISO_CHECK(a.automatic == c.automatic && a.adapt == c.adapt);
    }
    ISO_CHECK(!st::parseMtpAdapt(std::nullopt).set);
}

const char* const k_names[] = {"WHIRL_MTP", "WHIRL_MTP_Q4", "WHIRL_TAIL_CK", "WHIRL_DRAFT_HEAD", "WHIRL_SYS_MIN", "WHIRL_PREFILL_BATCH",
                               "WHIRL_PREFILL_CHUNK", "WHIRL_MTP_ADAPT", "WHIRL_DRAFT_MODEL", "WHIRL_DRAFT_VOCAB", "WHIRL_MODE"};

// the environment readers: unset -> the defaults both front ends had; flags = set and not "0"; warnings carry the
// server's text ("using <def>") or "ignored" where there is no default
void testEnv() {
    for (const char* n : k_names) whirl::platform::env::set(n, "");  // (_putenv_s with "" removes the variable)
    std::vector<std::string> warns;
    const st::Warn w = [&](const std::string& s) { warns.push_back(s); };
    ISO_CHECK(st::mtp());
    ISO_CHECK(st::mtpQ4());
    ISO_CHECK(st::tailCk());
    ISO_CHECK(!st::draftHeadQ4());
    ISO_CHECK(st::sysMin(4096, w) == 4096u);
    ISO_CHECK(!st::prefillBatch(w).has_value());
    ISO_CHECK(!st::prefillChunk(w).has_value());
    ISO_CHECK(!st::mtpAdapt().set);
    ISO_CHECK(!st::draftModel() && !st::draftVocab() && !st::mode());
    ISO_CHECK(warns.empty());
    whirl::platform::env::set("WHIRL_MTP", "0");
    whirl::platform::env::set("WHIRL_MTP_Q4", "no");  // not "0": on
    whirl::platform::env::set("WHIRL_TAIL_CK", "0");
    whirl::platform::env::set("WHIRL_DRAFT_HEAD", "q4");
    whirl::platform::env::set("WHIRL_SYS_MIN", "512");
    whirl::platform::env::set("WHIRL_PREFILL_BATCH", "12x");
    whirl::platform::env::set("WHIRL_PREFILL_CHUNK", "4096");
    whirl::platform::env::set("WHIRL_MTP_ADAPT", "auto");
    whirl::platform::env::set("WHIRL_MODE", "fast");
    ISO_CHECK(!st::mtp());
    ISO_CHECK(st::mtpQ4());
    ISO_CHECK(!st::tailCk());
    ISO_CHECK(st::draftHeadQ4());
    ISO_CHECK(st::sysMin(4096, w) == 512u);
    ISO_CHECK(!st::prefillBatch(w).has_value());
    ISO_CHECK(st::prefillChunk(w) == std::optional<std::uint32_t>(4096));
    ISO_CHECK(st::mtpAdapt().automatic);
    ISO_CHECK(st::mode() == std::optional<std::string>("fast"));
    ISO_CHECK(st::envGet("PREFILL_BATCH").has_value());  // a bad value still counts as set (server batch_fixed)
    ISO_CHECK(warns.size() == 1);
    if (warns.size() == 1) ISO_CHECK(warns[0] == "WHIRL_PREFILL_BATCH=12x is not a number; ignored");
    whirl::platform::env::set("WHIRL_SYS_MIN", "abc");
    ISO_CHECK(st::sysMin(4096, w) == 4096u);
    ISO_CHECK(warns.size() == 2);
    if (warns.size() == 2) ISO_CHECK(warns[1] == "WHIRL_SYS_MIN=abc is not a number; using 4096");
    ISO_CHECK(st::envU32("SYS_MIN", 7, nullptr) == 7u);  // no Warn: silent
    for (const char* n : k_names) whirl::platform::env::set(n, "");
}

// arch025-N10 (lane 1 leg 6): the two small-card / KV-room settings whirl-server read by hand (536d470 server_main.cpp:397-398,
// :512-513, :954-955, :1302-1307): references are the old expressions.
void testPoolReserve() {
    namespace st = whirl::settings;
    std::vector<std::string> warns;
    const st::Warn w = [&warns](const std::string& s) { warns.push_back(s); };
    const auto oldHeadroom = [&w]() {
        std::optional<std::uint32_t> h;
        if (st::envGet("VRAM_HEADROOM_MB")) h = st::envU32("VRAM_HEADROOM_MB", 0, w);
        return h;
    };
    for (const char* v : {"", "512", "0", "12x", "99999999999"}) {
        whirl::platform::env::set("WHIRL_VRAM_HEADROOM_MB", v);
        whirl::platform::env::set("WHIRL_POOL_RESERVE_MB", v);
        warns.clear();
        const std::optional<std::uint32_t> a = oldHeadroom();
        const std::size_t w_old = warns.size();
        const std::optional<std::uint32_t> b = st::vramHeadroomMb(w);
        ISO_CHECK(a == b && warns.size() == 2 * w_old);
        ISO_CHECK(st::poolReserveGiven() == st::envGet("POOL_RESERVE_MB").has_value());
        for (std::uint32_t def : {768u, 1536u, 3072u}) {
            warns.clear();
            const std::uint32_t r0 = st::envU32("POOL_RESERVE_MB", def, w);
            const std::size_t w0 = warns.size();
            ISO_CHECK(st::poolReserveMb(def, w) == r0 && warns.size() == 2 * w0);
        }
    }
    whirl::platform::env::set("WHIRL_VRAM_HEADROOM_MB", "12x");  // a bad value: warned, then 0 (not "unset")
    ISO_CHECK(st::vramHeadroomMb(nullptr) == std::optional<std::uint32_t>(0u));
    whirl::platform::env::set("WHIRL_VRAM_HEADROOM_MB", "");
    whirl::platform::env::set("WHIRL_POOL_RESERVE_MB", "");
    ISO_CHECK(!st::vramHeadroomMb(nullptr).has_value() && !st::poolReserveGiven());
}


// ---- arch025-N15 rest (lane 1 leg 7): the 14 other names both front ends read (DISCOVERY review 7) -----------------------
// verbatim pre-change code (f7e11d7): src/cli/decode.cpp:446-480 mtpOpts, src/server/server_main.cpp:734-752 / :1005-1010 /
// :1036 / :1652-1662, src/cli/cmd_chat.cpp:182 / :208 / src/cli/common.cpp:100
std::optional<std::uint32_t> old_srv_drafts(const std::optional<std::string>& v, std::uint32_t def, std::uint32_t mx) {
    std::optional<std::uint32_t> drafts_env;
    if (v) {
        std::uint32_t d = def;
        try {
            d = static_cast<std::uint32_t>(std::stoul(*v));
        } catch (const std::exception&) {
        }
        drafts_env = std::clamp(d, 1u, mx);
    }
    return drafts_env;
}
std::uint32_t old_cli_drafts(const std::optional<std::string>& dv, std::uint32_t def, std::uint32_t mx) {
    std::uint32_t drafts = def;
    if (dv) {
        try {
            drafts = std::clamp<std::uint32_t>(static_cast<std::uint32_t>(std::stoul(*dv)), 1, mx);
        } catch (...) {
            drafts = def;
        }
    }
    return drafts;
}
float old_srv_pmin(const std::optional<std::string>& v, float def) {
    float p_min = def;
    if (v) {
        try {
            p_min = std::stof(*v);
        } catch (const std::exception&) {
        }
    }
    return p_min;
}
float old_cli_pmin(const std::optional<std::string>& v, float def) {
    float p = def;
    if (v) {
        try {
            p = std::stof(*v);
        } catch (...) {
            p = 0;
        }
    }
    return p;
}
std::uint32_t old_cli_strtoul(const std::optional<std::string>& v, std::uint32_t base) {
    return v ? static_cast<std::uint32_t>(std::strtoul(v->c_str(), nullptr, 10)) : base;
}
std::uint32_t old_srv_ckmin(const std::optional<std::string>& x, std::uint32_t def) {
    std::uint32_t ck = def;
    if (x) {
        try {
            ck = static_cast<std::uint32_t>(std::stoul(*x));
        } catch (const std::exception&) {
        }
    }
    return ck;
}
bool sameF(float a, float b) { return a == b || (a != a && b != b); }

const char* const k_names2[] = {"WHIRL_MTP_DRAFTS", "WHIRL_MTP_PMIN", "WHIRL_MTP_NMIN", "WHIRL_MTP_FULLHEAD", "WHIRL_GDN_REPLAY",
                                "WHIRL_NGRAM", "WHIRL_NGRAM_MIN", "WHIRL_NGRAM_SLOPE", "WHIRL_NGRAM_MAX", "WHIRL_NGRAM_FORCE",
                                "WHIRL_PROFILE", "WHIRL_VIS_MODE", "WHIRL_VIS_CKPT_MIN", "WHIRL_VIS_DUMP"};

void testSpecVis() {
    for (const char* n : k_names2) whirl::platform::env::set(n, "");
    std::vector<std::string> warns;
    const st::Warn w = [&](const std::string& s) { warns.push_back(s); };
    // unset: every front end's former default
    ISO_CHECK(!st::mtpDrafts(8, 10).has_value());
    ISO_CHECK(st::mtpPmin(0.25f) == 0.25f);
    ISO_CHECK(st::mtpNmin(0, w) == 0u && st::mtpNmin(5, w) == 5u);
    ISO_CHECK(!st::mtpFullHead() && st::gdnReplay() && st::ngram(true) && !st::ngram(false));
    ISO_CHECK(st::ngramMin(3, w) == 3u && st::ngramMax(0, w) == 0u && !st::ngramForce(false) && st::ngramForce(true));
    ISO_CHECK(st::ngramSlope(0.015f) == 0.015f && st::profile() == 0u);
    ISO_CHECK(!st::visMode() && !st::visDump() && !st::visCkptMin(w).has_value());
    ISO_CHECK(warns.empty());

    std::vector<std::optional<std::string>> vals = {std::nullopt};
    for (const auto& v : k_vals) vals.push_back(v);
    for (const char* v : {"0.5", "-0.25", "1e-3", "nan", "inf", "2", "1", "q"}) vals.push_back(std::string(v));
    std::size_t n = 0, cli_pmin_diff = 0, cli_u32_diff = 0, srv_ck_diff = 0;
    for (const auto& v : vals) {
        // MTP_DRAFTS: the server's and the CLI's code agree on every text (the defaults 1 / 8 are in [1, max_drafts])
        for (std::uint32_t def : {1u, 8u}) {
            const std::optional<std::uint32_t> a = st::parseMtpDrafts(v, def, 10);
            ISO_CHECK(a == old_srv_drafts(v, def, 10));
            ISO_CHECK(a.value_or(def) == old_cli_drafts(v, def, 10));
            ISO_CHECK(a.has_value() == v.has_value());  // the CLI's "explicit draft count" test (!dv) is unchanged
        }
        // MTP_PMIN: the server's rule (an unreadable value keeps the default); the CLI used 0 there. Every mtpDefaults row
        // has p_min 0 (src/spec/policy.cpp), so with the defaults in use the two agree on every text.
        for (float def : {0.0f, 0.3f}) {
            const float a = st::parsePmin(v, def);
            ISO_CHECK(sameF(a, old_srv_pmin(v, def)));
            const float c = old_cli_pmin(v, def);
            if (!sameF(a, c)) {
                ++cli_pmin_diff;
                ISO_CHECK(def != 0.0f && c == 0.0f && a == def);  // only an unreadable value with a non-zero default
            }
        }
        // NGRAM_SLOPE: std::strtof on both sides
        ISO_CHECK(sameF(st::parseNgramSlope(v, 0.015f), v ? std::strtof(v->c_str(), nullptr) : 0.015f));
        // PROFILE: unset 0, "2" 2, anything else 1 (the server's eo.profile); the CLI's test was "set at all"
        const std::uint32_t pf = st::parseProfile(v);
        ISO_CHECK(pf == (!v ? 0u : (*v == "2" ? 2u : 1u)) && (pf != 0) == v.has_value());
        // the u32 names the CLI read with strtoul (MTP_NMIN, NGRAM_MIN, NGRAM_MAX) and the server's VIS_CKPT_MIN (stoul): now
        // the one u32 rule (the server's for the first three). They differ only on texts the rule calls malformed.
        for (std::uint32_t def : {0u, 3u, 1024u}) {
            const std::uint32_t a = v ? st::parseU32(*v).value_or(def) : def;
            if (a != old_cli_strtoul(v, def)) {
                ++cli_u32_diff;
                ISO_CHECK(v && !st::parseU32(*v).has_value());
            }
            if (a != old_srv_ckmin(v, def)) {
                ++srv_ck_diff;
                ISO_CHECK(v && !st::parseU32(*v).has_value());
            }
        }
        ++n;
    }
    // the readers on the real environment
    whirl::platform::env::set("WHIRL_MTP_DRAFTS", "12");
    whirl::platform::env::set("WHIRL_MTP_PMIN", "0.4");
    whirl::platform::env::set("WHIRL_MTP_NMIN", "2");
    whirl::platform::env::set("WHIRL_MTP_FULLHEAD", "0");  // set at all: the full head
    whirl::platform::env::set("WHIRL_GDN_REPLAY", "0");
    whirl::platform::env::set("WHIRL_NGRAM", "0");
    whirl::platform::env::set("WHIRL_NGRAM_MIN", "4");
    whirl::platform::env::set("WHIRL_NGRAM_SLOPE", "0.5");
    whirl::platform::env::set("WHIRL_NGRAM_MAX", "12x");
    whirl::platform::env::set("WHIRL_NGRAM_FORCE", "1");
    whirl::platform::env::set("WHIRL_PROFILE", "2");
    whirl::platform::env::set("WHIRL_VIS_MODE", "stream");
    whirl::platform::env::set("WHIRL_VIS_CKPT_MIN", "512");
    whirl::platform::env::set("WHIRL_VIS_DUMP", "dump_prefix");
    ISO_CHECK(st::mtpDrafts(8, 10) == std::optional<std::uint32_t>(10u));
    ISO_CHECK(st::mtpPmin(0.0f) == 0.4f && st::mtpNmin(0, w) == 2u && st::mtpFullHead() && !st::gdnReplay() && !st::ngram(true));
    ISO_CHECK(st::ngramMin(3, w) == 4u && st::ngramSlope(0.015f) == 0.5f && st::ngramForce(false) && st::profile() == 2u);
    ISO_CHECK(warns.empty());
    ISO_CHECK(st::ngramMax(0, w) == 0u);
    ISO_CHECK(warns.size() == 1);
    if (warns.size() == 1) ISO_CHECK(warns[0] == "WHIRL_NGRAM_MAX=12x is not a number; using 0");
    ISO_CHECK(st::visMode() == std::optional<std::string>("stream") && st::visDump() == std::optional<std::string>("dump_prefix"));
    ISO_CHECK(st::visCkptMin(w) == std::optional<std::uint32_t>(512u));
    whirl::platform::env::set("WHIRL_VIS_CKPT_MIN", "1k");
    ISO_CHECK(!st::visCkptMin(w).has_value() && warns.size() == 2);
    if (warns.size() == 2) ISO_CHECK(warns[1] == "WHIRL_VIS_CKPT_MIN=1k is not a number; ignored");  // the CLI's former text
    for (const char* nm : k_names2) whirl::platform::env::set(nm, "");
    std::printf("spec / vision settings: %zu texts; CLI MTP_PMIN differs on %zu (non-zero default only), CLI strtoul names on %zu, "
                "server VIS_CKPT_MIN on %zu (all malformed)\n",
                n, cli_pmin_diff, cli_u32_diff, srv_ck_diff);
}

// ---- arch025-lane1 leg 8: the four WHIRL_* values whirl-server still parsed by hand (5728296 src/server/server_main.cpp) --
// :999 SLOT_DRAFTS
std::optional<std::uint32_t> old_slot_drafts(const std::optional<std::string>& v) {
    std::optional<std::uint32_t> r;
    if (v) r = *v == "2" ? 2u : (*v != "0" ? 1u : 0u);
    return r;
}
// :1012-1017 GATHER_MS
std::optional<double> old_gather_ms(const std::optional<std::string>& v) {
    std::optional<double> r;
    if (v) {
        try {
            r = std::stod(*v);
        } catch (const std::exception&) {
        }
    }
    return r;
}
// :1068-1074 KV_RAM_MB (the value, or "not a number")
std::optional<std::uint64_t> old_kv_ram_mb(const std::string& v, bool& bad) {
    bad = false;
    try {
        return std::stoull(v);
    } catch (const std::exception&) {
        bad = true;
    }
    return std::nullopt;
}
// :516-521 SNAP_SETS (the DFlash2 pre-load fit)
std::optional<std::uint32_t> old_snap_sets(const std::optional<std::string>& v) {
    std::optional<std::uint32_t> r;
    if (v) {
        try {
            r = static_cast<std::uint32_t>(std::stoul(*v));
        } catch (const std::exception&) {
        }
    }
    return r;
}

// :1533-1538 KV_SSD_GB and :1642-1647 VIS_CACHE_MB (std::stoull, an unreadable value keeps the default)
std::optional<std::uint64_t> old_u64_lead(const std::optional<std::string>& v) {
    std::optional<std::uint64_t> r;
    if (v) {
        try {
            r = std::stoull(*v);
        } catch (const std::exception&) {
        }
    }
    return r;
}
// :1716-1717 GOV_RECOVER_S / GOV_HOLD_S (std::strtod: never fails, an unreadable text is 0)
std::optional<double> old_strtod(const std::optional<std::string>& v) {
    std::optional<double> r;
    if (v) r = std::strtod(v->c_str(), nullptr);
    return r;
}

const char* const k_names8[] = {"WHIRL_SLOT_DRAFTS", "WHIRL_GATHER_MS", "WHIRL_KV_RAM_MB", "WHIRL_SNAP_SETS", "WHIRL_KV_SSD_GB",
                                "WHIRL_VIS_IDLE_S", "WHIRL_VIS_CACHE_MB", "WHIRL_GOV_RECOVER_S", "WHIRL_GOV_HOLD_S"};

void testServerLeg8() {
    std::vector<std::optional<std::string>> vals = {std::nullopt};
    for (const auto& v : k_vals) vals.push_back(v);
    for (const char* v : {"2", "1", "0", "00", "02", "2x", "-1", "30", "0.5", "1e3", "-0.25", "nan", "inf", "x", "18446744073709551615",
                          "18446744073709551616", "  7", "7  "})
        vals.push_back(std::string(v));
    std::size_t n = 0;
    for (const auto& v : vals) {
        ISO_CHECK(st::parseSlotDrafts(v) == old_slot_drafts(v));
        const std::optional<double> g = st::parseGatherMs(v), og = old_gather_ms(v);
        ISO_CHECK(g.has_value() == og.has_value() && (!g || *g == *og || (*g != *g && *og != *og)));
        ISO_CHECK(st::parseU32Lead(v) == old_snap_sets(v));  // SNAP_SETS and VIS_IDLE_S (the same stoul code at :516 / :1616)
        ISO_CHECK(st::parseU64Lead(v) == old_u64_lead(v));    // KV_SSD_GB (:1533-1538), VIS_CACHE_MB (:1642-1647)
        {
            const std::optional<double> a = st::parseStrtod(v), b = old_strtod(v);  // GOV_RECOVER_S / GOV_HOLD_S (:1716-1717)
            ISO_CHECK(a.has_value() == b.has_value() && (!a || *a == *b || (*a != *a && *b != *b)));
        }
        const st::KvRamMb k = st::parseKvRamMb(v);
        ISO_CHECK(k.set == v.has_value());
        if (v) {
            bool bad = false;
            const std::optional<std::uint64_t> o = old_kv_ram_mb(*v, bad);
            ISO_CHECK(k.mb == o && k.mb.has_value() == !bad && k.text == *v);
        } else {
            ISO_CHECK(!k.mb.has_value() && k.text.empty());
        }
        ++n;
    }
    // the readers read the variable (unset -> nullopt / not set)
    for (const char* nm : k_names8) whirl::platform::env::set(nm, "");
    ISO_CHECK(!st::slotDrafts() && !st::gatherMs() && !st::kvRamMb().set && !st::snapSets());
    ISO_CHECK(!st::kvSsdGb() && !st::visIdleS() && !st::visCacheMb() && !st::govRecoverS() && !st::govHoldS());
    whirl::platform::env::set("WHIRL_KV_SSD_GB", "16");
    whirl::platform::env::set("WHIRL_VIS_IDLE_S", "5");
    whirl::platform::env::set("WHIRL_VIS_CACHE_MB", "x");
    whirl::platform::env::set("WHIRL_GOV_RECOVER_S", "2.5");
    whirl::platform::env::set("WHIRL_GOV_HOLD_S", "q");
    ISO_CHECK(st::kvSsdGb() == std::optional<std::uint64_t>(16u) && st::visIdleS() == std::optional<std::uint32_t>(5u) && !st::visCacheMb());
    ISO_CHECK(st::govRecoverS() == std::optional<double>(2.5) && st::govHoldS() == std::optional<double>(0.0));
    whirl::platform::env::set("WHIRL_SLOT_DRAFTS", "2");
    whirl::platform::env::set("WHIRL_GATHER_MS", "12.5");
    whirl::platform::env::set("WHIRL_KV_RAM_MB", "4096");
    whirl::platform::env::set("WHIRL_SNAP_SETS", "3");
    ISO_CHECK(st::slotDrafts() == std::optional<std::uint32_t>(2u) && st::gatherMs() == std::optional<double>(12.5));
    ISO_CHECK(st::kvRamMb().mb == std::optional<std::uint64_t>(4096u) && st::snapSets() == std::optional<std::uint32_t>(3u));
    whirl::platform::env::set("WHIRL_KV_RAM_MB", "lots");
    {
        const st::KvRamMb k = st::kvRamMb();
        ISO_CHECK(k.set && !k.mb && k.text == "lots");
    }
    for (const char* nm : k_names8) whirl::platform::env::set(nm, "");
    std::printf("server settings (leg 8): SLOT_DRAFTS / GATHER_MS / KV_RAM_MB / SNAP_SETS / KV_SSD_GB / VIS_IDLE_S / VIS_CACHE_MB / "
                "GOV_RECOVER_S / GOV_HOLD_S, %zu texts == 5728296\n", n);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("config.settings", {{"u32", testU32}, {"mtp_adapt", testMtpAdapt}, {"env", testEnv}, {"pool_reserve", testPoolReserve}, {"spec_vis", testSpecVis}, {"server_leg8", testServerLeg8}}, argc, argv);
}
