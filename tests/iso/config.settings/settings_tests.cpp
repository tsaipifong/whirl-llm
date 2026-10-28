// whirl-iso-config-settings-tests: src/config/run_settings.cpp built alone with its one declared dependency
// (platform.env) (ARCH-1 8.5 isolation, arch025-N15). Pins the shared settings rules against verbatim copies of the two
// former readers (3eef475): the server side unchanged on every value, the CLI side == the server rule (owner 2026-10-28:
// CLI / server duplicates merged; the differences are listed in progress\arch025-lane1.HANDOFF.md, N15 table).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "config/run_settings.h"
#include "platform/env.h"

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

}  // namespace

int main(int argc, char** argv) {
    return iso::run("config.settings", {{"u32", testU32}, {"mtp_adapt", testMtpAdapt}, {"env", testEnv}}, argc, argv);
}
