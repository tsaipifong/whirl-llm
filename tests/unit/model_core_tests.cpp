// WHIRL unit tests of module model.core (arch025 N7: moved verbatim from the single tests/unit/unit_tests.cpp; one file
// per module, built alone as whirl-unit-model-core-tests and together in whirl-tests).
// SPDX-License-Identifier: Apache-2.0

#include "../test_check.h"
#include "../test_env.h"
#include "whirl/value_scope.h"
#include "whirl/dflash_policy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <new>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <psapi.h>  // K32GetProcessMemoryInfo (arch025-surv testPlatformProcSurvivors)
#endif

using namespace whirl;
using whirl::test::g_fail;
using whirl::test::g_pass;
using whirl::test::throws;

namespace {
// ---------------------------------------------------------------------------
// DFlash2 host policy (whirl/dflash_policy.h): draft-source choice, environment, capture /
// ring sizing, draft limit, the server slot's context range and ingest plan
void testDflashPolicy() {
    namespace d = whirl::dflash;
    // --draft parsing
    CHECK(d::parseSource("mtp") == d::Source::mtp);
    CHECK(d::parseSource("dflash") == d::Source::dflash);
    CHECK(!d::parseSource("DFlash") && !d::parseSource("") && !d::parseSource("eagle"));
    CHECK_EQ(std::string(d::sourceName(d::Source::dflash)), std::string("dflash"));
    // choice: mtp stays mtp (a set WHIRL_DRAFT_MODEL is reported, not loaded)
    const std::optional<std::string> dm = std::string("D:/m/dflash2-q8_0.gguf"), none;
    {
        const d::Choice c = d::choose(d::Source::mtp, none, "", true);
        CHECK(c.source == d::Source::mtp && c.warning.empty());
        const d::Choice c2 = d::choose(d::Source::mtp, dm, "", true);
        CHECK(c2.source == d::Source::mtp && c2.warning.find("not loaded") != std::string::npos);
    }
    // dflash: every fallback is mtp with a reason; all conditions met -> dflash, silent
    CHECK(d::choose(d::Source::dflash, dm, "", true).source == d::Source::dflash);
    CHECK(d::choose(d::Source::dflash, dm, "", true).warning.empty());
    {
        const d::Choice c = d::choose(d::Source::dflash, none, "", true);
        CHECK(c.source == d::Source::mtp && c.warning.find("WHIRL_DRAFT_MODEL") != std::string::npos);
        const d::Choice c2 = d::choose(d::Source::dflash, dm, "hidden size 2048 != target 5120", true);
        CHECK(c2.source == d::Source::mtp && c2.warning.find("hidden size 2048") != std::string::npos &&
              c2.warning.find("dflash2-q8_0.gguf") != std::string::npos);
        const d::Choice c3 = d::choose(d::Source::dflash, dm, "", false);
        CHECK(c3.source == d::Source::mtp && c3.warning.find("dflash_") != std::string::npos);
        // an incompatible drafter is reported before missing kernels
        CHECK(d::choose(d::Source::dflash, dm, "x", false).warning.find("does not fit") != std::string::npos);
    }
    // --draft-model FILE vs WHIRL_DRAFT_MODEL: the command line wins, the environment is the fallback
    // (a debug override when no --draft-model is given); empty = not set (REVIEW-RC4-C P3-8)
    {
        const std::optional<std::string> cli = std::string("a.gguf"), env = std::string("b.gguf"), empty = std::string();
        CHECK(*d::resolveDraftModel(cli, none) == "a.gguf");
        CHECK(*d::resolveDraftModel(none, env) == "b.gguf");
        CHECK(*d::resolveDraftModel(cli, env) == "a.gguf");
        CHECK(*d::resolveDraftModel(cli, empty) == "a.gguf");
        CHECK(!d::resolveDraftModel(none, none) && !d::resolveDraftModel(empty, empty) && !d::resolveDraftModel(none, empty));
        // the log line names the source
        CHECK(d::draftModelNote(cli, env).find("WHIRL_DRAFT_MODEL=b.gguf ignored") != std::string::npos);
        CHECK(d::draftModelNote(none, env).find("from WHIRL_DRAFT_MODEL") != std::string::npos);
        CHECK(d::draftModelNote(cli, none).empty() && d::draftModelNote(none, none).empty() && d::draftModelNote(cli, cli).empty());
    }
    // relaxacc: MTP drafts only (DFlash2 keeps exact acceptance in fast mode)
    CHECK(d::relaxApplies(d::Source::mtp, true) && !d::relaxApplies(d::Source::mtp, false));
    CHECK(!d::relaxApplies(d::Source::dflash, true) && !d::relaxApplies(d::Source::dflash, false));
    // FR-DFL P2-3: the engine's per-slot rule (draft source codes 0 MTP, 1 n-gram, 2 DFlash2)
    CHECK(d::relaxAppliesTo(d::DraftKind::mtp, true) && !d::relaxAppliesTo(d::DraftKind::mtp, false));
    CHECK(!d::relaxAppliesTo(d::DraftKind::ngram, true) && !d::relaxAppliesTo(d::DraftKind::dflash, true));
    CHECK(static_cast<int>(d::DraftKind::mtp) == 0 && static_cast<int>(d::DraftKind::ngram) == 1 && static_cast<int>(d::DraftKind::dflash) == 2);
    // environment
    {
        std::map<std::string, std::string> envm;
        auto env = [&](const char* n) -> std::optional<std::string> {
            auto it = envm.find(n);
            if (it == envm.end()) return std::nullopt;
            return it->second;
        };
        std::vector<std::string> w;
        d::Params p = d::parseParams(env, w);
        CHECK(w.empty() && p.p_min == 0.0f && p.n_min == 0 && p.warm == 64 && !p.capture_rows);
        envm = {{"DFLASH_PMIN", "0.25"}, {"DFLASH_NMIN", "2"}, {"DFLASH_WARM", "0"}, {"DFLASH_CAPTURE_ROWS", "512"}};
        p = d::parseParams(env, w);
        CHECK(w.empty() && std::fabs(p.p_min - 0.25f) < 1e-7f && p.n_min == 2 && p.warm == 0 && p.capture_rows && *p.capture_rows == 512);
        envm = {{"DFLASH_PMIN", "1.5"}, {"DFLASH_NMIN", "-1"}, {"DFLASH_WARM", "6x"}, {"DFLASH_CAPTURE_ROWS", "0"}};
        p = d::parseParams(env, w);
        CHECK_EQ(w.size(), std::size_t{4});
        CHECK(p.p_min == 0.0f && p.n_min == 0 && p.warm == 64 && !p.capture_rows);
        w.clear();
        // FR-DFL P2-2: fewer capture rows than one verify of a full block (16) are refused
        for (const char* bad : {"4", "7", "15"}) {
            envm = {{"DFLASH_CAPTURE_ROWS", bad}};
            p = d::parseParams(env, w);
            CHECK(!p.capture_rows);
            CHECK(w.size() == 1 && w[0].find(">= 16") != std::string::npos);
            w.clear();
        }
        envm = {{"DFLASH_CAPTURE_ROWS", "16"}};
        p = d::parseParams(env, w);
        CHECK(w.empty() && p.capture_rows && *p.capture_rows == d::kMinCaptureRows);
        w.clear();
        envm = {{"DFLASH_PMIN", "1"}, {"DFLASH_NMIN", "99999999999"}, {"DFLASH_PMIN_X", "z"}};
        p = d::parseParams(env, w);
        CHECK(p.p_min == 1.0f && w.size() == 1 && w[0].find("DFLASH_NMIN") != std::string::npos);
        w.clear();
        envm = {{"DFLASH_PMIN", ".5"}};
        CHECK(d::parseParams(env, w).p_min == 0.5f && w.empty());
        envm = {{"DFLASH_PMIN", "."}};
        CHECK(d::parseParams(env, w).p_min == 0.0f && w.size() == 1);
        w.clear();
        envm = {{"DFLASH_PMIN", "0.1.2"}};
        CHECK(d::parseParams(env, w).p_min == 0.0f && w.size() == 1);
    }
    // capture rows / ring: the window (2048 without one), the batch caps the capture
    {
        d::Params p;
        CHECK_EQ(d::captureRows(p, 2048, 4096), 2048u);
        CHECK_EQ(d::captureRows(p, 0, 4096), 2048u);
        CHECK_EQ(d::captureRows(p, 4096, 1024), 1024u);
        p.capture_rows = 300;
        CHECK_EQ(d::captureRows(p, 2048, 4096), 300u);
        CHECK_EQ(d::ringRows(0), 2048u);
        CHECK_EQ(d::ringRows(4096), 4096u);
    }
    // ring pieces: wrap at the end, never more than the ring
    {
        d::RingPieces r = d::ringPieces(10, 5, 2048);
        CHECK(r.start == 10 && r.first == 5 && r.second == 0);
        r = d::ringPieces(2046, 5, 2048);
        CHECK(r.start == 2046 && r.first == 2 && r.second == 3);
        r = d::ringPieces(4096, 7, 2048);
        CHECK(r.start == 0 && r.first == 7 && r.second == 0);
        r = d::ringPieces(5, 3000, 2048);
        CHECK(r.start == 5 && r.first == 2043 && r.second == 5 && r.first + r.second == 2048);
        r = d::ringPieces(5, 0, 2048);
        CHECK(r.first == 0 && r.second == 0);
    }
    // draft limit: block 8 -> 7 drafts solo; shared verify rows / snapshots per slot; unfused 1
    {
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 1, true), 7u);
        CHECK_EQ(d::draftLimit(16, 15, 15, 16, 1, true), 15u);
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 2, true), 7u);
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 4, true), 3u);   // 16 rows / 4 slots - own row
        CHECK_EQ(d::draftLimit(8, 15, 15, 32, 4, true), 3u);   // snapshots 15 / 4
        CHECK_EQ(d::draftLimit(8, 15, 32, 32, 4, true), 7u);   // replay: snap_cap = vrows
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 1, false), 1u);
        CHECK_EQ(d::draftLimit(1, 15, 15, 16, 1, true), 0u);   // no drafter
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 0, true), 0u);
        CHECK_EQ(d::draftLimit(8, 15, 15, 16, 17, true), 0u);
    }
    // slot range + ingest plan (Engine::dflIngest / dflReady)
    {
        d::Range r;
        // prefill 0..99 in one forward, all captured
        d::Ingest g = d::planIngest(r, 100, 100, 0, 100, 0);
        CHECK(g.any && g.cap_row == 0 && g.n == 100 && g.pos == 0 && g.gap == 0 && r.lo == 0 && r.hi == 100);
        CHECK(d::ready(r, 100, 64) && !d::ready(r, 101, 64));
        // verify of 8 rows (slot rows 0..7), kept 3: positions 100..102
        g = d::planIngest(r, 8, 8, 0, 3, 100);
        CHECK(g.any && g.cap_row == 0 && g.n == 3 && g.pos == 100 && r.hi == 103 && r.lo == 0);
        // batched verify: this slot is rows 8..15 of 24, the capture holds the last 16 rows (8..23)
        g = d::planIngest(r, 16, 24, 8, 2, 103);
        CHECK(g.any && g.cap_row == 0 && g.n == 2 && g.pos == 103 && r.hi == 105);
        // capture of only the last 4 rows of a 2048-row chunk: a gap, the run restarts
        g = d::planIngest(r, 4, 2048, 0, 2048, 105);
        CHECK(g.any && g.cap_row == 0 && g.n == 4 && g.pos == 105 + 2044 && g.gap == 2044 && r.lo == 2149 && r.hi == 2153);
        CHECK(!d::ready(r, 2153, 64) && d::ready(r, 2153, 4) && d::ready(r, 2153, 0));
        // rows entirely outside the capture: gap counted, range untouched
        const d::Range before = r;
        g = d::planIngest(r, 4, 16, 0, 8, 2153);
        CHECK(!g.any && g.gap == 8 && r.lo == before.lo && r.hi == before.hi);
        // restore point (dflReset at 500): ready after warm rows
        r = d::Range{500, 500};
        CHECK(!d::ready(r, 500, 64));
        g = d::planIngest(r, 64, 64, 0, 64, 500);
        CHECK(d::ready(r, 564, 64) && r.lo == 500);
        d::Range r0{0, 10};
        CHECK(d::ready(r0, 10, 64));  // short prompt: min(pos, warm)
        CHECK_EQ(d::planIngest(r0, 8, 8, 0, 0, 10).n, 0u);
    }
    // FR-DFL P3-3: prefix reuse trims the run to positions < reuse that the ring still holds
    {
        d::Trim t = d::trimForReuse(d::Range{0, 5000}, 2000, 2048);  // ring holds 2952..4999: nothing below 2000
        CHECK(t.reset && t.r.lo == 2000 && t.r.hi == 2000);
        t = d::trimForReuse(d::Range{0, 3000}, 2000, 2048);  // ring holds 952..2999: [952, 2000) stays
        CHECK(!t.reset && t.r.lo == 952 && t.r.hi == 2000);
        t = d::trimForReuse(d::Range{0, 1500}, 2000, 2048);  // all in the ring, below reuse: unchanged
        CHECK(!t.reset && t.r.lo == 0 && t.r.hi == 1500);
        t = d::trimForReuse(d::Range{100, 5000}, 4000, 0);  // no ring limit: the old rule
        CHECK(!t.reset && t.r.lo == 100 && t.r.hi == 4000);
        t = d::trimForReuse(d::Range{3500, 5000}, 3000, 2048);  // run started after reuse: nothing left
        CHECK(t.reset && t.r.lo == 3000 && t.r.hi == 3000);
        t = d::trimForReuse(d::Range{0, 0}, 500, 2048);  // empty run: no reset (the next ingest counts the gap)
        CHECK(!t.reset && t.r.lo == 0 && t.r.hi == 0);
    }
    // FR-DFL P2-2: whirl chat's ingest after a verify (verifyIngest): capture rows < verify rows never
    // index before the capture buffer (was cap_n - (nd + 1): 4 - 8 wrapped to 4294967292)
    {
        d::Ingest g = d::verifyIngest(8, 8, 3, 100);  // full capture: rows 0..2 kept
        CHECK(g.any && g.cap_row == 0 && g.n == 3 && g.pos == 100 && g.gap == 0);
        g = d::verifyIngest(4, 8, 8, 100);  // capture holds verify rows 4..7, all kept
        CHECK(g.any && g.cap_row == 0 && g.n == 4 && g.pos == 104 && g.gap == 4 && g.cap_row + g.n <= 4);
        g = d::verifyIngest(4, 8, 6, 100);  // kept rows 0..5: rows 4, 5 are capture rows 0, 1
        CHECK(g.any && g.cap_row == 0 && g.n == 2 && g.pos == 104);
        g = d::verifyIngest(4, 8, 2, 100);  // kept rows 0..1 are not in the capture
        CHECK(!g.any && g.n == 0);
        g = d::verifyIngest(2048, 8, 8, 7);  // capture larger than the verify (cap_n = rows of the last forward)
        CHECK(g.any && g.cap_row == 0 && g.n == 8 && g.pos == 7);
    }
}


// OVH-C1: the MTP draft window (Model::attn_win) must be restored when the draft attention throws,
// otherwise the next request's trunk attention would inherit the window and drop keys.
void testValueScope() {
    std::int32_t win = 0;
    {
        whirl::ValueScope<std::int32_t> s(win, 0x4005);
        CHECK_EQ(win, 0x4005);
    }
    CHECK_EQ(win, 0);
    bool threw = false;
    try {
        whirl::ValueScope<std::int32_t> s(win, 0x4005);
        throw std::runtime_error("attnBlock failed");
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHECK(threw);
    CHECK_EQ(win, 0);  // restored by the destructor during unwinding
    win = 7;           // a nested scope restores the previous value, not blindly 0
    {
        whirl::ValueScope<std::int32_t> s(win, 9);
        CHECK_EQ(win, 9);
    }
    CHECK_EQ(win, 7);
}

}  // namespace

// the old main()'s calls of this module, in their order (tests/unit/unit_tests.cpp ORDER lists all modules)
const whirl::test::Suite& unitSuite_model_core() {
    static const whirl::test::Suite s{"model.core",
                                      {
                                          {"testValueScope", [] { testValueScope(); }},
                                          {"testDflashPolicy", [] { testDflashPolicy(); }},
                                      },
                                      nullptr};
    return s;
}

#ifndef WHIRL_UNIT_ALL  // this module's own program (cmake/unit.cmake whirl-unit-*-tests, or an isolated build of this file)
int main(int argc, char** argv) {
    return whirl::test::runSuites({&unitSuite_model_core()}, argc, argv, [] { whirl::test::clearWhirlEnv(); });
}
#endif
