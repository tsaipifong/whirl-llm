// whirl-iso-server-drafter-ctx-tests: the header-only module server.drafter_ctx (src/server/engine_drafter.h:
// DrafterCtx, the DFlash2 drafter context of a slot; arch025-engine S4, the bodies moved verbatim out of
// engine_decode.cpp:32-58 / engine.h:223-227 at 7ad196c) compiled with only its declared dependency headers (ARCH-1 8.5
// isolation). DrafterCtx owns nothing: it holds BatchStats& bst_, ServerModel& m_ and const EngineOptions& opt_ (the
// core.shared members it uses, same names). The checks pin what the move must keep, with expected values written out
// (not recomputed through whirl/dflash_policy.h): draftRoom's three cases, dflReset's guard / trim / range, dflIngest's
// guards, the captured / partly captured / not captured plans (the model calls, the slot range, bst_.dfl_gap
// accumulating) and dflReady's guard and warm-up rule.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "iso/fake_server_model.h"
#include "server/engine_drafter.h"

#include "device/null_backend.h"

#include <type_traits>

namespace {

namespace s = whirl::server;
using Fake = whirl::test::FakeServerModel;

// the layout rule of DESIGN v3 2.5: a logical object of references, trivially destroyed (~Engine destroys it first)
static_assert(std::is_trivially_destructible_v<s::DrafterCtx>);
static_assert(std::is_reference_v<decltype(s::DrafterCtx::bst_)> && std::is_reference_v<decltype(s::DrafterCtx::m_)> &&
              std::is_reference_v<decltype(s::DrafterCtx::opt_)>);
static_assert(std::is_same_v<decltype(s::DrafterCtx::bst_), s::BatchStats&>);
static_assert(std::is_same_v<decltype(s::DrafterCtx::m_), s::ServerModel&>);
static_assert(std::is_same_v<decltype(s::DrafterCtx::opt_), const s::EngineOptions&>);

struct Rig {
    whirl::device::NullBackend ops;
    Fake m{ops, 16};
    s::BatchStats bst;
    s::EngineOptions opt;
    s::DrafterCtx dc{bst, m, opt};
};

void testDraftRoom() {
    Rig r;
    r.m.dfl_max_drafts = 15;
    r.opt.n_draft = 3;
    ISO_CHECK(r.dc.draftRoom() == 0);          // neither MTP nor DFlash2
    r.opt.use_dflash = true;
    ISO_CHECK(r.dc.draftRoom() == 15);         // DFlash2: the drafter's block - 1
    r.opt.use_mtp = true;
    ISO_CHECK(r.dc.draftRoom() == 3);          // MTP first (n_draft), even with use_dflash set
    r.opt.use_dflash = false;
    r.opt.n_draft = 5;
    ISO_CHECK(r.dc.draftRoom() == 5);
    ISO_CHECK(r.m.dfl_calls.empty());
}

void testReset() {
    Rig r;
    s::Slot sl;
    sl.id = 7;
    sl.dfl_lo = 11;
    sl.dfl_hi = 40;
    r.dc.dflReset(sl, 500);                    // no DFlash2: nothing happens
    ISO_CHECK(sl.dfl_lo == 11 && sl.dfl_hi == 40 && r.m.dfl_calls.empty());
    r.opt.use_dflash = true;
    r.dc.dflReset(sl, 500);                    // trim the whole context, the run restarts at the restore point
    ISO_CHECK(sl.dfl_lo == 500 && sl.dfl_hi == 500);
    ISO_CHECK(r.m.dfl_calls.size() == 1 && r.m.dfl_calls[0].kind == 't' && r.m.dfl_calls[0].s == 7 && r.m.dfl_calls[0].a == 0);
    r.dc.dflReset(sl, 0);
    ISO_CHECK(sl.dfl_lo == 0 && sl.dfl_hi == 0 && r.m.dfl_calls.size() == 2 && r.m.dfl_calls[1].a == 0);
}

void testIngest() {
    Rig r;
    r.m.capture_n = 4;
    r.bst.dfl_gap = 5;
    s::Slot sl;
    sl.id = 2;
    sl.dfl_lo = sl.dfl_hi = 100;
    r.dc.dflIngest(sl, 10, 6, 4, 100);         // no DFlash2: nothing
    ISO_CHECK(r.m.dfl_calls.empty() && sl.dfl_lo == 100 && sl.dfl_hi == 100 && r.bst.dfl_gap == 5);
    r.opt.use_dflash = true;
    r.dc.dflIngest(sl, 10, 6, 0, 100);         // n == 0: nothing
    ISO_CHECK(r.m.dfl_calls.empty() && sl.dfl_hi == 100 && r.bst.dfl_gap == 5);
    // rows 6..9 of a 10-row verify, the last 4 captured: capture rows 0..3 at positions 100..103, run extends
    r.dc.dflIngest(sl, 10, 6, 4, 100);
    ISO_CHECK(r.m.dfl_calls.size() == 1);
    ISO_CHECK(r.m.dfl_calls[0].kind == 'i' && r.m.dfl_calls[0].s == 2 && r.m.dfl_calls[0].a == 0 && r.m.dfl_calls[0].b == 4 &&
              r.m.dfl_calls[0].c == 100);
    ISO_CHECK(sl.dfl_lo == 100 && sl.dfl_hi == 104 && r.bst.dfl_gap == 5);
    // a run that ends at 90 and rows for 100..103: 10 positions without features (counted), the run restarts at 100
    sl.dfl_lo = 50;
    sl.dfl_hi = 90;
    r.dc.dflIngest(sl, 10, 6, 4, 100);
    ISO_CHECK(r.m.dfl_calls.size() == 2 && r.m.dfl_calls[1].a == 0 && r.m.dfl_calls[1].b == 4 && r.m.dfl_calls[1].c == 100);
    ISO_CHECK(sl.dfl_lo == 100 && sl.dfl_hi == 104 && r.bst.dfl_gap == 15);
    // partly captured: rows 4..9 of a 10-row run, capture holds rows 6..9 -> capture rows 0..3 at positions 206..209
    sl.dfl_lo = 200;
    sl.dfl_hi = 204;
    r.dc.dflIngest(sl, 10, 4, 6, 204);
    ISO_CHECK(r.m.dfl_calls.size() == 3 && r.m.dfl_calls[2].a == 0 && r.m.dfl_calls[2].b == 4 && r.m.dfl_calls[2].c == 206);
    ISO_CHECK(sl.dfl_lo == 206 && sl.dfl_hi == 210 && r.bst.dfl_gap == 17);
    // none of the rows captured (rows 0..2, capture 6..9): a gap of 3, no model call, the range is kept
    r.dc.dflIngest(sl, 10, 0, 3, 210);
    ISO_CHECK(r.m.dfl_calls.size() == 3 && sl.dfl_lo == 206 && sl.dfl_hi == 210 && r.bst.dfl_gap == 20);
}

void testReady() {
    Rig r;
    r.opt.dfl_warm = 64;
    s::Slot sl;
    sl.pos = 300;
    sl.dfl_lo = 0;
    sl.dfl_hi = 300;
    ISO_CHECK(!r.dc.dflReady(sl));             // no DFlash2
    r.opt.use_dflash = true;
    ISO_CHECK(r.dc.dflReady(sl));
    sl.dfl_hi = 299;                           // the run does not end at pos
    ISO_CHECK(!r.dc.dflReady(sl));
    sl.dfl_lo = 240;                           // 60 rows < warm 64
    sl.dfl_hi = 300;
    ISO_CHECK(!r.dc.dflReady(sl));
    sl.dfl_lo = 236;                           // 64 rows
    ISO_CHECK(r.dc.dflReady(sl));
    sl.pos = sl.dfl_lo = sl.dfl_hi = 0;        // pos 0: min(pos, warm) = 0
    ISO_CHECK(r.dc.dflReady(sl));
    r.opt.dfl_warm = 70;
    sl.pos = sl.dfl_hi = 300;
    sl.dfl_lo = 236;
    ISO_CHECK(!r.dc.dflReady(sl));             // the option is read live (a reference)
    ISO_CHECK(r.m.dfl_calls.empty());
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.drafter_ctx",
                    {{"draft_room", testDraftRoom},
                     {"reset", testReset},
                     {"ingest", testIngest},
                     {"ready", testReady}},
                    argc, argv);
}
