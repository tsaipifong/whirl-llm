// v0252-hostwait (PLAN V1-61): WHIRL_CYCLE_SYNC (EngineOptions::cycle_sync) changes only how a decode cycle waits for
// the GPU and when the runtime gets its launches, never what runs:
//   bit 0 pinned: the control words (ServerModel::readCtlAsync) and, folded, the sampling / relaxacc readbacks go through
//                 pinned host memory with async copies on the model stream and ONE event wait per cycle (no blocking
//                 readCtl, no ops_.download in the cycle);
//   bit 1 spin:   that wait polls the event (eventDone) instead of eventSync; implies bit 0;
//   bit 2 kick:   the model records its event on its stream and queries it after every MTP step of the cycle
//                 (ServerModel::kick) and every EngineOptions::kick_layers layers of a decode / verify forward
//                 (ServerModel::setKick);
//   bit 3:        the same kicks by the event record alone, no query (v0252-hostwait2, REVIEW-v0252-hostwait A2).
// Same tokens in every setting: greedy MTP (exact acceptance), sampled MTP (specsample race), relaxacc greedy (fast, one
// pass ranks) and plain decode (no MTP) all give the cycle_sync 0 text (and the greedy reference where greedy).
// v0252-hostwait2 (REVIEW-v0252-hostwait B3): every run uses NullBackend's deferred completion (deferHostCopies): a
// readback lands only when the engine waits for the right event on the right stream, so "reads before the wait", "waits
// on the wrong event" and "copy on another stream" read stale bytes and change the text; the counts check that every
// readback landed (none left in flight) and that a spin wait really polls more than once. B4: the "batch | cycle sync"
// line reports what the cycles did (pinned waits, polls, blocking reads, kicks); the checks compare it with the mock's
// own counts.
// Included by server_tests.cpp inside its anonymous namespace (TestServer, baseSetup, makePrompt, completionBody ...).
// SPDX-License-Identifier: Apache-2.0

enum class CsKind { greedy_mtp, sampled_mtp, relax_mtp, plain };

const char* csKindName(CsKind k) {
    switch (k) {
    case CsKind::greedy_mtp: return "greedy_mtp";
    case CsKind::sampled_mtp: return "sampled_mtp";
    case CsKind::relax_mtp: return "relax_mtp";
    case CsKind::plain: return "plain";
    }
    return "?";
}

// the test's kick period: the mock has 8 layers, so a verify forward kicks after layers 2, 4 and 6 (run()'s rule)
constexpr std::uint32_t cs_kick_layers = 2;
constexpr std::uint64_t cs_fwd_kicks_per_verify = 3;

struct CsRun {
    std::vector<std::string> text;    // per request, in order
    std::vector<std::string> ref;     // greedy reference per request (greedy kinds; empty otherwise)
    std::uint64_t verifies = 0;       // decode cycles (one verify each)
    std::uint64_t mtp_steps = 0;      // MTP step calls (decode only)
    std::uint64_t read_ctl = 0;       // blocking control-word reads (ServerModel::readCtl)
    std::uint64_t ctl_async = 0;      // async control-word reads (ServerModel::readCtlAsync)
    std::uint32_t kick_layers = 0;    // what the engine set on the model
    bool kick_query = true;           // ... and whether its kicks query
    std::uint64_t kick_calls = 0, kick_fwd = 0;  // the model's kicks: engine calls, inside verify forwards
    std::uint64_t downloads = 0, event_sync = 0, event_done = 0, event_record = 0;  // Backend calls, whole run
    std::uint64_t sync_after_record = 0;  // event_sync calls whose previous Backend call is an event_record
    std::uint64_t sync_with_copy = 0;     // event_sync calls with a copy since the previous wait
    std::uint64_t landed = 0, in_flight = 0;  // deferred readbacks landed / still in flight at the end
    std::uint64_t poison = 0, hid_mismatch = 0;
    std::uint64_t mode_lines = 0;     // "decode: cycle sync N" lines (the configured mode, once per start)
    std::vector<std::uint64_t> stat;  // the last "batch | cycle sync" line: cs, pinned, polls, readctl, kicks, calls, fwd
};

ServerSetup csSetup(CsKind k, std::uint32_t cs, bool ctl_async) {
    ServerSetup su = baseSetup(k != CsKind::plain);
    su.mc.parallel = 2;
    su.mc.ctl_async = ctl_async;
    su.eo.ngram = false;
    // a fixed draft count: every cycle of an MTP request drafts, so the relaxacc / sampling readbacks run every cycle
    su.eo.mtp_auto = false;
    su.eo.n_draft = 3;
    su.eo.kick_layers = cs_kick_layers;
    if (k == CsKind::sampled_mtp) su.eo.spec_sample = true;
    if (k == CsKind::relax_mtp) {
        su.mc.relax_rank = true;          // the one-pass ranks (v0252-f1): the relax readback is max_rows * 12 bytes
        su.mc.mtp_miss_runner_up = true;  // wrong drafts inside relaxacc's window: the relaxed rule really keeps some
        su.mc.draft_miss = 0.4;
        su.eo.relax.on = true;
    }
    su.eo.cycle_sync = cs;
    return su;
}

std::string csBody(CsKind k, std::span<const std::uint32_t> p, std::uint32_t n) {
    if (k == CsKind::sampled_mtp)
        return std::format("{{\"prompt\":{},\"max_tokens\":{},\"temperature\":0.8,\"top_k\":40,\"seed\":7,\"ignore_eos\":true}}",
                           idsJson(p), n);
    return completionBody(p, n);
}

// the unsigned numbers of a log line after its marker, in order
std::vector<std::uint64_t> csNums(const std::string& s) {
    std::vector<std::uint64_t> v;
    std::uint64_t x = 0;
    bool in = false;
    for (const char c : s) {
        if (c >= '0' && c <= '9') {
            x = x * 10 + static_cast<std::uint64_t>(c - '0');
            in = true;
        } else if (in) {
            v.push_back(x);
            x = 0;
            in = false;
        }
    }
    if (in) v.push_back(x);
    return v;
}

CsRun csRunOne(CsKind k, std::uint32_t cs, bool ctl_async = true) {
    CsRun r;
    g_log.clear();  // no engine runs between two of these runs: the lines below are this run's
    TestServer srv(csSetup(k, cs, ctl_async));
    srv.ops().deferHostCopies(true);  // B3: before any request (no copy in flight yet)
    srv.ops().recordThreads(true);
    std::uint64_t seed = 6100;
    for (std::uint32_t n : {300u, 900u}) {
        const auto p = makePrompt(n, seed++);
        auto res = srv.post("/v1/completions", csBody(k, p, 96));
        CHECK_EQ(res.status, 200);
        r.text.push_back(completionText(res));
        if (k != CsKind::sampled_mtp) r.ref.push_back(decode(srv.model().greedyReference(p, 96, true)));
    }
    // B4: the engine writes its "batch | cycle sync" line when it goes idle (Scheduler::idleWait), but a stop that comes
    // first returns from runLoop before idleWait: wait (<= 3 s) for the line that counts every cycle of this run (pinned
    // waits + blocking reads == the cycles; the model's count is final once the last response is back)
    if (((cs & 15u) | ((cs & 2u) != 0 ? 1u : 0u)) != 0) {
        const std::uint64_t cycles = srv.model().verifies();
        for (int i = 0; i < 300; ++i) {
            std::vector<std::uint64_t> st;
            {
                std::lock_guard<std::mutex> lk(g_log.mu);
                for (const auto& l : g_log.lines)
                    if (const std::size_t p = l.find("batch | cycle sync "); p != std::string::npos) st = csNums(l.substr(p));
            }
            if (st.size() == 7 && st[1] + st[3] == cycles) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    srv.stopEngine();  // the engine loop is done: every count below is final (the engine and the model still exist)
    MockModel& m = srv.model();
    r.verifies = m.verifies();
    r.mtp_steps = m.mtpSteps();
    r.read_ctl = m.readCtlCalls();
    r.ctl_async = m.ctlAsyncCalls();
    r.kick_layers = m.kickLayers();
    r.kick_query = m.kickQuery();
    r.kick_calls = m.kickCount().calls;
    r.kick_fwd = m.kickCount().forward;
    r.poison = m.poison();
    r.hid_mismatch = m.hidMismatch();
    r.landed = srv.ops().deferredLanded();
    r.in_flight = srv.ops().deferredInFlight();
    using C = NullBackend::Call;
    const auto calls = srv.ops().callThreads();
    bool copy_since_wait = false;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        const C c = calls[i].first;
        if (c == C::download) r.downloads += 1;
        if (c == C::event_record) r.event_record += 1;
        if (c == C::event_done || c == C::event_done_checked) r.event_done += 1;
        if (c == C::copy_async) copy_since_wait = true;
        if (c == C::event_sync) {
            r.event_sync += 1;
            if (i > 0 && calls[i - 1].first == C::event_record) r.sync_after_record += 1;
            if (copy_since_wait) r.sync_with_copy += 1;
            copy_since_wait = false;
        }
    }
    srv.ops().recordThreads(false);
    srv.ops().deferHostCopies(false);
    {
        std::lock_guard<std::mutex> lk(g_log.mu);
        std::string last;
        for (const auto& l : g_log.lines) {
            if (const std::size_t p = l.find("batch | cycle sync "); p != std::string::npos) last = l.substr(p);
            if (l.find("decode: cycle sync ") != std::string::npos) r.mode_lines += 1;
        }
        r.stat = csNums(last);
    }
    return r;
}

void csPrint(CsKind k, std::uint32_t cs, const CsRun& r) {
    std::printf("  cycle_sync %s cs %u: %llu cycles, %llu MTP steps | readCtl %llu, readCtlAsync %llu | download %llu, event_sync %llu "
                "(after record %llu, with copy %llu), event_done %llu, event_record %llu | landed %llu, in flight %llu | kick "
                "layers %u%s, kicks %llu + %llu in forwards | log stat %zu numbers\n",
                csKindName(k), cs, static_cast<unsigned long long>(r.verifies), static_cast<unsigned long long>(r.mtp_steps),
                static_cast<unsigned long long>(r.read_ctl), static_cast<unsigned long long>(r.ctl_async),
                static_cast<unsigned long long>(r.downloads), static_cast<unsigned long long>(r.event_sync),
                static_cast<unsigned long long>(r.sync_after_record), static_cast<unsigned long long>(r.sync_with_copy),
                static_cast<unsigned long long>(r.event_done), static_cast<unsigned long long>(r.event_record),
                static_cast<unsigned long long>(r.landed), static_cast<unsigned long long>(r.in_flight), r.kick_layers,
                r.kick_query ? " (record + query)" : " (record only)", static_cast<unsigned long long>(r.kick_calls),
                static_cast<unsigned long long>(r.kick_fwd), r.stat.size());
}

// B4: the "batch | cycle sync N" line = {N, pinned waits, event polls, blocking readCtl, kicks asked, model kicks after
// MTP steps, model kicks inside forwards}; none for an effective cycle_sync 0
void csCheckStat(const CsRun& r, std::uint32_t cs, std::uint64_t pinned, std::uint64_t readctl) {
    if (cs == 0) {
        CHECK(r.stat.empty());
        CHECK_EQ(r.mode_lines, 0u);
        return;
    }
    CHECK_EQ(r.mode_lines, 1u);
    CHECK_EQ(r.stat.size(), 7u);
    if (r.stat.size() != 7) return;
    CHECK_EQ(r.stat[0], static_cast<std::uint64_t>(cs));
    CHECK_EQ(r.stat[1], pinned);
    if ((cs & 2u) != 0) CHECK(r.stat[2] >= 2 * pinned);  // every spin wait polled at least twice (the first answer: not done)
    else CHECK_EQ(r.stat[2], 0u);
    CHECK_EQ(r.stat[3], readctl);
    CHECK_EQ(r.stat[4], r.kick_calls);  // the engine asked exactly what the model did
    CHECK_EQ(r.stat[5], r.kick_calls);
    CHECK_EQ(r.stat[6], r.kick_fwd);
}

// Same tokens in every setting, every path; the cycle count does not move either (the waits move, not the work).
// Deferred completion is on: a wait in the wrong place reads stale control words / candidates and changes the text.
void testCycleSyncTokens() {
    for (CsKind k : {CsKind::greedy_mtp, CsKind::sampled_mtp, CsKind::relax_mtp, CsKind::plain}) {
        const CsRun base = csRunOne(k, 0);
        CHECK(base.verifies > 0);
        if (k != CsKind::sampled_mtp) {
            if (k == CsKind::relax_mtp) CHECK(base.text != base.ref);  // relaxacc really kept some wrong drafts
            else CHECK(base.text == base.ref);
        }
        for (std::uint32_t cs : {1u, 2u, 3u, 4u, 5u, 7u, 8u, 9u, 11u, 15u}) {
            const CsRun r = csRunOne(k, cs);
            csPrint(k, cs, r);
            CHECK(r.text == base.text);
            CHECK_EQ(r.verifies, base.verifies);
            CHECK_EQ(r.mtp_steps, base.mtp_steps);
            CHECK_EQ(r.poison, 0u);
            CHECK_EQ(r.hid_mismatch, 0u);
            CHECK_EQ(r.in_flight, 0u);  // every readback landed at a wait
        }
    }
}

// What each bit does to the cycle's Backend / model calls (the default 0 is the old map: one blocking readCtl per cycle,
// plus one blocking download per cycle for sampling and for relaxacc).
void testCycleSyncWaits() {
    for (CsKind k : {CsKind::greedy_mtp, CsKind::sampled_mtp, CsKind::relax_mtp, CsKind::plain}) {
        const CsRun r0 = csRunOne(k, 0);
        csPrint(k, 0, r0);
        const std::uint64_t V = r0.verifies;
        const std::uint64_t S = r0.mtp_steps;
        const std::uint64_t F = V * cs_fwd_kicks_per_verify;  // forward kicks of the cycles' verify forwards
        CHECK_EQ(r0.read_ctl, V);
        CHECK_EQ(r0.ctl_async, 0u);
        CHECK_EQ(r0.event_sync, 0u);
        CHECK_EQ(r0.kick_layers, 0u);
        CHECK_EQ(r0.kick_calls + r0.kick_fwd, 0u);
        CHECK_EQ(r0.landed, 0u);
        csCheckStat(r0, 0, 0, 0);
        if (k != CsKind::plain) CHECK(S >= V);  // at least one step per MTP cycle: the kick counts are not vacuous
        // bit 0: no blocking read left in the cycle, one event wait per cycle, recorded right before it, after the copies;
        // every readback (control words + folded) landed at that wait
        const CsRun r1 = csRunOne(k, 1);
        csPrint(k, 1, r1);
        CHECK_EQ(r1.read_ctl, 0u);
        CHECK_EQ(r1.ctl_async, V);
        CHECK_EQ(r1.event_sync, V);
        CHECK_EQ(r1.sync_after_record, V);
        CHECK_EQ(r1.sync_with_copy, V);
        // the folded readbacks (sampling top-k candidates every cycle of a sampling slot; relaxacc ranks every cycle whose
        // greedy relaxacc slot drafted): each one replaced one blocking download of the old cycle
        const std::uint64_t fold = r1.landed >= V ? r1.landed - V : 0;
        CHECK(r1.landed >= V);
        CHECK_EQ(r1.downloads + fold, r0.downloads);
        if (k == CsKind::sampled_mtp) CHECK_EQ(fold, V);
        else if (k == CsKind::relax_mtp) CHECK(fold > 0 && fold <= V);
        else CHECK_EQ(fold, 0u);
        CHECK_EQ(r1.event_record, r0.event_record + V);
        CHECK_EQ(r1.landed, V + fold);
        CHECK_EQ(r1.in_flight, 0u);
        CHECK_EQ(r1.kick_layers, 0u);
        CHECK_EQ(r1.kick_calls + r1.kick_fwd, 0u);
        csCheckStat(r1, 1, V, 0);
        // bit 1 (spin, implies bit 0): the same reads, the wait polls the event (no eventSync); the first answer of every
        // cycle is "not done" (deferred completion), so each wait polls at least twice
        for (std::uint32_t cs : {2u, 3u}) {
            const CsRun r = csRunOne(k, cs);
            csPrint(k, cs, r);
            CHECK_EQ(r.read_ctl, 0u);
            CHECK_EQ(r.ctl_async, V);
            CHECK_EQ(r.event_sync, 0u);
            CHECK(r.event_done >= r0.event_done + 2 * V);
            CHECK_EQ(r.downloads + fold, r0.downloads);
            CHECK_EQ(r.landed, V + fold);
            CHECK_EQ(r.in_flight, 0u);
            csCheckStat(r, 3, V, 0);
        }
        // bit 2 (kick): a record + query after every MTP step of the decode cycles (engine) and kick_layers-periodic ones
        // inside every verify forward (model)
        const CsRun r4 = csRunOne(k, 4);
        csPrint(k, 4, r4);
        CHECK_EQ(r4.read_ctl, V);
        CHECK_EQ(r4.ctl_async, 0u);
        CHECK_EQ(r4.kick_layers, cs_kick_layers);
        CHECK(r4.kick_query);
        CHECK_EQ(r4.kick_calls, S);
        CHECK_EQ(r4.kick_fwd, F);
        CHECK_EQ(r4.event_record, r0.event_record + S + F);
        CHECK(r4.event_done >= r0.event_done + S + F);  // (other queries, e.g. the page pool's, depend on timing)
        CHECK_EQ(r4.downloads, r0.downloads);
        CHECK_EQ(r4.landed, 0u);
        csCheckStat(r4, 4, 0, V);
        // bit 3 (record-only kick, A2): the same kick points; the engine asks the model for no query
        const CsRun r8 = csRunOne(k, 8);
        csPrint(k, 8, r8);
        CHECK_EQ(r8.read_ctl, V);
        CHECK_EQ(r8.kick_layers, cs_kick_layers);
        CHECK(!r8.kick_query);
        CHECK_EQ(r8.kick_calls, S);
        CHECK_EQ(r8.kick_fwd, F);
        CHECK_EQ(r8.event_record, r0.event_record + S + F);
        CHECK_EQ(r8.downloads, r0.downloads);
        csCheckStat(r8, 8, 0, V);
        // bits 0 + 3: the pinned wait and record-only kicks (no kick query)
        const CsRun r9 = csRunOne(k, 9);
        csPrint(k, 9, r9);
        CHECK_EQ(r9.read_ctl, 0u);
        CHECK_EQ(r9.ctl_async, V);
        CHECK_EQ(r9.event_sync, V);
        CHECK_EQ(r9.event_record, r0.event_record + V + S + F);
        CHECK(!r9.kick_query);
        CHECK_EQ(r9.landed, V + fold);
        CHECK_EQ(r9.in_flight, 0u);
        csCheckStat(r9, 9, V, 0);
        // bits 0 + 1 + 2
        const CsRun r7 = csRunOne(k, 7);
        csPrint(k, 7, r7);
        CHECK_EQ(r7.read_ctl, 0u);
        CHECK_EQ(r7.ctl_async, V);
        CHECK_EQ(r7.event_sync, 0u);
        CHECK_EQ(r7.event_record, r0.event_record + V + S + F);
        CHECK(r7.kick_query);
        CHECK_EQ(r7.kick_layers, cs_kick_layers);
        CHECK_EQ(r7.landed, V + fold);
        CHECK_EQ(r7.in_flight, 0u);
        csCheckStat(r7, 7, V, 0);
    }
    // a model without readCtlAsync: bit 0 keeps the blocking reads (and the same text)
    {
        const CsRun r0 = csRunOne(CsKind::relax_mtp, 0);
        const CsRun r = csRunOne(CsKind::relax_mtp, 3, false);
        csPrint(CsKind::relax_mtp, 3, r);
        CHECK(r.text == r0.text);
        CHECK_EQ(r.read_ctl, r0.verifies);
        CHECK_EQ(r.ctl_async, 0u);
        CHECK_EQ(r.event_sync, 0u);
        CHECK_EQ(r.downloads, r0.downloads);
        CHECK_EQ(r.landed, 0u);
        csCheckStat(r, 3, 0, r0.verifies);
    }
    // unknown bits are ignored with one warning (bits above 3 do nothing: the old cycle)
    {
        const CsRun r0 = csRunOne(CsKind::greedy_mtp, 0);
        const CsRun r = csRunOne(CsKind::greedy_mtp, 16);  // g_log is this run's only
        CHECK(r.text == r0.text);
        CHECK_EQ(r.read_ctl, r0.verifies);
        CHECK_EQ(r.event_record, r0.event_record);
        CHECK_EQ(g_log.count("cycle sync: unknown bits"), 1u);
        csCheckStat(r, 0, 0, 0);
    }
}

// v0252-hostwait2 (B3): the deferred-completion model of NullBackend itself (what the cycle_sync tests rely on;
// feat026-hostgap-r1's testCycleSyncDeferModel without its blocking-call counter)
void testCycleSyncDeferModel() {
    NullBackend nb;
    nb.deferHostCopies(true);
    device::Stream s = nb.streamCreateNonBlocking();
    device::Event e = nb.eventCreate();
    const device::DevPtr dev = nb.malloc(64);
    auto* pin = static_cast<std::uint8_t*>(nb.hostMalloc(64));
    auto* d = reinterpret_cast<std::uint8_t*>(dev);
    auto fill = [&](std::uint8_t v) { std::memset(d, v, 64); };
    const auto pinAddr = static_cast<device::DevPtr>(reinterpret_cast<std::uintptr_t>(pin));
    // 1. a readback is not there before the wait; an event query answers "not done" once, then lands it
    fill(1);
    nb.copyAsync(pinAddr, dev, 64, s);
    fill(9);  // later device work on the source does not reach the copy (taken at the call)
    CHECK_EQ(pin[0], 0u);
    nb.eventRecord(e, s);
    CHECK(!nb.eventDoneChecked(e));
    CHECK_EQ(pin[0], 0u);
    CHECK(nb.eventDoneChecked(e));
    CHECK_EQ(pin[0], 1u);
    CHECK(nb.eventDone(e));  // done stays done
    // 2. streamSync lands; 3. eventSync lands only what was before the event
    fill(2);
    nb.copyAsync(pinAddr, dev, 64, s);
    nb.streamSync(s);
    CHECK_EQ(pin[0], 2u);
    nb.eventRecord(e, s);
    fill(3);
    nb.copyAsync(pinAddr, dev, 64, s);
    nb.eventSync(e);
    CHECK_EQ(pin[0], 2u);
    CHECK_EQ(nb.deferredInFlight(), 1u);
    // 4. another stream's wait lands nothing (no implicit null-stream sync); deviceReached of the right stream lands
    nb.deviceReached(nullptr);
    CHECK_EQ(pin[0], 2u);
    device::Event e0 = nb.eventCreate();
    nb.eventRecord(e0, nullptr);
    nb.eventSync(e0);
    CHECK_EQ(pin[0], 2u);
    nb.deviceReached(s);
    CHECK_EQ(pin[0], 3u);
    // 5. device-to-device copies are not deferred; a pending readback into a freed block is dropped
    const device::DevPtr dev2 = nb.malloc(64);
    nb.copyAsync(dev2, dev, 64, s);
    CHECK_EQ(reinterpret_cast<std::uint8_t*>(dev2)[0], 3u);
    nb.copyAsync(pinAddr, dev, 64, s);
    CHECK_EQ(nb.deferredInFlight(), 1u);
    nb.hostFree(pin);
    CHECK_EQ(nb.deferredInFlight(), 0u);
    CHECK_EQ(nb.deferredLanded(), 3u);
    // 6. off: immediate, as before
    nb.deferHostCopies(false);
    auto* pin2 = static_cast<std::uint8_t*>(nb.hostMalloc(64));
    nb.copyAsync(static_cast<device::DevPtr>(reinterpret_cast<std::uintptr_t>(pin2)), dev, 64, s);
    CHECK_EQ(pin2[0], 3u);
    nb.hostFree(pin2);
    nb.free(dev2);
    nb.free(dev);
    nb.eventDestroy(e0);
    nb.eventDestroy(e);
    nb.streamDestroy(s);
    CHECK(nb.clean());
}
