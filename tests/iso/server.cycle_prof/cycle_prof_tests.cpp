// whirl-iso-server-cycle-prof-tests: the server's decode-cycle profile (src/server/engine_prof.*, arch025-engine S2:
// struct CycleProf and its print(), moved verbatim out of engine_internal.h) built alone (ARCH-1 8.5) with its declared
// dependencies: the server log (src/server/log.cpp, the "profile |" lines; a Log hook captures them) with what it calls
// (util.cpp, the platform layer) and the server model contract (server/backend.h headers; test double
// tests/iso/fake_server_model.h on the NullBackend). No engine, no HIP runtime.
// Pin: print()'s lines == the c1ca7ce engine_internal.h:31-55 format strings applied to the same counters (copied below
// as the reference), slot indices without cycles print nothing, every counter and last_end reset after print.
// SPDX-License-Identifier: Apache-2.0
#include "server/engine_prof.h"

#include "device/null_backend.h"
#include "iso/fake_server_model.h"
#include "iso/iso_test.h"
#include "server/log.h"

#include <array>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace sv = whirl::server;
using whirl::test::FakeServerModel;

std::vector<std::string> g_lines;
void hook(void*, std::string_view l) { g_lines.emplace_back(l); }

// the c1ca7ce body's lines for slot index a (the reference; counters as given, pm / pv the model's class times)
std::vector<std::string> reference(const sv::CycleProf& p, std::size_t a, const std::array<double, 10>& pm,
                                   const std::array<double, 10>& pv, const std::vector<const char*>& names) {
    std::vector<std::string> r;
    const double c = static_cast<double>(p.cycles[a]);
    double tm = 0, tv = 0;
    for (double v : pm) tm += v;
    for (double v : pv) tv += v;
    r.push_back(std::format(
        "profile | slots {} | {} cycles, {:.2f} rows, {:.2f} drafts/slot, {:.2f} tok/cycle | wall {:.2f} ms/cycle "
        "(pre-launch host {:.2f}, gap before cycle {:.2f}) | GPU mtp {:.2f} + verify {:.2f} ms | {:.1f} tok/s in "
        "cycles",
        a, p.cycles[a], static_cast<double>(p.rows[a]) / c, static_cast<double>(p.drafts[a]) / c / static_cast<double>(a),
        static_cast<double>(p.tokens[a]) / c, p.wall_ms[a] / c, p.host_ms[a] / c, p.gap_ms[a] / c, tm / c, tv / c,
        static_cast<double>(p.tokens[a]) * 1000.0 / (p.wall_ms[a] + p.gap_ms[a])));
    for (std::size_t i = 0; i < 10; ++i) {
        if (pm[i] > 0 || pv[i] > 0)
            r.push_back(std::format("profile | slots {} |   {:<9} mtp {:>6.3f}  verify {:>6.3f} ms/cycle", a,
                                    i < names.size() ? names[i] : "?", pm[i] / c, pv[i] / c));
    }
    r.push_back(std::format("profile | slots {} |   host: enqueue {:.3f} ms, after readCtl {:.3f} ms per cycle", a,
                            p.enq_ms[a] / c, p.post_ms[a] / c));
    return r;
}

bool endsWith(const std::string& s, const std::string& t) { return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0; }

void fillSlot(sv::CycleProf& p, std::size_t a, std::uint64_t k) {
    p.cycles[a] = 4 * k;
    p.rows[a] = 9 * k;
    p.drafts[a] = 7 * k;
    p.tokens[a] = 13 * k;
    p.wall_ms[a] = 10.25 * static_cast<double>(k);
    p.host_ms[a] = 1.5 * static_cast<double>(k);
    p.gap_ms[a] = 0.75 * static_cast<double>(k);
    p.post_ms[a] = 0.125 * static_cast<double>(k);
    p.enq_ms[a] = 0.0625 * static_cast<double>(k);
}

void testPrintLines() {
    whirl::device::NullBackend ops;
    FakeServerModel m(ops, 16);
    m.prof_names = {"gemv", "attn", "moe"};
    m.prof_mtp[0] = 2.0;
    m.prof_ver[1] = 3.5;
    m.prof_ver[2] = 0.5;
    sv::CycleProf p;
    fillSlot(p, 2, 1);  // only slot index 2 has cycles: one block of lines
    p.last_end = sv::TimePoint{};
    const std::array<double, 10> pm = m.prof_mtp, pv = m.prof_ver;
    const std::vector<std::string> want = reference(p, 2, pm, pv, m.prof_names);
    g_lines.clear();
    sv::Log::setConsole(false);
    sv::Log::setHook(hook, nullptr);
    p.print(m);
    sv::Log::setHook(nullptr, nullptr);
    ISO_CHECK(want.size() == 5);  // the summary, three class lines (gemv mtp, attn / moe verify), the host line
    ISO_CHECK(g_lines.size() == want.size());
    for (std::size_t i = 0; i < want.size() && i < g_lines.size(); ++i) ISO_CHECK(endsWith(g_lines[i], want[i]));
    // everything reset
    for (std::size_t a = 0; a < sv::CycleProf::n; ++a) {
        ISO_CHECK(p.cycles[a] == 0 && p.rows[a] == 0 && p.drafts[a] == 0 && p.tokens[a] == 0);
        ISO_CHECK(p.wall_ms[a] == 0 && p.host_ms[a] == 0 && p.gap_ms[a] == 0 && p.post_ms[a] == 0 && p.enq_ms[a] == 0);
    }
    ISO_CHECK(!p.last_end.has_value());
}

void testPrintTwoSlotsInOrder() {
    whirl::device::NullBackend ops;
    FakeServerModel m(ops, 16);
    sv::CycleProf p;
    fillSlot(p, 1, 2);
    fillSlot(p, 3, 1);
    const std::array<double, 10> z{};
    std::vector<std::string> want = reference(p, 1, z, z, {});
    for (const std::string& l : reference(p, 3, z, z, {})) want.push_back(l);
    g_lines.clear();
    sv::Log::setConsole(false);
    sv::Log::setHook(hook, nullptr);
    p.print(m);
    p.print(m);  // a second print right after: nothing left to report
    sv::Log::setHook(nullptr, nullptr);
    ISO_CHECK(want.size() == 4);  // no class lines (the model reports 0 ms)
    ISO_CHECK(g_lines.size() == want.size());
    for (std::size_t i = 0; i < want.size() && i < g_lines.size(); ++i) ISO_CHECK(endsWith(g_lines[i], want[i]));
}

void testDefaults() {
    const sv::CycleProf p;
    ISO_CHECK(sv::CycleProf::n == whirl::qwen35::gdn_max_seg + 1);
    ISO_CHECK(p.events && !p.last_end && !p.t_ready && p.cycles[0] == 0);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.cycle_prof",
                    {{"print_lines", testPrintLines}, {"print_two_slots", testPrintTwoSlotsInOrder}, {"defaults", testDefaults}},
                    argc, argv);
}
