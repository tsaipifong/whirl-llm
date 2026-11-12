// whirl-iso-server-sample-buffers-tests: the server engine's sampling buffers (src/server/engine_sampling.cpp, class
// SampleBuffers) built with only its declared dependencies: the sampling module (src/sampling, whirl/spec_sample) and the
// device / server contracts it reaches through ServerModel& and device::Backend& (headers; the host-memory NullBackend
// and tests/iso/fake_server_model.h as test doubles) (ARCH-1 8.5 isolation, arch025-lane1 leg 8). No engine, no HIP, no
// server log, no tokenizer. The checks are whirl-server-tests sample_buffers (tests/server/server_tests.cpp
// testSampleBuffers) on a 20000-token vocabulary (> k_big, so the escalation to the full row is reachable).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "iso/fake_server_model.h"
#include "server/engine_sampling.h"
#include "device/null_backend.h"
#include "sampling/sampler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>

using namespace whirl;
using namespace whirl::server;
using whirl::device::NullBackend;
using whirl::sampling::Sampler;
using whirl::test::FakeServerModel;

namespace {

constexpr std::uint32_t k_vocab = 20000;

void testAlloc() {
    NullBackend nb;
    FakeServerModel model(nb, k_vocab);
    const std::uint32_t V = model.cfg().n_vocab;
    ISO_CHECK(SampleBuffers::sampBytes() == static_cast<std::size_t>(max_rows) * k_small * 8 + max_rows * 8);
    for (const std::uint32_t topk : {0u, 4u}) {
        SampleBuffers sb(model, nb);
        sb.alloc(3, topk > 0, topk);
        ISO_CHECK(sb.sampHost().size() == SampleBuffers::sampBytes());
        ISO_CHECK(sb.sampDev() != 0);
        ISO_CHECK(sb.rowHost().size() == static_cast<std::size_t>(V));
        ISO_CHECK(sb.cands()->size() == std::max<std::size_t>(V, k_big));
        ISO_CHECK(sb.ctlHost().size() == static_cast<std::size_t>(3) * qwen35::ctl_words);
        const std::uint64_t rb = topk ? static_cast<std::uint64_t>(max_rows) * topk * 8 + max_rows * 8 : 0;
        ISO_CHECK(static_cast<std::uint64_t>(sb.relaxHost().size()) == rb);
        ISO_CHECK((sb.relaxDev() != 0) == (topk > 0));
        ISO_CHECK(sb.nVocab() == V);
        ISO_CHECK(sb.logitsRow(5) == model.logits() + static_cast<std::uint64_t>(5) * V * 4);
        ISO_CHECK(sb.inCands() != nullptr);
    }
}

void testRows() {
    NullBackend nb;
    FakeServerModel model(nb, k_vocab);
    const std::uint32_t V = model.cfg().n_vocab;
    // row reads on known logits: rows 0..2 (row 1 flat -> the top-k_small set stays open and escalates)
    SampleBuffers sb(model, nb);
    sb.alloc(3, false, 0);
    auto* lg = reinterpret_cast<float*>(model.logits());
    for (std::uint32_t r = 0; r < 3; ++r)
        for (std::uint32_t j = 0; j < V; ++j)
            lg[static_cast<std::size_t>(r) * V + j] = r == 1 ? 0.0f : static_cast<float>((j * 7919u + r * 31u) % 1000u) * 0.01f;
    ISO_CHECK(sb.logitAt(2, 17) == lg[2 * static_cast<std::size_t>(V) + 17]);
    const float* full = sb.loadFullRow(2);
    ISO_CHECK(std::equal(full, full + V, lg + 2 * static_cast<std::size_t>(V)));
    for (std::uint32_t r = 0; r < 3; ++r) {
        Sampler s;
        s.cands = sb.cands();
        s.temp = 1.0;
        s.inv_t = 1.0f;
        s.top_p = 0.9;
        sb.fetchCands(s, 3);
        ISO_CHECK(s.row_base == 0u);
        sb.prepareRow(s, r);
        ISO_CHECK(s.row == r);
        const float* x = lg + static_cast<std::size_t>(r) * V;
        ISO_CHECK(s.m == *std::max_element(x, x + V));
        ISO_CHECK(s.n > 0);
        ISO_CHECK(s.big_used <= 1 && s.full_used <= s.big_used);  // escalation: at most once each, full only after big
        if (r != 1) ISO_CHECK((*s.cands)[0].id == static_cast<std::uint32_t>(std::max_element(x, x + V) - x));
    }
}

// v0252-e1p1c (PLAN V1-93 P1c, the E1 one-cycle lookahead; WHIRL_E1=2): the ring of two control-word areas in pinned
// memory, one event and one hidden-row keep per entry.
//   allocAhead     only with a device accept mode (allocSpecAcc >= 2); two distinct entries, each `parallel` control areas
//   aheadSubmit    the accept kernel gets entry e's area as its host pointer (zero copy: no readCtl, no download); the
//                  cycle's hidden rows go to the sequence's MTP input and to entry e's keep (2 copies); ONE eventRecord
//   aheadWait      one eventSync of entry e's event, then entry e's words - an entry written by a later submit of the
//                  OTHER entry is not touched (the ring is why cycle n + 1 may finish before the host reads cycle n)
// The Backend calls are read off NullBackend::callThreads (no eventDone, no download, no streamSync in any of them).
void testAheadRing() {
    using Call = NullBackend::Call;
    const auto count = [](const NullBackend& b, Call c) {
        std::size_t n = 0;
        for (const auto& [k, t] : b.callThreads())
            if (k == c) ++n;
        return n;
    };
    NullBackend nb;
    FakeServerModel model(nb, k_vocab);
    const std::uint64_t E = model.cfg().n_embd;
    // an argument object the fake never reads (this module sees SpecAccArgs only by name: kernels/spec_accept.h is not
    // one of its headers)
    alignas(16) static unsigned char raw[4096] = {};
    const SpecAccArgs& aa = *reinterpret_cast<const SpecAccArgs*>(raw);
    {
        // no device accept mode: no ring
        SampleBuffers sb(model, nb);
        sb.alloc(2, false, 0);
        ISO_CHECK(!sb.aheadOk());
        ISO_CHECK(sb.allocSpecAcc(2, 1) == 1u);
        ISO_CHECK(!sb.allocAhead(2, 4) && !sb.aheadOk());
    }
    model.spec_ok = true;
    SampleBuffers sb(model, nb);
    sb.alloc(2, false, 0);
    ISO_CHECK(sb.allocSpecAcc(2, 3) == 3u);
    ISO_CHECK(sb.allocAhead(2, 4) && sb.aheadOk());
    ISO_CHECK(sb.aheadHid(0) != 0 && sb.aheadHid(1) != 0 && sb.aheadHid(0) != sb.aheadHid(1));
    ISO_CHECK(sb.aheadHid(1) >= sb.aheadHid(0) + 4 * E * 4 || sb.aheadHid(0) >= sb.aheadHid(1) + 4 * E * 4);  // 4 rows each
    // hidden rows of a "verify": row r of the model's hn holds the value 100 + r in every element
    auto* hn = reinterpret_cast<float*>(model.hn());
    for (std::uint32_t r = 0; r < 4; ++r)
        for (std::uint64_t j = 0; j < E; ++j) hn[r * E + j] = static_cast<float>(100 + r);
    nb.recordThreads(true);
    model.spec_mark = 11;
    sb.aheadSubmit(0, aa, 0, 1, 3);  // rows 1 .. 3 of the verify
    ISO_CHECK(model.spec_hosts.size() == 1 && model.spec_hosts[0] != nullptr);
    ISO_CHECK(count(nb, Call::event_record) == 1 && count(nb, Call::copy_async) == 2);
    const auto* keep0 = reinterpret_cast<const float*>(sb.aheadHid(0));
    ISO_CHECK(keep0[0] == 101.0f && keep0[E] == 102.0f && keep0[2 * E + E - 1] == 103.0f);
    // the next cycle goes to the other entry before the first one is read
    for (std::uint64_t j = 0; j < 4 * E; ++j) hn[j] = 7.0f;
    model.spec_mark = 22;
    sb.aheadSubmit(1, aa, 0, 0, 2);
    ISO_CHECK(model.spec_hosts.size() == 2 && model.spec_hosts[1] != nullptr && model.spec_hosts[1] != model.spec_hosts[0]);
    // the two areas do not overlap: `parallel` control areas each
    const std::ptrdiff_t gap = model.spec_hosts[1] - model.spec_hosts[0];
    ISO_CHECK(gap >= static_cast<std::ptrdiff_t>(2 * qwen35::ctl_words) || -gap >= static_cast<std::ptrdiff_t>(2 * qwen35::ctl_words));
    ISO_CHECK(count(nb, Call::event_record) == 2 && count(nb, Call::copy_async) == 4 && count(nb, Call::event_sync) == 0);
    ISO_CHECK(keep0[0] == 101.0f);  // entry 0's keep is not the one the second cycle wrote
    ISO_CHECK(reinterpret_cast<const float*>(sb.aheadHid(1))[0] == 7.0f);
    const std::int32_t* w0 = sb.aheadWait(0);
    ISO_CHECK(w0 == model.spec_hosts[0] && w0[0] == 11);
    ISO_CHECK(count(nb, Call::event_sync) == 1);
    const std::int32_t* w1 = sb.aheadWait(1);
    ISO_CHECK(w1 == model.spec_hosts[1] && w1[0] == 22);
    ISO_CHECK(count(nb, Call::event_sync) == 2);
    // E1 rules R-a / R-c / R-e: no event query, no D2H copy command, no stream sync
    ISO_CHECK(count(nb, Call::event_done) == 0 && count(nb, Call::event_done_checked) == 0);
    ISO_CHECK(count(nb, Call::download) == 0 && count(nb, Call::stream_sync) == 0);
    // the synchronous gpu-zc read keeps using the first area (WHIRL_E1=2's synchronous cycles)
    model.spec_mark = 33;
    const std::int32_t* ws = sb.readCycleCtl(3, 2 * qwen35::ctl_words, &aa);
    ISO_CHECK(ws != nullptr && ws[0] == 33 && model.spec_hosts.size() == 3);
    nb.recordThreads(false);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.sample_buffers", {{"alloc", testAlloc}, {"rows", testRows}, {"ahead_ring", testAheadRing}}, argc, argv);
}
