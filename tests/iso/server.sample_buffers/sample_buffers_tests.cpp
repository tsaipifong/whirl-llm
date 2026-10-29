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

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.sample_buffers", {{"alloc", testAlloc}, {"rows", testRows}}, argc, argv);
}
