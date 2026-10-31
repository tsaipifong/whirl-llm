// whirl-iso-forward-run-tests: the module forward.run (src/model/qwen35/forward/run.cpp: Model::forward / step, the
// device-resident decode step, the layer loop run, lendScratch) built over the forward files it calls - mtp_verify (which calls
// run.cpp back: the cycle is declared with the REAL file, progress/arch025-n8b3.PRIOR_ART.md design 2), attn, ffn, gdn, dispatch,
// vision_rows - and forward.state, WITHOUT prefill.cpp (ARCH-1 8.5 isolation, arch025-n8b3, PLAN N8 batch 3): a call into
// prefill.cpp, class Model, the engine or the server log is LNK2019 here. Host-only cases (argument checks before any launch,
// the scratch list lent to the vision encoder) on tests/iso/fake_hip.cpp.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Link doubles for this test program only (as tests/iso/forward.state): Context's TuneTable member defaults (model.core).
namespace whirl::qwen35 {
GemvR defaultGemvR() { return {}; }
GemvW defaultGemvW() { return {}; }
GemvR defaultGemvWHead() { return {}; }
}  // namespace whirl::qwen35

namespace {

namespace q = whirl::qwen35;

struct RunProbe : q::Context {
    using q::Context::attn_out;  // Scratch (a protected base of Context)
    using q::Context::ffn_g;
    using q::Context::ff_scratch;
    using q::Context::ffn_u;
    using q::Context::qf;
};

std::string codeOf(const std::function<void()>& f) {
    try {
        f();
    } catch (const q::ModelError& e) {
        return e.code();
    } catch (const std::logic_error&) {
        return "device";  // reached the HIP link double: the arguments were accepted
    }
    return "";
}

std::unique_ptr<RunProbe> probe() {
    auto c = std::make_unique<RunProbe>();
    c->max_batch = 8;
    c->max_ctx = 100;
    c->host_ids.assign(2 * 64, 0);
    return c;
}

void testForwardArgs() {
    auto c = probe();
    const std::vector<std::uint32_t> none, one(1, 3), full(8, 3), over(9, 3);
    ISO_CHECK(codeOf([&] { c->forward(none, 0); }) == "BatchTooLarge");
    ISO_CHECK(codeOf([&] { c->forward(over, 0); }) == "BatchTooLarge");
    ISO_CHECK(codeOf([&] { c->forward(full, 0); }) == "device");  // max_batch rows are allowed
    ISO_CHECK(codeOf([&] { c->forward(full, 92); }) == "device");  // ends exactly at max_ctx
    ISO_CHECK(codeOf([&] { c->forward(full, 93); }) == "ContextTooLong");
    ISO_CHECK(codeOf([&] { c->forward(one, 99); }) == "device");
}

void testStep() {
    auto c = probe();
    ISO_CHECK(codeOf([&] { c->step(3, 99); }) == "device");  // one token at the last position
    ISO_CHECK(codeOf([&] { c->step(3, 100); }) == "ContextTooLong");
}

void testLendScratch() {
    auto c = probe();
    c->cfg.n_embd = 16;
    c->cfg.n_ff = 32;
    c->cfg.n_head = 2;
    c->cfg.head_dim = 8;
    c->ff_scratch = 32;
    c->ffn_g = 0x1000;
    c->ffn_u = 0;  // not allocated: skipped
    c->qf = 0x3000;
    c->attn_out = 0x4000;
    std::array<std::array<std::uint64_t, 2>, 8> out{};
    const std::size_t n = c->lendScratch(out);
    ISO_CHECK(n == 3);
    ISO_CHECK(out[0][0] == 0x1000 && out[0][1] == 8ull * 32 * 4);         // ffn_g: B * ff_scratch f32
    ISO_CHECK(out[1][0] == 0x3000 && out[1][1] == 8ull * 2 * 8 * 2 * 4);  // qf: B * n_head * head_dim * 2 f32
    ISO_CHECK(out[2][0] == 0x4000 && out[2][1] == 8ull * 2 * 8 * 4);      // attn_out
    std::array<std::array<std::uint64_t, 2>, 2> two{};
    ISO_CHECK(c->lendScratch(two) == 2 && two[1][0] == 0x3000);  // at most out.size() regions
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward.run", {{"forward_args", testForwardArgs}, {"step", testStep}, {"lend_scratch", testLendScratch}}, argc, argv);
}
