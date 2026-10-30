// whirl-iso-forward-tests: the forward sources (src/model/qwen35/forward/*.cpp, modules forward.dispatch .. forward.gdn)
// and the execution context they run on (src/model/qwen35/context.cpp, module forward.state) built alone (ARCH-1 8.5,
// arch025-lane2 leg 3), with their declared dependencies: the op planners (src/ops), the kernel ABI tables, the GGUF
// reader, the device contract (src/device/backend_hip.cpp), the launch log, lib whirl's util + the platform layer it
// uses, and a link double of the HIP layer (fake_hip.cpp: every device call throws). No class Model, no server: a
// forward source that calls Model::load, the engine or the server log is LNK2019 here, and one that includes
// whirl/model.h or server/engine.h is C1083. The tests are the host-only parts of forward.state.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/qwen35/context.h"

#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "forward_state_cases.inc"  // arch025-n8b2: the forward.state cases (moved verbatim)

namespace {

namespace q = whirl::qwen35;

// arch025-mutcat (PLAN 5.5, mutant x12-f4b-settokens): a batch of up to max_small_batch tokens takes the kernel-arg
// path (setTokensDev), which writes the host row state (row_n / row_pos / row_base) that attention's rows_known reads
// (forward/attn.cpp: rows_known = row_n == n -> grouped ws4g); the host upload path above it does not. The state is
// written before the launch, so the link double's refusal (every device call throws) still leaves it to be read here.
}  // namespace

// Link doubles for this test program only: Context's TuneTable member defaults (src/model/qwen35/tune_table.h;
// defined in src/model/config.cpp, the model core, which is not a forward dependency). Only this test's Context
// constructor uses them (the control build linked without them before testSetTokensSmallBatchRows), and no value is read.
namespace whirl::qwen35 {
GemvR defaultGemvR() { return {}; }
GemvW defaultGemvW() { return {}; }
GemvR defaultGemvWHead() { return {}; }
}  // namespace whirl::qwen35

namespace {

struct TokenProbe : q::Context {
    using q::Context::setTokens;
};

void testSetTokensSmallBatchRows() {
    for (const std::uint32_t n : {1u, q::max_small_batch}) {
        auto c = std::make_unique<TokenProbe>();
        c->host_ids.assign(2 * 64, 0);
        c->row_n = 3;
        const std::vector<std::uint32_t> toks(n, 7);
        bool refused = false;
        try {
            c->setTokens(0, toks, 100);
        } catch (const std::logic_error&) {
            refused = true;  // the launch (or, wrongly, the upload) reaches the link double
        }
        ISO_CHECK(refused);
        ISO_CHECK(c->row_n == n);  // the rows of this batch are known: rows_known holds for it
        ISO_CHECK(c->row_pos[0] == 100 && c->row_pos[n - 1] == static_cast<std::int32_t>(100 + n - 1));
    }
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("forward",
                    {{"tune_bucket_of_each_size", testTuneBucketOfEachSize},
                     {"tune_bucket_edges", testTuneBucketEdges},
                     {"op_class_names", testOpClassNames},
                     {"profile_host_only", testProfileHostOnly},
                     {"draft_window_arg", testDraftWindowArg},
                     {"set_tokens_small_batch_rows", testSetTokensSmallBatchRows}},
                    argc, argv);
}
