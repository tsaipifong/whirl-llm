// whirl-iso-model-qwen35-arch-tests: the product architecture table (src/model/qwen35/module.cpp, model.qwen35.arch)
// built alone (ARCH-1 8.5, arch025-lane2 leg 5) with the model core and what is below it, spec.policy and the HIP link
// double (tests/iso/fake_hip.cpp): no server, no engine, no ServerModel adapter. The table's one row serves the qwen35
// family's names (the array of qwen35/arch_names.h that model.core's Config::fromGguf checks) with the qwen35 functions.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/arch_registry.h"
#include "model/qwen35/arch_names.h"
#include "whirl/model.h"

#include <iterator>
#include <string>

// the embedded draft vocabulary (whirl-bin2c output in the product build; draft_vocab_embed.cpp of the model core
// references it): one id, little-endian u32, as in tests/iso/model.core
extern const unsigned char whirl_draft_vocab_64k[] = {0x01, 0x00, 0x00, 0x00};
extern const unsigned long long whirl_draft_vocab_64k_size = sizeof(whirl_draft_vocab_64k);

namespace {

namespace m = whirl::model;
namespace q = whirl::qwen35;

void testRow() {
    const m::ArchTable& t = m::builtinArchs();
    ISO_CHECK(t.size() == 1 && t[0].family == "qwen35");
    // the row lists the family's names array itself (one definition: arch_names.h's inline variable)
    ISO_CHECK(t[0].names.data() == q::arch_names && t[0].names.size() == std::size(q::arch_names));
    ISO_CHECK(t[0].fromGguf == &q::Config::fromGguf && t[0].validateTensors != nullptr);
    ISO_CHECK(t[0].sizing.estimateLoad == &q::estimateLoad && t[0].sizing.bufferBytes == &q::bufferBytes &&
              t[0].sizing.resolveFdSplits == &q::resolveFdSplits && t[0].sizing.dflashWeightBytes == &q::dflashWeightBytes &&
              t[0].sizing.mtpDefaults == &q::mtpDefaults && t[0].sizing.max_small_batch == q::max_small_batch);
}

void testLookups() {
    ISO_CHECK(m::findArch("qwen35") == &m::builtinArchs()[0] && m::findArch("qwen35moe") == &m::builtinArchs()[0]);
    ISO_CHECK(m::findArch("llama") == nullptr);
    // the product table and model.core's names-only family table give the same message (one family)
    ISO_CHECK(m::unsupportedArchText("llama") ==
              "general.architecture = 'llama' (WHIRL supports qwen35 and qwen35moe only)");
    ISO_CHECK(m::unsupportedArchText("llama") == m::unsupportedArchText("llama", m::ArchTable(q::arch_names_table)));
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("model.qwen35.arch", {{"row", testRow}, {"lookups", testLookups}}, argc, argv);
}
