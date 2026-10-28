// whirl-iso-model-arch-registry-tests: src/model/arch_registry.cpp built alone, with the header-only types it declares
// (model_types.h, kv_format.h, spec/policy.h types) (ARCH-1 8.5 isolation, arch025-isoall). The product table
// (builtinArchs, src/model/qwen35/module.cpp) is NOT part of the module: every call passes its own table.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/arch_registry.h"

#include <array>
#include <string>

using namespace whirl;

namespace {

const std::array<model::ArchName, 2> kNamesA{{{"famA", false}, {"famAmoe", true}}};
const std::array<model::ArchName, 1> kNamesB{{{"famB", false}}};
const std::array<model::ArchModule, 2> kTable{{{"A", kNamesA}, {"B", kNamesB}}};

void testLookup() {
    const model::ArchTable t(kTable);
    ISO_CHECK(model::findArch("famAmoe", t) == &kTable[0]);
    ISO_CHECK(model::findArch("famB", t) == &kTable[1]);
    ISO_CHECK(model::findArch("llama", t) == nullptr);
    const model::ArchName* n = model::findArchName("famAmoe", t);
    ISO_CHECK(n != nullptr && n->moe);
}

void testText() {
    const model::ArchTable t(kTable);
    ISO_CHECK(model::supportedArchs(t) == "famA, famAmoe and famB");
    ISO_CHECK(model::unsupportedArchText("llama", t) ==
              "general.architecture = 'llama' (WHIRL supports famA, famAmoe and famB only)");
}

}  // namespace

int main(int argc, char** argv) { return iso::run("model.arch_registry", {{"lookup", testLookup}, {"text", testText}}, argc, argv); }
