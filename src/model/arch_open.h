// The architecture module of an opened GGUF, for whirl.exe and whirl-server (arch025-N15d; the server's buildModel
// did this inline, the CLI called qwen35::Config::fromGguf directly): general.architecture -> model::findArch (the
// table of src/model/arch_registry.h), an unsupported name -> ModelError UnsupportedArch with the table's text (the
// same code and text Config::fromGguf throws), else the module and its config (ArchModule::fromGguf).
// Header-only: both callers already include whirl/model.h (ModelError).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "model/arch_registry.h"
#include "whirl/gguf.h"
#include "whirl/model.h"

#include <string_view>

namespace whirl::model {

struct OpenedArch {
    const ArchModule* am = nullptr;  // never null
    qwen35::Config cfg;
};

inline OpenedArch openArch(const gguf::File& f, const ArchTable& table = builtinArchs()) {
    const std::string_view name = f.getStringOr("general.architecture", "");
    const ArchModule* const am = findArch(name, table);
    if (!am) throw qwen35::ModelError("UnsupportedArch", unsupportedArchText(name, table));
    return {am, am->fromGguf(f)};
}

}  // namespace whirl::model
