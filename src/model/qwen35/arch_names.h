// The qwen35 family's GGUF architecture names (arch025-lane2 leg 5): data, below both users. Config::fromGguf
// (src/model/config.cpp, model.core) checks a GGUF's general.architecture against them and takes the variant (moe) from
// them; the product architecture table (src/model/qwen35/module.cpp, model.qwen35) lists the same array in its row.
// SPDX-License-Identifier: Apache-2.0
//
// Why here and not through builtinArchs(): module.cpp's row points at Config::fromGguf, so a config reader that names
// the product table makes the model core and the family module need each other at link time. As in llama.cpp
// (src/llama-arch.cpp LLM_ARCH_NAMES: names are a data table the loader checks; the per-arch code is a separate switch),
// the names are data the model core owns and the table above it reuses (progress\arch025-lane2.PRIOR_ART.md §8).
// With one family the names-only table below lists exactly the product table's names, so the UnsupportedArch text is
// the same bytes (whirl-model-tests testArchRegistry pins both facts).

#pragma once

#include "model/arch_registry.h"  // ArchName, ArchModule, ArchTable

namespace whirl::qwen35 {

// {"qwen35", dense SwiGLU FFN}, {"qwen35moe", MoE FFN}; one definition for every TU (inline variable)
inline constexpr model::ArchName arch_names[] = {{"qwen35", false}, {"qwen35moe", true}};

// The family's names as a one-row table without functions, for findArchName / unsupportedArchText.
inline constexpr model::ArchModule arch_names_table[] = {{"qwen35", arch_names}};

}  // namespace whirl::qwen35
