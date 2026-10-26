// The qwen35 family module (arch025-M5, ADR-0004 decision 2): the product's architecture table. One family serves the
// GGUF names qwen35 (dense SwiGLU FFN) and qwen35moe (MoE FFN); its config reader, tensor check and pre-load sizing are
// the qwen35 functions the server called directly before M5.
// SPDX-License-Identifier: Apache-2.0
//
// The table is explicit (no static self-registration, src/model/arch_registry.h). A new family adds its directory
// (src/model/<family>/) and one row here.

#include "model/arch_registry.h"
#include "whirl/model.h"

namespace whirl::model {

namespace {

void validateQwen35(const qwen35::Config& cfg, const gguf::File& f) { cfg.validateTensors(f); }

constexpr ArchName qwen35_names[] = {{"qwen35", false}, {"qwen35moe", true}};

constexpr ArchModule builtin_modules[] = {
    {"qwen35", qwen35_names, &qwen35::Config::fromGguf, &validateQwen35,
     {&qwen35::estimateLoad, &qwen35::bufferBytes, &qwen35::resolveFdSplits, &qwen35::dflashWeightBytes, qwen35::max_small_batch,
      &qwen35::mtpDefaults}},
};

constexpr ArchTable builtin_table{builtin_modules};

}  // namespace

const ArchTable& builtinArchs() { return builtin_table; }

}  // namespace whirl::model
