// The qwen35 family module (arch025-M5, ADR-0004 decision 2): the product's architecture table. One family serves the
// GGUF names qwen35 (dense SwiGLU FFN) and qwen35moe (MoE FFN); its config reader, tensor check and pre-load sizing are
// the qwen35 functions the server called directly before M5.
// SPDX-License-Identifier: Apache-2.0
//
// The table is explicit (no static self-registration, src/model/arch_registry.h). A new family adds its directory
// (src/model/<family>/) and one row here. The row's load entry (arch025-lane2 leg 6) is qwen35::Model::load;
// loadServerModel (the server's load) is defined here too, because it hands the server the qwen35 model.

#include "model/arch_registry.h"
#include "model/qwen35/arch_names.h"  // the family's GGUF names (model.core's config reader checks the same array)
#include "whirl/model.h"

namespace whirl::model {

namespace {

void validateQwen35(const qwen35::Config& cfg, const gguf::File& f) { cfg.validateTensors(f); }

// the load entry (arch025-lane2 leg 6, X-16): qwen35::Model::load as whirl-server called it before (same arguments);
// the model goes back to the caller's std::unique_ptr in loadServerModel below
void freeQwen35(qwen35::Model* m) { delete m; }
Loaded loadQwen35(const gguf::File& f, std::uint32_t max_ctx, qwen35::LoadStats& stats, const qwen35::LoadOptions& lo) {
    Loaded r;
    r.qwen35 = Owned<qwen35::Model>(qwen35::Model::load(f, max_ctx, stats, lo).release(), &freeQwen35);
    return r;
}

constexpr ArchModule builtin_modules[] = {
    {"qwen35", qwen35::arch_names, &qwen35::Config::fromGguf, &validateQwen35,
     {&qwen35::estimateLoad, &qwen35::bufferBytes, &qwen35::resolveFdSplits, &qwen35::dflashWeightBytes, qwen35::max_small_batch,
      &qwen35::mtpDefaults},
     &loadQwen35},
};

constexpr ArchTable builtin_table{builtin_modules};

}  // namespace

const ArchTable& builtinArchs() { return builtin_table; }

std::unique_ptr<qwen35::Model> loadServerModel(const ArchModule& am, const gguf::File& f, std::uint32_t max_ctx,
                                               qwen35::LoadStats& stats, const qwen35::LoadOptions& lo) {
    Loaded r = loadArch(am, f, max_ctx, stats, lo);
    if (!r.qwen35)
        throw qwen35::ModelError("UnsupportedArch", "model family '" + std::string(am.family) + "' (general.architecture = '" +
                                                        std::string(f.getStringOr("general.architecture", "")) +
                                                        "'): its load gives no model whirl-server can run (the server runs the "
                                                        "qwen35 family)");
    return std::unique_ptr<qwen35::Model>(r.qwen35.release());
}

}  // namespace whirl::model
