// Architecture registry (arch025-M5; ADR-0004 decision 1, PLAN C2 / C3): the GGUF `general.architecture` names WHIRL
// loads, the family module that serves each, and the module's load-time sizing (the server's pre-load VRAM fit).
// SPDX-License-Identifier: Apache-2.0
//
// Host-only library whirl_arch (cmake/arch.cmake): the server core and the model library link it. builtinArchs() is the
// explicit table of the model library (src/model/qwen35/module.cpp); tests pass their own table to findArch, so the CPU
// test executables never need the model library. No static self-registration: whirl_model is a STATIC library, and an
// object nothing references would be dropped by the linker (progress\arch025-M5.PRIOR_ART.md §2).
//
// What goes through the module in 0.2.5: the arch-name check and the config (Config::fromGguf takes moe from the table),
// and the server's pre-load sizing (server_main.cpp). Model::load and everything after it still use qwen35::Model
// directly, and the CLI's applyRuntimeEnv writes Model members (ADR-0004 "Honest limit"; the C line moves the rest).

#pragma once

#include "spec/policy.h"        // MtpDefaults
#include "whirl/kv_format.h"    // KvFormat
#include "whirl/model_types.h"  // qwen35::Config, qwen35::LoadEstimate, hip::Arch

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace whirl::gguf {
class File;
}  // namespace whirl::gguf

namespace whirl::model {

// One GGUF architecture name and the variant of its family it selects.
struct ArchName {
    std::string_view name;
    bool moe = false;  // the family's MoE variant (qwen35moe)
};

// The pre-load sizing a family module serves (the server's VRAM fit before Model::load; same signatures as qwen35's).
struct ArchSizing {
    qwen35::LoadEstimate (*estimateLoad)(const gguf::File& f, const qwen35::Config& cfg, bool embd_on_host, bool no_mtp) = nullptr;
    std::uint64_t (*bufferBytes)(const qwen35::Config& cfg, std::uint32_t B, std::uint64_t ffs, std::uint64_t max_elems,
                                 std::uint32_t fd_splits) = nullptr;
    std::uint32_t (*resolveFdSplits)(KvFormat kv) = nullptr;
    std::uint64_t (*dflashWeightBytes)(const gguf::File& draft) = nullptr;
    std::uint32_t max_small_batch = 0;  // rows of a decode / verify step (the per-slot replay rows)
    qwen35::MtpDefaults (*mtpDefaults)(bool moe, hip::Arch arch) = nullptr;
};

// One model family: its GGUF names, its config reader and tensor check, its sizing.
struct ArchModule {
    std::string_view family;
    std::span<const ArchName> names;
    qwen35::Config (*fromGguf)(const gguf::File& f) = nullptr;
    void (*validateTensors)(const qwen35::Config& cfg, const gguf::File& f) = nullptr;
    ArchSizing sizing;
};

using ArchTable = std::span<const ArchModule>;

// The product's table (the model library, src/model/qwen35/module.cpp).
const ArchTable& builtinArchs();

// The module serving GGUF architecture `name`, nullptr when no row of `table` has it.
const ArchModule* findArch(std::string_view name, const ArchTable& table = builtinArchs());
// The name's row (its variant), nullptr when unknown.
const ArchName* findArchName(std::string_view name, const ArchTable& table = builtinArchs());
// Every name of the table in order, joined as "a", "a and b", "a, b and c".
std::string supportedArchs(const ArchTable& table = builtinArchs());
// The UnsupportedArch detail: "general.architecture = '<name>' (WHIRL supports <supportedArchs> only)"
// (with the product table: byte-identical to v0.2.4 0e35407's literal in config.cpp).
std::string unsupportedArchText(std::string_view name, const ArchTable& table = builtinArchs());

// The server's per-slot recurrent state (st1: conv + DeltaNet state of every DeltaNet layer) and its replay rows
// (pend1: max_small_batch rows of conv input and gate / state rows), from the config and the module's sizing; the
// server's dGPU fit, its UMA check and the tests use this one formula.
struct SlotStateBytes {
    std::uint64_t st1 = 0;
    std::uint64_t pend1 = 0;
};
SlotStateBytes slotStateBytes(const ArchSizing& s, const qwen35::Config& cfg);

}  // namespace whirl::model
