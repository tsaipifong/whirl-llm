// Architecture registry: table lookup, the UnsupportedArch text, the load entry (arch025-lane2 leg 6) and the server's
// per-slot state bytes (arch025-M5).
// SPDX-License-Identifier: Apache-2.0

#include "model/arch_registry.h"

namespace whirl::model {

const ArchModule* findArch(std::string_view name, const ArchTable& table) {
    for (const ArchModule& m : table)
        for (const ArchName& n : m.names)
            if (n.name == name) return &m;
    return nullptr;
}

const ArchName* findArchName(std::string_view name, const ArchTable& table) {
    for (const ArchModule& m : table)
        for (const ArchName& n : m.names)
            if (n.name == name) return &n;
    return nullptr;
}

std::string supportedArchs(const ArchTable& table) {
    std::size_t total = 0;
    for (const ArchModule& m : table) total += m.names.size();
    std::string s;
    std::size_t i = 0;
    for (const ArchModule& m : table)
        for (const ArchName& n : m.names) {
            if (i > 0) s += i + 1 == total ? " and " : ", ";
            s += n.name;
            ++i;
        }
    return s;
}

std::string unsupportedArchText(std::string_view name, const ArchTable& table) {
    return "general.architecture = '" + std::string(name) + "' (WHIRL supports " + supportedArchs(table) + " only)";
}

Loaded loadArch(const ArchModule& am, const gguf::File& f, std::uint32_t max_ctx, qwen35::LoadStats& stats,
                const qwen35::LoadOptions& lo) {
    if (!am.load) return {};
    return am.load(f, max_ctx, stats, lo);
}

SlotStateBytes slotStateBytes(const ArchSizing& s, const qwen35::Config& cfg) {
    std::uint64_t n_gdn_c = 0;
    for (std::uint32_t li = 0; li < cfg.n_layer; ++li)
        if (!cfg.isAttn(li)) ++n_gdn_c;
    SlotStateBytes r;
    r.st1 = n_gdn_c * (static_cast<std::uint64_t>(cfg.d_conv - 1) * cfg.convCh() * 4 +
                       static_cast<std::uint64_t>(cfg.n_v_heads) * cfg.d_state * cfg.headV() * 4);
    r.pend1 = n_gdn_c * (static_cast<std::uint64_t>(s.max_small_batch) * cfg.convCh() * 4 +
                         static_cast<std::uint64_t>(s.max_small_batch) * cfg.n_v_heads * (cfg.d_state + cfg.headV() + 2) * 4);
    return r;
}

}  // namespace whirl::model
