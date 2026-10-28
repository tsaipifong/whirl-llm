// Run settings shared by the CLI and the server (arch025-N15). See run_settings.h.
// SPDX-License-Identifier: Apache-2.0

#include "config/run_settings.h"

#include "platform/env.h"

#include <exception>
#include <format>
#include <string>

namespace whirl::settings {

std::optional<std::string> envGet(std::string_view name) {
    return platform::env::getUtf8(std::string("WHIRL_") + std::string(name));
}

bool envFlag(std::string_view name, bool def) {
    if (auto v = envGet(name)) return *v != "0";
    return def;
}

// (moved from src/server/server_main.cpp envU32, the server's rule; FR-SRV P3-10)
std::optional<std::uint32_t> parseU32(std::string_view v) {
    try {
        const std::string s(v);
        std::size_t used = 0;
        const unsigned long x = std::stoul(s, &used);
        if (used == s.size() && x <= 0xffffffffull) return static_cast<std::uint32_t>(x);
    } catch (const std::exception&) {
    }
    return std::nullopt;
}

std::uint32_t envU32(std::string_view name, std::uint32_t def, const Warn& warn) {
    if (auto v = envGet(name)) {
        if (auto x = parseU32(*v)) return *x;
        if (warn) warn(std::format("WHIRL_{}={} is not a number; using {}", name, *v, def));
    }
    return def;
}

std::optional<std::uint32_t> envU32(std::string_view name, const Warn& warn) {
    if (auto v = envGet(name)) {
        if (auto x = parseU32(*v)) return x;
        if (warn) warn(std::format("WHIRL_{}={} is not a number; ignored", name, *v));
    }
    return std::nullopt;
}

// (the server's rule, src/server/server_main.cpp MTP_ADAPT before arch025-N15)
MtpAdapt parseMtpAdapt(const std::optional<std::string>& v) {
    MtpAdapt r;
    if (!v) return r;
    r.set = true;
    r.automatic = *v == "auto";
    try {
        r.adapt = static_cast<std::uint32_t>(std::stoul(*v));
    } catch (const std::exception&) {
        r.adapt = 0;
    }
    return r;
}

MtpAdapt mtpAdapt() { return parseMtpAdapt(envGet("MTP_ADAPT")); }

}  // namespace whirl::settings
