// whirl chat --draft dflash: where WHIRL_DFLASH_DUMP sends the captured target features (cmd_chat.cpp).
// SPDX-License-Identifier: Apache-2.0
//
// Header-only, so the CPU unit tests pin the rule without the GPU chat path (arch025-gaptests G19).
#pragma once

#include <optional>
#include <string>

namespace cli {  // the CLI's namespace (src/cli/cli.h)

// The dump folder for a WHIRL_DFLASH_DUMP value (nullopt = unset): the value itself, or nullopt (no dump) when it is
// unset or set but empty. bug-emptyrootpaths (37ad538): an empty value used to reach dflashDumpFeatures, where MSVC
// create_directories("") throws and "" + "/feats.npy" names the drive root. cmd_chat.cpp prints
// "WHIRL_DFLASH_DUMP is empty: no feature dump" for the empty case.
inline std::optional<std::string> dflashDumpDir(const std::optional<std::string>& env_value) {
    if (env_value && env_value->empty()) return std::nullopt;
    return env_value;
}

}  // namespace cli
