// Environment variable lookup: WHIRL_<name> (defined in src/config/env_snapshot.cpp).
// arch025-lane2 D2.2: moved here from include/whirl/model.h (model.h includes this header), so env_snapshot.cpp no longer
// includes the whole qwen35 Model for two declarations.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace whirl {

// Environment variable lookup: WHIRL_<name>.
// Returns nullopt when it is not set.
std::optional<std::string> envGet(std::string_view name);
// envGet(name) present and not "0".
bool envFlag(std::string_view name, bool def = false);

}  // namespace whirl
