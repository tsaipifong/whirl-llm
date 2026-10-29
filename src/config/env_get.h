// Environment variable lookup: WHIRL_<name> (module config.settings; defined in src/config/run_settings.cpp).
// arch025-lane2 D2.2: moved here from include/whirl/model.h (model.h includes this header), so env_snapshot.cpp no longer
// includes the whole qwen35 Model for two declarations.
// arch025-N6 (lane 1 leg 11): whirl::envGet / envFlag (defined in src/config/env_snapshot.cpp) and whirl::settings::envGet /
// envFlag (src/config/run_settings.cpp) were the same two functions; the model layer's names are now using-declarations
// of config.settings' ones (llama.cpp: one env reader, common/arg.cpp get_value_from_env). whirl_model links whirl_settings.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <string_view>

namespace whirl::settings {

// WHIRL_<name> (UTF-8 of the wide value, platform::env::getUtf8). nullopt when unset.
std::optional<std::string> envGet(std::string_view name);
// WHIRL_<name> set and not "0"; `def` when unset (the former envFlag / envOn).
bool envFlag(std::string_view name, bool def = false);

}  // namespace whirl::settings

namespace whirl {

using settings::envFlag;
using settings::envGet;

}  // namespace whirl
