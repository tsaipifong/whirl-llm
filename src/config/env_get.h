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
// v0252-probe (PLAN V1-123 / V1-119; probe branch): WHIRL_RELAX_RANK_CHECK, the runtime switch of relaxacc's one-pass rank
// check. `def` = device::Profile::relax_rank_check, so unset = the profile's value; "0" = off, any other value = on. Read at
// the load (src/model/loader.cpp -> Context::relax_rank_check, read by Context::setRelaxRank / relaxRankEnqueue) and at
// src/model/tune.cpp k_relax_rank, so host side and kernel choice agree. Here and not in run_settings.h: the model layer includes this header only.
inline bool relaxRankCheck(bool def) { return envFlag("RELAX_RANK_CHECK", def); }

}  // namespace whirl::settings

namespace whirl {

using settings::envFlag;
using settings::envGet;

}  // namespace whirl
