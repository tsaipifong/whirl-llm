// WHIRL_DUMP_EXEC and the --dump-only exit after load + tune (arch025-S0a; see whirl/dump.h for the switches).
// arch025-lane2 N3: moved here from include/whirl/dump.h, so dump.h (the numerics-level KV dump, included by
// src/common/numerics.cpp) no longer names the qwen35 Model and dump + numerics form one low-level module.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace whirl::qwen35 {
class Model;
// WHIRL_DUMP_EXEC: the "D exec:" lines (src/model/exec_dump.cpp)
std::string execDump(const Model& m);
// end of loadOrTune: the exec dump when WHIRL_DUMP_EXEC is on; exit(0) when --dump-only
void afterLoadTune(const Model& m);
}  // namespace whirl::qwen35
