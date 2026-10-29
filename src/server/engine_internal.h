// Server engine, the types engine.cpp and its split files (engine_decode / _pages / _prefill / _queue / _tier .cpp)
// share (arch025-N4f; were engine.cpp's own): one definition, so a KvPoolFull thrown in one file is the type another
// file's handler sees. Not part of the engine's interface (engine.h).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine.h"
#include "engine_prof.h"  // CycleProf (arch025-engine S2)
#include "log.h"

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace whirl::server {

struct KvPoolFull : std::runtime_error {
    KvPoolFull() : std::runtime_error("KvPoolFull") {}
};

}  // namespace whirl::server
