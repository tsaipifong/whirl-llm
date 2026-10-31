// Server engine, what engine.cpp and its split files (engine_decode / _pages / _prefill / _queue / _tier .cpp) include
// beyond engine.h (arch025-N4f; were engine.cpp's own): CycleProf (engine_prof.h, arch025-engine S2) and the log.
// KvPoolFull moved to server/engine_common.h (arch025-engine S3, module server.engine_types). Not part of the engine's
// interface (engine.h).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine.h"
#include "engine_prof.h"  // CycleProf (arch025-engine S2)
#include "log.h"
#include "server/engine_common.h"  // KvPoolFull (arch025-engine S3)

#include <array>
#include <cstdint>
#include <optional>
#include <stdexcept>
