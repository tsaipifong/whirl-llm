// Identity of everything that decides the bits of a host tier entry (moved out
// of server_main.cpp, refactor R-1). An entry is restored only by a process
// with the same fingerprint.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace whirl::cache {

struct FingerprintInputs {
    std::uint64_t exe_id = 0;
    std::string model_base;  // model file name (no directory)
    std::uint64_t tensors = 0, bytes = 0;
    std::string kv_name;
    bool use_mtp = false;
    std::uint64_t n_gdn = 0;
    std::uint64_t ck_bytes = 0, page_bytes = 0;
    std::uint64_t max_batch = 0;
    // numerics mode and its items (numerics::logLine): a --balance run's KV /
    // DeltaNet state must never be restored by a --precise process
    std::string numerics;
    // WHIRL_* environment (any order; filtered by fingerprintEnvKept)
    std::vector<std::pair<std::string, std::string>> env;
};

std::uint64_t tierFingerprint(const FingerprintInputs& in);

// A WHIRL_* variable that changes numerics (not a path / size / logging knob).
bool fingerprintEnvKept(std::string_view key);

// The process environment's kept WHIRL_* variables (empty off Windows).
std::vector<std::pair<std::string, std::string>> fingerprintEnv();

}  // namespace whirl::cache
