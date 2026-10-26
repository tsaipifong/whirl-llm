// Internal header of the Model forward sources (src/model/qwen35/forward/*.cpp, arch025-F1..F4, M5): the includes they
// share and the small conversion helpers ti / I / D. The helpers stay in an anonymous namespace (arch025 PLAN C8):
// every forward TU gets its own internal-linkage copy, as when they were file-local to forward.cpp. Include only from
// the forward sources (not from a public header or an inline member).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "whirl/model.h"

#include "whirl/ops/attention.h"
#include "whirl/ops/gdn.h"
#include "whirl/ops/matmul.h"
#include "whirl/ops/moe.h"
#include "whirl/ops/mtp.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace whirl::qwen35 {

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using i32 = std::int32_t;

constexpr std::size_t ti(GgmlType t) { return static_cast<std::size_t>(t); }
inline i32 I(u64 v) { return static_cast<i32>(v); }

hip::Dim3 D(u64 x, u64 y = 1, u64 z = 1) { return hip::Dim3{static_cast<unsigned>(x), static_cast<unsigned>(y), static_cast<unsigned>(z)}; }

}  // namespace

}  // namespace whirl::qwen35
