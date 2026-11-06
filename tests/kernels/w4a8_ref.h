// v0252-f16 (F16 P1a) STUB - test-first state (commit 1): the CPU reference does nothing so
// tests/kernels/unit/w4a8_ref_tests.cpp fails before the real code. Replaced by the real header in commit 2.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace w4a8ref {
inline void quantizeTiled(const float *, int, int, std::uint8_t *) {}
inline std::int32_t blockDot(const std::uint8_t *, std::size_t, int, const std::uint8_t *, int, int, int) { return 0; }
inline void gemm(const std::uint8_t *, std::size_t, const std::uint8_t *, float *, int, int, int, int) {}
inline double weightValue(const std::uint8_t *, int) { return 1.0; }
inline void gemmF64(const std::uint8_t *, std::size_t, const float *, double *, int, int, int) {}
}  // namespace w4a8ref
