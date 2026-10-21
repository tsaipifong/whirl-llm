// REVIEW-GOV-2: link stub for the CPU-only (WHIRL_WITH_HIP=OFF) build of the restore test. The test sends no
// images, so vision::expand is never called; the real one lives in the HIP-only vision library.
// SPDX-License-Identifier: Apache-2.0
#include "whirl/vision.h"

#include <cstdlib>

namespace whirl::vision {
std::vector<std::uint32_t> expand(std::span<const std::uint32_t>, std::uint32_t, std::span<const Prepared* const>,
                                  std::span<qwen35::VisSpan>) {
    std::abort();
}
}  // namespace whirl::vision
