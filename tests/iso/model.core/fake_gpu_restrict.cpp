// Link double of the GPU restriction half of whirl::gpusel (src/hip/gpu_restrict.cpp, whirl_hip: it enumerates the GPUs
// through the HIP runtime and sets the runtime's device visibility) for the model.core isolation program (ARCH-1 8.5,
// arch025-lane2 leg 4). The model core's device choice (src/model/device.cpp) calls it; a host-only test has no runtime
// to ask, so every entry throws. The host half (src/common/gpu_select.cpp: parsing, the default choice, the texts) is a
// declared dependency and runs for real. Only what model.core references is defined.
// SPDX-License-Identifier: Apache-2.0
#include "whirl/gpu_select.h"

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace whirl::gpusel {

namespace {

[[noreturn]] void gpu(const char* op) {
    throw std::logic_error(std::string("iso fake gpu_restrict: ") + op + " reached the device runtime");
}

}  // namespace

const Restriction& restrictToSelected(std::string_view) { gpu("restrictToSelected"); }
std::vector<GpuRecord> enumerate() { gpu("enumerate"); }
std::string userVisibilityVarHere() { gpu("userVisibilityVarHere"); }
std::string noGpuTextHere() { gpu("noGpuTextHere"); }
int chosenPosition(const std::vector<GpuRecord>&, std::string_view) { gpu("chosenPosition"); }

}  // namespace whirl::gpusel
