// The model layer's GPU choice entry points (defined in src/model, qwen35 namespace): pickDevice and the s8-719
// restriction restrictDevice. Moved verbatim from include/whirl/model.h (arch025-lane1 leg 3) so the front ends' shared
// GPU choice (src/config/device_choice.h, module config.device) does not include the Model header; model.h includes
// this file, so every model.h user sees the same declarations.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string_view>

namespace whirl::qwen35 {
// HIP device for `spec` (gpusel::selectGpus: pci:BUS:DEVICE, index, uuid:, r9700 / 8060s, a name part);
// empty: WHIRL_DEVICE, else gpusel::defaultGpu (the most GPU memory). After restrictDevice() restricted
// the runtime it is the one visible GPU. Takes the per-GPU process mutex, keyed by uuid / PCI address
// (WHIRL_GPU_SHARE=1 skips it).
int pickDevice(std::string_view spec);
}  // namespace whirl::qwen35
namespace whirl::gpusel {
struct Restriction;
}
namespace whirl::qwen35 {
// s8-719 root fix: decide the GPU and make the HIP runtime see only it (gpusel::restrictToSelected), before
// the first HIP call of whirl.exe / whirl-server.exe; a bad --device becomes ModelError("DeviceChoice").
const gpusel::Restriction& restrictDevice(std::string_view spec);
}  // namespace whirl::qwen35
