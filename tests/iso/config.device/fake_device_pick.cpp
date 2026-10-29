// Link double of the model layer's GPU choice (src/model/device.cpp: qwen35::restrictDevice / pickDevice, declared in include/whirl/device_pick.h)
// for the isolation program of config.device (ARCH-1 8.5, arch025-iso4). The real functions probe the GPUs through the HIP runtime and take the per-GPU
// process mutex; the module under test (src/config/device_choice.h) only orders the two calls and maps their results, which the double lets a
// test set and observe.
// SPDX-License-Identifier: Apache-2.0
#include "iso/config.device/fake_device_pick.h"
#include "whirl/device_pick.h"

namespace iso_fake {

DevicePick& devicePick() {
    static DevicePick d;
    return d;
}

}  // namespace iso_fake

namespace whirl::qwen35 {

const gpusel::Restriction& restrictDevice(std::string_view spec) {
    iso_fake::devicePick().calls.push_back("restrict:" + std::string(spec));
    return iso_fake::devicePick().restriction;
}

int pickDevice(std::string_view spec) {
    iso_fake::devicePick().calls.push_back("pick:" + std::string(spec));
    return iso_fake::devicePick().picked;
}

}  // namespace whirl::qwen35
