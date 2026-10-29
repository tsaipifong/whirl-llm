// Controls of the link double tests/iso/config.device/fake_device_pick.cpp (ARCH-1 8.5 isolation of config.device, arch025-iso4).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "whirl/gpu_select.h"

#include <string>
#include <vector>

namespace iso_fake {

// what the double returns, and what it saw (in call order: "restrict:<spec>" / "pick:<spec>")
struct DevicePick {
    whirl::gpusel::Restriction restriction;
    int picked = 0;
    std::vector<std::string> calls;
};
DevicePick& devicePick();

}  // namespace iso_fake
