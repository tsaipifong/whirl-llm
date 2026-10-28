// whirl devices: the AMD GPUs the driver reports (arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"
#include "platform/env.h"  // HIP_VISIBLE_DEVICES (arch025-N10: no std::getenv outside the platform layer)

namespace cli {

const char* k_help_devices =
    "usage: whirl devices\n"
    "\n"
    "Lists the AMD GPUs the driver reports (index, name, architecture, memory, compute units),\n"
    "whether this build has GPU kernels for each of them, and which one is used by default\n"
    "(the GPU with the most GPU memory; --device / WHIRL_DEVICE pick another: pci:BUS:DEVICE, as Task Manager >\n"
    "Performance > GPU > Physical location shows it, stays the same when GPUs are added or removed; an index can change).\n";

int cmdDevices() {
    const std::vector<whirl::gpusel::GpuRecord> devs = whirl::gpusel::enumerate();
    if (devs.empty()) {
        // arch025-c2fu: HIP_VISIBLE_DEVICES= (or another visibility variable) hides every GPU: name it, no driver advice
        if (!whirl::gpusel::userVisibilityVarHere().empty()) {
            std::fputs(("whirl: error: no AMD GPU is visible to WHIRL: " + whirl::gpusel::noGpuTextHere() + ".\n").c_str(), stderr);
            return app::exit_gpu;
        }
        std::fputs("whirl: error: the AMD graphics driver reports no GPU.\n"
                   "  WHIRL needs an AMD Radeon AI PRO R9700 (gfx1201) or a Radeon 8060S (gfx1151). If one is\n"
                   "  installed, install AMD Software:\n"
                   "  Adrenalin Edition 26.8.1 or newer (https://www.amd.com/en/support) and restart Windows.\n",
                   stderr);
        return app::exit_gpu;
    }
    // every GPU (this command never restricts the runtime); [default] = the GPU used without --device /
    // WHIRL_DEVICE (gpusel::defaultGpu: the most GPU memory; an integrated GPU counts its carve-out, UMA-1)
    if (const auto v = whirl::platform::env::get("HIP_VISIBLE_DEVICES"))
        out(fmt("  (HIP_VISIBLE_DEVICES=%s is set: only those GPUs are listed, numbered from 0)\n", v->c_str()));
    out(whirl::gpusel::listingText(devs, whirl::gpusel::defaultGpu(devs).index));
    return 0;
}

}  // namespace cli
