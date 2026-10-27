// whirl devices: the AMD GPUs the driver reports (arch025-C2: from src/cli/main.cpp)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

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
        std::fputs("whirl: error: the AMD graphics driver reports no GPU.\n"
                   "  WHIRL needs an AMD Radeon AI PRO R9700 (gfx1201) or a Radeon 8060S (gfx1151). If one is\n"
                   "  installed, install AMD Software:\n"
                   "  Adrenalin Edition 26.8.1 or newer (https://www.amd.com/en/support) and restart Windows.\n",
                   stderr);
        return app::exit_gpu;
    }
    // every GPU (this command never restricts the runtime); [default] = the GPU used without --device /
    // WHIRL_DEVICE (gpusel::defaultGpu: the most GPU memory; an integrated GPU counts its carve-out, UMA-1)
    if (const char* v = std::getenv("HIP_VISIBLE_DEVICES")) out(fmt("  (HIP_VISIBLE_DEVICES=%s is set: only those GPUs are listed, numbered from 0)\n", v));
    out(whirl::gpusel::listingText(devs, whirl::gpusel::defaultGpu(devs).index));
    return 0;
}

}  // namespace cli
