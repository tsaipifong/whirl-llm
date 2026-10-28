// The GPU choice of the two front ends (arch025-N15b): `whirl` (src/cli/build_model.cpp selectDevice) and whirl-server
// (src/server/server_main.cpp) chose the HIP device with the same steps written twice. One rule here:
//   WHIRL_HIP_DEVICE=N (debug override, the settings u32 rule: a bad value warns and uses 0) -> device N, shown N;
//   else the s8-719 root fix: restrictDevice(spec) (the runtime sees only the chosen GPU), then pickDevice(spec);
//   shown = the index `whirl devices` / --list-devices prints (before the restriction).
// The restriction's note goes to `note` right after restrictDevice, before pickDevice (the server's order; server: logI,
// CLI: stderr "whirl: ..."); it is also returned. Header-only:
// it uses the model layer's device selection (include/whirl/device_pick.h), which both front ends link. llama.cpp shape:
// one --device handling in common (common/arg.cpp), used by llama-cli and llama-server (arch025-lane1.PRIOR_ART.md).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "config/run_settings.h"
#include "whirl/gpu_select.h"
#include "whirl/device_pick.h"  // qwen35::restrictDevice / pickDevice (not the Model header)

#include <string>
#include <string_view>

namespace whirl::settings {

struct DeviceChoice {
    int dev = 0;       // the HIP device to use
    int shown = 0;     // its index as the device list prints it (before the restriction)
    std::string note;  // the restriction's log line ("" = none)
};

inline DeviceChoice chooseDevice(std::string_view spec, const Warn& warn, const Warn& note) {
    DeviceChoice c;
    if (envGet("HIP_DEVICE")) {
        c.dev = c.shown = static_cast<int>(envU32("HIP_DEVICE", 0, warn));
        return c;
    }
    // s8-719 root fix: choose the GPU and make the HIP runtime see only it, before the first HIP call
    // (a pinned host buffer is charged on every GPU the runtime sees: hipError 719 with two GPUs)
    const gpusel::Restriction& r = qwen35::restrictDevice(spec);
    c.note = r.note;
    if (!c.note.empty() && note) note(c.note);
    c.dev = qwen35::pickDevice(spec);
    c.shown = r.active ? r.all[r.selected[0]].index : c.dev;
    return c;
}

}  // namespace whirl::settings
