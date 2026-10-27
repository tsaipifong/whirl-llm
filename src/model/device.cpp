// Device selection and the one-process-per-GPU lock.
// SPDX-License-Identifier: Apache-2.0
// Started as a reimplementation of the WHIRL Zig research prototype's gguf_cli.zig (pickDevice, lockGpu);
// since 0.2.3 (s8-719 root fix) the matching and the default are whirl::gpusel (src/common/gpu_select.cpp)
// and the lock is keyed by the GPU's uuid / PCI address instead of its index.

#include "whirl/model.h"
#include "whirl/gpu_select.h"
#include "config/env_snapshot.h"
#include "platform/proc.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace whirl::qwen35 {

namespace {

// One model process per GPU: a named mutex per GPU (gpusel::lockName: uuid, else PCI address - after the
// restriction every WHIRL process sees its GPU as index 0), held until the process exits. A second WHIRL
// process on the same GPU waits (reporting every 30 s, up to WHIRL_GPU_WAIT seconds, default 1800).
// WHIRL_GPU_SHARE=1 skips it.
void lockGpu(const gpusel::GpuRecord& g, const whirl::config::EnvSnapshot& env) {
    if (const auto& v = env.gpu_share; v && !v->empty() && (*v)[0] != '0') return;
    namespace pproc = platform::proc;
    const std::wstring name = gpusel::lockMutexName(g);
    pproc::Handle h = pproc::createMutex(name.c_str());
    if (!h) return;
    unsigned limit_s = 1800;
    if (const auto& v = env.gpu_wait) {
        try {
            limit_s = static_cast<unsigned>(std::stoul(*v));
        } catch (...) {
        }
    }
    unsigned waited = 0;
    for (;;) {
        const std::uint32_t r = pproc::wait(h, 30000);
        if (r == pproc::kWaitObject0 || r == pproc::kWaitAbandoned) return;  // held until exit (never closed)
        waited += 30;
        std::fprintf(stderr, "whirl: another WHIRL process is using GPU pci:%d:%d (%s); waiting (%u s)\n", g.pci_bus, g.pci_device,
                     g.name.c_str(), waited);
        if (waited >= limit_s) throw ModelError("GpuBusy");
    }
}

}  // namespace

const gpusel::Restriction& restrictDevice(std::string_view spec) {
    try {
        return gpusel::restrictToSelected(spec);
    } catch (const gpusel::SelectError& e) {
        throw ModelError("DeviceChoice", e.what());
    }
}

int pickDevice(std::string_view spec_arg) {
    // arch025-N1: read once per call (before any load: there is no LoadOptions yet); gpusel writes only
    // HIP_VISIBLE_DEVICES, never a WHIRL_* name, so the values are the ones the reads below saw before
    const whirl::config::EnvSnapshot env = whirl::config::EnvSnapshot::capture();
    std::vector<gpusel::GpuRecord> recs = gpusel::enumerate();
    if (recs.empty()) throw ModelError("NoTargetGpu", "the AMD graphics driver reports no GPU");
    int pos = 0;
    try {
        pos = gpusel::chosenPosition(recs, spec_arg);  // 0 when restricted, else --device / WHIRL_DEVICE / default
    } catch (const gpusel::SelectError& e) {
        const bool none = spec_arg.empty() && !env.device;
        if (none) throw ModelError("NoTargetGpu", e.what());
        throw ModelError("DeviceChoice", e.what());
    }
    lockGpu(recs[pos], env);
    return recs[pos].index;
}

}  // namespace whirl::qwen35
