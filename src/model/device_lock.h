// The one-process-per-GPU lock of src/model/device.cpp (pickDevice calls it after the GPU is chosen).
// SPDX-License-Identifier: Apache-2.0
//
// arch025-surv: declared here (it was file-local) only so whirl-model-tests can run it on a test-only GpuRecord
// (mutation survivor pl4-wait-abandoned: an abandoned mutex, i.e. a WHIRL process killed while it held the GPU, must be
// taken at once). Not part of the model interface.
#pragma once

#include "config/env_snapshot.h"
#include "whirl/gpu_select.h"

namespace whirl::qwen35 {

// The named mutex gpusel::lockMutexName(g), waited for in 30 s steps up to WHIRL_GPU_WAIT (default 1800) seconds, then
// ModelError("GpuBusy"); returns at once when it is free or abandoned, and when WHIRL_GPU_SHARE is set (not "0").
// The mutex is held until the process exits (never closed).
void lockGpu(const gpusel::GpuRecord& g, const whirl::config::EnvSnapshot& env);

}  // namespace whirl::qwen35
