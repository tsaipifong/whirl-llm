// whirl-iso-hip-restrict-tests: the module hip.restrict (src/hip/gpu_restrict.cpp, the GPU visibility restriction of whirl.exe / whirl-server) built with
// its declared dependencies (gpu_select, the HIP runtime layer, platform env) and run (ARCH-1 8.5 isolation, arch025-iso4). No unit test existed. The
// restriction decides once per process, and its normal path probes the GPUs through a helper process and the HIP runtime, which a CPU test must not do;
// the checks here are the paths that decide WITHOUT a GPU: the user's own visibility variable (left as given, no probe), WHIRL_GPU_RESTRICT=0, and
// the two environment readers (userVisibilityVarHere / noGpuTextHere). The probing path is covered by the srv_gpu_select / srv_kvram719 GPU steps.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "whirl/gpu_select.h"

#include <cstdlib>
#include <string>

namespace {

namespace gs = whirl::gpusel;

void setVar(const char* k, const char* v) { _putenv_s(k, v ? v : ""); }  // "" removes the variable from this process

void testEnvironmentReaders() {
    for (const char* k : {"HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"}) setVar(k, nullptr);
    ISO_CHECK(gs::userVisibilityVarHere().empty());
    ISO_CHECK(gs::noGpuTextHere() == "the AMD graphics driver reports no GPU");
    setVar("ROCR_VISIBLE_DEVICES", "1");
    ISO_CHECK(gs::userVisibilityVarHere() == "ROCR_VISIBLE_DEVICES");
    ISO_CHECK(gs::noGpuTextHere().find("ROCR_VISIBLE_DEVICES=1 is set and may hide the GPU") == 0);
    setVar("ROCR_VISIBLE_DEVICES", nullptr);
    setVar("HIP_VISIBLE_DEVICES", "0");  // the first name of the list wins
    setVar("CUDA_VISIBLE_DEVICES", "1");
    ISO_CHECK(gs::userVisibilityVarHere() == "HIP_VISIBLE_DEVICES");
    setVar("CUDA_VISIBLE_DEVICES", nullptr);
    setVar("HIP_VISIBLE_DEVICES", nullptr);
}

// restrictToSelected decides once per process, so this one case sets up the user's variable before the first call
void testUserVariableLeftAsGiven() {
    setVar("HIP_VISIBLE_DEVICES", "0");
    setVar("WHIRL_DEVICE", nullptr);
    ISO_CHECK(!gs::current().done);  // nothing decided yet
    const gs::Restriction& r = gs::restrictToSelected("");
    ISO_CHECK(r.done && !r.active && r.all.empty() && r.selected.empty());  // no probe, no restriction
    ISO_CHECK(r.note == gs::userVarNote("HIP_VISIBLE_DEVICES", "0"));
    ISO_CHECK(&gs::restrictToSelected("anything") == &r && &gs::current() == &r);  // later calls return the first result
    ISO_CHECK(r.note.find("HIP_VISIBLE_DEVICES=0 is set, so WHIRL leaves the GPU visibility as given") != std::string::npos);
    setVar("HIP_VISIBLE_DEVICES", nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    // the environment readers first (they need the variables unset), then the one-shot decision
    return iso::run("hip.restrict", {{"environment_readers", testEnvironmentReaders}, {"user_variable_left_as_given", testUserVariableLeftAsGiven}}, argc, argv);
}
