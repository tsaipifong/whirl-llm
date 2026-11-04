// whirl-vis: standalone vision encoder test (see vis_cli.cpp).
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-vis vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]

#include "platform/path_budget.h"
#include "whirl/vision.h"

#include <cstdio>
#include <optional>
#include <string>

// arch025-n8b2: the body is whirl::vision::visMain (src/vision/vis_cli.cpp, moved verbatim), so the vision tests can run it.
// PATH-256 (v0.2.5.1 PLAN V1-22, pathlen256.DISCOVERY.md 2c #2): an install folder longer than 150 UTF-8 bytes is refused
// first, before visMain (its --whirl-gpu-probe branch) and anything that can load the AMD runtime; exit 2.
int main(int argc, char** argv) {
    if (const std::optional<std::string> refuse = whirl::platform::path_budget::installDirRefusal("whirl-vis")) {
        std::fputs(refuse->c_str(), stderr);
        return 2;
    }
    return whirl::vision::visMain(argc, argv);
}
