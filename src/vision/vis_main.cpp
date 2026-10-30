// whirl-vis: standalone vision encoder test (see vis_cli.cpp).
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-vis vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]

#include "whirl/vision.h"

// arch025-n8b2: the body is whirl::vision::visMain (src/vision/vis_cli.cpp, moved verbatim), so the vision tests can run it.
int main(int argc, char** argv) { return whirl::vision::visMain(argc, argv); }
