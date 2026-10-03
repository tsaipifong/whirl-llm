// whirl-vis: standalone vision encoder test (see vis_cli.cpp).
// SPDX-License-Identifier: Apache-2.0
//
//   whirl-vis vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]

#include "whirl/vision.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2 || std::strcmp(argv[1], "vis-encode") != 0) {
        std::printf("usage: whirl-vis vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream]\n");
        return 2;
    }
    std::vector<std::string> args(argv + 2, argv + argc);
    try {
        return whirl::vision::cliVisEncode(args);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "whirl-vis: %s\n", e.what());
        return 1;
    }
}
