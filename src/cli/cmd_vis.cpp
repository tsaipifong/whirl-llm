// whirl vis-encode: encode one image with a vision encoder (diagnostic; arch025-C2: wmain's vis-encode block)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

const char* k_help_vis =
    "usage: whirl vis-encode MMPROJ.gguf IMAGE [OUT.f32] [--reps N] [--mode auto|resident|stream] [--device D]\n"
    "\n"
    "Encodes one image with a vision encoder (diagnostic) and reports the timing; OUT.f32\n"
    "receives the projected embeddings.\n"
    "\n"
    "options:\n"
    "  --reps N                   encode N times (timing)\n"
    "  --mode M                   auto (default), resident (weights in VRAM), stream (layer by layer)\n"
    "  --device D, -dev D         the GPU, as for whirl chat (default: the one with the most GPU memory)\n";

#if WHIRL_HAVE_VISION
// argv: whirl's whole command line (argv[1] = "vis-encode")
int cmdVisEncode(std::vector<std::string>& argv) {
    try {
        const std::span<const std::string> va = std::span<const std::string>(argv).subspan(2);
        if (va.size() >= 2) {
            app::requireModelFile(va[0], "vision encoder (mmproj) file");
            app::requireModelFile(va[1], "image file");
        }
        return vision::cliVisEncode(va);
    } catch (const std::exception& e) {
        return app::explain(e, "whirl");
    }
}
#endif

}  // namespace cli
