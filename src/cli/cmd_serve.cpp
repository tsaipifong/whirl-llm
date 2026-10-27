// whirl serve: the OpenAI-compatible HTTP server, the same as whirl-server.exe (arch025-C2: wmain's serve block)
// SPDX-License-Identifier: Apache-2.0

#include "cli/cli.h"

namespace cli {

#if WHIRL_HAVE_SERVER
// argv: whirl's whole command line (argv[1] = "serve")
int cmdServe(std::vector<std::string>& argv) {
    // the server parses its own options: argv[0] = program, then everything after "serve"
    std::vector<char*> sargv;
    sargv.push_back(argv[0].data());
    for (std::size_t i = 2; i < argv.size(); ++i) sargv.push_back(argv[i].data());
    sargv.push_back(nullptr);
    return whirl::server::serveMain(static_cast<int>(sargv.size() - 1), sargv.data(), "whirl serve");
}
#endif

}  // namespace cli
