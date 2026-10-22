// whirl-server executable: UTF-16 command line -> UTF-8 argv -> serveMain.
// SPDX-License-Identifier: Apache-2.0

#include "server_main.h"
#include "release/release.h"
#include "whirl/common.h"
#include "whirl/gpu_select.h"

#include <string>
#include <vector>

int wmain(int argc, wchar_t** wargv) {
    whirl::app::initConsole();
    // s8-719 root fix: the GPU list helper (a child of this program, see gpu_select.h); never interactive
    if (argc == 2 && std::wstring(wargv[1]) == L"--whirl-gpu-probe") return whirl::gpusel::probeMain();
    std::vector<std::string> args;
    bool info_only = false;  // --help / --version need no GPU
    for (int i = 0; i < argc; ++i) {
        args.push_back(whirl::narrow(wargv[i]));
        const std::string& a = args.back();
        if (i > 0 && (a == "-h" || a == "--help" || a == "/?" || a == "-V" || a == "--version")) info_only = true;
    }
    if (!info_only)
        if (const int pf = whirl::app::gpuPreflight(); pf != whirl::app::exit_ok) return pf;
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    return whirl::server::serveMain(argc, argv.data());
}
