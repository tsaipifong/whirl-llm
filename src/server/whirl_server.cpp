// whirl-server executable: UTF-16 command line -> UTF-8 argv -> serveMain.
// SPDX-License-Identifier: Apache-2.0

#include "server_main.h"
#include "platform/path_budget.h"
#include "platform/paths.h"
#include "release/release.h"
#include "whirl/common.h"
#include "whirl/gpu_select.h"

#include <cstdio>
#include <optional>
#include <string>
#include <vector>

int wmain(int argc, wchar_t** wargv) {
    whirl::app::initConsole();
    // PATH-256 (v0.2.5.1 PLAN V1-22, pathlen256.DISCOVERY.md 2c #2): an install folder longer than 150 UTF-8 bytes is
    // refused first, for every argument (also --version, --help and the probe), before the probe branch and
    // gpuPreflight, so before anything can load the AMD runtime (amdhip64 is loaded only by hip::runtime::load)
    if (const std::optional<std::string> refuse = whirl::platform::path_budget::installDirRefusal("whirl-server")) {
        std::fputs(refuse->c_str(), stderr);
        return whirl::app::exit_usage;
    }
    // s8-719 root fix: the GPU list helper (a child of this program, see gpu_select.h); never interactive
    if (argc == 2 && std::wstring(wargv[1]) == L"--whirl-gpu-probe") return whirl::gpusel::probeMain();
    // PATH-256 2c #3: a LOCALAPPDATA over 187 UTF-16 units is not used (no tune cache, no default SSD folder, the log on
    // the console only); said once, not by the probe child
    if (const std::optional<std::string> w = whirl::platform::paths::localAppDataWarning()) std::fputs(("whirl-server: " + *w).c_str(), stderr);
    std::vector<std::string> args = whirl::narrowArgs(argc, wargv);
    bool info_only = false;  // --help / --version need no GPU
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string& a = args[i];
        if (a == "-h" || a == "--help" || a == "/?" || a == "-V" || a == "--version") info_only = true;
    }
    if (!info_only)
        if (const int pf = whirl::app::gpuPreflight(); pf != whirl::app::exit_ok) return pf;
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);
    return whirl::server::serveMain(argc, argv.data());
}
