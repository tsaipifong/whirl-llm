// Windows launcher entry points, called from src/cli/main.cpp (whirl.exe). See launcher.h.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <string>

namespace whirl::launcher {

// The hidden argument of the tray's chat window: `whirl --whirl-launcher-chat HOST:PORT`.
inline constexpr const char* k_chat_arg = "--whirl-launcher-chat";
// `whirl --setup`: ask the setup questions again, save, start.
inline constexpr const char* k_setup_arg = "--setup";

// Double-click start (decideStart() == launcher) or `whirl --setup` (force_setup): settings or setup, the server
// child, the tray. Returns the process exit code.
int launcherMain(bool force_setup);
// The console chat window against a running server.
int chatMain(const std::string& host_port);

}  // namespace whirl::launcher
