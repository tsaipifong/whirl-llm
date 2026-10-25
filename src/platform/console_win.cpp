// Platform layer, console: the Windows implementation of platform/console.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL1: the console calls of src/release/release.cpp (initConsole / restoreConsole / isConsole) and the binary
// stdout of src/cli/main.cpp, with the arguments they had in v0.2.4.

#include "platform/console.h"

#include <fcntl.h>
#include <io.h>

#include <cstdio>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

static_assert(CP_UTF8 == whirl::platform::console::kCodePageUtf8);

namespace whirl::platform::console {

bool isConsole(Stream s) {
    HANDLE h = GetStdHandle(s == Stream::out ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
    DWORD mode = 0;
    return h != nullptr && h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode) != 0;
}

unsigned outputCodePage() { return GetConsoleOutputCP(); }

bool setOutputCodePage(unsigned cp) { return SetConsoleOutputCP(cp) != 0; }

void setStdoutBinary() { _setmode(_fileno(stdout), _O_BINARY); }

}  // namespace whirl::platform::console
