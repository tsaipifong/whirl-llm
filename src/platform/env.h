// Platform layer: process environment (arch025 PL1). See docs/dev/platform.md.
// SPDX-License-Identifier: Apache-2.0
//
// Rules of every src/platform header (ADR-0004 Decision 5): no Windows / CRT-platform header here (checked by
// src/platform/headers_win.cpp at compile time and by tools/gates/platform_check.ps1); plain C++ types only; the
// Windows implementation is env_win.cpp, chosen by cmake/platform.cmake. No virtual interface.
#pragma once

#include <optional>
#include <string>
#include <vector>

namespace whirl::platform::env {

// One variable of the CRT's narrow environment (_dupenv_s: the ANSI code page, read at the call). nullopt when unset.
// The single raw read behind envGet / EnvSnapshot::capture (src/config), the VRAM cap, the kernel table switches and
// LOCALAPPDATA up to PL1. Moved unchanged from the four _dupenv_s copies of v0.2.4.
std::optional<std::string> get(const std::string& name);

// One variable of the CRT's wide environment (_wdupenv_s: UTF-16, read at the call) as UTF-8. nullopt when unset,
// "" when set but empty (as get()). For values that become file system paths through widen(): the narrow table is
// the ANSI code page, which is UTF-8 only in a process with the activeCodePage manifest (whirl.exe /
// whirl-server.exe), so get() mangles a non-ASCII user folder in every other program (bug-lappdata). For ASCII values
// the bytes equal get()'s. Used by paths::localAppData().
std::optional<std::string> getUtf8(const std::string& name);

// _putenv_s(name, value): the CRT's and the process environment. Returns its errno_t (0 = ok); callers that ignored
// it before keep ignoring it. The one product write: GPU_RESOURCE_CACHE_SIZE before the HIP runtime starts.
int set(const char* name, const char* value);

// The process environment block (GetEnvironmentStringsW), one entry per string ("NAME=value", also the hidden
// "=C:=C:\..." entries), in the order the OS returns it; UTF-16, unconverted. Empty when the call fails.
// The SSD-cache fingerprint (src/cache/fingerprint.cpp) filters and converts it.
std::vector<std::wstring> block();

// True when the variable exists in the process environment (even when empty). First PL1 step: used by the test-only
// NullBackend leak ledger (WHIRL_LEAKCHECK_QUIET, bug-nullbackendenv).
bool isSet(const char* name);

}  // namespace whirl::platform::env
