// Platform layer skeleton (arch025 PL0): compile-time proof that the src/platform headers stay platform-free.
// SPDX-License-Identifier: Apache-2.0
//
// Every public header of the platform layer is included first, before anything else; if one of them pulled in
// <windows.h> or Winsock (directly or through another header), the guard macros below are defined and the build
// stops. PL1-PL4 add the real implementations next to this file (env_win.cpp, fs_win.cpp, net_win.cpp, ...), each
// including <windows.h> itself; this file stays first in cmake/platform.cmake and includes no platform header.
// See docs/dev/platform.md.

#include "platform/args.h"
#include "platform/console.h"
#include "platform/env.h"
#include "platform/fs.h"
#include "platform/fs_fake.h"
#include "platform/net.h"
#include "platform/net_fake.h"
#include "platform/paths.h"
#include "platform/proc.h"
#include "platform/sys.h"
#include "platform/time.h"

#if defined(_WINDOWS_) || defined(_WINSOCKAPI_) || defined(_WINSOCK2API_) || defined(_INC_WINDOWS)
#error "a src/platform header includes <windows.h> / Winsock; headers must stay platform-free (ADR-0004 #5)"
#endif

namespace whirl::platform {

// One defined symbol so the static library is never empty (lib.exe LNK4221) before PL1 adds the first
// implementation file. Nothing references it; the linker drops this object from every executable.
extern const char kLayerName[] = "win32";

}  // namespace whirl::platform
