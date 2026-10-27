// Platform layer, UTF-8 <-> UTF-16: the Windows implementation of platform/args.h.
// SPDX-License-Identifier: Apache-2.0
//
// arch025 PL4: the two conversion calls of src/common/util.cpp (whirl::widen / whirl::narrow), arguments as in v0.2.4.

#include "platform/args.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace whirl::platform::args {

int wideFromUtf8(const char* s, int n, wchar_t* out, int cap) { return MultiByteToWideChar(CP_UTF8, 0, s, n, out, cap); }

int utf8FromWide(const wchar_t* s, int n, char* out, int cap) {
    return WideCharToMultiByte(CP_UTF8, 0, s, n, out, cap, nullptr, nullptr);
}

}  // namespace whirl::platform::args
