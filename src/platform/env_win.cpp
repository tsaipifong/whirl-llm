// Platform layer, process environment (arch025 PL1 first step). SPDX-License-Identifier: Apache-2.0
#include "platform/env.h"

#include <cstdlib>

namespace whirl::platform::env {

bool isSet(const char* name) {
    char* buf = nullptr;
    size_t len = 0;
    // _dupenv_s keeps MSVC quiet about getenv; a variable set to the empty string counts as set, as getenv != nullptr
    const bool set = _dupenv_s(&buf, &len, name) == 0 && buf != nullptr;
    std::free(buf);
    return set;
}

}  // namespace whirl::platform::env
