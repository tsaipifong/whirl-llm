# WHIRL - platform layer (src/platform, arch025 PL0; ADR-0004 Decision 5)
# SPDX-License-Identifier: Apache-2.0
#
# Included from the top-level CMakeLists.txt before the core library. One header per concern
# (src/platform/<concern>.h, platform-free) and one Windows implementation per concern (<concern>_win.cpp), chosen
# here at link time; no virtual interface. whirl_platform is the bottom layer: it links no other WHIRL target.
# PL0 is the skeleton only (headers_win.cpp: the compile-time "headers stay platform-free" check); PL1-PL4 add
# env / paths / console / time, fs, net (ws2_32 moves here), proc / sys and their call sites. See docs/dev/platform.md.

add_library(whirl_platform STATIC
  ${CMAKE_SOURCE_DIR}/src/platform/headers_win.cpp
  ${CMAKE_SOURCE_DIR}/src/platform/env_win.cpp)
# callers include "platform/<concern>.h"; never put src/platform itself on an include path (its time.h would shadow
# the C library's <time.h>)
target_include_directories(whirl_platform PUBLIC ${CMAKE_SOURCE_DIR}/src)
